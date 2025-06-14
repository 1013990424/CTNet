import torch
import torch.nn as nn
import torch.nn.functional as F

def rgb2hsv(img):
    eps = 1e-8
    hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

    hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 2] == img.max(1)[0]]
    hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 1] == img.max(1)[0]]
    hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 0] == img.max(1)[0]]) % 6

    hue[img.min(1)[0] == img.max(1)[0]] = 0.0
    hue = hue / 6

    saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + eps)
    saturation[img.max(1)[0] == 0] = 0

    value = img.max(1)[0]

    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)

    return hue, saturation, value

class attention(nn.Module):
    def __init__(self, input_dim):
        super(attention, self).__init__()
        self.fc1 = nn.Conv2d(input_dim, input_dim//16, kernel_size=1)
        self.fc2 = nn.Conv2d(input_dim//16, input_dim, kernel_size=1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        vec = self.avg(x)
        vec = self.fc1(vec)
        vec = self.relu(vec)
        vec = self.fc2(vec)
        vec = self.sigmoid(vec)
        x = x + x * vec

        return x

class encode_without_help(nn.Module):
    def __init__(self, input_dim):
        super(encode_without_help, self).__init__()
        # encoder
        self.conv1 = nn.Conv2d(input_dim, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv5 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu = nn.LeakyReLU()
        self.pool1 = nn.Conv2d(16, 16, 3, 2, 1)
        self.pool2 = nn.Conv2d(32, 32, 3, 2, 1)

    def forward(self, x):
        # encoder
        dn1 = self.conv1(x)
        dn1 = self.relu(dn1)
        dn1 = self.conv2(dn1)
        dn1 = self.relu(dn1)

        dn2 = self.pool1(dn1)
        dn2 = self.conv3(dn2)
        dn2 = self.relu(dn2)
        dn2 = self.conv4(dn2)
        dn2 = self.relu(dn2)

        dn3 = self.pool2(dn2)
        dn3 = self.conv5(dn3)
        dn3 = self.relu(dn3)
        dn3 = self.conv6(dn3)
        dn3 = self.relu(dn3)

        return dn1, dn2, dn3

class Block(nn.Module):
    def __init__(self, input_dim):
        super(Block, self).__init__()
        self.up_dim = input_dim - input_dim % 6 + 6
        self.temp = self.up_dim // 6
        self.conv1 = nn.Conv2d(input_dim, input_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(input_dim, self.up_dim, 1, 1, 0)
        self.conv3 = nn.Conv2d(self.temp, input_dim * 2, 3, 1, 1)
        self.conv4 = nn.Conv2d(input_dim * 2, input_dim * 2, 3, 1, 1)
        self.conv5 = nn.Conv2d(input_dim * 12, input_dim * 2, 1, 1, 0)
        self.conv6 = nn.Conv2d(input_dim * 3, input_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.at = attention(self.up_dim)

    def forward(self, hfea, sfea, vfea):
        """pre"""
        c = sfea * vfea
        c = self.lrelu(self.conv1(c))
        x = c * self.lrelu(self.conv1(hfea))
        x = self.lrelu(self.conv1(x))
        m = vfea - c
        m = self.lrelu(self.conv1(m))
        """up"""
        hfea = self.lrelu(self.conv2(hfea))
        """group"""
        batchsize, num_channels, height, width = hfea.data.size()
        hfea = hfea.reshape(batchsize, self.temp, 6, height, width)
        hfea = hfea.permute(0, 2, 1, 3, 4)  # 交换不同组的信息
        hfea = hfea.reshape(batchsize, num_channels, height, width)
        feas = []
        hfeas = torch.chunk(hfea, 6, 1)
        fea = torch.cat([x, c], dim=1)
        for block in hfeas:
            group_fea = self.sigmoid(self.conv3(block)) * fea
            feas.append(group_fea)
        fea = torch.cat(feas, dim=1)
        """tail"""
        fea = self.lrelu(self.conv5(fea))
        fea = torch.cat([fea, m], dim=1)
        fea = self.lrelu(self.conv6(fea))

        return fea

class Aggregate_Block(nn.Module):
    def __init__(self, input_dim):
        super(Aggregate_Block, self).__init__()
        self.conv_rgb2hsv = nn.Conv2d(64, 1, 3, 1, 1)
        self.conv_hsv2rgb = nn.Conv2d(input_dim, 1, 3, 1, 1)
        self.conv1 = nn.Conv2d(input_dim, 1, 3, 1, 1)
        self.conv2 = nn.Conv2d(input_dim * 2, input_dim * 2, 3, 1, 1)
        self.fc1 = nn.Conv2d(input_dim, input_dim // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(input_dim // 16, input_dim, kernel_size=1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU()

    def forward(self, rgb_fea, hsv_fea):
        fea = rgb_fea - hsv_fea
        # RGB的通道
        vec = self.avg(fea)
        vec = self.fc1(vec)
        vec = self.lrelu(vec)
        vec = self.fc2(vec)
        rgb_vec = self.sigmoid(vec)
        rgb_mask = self.sigmoid(self.conv1(fea))
        vec1 = self.avg(fea)
        vec1 = self.fc1(vec1)
        vec1 = self.lrelu(vec1)
        vec1 = self.fc2(vec1)
        hsv_vec = self.sigmoid(vec1)
        hsv_mask = self.sigmoid(self.conv1(fea))
        rgb_fea = hsv_mask * hsv_vec * rgb_fea + rgb_fea
        hsv_fea = rgb_mask * rgb_vec * hsv_fea + hsv_fea
        fea = torch.cat([rgb_fea, hsv_fea], dim=1)
        fea = self.lrelu(self.conv2(fea))

        return fea

class hsv_Unet(nn.Module):
    def __init__(self, out_dim):
        super(hsv_Unet, self).__init__()
        # decoder
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv11 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(32 + 64, 32, 3, 1, 1)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(16 + 32, 16, 3, 1, 1)
        self.conv33 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, out_dim, 3, 1, 1)
        # other
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, vfea1, vfea2, vfea3):
        # decode
        dn6 = self.relu(self.conv1(vfea3))
        dn5 = self.relu(self.conv11(dn6))  # 64
        fea = F.interpolate(dn5, scale_factor=2, mode='nearest')
        fea = torch.cat([fea, vfea2], dim=1)
        dn4 = self.relu(self.conv2(fea))
        dn3 = self.relu(self.conv22(dn4))  # 32
        fea = F.interpolate(dn3, scale_factor=2, mode='nearest')
        fea = torch.cat([fea, vfea1], dim=1)
        dn2 = self.relu(self.conv3(fea))
        dn1 = self.relu(self.conv33(dn2))  # 16
        fea = self.sigmoid(self.conv4(dn1))

        return dn2, dn4, dn6, fea

#normal decode
class MCFNet(nn.Module):
    def __init__(self):
        super(MCFNet, self).__init__()
        # encoder
        self.hsv_encode = encode_without_help(1)
        self.rgb_encode = encode_without_help(3)
        self.hsv_Unet = hsv_Unet(3)
        # other
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1 = nn.Conv2d(64 + 64 * 2, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64 + 32 * 2, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64 + 16 * 2, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.out64 = nn.Conv2d(64, 3, 3, 1, 1)
        self.out32 = nn.Conv2d(32, 3, 3, 1, 1)
        self.out16 = nn.Conv2d(16, 3, 3, 1, 1)
        self.relu = nn.LeakyReLU()
        self.down1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.down2 = nn.Conv2d(32, 16, 3, 1, 1)
        #aggregation
        self.aggregation64 = Aggregate_Block(64)
        self.aggregation32 = Aggregate_Block(32)
        self.aggregation16 = Aggregate_Block(16)
        #attention
        self.at1 = attention(64 + 64 * 2)
        self.at2 = attention(64 + 32 * 2)
        self.at3 = attention(64 + 16 * 2)
        self.Block64 = Block(64)
        self.Block32 = Block(32)
        self.Block16 = Block(16)

    def forward(self, x):
        #得到hsv
        h, s, v = rgb2hsv(x)
        #32 3
        hfea1, hfea2, hfea3 = self.hsv_encode(h)
        sfea1, sfea2, sfea3 = self.hsv_encode(s)
        vfea1, vfea2, vfea3 = self.hsv_encode(v)
        rgb_fea1, rgb_fea2, rgb_fea3 = self.rgb_encode(x)
        vfea1, vfea2, vfea3, v = self.hsv_Unet(vfea1, vfea2, vfea3)
        fea = self.relu(self.conv0(rgb_fea3))
        # decode
        hsv_fea3 = self.Block64(hfea3, sfea3, vfea3)
        aggregate_fea3 = self.aggregation64(rgb_fea3, hsv_fea3)# 128
        fea = torch.cat([fea, aggregate_fea3], dim=1)# 128 + 64
        fea = self.at1(fea)
        fea = self.relu(self.conv1(fea))
        fea = self.relu(self.conv2(fea))
        #fea = self.relu(self.down1(fea))# 32
        rgb_x1 = self.out64(fea)
        fea = F.interpolate(fea, scale_factor=2, mode='nearest')

        hsv_fea2 = self.Block32(hfea2, sfea2, vfea2)
        aggregate_fea2 = self.aggregation32(rgb_fea2, hsv_fea2)# 64
        fea = torch.cat([fea, aggregate_fea2], dim=1)# 64 + 32
        fea = self.at2(fea)
        fea = self.relu(self.conv3(fea))
        fea = self.relu(self.conv4(fea))
        #fea = self.relu(self.down2(fea))# 16
        rgb_x2 = self.out64(fea) + F.interpolate(rgb_x1, scale_factor=2, mode='nearest')
        fea = F.interpolate(fea, scale_factor=2, mode='nearest')

        hsv_fea1 = self.Block16(hfea1, sfea1, vfea1)
        aggregate_fea1 = self.aggregation16(rgb_fea1, hsv_fea1)# 32
        fea = torch.cat([fea, aggregate_fea1], dim=1)# 32 + 16
        fea = self.at3(fea)
        fea = self.relu(self.conv5(fea))
        fea = self.relu(self.conv6(fea))
        rgb_x4 = self.out64(fea) + F.interpolate(rgb_x2, scale_factor=2, mode='nearest')

        return v, rgb_x4

model = MCFNet()
img = torch.rand(1,3, 32, 32)
pre = model(img)