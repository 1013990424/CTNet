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

class hsv_Unet(nn.Module):
    def __init__(self, out_dim):
        super(hsv_Unet, self).__init__()
        # decoder
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1 = nn.Conv2d(64 * 2, 64, 3, 1, 1)
        self.conv11 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(32 * 1 + 64, 64, 3, 1, 1)
        self.conv22 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(16 * 1 + 64, 64, 3, 1, 1)
        self.conv33 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 1, 3, 1, 1)
        self.out = nn.Conv2d(3, 3, 3, 1, 1)
        # other
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.hsv_encode = encode_without_help(1)

    def forward(self, x):
        # 得到hsv
        h, s, v = rgb2hsv(x)
        # 32 3
        vfea1, vfea2, vfea3 = self.hsv_encode(v)
        fea = self.relu(self.conv0(vfea3))
        # decode
        fea = torch.cat([fea, vfea3], dim=1)
        dn6 = self.relu(self.conv1(fea))
        dn5 = self.relu(self.conv11(dn6))  # 64
        fea = F.interpolate(dn5, scale_factor=2, mode='nearest')
        fea = torch.cat([fea, vfea2], dim=1)
        dn4 = self.relu(self.conv2(fea))
        dn3 = self.relu(self.conv22(dn4))  # 32
        fea = F.interpolate(dn3, scale_factor=2, mode='nearest')
        fea = torch.cat([fea, vfea1], dim=1)
        dn2 = self.relu(self.conv3(fea))
        dn1 = self.relu(self.conv33(dn2))  # 16
        v = self.sigmoid(self.conv4(dn1))
        fea = torch.cat([h, s, v], dim=1)
        fea = self.sigmoid(self.out(fea))

        return v, fea

#normal decode
class MCFNet(nn.Module):
    def __init__(self):
        super(MCFNet, self).__init__()
        # encoder
        self.hsv_encode = encode_without_help(1)
        self.rgb_encode = encode_without_help(3)
        self.hsv_Unet = hsv_Unet(1)
        # other
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1 = nn.Conv2d(64 + 64, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64 + 32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64 + 16, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.out64 = nn.Conv2d(64, 3, 3, 1, 1)
        self.out32 = nn.Conv2d(32, 3, 3, 1, 1)
        self.out16 = nn.Conv2d(16, 3, 3, 1, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        #32 3
        v, hsv = self.hsv_Unet(x)
        rgb_fea1, rgb_fea2, rgb_fea3 = self.rgb_encode(x)
        fea = self.relu(self.conv0(rgb_fea3))
        # decode
        fea = torch.cat([fea, rgb_fea3], dim=1)  # 128 + 64
        fea = self.relu(self.conv1(fea))
        fea = self.relu(self.conv2(fea))
        rgb_x1 = self.out64(fea)
        fea = F.interpolate(fea, scale_factor=2, mode='nearest')

        fea = torch.cat([fea, rgb_fea2], dim=1)# 64 + 32
        fea = self.relu(self.conv3(fea))
        fea = self.relu(self.conv4(fea))
        rgb_x2 = self.out64(fea) + F.interpolate(rgb_x1, scale_factor=2, mode='nearest')
        fea = F.interpolate(fea, scale_factor=2, mode='nearest')

        fea = torch.cat([fea, rgb_fea1], dim=1)# 32 + 16
        fea = self.relu(self.conv5(fea))
        fea = self.relu(self.conv6(fea))
        rgb_x4 = self.out64(fea) + F.interpolate(rgb_x2, scale_factor=2, mode='nearest')
        rgb_x4 = rgb_x4 + hsv
        
        return v, rgb_x4
