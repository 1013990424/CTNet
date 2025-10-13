import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out

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

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)

class Block(nn.Module):
    def __init__(self, input_dim):
        super(Block, self).__init__()
        self.extended_dim = input_dim - input_dim % 6 + 6
        self.group_dim = self.extended_dim // 6

        # conv layers
        self.conv_fusion1 = nn.Conv2d(input_dim, input_dim, 1, 1, 0)
        self.conv_fusion2 = nn.Conv2d(input_dim, input_dim, 1, 1, 0)
        self.conv_fusion3 = nn.Conv2d(input_dim, input_dim, 1, 1, 0)
        self.conv_fusion4 = nn.Conv2d(input_dim, input_dim, 1, 1, 0)
        self.conv_expand = nn.Conv2d(input_dim, self.extended_dim, 1, 1, 0)
        self.conv_group = nn.Conv2d(self.group_dim, input_dim * 2, 1, 1, 0)
        self.conv_merge1 = nn.Conv2d(input_dim * 12, input_dim * 2, 1, 1, 0)
        self.conv_merge2 = nn.Conv2d(input_dim * 3, input_dim, 1, 1, 0)

        self.act_lrelu = nn.LeakyReLU()
        self.act_sigmoid = nn.Sigmoid()

    def forward(self, high_feat, spatial_feat, var_feat):
        """Feature fusion"""
        cross_feat = spatial_feat * var_feat
        cross_feat = self.act_lrelu(self.conv_fusion1(cross_feat))

        fusion_feat = cross_feat * self.act_lrelu(self.conv_fusion2(high_feat))
        fusion_feat = self.act_lrelu(self.conv_fusion3(fusion_feat))

        diff_feat = var_feat - cross_feat
        diff_feat = self.act_lrelu(self.conv_fusion4(diff_feat))

        """Feature expansion"""
        high_feat = self.act_lrelu(self.conv_expand(high_feat))

        """Group processing"""
        b, c, h, w = high_feat.data.size()
        high_feat = high_feat.reshape(b, self.group_dim, 6, h, w)
        high_feat = high_feat.permute(0, 2, 1, 3, 4)  # swap group dimension
        high_feat = high_feat.reshape(b, c, h, w)

        group_feats = []
        high_feat_groups = torch.chunk(high_feat, 6, 1)
        fusion_input = torch.cat([fusion_feat, cross_feat], dim=1)

        for group in high_feat_groups:
            group_output = self.act_sigmoid(self.conv_group(group)) * fusion_input
            group_feats.append(group_output)

        combined_feat = torch.cat(group_feats, dim=1)

        """Tail fusion"""
        combined_feat = self.act_lrelu(self.conv_merge1(combined_feat))
        combined_feat = torch.cat([combined_feat, diff_feat], dim=1)
        combined_feat = self.act_lrelu(self.conv_merge2(combined_feat))

        return combined_feat


class AggregateBlock(nn.Module):
    def __init__(self, input_dim):
        super(AggregateBlock, self).__init__()
        # mask generation
        self.conv_rgb_mask = nn.Conv2d(input_dim, 1, 3, 1, 1)
        self.conv_hsv_mask = nn.Conv2d(input_dim, 1, 3, 1, 1)

        # fusion
        self.conv_fusion = nn.Conv2d(input_dim * 2, input_dim, 1, 1, 0)

        # attention layers
        self.fc_rgb_down = nn.Conv2d(input_dim, input_dim // 16, kernel_size=1)
        self.fc_rgb_up = nn.Conv2d(input_dim // 16, input_dim, kernel_size=1)
        self.fc_hsv_down = nn.Conv2d(input_dim, input_dim // 16, kernel_size=1)
        self.fc_hsv_up = nn.Conv2d(input_dim // 16, input_dim, kernel_size=1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.act_sigmoid = nn.Sigmoid()
        self.act_lrelu = nn.LeakyReLU()

    def forward(self, rgb_feat, hsv_feat):
        diff_feat = rgb_feat - hsv_feat

        # RGB attention
        rgb_vec = self.global_pool(diff_feat)
        rgb_vec = self.act_lrelu(self.fc_rgb_down(rgb_vec))
        rgb_vec = self.fc_rgb_up(rgb_vec)
        rgb_vec = self.act_sigmoid(rgb_vec)
        rgb_mask = self.act_sigmoid(self.conv_rgb_mask(diff_feat))

        # HSV attention
        hsv_vec = self.global_pool(diff_feat)
        hsv_vec = self.act_lrelu(self.fc_hsv_down(hsv_vec))
        hsv_vec = self.fc_hsv_up(hsv_vec)
        hsv_vec = self.act_sigmoid(hsv_vec)
        hsv_mask = self.act_sigmoid(self.conv_hsv_mask(diff_feat))

        # apply attention
        rgb_feat = hsv_mask * hsv_vec * rgb_feat + rgb_feat
        hsv_feat = rgb_mask * rgb_vec * hsv_feat + hsv_feat

        fused_feat = torch.cat([rgb_feat, hsv_feat], dim=1)
        fused_feat = self.conv_fusion(fused_feat)

        return fused_feat

class MS_MSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MS_MSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)

        return x

def default_conv(in_channels, out_channels, kernel_size=3, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class PALayer(nn.Module):
    def __init__(self, channel, bias=False):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=bias),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)

        return x * y

class CALayer(nn.Module):
    def __init__(self, channel, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)

        return x * y

def default_conv(in_channels, out_channels, kernel_size=3, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class FABlock(nn.Module):
    def __init__(self, dim, conv=default_conv, kernel_size=3, bias=False):
        super(FABlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=bias)
        self.conv2 = conv(dim, dim, kernel_size, bias=bias)
        self.conv3 = conv(dim * 2, dim, 1, bias=bias)
        self.act = nn.LeakyReLU()
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act(res)
        res = self.conv2(res)
        ca_branch = self.calayer(res)
        pa_branch = self.palayer(res)
        res = torch.cat([pa_branch, ca_branch], dim=1)
        res = self.conv3(res)
        res += x

        return res

class MSAB(nn.Module):
    def __init__(self, dim, num_blocks, num_head, bias):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, num_heads=num_head, bias=bias),
                FeedForward(dim=dim),
            ]))
        self.norm = LayerNorm(dim)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        for (attn, ff) in self.blocks:
            x = attn(self.norm(x)) + x
            x = ff(self.norm(x)) + x

        return x

class Net(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=64, stage=2, num_blocks=[2, 4, 6], bias=False):
        super(Net, self).__init__()
        self.dim = dim
        self.stage = stage

        # -------------------HSV----------------------------
        self.h_embedding = nn.Sequential(
            nn.Conv2d(1, self.dim, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False),
        )

        self.s_embedding = nn.Sequential(
            nn.Conv2d(1, self.dim, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False),
        )

        self.v_embedding = nn.Sequential(
            nn.Conv2d(1, self.dim, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False),
        )
        # Encoder
        self.H_encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.H_encoder_layers.append(nn.ModuleList([
                FABlock(dim=dim_stage),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        self.S_encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.S_encoder_layers.append(nn.ModuleList([
                FABlock(dim=dim_stage),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        self.V_encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.V_encoder_layers.append(nn.ModuleList([
                FABlock(dim=dim_stage),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        self.V_decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.V_decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                FABlock(dim=dim_stage // 2),
            ]))
            dim_stage //= 2

        dim_stage = dim
        self.trasfrom_layers = nn.ModuleList([])
        for i in range(stage):
            self.trasfrom_layers.append(Block(dim_stage))
            dim_stage *= 2
        # -------------------RGB----------------------------

        # Input projection
        self.embedding = nn.Sequential(
            nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False),
        )

        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(
                    dim=dim_stage, num_blocks=num_blocks[i], num_head=dim_stage // dim, bias=bias),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck（修改部分：添加 DINO 特征融合）
        self.bottleneck = MSAB(
            dim=dim_stage, num_blocks=num_blocks[-1],  num_head=dim_stage // dim, bias=bias)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                Aggregate_Block(dim_stage // 2),
                MSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], num_head=dim_stage // dim, bias=bias),
            ]))
            dim_stage //= 2
        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.v_mapping = nn.Conv2d(self.dim, 1, 3, 1, 1, bias=False)
        #### activation function
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # -------------------HSV----------------------------
        h, s, v = rgb2hsv(x)
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Embedding
        hfea = self.h_embedding(h)
        sfea = self.s_embedding(s)
        vfea = self.v_embedding(v)

        hfea_encoder = []
        for (MSAB, FeaDownSample) in self.H_encoder_layers:
            hfea = MSAB(hfea)
            hfea_encoder.append(hfea)
            hfea = FeaDownSample(hfea)

        sfea_encoder = []
        for (MSAB, FeaDownSample) in self.S_encoder_layers:
            sfea = MSAB(sfea)
            sfea_encoder.append(sfea)
            sfea = FeaDownSample(sfea)

        vfea_encoder = []
        for (MSAB, FeaDownSample) in self.V_encoder_layers:
            vfea = MSAB(vfea)
            vfea_encoder.append(vfea)
            vfea = FeaDownSample(vfea)

        vfea_decoder = []
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.V_decoder_layers):
            vfea = FeaUpSample(vfea)
            vfea = Fution(torch.cat([vfea, vfea_encoder[self.stage - 1 - i]], dim=1))
            vfea = LeWinBlcok(vfea)
            vfea_decoder.append(vfea)

        v = self.v_mapping(vfea)
        hsv_fea = []
        for i, (Fution) in enumerate(self.trasfrom_layers):
            fusion_fea = Fution(hfea_encoder[i], sfea_encoder[i], vfea_decoder[self.stage - i - 1])
            hsv_fea.append(fusion_fea)

        # -------------------RGB----------------------------
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for (MSAB, FeaDownSample) in self.encoder_layers:
            fea = MSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, Aggregate, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            aggregate_fea = Aggregate(fea_encoder[self.stage-1-i], hsv_fea[self.stage-1-i])
            fea = Fution(torch.cat([fea, aggregate_fea], dim=1))
            fea = LeWinBlcok(fea)

        # Mapping
        out = self.mapping(fea)

        return v, out

def measure_model(model, input_size=(1, 3, 224, 224), device='cuda:2'):
    from thop import profile, clever_format
    import time
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(input_size).to(device)

    # 参数量和 FLOPs
    flops, params = profile(model, inputs=(dummy_input,))
    flops, params = clever_format([flops, params], "%.3f")

    # 推理时间（多次取平均）
    with torch.no_grad():
        times = []
        for _ in range(100):
            start_time = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize(device=device)  # 指定 GPU 同步
            end_time = time.time()
            times.append(end_time - start_time)
        avg_time = sum(times) / len(times)

    print(f"Params: {params}")
    print(f"FLOPs: {flops}")
    print(f"Inference Time (on {device}): {avg_time * 1000:.3f} ms")

    return params, flops, avg_time

if __name__ == '__main__':
    model = Net(num_blocks=[2,2,4], dim=48)
    measure_model(model)
