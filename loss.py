import torch.nn as nn
import torch
import pytorch_ssim
from torch.nn import functional as F

#定义感知loss
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)

#定义感知loss
class Perception_loss(nn.Module):
    def __init__(self):
        super(Perception_loss, self).__init__()
        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, fake_img_hr, target_img_hr):
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(fake_img_hr), self.loss_network(target_img_hr))
        return perception_loss

#SSIM
class ssim_loss(nn.Module):
    def __init__(self):
        super(ssim_loss, self).__init__()
        self.criterion_ssim = pytorch_ssim.SSIM(window_size=11)#SSIM loss
    def forward(self, pre, gt):
        loss = 1 - self.criterion_ssim(pre, gt)
        return loss

#PSNR
class psnr_loss(nn.Module):
    def __init__(self):
        super(psnr_loss, self).__init__()
        self.criterion_mse = nn.MSELoss()

    def forward(self, pre, gt):
        loss = -1 * torch.log10(1 / self.criterion_mse(pre, gt))
        return loss


class KDLoss(nn.Module):
    """
    Args:
        loss_weight (float): Loss weight for KD loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, temperature=0.15):
        super(KDLoss, self).__init__()

        self.loss_weight = loss_weight
        self.temperature = temperature

    def forward(self, S1_fea, S2_fea):
        """
        Args:
            S1_fea (List): contain shape (N, L) vector.
            S2_fea (List): contain shape (N, L) vector.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        loss_KD_dis = 0
        loss_KD_abs = 0
        for i in range(len(S1_fea)):
            S2_distance = F.log_softmax(S2_fea[i] / self.temperature, dim=1)
            S1_distance = F.softmax(S1_fea[i].detach() / self.temperature, dim=1)
            loss_KD_dis += F.kl_div(
                S2_distance, S1_distance, reduction='batchmean')
            loss_KD_abs += nn.L1Loss()(S2_fea[i], S1_fea[i].detach())
        return self.loss_weight * loss_KD_dis, self.loss_weight * loss_KD_abs