import os
import numpy as np
import torch
import math
import pytorch_ssim
import lpips
from skimage import io
from tqdm import tqdm
import cv2

from torchvision import transforms

# 模型输出图路径前缀和GT路径
RESULTS_BASE = "/home/pc/Desktop/LOL-v2/real_captured/TEST/"
GT_PATH = "/data/low_light_dataset/LOL-v2/Real_captured/Test/Normal/"

# 转换为 Tensor
to_tensor = transforms.ToTensor()

# LPIPS 模型
lpips_model = lpips.LPIPS(net='alex').cuda()

# PSNR
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# 单个模型评估函数
def evaluate_folder(model_name):
    pred_dir = os.path.join(RESULTS_BASE, model_name)
    gt_dir = GT_PATH

    psnr_sum, ssim_sum, lpips_sum = 0, 0, 0
    count = 0

    for img_name in tqdm(os.listdir(pred_dir), desc=f"Evaluating {model_name}"):
        pred_path = os.path.join(pred_dir, img_name)
        gt_name = 'normal' + img_name.split('low')[1] if 'low' in img_name else img_name
        gt_path = os.path.join(gt_dir, gt_name)

        if not os.path.exists(gt_path):
            continue

        pred = io.imread(pred_path).astype(np.float32) / 255.0
        gt = io.imread(gt_path).astype(np.float32) / 255.0

        # 尺寸对齐
        h, w = min(pred.shape[0], gt.shape[0]), min(pred.shape[1], gt.shape[1])
        pred = pred[:h, :w]
        gt = gt[:h, :w]

        # 灰度均衡调整
        mean_gray_pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY).mean()
        mean_gray_gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).mean()
        pred = np.clip(pred * (mean_gray_gt / (mean_gray_pred + 1e-8)), 0, 1)

        # 指标计算
        psnr_score = psnr(pred, gt)
        pred_tensor = to_tensor(pred).unsqueeze(0).cuda()
        gt_tensor = to_tensor(gt).unsqueeze(0).cuda()
        ssim_score = pytorch_ssim.ssim(pred_tensor, gt_tensor).item()
        lpips_score = lpips_model(pred_tensor * 2 - 1, gt_tensor * 2 - 1).item()

        psnr_sum += psnr_score
        ssim_sum += ssim_score
        lpips_sum += lpips_score
        count += 1

    return psnr_sum / count, ssim_sum / count, lpips_sum / count

# ========== 主程序 ========== #
if __name__ == "__main__":
    model_names = ["Restormer", "MIRNet", "NAFNet", "MambaIR", "UFormer", "RetinexFormer", "SNR"]

    print("Model\tPSNR\tSSIM\tLPIPS")
    for name in model_names:
        avg_psnr, avg_ssim, avg_lpips = evaluate_folder(name)
        print(f"{name}\t{avg_psnr:.4f}\t{avg_ssim:.4f}\t{avg_lpips:.4f}")
