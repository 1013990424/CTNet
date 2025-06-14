import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
import pytorch_ssim
from torchvision import transforms
import cv2
from scipy.ndimage import maximum_filter

def get_local_max(image: np.ndarray, win: int = 7) -> np.ndarray:
    return maximum_filter(image, size=(2 * win + 1, 2 * win + 1), mode='reflect')


def compute_loe_matlab_style(input_img: np.ndarray, enhanced_img: np.ndarray) -> float:
    input_img = input_img.astype(np.float64)
    enhanced_img = enhanced_img.astype(np.float64)

    H, W, _ = input_img.shape
    mind = min(H, W)
    step = max(1, int(np.floor(mind / 50)))

    input_light = np.max(input_img, axis=2)
    enhanced_light = np.max(enhanced_img, axis=2)

    input_light = get_local_max(input_light, win=7)
    enhanced_light = get_local_max(enhanced_light, win=7)

    blkm = H // step
    blkn = W // step
    ipic_ds = np.zeros((blkm, blkn))
    epic_ds = np.zeros((blkm, blkn))

    for i in range(blkm):
        for j in range(blkn):
            ipic_ds[i, j] = input_light[i * step, j * step]
            epic_ds[i, j] = enhanced_light[i * step, j * step]

    LOE = np.zeros((blkm, blkn))
    for i in range(blkm):
        for j in range(blkn):
            flag1 = ipic_ds >= ipic_ds[i, j]
            flag2 = epic_ds >= epic_ds[i, j]
            flag = flag1 != flag2
            LOE[i, j] = np.sum(flag)

    return np.mean(LOE)

# LPIPS 初始化
lpips_fn = lpips.LPIPS(net='alex').cuda()

# 归一化函数
def normalization(img):
    img = np.array(img, dtype="float32") / 255.0
    return img

# 图像评估函数
def eval_images(enhanced_root, gt_root, TEST_PREFIXES):
    psnr_list, ssim_list, lpips_list, loe_list = [], [], [], []

    for folder_name in os.listdir(enhanced_root):
        if folder_name not in TEST_PREFIXES:
            continue

        enhanced_folder = os.path.join(enhanced_root, folder_name)
        gt_folder = os.path.join(gt_root, folder_name)

        enhanced_list = sorted(os.listdir(enhanced_folder))
        gt_list = sorted(os.listdir(gt_folder))

        assert len(gt_list) == 1, f"图像数量不一致：{folder_name}"

        for enh_name in enhanced_list:
            enh_path = os.path.join(enhanced_folder, enh_name)
            gt_name = gt_list[0]
            gt_path = os.path.join(gt_folder, gt_name)

            # 读取图像
            enh_img = Image.open(enh_path).convert("RGB")
            gt_img = np.load(gt_path)

            # 转为 numpy 并归一化
            enh_np = np.array(enh_img)
            enh_np = normalization(enh_np)
            gt_np = cv2.resize(gt_img, (960, 512))
            gt_np = normalization(gt_np)

            # PSNR / SSIM / LOE
            psnr_val = psnr(gt_np, enh_np, data_range=1.0)
            ssim_val = ssim(gt_np, enh_np, data_range=1.0, channel_axis=-1)
            loe_val = compute_loe_matlab_style(gt_np, enh_np)

            # LPIPS
            transform = transforms.ToTensor()
            enh_tensor = transform(enh_np).unsqueeze(0).cuda()
            gt_tensor = transform(gt_np).unsqueeze(0).cuda()
            lpips_val = lpips_fn(gt_tensor, enh_tensor).item()

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            loe_list.append(loe_val)
            lpips_list.append(lpips_val)

            print(f"{folder_name}/{enh_name} — PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}")

    print("\n====== 测试完成 ======")
    print(f"平均 PSNR : {np.mean(psnr_list):.2f}")
    print(f"平均 SSIM : {np.mean(ssim_list):.4f}")
    print(f"平均 LOE : {np.mean(loe_list):.4f}")
    print(f"平均 LPIPS: {np.mean(lpips_list):.4f}")


# ---------------- 设置路径 ----------------
TEST_PREFIXES = set([
    '10022', '10034', '10126', '10055', '10193', '10087', '10016', '10172',
     '10116', '10068', '10032', '10167', '10198', '10178', '10074', '10217',
     '10187', '10139', '10125', '10030', '10006', '10192', '10191', '10103',
     '10163', '10185', '10213', '10176', '10170', '10011', '10101', '10035',
     '10111', '10082', '10054', '10040', '10077', '10199', '10140', '10203',
     '10227', '10105', '10045', '10093', '10228', '10069', '10003', '10106',
     '10162', '10226'
])

# 设置增强结果路径和 GT 图像路径（均为 PNG / JPG 格式图像）
enhanced_path = '/home/pc/Desktop/SID/CTNet'
gt_path = '/data/low_light_dataset/SID/long_sid2'

eval_images(enhanced_path, gt_path, TEST_PREFIXES)
