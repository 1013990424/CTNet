import os
import torch
import numpy as np
from PIL import Image
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr
import pytorch_ssim
from CTNet import Net
import lpips  # LPIPS 需要网络支持
import cv2
from torchvision import transforms
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

TEST_IDS = set([
    '10022', '10034', '10126', '10055', '10193', '10087', '10016', '10172',
     '10116', '10068', '10032', '10167', '10198', '10178', '10074', '10217',
     '10187', '10139', '10125', '10030', '10006', '10192', '10191', '10103',
     '10163', '10185', '10213', '10176', '10170', '10011', '10101', '10035',
     '10111', '10082', '10054', '10040', '10077', '10199', '10140', '10203',
     '10227', '10105', '10045', '10093', '10228', '10069', '10003', '10106',
     '10162', '10226'
])
# ---------------- 模型初始化 ----------------
model = Net(num_blocks=[2, 2, 4], dim=48)
model_path = '../output/SID.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=True)
model = model.cuda()
model.eval()

print("模型加载完成。")

# ---------------- LPIPS 初始化 ----------------
lpips_fn = lpips.LPIPS(net='alex').cuda()

# ---------------- 归一化函数 ----------------
def normalization(img):
    img = np.array(img, dtype="float32")
    img = (img * 1.0) / 255
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
    return img_norm

# ---------------- SDSD 测试函数 ----------------
def SDSD_eval(input_root, gt_root, output_root, TEST_PREFIXES):
    os.makedirs(output_root, exist_ok=True)

    psnr_list, ssim_list, lpips_list, loe_list = [], [], [], []

    for folder_name in os.listdir(input_root):
        if folder_name not in TEST_PREFIXES:
            continue
        input_folder = os.path.join(input_root, folder_name)
        gt_folder = os.path.join(gt_root, folder_name)
        output_folder = os.path.join(output_root, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        input_list = sorted(os.listdir(input_folder))
        gt_list = sorted(os.listdir(gt_folder))

        assert len(gt_list) == 1, f"图像数量不一致：{folder_name}"

        for img_name in input_list:
            input_path = os.path.join(input_folder, img_name)
            gt_name = gt_list[0]
            gt_path = os.path.join(gt_folder, gt_name)

            # 读取和预处理低光图像
            img = np.load(input_path)
            img = cv2.resize(img, (960, 512))
            img = normalization(img)

            img_tensor = torch.from_numpy(np.moveaxis(img, -1, 0)).float().unsqueeze(0).cuda()

            with torch.no_grad():
                _, output = model(img_tensor)
                output = output.squeeze().clamp(0, 1).cpu().numpy()
                output_img = np.moveaxis(output, 0, -1)

            output_uint8 = (output_img * 255).astype(np.uint8)
            save_name = img_name.split('.')[0] + '.png'
            Image.fromarray(output_uint8).save(os.path.join(output_folder, save_name))

            # 读取并裁剪 GT 图像
            gt_img = np.load(gt_path)
            gt_img = cv2.resize(gt_img, (960, 512))
            gt_img = np.array(gt_img, dtype="float32")
            gt_img = gt_img / 255.0

            # PSNR / SSIM / LOE
            psnr_val = psnr(gt_img, output_img, data_range=1.0)
            loe_val = compute_loe_matlab_style(gt_img, output_img)

            # LPIPS
            transform = transforms.ToTensor()
            enh_tensor = transform(output_img).unsqueeze(0).cuda()
            gt_tensor = transform(gt_img).unsqueeze(0).cuda()
            lpips_val = lpips_fn(gt_tensor, enh_tensor).item()

            # SSIM
            ssim_val = pytorch_ssim.ssim(enh_tensor, gt_tensor).item()

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            loe_list.append(loe_val)
            lpips_list.append(lpips_val)

            print(f"{folder_name}/{img_name} — PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}")

    print("\n====== 测试完成 ======")
    print(f"平均 PSNR : {np.mean(psnr_list):.2f}")
    print(f"平均 SSIM : {np.mean(ssim_list):.4f}")
    print(f"平均 LOE : {np.mean(loe_list):.4f}")
    print(f"平均 LPIPS: {np.mean(lpips_list):.4f}")


# ---------------- 设置路径 ----------------
input_path = "/data/low_light_dataset/SID/short_sid2"
gt_path = "/data/low_light_dataset/SID/long_sid2"
output_path = "/home/pc/Desktop/SID/CTNet"

SDSD_eval(input_path, gt_path, output_path, TEST_IDS)
