import os
import numpy as np
import skimage.io as io
import torch
import lpips
from tqdm import tqdm
import cv2

# ==== 配置路径 ====
name = 'LOL-v2'
if name == 'LOL-v2':
    results_root = "/home/pc/Desktop/LOL-v2/real_captured/TEST"
    gt_root = "/data/low_light_dataset/LOL-v2/Real_captured/Test/Normal"
    input_root = "/data/low_light_dataset/LOL-v2/Real_captured/Test/Low"  # 原始输入图像所在方法文件夹名称
    ph, pw = 64, 128
    save_dir = "LOL/normal"
else:
    results_root = "/home/pc/Desktop/LOL/eval15"
    gt_root = "/data/low_light_dataset/LOL/eval15/high"
    input_root = "/data/low_light_dataset/LOL/eval15/low"  # 原始输入图像所在方法文件夹名称
    ph, pw = 96, 144
    save_dir = "LOL/v1"

our_method = "CTNet"
lpips_margin = 0.01

# 拼接顺序（必须与文件夹名称一致）
method_order = ["LLFormer", "SNR", "RetinexFormer", "MRQ", "CIDNet", "CTNet"]
os.makedirs(save_dir, exist_ok=True)
assert our_method in method_order, f"{our_method} 不在 method_order 中"

loss_fn = lpips.LPIPS(net='alex').cuda()

counter = 0

def load_image(path):
    if path.endswith(".npy"):
        img = np.load(path)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 1:
            img = np.concatenate([img]*3, axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
    else:
        img = io.imread(path)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
    return img

def to_tensor(img):
    return torch.tensor(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()

base_img_list = sorted(os.listdir(os.path.join(results_root, our_method)))
gt_img_list = sorted(os.listdir(os.path.join(gt_root)))
input_img_list = sorted(os.listdir(os.path.join(input_root)))
num_images = min(len(base_img_list), len(gt_img_list), *[
    len(os.listdir(os.path.join(results_root, m))) for m in method_order
])

for i in range(num_images):
    try:
        method_imgs = {}
        lpips_scores = {}

        # === 处理 input 图像 ===
        input_path = os.path.join(input_root, input_img_list[i])
        input_img = load_image(input_path)

        # === 处理 GT 图像 ===
        gt_path = os.path.join(gt_root, gt_img_list[i])
        gt_img = load_image(gt_path)
        gt_tensor = to_tensor(gt_img)

        # === 遍历所有方法图像 ===
        for method in method_order:
            img_list = sorted(os.listdir(os.path.join(results_root, method)))
            if i >= len(img_list):
                raise ValueError(f"{method} 缺第 {i} 张图")
            img_name = img_list[i]
            img_path = os.path.join(results_root, method, img_name)
            img = load_image(img_path)
            method_imgs[method] = img

        # === 拼图并保存 ===
        H, W = 400, 600
        best_margin = -1
        best_coord = (0, 0)

        def concat_with_padding(images, pad=5, axis=1, color=(255, 255, 255)):
            pad_shape = list(images[0].shape)
            pad_shape[axis] = pad
            pad_img = np.full(pad_shape, color, dtype=np.uint8)

            result = images[0]
            for img in images[1:]:
                result = np.concatenate([result, pad_img, img], axis=axis)
            return result

        # === 拼接全图（上方） ===
        row_imgs = [input_img] + [method_imgs[m] for m in method_order] + [gt_img]
        concat_img = concat_with_padding(row_imgs, pad=5, axis=1, color=(255, 255, 255))  # 横向 padding

        # === 保存 ===
        save_name = f"img_{counter:03d}.png"
        print(save_name)
        io.imsave(os.path.join(save_dir, save_name), concat_img)
        counter += 1

    except Exception as e:
        print(f"[跳过第 {i} 张图，错误：{e}")
        continue

print(f"✅ 完成！共保存 {counter} 张我们方法最优的拼图到：{save_dir}")
