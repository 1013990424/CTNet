import os
import numpy as np
import skimage.io as io
import torch
import lpips
from tqdm import tqdm
import cv2

# ==== 配置路径 ====
name = 'SID'
if name == 'SID':
    results_root = "/home/pc/Desktop/SID"
    gt_root = "/data/low_light_dataset/SID/long_sid2"
    save_dir = "SDSD_SID/SID"
    input_root = "/data/low_light_dataset/SID/short_sid2"  # 原始输入图像所在方法文件夹名称
if name == 'SDSD_outdoor':
    results_root = "/home/pc/Desktop/SDSD/outdoor"
    gt_root = "/data/low_light_dataset/SDSD/outdoor/outdoor_static_np/GT"
    save_dir = "SDSD_SID/SDSD_outdoor"
    input_root = "/data/low_light_dataset/SDSD/outdoor/outdoor_static_np/input"  # 原始输入图像所在方法文件夹名称
if name == 'SDSD_indoor':
    results_root = "/home/pc/Desktop/SDSD/indoor"
    gt_root = "/data/low_light_dataset/SDSD/indoor/indoor_static_np/GT"
    save_dir = "SDSD_SID/SDSD_indoor"
    input_root = "/data/low_light_dataset/SDSD/indoor/indoor_static_np/input"  # 原始输入图像所在方法文件夹名称
our_method = "CTNet"
ph, pw = 128, 256
lpips_margin = 0.04

# 拼接顺序（必须与文件夹名称一致）
method_order = ["LLFormer", "SNR", "RetinexFormer", "MRQ", "CIDNet", "CTNet"]
os.makedirs(save_dir, exist_ok=True)
assert our_method in method_order, f"{our_method} 不在 method_order 中"

loss_fn = lpips.LPIPS(net='alex').cuda()

# 遍历 pair 文件夹（以我们方法为基准）
pair_dirs = sorted([d for d in os.listdir(os.path.join(results_root, our_method)) if os.path.isdir(os.path.join(results_root, our_method, d))])

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

for pair in tqdm(pair_dirs):
    # 确保每个方法都有这个 pair
    if not all(os.path.exists(os.path.join(results_root, m, pair)) for m in method_order):
        continue
    if not os.path.exists(os.path.join(gt_root, pair)):
        continue
    if not os.path.exists(os.path.join(input_root, pair)):
        continue
    # 获取图像列表（以 CTNet 为基准）
    base_img_list = sorted(os.listdir(os.path.join(results_root, our_method, pair)))
    gt_img_list = sorted(os.listdir(os.path.join(gt_root, pair)))
    input_img_list = sorted(os.listdir(os.path.join(input_root, pair)))
    num_images = min(len(base_img_list), len(gt_img_list), *[
        len(os.listdir(os.path.join(results_root, m, pair))) for m in method_order
    ])

    for i in range(num_images):
        try:
            method_imgs = {}
            lpips_scores = {}

            # === 处理 input 图像 ===
            input_path = os.path.join(input_root, pair, input_img_list[i])
            input_img = load_image(input_path)
            input_img = cv2.resize(input_img, (960, 512))

            # === 处理 GT 图像 ===
            gt_path = os.path.join(gt_root, pair, gt_img_list[i])
            gt_img = load_image(gt_path)
            gt_img = cv2.resize(gt_img, (960, 512))
            gt_tensor = to_tensor(gt_img)

            # === 遍历所有方法图像 ===
            for method in method_order:
                img_list = sorted(os.listdir(os.path.join(results_root, method, pair)))
                if i >= len(img_list):
                    raise ValueError(f"{method} 缺第 {i} 张图")
                img_name = img_list[i]
                img_path = os.path.join(results_root, method, pair, img_name)
                img = load_image(img_path)
                img = cv2.resize(img, (960, 512))
                method_imgs[method] = img
                pred_tensor = to_tensor(img)
                lpips_scores[method] = loss_fn(gt_tensor, pred_tensor).item()

            # === 如果我们不是最优，或者优势不明显，跳过 ===
            sorted_methods = sorted(lpips_scores.items(), key=lambda x: x[1])
            best_method, best_score = sorted_methods[0]
            second_best_score = sorted_methods[1][1]

            if best_method != our_method or (second_best_score - best_score) < lpips_margin:
                continue

            # === 拼图并保存 ===
            H, W = 512, 960

            best_margin = -1
            best_coord = (0, 0)

            for y in range(0, H, ph):
                for x in range(0, W, pw):
                    gt_patch = to_tensor(gt_img[y:y + ph, x:x + pw])
                    our_patch = to_tensor(method_imgs[our_method][y:y + ph, x:x + pw])
                    our_score = loss_fn(gt_patch, our_patch).item()
                    worst_score = min([
                        loss_fn(gt_patch, to_tensor(method_imgs[m][y:y + ph, x:x + pw])).item()
                        for m in method_order if m != our_method
                    ])
                    margin = worst_score - our_score
                    if margin > best_margin:
                        best_margin = margin
                        best_coord = (x, y)

            crop_x, crop_y = best_coord
            crop_w, crop_h = pw, ph

            # === 在每张图上绘制红框（包括 input、每个方法、gt）===
            def draw_red_rect(img, x, y, w, h):
                return cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (255, 0, 0), 10)

            input_img_boxed = draw_red_rect(input_img, crop_x, crop_y, crop_w, crop_h)
            gt_img_boxed = draw_red_rect(gt_img, crop_x, crop_y, crop_w, crop_h)

            method_imgs_boxed = {}
            for m in method_order:
                method_imgs_boxed[m] = draw_red_rect(method_imgs[m], crop_x, crop_y, crop_w, crop_h)

            # === 拼接全图行（含框），添加5像素分隔条 ===
            def concat_with_padding(images, pad=5, axis=1, color=(255, 255, 255)):
                pad_shape = list(images[0].shape)
                pad_shape[axis] = pad
                pad_img = np.full(pad_shape, color, dtype=np.uint8)

                result = images[0]
                for img in images[1:]:
                    result = np.concatenate([result, pad_img, img], axis=axis)
                return result

            # === 拼接全图（上方） ===
            row_imgs = [input_img_boxed] + [method_imgs_boxed[m] for m in method_order] + [gt_img_boxed]
            concat_img = concat_with_padding(row_imgs, pad=5, axis=1, color=(255, 255, 255))  # 横向 padding

            # === 1. 计算缩放后的 patch 尺寸 ===
            zoom_patches = []
            zoom_patches.append(input_img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w])
            for m in method_order:
                zoom_patch = method_imgs[m][crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
                zoom_patches.append(zoom_patch)
            zoom_patches.append(gt_img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w])

            h_concat, w_concat, _ = concat_img.shape
            num_patches = len(zoom_patches)
            num_gaps = num_patches - 1
            gap_width = 5

            # 计算每个 patch 的目标宽度（总宽度要减去 gap 总宽度）
            target_patch_width = (w_concat - gap_width * num_gaps) // num_patches
            target_patch_height = int(crop_h * (target_patch_width / crop_w))  # 等比例缩放

            # === 2. 逐个缩放 patch，插入 5px 分隔 ===
            resized_zoom_patches = []
            for i, patch in enumerate(zoom_patches):
                patch_resized = cv2.resize(patch, (target_patch_width, target_patch_height), interpolation=cv2.INTER_AREA)
                resized_zoom_patches.append(patch_resized)
                if i != num_patches - 1:
                    resized_zoom_patches.append(
                        np.full((target_patch_height, gap_width, 3), 255, dtype=np.uint8))  # 白色间隔

            # === 3. 拼接 zoom row ===
            zoom_row_final = np.concatenate(resized_zoom_patches, axis=1)

            # === 4. 添加上下间隔，并拼接最终图像 ===
            separator = np.full((5, w_concat, 3), 255, dtype=np.uint8)  # 白色分隔条
            final_img = np.concatenate([concat_img, separator, zoom_row_final], axis=0)

            # === 保存 ===
            save_name = f"{pair}_img_{i:03d}.png"
            print(save_name)
            io.imsave(os.path.join(save_dir, save_name), final_img)
            counter += 1

        except Exception as e:
            print(f"[跳过] {pair} 第 {i} 张图，错误：{e}")
            continue

print(f"✅ 完成！共保存 {counter} 张我们方法最优的拼图到：{save_dir}")
