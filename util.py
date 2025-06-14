import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def count_dataset():
    file_list = os.listdir('/home/pc/Desktop/SDSD/indoor/RetinexFormer')
    count = 0
    path = '/home/pc/Desktop/SDSD/indoor/RetinexFormer'
    for file in os.listdir(path):
        if file in file_list:
            for npy in os.listdir(os.path.join(path, file)):
                count += 1
    print(count)

    count = 0
    path = '/data/low_light_dataset/SDSD/indoor/indoor_static_np/input'
    for file in os.listdir(path):
        if file not in file_list:
            for npy in os.listdir(os.path.join(path, file)):
                if npy.endswith('.npy'):
                    count += 1
    print(count)

    file_list = os.listdir('/home/pc/Desktop/SDSD/outdoor/RetinexFormer')
    count = 0
    path = '/home/pc/Desktop/SDSD/outdoor/RetinexFormer'
    for file in os.listdir(path):
        if file in file_list:
            for npy in os.listdir(os.path.join(path, file)):
                count += 1
    print(count)

    count = 0
    path = '/data/low_light_dataset/SDSD/outdoor/outdoor_static_np/input'
    for file in os.listdir(path):
        if file not in file_list:
            for npy in os.listdir(os.path.join(path, file)):
                if npy.endswith('.npy'):
                    count += 1
    print(count)

    file_list = os.listdir('/home/pc/Desktop/SID/RetinexFormer')
    count = 0
    path = '/home/pc/Desktop/SID/RetinexFormer'
    for file in os.listdir(path):
        if file in file_list:
            for npy in os.listdir(os.path.join(path, file)):
                count += 1
    print(count)

    count = 0
    path = '/data/low_light_dataset/SID/short_sid2'
    for file in os.listdir(path):
        if file not in file_list:
            for npy in os.listdir(os.path.join(path, file)):
                if npy.endswith('.npy'):
                    count += 1
    print(count)

    file_list = os.listdir('/home/pc/Desktop/SMID/RetinexFormer')
    count = 0
    path = '/home/pc/Desktop/SMID/RetinexFormer'
    for file in os.listdir(path):
        if file in file_list:
            for npy in os.listdir(os.path.join(path, file)):
                count += 1
    print(count)

    count = 0
    path = '/data/low_light_dataset/SMID/smid/SMID_LQ_np'
    for file in os.listdir(path):
        if file not in file_list:
            for npy in os.listdir(os.path.join(path, file)):
                if npy.endswith('.npy'):
                    count += 1
    print(count)

def visualize_npy(path):
    img = np.load(path)
    print(f"Shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")

    # 如果不是 uint8 且值在 0~255 范围内，先转类型
    if img.dtype != np.uint8 and img.max() <= 255:
        img = img.astype(np.uint8)

    # 判断通道顺序
    if img.ndim == 3:
        print(img.shape)
        if img.shape[0] == 3:  # 可能是 CHW 格式
            img = np.transpose(img, (1, 2, 0))
        img = cv2.resize(img, (960, 512))
        print(img.shape)
        print(np.max(img))
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')

    plt.axis('off')
    plt.show()

count_dataset()
#visualize_npy('/data/low_light_dataset/SDSD/indoor/indoor_static_np/GT/pair1/0031.npy')
