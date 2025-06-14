import os
import random
import numpy as np
import glob
import cv2
import torch
from torch.utils.data import Dataset

TEST_PREFIXES = [
    'pair21', 'pair4', 'pair11_2', 'pair1', 'pair9_2',
    'pair19_2', 'pair11', 'pair4_2', 'pair19', 'pair1_2', 'pair21_2', 'pair9'
]

def normalization(img):
    img = np.array(img, dtype="float32")
    img = (img * 1.0) / 255
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
    return img_norm

def data_augmentation(image, mode):
    if mode == 0:
        return image
    elif mode == 1:
        return np.flipud(image)
    elif mode == 2:
        return np.rot90(image)
    elif mode == 3:
        return np.flipud(np.rot90(image))
    elif mode == 4:
        return np.rot90(image, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(image, k=2))
    elif mode == 6:
        return np.rot90(image, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(image, k=3))

def loader(path, size, type):
    img = np.load(path[0])
    label = np.load(path[1])
    if type == 'train':
        img = cv2.resize(img, (960, 512))
        label = cv2.resize(label, (960, 512))
        rh = np.random.randint(0, img.shape[0] - size)
        rw = np.random.randint(0, img.shape[1] - size)
        img = img[rh: rh + size, rw: rw + size]
        label = label[rh: rh + size, rw: rw + size]
        # 数据增强
        rand_mode = random.randint(0, 7)
        img = data_augmentation(img, rand_mode)
        label = data_augmentation(label, rand_mode)
    else:  # test mode
        img = cv2.resize(img, (960, 512))
        label = cv2.resize(label, (960, 512))
    img = normalization(img)
    label = np.array(label, dtype="float32") / 255.0

    img = img.transpose(2, 0, 1).astype(np.float32)
    label = label.transpose(2, 0, 1).astype(np.float32)

    return img, label

class SDSDTrainDataset(Dataset):
    def __init__(self, root, size=384, transform=None):
        super().__init__()
        self.input_root = os.path.join(root, 'input')
        self.gt_root = os.path.join(root, 'GT')
        self.size = size
        self.transform = transform
        self.data = []

        all_folders = os.listdir(self.input_root)
        print('training')
        for folder in all_folders:
            if not os.path.isdir(os.path.join(self.input_root, folder)):
                continue
            if folder in TEST_PREFIXES:
                continue  # skip test folders
            input_folder = os.path.join(self.input_root, folder)
            gt_folder = os.path.join(self.gt_root, folder)
            input_paths = sorted(glob.glob(os.path.join(input_folder, '*')))
            gt_paths = sorted(glob.glob(os.path.join(gt_folder, '*')))
            for i in range(len(input_paths)):
                self.data.append([input_paths[i], gt_paths[i]])
        print(f"[Train] Total samples: {len(self.data)}")

    def __getitem__(self, index):
        img, label = loader(self.data[index], self.size, 'train')
        return img, label

    def __len__(self):
        return len(self.data)

class SDSDTestDataset(Dataset):
    def __init__(self, root, size=384, transform=None):
        super().__init__()
        self.input_root = os.path.join(root, 'input')
        self.gt_root = os.path.join(root, 'GT')
        self.size = size
        self.transform = transform
        self.data = []

        all_folders = os.listdir(self.input_root)
        print('testing')
        for folder in all_folders:
            if not os.path.isdir(os.path.join(self.input_root, folder)):
                continue
            if folder not in TEST_PREFIXES:
                continue  # skip train folders
            input_folder = os.path.join(self.input_root, folder)
            gt_folder = os.path.join(self.gt_root, folder)
            input_paths = sorted(glob.glob(os.path.join(input_folder, '*')))
            gt_paths = sorted(glob.glob(os.path.join(gt_folder, '*')))
            for i in range(len(input_paths)):
                self.data.append([input_paths[i], gt_paths[i]])
        print(f"[Test] Total samples: {len(self.data)}")

    def __getitem__(self, index):
        img, label = loader(self.data[index], self.size, 'test')
        return img, label

    def __len__(self):
        return len(self.data)
