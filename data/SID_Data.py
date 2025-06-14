import os
import random
import glob
import numpy as np
import cv2
from torch.utils.data import Dataset

TEST_IDS = set([
    '10022', '10034', '10126', '10055', '10193', '10087', '10016', '10172',
     '10116', '10068', '10032', '10167', '10198', '10178', '10074', '10217',
     '10187', '10139', '10125', '10030', '10006', '10192', '10191', '10103',
     '10163', '10185', '10213', '10176', '10170', '10011', '10101', '10035',
     '10111', '10082', '10054', '10040', '10077', '10199', '10140', '10203',
     '10227', '10105', '10045', '10093', '10228', '10069', '10003', '10106',
     '10162', '10226'
])

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

class SIDTrainDataset(Dataset):
    def __init__(self, root, size=384, transform=None):
        super().__init__()
        self.input_root = os.path.join(root, 'short_sid2')
        self.gt_root = os.path.join(root, 'long_sid2')
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
                self.data.append([input_paths[i], gt_paths[0]])
        print(f"[Train] Total samples: {len(self.data)}")

    def __getitem__(self, index):
        img, label = loader(self.data[index], self.size, 'train')
        return img, label

    def __len__(self):
        return len(self.data)

class SIDTestDataset(Dataset):
    def __init__(self, root, size=384, transform=None):
        super().__init__()
        self.input_root = os.path.join(root, 'short_sid2')
        self.gt_root = os.path.join(root, 'long_sid2')
        self.size = size
        self.transform = transform
        self.data = []

        all_folders = os.listdir(self.input_root)
        print('testing')
        for folder in all_folders:
            if not os.path.isdir(os.path.join(self.input_root, folder)):
                continue
            if folder not in TEST_IDS:
                continue  # skip train folders
            input_folder = os.path.join(self.input_root, folder)
            gt_folder = os.path.join(self.gt_root, folder)
            input_paths = sorted(glob.glob(os.path.join(input_folder, '*')))
            gt_paths = sorted(glob.glob(os.path.join(gt_folder, '*')))
            for i in range(len(input_paths)):
                self.data.append([input_paths[i], gt_paths[0]])
        print(f"[Test] Total samples: {len(self.data)}")

    def __getitem__(self, index):
        img, label = loader(self.data[index], self.size, 'test')
        return img, label

    def __len__(self):
        return len(self.data)
