import os
import random
import glob
import numpy as np
import cv2
from torch.utils.data import Dataset

SMID_TEST_IDS = set([
    '0106', '0170', '0166', '0129', '0009', '0075', '0037', '0039', '0092', '0104', '0051',
    '0052', '0177', '0048', '0167', '0154', '0007', '0020', '0036', '0066', '0180', '0172',
    '0105', '0099', '0147', '0091', '0012', '0196', '0169', '0139', '0047', '0013', '0088',
    '0050', '0145', '0107', '0010', '0151', '0076', '0175', '0191', '0065', '0083', '0078',
    '0157', '0103', '0181', '0049', '0153'
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

class SMIDTrainDataset(Dataset):
    def __init__(self, root, size=384, transform=None):
        super().__init__()
        self.input_root = os.path.join(root, 'SMID_LQ_np')
        self.gt_root = os.path.join(root, 'SMID_Long_np')
        self.size = size
        self.transform = transform
        self.data = []

        all_folders = os.listdir(self.input_root)
        print('training')
        for folder in all_folders:
            if not os.path.isdir(os.path.join(self.input_root, folder)):
                continue
            # if folder in TEST_PREFIXES:
            #     continue  # skip test folders
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

class SMIDTestDataset(Dataset):
    def __init__(self, root, size=384, transform=None):
        super().__init__()
        self.input_root = os.path.join(root, 'SMID_LQ_np')
        self.gt_root = os.path.join(root, 'SMID_Long_np')
        self.size = size
        self.transform = transform
        self.data = []

        all_folders = os.listdir(self.input_root)
        print('testing')
        for folder in all_folders:
            if not os.path.isdir(os.path.join(self.input_root, folder)):
                continue
            if folder not in SMID_TEST_IDS:
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
