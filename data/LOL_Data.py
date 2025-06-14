from torch.utils.data import Dataset
from PIL import Image
from skimage import io, color, exposure
import os
import random
import numpy as np
def normalization(img):
    img = np.array(img, dtype="float32")
    img = (img * 1.0) / 255
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
    return img_norm

def loader(path, size, type):
    img = io.imread(path[0])
    label = io.imread(path[1])
    if type =='train':
        rh = np.random.randint(0, img.shape[0] - size)
        rw = np.random.randint(0, img.shape[1] - size)
        img = img[rh: rh + size, rw: rw + size]
        label = label[rh: rh + size, rw: rw + size]
        # 数据增强
        rand_mode = random.randint(0, 7)
        img = data_augmentation(img, rand_mode)
        label = data_augmentation(label, rand_mode)
    # 归一化
    img = normalization(img)
    label = np.array(label, dtype="float32")
    label = label / 255.0
    img = img.transpose(2, 0, 1).astype(np.float32)
    label = label.transpose(2, 0, 1).astype(np.float32)

    return img, label

def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

class TrainData(Dataset):
    def __init__(self, size=384, loader=loader,  transform=None):
        super(Dataset, self).__init__()
        self.path = '/data/low_light_dataset/LOL/'
        self.img = os.listdir(self.path + 'our485/low')
        self.label = os.listdir(self.path + 'our485/high')
        self.transform = transform
        self.size = size
        self.loader = loader
        self.data = []
        for i in range(len(self.img)):
            self.data.append([self.path + 'our485/low/' + self.img[i], self.path + 'our485/high/' + self.img[i]])

        # self.path = '/data/low_light_dataset/LOL/'
        # self.img = os.listdir(self.path + 'eval15/low')
        # self.label = os.listdir(self.path + 'eval15/high')
        # for i in range(len(self.img)):
        #     self.data.append([self.path + 'eval15/low/' + self.img[i], self.path + 'eval15/high/' + self.img[i]])
        print(len(self.data))

    def __getitem__(self, item):
        path = self.data[item]
        img, label = self.loader(path, self.size, 'train')
        return img, label

    def __len__(self):
        return len(self.img)

class TestData(Dataset):
    def __init__(self, size=384, loader=loader,  transform=None):
        super(Dataset, self).__init__()
        self.path = '/data/low_light_dataset/LOL/'
        self.img = os.listdir(self.path + 'eval15/low')
        self.label = os.listdir(self.path + 'eval15/high')
        self.transform = transform
        self.size = size
        self.loader = loader
        self.data = []
        for i in range(len(self.img)):
            self.data.append([self.path + 'eval15/low/' + self.img[i], self.path + 'eval15/high/' + self.img[i]])

    def __getitem__(self, item):
        path = self.data[item]
        img, label = self.loader(path, self.size, 'test')
        return img, label

    def __len__(self):
        return len(self.img)

