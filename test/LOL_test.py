#测试
import time
from CTNet import Net
from PIL import Image
from skimage import io, color
import pytorch_ssim
import torch
import numpy as np
import os
import math

def psnr(img1, img2):
    # print(img1.size())
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

device = torch.device('cuda:2')
#模型初始化
model = Net()
model_path = './output/LOL.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:2')), strict=True)
model = model.to(device)
model.eval()

#归一化
def normalization(img):
    img = np.array(img, dtype="float32")
    img = (img * 1.0) / 255
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
    return img_norm

def test_data(low_path, high_path):
    ssims = 0
    psnrs = 0
    test_size = len(os.listdir(low_path))
    for img_name in os.listdir(low_path):
        img = io.imread(low_path + '/' + img_name)
        gt = io.imread(high_path + '/' + img_name)
        crop_x = img.shape[0] % 8
        crop_y = img.shape[1] % 8
        img = img[crop_x:, crop_y:, :]
        img = normalization(img)
        gt = np.array(gt, dtype="float32")
        gt = gt / 255.0
        img = img.transpose(2, 0, 1).astype(np.float32)
        gt = gt.transpose(2, 0, 1).astype(np.float32)
        gt = torch.from_numpy(gt).float()
        img = torch.from_numpy(img).float()  # [w h 3] -> [3 w h]
        img = img.unsqueeze(0)
        gt = gt.unsqueeze(0)
        # 得到结果
        img = img.to(device)
        gt = gt.to(device)
        with torch.no_grad():
            _,  img = model(img)
            img = img.data
            ssims += pytorch_ssim.ssim(img.clamp_(0, 1), gt).cpu().numpy()
            psnrs += psnr(img.clamp_(0, 1), gt)
            img = img.squeeze().float()  # 删除维度为1的维度
            img = img.cpu().clamp_(0, 1).numpy()  # 把元素的值截取到0-1之间
            img = np.moveaxis(img, 0, -1)  # [3  w*4  h* 4] ->[w*4  h*4  3]
            img = img * 255
            img = np.array(img, np.uint8)

        # 保存
        temp = Image.fromarray(img)
        temp.save('./LOL_result/' + img_name)
    print(test_size)
    print(psnrs / test_size, ssims / test_size)

print('加载完成')
#start = time.clock()
path = ('/data/low_light_dataset/LOL/')
low_path = path + "eval15/low"
high_path = path + "eval15/high"
test_data(low_path, high_path)
#elapsed = (time.clock() - start)
#print("Time used:", elapsed)
