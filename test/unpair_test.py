#测试
from transformer import Net, rgb2hsv
from PIL import Image
from skimage import io, color
import torch
import numpy as np
import os

#模型初始化
device = torch.device('cuda:2')
model = Net(num_blocks=[2, 2, 4], dim=48)
model_path = './output/unpair.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=True)
model = model.to(device)
model.eval()
print(model_path)
def normalization(img):
    img = np.array(img, dtype="float32")
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
    return img_norm

def unpair_inference(path, save_path):
    for img_name in os.listdir(path):
        low_name = img_name.split('.')[0]
        img = io.imread(path + '/' + img_name)
        crop_x = img.shape[0] % 8
        crop_y = img.shape[1] % 8
        if img.shape[0] > 700:
            crop_x += 128
        if img.shape[1] > 700:
            crop_y += 128
        if img.shape[0] > 1500:
            crop_x += 128
        if img.shape[1] > 1500:
            crop_y += 128
        if img.shape[0] > 2000:
            crop_x += 256
        if img.shape[1] > 2000:
            crop_y += 256
        img = img[crop_x:, crop_y:, :]
        img = normalization(img)
        img = torch.from_numpy(np.moveaxis(img, -1, 0)).float()  # [w h 3] -> [3 w h]
        img = img.unsqueeze(0)
        # 得到结果
        img = img.to(device)
        with torch.no_grad():
            _,  img = model(img)
            img = img.data
            img = img.squeeze().float()  # 删除维度为1的维度
            img = img.cpu().clamp_(0, 1).numpy()  # 把元素的值截取到0-1之间
            img = np.moveaxis(img, 0, -1)  # [3  w*4  h* 4] ->[w*4  h*4  3]
            img = img * 255
            img = np.array(img, np.uint8)

        # 保存
        temp = Image.fromarray(img)
        temp.save(save_path + low_name + '.png')

print('加载完成')
for name in ['DICM', 'LIME', 'MEF', 'NPE', 'VV']:
    print(name)
    path = '/data/low_light_dataset/unpair/' + name
    save_path = '/home/pc/Desktop/unpair/' + name + '/CTNet/'
    unpair_inference(path, save_path)
