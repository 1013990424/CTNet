#测试
from transformer import Net, rgb2hsv
from PIL import Image
from skimage import io, color
import torch
import numpy as np
import os
#模型初始化
model = Net()
model_path = './output/syn/best.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=True)
model = model.cuda()
model.eval()

def normalization(img):
    img = np.array(img, dtype="float32")
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
    return img_norm

def test_data(path):
    for img_name in os.listdir(path):
        low_name = img_name
        img = io.imread(path + '/' + low_name)
        crop_x = img.shape[0] % 8
        crop_y = img.shape[1] % 8
        img = img[crop_x:, crop_y:, :]
        img = normalization(img)
        img = torch.from_numpy(np.moveaxis(img, -1, 0)).float()  # [w h 3] -> [3 w h]
        img = img.unsqueeze(0)
        # 得到结果
        img = img.cuda()
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
        temp.save('./syn_result/' + img_name)

print('加载完成')
path = "../../datas/LOL-v2/Synthetic/Test/Low"
test_data(path)
