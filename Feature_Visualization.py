import torch
import numpy as np
import os, argparse, cv2
from visual_model import MCFNet
from torch.utils.data import Dataset
from skimage import io
from torch.utils.data import DataLoader

def normalization(img):
    img = np.array(img, dtype="float32")
    img = (img * 1.0) / 255
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
    return img_norm

def loader(path):
    img = io.imread(path)
    name = path.split('/')[-1]
    # 归一化
    img = normalization(img)
    img = img.transpose(2, 0, 1).astype(np.float32)

    return img, name

class TestData(Dataset):
    def __init__(self, size=384, loader=loader, transform=None):
        super(Dataset, self).__init__()
        self.img = os.listdir("../LOL-v2/Real_captured/Train/Low")
        self.transform = transform
        self.size = size
        self.loader = loader
        self.data = []
        for i in range(len(self.img)):
            self.data.append('../LOL-v2/Real_captured/Train/Low/' + self.img[i])

    def __getitem__(self, item):
        path = self.data[item]
        img, name = self.loader(path)
        return img, name

    def __len__(self):
        return len(self.img)

def heatmap(feat_viz, ori_img, save_path=None):
    feat_viz = torch.mean(feat_viz, dim=1, keepdim=True).data.cpu().numpy().squeeze()
    feat_viz = (feat_viz - feat_viz.min()) / (feat_viz.max() - feat_viz.min() + 1e-8)

    ori_img = ori_img.data.cpu().numpy().squeeze()
    ori_img = ori_img.transpose((1, 2, 0))
    ori_img = ori_img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    ori_img = ori_img[:, :, ::-1]
    # img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    ori_img = np.uint8(255 * ori_img)
    feat_viz = np.uint8(255 * feat_viz)
    feat_viz = cv2.applyColorMap(feat_viz, cv2.COLORMAP_JET)
    #feat_viz = cv2.resize(feat_viz, (320, 320))
    #ori_img = cv2.resize(ori_img, (320, 320))
    # print(feat_viz.shape, ori_img.shape)
    feat_viz = cv2.addWeighted(ori_img, 0.5, feat_viz, 0.5, 0)

    cv2.imwrite(save_path, feat_viz)
    # cv2.imshow('img', feat_viz)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


model = MCFNet()
model_path = './pretrained/299.pth'
model.load_state_dict(torch.load(model_path), strict=True)
model.cuda()
model.eval()

#os.makedirs(save_path, exist_ok=True)
data = TestData()
test_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=1)
for idx, (image, img_name) in enumerate(test_loader):
    image = image.cuda()
    img_name = img_name[0]
    rgb_fea1, hsv_fea1, aggregate_fea1 = model(image)
    save_path = './Feature_Visualization/origin/'
    heatmap(feat_viz=rgb_fea1, ori_img=image, save_path=save_path + 'rgb_' + img_name)
    heatmap(feat_viz=hsv_fea1, ori_img=image, save_path=save_path + 'hsv_' + img_name)
    heatmap(feat_viz=aggregate_fea1, ori_img=image, save_path=save_path + 'aggregate_' + img_name)
    # for i in range(0, 3):
    #     for j in range(0, 4):
    #         for k in range(0, 2):
    #             cur_feat_viz = feat_viz[i][j][k]
    #             label = 'feat' if k == 0 else 'guid'
    #             img_name = name.split('.')[0] + '_level{}_GRA{}_'.format(i+3, j+1) + label + '.png'
    #             heatmap(feat_viz=cur_feat_viz, ori_img=image, save_path=save_path+img_name)
    #             print('> Dataset: {}, Image: {}'.format(_data_name, save_path+img_name))
    # res = res2
    # res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    # res = res.sigmoid().data.cpu().numpy().squeeze()
    # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    # misc.imsave(save_path+name, res)