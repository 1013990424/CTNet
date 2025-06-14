from transformer import Net, rgb2hsv
import numpy as np
import cv2
from torchvision.models import vgg16
from loss import ssim_loss, LossNetwork
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_ssim
import torch.nn as nn
import torch
import warnings
import random
import math
import piq

warnings.filterwarnings("ignore")
print(torch.__version__)
print(torch.cuda.is_available())
random.seed(0)

torch.distributed.init_process_group("nccl")
rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
device_id = rank % torch.cuda.device_count()
device = torch.device(device_id)

def psnr(img1, img2):
    # print(img1.size())
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

#参数
data_name = 'SID'
use_gamma= False
if data_name == 'LOL':
    from data.LOL_Data import TrainData, TestData
    model = Net(num_blocks=[2, 2, 4], dim=48)
    train_dataset = TrainData(size=384)
    test_dataset = TestData()
    batchsize = 1
if data_name == 'real':
    from data.real_Data import TrainData, TestData
    model = Net(num_blocks=[2, 2, 4], dim=48)
    train_dataset = TrainData(size=224)
    test_dataset = TestData()
    batchsize = 1
if data_name == 'syn':
    from data.syn_Data import TrainData, TestData
    model = Net(num_blocks=[2, 2, 4], dim=48)
    train_dataset = TrainData(size=256 + 96)
    test_dataset = TestData()
    batchsize = 1
if data_name == 'unpair':
    from data.syn_Data import TrainData, TestData
    model = Net(num_blocks=[2, 2, 4], dim=48)
    train_dataset = TrainData(size=256 + 96)
    test_dataset = TestData()
    batchsize = 1
    use_gamma = True
if data_name == 'SDSD_indoor':
    from data.SDSD_indoor_Data import SDSDTrainDataset, SDSDTestDataset
    model = Net(num_blocks=[2, 2, 4], dim=48)
    train_dataset = SDSDTrainDataset('/data/low_light_dataset/SDSD/indoor/indoor_static_np', size=256)
    test_dataset = SDSDTestDataset('/data/low_light_dataset/SDSD/indoor/indoor_static_np')
    batchsize = 2
if data_name == 'SDSD_outdoor':
    from data.SDSD_outdoor_Data import SDSDTrainDataset, SDSDTestDataset
    model = Net(num_blocks=[2, 2, 4], dim=48)
    train_dataset = SDSDTrainDataset('/data/low_light_dataset/SDSD/outdoor/outdoor_static_np', size=256)
    test_dataset = SDSDTestDataset('/data/low_light_dataset/SDSD/outdoor/outdoor_static_np')
    batchsize = 2
if data_name == 'SMID':
    from data.SMID_Data import SMIDTrainDataset, SMIDTestDataset
    model = Net(num_blocks=[2, 2, 4], dim=48)
    train_dataset = SMIDTrainDataset('/data/low_light_dataset/SMID/smid', size=256)
    test_dataset = SMIDTestDataset('/data/low_light_dataset/SMID/smid')
    batchsize = 1
if data_name == 'SID':
    from data.SID_Data import SIDTrainDataset, SIDTestDataset
    model = Net(num_blocks=[2, 2, 4], dim=48)
    train_dataset = SIDTrainDataset('/data/low_light_dataset/SID', size=256)
    test_dataset = SIDTestDataset('/data/low_light_dataset/SID')
    batchsize = 2

print(data_name)
learning_rate = 2e-4
total_iters = 500000
num_iter_per_epoch = math.ceil(len(train_dataset)  / (batchsize * torch.cuda.device_count()))
total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
traindata_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False, num_workers=4, sampler=train_sampler)
testdata_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
for param in vgg_model.parameters():
    param.requires_grad = False
loss_network = LossNetwork(vgg_model)
loss_network.eval()

model = model.to(device)
model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
mse_loss = nn.MSELoss()
normalLoss = ssim_loss()
l1_loss = F.smooth_l1_loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-5)
#scheduler = lr_scheduler.CosineAnnealingRestartCyclicLR(optimizer, periods = [92000, 208000], restart_weights = [1,1], eta_mins = [0.000285,0.000001])
if torch.cuda.is_available():
    normalLoss = normalLoss.to(device)
    mse_loss = mse_loss.to(device)

is_master = rank == 0
global_psnr = 0
batch_num = 0
train_psnr = 0
for epoch in range(1, total_epochs+1):
    epoch_loss = 0
    test_loss = 0
    for idx, (img, label_x4) in enumerate(traindata_loader):
        if torch.cuda.is_available():
            img = img.to(device)
            label_x4 = label_x4.to(device)
        #获得lab和hsv
        _, _, v = rgb2hsv(label_x4)
        #计算结果
        model.zero_grad()
        # use random gamma function (enhancement curve) to improve generalization
        if use_gamma:
            gamma = random.randint(60, 120) / 100.0
            pre_v, pre_rgb = model(img ** gamma)
        else:
            pre_v, pre_rgb = model(img)
        #损失函数
        test_loss += mse_loss(pre_rgb, label_x4).data.item()
        if 'SID' not in data_name:
            loss = l1_loss(pre_rgb, label_x4) + 0.04 * loss_network(pre_rgb, label_x4) + mse_loss(pre_v, v)
        else:
            loss = normalLoss(pre_rgb, label_x4) + mse_loss(pre_v, v)
        epoch_loss += loss.data.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        batch_num += 1
        if is_master:
            if batch_num % 100 == 0:
                num = 0
                ssims = 0
                psnrs = 0
                model.eval()
                with torch.no_grad():
                    for idx, (img, label) in enumerate(testdata_loader):
                        x = img.to(device)  # (B, C, H, W)
                        y = label.to(device)

                        _, pre_rgb = model(x)  # (B, C, H, W)

                        # 归一化到 [0, 1]，如果还没归一化
                        pre_rgb = pre_rgb.clamp(0, 1)
                        y = y.clamp(0, 1)
                        # 计算ssim 和 psnr
                        pre_rgb = pre_rgb.squeeze().permute(1, 2, 0).cpu().numpy()
                        y = y.squeeze()
                        y = y.permute(1, 2, 0).cpu().numpy()
                        psnrs += psnr(pre_rgb * 255.0, y * 255.0)
                        pre_rgb = torch.Tensor(pre_rgb).unsqueeze(0).permute(0, 3, 1, 2)
                        y = torch.Tensor(y).unsqueeze(0).permute(0, 3, 1, 2)
                        ssims += pytorch_ssim.ssim(pre_rgb, y)
                        num += x.size(0)
                avg_psnr = psnrs / num
                avg_ssim = ssims / num

                # 保存最好的
                if avg_psnr > global_psnr and avg_psnr < 26:
                    global_psnr = avg_psnr
                    torch.save(model.module.state_dict(), './output/' + data_name + '.pth')
                    print('batch_{}_loss :{} psnr :{} ssim :{}'.format(str(batch_num), str(epoch_loss), str(avg_psnr),
                                                                       str(avg_ssim)))
                    print('------------------------saving the best model------------------------')
                    continue

            if batch_num % 1000 == 0:
                print('batch_{}_loss :{} psnr :{} ssim :{}'.format(str(batch_num), str(epoch_loss), str(avg_psnr),
                                                                   str(avg_ssim)))
            if batch_num % 5000 == 0:
                if data_name == 'unpair':
                    torch.save(model.module.state_dict(), './output/' + data_name + str(batch_num) + '.pth')
