from real_Data import TrainData
from model3 import MCFNet, rgb2hsv
from loss import normal_loss
from torch.optim import Adam
from torch.utils.data import DataLoader
import pytorch_ssim
import torch.nn as nn
import torch
import warnings
import random
warnings.filterwarnings("ignore")
print(torch.__version__)
print(torch.cuda.is_available())
random.seed(0)
#参数
learning_rate = 1e-4
epochs = 300
batchsize = 4

data = TrainData(256)
traindata_loader = DataLoader(data, batch_size=batchsize, shuffle=True, num_workers=1)
model = MCFNet()
mse_loss = nn.MSELoss()
normalLoss = normal_loss()
optimizer = Adam(model.parameters(), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[200], gamma=0.5, last_epoch=-1)
if torch.cuda.is_available():
    model = model.cuda()
    normalLoss = normalLoss.cuda()
    mse_loss = mse_loss.cuda()

min_loss = 1e5
for epoch in range(1, epochs+1):
    epoch_loss = 0
    test_loss = 0
    for idx, (img, label_x4) in enumerate(traindata_loader):
        if torch.cuda.is_available():
            img = img.cuda()
            label_x4 = label_x4.cuda()
        # 获得lab和hsv
        _, _, v = rgb2hsv(label_x4)
        # 计算结果
        model.zero_grad()
        pre_v, pre_rgb = model(img)
        # 损失函数
        test_loss += mse_loss(pre_rgb, label_x4).data.item()
        loss = normalLoss(pre_rgb, label_x4) + mse_loss(pre_v, v)
        epoch_loss += loss.data.item()
        loss.backward()
        optimizer.step()

    #scheduler.step(epoch)
    # 保存最好的
    min_loss = test_loss
    torch.save(model.state_dict(), './without_CIB/' + str(epoch) + '.pth')

    print('epoch_' + str(epoch) + "_loss :" + str(epoch_loss) + " val_loss:" + str(test_loss))


