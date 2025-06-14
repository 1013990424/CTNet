from PIL import Image
import numpy as np
import matplotlib.pylab as plt
from skimage import io, color, exposure
import os
"""忽略warning"""
import warnings


warnings.filterwarnings("ignore")

def compare(name):
    #low = io.imread('C:/Users/loner/Desktop/低光照增强数据集/LOL/eval15/low/'+str(name)+'.png')
    #high = io.imread('C:/Users/loner/Desktop/低光照增强数据集/LOL/eval15/high/' + str(name) + '.png')
    name = name.split('low')[1]
    #low = io.imread('C:/Users/loner/Desktop/低光照增强数据集/LOL-v2/Real_captured/Train/Low/low' + name)
    #high = io.imread('C:/Users/loner/Desktop/低光照增强数据集/LOL-v2/Real_captured/Train/Normal/normal' + name)

    low = io.imread('C:/Users/loner/Desktop/低光照增强数据集/LOL-v2/Synthetic/Train/Low/' + name)
    high = io.imread('C:/Users/loner/Desktop/低光照增强数据集/LOL-v2/Synthetic/Train/Normal/' + name)

    low_hsv = color.rgb2hsv(low)
    high_hsv = color.rgb2hsv(high)

    high_v = high_hsv[:, :, 2]
    low_hsv[:, :, 2] = high_v

    img = color.hsv2rgb(low_hsv)

    io.imsave('./motivation/' + name + '.jpg', img)

def combine(name):
    print(name)
    path0 = './motivation/LOL-v2'
    path1 = 'C:/Users/loner/Desktop/低光照增强数据集/LOL-v2/Real_captured/Train/Normal/normal'
    name = name.split('.')[0]
    name = name.split('normal')[1]
    img0 = Image.open(path0 + '/' + name + '.png.jpg')
    img1 = Image.open(path1 + name + '.png')
    target = Image.new('RGB', (600 * 2 + 5 * 1, 400), color=(255, 255, 255))
    target.paste(img0, (0, 0, img1.size[0], img1.size[1]))
    target.paste(img1, (img1.size[0] * 1 + 5 * 1, 0, img1.size[0] * 2 + 5 * 1, img1.size[1]))
    target.save('./motivation/' + name + '.png')

def HScombine(name):

    #low = io.imread('C:/Users/loner/Desktop/低光照增强数据集/LOL-v2/Real_captured/Train/Low/low' + name)
    #high = io.imread('C:/Users/loner/Desktop/低光照增强数据集/LOL-v2/Real_captured/Train/Normal/normal' + name)

    low = io.imread('C:/Users/loner/Desktop/低光照增强数据集/LOL-v2/Synthetic/Train/Low/' + name)
    high = io.imread('C:/Users/loner/Desktop/低光照增强数据集/LOL-v2/Synthetic/Train/Normal/' + name)

    low_hsv = color.rgb2hsv(low)
    high_hsv = color.rgb2hsv(high)

    low_H = low_hsv[:, :, 0]
    low_S = low_hsv[:, :, 1]
    high_H = high_hsv[:, :, 0]
    high_S = high_hsv[:, :, 1]

    w, h, _ = low.shape

    low = Image.fromarray(low)
    low_H = Image.fromarray(np.uint8(low_H * 255))
    low_S = Image.fromarray(np.uint8(low_S * 255))
    high_H = Image.fromarray(np.uint8(high_H * 255))
    high_S = Image.fromarray(np.uint8(high_S * 255))
    high = Image.fromarray(high)

    target = Image.new('RGB', (h * 6 + 5 * 5, w), color=(255, 255, 255))
    target.paste(low, (0, 0, h, w))
    target.paste(low_H, (h * 1 + 5 * 1, 0, h * 2 + 5 * 1, w))
    target.paste(low_S, (h * 2 + 5 * 2, 0, h * 3 + 5 * 2, w))
    target.paste(high_H, (h * 3 + 5 * 3, 0, h * 4 + 5 * 3, w))
    target.paste(high_S, (h * 4 + 5 * 4, 0, h * 5 + 5 * 4, w))
    target.paste(high, (h * 5 + 5 * 5, 0, h * 6 + 5 * 5, w))
    target.save('./motivation/LOL-v2-syn/' + name + '.png')

def Crop():
    imgs = os.listdir('./motivation/decom_net_train_result')
    for img_name in imgs:
        img = Image.open('./motivation/decom_net_train_result/' + img_name)
        ref = img.crop((0, 0, 600, 400))
        ill = img.crop((600, 0, 1200, 400))
        name = img_name.split('.')[0]
        ref.save('./motivation/decom_result/ref_' + name + '.png')
        ill.save('./motivation/decom_result/ill_' + name + '.png')

def Deccombine(name):

    #low = io.imread('C:/Users/loner/Desktop/低光照增强数据集/LOL-v2/Real_captured/Train/Low/low' + name)
    #high = io.imread('C:/Users/loner/Desktop/低光照增强数据集/LOL-v2/Real_captured/Train/Normal/normal' + name)

    low_ref = Image.open('./motivation/decom_result/ref_low_' + name)
    low_ill = Image.open('./motivation/decom_result/ill_low_' + name)
    high_ref = Image.open('./motivation/decom_result/ref_high_' + name)
    high_ill = Image.open('./motivation/decom_result/ill_high_' + name)

    w, h = low_ref.size
    target = Image.new('RGB', (w * 4 + 5 * 3, h), color=(255, 255, 255))
    target.paste(low_ref, (0, 0, w, h))
    target.paste(low_ill, (w * 1 + 5 * 1, 0, w * 2 + 5 * 1, h))
    target.paste(high_ref, (w * 2 + 5 * 2, 0, w * 3 + 5 * 2, h))
    target.paste(high_ill, (w * 3 + 5 * 3, 0, w * 4 + 5 * 3, h))
    target.save('./motivation/' + name + '.png')

def KinDccombine(name, i):

    low = io.imread('C:/Users/loner/Desktop/低光照增强数据集/LOL/eval15/low/' + name)
    kind_name = name.split('.')[0]
    kind_name += '_kindle.png'
    kind = io.imread('C:/Users/loner/Desktop/LOL/KinD/' + kind_name)
    high = io.imread('C:/Users/loner/Desktop/低光照增强数据集/LOL/eval15/high/' + name)

    low = Image.fromarray(low)
    kind = Image.fromarray(kind)
    high = Image.fromarray(high)

    num = str(i)
    low_ref = Image.open('./motivation/decom_result/ref_low_' + num + '_2000.png')
    #low_ill = Image.open('./motivation/decom_result/ill_low_' + num + '_2000.png')
    high_ref = Image.open('./motivation/decom_result/ref_high_' + num + '_2000.png')
    #high_ill = Image.open('./motivation/decom_result/ill_high_' + num + '_2000.png')

    w, h = low.size
    target = Image.new('RGB', (w * 5 + 5 * 4, h), color=(255, 255, 255))
    target.paste(low, (0, 0, w, h))
    target.paste(kind, (w * 1 + 5 * 1, 0, w * 2 + 5 * 1, h))
    target.paste(high, (w * 2 + 5 * 2, 0, w * 3 + 5 * 2, h))
    target.paste(low_ref, (w * 3 + 5 * 3, 0, w * 4 + 5 * 3, h))
    target.paste(high_ref, (w * 4 + 5 * 4, 0, w * 5 + 5 * 4, h))
    target.save('./motivation/KinD/' + name + '.png')

def Visccombine(name):

    low = io.imread('C:/Users/loner/Desktop/低光照增强数据集/LOL-v2/Real_captured/Test/Low/' + name)
    kind_name = name.split('.')[0]
    kind_name += '_kindle.png'
    kind_name = kind_name.split('low')[1]
    kind = io.imread('C:/Users/loner/Desktop/LOL_v2/真实数据集/KinD/' + kind_name)
    ctnet = io.imread('C:/Users/loner/Desktop/LOL_v2/真实数据集/CTNet/' + name)
    high_name = name.split('low')[1]
    high = io.imread('C:/Users/loner/Desktop/低光照增强数据集/LOL-v2/Real_captured/Test/Normal/normal' + high_name)

    low = Image.fromarray(low)
    kind = Image.fromarray(kind)
    ctnet = Image.fromarray(ctnet)
    high = Image.fromarray(high)

    w, h = low.size
    target = Image.new('RGB', (w * 4 + 5 * 3, h), color=(255, 255, 255))
    target.paste(low, (0, 0, w, h))
    target.paste(kind, (w * 1 + 5 * 1, 0, w * 2 + 5 * 1, h))
    target.paste(ctnet, (w * 2 + 5 * 2, 0, w * 3 + 5 * 2, h))
    target.paste(high, (w * 3 + 5 * 3, 0, w * 4 + 5 * 3, h))
    target.save('./motivation/visualization/' + name + '.png')


#Crop()
#names = ['1_2000.png', '2_2000.png', '3_2000.png']
#names = [1, 22, 23, 55, 79, 111, 146, 179, 493, 547, 665, 669, 748, 778, 780]
#names = os.listdir('C:/Users/loner/Desktop/低光照增强数据集/LOL-v2/Real_captured/Train/Normal')
names = os.listdir('C:/Users/loner/Desktop/低光照增强数据集/LOL-v2/Real_captured/Test/Low')
#names = os.listdir('C:/Users/loner/Desktop/低光照增强数据集/LOL/eval15/low')
#names = ['00450', '00179', '00634', '00001', '00006', '00021', '00052', '00060', '00132', '00157', '00233', '00342', '00460', '00464', '00478', '00539', '00561', '00564', '00566', '00568', '00658','00643','00637']
#names = os.listdir('C:/Users/loner/Desktop/低光照增强数据集/LOL-v2/Synthetic/Train/Normal')
# for i in range(len(names)):
#     #compare(names[i])
#     #combine(names[i])
#     #HScombine(names[i])
#     #Deccombine(names[i])
#     #KinDccombine(names[i], i+1)
#     Visccombine(names[i])