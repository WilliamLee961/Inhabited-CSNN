import os
import sys
import glob
import numpy as np
import pandas as pd
from PIL import Image

import pickle
import gzip

import torch
import torch.nn as nn
import scipy.io as sio
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torchvision.models as models
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("use device :{}".format(device))

MAX_EPOCH = 15 #25
BATCH_SIZE = 16
LR = 0.0002 # 0.001
log_interval = 10
val_interval = 1
classes = 2
start_epoch = -1
lr_decay_step = 7

img_n = open('./img_names_resnet18','rb')  # 以二进制读模式（rb）打开pkl文件
img_name = pickle.load(img_n)  # 读取存储的pickle文件
print(img_name.shape)
# 将 img_name 转换为字符串列表
img_name = [str(name) for name in img_name]
print(img_name[0])
# 定义图片存储的目录
image_dir = './datasets'

# 定义一个函数，根据图片名称获取图片数据
def get_image_data(img_name):
    img_path = os.path.join(image_dir, img_name)
    if os.path.exists(img_path):
        with Image.open(img_path) as img:
            img_data = img.copy() # 读取图片数据
            return img_data
    else:
        return None

# example 获取第一张图片的数据

first_img_name = img_name[0]
first_img_data = get_image_data(first_img_name) #PIL.Image.Image 512*512*3
if first_img_data:
    # print("Shape of First image data:")
    # PIL.Image对象没有shape属性，需要把该对象转换为Numpy数组
    first_img_array = np.array(first_img_data)
    # print(first_img_array) #512*512*3
else:
    print("First image not found.")
# img_data = pickle.load(img)
# datas = ... 从原始数据集中下载from服务器）
# print("img_name.shape: ", np.array(img_name).shape)   
# print(img_data)
last_part = [name.split('-')[-1] for name in img_name] #pos.png, , smi ...
labels_value = [name.split('.')[0] for name in last_part] 

labels = []
for label_value in labels_value:
    if label_value == 'pos':
        labels.append(1)
    elif label_value == 'neg':
        labels.append(0)
    else:
        threshold = 60 # 为渐变图片设定阈值，阈值就是渐变图片从看得清到看不清的一个程度 大于阈值看不清定为neg
        if int(label_value) > threshold:
            labels.append(0)
        else:
            labels.append(1)
# print("label 0-200:", labels[:200])
# img_data = get_image_data(img_name)
df = pd.DataFrame({
    'name': img_name,
    'label': labels,
    # 'data': img_data
})

# 保存到CSV文件
# df.to_csv('./csvdata/img_label.csv', index=False)

# 我想要读入Gradient Image 文件夹中的图片，得到其中图片个数
folder_path = './datasets'
file_list = os.listdir(folder_path)
if not os.path.exists(folder_path):
    print(f"文件夹 '{folder_path}' 不存在！")

image_extensions = 'png'
image_files = [file for file in file_list if file.endswith(image_extensions)]

num_images = len(image_files)
# print("images数量: ", num_images)

f = pd.read_csv('./csvdata/img_label.csv')
df_f = pd.DataFrame(f)
first_df_f = df_f[10:11]
# dataframe 中包括 10  dfi-1-005-65.png Name: name, dtype: object 
print("name:", first_df_f['name'].iloc[0])
print("label:", first_df_f['label'].iloc[0])
print(df_f.shape) # (25920,2)
def load_data (path):
    x = []
    y = []
    z = []
    listing = os.listdir(path)
    for i in range(len(df_f)):
        document = df_f[i:i+1] # df_f[i:i+1]：这种写法会返回 DataFrame df_f 中索引从 i 到 i+1（不包括 i+1）的行，返回的数据类型是 DataFrame。
        df_name = document['name'].iloc[0] # 提取pandas.DataFrame中的对象需要用到iloc方法。
        df_label = document['label'].iloc[0]
        
        # 只有在图片存在时才更新label和name
        for image in listing:
            if str(df_name) == str(image):
                img = Image.open(path + '/' + image)
                img = img.resize((512,512))
                img = img.convert('RGB')
                img = np.array(img)
                x.append(img) #将值填入x
                y.append(int(df_label))
                z.append(str(df_name))
                # print('loading : '+ df_name)#打印信息
                break
        # else:
        #     print(f'File not found: {df_name}')
        
    x_ = np.array(x)
    y_ = np.array(y)
    z_ = np.array(z)
    return x_, y_, z_

data,label,name = load_data("./datasets")

# 训练集的数据增强
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # 水平翻转
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), #随机颜色调整
    # transforms.ToTensor(), # 
    # transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化
])

# 对X_train, X_test 数据集去重
unique_name, unique_indices = np.unique(name, axis=0, return_index = True)
print(unique_name.shape)
labels = label[unique_indices]
unique_data = data[unique_indices]
X_train, X_test, y_train, y_test = train_test_split(unique_data, labels, test_size=0.2, random_state=4)



# 训练集数据维度的调整：N H W C
X_train = np.reshape(X_train,(X_train.shape[0],512,512,3))
# 测试集数据维度的调整：N H W C
X_test = np.reshape(X_test,(X_test.shape[0],512,512,3))
save_train_dir = './data/img_train'
# 定义每个原始图像生成的增强图像数量
num_argumented_per_image = 5 # 每张原始图像生成5张增强图像
# 保存增强后的训练集图像
X_train_augmented = []
y_train_augmented = []
for i, (img, label1) in enumerate(zip(X_train, y_train)): #不能用label表示，不然会污染之前的label数组
    # 创建类别子目录（如果不存在）
    class_dir = os.path.join(save_train_dir, str(label1))
    os.makedirs(class_dir, exist_ok=True)
    # 先把原有训练集加入
    X_train_augmented.append(img)
    y_train_augmented.append(label1)
    img_pil = Image.fromarray((img).astype(np.uint8))  # 假设 img 的值范围是 [0, 255]
    for j in range(num_argumented_per_image):
        img_augmented = train_transform(img_pil) # 对图像进行增强
        X_train_augmented.append(np.array(img_augmented)) # 将增强后的图像转换会Numpy数组添加到 X_train
        y_train_augmented.append(label1)
        # img_augmented.save(os.path.join(class_dir, f'train_{i}_aug_{j}.png'))
        # print(f'已保存图像: {os.path.join(class_dir, f"train_{i}_aug_{j}.png")}')


# 将图片数据转换为Numpy 数组并且归一化(暂时先不归一化，内存不够)
X_train_augmented = np.array(X_train_augmented)
X_test = np.array(X_test) 
y_train_augmented = np.array(y_train_augmented)
y_test = np.array(y_test)

# 打印增强后的训练集数据形状
# print('X_train_augmented shape:', X_train_augmented.shape)
# print('y_train_augmented_shape:', y_train_augmented.shape)

# # X_train_augmented,X_test Min_Max归一化 这种方式问题在于转array需要太大的内存和时间，不如直接/255
# x_min = np.min(X_train_augmented, axis=(0,1,2), keepdims=True)
# x_max = np.max(X_train_augmented, axis=(0,1,2), keepdims=True) 
# print(x_min.shape) # 应该是每个通道最小值，共三个通道
# X_train_augmented = (X_train_augmented - x_min) / (x_max - x_min) # 广播机制 每个元素逗进行这样的计算

# # 测试集归一化
# x_min_test = np.min(X_test, axis=(0,1,2), keepdims=True)
# x_max_test = np.max(X_test, axis=(0,1,2), keepdims=True)
# X_test = (X_test - x_min_test)/(x_max_test - x_min_test)

# 将数据转换为PyTorch张量
X_train_augmented = torch.as_tensor(X_train_augmented)
X_test = torch.as_tensor(X_test)
y_train_augmented = torch.as_tensor(y_train_augmented)
y_test = torch.as_tensor(y_test)
print(X_train_augmented.shape)

# ============================ step 2/5 模型 ============================
class resnet18_ft(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        net = models.resnet18(pretrained=True)
        net.fc= nn.Sequential()    # 将分类层（fc）置空
        self.features = net
        self.fc= nn.Sequential(    # 定义一个卷积网络结构
            nn.Linear(512*1*1, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )
        
    def forward(self, x):
        x1 = self.features(x)
        x2 = x1.view(-1, 512)
        x2 = self.fc(x2)
        print(x1.shape) #[1,512]
        # print("x2 shape:", x2.shape) [1,2]
        return x2, x1      # output,feature
    
criterion = nn.CrossEntropyLoss()    # 选择损失函数
# 法2 : conv 小学习率
# flag = 0
net = resnet18_ft(2) # 这里的net是我们微调后的
net = net.to(device)
optimizer = optim.Adam(net.parameters(), lr=LR)               # 选择优化器
# optimizer.state = dict() # 清空之前优化器中的状态
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)     # 设置学习率下降策略 每隔7个epoch学习率降为原来的十分之一

# ---------------------训练-----------------------
train_curve = list()
valid_curve = list()

for epoch in range(start_epoch + 1, MAX_EPOCH):
    loss_mean = 0.
    correct = 0.
    total = 0.

    net.train() # 将模型调为训练模式
    for i, data in enumerate(X_train_augmented): #9804个
        # forward
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        inputs = data
        inputs = inputs.transpose(0, 2)
        inputs = torch.unsqueeze(inputs, dim=0)
        labels = y_train_augmented[i]
        labels = torch.unsqueeze(labels, dim=0)
        inputs = inputs.float()
        labels = labels.long()
        print(labels.shape) # 应该是(batch_size,)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, features = net(inputs) # 这样的forward返回的是feature吧
        # 这样的方式每次只处理一个样本，效率较低，其实可以一个batch一个batch读

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        # update weights
        optimizer.step()

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # correct += (predicted == labels).squeeze().cpu().sum().numpy()
        # 这里有更好的写法：
        correct += (predicted == labels).sum().item()

        # 打印训练信息
        loss_mean += loss.item()
        train_curve.append(loss.item())
        if (i+1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, i+1, len(X_train_augmented), loss_mean, correct / total))
            loss_mean = 0.

            # if flag_m1:
            # print("epoch:{} conv1.weights[0, 0, ...] :\n {}".format(epoch, resnet18_ft.conv1.weight[0, 0, ...]))

    scheduler.step()  # 更新学习率

    # validate the model
    if (epoch+1) % val_interval == 0:
        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(X_test):
                inputs = data
                inputs = inputs.transpose(0,2)
                inputs = torch.unsqueeze(inputs, dim=0)
                labels = y_test[j]
                labels = torch.unsqueeze(labels, dim=0)
                inputs = inputs.float()
                labels = labels.long()
                inputs, labels = inputs.to(device), labels.to(device)
                
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

                outputs, features = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                loss_val += loss.item()

            loss_val_mean = loss_val/len(X_test)
            valid_curve.append(loss_val_mean)
            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, j+1, len(X_test), loss_val_mean, correct_val / total_val))
        net.train()

# 最后使用训练后的微调预训练resnet提取图片特征（fc之前的向量）

    
