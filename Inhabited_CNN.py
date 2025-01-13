#%%
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import os
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.cuda.set_device(0)


class Inhabited_AlexNet(nn.Module):
    def __init__(self,add_model=""):
        super(Inhabited_AlexNet, self).__init__()
        self.norm=add_model
        self.features = nn.Sequential(
            # x的输入: [b,3,227,227]  3是彩色图片三通道
            nn.Conv2d(3, 96, kernel_size=(11,11), stride=4, padding=1),
            #[b,96,55,55]
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(96,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True),
            LateralInhibition(96),
            # [b,96,55,55]
            nn.MaxPool2d(kernel_size=(3,3), stride=2),
            # [b,96,27,27]
            nn.Conv2d(96, 256, kernel_size=(5,5), stride=1, padding=2),
            # [b,256,27,27]
            nn.ReLU(inplace=True),
            LateralInhibition(256),
            #nn.BatchNorm2d(256,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True),
            # [b,256,27,27]
            nn.MaxPool2d(kernel_size=(3,3), stride=2),
            #[b,256,13,13]
            nn.Conv2d(256, 384, kernel_size=(3,3), stride=1, padding=1),
            # [b,384,13,13]
            nn.ReLU(inplace=True),
            LateralInhibition(384),#nn.BatchNorm2d(384,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True),
            # [b,384,13,13]
            nn.Conv2d(384, 384, kernel_size=(3,3), stride=1, padding=1),
            # [b,384,13,13]
            nn.ReLU(inplace=True),
            LateralInhibition(384),#nn.BatchNorm2d(384,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True),
            # [b,384,13,13]
            nn.Conv2d(384, 256, kernel_size=(3,3), padding=1),
            # [b,256,13,13]
            nn.ReLU(inplace=True),
            LateralInhibition(256),#nn.BatchNorm2d(256,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True),
            # [b,256,13,13]
            nn.MaxPool2d(kernel_size=(3,3), stride=2),
            # [b,256,6,6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_activations(self,x):
        out = self.features(x)
        return out
    def get_last_activations(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        for i in range(3):
            x = self.classifier[i](x)
        return x

    def draw(self,x):
        before=[]
        after=[]
        raw_image=x
        for index,layer in enumerate(self.features):
            x=layer(x)
            if "ReLU" in str(layer):
                before.append(x)
            if "LateralInhibition" in str(layer):
                after.append(x)
        return before,after,raw_image


    def logits_activations(self,x,layer_names="ReLU",suppress={}):
        if layer_names !="ReLU":
            print("Sorry! fuck Poor function!")
        layer_values={}

        for index,layer in enumerate(self.features):
            x=layer(x)
            if layer_names in str(layer):
                if suppress!={}:
                    sup_mask_sized_as_x = suppress[index].expand(-1, x.size()[1], -1, -1)
                    x = x.where(sup_mask_sized_as_x == 0, torch.zeros_like(x))
                
                layer_values[index]=x
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x,layer_values

def to_tensor(x, **kwargs):
    CUDA = torch.cuda.is_available()
    device = 'cuda' if CUDA else 'cpu'
    return torch.tensor(x, device=device, **kwargs)
def to_np(x):
    return x.detach().cpu().numpy()

# 抑制结构（代码里唯一有用的地方）
class LateralInhibition(torch.nn.Module):
    def __init__(self, c , l=7, a=0.1, b=0.9):
        super().__init__()
        self.len = l
        self.channel=c
        assert self.len % 2 == 1
        self.a = a
        self.b = b
        self.register_buffer(
            'inhibition_kernel',
            to_tensor(self.mex_hat(l), dtype=torch.float32).view(1, 1, 1, -1))

    def forward(self, x):  # as argument we get max-c map with dimensions 'batch x c x n x n'
        assert x.size(2) == x.size(3)
        len_ = self.len
        pad = len_ // 2
        batches = x.size(0)
        n = x.size(2)

        # unfold x to LIZs for each pixel:
        x_unf = F.unfold(x, (len_, len_), padding=(pad, pad)) # (batch,C*len_*len_,n*n)
        # next line is needed for extend tensor size (from 'batch x kernel x n*n' to 'batch x 1 x kernel x n*n'):
        x_unf = x_unf.view(batches, self.channel,len_*len_, n*n) # (batch,C,len_*len_,n*n)
        x_unf = x_unf.transpose(2,3)
        # select all middle points in LIZs:
        mid_vals = x.view(x.size(0), self.channel, n*n, 1) # (batch,C,n*n,1)
        
        average_term = torch.exp(-x_unf.mean(-1, keepdim=True)).view(batches, self.channel, n, n)
        
        differential_term = (self.inhibition_kernel * F.relu(x_unf - mid_vals)
                            ).sum(3, keepdim=True).view(batches, self.channel, n, n)
        
        suppression_mask = self.a * average_term + self.b * differential_term
        assert x.shape == suppression_mask.shape
        
        # normalization
        E=suppression_mask.mean()
        sigma=suppression_mask.std()
        suppression_mask=(suppression_mask-E)/(sigma**2+0.0001)**0.5
        #suppression_mask_norm = (suppression_mask ** 2).sum() ** 0.5
        #suppression_mask = suppression_mask / suppression_mask_norm
        # because all values are non-negative we can do this:
        #filter_ = x > suppression_mask
        #suppression_mask = x.where(filter_, torch.zeros_like(x))
        return suppression_mask
       
        # PROBLEM WITH GATING: non-normalized partial computations
    
    # 生物学概念 Mexican Hat
    def mex_hat(self,d):
        grid = (np.mgrid[:d, :d] - d//2) * 1.0
        eucl_grid = (grid**2).sum(0) ** 0.5  # euclidean distances
        eucl_grid /= d  # normalize by LIZ length
        return eucl_grid * np.exp(-eucl_grid)  # mex_hat function values
if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x=torch.randn([10,3,227,227])
    x=x.to(device)
    model=Inhabited_AlexNet()
    model=model.to(device)
    # y=model.get_activations(x)
    y=model.get_last_activations(x)
    print(y.shape)
    from models import  classifier
    new_model=classifier()
    new_model=new_model.to(device)
    features = y.view(y.size(0), -1)
    y=new_model(features)
    print(y)
