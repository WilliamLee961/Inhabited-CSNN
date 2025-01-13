import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




from spikingjelly.activation_based import encoding, neuron, layer, functional, surrogate







NUM_CLASSES = 8
def to_tensor(x, **kwargs):
    # CUDA = torch.cuda.is_available()
    # device = 'cuda' if CUDA else 'cpu'
    device='cpu'
    return torch.tensor(x, device=device, **kwargs)
def to_np(x):
    return x.detach().cpu().numpy()


# 改进版抑制结构,修改其代码使其能够处理五维数据，后三维度
#fre*bat*channel*n*n
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

    def forward(self, x):  # as argument we get max-c map with dimensions 'fre*batch x c x n x n'
        assert x.size(3) == x.size(4)
        len_ = self.len
        pad = len_ // 2
        fre=x.size(0)
        batches = x.size(1)
        n = x.size(3)

        # unfold x to LIZs for each pixel:
        x_=x.view(fre*batches,self.channel,n,n)
        x_unf = F.unfold(x_, (len_, len_), padding=(pad, pad)) # (batch,C*len_*len_,n*n)
        # next line is needed for extend tensor size (from 'batch x kernel x n*n' to 'batch x 1 x kernel x n*n'):
        x_unf = x_unf.view(fre,batches, self.channel,len_*len_, n*n) # (fre,batch,C,len_*len_,n*n)
        x_unf = x_unf.transpose(3,4)
        # select all middle points in LIZs:
        mid_vals = x.view(fre,batches,self.channel, n*n, 1) # (batch,C,n*n,1)
        
        average_term = torch.exp(-x_unf.mean(-1, keepdim=True)).view(fre,batches, self.channel, n, n)
        
        differential_term = (self.inhibition_kernel * F.relu(x_unf - mid_vals)
                            ).sum(4, keepdim=True).view(fre,batches, self.channel, n, n)
        
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


class Inhabited_CSNN(nn.Module):
    def __init__(self, time_step: int=8):
        super().__init__()
        self.num_classes = NUM_CLASSES
        self.time_step = time_step

        self.feature = nn.Sequential(
            # (BN, 3, 128, 128)
            layer.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            LateralInhibition(64),
            layer.BatchNorm2d(64),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2),
            layer.Conv2d(64, 32, kernel_size=3, padding=0),
            LateralInhibition(32),
            layer.BatchNorm2d(32),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2),
            layer.Conv2d(32, 32, kernel_size=3, padding=1),
            LateralInhibition(32),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Conv2d(32, 32, kernel_size=3, padding=1),
            LateralInhibition(32),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Conv2d(32, 16, kernel_size=3, padding=1),
            LateralInhibition(16),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2)
        )

        self.classify = nn.Sequential(
            layer.Flatten(),
            layer.Linear(16 * 7 * 7, 100),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Linear(100, self.num_classes),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.time_step, 1, 1, 1, 1)   # (T, BN, 3, 128, 128)
        x_seq = self.feature(x_seq)
        x_seq = self.classify(x_seq)    # (T, BN, 2)
        fr = x_seq.mean(0)  # (BN, 2)
        return fr
    def get_activations(self,x):
        x_seq = x.unsqueeze(0).repeat(self.time_step, 1, 1, 1, 1)  # (T, BN, 3, 128, 128)
        x = self.feature(x_seq)
        x=x.transpose(0,1)
        return x


class Feedback_CNN(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):
        super(Feedback_CNN, self).__init__()
        self.features = nn.Sequential(
            # input size of 128(batch size)*3*32*32
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),     # 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                              # 8
            nn.Conv2d(64, 192, kernel_size=3, padding=1),             # 8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                              # 4
            nn.Conv2d(192, 384, kernel_size=3, padding=1),            # 4
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),            # 4
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),            # 4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                              # 2
        )
        self.transpose = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3),              # 4
            nn.Sigmoid(),
            nn.ConvTranspose2d(128, 64, kernel_size=5),               # 8
            nn.Sigmoid(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),      # 16
            nn.Sigmoid(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),       # 32
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, num_classes),
            
        )

    def forward(self, x):
        out = self.features(x)
        fb = self.transpose(out)
        out = self.features(x*fb)
        
        out = out.view(out.size(0), 256 * 2 * 2)
        out = self.classifier(out)
        out = F.log_softmax(out,dim=1)
        return out
    
    
    def get_activations(self,x):
        out = self.features(x)
        fb = self.transpose(out)
        out = self.features(x*fb)
        return out







if __name__=='__main__':
    # 准备一个数据集并导入
    
    # x=torch.randn([8,20,3,128,128])
    x=torch.randn([20,3,128,128])
    # CUDA = torch.cuda.is_available()
    # device = 'cuda' if CUDA else 'cpu'
    # x=x.to(device)
    model=Inhabited_CSNN()
    # model.to(device)
    # model=LateralInhibition(3)
    # save_path='/home/ubuntu/files/self_brain_score/Trained_model/CSNN_origin.pth'
    # import os
    # if os.path.isfile(save_path):
    #     model.load_state_dict(torch.load(save_path)['state_dict'])
    #     print('model reload done!')

    y=model(x)
    print(y.shape)
    if torch.cuda.is_available():
        print('CUDA is available. GPU device is:', torch.cuda.get_device_name(0))

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    # r = score(model=model.eval(),first_time=True)
    # print(r)


