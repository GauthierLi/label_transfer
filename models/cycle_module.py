import sys
sys.path.append(r"/media/gauthierli-org/CodingSpace1/code/label_transfer/CFG")
import cfg
import torch
import torch.nn as nn
from torchvision.models import resnet50

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        out = x * torch.tanh(F.softplus(x))
        return out

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,ks=4,pad=2):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,kernel_size=ks,padding=pad,bias=False)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = Mish() # nn.LeakyReLU(inplace=True)
        # self.relu_s1 = nn.Gelu()

    def forward(self,x):
        hx = x
        xout = self.relu_s1(self.bn_s1(hx+self.conv_s1(hx)))
        return xout

class res1m1(nn.Module):
    def __init__(self):
        self.conv = REBNCONV(128, cfg.latent_dim)

class convDown2(nn.Module):
    def __init__(self):
        self.blockA = nn.Sequential(*[REBNCONV(3, 64), REBNCONV(64, 128)])
        self.blockB = nn.Sequential(*[REBNCONV(128, 256), nn.Flatten()])
        self.blockC = nn.Sequential(*[])
        




class cycleBlock(nn.Module):
    def __init__(self, weight='pretrained'):
        self.fea_extra = resnet50(weight=weight)

