import sys
sys.path.append(r"CFG/")
import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        out = x * torch.tanh(F.softplus(x))
        return out

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,ks=4,stride=2,pad=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,kernel_size=ks,padding=pad,stride=2,bias=False)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.LeakyReLU(inplace=True)
        # self.relu_s1 = nn.Gelu()

    def forward(self,x):
        xout = self.relu_s1(self.bn_s1(self.conv_s1(x)))
        return xout

class res1m1(nn.Module):
    def __init__(self):
        super(res1m1, self).__init__()
        self.conv1 = REBNCONV(128,cfg.latent_dim, ks=1, pad=0)
        self.branch0 = nn.Sequential(*[REBNCONV(128,128),
                                    nn.Conv2d(128, cfg.latent_dim, kernel_size=1, padding=0)])
        self.branch1 = nn.Conv2d(cfg.latent_dim, cfg.latent_dim, kernel_size=1, padding=0)
        self.branch2 = nn.Conv2d(cfg.latent_dim, cfg.latent_dim, kernel_size=1, padding=0) # cat :inp 2*latent_dim

    def forward(self, x):
        hx = self.branch0(x)
        x = self.conv1(x)
        x = self.branch1(x)
        out = hx + x 
        # out = torch.cat([hx,x], 0)
        out = self.branch2(out)
        return out

def _upsample_like(src,tar):
    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear')
    return src

class cycleBlock(nn.Module):
    def __init__(self, classes=cfg.classes):
        r"""
         @input: classes number
        """
        super(cycleBlock, self).__init__()
        self.blockA = nn.Sequential(*[REBNCONV(3, 64), REBNCONV(64, 128)])
        self.blockB = nn.Sequential(*[REBNCONV(128, 256), nn.Flatten(),
                                     nn.Linear(4 * cfg.img_size ** 2, 1024, bias=cfg.bias)],
                                     nn.BatchNorm1d(1024),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(1024, 64))
        self.blockC = res1m1()
        self.blockD = nn.Sequential(*[nn.Linear(64, cfg.latent_dim * classes),
                                     nn.BatchNorm1d(cfg.latent_dim * classes),
                                     nn.LeakyReLU(inplace=True)])

        self.GAP = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        feature1 = self.blockA(x)

        local_features = self.blockC(feature1)
        lin_features1 = self.blockB(feature1)

        class_representations = self.blockD(lin_features1)
        class_representations = class_representations.view(-1, cfg.latent_dim,  cfg.classes)

        masks = (torch.einsum('bpk, bpuv -> bkuv', [class_representations, local_features]))
        labels = torch.softmax(self.GAP(masks).squeeze(dim=-1).squeeze(dim=-1), dim=-1)
        
        masks = _upsample_like(masks, x)
        return labels, masks

if __name__ == '__main__':
    # img = torch.randn((4, 3, 512, 512)).to("cuda")
    # net = cycleBlock().to("cuda")

    # f = net(img)
    # print(f[0].shape, f[1].shape)

    a=torch.rand((3, 4, 2, 2))
    b=torch.tensor([[[1,0],[0,0]], [[0,1],[0,0]], [[0,0],[1,0]]])
    print(1-b)
    c = torch.einsum("bchw, bhw->bchw", [a,b])
    # c=a*b
    # d=torch.mul(a,b)
    print(a)
    # print(b.shape)
    print(c)
    # print(d)
