import os
import sys
import pdb
import torch
sys.path.append(r"/media/gauthierli-org/CodingSpace1/code/label_transfer/CFG")
sys.path.append(r"/media/gauthierli-org/CodingSpace1/code/label_transfer/models")

import cfg

import numpy as np
import torch.nn as nn
from encoders import DoubleConv
import torch.nn.functional as F
from torch.autograd import Variable

# class GlobalMI(nn.Module):
#     """with great parameters of weight decays !!!!!!!!!"""
#     def __init__(self):
#         super(GlobalMI, self).__init__()
#         self.lin = nn.Sequential(nn.Linear(cfg.latent_dim*cfg.img_size//32*cfg.img_size//32 + cfg.latent_dim, 1024, bias=True), nn.LeakyReLU(),
#                                  nn.Linear(1024, 512, bias=True), nn.LeakyReLU(),
#                                  nn.Linear(512,  256, bias=True), nn.LeakyReLU(),
#                                  nn.Linear(256, 1, bias=True))
    
#     def forward(self, features, representation):
#         B,C,W,H = features.shape
#         features = features.view(B, -1)
#         x = torch.cat([features, representation],dim=1 )
#         # pdb.set_trace()
#         return nn.Sigmoid()(self.lin(x)).squeeze(dim=1)


class LocalMI(nn.Module):
    "new version, not normalized"
    def __init__(self):
        super(LocalMI, self).__init__()
        self.transconv1 = nn.ConvTranspose2d(in_channels=cfg.latent_dim, out_channels=cfg.latent_dim, stride=2,padding=1,output_padding=1,kernel_size=4,groups=cfg.latent_dim)
        self.transconv2 = nn.ConvTranspose2d(in_channels=cfg.latent_dim, out_channels=cfg.latent_dim, stride=2,padding=1,output_padding=1,kernel_size=4,groups=cfg.latent_dim)
        self.transconv3 = nn.ConvTranspose2d(in_channels=cfg.latent_dim, out_channels=cfg.latent_dim, stride=1,padding=1,output_padding=0,kernel_size=1,groups=cfg.latent_dim)
        self.in1 = nn.InstanceNorm2d(num_features=cfg.latent_dim)
        self.activate = nn.LeakyReLU()

    def forward(self, features, representation, mean=True):
        assert features.shape[1] == representation.shape[-1], f"features dimention {features.shape[1]} not match representation dimention {representation.shape[-1]}..."

        # features_norm = 1/torch.norm(features, p=2, dim=1)
        # representation_norm = 1/torch.norm(representation, p=2, dim=1)
        # features = torch.einsum('bchw, bhw -> bchw', [features, features_norm])
        # representation = torch.einsum('bc, b -> bc', [representation, representation_norm])

        # features = self.activate(self.in1(self.transconv1(features)))
        # features = self.activate(self.in1(self.transconv2(features)))
        # features = self.activate(self.in1(self.transconv3(features)))

        out = nn.Sigmoid()(torch.einsum('bchw, bc -> bhw', [features, representation]))
        if mean:
            out = out.mean(dim=(1,2))
            return out
        else:
            print("map size", out.shape)
            return F.interpolate(out.unsqueeze(dim=1),size=[cfg.img_size,cfg.img_size],mode='bilinear').squeeze(dim=1)
    
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class TotalMI(nn.Module):
    def __init__(self):
        super().__init__()
        self.lmi = LocalMI()
        # self.gmi = GlobalMI()
    
    def _get_focus(self, labels):
        B, _ = labels.shape
        mapmat = Variable(torch.zeros(B,B), requires_grad=False)
        for i in range(B):
            for j in range(B):
                if labels[i].equal(labels[j]):
                    mapmat[i][j] = 1
        return mapmat

    def forward(self, features, representation):
        B,C,H,W = features.shape
        lmi_scoremap = torch.zeros(B,B)
        # gmi_scoremap = torch.zeros(B,B)

        for i in range(B):
            for j in range(B):
                lmi_scoremap[i][j] = self.lmi(features[i].unsqueeze(dim=0), representation[j].unsqueeze(dim=0))
                # gmi_scoremap[i][j] = self.gmi(features[i].unsqueeze(dim=0), representation[j].unsqueeze(dim=0))

        # lmi_scoremap = nn.Softmax(dim=1)(lmi_scoremap)

        # gmi_scoremap = nn.Softmax(dim=1)(gmi_scoremap)

        focus = Variable(torch.eye(B, dtype=torch.float32), requires_grad=False) # self._get_focus(labels)
        zero_focus = Variable(1 - focus , requires_grad=False)

        sp = nn.Softplus()
        lmi_score = sp(- focus * lmi_scoremap).mean() - sp(focus * lmi_scoremap).mean()
        # gmi_score = (zero_focus * gmi_scoremap).sum() - (focus * lmi_scoremap).sum()

        return lmi_score

def test_enisum():
    a = torch.tensor([1.,2.,3.])
    b = torch.ones((3,2,2))
    print(a.shape, b.shape)
    c = torch.einsum('a, abc -> bc', [a, b])
    
    norm = torch.norm(b, p=2,dim=0 )
    norm_inv = 1/norm
    print(norm)
    print(norm_inv * 3)

if __name__ == "__main__":
    # test_enisum()

    tst_features = torch.ones(4, cfg.latent_dim, 16, 16)
    tst_rep = torch.randn(4, cfg.latent_dim)

    # gmi = GlobalMI()
    lmi = LocalMI()
    glo = TotalMI()
    # print(gmi(tst_features, tst_rep))
    print("lmi", lmi(tst_features, tst_rep, mean=False).shape)
    print("glo", glo(tst_features, tst_rep))
    