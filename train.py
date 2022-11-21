import sys 
sys.path.append(r"/media/gauthierli-org/CodingSpace1/code/label_transfer/CFG")

import os
import pdb
import torch
import numpy as np
import torch.nn as nn
import CFG.cfg as cfg
import torch.optim as optim

from tqdm import tqdm
from utils.vis import dynamic_pic
from models.miNets import TotalMI, LocalMI
from torch.utils.data import Dataset, DataLoader
from dataloaders.dataloader import cell_dataloader, flowers_dataloader
from models.encoders import Encoder, Decoder, feature_compress, Discriminator, UnetEncoder, U2Encoder, U2Decoder

"""
rethinking: in order to give s robostic training, it's better to make some masks or noise to force the local mask to convergence.

the next trial: change the activate function into Gelu
"""

Vusialize_iter = 10

def train_one_epoch(cur_epoch, models, train_loader, monitor):
    total_recons_loss, total_mi_loss, total_label_loss = 0., 0., 0.
    N = len(train_loader)
    pbar = tqdm(train_loader, desc=f"[epoch {epoch} training]")

    recons_lf = nn.MSELoss()
    bce_lf = nn.BCELoss()

    for num, (img, label) in enumerate(pbar):
        img = img.to(cfg.device)
        label = label.to(cfg.device)

        # 1 reconstract
        models["encoder"]["net"].train()
        models["decoder"]["net"].train()
        models["feature_compress"]["net"].eval()
        models["mi"]["net"].eval()
        models["encoder"]["optim"].zero_grad()
        models["decoder"]["optim"].zero_grad()

        if cfg.net_type == "U2Net":
            features = models["encoder"]["net"](img)
            reconstruct = models["decoder"]["net"](features)
            # pdb.set_trace()
            recons_loss = 0
            for rec in reconstruct:
                recons_loss += recons_lf(img, rec)
        else:
            feature = models["encoder"]["net"](img)
            reconstruct = models["decoder"]["net"](feature)
            recons_loss = recons_lf(img , reconstruct) 
            

        total_recons_loss += recons_loss.item()
        if num % Vusialize_iter == 0:
            monitor(num, recons_loss.item(), category="recons", mode="line", drop_x=True)
            monitor((255 * img[0].permute(1,2,0).cpu().detach().numpy()).astype("uint8"), 0, category="ori", mode="figure")
            if cfg.net_type == "U2Net":
                monitor((255 * reconstruct[0][0].permute(1,2,0).cpu().detach().numpy()).astype("uint8"), 0, category="rec", mode="figure")
            else:
                monitor((255 * reconstruct[0].permute(1,2,0).cpu().detach().numpy()).astype("uint8"), 0, category="rec", mode="figure")
        recons_loss.backward()

        # 2. maximum mi and minimize label loss
        models["encoder"]["net"].train()
        models["feature_compress"]["net"].train()
        models["mi"]["net"].train()
        models["decoder"]["net"].eval()
        models["encoder"]["optim"].zero_grad()
        models["feature_compress"]["optim"].zero_grad()
        models["mi"]["optim"].zero_grad()

        if cfg.net_type == "U2Net":
            feature = models["encoder"]["net"](img)[-1]
        else:
            feature = models["encoder"]["net"](img)
        representation = models["feature_compress"]["net"](feature)

        mi_loss = models["mi"]["net"](feature, representation) # + torch.norm(representation, p=2, dim=1).mean()
        label_loss = bce_lf(representation, label)
        total_mi_loss += mi_loss.item()
        total_label_loss += label_loss.item()
        if num % Vusialize_iter == 0:
            monitor(num, mi_loss.item(), category="mi loss", drop_x=True)
            monitor(num, label_loss.item(), category="label loss", drop_x=True)
        mi_label_loss = mi_loss + 10 * label_loss

        mi_label_loss.backward()
        models["encoder"]["optim"].step()
        models["decoder"]["optim"].step()
        models["feature_compress"]["optim"].step()
        models["mi"]["optim"].step()

        # vusualize_mask
        if num % Vusialize_iter == 0:
            mask = models["mi"]["net"].lmi(feature, representation, False)
            monitor(((mask[0] * 255).detach().cpu().numpy()).astype("uint8"),0, category="mask", mode="figure")
            # draw
            monitor.draw(joint=["recons", "mi loss","label loss" , "ori", "rec", "mask"],
                        row_max=2, pause=0, save_path=os.path.join(cfg.log_img, "view.png"))

    total = total_recons_loss / N + total_mi_loss / N + total_label_loss / N
    monitor.draw(joint=["recons", "mi loss","label loss" , "ori", "rec", "mask"], 
                    row_max=2, pause=0, save_path=os.path.join(cfg.log_img, f"epoch{epoch}.png"))
    return total

if __name__ == "__main__":
    if not os.path.isdir(cfg.ckpt_path):
        os.makedirs(cfg.ckpt_path)
    if not os.path.isdir(cfg.log_img):
        os.makedirs(cfg.log_img)
    
    monitor = dynamic_pic(2000)
    mapdict, train_loader = flowers_dataloader()
    print(mapdict)

    if cfg.net_type == "U2Net":
        encoder = U2Encoder(latent_dim=cfg.latent_dim).to(cfg.device)
        decoder = U2Decoder().to(cfg.device)
        tmi_loss = TotalMI().to(cfg.device)
    else:
        encoder = UnetEncoder(latent_dim=cfg.latent_dim).to(cfg.device)
        decoder = Decoder().to(cfg.device)
        tmi_loss = TotalMI().to(cfg.device)
    fea_compress = feature_compress().to(cfg.device)
    discrim = Discriminator().to(cfg.device)

    optim_en = optim.AdamW(encoder.parameters(), lr=cfg.lr,weight_decay=cfg.wd)
    optim_de = optim.AdamW(decoder.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    optim_feac = optim.AdamW(fea_compress.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    optim_tmi = optim.AdamW(tmi_loss.parameters(), lr=cfg.lr)
    optim_discrim = optim.AdamW(discrim.parameters(), lr=cfg.lr, weight_decay=cfg.wd)


    lrscd_en = optim.lr_scheduler.ExponentialLR(optim_en, gamma=0.1)
    lrscd_de = optim.lr_scheduler.ExponentialLR(optim_de, gamma=0.1)
    lrscd_feac = optim.lr_scheduler.ExponentialLR(optim_feac, gamma=0.1)
    lrscd_tmi = optim.lr_scheduler.ExponentialLR(optim_tmi, gamma=0.1)
    lrscd_discrim = optim.lr_scheduler.ExponentialLR(optim_discrim, gamma=0.1)

    # lrscd_en = optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_en, T_0=10, T_mult=2, eta_min=1e-8)
    # lrscd_de = optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_de, T_0=10, T_mult=2, eta_min=1e-8)
    # lrscd_feac = optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_feac, T_0=10, T_mult=2, eta_min=1e-8)
    # lrscd_mi = optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_mi, T_0=10, T_mult=2, eta_min=1e-8)
    # lrscd_discrim = optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_discrim, T_0=10, T_mult=2, eta_min=1e-8)


    models = {"encoder": {"net":encoder, "optim":optim_en, "lr_scd": lrscd_en},
              "decoder": {"net":decoder, "optim":optim_de, "lr_scd": lrscd_de},
              "feature_compress": {"net":fea_compress, "optim":optim_feac, "lr_scd": lrscd_feac},
              "mi":{"net":tmi_loss, "optim":optim_tmi, "lr_scd": lrscd_tmi},
              "discriminator":{"net":discrim, "optim":optim_discrim, "lr_scd": lrscd_discrim}}
    
    if cfg.resume:
        ckpt = torch.load(cfg.resume_path)
        cur_epoch = ckpt['epoch']
        for key in models:
            models[key]["net"].load_state_dict(ckpt[key])

    best_loss = 999999999999999
    for epoch in range(cur_epoch, cfg.epoch) if cfg.resume else range(cfg.epoch):
        loss = train_one_epoch(epoch, models,train_loader, monitor)
        for key in models:
            models[key]["lr_scd"].step()

        ckpt = dict()
        for key in models:
            ckpt[key] = models[key]["net"].state_dict()
        ckpt['epoch'] = epoch
        torch.save(ckpt, os.path.join(cfg.ckpt_path, f"last_epoch.pth"))
        if loss < best_loss:
            best_loss = loss
            torch.save(ckpt, os.path.join(cfg.ckpt_path, f"best_epoch.pth"))