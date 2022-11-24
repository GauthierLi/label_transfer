import os
import sys
import pdb
import torch
import numpy as np
import torch.nn as nn
import CFG.cfg as cfg
import torch.optim as optim

from tqdm import tqdm
from utils.vis import dynamic_pic
from torch.utils.data import Dataset, DataLoader
from dataloaders.dataloader import flowers_dataloader
from models.cycle_module import cycleBlock

"""
rethinking: in order to give s robostic training, it's better to make some masks or noise to force the local mask to convergence.

the next trial: change the activate function into Gelu
"""

Vusialize_iter = 10

def train_one_epoch(cur_epoch, train_loader, models,monitor):
    total_loss = 0.
    N = len(train_loader)
    pbar = tqdm(train_loader)
    lf = nn.BCELoss()
    models['net'].train()
    for num,(img, label) in enumerate(pbar):
        models['opt'].zero_grad()
        img = img.to(cfg.device)
        label = label.to(cfg.device)

        # first supervised
        pred_label, pred_masks = models['net'](img)
        # pdb.set_trace()
        print(torch.argmax(pred_label[0]).item(), torch.argmax(label[0]).item())
        cur_loss = lf(pred_label, label)

        class_of_obj = torch.argmax(pred_label,dim=1)
        masks = torch.stack([pred_masks[i][class_of_obj[i]] for i in range(cfg.bs)], dim=0).detach()

        if epoch % 2 > 10:
            # second supervised
            img1 = torch.mul(img, (1-masks).unsqueeze(dim=1)).detach()
            img2 = torch.mul(img, masks.unsqueeze(dim=1)).detach()

            pred_label1, pred_masks1 = models['net'](img1)
            background = torch.zeros_like(pred_label1)
            background[:,-1] = 1
            cur_loss1 = lf(pred_label1, background)
            pred_label2, pred_masks2 = models['net'](img2)
            cur_loss2 = lf(pred_label2, label)

            class_of_obj1 = torch.argmax(pred_label1,dim=1)
            masks1 = torch.stack([pred_masks1[i][class_of_obj1[i]] for i in range(cfg.bs)], dim=0).detach()
            class_of_obj2 = torch.argmax(pred_label2,dim=1)
            masks2 = torch.stack([pred_masks2[i][class_of_obj2[i]] for i in range(cfg.bs)], dim=0).detach()

            monitor(num, cur_loss2.item(), category="cur_loss2", mode="line", drop_x=True)
            monitor(num, cur_loss1.item(), category="cur_loss1", mode="line", drop_x=True)

            expect_true = cur_loss + cur_loss2
            expect_false = cur_loss1
            (expect_true + 0.01 * expect_false).backward()
            models['opt'].step()

            monitor(img2[0].permute(1,2,0).cpu().detach().numpy(), 0, category="img2", mode="figure")
            monitor(nn.Softmax()(masks1[0]).cpu().detach().numpy(), 0, category="mask1", mode="figure")
            monitor(nn.Softmax()(masks2[0]).cpu().detach().numpy(), 0, category="mask2", mode="figure")
        else:
            cur_loss.backward()
            models['opt'].step()

        monitor(num, cur_loss.item(), category="cur_loss0", mode="line", drop_x=True)
        monitor(img[0].permute(1,2,0).cpu().detach().numpy(), 0, category="img", mode="figure")
        monitor(nn.Softmax()(masks[0]).cpu().detach().numpy(), 0, category="mask0", mode="figure")
        

        pbar.desc = "[epoch:{:}, lr:{:.6f} loss {:.4f}]"\
                    .format(cur_epoch, models['opt'].state_dict()['param_groups'][0]['lr'],cur_loss.item())

        # draw 
        if num % 10 == 0:
            monitor.draw(joint=["img", "img2", "mask0", "mask1", 'mask2', 'cur_loss0',  'cur_loss2' ,'cur_loss1'],
                        row_max=3, pause=0, save_path=os.path.join(cfg.log_img, "view.png"))
    monitor.draw(joint=["img", "img2", "mask0", "mask1", 'mask2', 'cur_loss0',  'cur_loss2' ,'cur_loss1'],
                        row_max=3, pause=0, save_path=os.path.join(cfg.log_img, f"epoch_{cur_epoch}.png"))
    
    return total_loss / float(N)

if __name__ == "__main__":
    if not os.path.isdir(cfg.ckpt_path):
        os.makedirs(cfg.ckpt_path)
    if not os.path.isdir(cfg.log_img):
        os.makedirs(cfg.log_img)
    
    monitor = dynamic_pic(2000)
    mapdict, train_loader = flowers_dataloader()
    print(mapdict)

    net = cycleBlock().to(cfg.device)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    lr_schedule = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=24,T_mult=2,eta_min=1e-6)

    models_dict = {"net":net, "opt":optimizer, "lr_sch":lr_schedule}
    best_loss = 999999999999999
    for epoch in range(cfg.epoch):
        loss = train_one_epoch(epoch, train_loader, models_dict, monitor)
        models_dict['lr_sch'].step()
        ckpt = {"state_dict":models_dict['net'].state_dict(), "epoch":epoch}
        torch.save(ckpt, os.path.join(cfg.ckpt_path, f"last_epoch.pth"))
        if loss < best_loss:
            best_loss = loss
            torch.save(ckpt, os.path.join(cfg.ckpt_path, f"best_epoch.pth"))