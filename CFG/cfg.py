import torch
# general parameters
name = "second_trial_with_sp"

bs = 4  
nw = 0
lr=1e-3
wd = 1e-3
epoch = 500
img_size = 256
latent_dim = 20
device = "cuda" if torch.cuda.is_available else "cpu"

net_type='U2Net'

# data path
data_path = "../../data/small_img1k" 
ckpt_path = f"./checkpoint/U2net_{name}"
log_img = f"./log_img/U2net_{name}"

# resume
resume = False
resume_path = r""