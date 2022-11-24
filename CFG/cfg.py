import torch
# general parameters
name = "second_trial_with_sp_3rd_128"

bs = 2
nw = 0
lr=1e-3
wd = 1e-3
classes = 21
epoch = 500
img_size = 128
latent_dim = 1024
device = "cuda" if torch.cuda.is_available else "cpu"

# framework info
feature_compress_drop_rate = 0.8
bias = True

# data path
data_path = "../../data/small_img1k" 
ckpt_path = f"./checkpoint/U2net_{name}"
log_img = f"./log_img/U2net_{name}"

# resume
resume = False
resume_path = r""