"""
Invert some images into another GAN and analyze their IS and FID
"""
import torch
import numpy as np
from pytorch_gan_metrics.utils import ImageDataset
from pytorch_gan_metrics.core  import torch_cov, get_inception_feature, calculate_inception_score, calculate_frechet_distance
from pytorch_pretrained_biggan import BigGAN
from core.utils.GAN_utils import upconvGAN, BigGAN_wrapper, loadBigGAN
from core.utils import saveallforms, showimg, show_imgrid, save_imgrid
from core.utils.GAN_invert_utils import GAN_invert, GAN_invert_with_scheduler
biggan = BigGAN.from_pretrained("biggan-deep-256")
biggan.eval().requires_grad_(False).cuda()
BG = BigGAN_wrapper(biggan)
G = upconvGAN("fc6")
G.cuda().requires_grad_(False).eval()

imageset_str = "BigGAN_norm_std07"
BG_rn_fun = lambda batch_size: \
        BG.visualize(0.7 * torch.randn(batch_size, 256, device="cuda"))
#%%
imageset_str = "BigGAN_1000cls_std07"
BG_cls_fun = lambda batch_size: BG.visualize(BG.sample_vector(batch_size, class_id=None))

#%%
target_img = BG_cls_fun(10)
z_init = torch.randn(10, 4096, device="cuda")
z_opt, img_opt, losses = GAN_invert(G, target_img, z_init, lr=2e-3, max_iter=2000, print_progress=False)
#%%
target_img = BG_cls_fun(40)
z_init = torch.randn(40, 4096, device="cuda")
z_opt, img_opt, losses = GAN_invert(G, target_img, z_init, lr=2e-3, max_iter=2000, print_progress=False)

#%%
show_imgrid(img_opt)
show_imgrid(target_img)
#%%
import sys
from os.path import join
if sys.platform == "linux":
    Hdir_fc6 = join("/home/biw905/Hessian", "Evolution_Avg_Hess.npz")  #r"/scratch/binxu/GAN_hessian/FC6GAN/summary/Evolution_Avg_Hess.npz"
else:
    Hdir_fc6 = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\fc6GAN\Evolution_Avg_Hess.npz"

H = np.load(Hdir_fc6)
#%%
# H_avg = H["H_avg"]
eigvecs = H["eigvect_avg"]
eigvecs_th = torch.from_numpy(eigvecs).float().cuda()
#%%
from torch.optim import Adam
from tqdm import trange
def GAN_Hessian_invert(G, target_img, eigvecs=None, z_init=None, lr=2e-3, weight_decay=0e-4, max_iter=5000, print_progress=True):
    if z_init is None:
        z_init = torch.randn(5, 4096, device="cuda")
    z_init = z_init.detach().clone()
    if target_img.device != "cuda":
        target_img = target_img.cuda()
    w_opt = z_init if eigvecs is None else (z_init @ eigvecs)
    w_opt.requires_grad_(True)
    opt = Adam([w_opt], lr=lr, weight_decay=weight_decay)
    pbar = trange(max_iter)
    for i in pbar:
        z_opt = w_opt if eigvecs is None else (w_opt @ eigvecs.t())
        img_opt = G.visualize(z_opt)
        losses = ((img_opt - target_img) ** 2).mean(dim=(1, 2, 3))  # changed from mean to sum
        loss = losses.sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        pbar.set_description(f"loss: {losses.mean().item():.3f}")
        if print_progress:
            print(i, losses.mean().item())
    img_opt = G.visualize(z_opt.detach())
    return z_opt, img_opt
#%%
target_img = BG_cls_fun(10)
z_init = torch.randn(10, 4096, device="cuda")
#%%
z_opt, img_opt_H = GAN_Hessian_invert(G, target_img, eigvecs=eigvecs_th, z_init=z_init, lr=2e-3, weight_decay=0e-4, max_iter=2000, print_progress=False)
# 1999 0.018893277272582054
z_opt, img_opt_noH = GAN_Hessian_invert(G, target_img, eigvecs=None, z_init=z_init, lr=2e-3, weight_decay=0e-4, max_iter=2000, print_progress=False)
# 0.015:
#%%
show_imgrid(img_opt_H, nrow=5)
show_imgrid(img_opt_noH, nrow=5)
