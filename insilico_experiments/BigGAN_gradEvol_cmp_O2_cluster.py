""" Cluster version of BigGAN Evol """
import math
import sys
sys.path.append(r"/home/biw905/Github/Neuro-ActMax-GAN-comparison")
import os
import tqdm
import numpy as np
from os.path import join
import matplotlib.pylab as plt
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample, one_hot_from_names, save_as_images)
from core.utils.CNN_scorers import TorchScorer
from core.utils.GAN_utils import BigGAN_wrapper, upconvGAN, loadBigGAN
from core.utils.grad_RF_estim import grad_RF_estimate, gradmap2RF_square
from core.utils.layer_hook_utils import get_module_names, layername_dict, register_hook_by_module_names
from core.utils.Optimizers import CholeskyCMAES, HessCMAES, ZOHA_Sphere_lr_euclid
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms

#%%
if sys.platform == "linux":
    # rootdir = r"/scratch/binxu/BigGAN_Optim_Tune_new"
    # Hdir_BigGAN = r"/scratch/binxu/GAN_hessian/BigGAN/summary/H_avg_1000cls.npz"
    # Hdir_fc6 = r"/scratch/binxu/GAN_hessian/FC6GAN/summary/Evolution_Avg_Hess.npz"
    # O2 path interface
    scratchdir = "/n/scratch3/users/b/biw905"  # os.environ['SCRATCH1']
    rootdir = join(scratchdir, "GAN_Evol_cmp")
    Hdir_BigGAN = join("/home/biw905/Hessian", "H_avg_1000cls.npz")  #r"/scratch/binxu/GAN_hessian/BigGAN/summary/H_avg_1000cls.npz"
    Hdir_fc6 = join("/home/biw905/Hessian", "Evolution_Avg_Hess.npz")  #r"/scratch/binxu/GAN_hessian/FC6GAN/summary/Evolution_Avg_Hess.npz"
else:
    # rootdir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_Optim_Tune_tmp"
    rootdir = r"D:\Cluster_Backup\GAN_Evol_cmp" #r"E:\Monkey_Data\BigGAN_Optim_Tune_tmp"
    Hdir_BigGAN = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN\H_avg_1000cls.npz"
    Hdir_fc6 = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\fc6GAN\Evolution_Avg_Hess.npz"

#%%
# from argparse import ArgumentParser
# parser = ArgumentParser()
# parser.add_argument("--net", type=str, default="alexnet", help="Network model to use for Image distance computation")
# parser.add_argument("--layer", type=str, default="fc6", help="Network model to use for Image distance computation")
# parser.add_argument("--chans", type=int, nargs='+', default=[0, 25], help="")
# parser.add_argument("--G", type=str, default="BigGAN", help="")
# parser.add_argument("--optim", type=str, nargs='+', default=["HessCMA", "HessCMA_class", "CholCMA", "CholCMA_prod", "CholCMA_class"], help="")
# parser.add_argument("--steps", type=int, default=100, help="")
# parser.add_argument("--reps", type=int, default=2, help="")
# parser.add_argument("--RFresize", type=bool, default=False, help="")
# args = parser.parse_args() # ["--G", "BigGAN", "--optim", "HessCMA", "CholCMA","--chans",'1','2','--steps','100',"--reps",'2']
#%% Select GAN
def load_GAN(name):
    if name == "BigGAN":
        BGAN = BigGAN.from_pretrained("biggan-deep-256")
        BGAN.eval().cuda()
        for param in BGAN.parameters():
            param.requires_grad_(False)
        G = BigGAN_wrapper(BGAN)
    elif name == "fc6":
        G = upconvGAN("fc6")
        G.eval().cuda()
        for param in G.parameters():
            param.requires_grad_(False)
    else:
        raise ValueError("Unknown GAN model")
    return G


def load_Hessian(name):
    # Select Hessian
    try:
        if name == "BigGAN":
            H = np.load(Hdir_BigGAN)
        elif name == "fc6":
            H = np.load(Hdir_fc6)
        else:
            raise ValueError("Unknown GAN model")
    except:
        print("Hessian not found for the specified GAN")
        H = None
    return H
#%%
from easydict import EasyDict as edict
args = edict()
args.G = "BigGAN"
args.net = "resnet50"
#%%
G = load_GAN(args.G)
Hdata = load_Hessian(args.G)
scorer = TorchScorer(args.net, imgpix=227)
#%%
# scorer.select_unit(("resnet50_linf8", ".layer3", 5, 7, 7), allow_grad=True)
#%%
# from core.utils.layer_hook_utils import get_module_names
# from torchvision.transforms import ToTensor, Compose, Normalize, Resize
# preprocess = Compose([
#     Resize(224),
#     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ToTensor(),
# ])
# get_module_names(scorer.model, input_size=(3, 224, 224), device="cuda");
#%%
evc = torch.tensor(Hdata["eigvects_avg"]).cuda()
#%%
from torch.optim import SGD, Adam
scorer.select_unit(("resnet50_linf8", ".layer4", 15, 7, 7), allow_grad=True)
hess_param = True
z = G.sample_vector(5).cuda()
if hess_param:
    w = z @ evc
else:
    w = z
w.requires_grad_(True)
optim = Adam([w], lr=0.01)
score_traj = []
z_traj = []
for i in range(100):
    optim.zero_grad()
    z = (w @ evc.t()) if hess_param else w
    img = G.visualize(z)
    score = scorer.score_tsr_wgrad(img)
    score_traj.append(score.detach().cpu())
    z_traj.append(z.detach().cpu())
    loss = - score.sum()
    loss.backward()
    optim.step()
    zero_mask = (score == 0)
    if zero_mask.sum() > 0:
        new_z = G.sample_vector(zero_mask.sum())
        w.data[zero_mask] = new_z @ evc if hess_param else new_z
    # zero_mask = (score == 0).float()[:, None]
    # z.data.add_(zero_mask * torch.randn_like(z) * 0.05)
    # z.data[zero_mask,] = (1 - zero_mask) * z.data + \
    #          (zero_mask * G.sample_vector(5).cuda())
    print("  ".join(["%.2f"%s for s in score.detach()]))

scorer.cleanup()
idx = torch.argsort(score.detach().cpu(), descending=True)
score_traj = torch.stack(score_traj)
z_traj = torch.stack(z_traj)
img = img.detach()[idx]
z_traj = z_traj[:, idx, :]  # sort the sample
score_traj = score_traj[:, idx]   # sort the sample
noise_norm = z_traj[:, :, :128].norm(dim=-1)
class_norm = z_traj[:, :, 128:].norm(dim=-1)
#%%
figh, axs = plt.subplots(1, 3, figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.plot(score_traj)
plt.ylabel("score traj")
plt.subplot(1, 3, 2)
plt.plot(noise_norm)
plt.ylabel("noise norm")
plt.subplot(1, 3, 3)
plt.plot(class_norm)
plt.ylabel("class norm")
plt.suptitle("BigGAN")
plt.tight_layout()
saveallforms(savedir, "score_traj")
plt.show()

#%%
show_imgrid(img, nrow=5, )