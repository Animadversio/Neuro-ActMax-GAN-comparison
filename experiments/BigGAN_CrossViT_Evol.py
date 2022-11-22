import os
from os.path import join
import torch
import torch.nn as nn
from timm import create_model
from core.utils import show_imgrid
from torch.optim import Adam, SGD
import numpy as np
import matplotlib.pyplot as plt
from core.utils import upconvGAN, featureFetcher, featureFetcher_module, CholeskyCMAES
from core.utils import saveallforms, save_imgrid
import matplotlib
from core.utils.GAN_utils import loadBigGAN, BigGAN_wrapper
#%%
# Load BigGAN
BG = BigGAN_wrapper(loadBigGAN().cuda().eval())
# Load CrossViT
model = create_model('crossvit_18_dagger_408', pretrained=True)
model.cuda()
for param in model.parameters():
    param.requires_grad = False
#%% Example of using BigGAN
z = BG.sample_vector(1)
img = BG.visualize(z)
#%%
def BG_grad_evol_scorer(get_score_all_fun, steps=200, lr=0.002, return_zs=True):
    z = 0.1 * BG.sample_vector(1)
    z.requires_grad = True
    optimizer = SGD([z], lr=lr)
    score_traj = []
    zs = []
    for i in range(steps):
        zs.append(z.detach().cpu().clone())
        img = BG.visualize(z)
        model(img)
        score = get_score_all_fun()
        loss = -score.mean()
        loss.backward()
        optimizer.step()
        print(i, f"score {score.item():.3f} z {z.norm().item():.3f} zgrad  {z.grad.norm().item():.3f}")
        optimizer.zero_grad()
        score_traj.append(score.item())
    zs = torch.cat(zs, dim=0)
    show_imgrid(img, )
    if return_zs:
        return zs, img, np.array(score_traj)
    else:
        return z, img, np.array(score_traj)
#%%
savedir = r"F:\insilico_exps\GAN_crossViT\crossvit_18_dagger_384"

keystr = "B1_B1_0_norm1"
module = model.blocks[1].blocks[1][0].norm1
ch_unit = 300, 4
os.makedirs(join(savedir, keystr+"_BigGAN"), exist_ok=True)
fetcher = featureFetcher_module()
for chan in [50, 100, 150, 200, 250, 300]:
    for unit in range(20):
        ch_unit = chan, unit
        fetcher.record_module(module, keystr, ingraph=True)
        get_score_all_fun = lambda: fetcher[keystr][:, ch_unit[0], ch_unit[1]]
        for repi in range(10):
            zs, img, score_traj = BG_grad_evol_scorer(get_score_all_fun, steps=200, lr=0.002, return_zs=True)
            save_imgrid(img, join(savedir, keystr+"_BigGAN", "proto_%s_%d_%d_rep%d_BigGAN.png" % (keystr, ch_unit[0], ch_unit[1], repi)))
            np.savez(join(savedir, keystr+"_BigGAN", "BG_evol_%s_%d_%d_rep%d_BigGAN.npz" % (keystr, ch_unit[0], ch_unit[1], repi)),
                     score_traj=score_traj, z=z[-1, :].detach().cpu(), z_traj=zs.detach().cpu())

        fetcher.cleanup()
#%%
z = 0.1 * BG.sample_vector(1)
# os.makedirs(join(savedir, keystr), exist_ok=True)
keystr = "B1_B1_0_norm1"
module = model.blocks[1].blocks[1][0].norm1
ch_unit = 300, 4
fetcher = featureFetcher_module()
fetcher.record_module(module, keystr, ingraph=True)
get_score_all_fun = lambda: fetcher[keystr][:, ch_unit[0], ch_unit[1]]
z.requires_grad = True
optimizer = SGD([z], lr=0.002)
for i in range(200):
    img = BG.visualize(z)
    model(img)
    score = get_score_all_fun()
    loss = -score.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(i, f"score {score.item():.3f} z {z.norm().item():.3f} zgrad  {z.grad.norm().item():.3f}")
fetcher.cleanup()
#%%
show_imgrid(img)
