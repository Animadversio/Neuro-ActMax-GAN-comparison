
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, LinearLR
from tqdm import trange, tqdm
from pathlib import Path
from core.utils.GAN_utils import loadBigGAN, BigGAN_wrapper, upconvGAN
from core.utils.plot_utils import saveallforms, show_imgrid, save_imgrid
from neuro_data_analysis.neural_data_lib import get_expstr, load_neural_data, parse_montage
import os
from os.path import join
outdir = r"E:\OneDrive - Harvard University\SfN2023\FigureAnims\Schematics_obj_cluster\src"
#%%
os.makedirs(outdir, exist_ok=True)
#%%
BG = loadBigGAN()
BGW = BigGAN_wrapper(BG)
#%%
G = upconvGAN("fc6")
G.eval().cuda()
#%%
class_id1 = 382
torch.manual_seed(42)
z = BGW.sample_vector(5, class_id=class_id1)
imgs = BGW.visualize(z)
show_imgrid(imgs)
for i in range(5):
    save_imgrid(imgs[i], join(outdir, f"obj_{class_id1}_{i:02d}.png"))
#%%
class_vec = z[1, 128:]
noise_vec = z[1, :128]
# add noise to the noise vector
noise_vec_pert = noise_vec[None, ] + torch.randn(10, 128).cuda() * 0.4
z_pert = torch.cat([noise_vec_pert,
                    class_vec.repeat(10, 1), ], dim=1)
imgs_pert = BGW.visualize(z_pert)
show_imgrid(imgs_pert, nrow=5)
for i in range(len(imgs_pert)):
    save_imgrid(imgs_pert[i], join(outdir, f"obj_{class_id1}_1_pert{i:02d}.png"))
#%% save each image to disk
class_id2 = 225
torch.manual_seed(45)
z2 = BGW.sample_vector(5, class_id=class_id2)
imgs2 = BGW.visualize(z2)
show_imgrid(imgs2)
for i in range(5):
    save_imgrid(imgs2[i], join(outdir, f"obj_{class_id2}_{i:02d}.png"))
#%%
#%%
class_vec = z2[1, 128:]
noise_vec = z2[1, :128]
# add noise to the noise vector
noise_vec_pert = noise_vec[None, ] + torch.randn(10, 128).cuda() * 0.4
z_pert = torch.cat([noise_vec_pert,
                    class_vec.repeat(10, 1), ], dim=1)
imgs_pert2 = BGW.visualize(z_pert)
show_imgrid(imgs_pert2, nrow=5)
for i in range(len(imgs_pert)):
    save_imgrid(imgs_pert2[i], join(outdir, f"obj_{class_id2}_1_pert{i:02d}.png"))
#%%
# linear interpolation between z1[1] and z2[1]
ticks = torch.linspace(0, 1, 10).cuda()
z_interp = z[1, None, ] * ticks[:, None] + z2[1, None, ] * (1 - ticks[:, None])
imgs_interp = BGW.visualize(z_interp)
show_imgrid(imgs_interp, nrow=5)
#%% save each image to disk
for i in range(len(imgs_interp)):
    save_imgrid(imgs_interp[i], join(outdir, f"obj_{class_id1}_{class_id2}_interp{i:02d}.png"))

#%%

