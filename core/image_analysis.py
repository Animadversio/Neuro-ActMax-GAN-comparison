"""
Quantify naturalness of generated images.
    Compare the BigGAN and FC6 GAN images in terms of FID score.
    and Fourier power spectrum.

! pip install pytorch-gan-metrics
! pip install pytorch-pretrained-biggan
"""

import os
from os.path import join
import torch
import numpy as np
from pytorch_gan_metrics import get_inception_score, get_fid
from pytorch_gan_metrics.utils import ImageDataset
from pytorch_gan_metrics.core  import torch_cov, get_inception_feature, calculate_inception_score, calculate_frechet_distance
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop
from torch.utils.data import DataLoader
savedir = r"E:\OneDrive - Harvard University\GAN_imgstats_cmp\Inception"
#%%
img_size = 256
INroot = r"E:\Datasets\imagenet-valid\valid"
transform = Compose([Resize([img_size, img_size]),
                     CenterCrop([img_size, img_size]),
                     ToTensor()])
# dataset = ImageDataset(r"E:\Datasets\imagenet-valid\valid")
INdataset = ImageDataset(root=INroot, transform=transform)
#%% INet
batch_size = 80
num_workers = os.cpu_count()
loader = DataLoader(INdataset, batch_size=batch_size, num_workers=num_workers)
acts, probs = get_inception_feature(
    loader, dims=[2048, 1008], use_torch=True, verbose=True)
mu = torch.mean(acts, dim=0).cpu().numpy()
sigma = torch_cov(acts, rowvar=False).cpu().numpy()
np.savez_compressed(r"E:\Datasets\imagenet-valid\inception_stats.npz", mu=mu, sigma=sigma)
inception_score, IS_std = calculate_inception_score(probs, 10, use_torch=True)
np.savez_compressed(r"E:\Datasets\imagenet-valid\IS_stats.npz", IS=inception_score, IS_std=IS_std)
# (211.13711547851562, 3.3677103519439697)
#%% FC6
from core.utils.GAN_utils import upconvGAN, BigGAN_wrapper, loadBigGAN
class GANDataloader(DataLoader):
    """A hacky way to create a dataloader from a GAN model.
        Saving the images to disk will increase IO time.

        :arg
            render_fun: a function that takes batch_size as input and returns a batch of images.
            batch_size: batch size for the dataloader.
            total_imgnum: total number of images to be generated.

        :example
            FG_fun = lambda batch_size: \
                    FG.visualize(torch.randn(batch_size, 4096, device="cuda"))
            FC6_loader = GANDataloader(FG_fun, 40, 120)
            for imgtsr in iter(FC6_loader):
                print(imgtsr.shape)
    """
    def __init__(self, render_fun, batch_size, total_imgnum, ):
        # self.dataset =
        self.render_fun = render_fun
        self.batch_size = batch_size
        self.total_imgnum = total_imgnum
        self.dataset = [None] * total_imgnum  # dummy dataset
        super().__init__(self.dataset, batch_size=batch_size, shuffle=False, )

    def __iter__(self):
        for i in range(0, self.total_imgnum, self.batch_size):
            yield self.render_fun(self.batch_size)

    def __len__(self):
        return (1 + self.total_imgnum // self.batch_size)
#%%

FG_fun = lambda batch_size: \
        FG.visualize(torch.randn(batch_size, 4096, device="cuda"))
FC6_loader = GANDataloader(FG_fun, 40, 120)
for imgtsr in iter(FC6_loader):
    print(imgtsr.shape)
#%% Load in FC6 GAN
FG = upconvGAN()
FG.eval().requires_grad_(False)
FG.cuda()
#%%
imageset_str = "FC6_std4"
FG4_fun = lambda batch_size: \
        FG.visualize(4 * torch.randn(batch_size, 4096, device="cuda"))
FC6_loader = GANDataloader(FG4_fun, batch_size=40, total_imgnum=50000)
with torch.no_grad():
    acts, probs = get_inception_feature(FC6_loader, dims=[2048, 1008], use_torch=True, verbose=True)
mu = torch.mean(acts, dim=0).cpu().numpy()
sigma = torch_cov(acts, rowvar=False).cpu().numpy()
np.savez_compressed(join(savedir, f"{imageset_str}_inception_stats.npz"), mu=mu, sigma=sigma)
inception_score, IS_std = calculate_inception_score(probs, 10, use_torch=True)
np.savez(join(savedir, f"{imageset_str}_IS_stats.npz"), IS=inception_score, IS_std=IS_std)
#%%
with np.load(join(savedir, f"{'FC6_std4'}_IS_stats.npz")) as f:
    IS = f["IS"]
    IS_std = f["IS_std"]
    print(IS, IS_std)
#%%
imageset_str = 'FC6_std4'
with np.load(join(savedir, f"{imageset_str}_inception_stats.npz")) as f:
    mu1 = f["mu"]
    sigma1 = f["sigma"]

imageset_str = 'INet'
with np.load(join(savedir, f"{imageset_str}_inception_stats.npz")) as f:
    mu2 = f["mu"]
    sigma2 = f["sigma"]

fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6)
# FC6 vs INet : 197.0278012218725
# Inception Score :  3.3964710235595703 +- 0.03907453641295433
#%% Load in BigGAN
# biggan = loadBigGAN("biggan-deep-256")
from pytorch_pretrained_biggan import BigGAN
biggan = BigGAN.from_pretrained("biggan-deep-256")
biggan.eval().requires_grad_(False).cuda()
BG = BigGAN_wrapper(biggan)
#%% BigGAN
imageset_str = "BigGAN_norm_std008"
BG_rn_fun = lambda batch_size: \
        BG.visualize(0.08 * torch.randn(batch_size, 256, device="cuda"))
BG_loader = GANDataloader(BG_rn_fun, batch_size=20, total_imgnum=50000)
with torch.no_grad():
    acts, probs = get_inception_feature(BG_loader, dims=[2048, 1008], use_torch=True, verbose=True)
mu = torch.mean(acts, dim=0).cpu().numpy()
sigma = torch_cov(acts, rowvar=False).cpu().numpy()
np.savez_compressed(join(savedir, f"{imageset_str}_inception_stats.npz"), mu=mu, sigma=sigma)
inception_score, IS_std = calculate_inception_score(probs, 10, use_torch=True)
np.savez(join(savedir, f"{imageset_str}_IS_stats.npz"), IS=inception_score, IS_std=IS_std)
torch.save({"acts": acts, "probs":probs}, join(savedir, f"{imageset_str}_act_prob.pt"))
#%%
fid_BG_rnd = calculate_frechet_distance(mu, sigma, mu2, sigma2, eps=1e-6)
print(fid_BG_rnd)
# FID  BigGAN vs INet:  44.4272151684894
# Inception Score : 25.177 +- 0.5187
#%%
savedir = "/home/binxuwang/DL_Projects/GAN-fids"
imageset_str = "BigGAN_1000cls_std07"
BG_cls_fun = lambda batch_size: BG.visualize(BG.sample_vector(batch_size, class_id=None))
BG_cls_loader = GANDataloader(BG_cls_fun, batch_size=80, total_imgnum=50000)
with torch.no_grad():
    acts, probs = get_inception_feature(BG_cls_loader, dims=[2048, 1008], use_torch=True, verbose=True)
mu = torch.mean(acts, dim=0).cpu().numpy()
sigma = torch_cov(acts, rowvar=False).cpu().numpy()
np.savez_compressed(join(savedir, f"{imageset_str}_inception_stats.npz"), mu=mu, sigma=sigma)
inception_score, IS_std = calculate_inception_score(probs, 10, use_torch=True)
np.savez(join(savedir, f"{imageset_str}_IS_stats.npz"), IS=inception_score, IS_std=IS_std)
torch.save({"acts": acts, "probs":probs}, join(savedir, f"{imageset_str}_act_prob.pt"))
print(inception_score, IS_std)
#%%
with np.load(join(savedir, f"{'INet'}_inception_stats.npz")) as f:
    mu2 = f["mu"]
    sigma2 = f["sigma"]
fid_BG_cls = calculate_frechet_distance(mu, sigma, mu2, sigma2, eps=1e-6)
print(fid_BG_cls)
# BigGAN class vs INet: 9.333615632573526
# Inception Score: 224.3373260498047 3.763425350189209
#%%
imageset_str = "BigGAN_1000cls_std10"
BG_cls2_fun = lambda batch_size: BG.visualize(torch.cat(
    (torch.randn(128, batch_size, device="cuda"),
    BG.BigGAN.embeddings.weight[:, torch.randint(1000, size=(batch_size,), device="cuda")],)).T
)
BG_cls2_loader = GANDataloader(BG_cls2_fun, batch_size=80, total_imgnum=50000)
with torch.no_grad():
    acts, probs = get_inception_feature(BG_cls2_loader, dims=[2048, 1008], use_torch=True, verbose=True)
mu = torch.mean(acts, dim=0).cpu().numpy()
sigma = torch_cov(acts, rowvar=False).cpu().numpy()
np.savez_compressed(join(savedir, f"{imageset_str}_inception_stats.npz"), mu=mu, sigma=sigma)
inception_score, IS_std = calculate_inception_score(probs, 10, use_torch=True)
np.savez(join(savedir, f"{imageset_str}_IS_stats.npz"), IS=inception_score, IS_std=IS_std)
torch.save({"acts": acts, "probs":probs}, join(savedir, f"{imageset_str}_act_prob.pt"))
print(inception_score, IS_std)
#%%