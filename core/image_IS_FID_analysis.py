"""
Quantify naturalness of generated images. through FID and IS.
    Compare the BigGAN and FC6 GAN images in terms of FID score.
    and Fourier power spectrum.

Requirements
! pip install pytorch-gan-metrics
! pip install pytorch-pretrained-biggan
"""
import sys
import os
from os.path import join

import matplotlib.pyplot as plt
import torch
import numpy as np
from pytorch_gan_metrics import get_inception_score, get_fid
from pytorch_gan_metrics.utils import ImageDataset
from pytorch_gan_metrics.core  import torch_cov, get_inception_feature, calculate_inception_score, calculate_frechet_distance
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from core.utils import saveallforms, showimg, show_imgrid, save_imgrid
if sys.platform == "linux" and os.getlogin() == 'binxuwang':
    savedir = "/home/binxuwang/DL_Projects/GAN-fids"
else:
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
np.savez_compressed(rf"E:\Datasets\imagenet-valid\{'INet'}_inception_stats.npz", mu=mu, sigma=sigma)
inception_score, IS_std = calculate_inception_score(probs, 10, use_torch=True)
np.savez_compressed(rf"E:\Datasets\imagenet-valid\{'INet'}_IS_stats.npz", IS=inception_score, IS_std=IS_std)
# Inception score: 211.13711547851562 +- 3.3677103519439697
#%%
with np.load(join(savedir, f"{'INet'}_inception_stats.npz")) as f:
    mu_INet = f["mu"]
    sigma_INet = f["sigma"]
#%% FC6
from core.utils.GAN_utils import upconvGAN, BigGAN_wrapper, loadBigGAN
class GANDataloader(DataLoader):
    """A hacky way to create a dataloader from a GAN model.
        Create images on the fly and stream them to the reader.
        Saving the images to disk will increase IO time.
        Binxu

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
#%% Testing ground
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
fid_w_INet = calculate_frechet_distance(mu, sigma, mu_INet, sigma_INet, eps=1e-6)
print(imageset_str)
print("FID",fid_w_INet)
print("Inception Score",inception_score, IS_std)
# FC6 vs INet : 197.0278012218725
# Inception Score :  3.3964710235595703 +- 0.03907453641295433
#%% Load in BigGAN
# biggan = loadBigGAN("biggan-deep-256")
from pytorch_pretrained_biggan import BigGAN
biggan = BigGAN.from_pretrained("biggan-deep-256")
biggan.eval().requires_grad_(False).cuda()
BG = BigGAN_wrapper(biggan)

#%% BigGAN
"""Sample class vector and noise vector from isotropic gaussian """
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
fid_BG_rnd = calculate_frechet_distance(mu, sigma, mu_INet, sigma_INet, eps=1e-6)
print(fid_BG_rnd)
# FID  BigGAN_random samples vs INet:  44.4272151684894
# Inception Score : 25.177 +- 0.5187

#%%
savedir = "/home/binxuwang/DL_Projects/GAN-fids"
imageset_str = "BigGAN_norm_std07"
BG_rn_fun = lambda batch_size: \
        BG.visualize(0.7 * torch.randn(batch_size, 256, device="cuda"))

BG_loader = GANDataloader(BG_rn_fun, batch_size=80, total_imgnum=50000)
with torch.no_grad():
    acts, probs = get_inception_feature(BG_loader, dims=[2048, 1008], use_torch=True, verbose=True)
mu = torch.mean(acts, dim=0).cpu().numpy()
sigma = torch_cov(acts, rowvar=False).cpu().numpy()
np.savez_compressed(join(savedir, f"{imageset_str}_inception_stats.npz"), mu=mu, sigma=sigma)
inception_score, IS_std = calculate_inception_score(probs, 10, use_torch=True)
np.savez(join(savedir, f"{imageset_str}_IS_stats.npz"), IS=inception_score, IS_std=IS_std)
torch.save({"acts": acts, "probs":probs}, join(savedir, f"{imageset_str}_act_prob.pt"))
fid_w_INet = calculate_frechet_distance(mu, sigma, mu_INet, sigma_INet, eps=1e-6)
print(imageset_str)
print("FID",fid_w_INet)
print("Inception Score",inception_score, IS_std)
# FID 88.07192380266332
# Inception Score 13.241793632507324 0.22063525021076202
#%%
imageset_str = "BigGAN_norm_std12"
BG_rn_fun = lambda batch_size: \
        BG.visualize(1.2 * torch.randn(batch_size, 256, device="cuda"))
BG_loader = GANDataloader(BG_rn_fun, batch_size=80, total_imgnum=50000)
with torch.no_grad():
    acts, probs = get_inception_feature(BG_loader, dims=[2048, 1008], use_torch=True, verbose=True)
mu = torch.mean(acts, dim=0).cpu().numpy()
sigma = torch_cov(acts, rowvar=False).cpu().numpy()
np.savez_compressed(join(savedir, f"{imageset_str}_inception_stats.npz"), mu=mu, sigma=sigma)
inception_score, IS_std = calculate_inception_score(probs, 10, use_torch=True)
np.savez(join(savedir, f"{imageset_str}_IS_stats.npz"), IS=inception_score, IS_std=IS_std)
torch.save({"acts": acts, "probs":probs}, join(savedir, f"{imageset_str}_act_prob.pt"))
fid_w_INet = calculate_frechet_distance(mu, sigma, mu_INet, sigma_INet, eps=1e-6)
print(imageset_str)
print("FID",fid_w_INet)
print("Inception Score",inception_score, IS_std)
# FID 88.07192380266332
# Inception Score 13.241793632507324 0.22063525021076202
#%%
"""Sample class vector from 1000 clasees, and noise vector with std = 0.7. better quality than 1.0"""
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
fid_BG_cls = calculate_frechet_distance(mu, sigma, mu_INet, sigma_INet, eps=1e-6)
print(fid_BG_cls)
# BigGAN class vs INet: 9.333615632573526
# Inception Score: 224.3373260498047 3.763425350189209
#%%
"""Sample class vector from 1000 clasees, and noise vector with std = 1.0. worse than 0.7 """
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
imageset_str = "white_noise"
wnoise_fun = lambda batch_size: torch.rand(batch_size, 3, 256, 256)
wnoise_loader = GANDataloader(wnoise_fun, batch_size=80, total_imgnum=50000)
with torch.no_grad():
    acts, probs = get_inception_feature(wnoise_loader, dims=[2048, 1008], use_torch=True, verbose=True)
mu = torch.mean(acts, dim=0).cpu().numpy()
sigma = torch_cov(acts, rowvar=False).cpu().numpy()
np.savez_compressed(join(savedir, f"{imageset_str}_inception_stats.npz"), mu=mu, sigma=sigma)
inception_score, IS_std = calculate_inception_score(probs, 10, use_torch=True)
np.savez(join(savedir, f"{imageset_str}_IS_stats.npz"), IS=inception_score, IS_std=IS_std)
torch.save({"acts": acts, "probs":probs}, join(savedir, f"{imageset_str}_act_prob.pt"))
print("IS+-std ", inception_score, IS_std)
fid = calculate_frechet_distance(mu, sigma, mu_INet, sigma_INet)
print("Fid", fid)
#%%
amp = torch.fft.fft2(torch.rand(3, 256, 256))
freq1d = torch.fft.fftfreq(256)
freq2d = torch.sqrt(freq1d[:, None]**2 + freq1d[None, :]**2)
freq2d[0, 0] = 1
amp = amp / freq2d
amp[0, 0] = 0
pinknoise = torch.fft.ifft2(amp).real #()
pinknoise = (pinknoise - pinknoise.mean(dim=(-3, -2, -1))) / pinknoise.std(dim=(-3, -2, -1), keepdim=True) * 0.2 + 0.5
pinknoise = pinknoise.clamp(0, 1)
plt.imshow(pinknoise.permute(1, 2, 0))
plt.show()
#%%


show_imgrid(pink_noise(16), nrow=4, padding=2)
#%%
plt.figure()
plt.imshow(pinknoise)
plt.show()
#%%
imageset_str = "pink_noise"
def pink_noise(batch_size, generator=None):
    amp = torch.fft.fft2(torch.rand(batch_size, 3, 256, 256, generator=generator))
    freq1d = torch.fft.fftfreq(256)
    freq2d = torch.sqrt(freq1d[:, None] ** 2 + freq1d[None, :] ** 2)
    freq2d[0, 0] = 1
    amp = amp / freq2d
    amp[0, 0] = 0
    pinknoise = torch.fft.ifft2(amp).real  # ()
    pinknoise = (pinknoise - pinknoise.mean(dim=(-3, -2, -1), keepdim=True)) / \
                pinknoise.std(dim=(-3, -2, -1), keepdim=True) * 0.2 + 0.5
    pinknoise = pinknoise.clamp(0, 1)
    return pinknoise


pnoise_loader = GANDataloader(pink_noise, batch_size=80, total_imgnum=50000)
with torch.no_grad():
    acts, probs = get_inception_feature(pnoise_loader, dims=[2048, 1008], use_torch=True, verbose=True)
mu = torch.mean(acts, dim=0).cpu().numpy()
sigma = torch_cov(acts, rowvar=False).cpu().numpy()
np.savez_compressed(join(savedir, f"{imageset_str}_inception_stats.npz"), mu=mu, sigma=sigma)
inception_score, IS_std = calculate_inception_score(probs, 10, use_torch=True)
np.savez(join(savedir, f"{imageset_str}_IS_stats.npz"), IS=inception_score, IS_std=IS_std)
torch.save({"acts": acts, "probs":probs}, join(savedir, f"{imageset_str}_act_prob.pt"))
print("IS+-std ", inception_score, IS_std)
fid = calculate_frechet_distance(mu, sigma, mu_INet, sigma_INet)
print("Fid", fid)

#%%
print(inception_score, IS_std)

#%% Summary stats for all GANs
import pandas as pd
import seaborn as sns
#%%
with np.load(join(savedir, f"{'INet'}_inception_stats.npz")) as f:
    mu_INet = f["mu"]
    sigma_INet = f["sigma"]

df = []
for imgset_lab in ["INet", 'FC6_std4',
                   "BigGAN_norm_std07", "BigGAN_norm_std008",
                   "BigGAN_1000cls_std07", "BigGAN_1000cls_std10"]:
    with np.load(join(savedir, f"{imgset_lab}_inception_stats.npz")) as f:
        mu, sigma = f["mu"], f["sigma"]
    fid_w_INet = calculate_frechet_distance(mu, sigma, mu_INet, sigma_INet, eps=1e-6)
    print(f"{imgset_lab} vs INet: {fid_w_INet}")
    with np.load(join(savedir, f"{imgset_lab}_IS_stats.npz")) as f:
        IS, IS_std = f["IS"], f["IS_std"]
    print(f"Inception Score {IS}+-{IS_std}")
    df.append({"imgset": imgset_lab, "FID": fid_w_INet, "IS": IS, "IS_std": IS_std})

df = pd.DataFrame(df)
df = df.astype({"imgset": str, "FID": float, "IS": float, "IS_std": float})
df.to_csv(join(savedir, "GAN_FID_IS.csv"))
#%%
df = pd.read_csv(join(savedir, "GAN_FID_IS.csv"))
#%%
#%%
df_val = df.copy()
df_val["FID"].iloc[0] = np.nan  # INet is not a GAN, so no FID
plt.figure(figsize=(5, 7))
sns.barplot(x="imgset", y="FID", data=df_val)
plt.ylabel("Frechet Inception Distance")
plt.xticks(rotation=45)
plt.tight_layout()
saveallforms(savedir, "GAN_FID_barplot")
plt.show()
#%%
plt.figure(figsize=(5, 7))
sns.barplot(x="imgset", y="IS", data=df_val)
plt.errorbar(x = np.arange(len(df)), y = df['IS'],
            yerr=df['IS_std'], fmt='none', c= 'black', capsize = 2)
plt.ylabel("Inception Score")
plt.xticks(rotation=45)
plt.tight_layout()
saveallforms(savedir, "GAN_IS_barplot")
plt.show()
#%%
plot_rows = ["INet", "BigGAN_1000cls_std07", "BigGAN_norm_std008", "FC6_std4", "pink_noise", "white_noise"]
df_val = df.copy()
df_val["FID"].iloc[0] = np.nan  # INet is not a GAN, so no FID
plt.figure(figsize=(5, 7))
sns.barplot(x="imgset", y="FID", order=plot_rows, data=df_val)
plt.ylabel("Frechet Inception Distance")
plt.xticks(rotation=45)
plt.tight_layout()
saveallforms(savedir, "GAN_FID_barplot_selective")
plt.show()
plt.figure(figsize=(5, 7))
sns.barplot(x="imgset", y="IS", order=plot_rows, data=df_val)
plt.errorbar(x=np.arange(len(plot_rows)), y=df_val.set_index("imgset").loc[plot_rows]['IS'],
            yerr=df_val.set_index("imgset").loc[plot_rows]['IS_std'], fmt='none', c= 'black', capsize = 2)
plt.ylabel("Inception Score")
plt.xticks(rotation=45)
plt.tight_layout()
saveallforms(savedir, "GAN_IS_barplot_selective")
plt.show()
#%%
# "INet": image net images
# "BigGAN_1000cls_std07": samples from BigGAN using the trained random vectors
# "BigGAN_norm_std008": samples from BigGAN using Gaussian random latent vectors. (like in evolution)
# "FC6_std4": DeepSim FC6.
# "pink_noise": 1/f noise  (matching natural image mean, std and spectrum)
# "white_noise": i.i.d. uniform noise
#%%
imgtsrs_tmp = BG.visualize(0.08*torch.randn(20, 256, device="cuda"))
mtg = show_imgrid(imgtsrs_tmp, nrow=5,)
save_imgrid(imgtsrs_tmp, join(savedir, "BigGAN_norm_std008_samples.jpg"), nrow=5)
plt.figure(figsize=(10, 8))
showimg(plt.gca(), mtg)
plt.tight_layout()
plt.show()
#%%
imgtsrs_tmp = BG.visualize(torch.cat(
    (torch.randn(128, 20, device="cuda"),
    BG.BigGAN.embeddings.weight[:, torch.randint(1000, size=(20,), device="cuda")],)).T
)
mtg = show_imgrid(imgtsrs_tmp, nrow=5,)
save_imgrid(imgtsrs_tmp, join(savedir, "BigGAN_1000cls_std10_samples.jpg"), nrow=5)
plt.figure(figsize=(10, 8))
showimg(plt.gca(), mtg)
plt.tight_layout()
plt.show()
#%%
imgtsrs_tmp = BG.visualize(BG.sample_vector(20, class_id=None))
mtg = show_imgrid(imgtsrs_tmp, nrow=5,)
save_imgrid(imgtsrs_tmp, join(savedir, "BigGAN_1000cls_std07_samples.jpg"), nrow=5)
plt.figure(figsize=(10, 8))
showimg(plt.gca(), mtg)
plt.tight_layout()
plt.show()
#%%
imgtsrs_tmp = FG.visualize(4 * torch.randn(20, 4096, device="cuda"))
mtg = show_imgrid(imgtsrs_tmp, nrow=5,)
save_imgrid(imgtsrs_tmp, join(savedir, "FC6_std4_samples.jpg"), nrow=5)
plt.figure(figsize=(10, 8))
showimg(plt.gca(), mtg)
plt.tight_layout()
plt.show()
#%%
imgtsrs_tmp = pink_noise(20,)
mtg = show_imgrid(imgtsrs_tmp, nrow=5,)
save_imgrid(imgtsrs_tmp, join(savedir, "pink_noise_samples.jpg"), nrow=5)
plt.figure(figsize=(10, 8))
showimg(plt.gca(), mtg)
plt.tight_layout()
plt.show()

#%% Generate samples
outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\GAN_image_statistics\samples"
mtg = save_imgrid(INdataset[42], join(outdir, "ImageNet_samples.jpg"), nrow=1)
#%%
img = FG.visualize(4 * torch.randn(1, 4096, device="cuda", generator=torch.cuda.manual_seed(42)))
mtg = save_imgrid(img, join(outdir, "DeePSim_std4_samples.jpg"), nrow=1)
#%%
#%%
# torch.random.manual_seed(100)
img = BG.visualize(BG.sample_vector(1, class_id=None))
mtg = save_imgrid(img, join(outdir, "BigGAN_1000cls_samples.jpg"), nrow=1)
#%%
img = BG.visualize(0.08 * torch.randn(1, 256, device="cuda", generator=torch.cuda.manual_seed(0)))
mtg = save_imgrid(img, join(outdir, "BigGAN_norm_std008_samples.jpg"), nrow=1)
#%%
mtg = save_imgrid(pink_noise(1, generator=torch.cuda.manual_seed(42)),
                  join(outdir, "pinknoise_samples.jpg"), nrow=1)
#%%
mtg = save_imgrid(torch.rand(3,256,256, generator=torch.cuda.manual_seed(42)),
                  join(outdir, "whitenoise_samples.jpg"), nrow=1)
