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
#%%
import os
from os.path import join
from typing import List, Union, Tuple, Optional
from glob import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, GaussianBlur
from torchvision.transforms.functional import to_tensor
class ImageDataset_filter(Dataset):
    """An simple image dataset for calculating inception score and FID."""

    def __init__(self, root, glob_pattern="*", exts=['png', 'jpg', 'JPEG'], transform=None,
                 num_images=None):
        """Construct an image dataset.

        Args:
            root: Path to the image directory. This directory will be
                  recursively searched.
            exts: List of extensions to search for.
            transform: A torchvision transform to apply to the images. If
                       None, the images will be converted to tensors.
            num_images: The number of images to load. If None, all images
                        will be loaded.
        """
        self.paths = []
        self.transform = transform
        for ext in exts:
            self.paths.extend(
                list(glob(
                    os.path.join(root, glob_pattern+'.%s' % ext), recursive=True)))
        self.paths = self.paths[:num_images]

    def __len__(self):              # noqa
        return len(self.paths)

    def __getitem__(self, idx):     # noqa
        image = Image.open(self.paths[idx])
        image = image.convert('RGB')        # fix ImageNet grayscale images
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = to_tensor(image)
        return image

#%%
sumdir = "/n/scratch3/users/b/biw905/GAN_sample_fid/summary"
os.makedirs(sumdir, exist_ok=True)
with np.load(join(sumdir, f"{'INet'}_inception_stats.npz")) as f:
    mu_INet = f["mu"]
    sigma_INet = f["sigma"]
#%%
imageset_str = "BigGAN_1000cls_std07"
imgroot = "/n/scratch3/users/b/biw905/GAN_sample_fid/BigGAN_1000cls_std07_invert"
FCimgdataset = ImageDataset_filter(imgroot, glob_pattern="FC_invert*", transform=None)
BGimgdataset = ImageDataset_filter(imgroot, glob_pattern="BG*", transform=None)
BGBlurimgdataset = ImageDataset_filter(imgroot, glob_pattern="BG*",
                           transform=Compose([ToTensor(), GaussianBlur([15, 15], sigma=9)]))
print(len(FCimgdataset), len(BGimgdataset))

imgloader = DataLoader(FCimgdataset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
with torch.no_grad():
    acts, probs = get_inception_feature(imgloader, dims=[2048, 1008], use_torch=True, verbose=True)
mu = torch.mean(acts, dim=0).cpu().numpy()
sigma = torch_cov(acts, rowvar=False).cpu().numpy()
np.savez_compressed(join(sumdir, f"{imageset_str}_FC_invert_inception_stats.npz"), mu=mu, sigma=sigma)
inception_score, IS_std = calculate_inception_score(probs, 10, use_torch=True)
fid_w_INet = calculate_frechet_distance(mu, sigma, mu_INet, sigma_INet, eps=1e-6)
np.savez(join(sumdir, f"{imageset_str}_FC_invert_IS_stats.npz"), IS=inception_score, IS_std=IS_std, FID=fid_w_INet)
print(imageset_str, "FC Inverted")
print("FID", fid_w_INet)
print("Inception Score", inception_score, IS_std)
#%%
imgloader = DataLoader(BGimgdataset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
with torch.no_grad():
    acts, probs = get_inception_feature(imgloader, dims=[2048, 1008], use_torch=True, verbose=True)
mu = torch.mean(acts, dim=0).cpu().numpy()
sigma = torch_cov(acts, rowvar=False).cpu().numpy()
np.savez_compressed(join(sumdir, f"{imageset_str}_BG_inception_stats.npz"), mu=mu, sigma=sigma)
inception_score, IS_std = calculate_inception_score(probs, 10, use_torch=True)
fid_w_INet = calculate_frechet_distance(mu, sigma, mu_INet, sigma_INet, eps=1e-6)
np.savez(join(sumdir, f"{imageset_str}_BG_IS_stats.npz"), IS=inception_score, IS_std=IS_std, FID=fid_w_INet)
print(imageset_str, "Original")
print("FID", fid_w_INet)
print("Inception Score", inception_score, IS_std)
#%%
imgloader = DataLoader(BGBlurimgdataset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
with torch.no_grad():
    acts, probs = get_inception_feature(imgloader, dims=[2048, 1008], use_torch=True, verbose=True)
mu = torch.mean(acts, dim=0).cpu().numpy()
sigma = torch_cov(acts, rowvar=False).cpu().numpy()
np.savez_compressed(join(sumdir, f"{imageset_str}_BGBlur_inception_stats.npz"), mu=mu, sigma=sigma)
inception_score, IS_std = calculate_inception_score(probs, 10, use_torch=True)
fid_w_INet = calculate_frechet_distance(mu, sigma, mu_INet, sigma_INet, eps=1e-6)
np.savez(join(sumdir, f"{imageset_str}_BGBlur_IS_stats.npz"), IS=inception_score, IS_std=IS_std, FID=fid_w_INet)
print(imageset_str, "Blur")
print("FID", fid_w_INet)
print("Inception Score", inception_score, IS_std)
#%%
imageset_str = "resnet50_linf8_gradevol"
imgroot = "/n/scratch3/users/b/biw905/GAN_sample_fid/resnet50_linf8_gradevol"
gradevolimgdataset = ImageDataset_filter(imgroot, glob_pattern="class*", transform=None)
print(len(gradevolimgdataset), )
imgloader = DataLoader(gradevolimgdataset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
with torch.no_grad():
    acts, probs = get_inception_feature(imgloader, dims=[2048, 1008], use_torch=True, verbose=True)
mu = torch.mean(acts, dim=0).cpu().numpy()
sigma = torch_cov(acts, rowvar=False).cpu().numpy()
np.savez_compressed(join(sumdir, f"{imageset_str}_FC_gradevol_inception_stats.npz"), mu=mu, sigma=sigma)
inception_score, IS_std = calculate_inception_score(probs, 10, use_torch=True)
fid_w_INet = calculate_frechet_distance(mu, sigma, mu_INet, sigma_INet, eps=1e-6)
np.savez(join(sumdir, f"{imageset_str}_FC_gradevol_IS_stats.npz"), IS=inception_score, IS_std=IS_std, FID=fid_w_INet)
print(imageset_str, "FC Evolved")
print("FID", fid_w_INet)
print("Inception Score", inception_score, IS_std)
#%%
imageset_str = "resnet50_linf8_gradevol_avgpool"
imgroot = "/n/scratch3/users/b/biw905/GAN_sample_fid/resnet50_linf8_gradevol_avgpool"
gradevolimgdataset = ImageDataset_filter(imgroot, glob_pattern="class*", transform=None)
print(len(gradevolimgdataset), )
imgloader = DataLoader(gradevolimgdataset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
with torch.no_grad():
    acts, probs = get_inception_feature(imgloader, dims=[2048, 1008], use_torch=True, verbose=True)
mu = torch.mean(acts, dim=0).cpu().numpy()
sigma = torch_cov(acts, rowvar=False).cpu().numpy()
np.savez_compressed(join(sumdir, f"{imageset_str}_FC_gradevol_inception_stats.npz"), mu=mu, sigma=sigma)
inception_score, IS_std = calculate_inception_score(probs, 10, use_torch=True)
fid_w_INet = calculate_frechet_distance(mu, sigma, mu_INet, sigma_INet, eps=1e-6)
np.savez(join(sumdir, f"{imageset_str}_FC_gradevol_IS_stats.npz"), IS=inception_score, IS_std=IS_std, FID=fid_w_INet)
print(imageset_str, "FC Evolved")
print("FID", fid_w_INet)
print("Inception Score", inception_score, IS_std)

#%%
imageset_str = "resnet50_linf8_gradevol_layer4"
imgroot = "/n/scratch3/users/b/biw905/GAN_sample_fid/resnet50_linf8_gradevol_layer4"
gradevolimgdataset = ImageDataset_filter(imgroot, glob_pattern="class*", transform=None)
print(len(gradevolimgdataset), )
imgloader = DataLoader(gradevolimgdataset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
with torch.no_grad():
    acts, probs = get_inception_feature(imgloader, dims=[2048, 1008], use_torch=True, verbose=True)
mu = torch.mean(acts, dim=0).cpu().numpy()
sigma = torch_cov(acts, rowvar=False).cpu().numpy()
np.savez_compressed(join(sumdir, f"{imageset_str}_FC_gradevol_inception_stats.npz"), mu=mu, sigma=sigma)
inception_score, IS_std = calculate_inception_score(probs, 10, use_torch=True)
fid_w_INet = calculate_frechet_distance(mu, sigma, mu_INet, sigma_INet, eps=1e-6)
np.savez(join(sumdir, f"{imageset_str}_FC_gradevol_IS_stats.npz"), IS=inception_score, IS_std=IS_std, FID=fid_w_INet)
print(imageset_str, "FC Evolved")
print("FID", fid_w_INet)
print("Inception Score", inception_score, IS_std)
#%%
imageset_str = "resnet50_linf8_gradevol_layer3"
imgroot = "/n/scratch3/users/b/biw905/GAN_sample_fid/resnet50_linf8_gradevol_layer3"
gradevolimgdataset = ImageDataset_filter(imgroot, glob_pattern="class*", transform=None)
print(len(gradevolimgdataset), )
imgloader = DataLoader(gradevolimgdataset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
with torch.no_grad():
    acts, probs = get_inception_feature(imgloader, dims=[2048, 1008], use_torch=True, verbose=True)
mu = torch.mean(acts, dim=0).cpu().numpy()
sigma = torch_cov(acts, rowvar=False).cpu().numpy()
np.savez_compressed(join(sumdir, f"{imageset_str}_FC_gradevol_inception_stats.npz"), mu=mu, sigma=sigma)
inception_score, IS_std = calculate_inception_score(probs, 10, use_torch=True)
fid_w_INet = calculate_frechet_distance(mu, sigma, mu_INet, sigma_INet, eps=1e-6)
np.savez(join(sumdir, f"{imageset_str}_FC_gradevol_IS_stats.npz"), IS=inception_score, IS_std=IS_std, FID=fid_w_INet)
print(imageset_str, "FC Evolved")
print("FID", fid_w_INet)
print("Inception Score", inception_score, IS_std)




#%%
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
