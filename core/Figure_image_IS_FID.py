#%% Summary stats for all GANs
import sys
import os
from os.path import join
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pytorch_gan_metrics.utils import ImageDataset
from pytorch_gan_metrics.core  import calculate_frechet_distance, torch_cov, get_inception_feature, calculate_inception_score
from core.utils.plot_utils import saveallforms
if sys.platform == "linux" and os.getlogin() == 'binxuwang':
    savedir = "/home/binxuwang/DL_Projects/GAN-fids"
else:
    savedir = r"E:\OneDrive - Harvard University\GAN_imgstats_cmp\Inception"
figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\GAN_image_statistics\src"
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
#%%
statssavedir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\GAN_image_statistics\src"
df = pd.read_csv(join(statssavedir, "GAN_FID_IS.csv"), index_col=0)
for imgset_name in [
    "BigGAN_1000cls_std07_BG",
    "BigGAN_1000cls_std07_FC_invert",
    "BigGAN_1000cls_std07_BGBlur",
    "resnet50_linf8_gradevol_FC_gradevol",
    "resnet50_linf8_gradevol_avgpool_FC_gradevol",
    "resnet50_linf8_gradevol_layer4_FC_gradevol",
    "resnet50_linf8_gradevol_layer3_FC_gradevol",
]:
    with np.load(join(statssavedir, f"{imgset_name}_inception_stats.npz")) as f:
        mu, sigma = f["mu"], f["sigma"]
    with np.load(join(statssavedir, f"{imgset_name}_IS_stats.npz")) as f:
        IS, IS_std, FID = f["IS"], f["IS_std"], f["FID"]
    df = df.append({"imgset": imgset_name, "FID": FID, "IS": IS, "IS_std": IS_std}, ignore_index=True)
df.to_csv(join(statssavedir, "GAN_FID_IS_full.csv"), )
# df.to_csv(join(savedir, "GAN_FID_IS.csv"))
#%%
# set the variable types to be float
df = df.astype({"imgset": str, "FID": float, "IS": float, "IS_std": float})
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
plot_rows = ["INet",
    "BigGAN_1000cls_std07",
    "resnet50_linf8_gradevol_FC_gradevol",
    "resnet50_linf8_gradevol_avgpool_FC_gradevol",
    "resnet50_linf8_gradevol_layer4_FC_gradevol",
    "resnet50_linf8_gradevol_layer3_FC_gradevol",]
row_labels = ["ImageNet", "BigGAN", "fc\nEvol", "avgpool\nEvol", "layer4\nEvol", "layer3\nEvol"]
df_val = df.copy()
df_val["FID"].iloc[0] = np.nan  # INet is not a GAN, so no FID
plt.figure(figsize=(5, 7))
sns.barplot(x="imgset", y="FID", order=plot_rows, data=df_val)
plt.ylabel("Frechet Inception Distance", fontsize=15)
plt.gca().set_xticklabels(row_labels, fontsize=13)#rotation=45)
plt.suptitle("Effect of ResNet50-robust Evolution on Image Statistics")
plt.tight_layout()
saveallforms([figdir, savedir], "GAN_FID_barplot_gradevol_cmp")
plt.show()

plt.figure(figsize=(5, 7))
sns.barplot(x="imgset", y="IS", order=plot_rows, data=df_val)
plt.errorbar(x=np.arange(len(plot_rows)), y=df_val.set_index("imgset").loc[plot_rows]['IS'],
            yerr=df_val.set_index("imgset").loc[plot_rows]['IS_std'], fmt='none', c= 'black', capsize = 2)
plt.ylabel("Inception Score", fontsize=15)
plt.gca().set_xticklabels(row_labels, fontsize=13)#rotation=45)
plt.suptitle("Effect of ResNet50-robust Evolution on Image Statistics")
plt.tight_layout()
saveallforms([figdir, savedir], "GAN_IS_barplot_gradevol_cmp")
plt.show()
#%%
plot_rows = ["INet",
    # "BigGAN_1000cls_std07",
    "BigGAN_1000cls_std07_BG",
    "BigGAN_1000cls_std07_BGBlur",
    "BigGAN_1000cls_std07_FC_invert",
             'FC6_std4',]
row_labels = ["ImageNet", "BigGAN", "BigGAN\nBlur", "DeePSim\nInvert", "DeePSim",]
df_val = df.copy()
df_val["FID"].iloc[0] = np.nan  # INet is not a GAN, so no FID
plt.figure(figsize=(5, 7))
sns.barplot(x="imgset", y="FID", order=plot_rows, data=df_val)
plt.ylabel("Frechet Inception Distance", fontsize=15)
plt.gca().set_xticklabels(row_labels, fontsize=13)#rotation=45)
plt.suptitle("Effect of DeePSim Inversion on Image Statistics")
plt.tight_layout()
saveallforms([figdir, savedir], "GAN_FID_barplot_FC_invert_cmp")
plt.show()

plt.figure(figsize=(5, 7))
sns.barplot(x="imgset", y="IS", order=plot_rows, data=df_val)
plt.errorbar(x=np.arange(len(plot_rows)), y=df_val.set_index("imgset").loc[plot_rows]['IS'],
            yerr=df_val.set_index("imgset").loc[plot_rows]['IS_std'], fmt='none', c= 'black', capsize = 2)
plt.ylabel("Inception Score", fontsize=15)
plt.gca().set_xticklabels(row_labels, fontsize=13)#rotation=45)
plt.suptitle("Effect of DeePSim Inversion on Image Statistics")
plt.tight_layout()
saveallforms([figdir, savedir], "GAN_IS_barplot_FC_invert_cmp")
plt.show()