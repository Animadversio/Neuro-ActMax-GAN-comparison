import torch
from pathlib import Path
import pickle as pkl
import pandas as pd
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms
from core.utils.stats_utils import paired_strip_plot
# Model
# yolomodel = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
# plt.switch_backend('module://backend_interagg')
# saveroot = Path(r"/n/scratch3/users/b/biw905/GAN_sample_fid")
saveroot = Path(r"F:\insilico_exps\GAN_sample_fid")
sumdir = (saveroot / "yolo_summary")
sumdir.mkdir(exist_ok=True)
df_all = {}
for imgdir_name in [
    "DeePSim_4std",
    "BigGAN_trunc07",
    "BigGAN_std_008",
    "BigGAN_1000cls_std07",
    "BigGAN_1000cls_std07_FC_invert",
    "resnet50_linf8_gradevol",
    "resnet50_linf8_gradevol_avgpool",
    "resnet50_linf8_gradevol_layer4",
    "resnet50_linf8_gradevol_layer3",
]:
    df = pd.read_csv(sumdir / f"{imgdir_name}_yolo_stats.csv", index_col=0)
    df["imgdir_name"] = imgdir_name
    df_all[imgdir_name] = (df)
df_all = pd.concat(df_all.values())
#%%
tabdir = Path(r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables")
df_all.to_csv(sumdir / "all_yolo_stats.csv")
df_all.to_csv(tabdir / "all_yolo_stats.csv")
#%%
import seaborn as sns
figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\GAN_image_statistics\yolo_src"
# use frequency instead of count?
sns.displot(data=df_all.reset_index(), x="confidence", hue="imgdir_name",
            kind="kde", fill=True, )
saveallforms(figdir, "yolo_confidence_kdedist", plt.gcf(), )
plt.show()
#%%
sns.displot(data=df_all.reset_index(), x="confidence", hue="imgdir_name",
            kind="hist", stat="probability", fill=True, kde=True)
saveallforms(figdir, "yolo_confidence_histdist", plt.gcf(), )
plt.show()
#%%
"""hist plot how is it normalized?"""
sns.displot(data=df_all.reset_index(), x="confidence", hue="imgdir_name",
            kind="kde", fill=True, cut=0, )
saveallforms(figdir, "yolo_confidence_kdedist", plt.gcf(), )
plt.show()
#%%
"""hist plot how is it normalized?"""
sns.displot(data=df_all.reset_index(), x="confidence", hue="imgdir_name",
            kind="kde", fill=True, cut=0, hue_order=
            [ "DeePSim_4std",
            "BigGAN_trunc07",
            "BigGAN_std_008",])
saveallforms(figdir, "yolo_confidence_kdedist_GAN_cmp", plt.gcf(), )
plt.show()
#%%
"""hist plot how is it normalized?"""
sns.displot(data=df_all.reset_index(), x="confidence", hue="imgdir_name",
            kind="kde", fill=True, cut=0, hue_order=
            ["BigGAN_1000cls_std07",
    "BigGAN_1000cls_std07_FC_invert",])
#"DeePSim_4std",
saveallforms(figdir, "yolo_confidence_kdedist_BG_invert_cmp", plt.gcf(), )
plt.show()
#%%
"""hist plot how is it normalized?"""
sns.displot(data=df_all.reset_index(), x="confidence", hue="imgdir_name",
            kind="kde", fill=True, cut=0, hue_order=
            ["resnet50_linf8_gradevol",
    "resnet50_linf8_gradevol_avgpool",
    "resnet50_linf8_gradevol_layer4",
    "resnet50_linf8_gradevol_layer3",])
saveallforms(figdir, "yolo_confidence_kdedist_resnet_gradevol", plt.gcf(), )
plt.show()
#%%
sns.displot(data=df_all[df_all.imgdir_name=="BigGAN_std_008"], x="confidence", hue="imgdir_name",
            kind="hist", stat="probability", fill=True, )
plt.show()
#%%
sns.histplot(data=df_all[df_all.imgdir_name=="BigGAN_std_008"], x="confidence", hue="imgdir_name",
             stat="probability", fill=True, )
plt.show()
#%%
# bar plot of fraction of non na confidence
df_all["non_na"] = df_all["confidence"].isna()
df_all["non_na"].groupby(df_all["imgdir_name"]).mean().plot.bar()
plt.show()

#%%
