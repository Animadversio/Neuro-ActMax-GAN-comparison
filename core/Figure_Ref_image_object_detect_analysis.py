import torch
from pathlib import Path
import pickle as pkl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from core.utils.plot_utils import saveallforms
from core.utils.stats_utils import paired_strip_plot
# Model
# yolomodel = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
plt.switch_backend('module://backend_interagg')
# saveroot = Path(r"/n/scratch3/users/b/biw905/GAN_sample_fid")
#%% load stats for reference image dataset
saveroot = Path(r"F:\insilico_exps\GAN_sample_fid")
sumdir = (saveroot / "yolo_summary")
sumdir_obj = (saveroot / "yolo_objconf_summary")
df_all = {}
df_obj_all = {}
for imgdir_name in [
    "pink_noise",
    "imagenet_valid",
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
    df_all[imgdir_name] = df
    df_obj = pd.read_csv(sumdir_obj / f"{imgdir_name}_yolo_objconf_stats.csv", index_col=0)
    df_obj["imgdir_name"] = imgdir_name
    df_obj_all[imgdir_name] = df_obj

df_all = pd.concat(df_all.values())
df_obj_all = pd.concat(df_obj_all.values())
#%%
tabdir = Path(r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables")
df_all.to_csv(sumdir / "GAN_samples_all_yolo_stats.csv")
df_all.to_csv(tabdir / "GAN_samples_all_yolo_stats.csv")
df_obj_all.to_csv(sumdir_obj / "GAN_samples_all_yolo_objconf_stats.csv")
df_obj_all.to_csv(tabdir / "GAN_samples_all_yolo_objconf_stats.csv")


#%% Reload the table and analysis objectness
tabdir = Path(r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables")
df_all = pd.read_csv(tabdir / "GAN_samples_all_yolo_stats.csv", index_col=0)
#%%
df_all["confidence_fill0"] = df_all["confidence"].fillna(0)
df_obj_all["confidence_fill0"] = df_obj_all["confidence"].fillna(0)
df_obj_all["obj_confidence_fill0"] = df_obj_all["obj_confidence"].fillna(0)
df_obj_all["cls_confidence_fill0"] = df_obj_all["cls_confidence"].fillna(0)
#%%
# show more columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)
#%%
# count rate of non zero confidence
sumstatdf = pd.concat([df_all.groupby("imgdir_name").agg({"confidence_fill0": ["mean", "sem"], "confidence":["mean", "sem"]}),
           df_all.groupby("imgdir_name").agg(detect_rate=("n_objs", lambda x: 1-np.mean(x == 0)))],
        axis=1)
sumstatdf.to_csv(tabdir / "GAN_samples_all_yolo_stats_summary.csv")
#%%
df_all.groupby("imgdir_name").agg(detect_rate=("n_objs", lambda x: 1-np.mean(x == 0)))
#%%
sumstatdf = pd.concat([df_obj_all.groupby("imgdir_name").agg({"confidence": ["mean", "sem"],
                                                   "confidence_fill0": ["mean", "sem"],
                                                   "obj_confidence": ["mean", "sem"],
                                                   "obj_confidence_fill0": ["mean", "sem"],
                                                   "cls_confidence": ["mean", "sem"],
                                                   "cls_confidence_fill0": ["mean", "sem"],
                                                   }),
              df_obj_all.groupby("imgdir_name").agg(detect_rate=("n_objs", lambda x: 1-np.mean(x == 0)))],
                axis=1)
sumstatdf.to_csv(tabdir / "GAN_samples_all_yolo_objconf_stats_summary.csv")
sumstatdf


#%%
outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Figure_Evol_objectness\GAN_ref_src"
#%%
plt.figure(figsize=[5, 6])
sns.barplot(data=sumstatdf.reset_index(), x="imgdir_name", y="detect_rate",
            order=["imagenet_valid", "BigGAN_std_008", "DeePSim_4std", "pink_noise"],)
plt.gca().set_xticklabels(["ImageNet", "BigGAN", "DeePSim", "Pink noise"])
plt.xlabel("Image Space Name", fontsize=14)
plt.ylabel("Object Detection Rate", fontsize=14)
plt.suptitle("Object Detection Rate of YOLOv5 on ImageNet and GAN Samples")
plt.xticks(rotation=0)
saveallforms(outdir, "yolo_detect_rate_select", plt.gcf(), )
plt.show()

#%%
plt.figure(figsize=[4.5, 6])
sns.violinplot(data=df_all.reset_index(), x="imgdir_name", y="confidence",
            order=["imagenet_valid", "BigGAN_std_008", "DeePSim_4std", ], cut=0, )  # "pink_noise"
plt.gca().set_xticklabels(["ImageNet", "BigGAN", "DeePSim", ])  # "Pink noise"
plt.xlabel("Image Space Name", fontsize=14)
plt.ylabel("Max Confidence", fontsize=14)
plt.suptitle("Objectness of ImageNet and GAN Samples")
plt.xticks(rotation=0)
saveallforms(outdir, "yolo_objectness_select", plt.gcf(), )
plt.show()
#%%
for valkey in ["confidence", "confidence_fill0",
            "obj_confidence", "obj_confidence_fill0",
            "cls_confidence", "cls_confidence_fill0"]:
    plt.figure(figsize=[4.5, 6])
    sns.violinplot(data=df_obj_all.reset_index(), x="imgdir_name", y=valkey,
                order=["imagenet_valid", "BigGAN_std_008", "DeePSim_4std", ], cut=0, )  # "pink_noise"
    plt.gca().set_xticklabels(["ImageNet", "BigGAN", "DeePSim", ])  # "Pink noise"
    plt.xlabel("Image Space Name", fontsize=14)
    plt.ylabel(f"Max {valkey}", fontsize=14)
    plt.suptitle("Objectness of ImageNet and GAN Samples")
    plt.xticks(rotation=0)
    saveallforms(outdir, f"yolo_{valkey}_select", plt.gcf(), )
    plt.show()
#%%
for valkey in ["confidence", "obj_confidence", "cls_confidence"]:
    plt.figure(figsize=[4.5, 6])
    sns.violinplot(data=df_obj_all.reset_index(), x="imgdir_name", y=valkey,
                   order=["imagenet_valid", "BigGAN_std_008", "DeePSim_4std"],
                   cut=0, )
    sns.pointplot(data=df_obj_all.reset_index(), x="imgdir_name", y=valkey+"_fill0",
                  order=["imagenet_valid", "BigGAN_std_008", "DeePSim_4std"],
                  n_boot=0, errorbar="se", linestyles="none", color="red",
                  capsize=.4, errwidth=1)
    plt.gca().set_xticklabels(["ImageNet", "BigGAN (RND)", "DeePSim"])
    plt.xlabel("Image Space Name", fontsize=14)
    plt.ylabel(f"Max {valkey}", fontsize=14)
    plt.suptitle("Objectness of ImageNet and GAN Samples")
    plt.xticks(rotation=0)
    saveallforms(outdir, f"yolo_{valkey}_select_pnt", plt.gcf(), )
    plt.show()
#%%
plt.figure(figsize=[5, 4.5])
sns.kdeplot(data=df_all.reset_index(), x="confidence", hue="imgdir_name", cut=0,  fill=True,
            hue_order=["imagenet_valid", "BigGAN_std_008", "DeePSim_4std"],
            )
plt.suptitle("Objectness of ImageNet and GAN Samples")
plt.gca().get_legend().set_title("Image Space Name")
sns.move_legend(plt.gca(), "upper left",)
# change the texts
for text_obj, legend_str in zip(plt.gca().get_legend().texts,
                                ["ImageNet", "BigGAN (RND)", "DeePSim"]):
    text_obj.set_text(legend_str)
saveallforms(outdir, "yolo_objectness_select_hist", plt.gcf(), )
plt.show()

#%%
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
            ["imagenet_valid",
            "BigGAN_trunc07",
            "BigGAN_std_008",
            "DeePSim_4std",])
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
"""hist plot of fraction of non na confidence"""
df_all["NA_detect"] = df_all["confidence"].isna()
#%%
plt.figure(figsize=(5, 6))
sns.barplot(data=df_all.reset_index(), x="imgdir_name", y="NA_detect",
            order=["imagenet_valid", "BigGAN_trunc07", "BigGAN_std_008", "DeePSim_4std"])
plt.gca().set_xticklabels(["ImageNet-valid", "BigGAN", "BigGAN (rnd)", "DeePSim"])
plt.xticks(rotation=0)
plt.xlabel("image space")
plt.ylabel("fraction of non-detection")
plt.suptitle("BigGAN vs DeePSim")
plt.tight_layout()
saveallforms(figdir, "yolo_detect_rate_GAN_cmp", plt.gcf(), )
plt.show()
#%%
plt.figure(figsize=(5, 6))
sns.barplot(data=df_all.reset_index(), x="imgdir_name", y="NA_detect",
            order=["BigGAN_1000cls_std07", "BigGAN_1000cls_std07_FC_invert"])
plt.gca().set_xticklabels(["BigGAN", "inverted BigGAN\nin DeePSim"])
# plt.xticks(rotation=45)
plt.xlabel("image space")
plt.ylabel("fraction of non-detection")
plt.suptitle("BigGAN vs inverted into DeePSim")
plt.tight_layout()
saveallforms(figdir, "yolo_detect_rate_BG_invert_cmp", plt.gcf(), )
plt.show()
#%%
plt.figure(figsize=(5, 6))
sns.barplot(data=df_all.reset_index(), x="imgdir_name", y="NA_detect",
            order=["resnet50_linf8_gradevol",
                   "resnet50_linf8_gradevol_avgpool",
                   "resnet50_linf8_gradevol_layer4",
                   "resnet50_linf8_gradevol_layer3",])
plt.gca().set_xticklabels(["fc", "avgpool", "layer4", "layer3"])
plt.xticks(rotation=0)
plt.suptitle("Gradient Evolution of DeePSim by ResNet50_linf8")
plt.xlabel("image space")
plt.ylabel("fraction of non-detection")
plt.tight_layout()
saveallforms(figdir, "yolo_detect_rate_resnet_gradevol", plt.gcf(), )
plt.show()

#%%
sns.displot(data=df_all[df_all.imgdir_name == "BigGAN_std_008"], x="confidence", hue="imgdir_name",
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
