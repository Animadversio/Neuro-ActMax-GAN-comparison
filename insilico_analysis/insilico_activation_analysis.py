import shutil
import os
import re
import glob
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from os.path import join
from easydict import EasyDict as edict
import numpy as np
import pandas as pd
from core.utils.plot_utils import saveallforms

rootdir = r"F:\insilico_exps\GAN_Evol_cmp"
os.makedirs(join(rootdir, "summary"), exist_ok=True)
figdir = join(rootdir, "summary")
netname_prefix = "effnet_"
df_evol = pd.read_csv(join(rootdir, "summary", f"{netname_prefix}raw_summary.csv"))
#%%
maxactdf_evol = df_evol.groupby(["netname", "layer", "RFresize", "unitid",]).agg({"score": "max"})
df_evol_norm = df_evol.merge(maxactdf_evol,
                            on=["netname", "layer", "RFresize", "unitid"], suffixes=("", "_max"))
df_evol_norm["score_norm"] = df_evol_norm["score"] / df_evol_norm["score_max"]
df_evol_norm["maxscore_norm"] = df_evol_norm["maxscore"] / df_evol_norm["score_max"]
df_evol_norm.to_csv(join(rootdir, "summary", f"{netname_prefix}raw_summary_norm.csv"), index=False)
#%%
df_evol_norm["layershort"] = df_evol_norm["layer"].apply(lambda x: x.replace(".SelectAdaptivePool2dglobal_pool", "cpool")\
                                          .replace('.Linearclassifier', "fc").replace(".blocks", "block"))
for netname, netdf in df_evol_norm.groupby(["netname"]):
    netname_short = netname.replace("tf_efficientnet_b6", "effnet")
    # rename entries in columns layer
    meansumdf_norm = netdf.groupby(["layershort", "RFresize", "optimmethod", "GANname"]) \
        .agg({"score_norm": "mean", "maxscore_norm": "mean",
              "maxstep": "mean"})  # .to_csv(join(rootdir, "summary_mean.csv"))
    meansum_norm_pivot = meansumdf_norm.pivot_table(index=["layershort", ],
                                                    columns=["GANname", "optimmethod", ], values="score_norm",
                                                    sort=False)
    meansum_norm_pivot_T = meansumdf_norm.pivot_table(index=["optimmethod", "GANname"],
                                                      columns=["layershort", "RFresize"], values="score_norm", sort=False)
    # %%
    meansum_norm_pivot.plot.bar(figsize=(8, 5))
    # xticks rotate
    plt.ylabel("score normalized by unit max")
    plt.xticks(rotation=40)
    plt.tight_layout()
    plt.suptitle(netname_short)
    saveallforms(figdir, f"{netname_short}_score_norm_by_layer_net_RF_bar")
    plt.show()
    # %%
    meansum_norm_pivot_T.plot.bar(figsize=(8, 5))
    # xticks rotate
    plt.ylabel("score normalized by unit max")
    plt.xticks(rotation=40)
    plt.tight_layout()
    plt.suptitle(netname_short)
    saveallforms(figdir, f"{netname_short}_score_norm_by_optim_GAN_bar")
    plt.show()
#%%

mask = ~df_evol_norm.optimmethod.isin(["CholCMA_class",])
meansumdf_norm = df_evol_norm[mask].groupby(["netname", "layer", "RFresize", "optimmethod", "GANname"])\
    .agg({"score_norm":"mean","maxscore_norm":"mean","maxstep":"mean"})#.to_csv(join(rootdir, "summary_mean.csv"))
meansum_norm_pivot = meansumdf_norm.pivot_table(index=["netname", "layer", ],
                        columns=["GANname", "optimmethod",], values="score_norm", sort=False)
meansum_norm_pivot_T = meansumdf_norm.pivot_table(index=["optimmethod", "GANname"],
                        columns=["layer", "RFresize"], values="score_norm", sort=False)
#%%
meansum_norm_pivot.plot.bar(figsize=(8, 5))
# xticks rotate
plt.ylabel("score normalized by unit max")
plt.xticks(rotation=40)
plt.tight_layout()
saveallforms(figdir, f"{netname_prefix}score_norm_by_layer_net_RF_bar")
plt.show()
#%%
meansum_norm_pivot_T.plot.bar(figsize=(8, 5))
# xticks rotate
plt.ylabel("score normalized by unit max")
plt.xticks(rotation=40)
plt.tight_layout()
saveallforms(figdir, f"{netname_prefix}score_norm_by_optim_GAN_bar")
plt.show()

#%% Resnet50_linf8
rootdir = r"F:\insilico_exps\GAN_Evol_cmp"
os.makedirs(join(rootdir, "summary"), exist_ok=True)
figdir = join(rootdir, "summary")
netname_prefix = "resnet50_linf8_"
df_evol = pd.read_csv(join(rootdir, "summary", f"resnet50_linf8_raw_summary.csv"))
#%%
maxactdf_evol = df_evol.groupby(["netname", "layer", "RFresize", "unitid",]).agg({"score": "max"})
df_evol_norm = df_evol.merge(maxactdf_evol,
                            on=["netname", "layer", "RFresize", "unitid"], suffixes=("", "_max"))
df_evol_norm["score_norm"] = df_evol_norm["score"] / df_evol_norm["score_max"]
df_evol_norm["maxscore_norm"] = df_evol_norm["maxscore"] / df_evol_norm["score_max"]
df_evol_norm.to_csv(join(rootdir, "summary", f"{netname_prefix}raw_summary_norm.csv"), index=False)
#%%
df_evol_norm["layershort"] = df_evol_norm["layer"].apply(lambda x: x.replace(".Linearfc", "fc")\
                                          .replace('.Bottleneck', "B").replace(".layer", "block"))
for netname, netdf in df_evol_norm.groupby(["netname"]):
    netname_short = netname
    # rename entries in columns layer
    meansumdf_norm = netdf.groupby(["layershort", "RFresize", "optimmethod", "GANname"]) \
        .agg({"score_norm": "mean", "maxscore_norm": "mean",
              "maxstep": "mean"})  # .to_csv(join(rootdir, "summary_mean.csv"))
    meansum_norm_pivot = meansumdf_norm.pivot_table(index=["layershort", "RFresize"],
                                                    columns=["GANname", "optimmethod", ], values="score_norm",
                                                    sort=False)
    meansum_norm_pivot_T = meansumdf_norm.pivot_table(index=["optimmethod", "GANname"],
                                                      columns=["layershort", "RFresize"], values="score_norm", sort=False)
    # %%
    meansum_norm_pivot.plot.bar(figsize=(8, 5))
    # xticks rotate
    plt.ylabel("score normalized by unit max")
    plt.xticks(rotation=40)
    plt.tight_layout()
    plt.suptitle(netname_short)
    saveallforms(figdir, f"{netname_short}_score_norm_by_layer_net_RF_bar")
    plt.show()
    # %%
    meansum_norm_pivot_T.plot.bar(figsize=(8, 5))
    # xticks rotate
    plt.ylabel("score normalized by unit max")
    plt.xticks(rotation=40)
    plt.tight_layout()
    plt.suptitle(netname_short)
    saveallforms(figdir, f"{netname_short}_score_norm_by_optim_GAN_bar")
    plt.show()

#%% Resnet50
rootdir = r"F:\insilico_exps\GAN_Evol_cmp"
os.makedirs(join(rootdir, "summary"), exist_ok=True)
figdir = join(rootdir, "summary")
netname_prefix = "resnet50_"
df_evol = pd.read_csv(join(rootdir, "summary", f"raw_summary.csv"))
#%%
maxactdf_evol = df_evol.groupby(["netname", "layer", "RFresize", "unitid",]).agg({"score": "max"})
df_evol_norm = df_evol.merge(maxactdf_evol,
                            on=["netname", "layer", "RFresize", "unitid"], suffixes=("", "_max"))
df_evol_norm["score_norm"] = df_evol_norm["score"] / df_evol_norm["score_max"]
df_evol_norm["maxscore_norm"] = df_evol_norm["maxscore"] / df_evol_norm["score_max"]
df_evol_norm.to_csv(join(rootdir, "summary", f"{netname_prefix}raw_summary_norm.csv"), index=False)
#%%
df_evol_norm["layershort"] = df_evol_norm["layer"].apply(lambda x: x.replace(".Linearfc", "fc")\
                                          .replace('.Bottleneck', "B").replace(".layer", "block"))
# set dtypes for columns ["layershort", "RFresize", "optimmethod", "GANname"] as str
for col in ["layershort", "optimmethod", "GANname"]:
    df_evol_norm[col] = df_evol_norm[col].astype(str)
df_evol_norm["RFresize"] = df_evol_norm["RFresize"].astype(bool)
#%%
import seaborn as sns

for netname, netdf in df_evol_norm.groupby(["netname"]):
    netname_short = netname
    netdf.layershort.unique()
    msk = netdf["layershort"].isin(['block1B1', 'block2B3', 'block3B5', 'block4B2', 'fc'])
    msk =  msk & ~ df_evol_norm.optimmethod.isin(['CholCMA_class'])
    # rename entries in columns layer
    # summarize by mean
    meansumdf_norm = netdf[msk].groupby(["layershort", "RFresize", "optimmethod", "GANname"]) \
        .agg({"score_norm": "mean", "maxscore_norm": "mean", "maxstep": "mean"})
    # .to_csv(join(rootdir, "summary_mean.csv"))
    # transform all indices as columns
    meansum_norm_pivot = meansumdf_norm.pivot_table(index=["layershort", "RFresize"],
                    columns=["GANname", "optimmethod", ], values="score_norm",
                     sort=False)
    meansum_norm_pivot_T = meansumdf_norm.pivot_table(index=["optimmethod", "GANname"],
                                                      columns=["layershort", "RFresize"], values="score_norm", sort=False)
    # %%
    meansum_norm_pivot.plot.bar(figsize=(8, 5))
    # xticks rotate
    plt.ylabel("score normalized by unit max")
    plt.xticks(rotation=40)
    plt.tight_layout()
    plt.suptitle(netname_short)
    saveallforms(figdir, f"{netname_short}_score_norm_by_layer_net_RF_bar")
    plt.show()
    # %%
    # TODO: violinplot properly
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=netdf[msk], y="score_norm",
                   x=netdf[msk][["layershort", "RFresize"]].apply(tuple, axis=1),
                   hue=netdf[msk][["GANname", "optimmethod"]].apply(tuple, axis=1),
                   bw=0.3, cut=0, scale="width", scale_hue=False, width=0.7, inner="quartile", linewidth=0.5,
                   hue_order=[('BigGAN', 'CholCMA'), ('BigGAN', 'HessCMA'),
                             ('fc6', 'CholCMA_fc6'), ('fc6', 'HessCMA500_fc6')]) #palette="Set2",
    # meansum_norm_pivot.plot.violin(figsize=(8, 5))
    # xticks rotate
    plt.ylabel("score normalized by unit max")
    plt.xticks(rotation=40)
    plt.tight_layout()
    plt.suptitle(netname_short)
    saveallforms(figdir, f"{netname_short}_score_norm_by_layer_net_RF_violin")
    plt.show()
    # %%
    meansum_norm_pivot_T.plot.bar(figsize=(8, 5))
    # xticks rotate
    plt.ylabel("score normalized by unit max")
    plt.xticks(rotation=40)
    plt.tight_layout()
    plt.suptitle(netname_short)
    saveallforms(figdir, f"{netname_short}_score_norm_by_optim_GAN_bar")
    plt.show()
