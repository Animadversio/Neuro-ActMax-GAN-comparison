#%%

import shutil
import os
import re
import glob

import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from os.path import join
from easydict import EasyDict as edict
from core.utils.montage_utils import crop_from_montage, crop_all_from_montage
from core.utils.montage_utils import make_grid, make_grid_np, make_grid_T
from collections import defaultdict
import pickle as pkl
from core.utils.plot_utils import saveallforms
sumdir = r"F:\insilico_exps\GAN_Evol_cmp\summary"

#%%
grad_df = pd.read_csv(Path(sumdir) / f"raw_summary_grad.csv", index_col=0)
evol_df = pd.read_csv(Path(sumdir) / f"raw_summary.csv", )
evol_df_reslinf8 = pd.read_csv(Path(r"E:\Cluster_Backup\GAN_Evol_cmp\summary") / f"raw_summary.csv", )
#%%
# replace "resnet50_linf_8" by "resnet50_linf8" in "netname" column
evol_df_reslinf8["netname"] = evol_df_reslinf8["netname"].str.replace("resnet50_linf_8", "resnet50_linf8")
#%%
cmb_df  = pd.concat((evol_df, evol_df_reslinf8, grad_df), axis=0)  # pd.merge(grad_df, evol_df, on=["netname", "layer", "unitid", "x", "y", "RFresize"])
cmb_df.to_csv(Path(sumdir) / f"raw_summary_cmb_grad-evol.csv")
#%%
"""Average scores for each unit x method x GAN
Then pivot the table to get a matrix of scores for each unit x method x GAN"""
meansummary = cmb_df.groupby(by=["netname", "layer", "unitid", "RFresize", "optimmethod", "GANname"], dropna=False).mean()
mean_pivot = meansummary.pivot_table(index=["netname", "layer", "unitid", "RFresize"], columns=["GANname", "optimmethod", ], values="score")
#%%
mean_pivot.to_csv(Path(sumdir) / f"optimizer_unit_summary.csv")
meansummary.to_csv(Path(sumdir) / f"optimizer_unit_summary_raw.csv")


#%%
"""Find the units with paired gradient & gradeint free evolutions """
targetoptims = ["Adam001", "Adam001Hess", "CholCMA", "HessCMA",
                "Adam01",  "Adam01Hess", "HessCMA500_fc6", ]
targetoptims = [("BigGAN", "Adam001"), ("BigGAN", "Adam001Hess"), ("BigGAN", "CholCMA"), ("BigGAN", "HessCMA"),
                ("fc6", "Adam01"),  ("fc6", "Adam01Hess"), ("fc6", "HessCMA500_fc6"), ]
# find rows in which all targetoptims are has non nan value in mean_pivot
targetrows = mean_pivot.loc[:, targetoptims].notna().all(axis=1)
# filter the table with these matched methods
mean_pivot_matched = mean_pivot[targetrows].loc[:, targetoptims]
#%%
mean_pivot_matched.to_csv(Path(sumdir) / f"optimizer_unit_summary_matched.csv")
#%%
mean_pivot_matched.T.plot(kind="bar", figsize=(20, 10))
plt.savefig(Path(sumdir) / f"optimizer_unit_summary_matched.png")
plt.show()

#%%
"""Normalize each unit's score by the max score across methods of the unit"""
layermean_pivot_matched = mean_pivot_matched\
    .groupby(by=["netname", "layer", "RFresize"]).mean()
layermean_pivot_matched.T.plot(kind="bar", figsize=(6, 4))
plt.savefig(Path(sumdir) / f"optimizer_unit_summary_matched_layermean.png")
plt.show()
#%% Normalize the activation by the max of mean activation of each unit.
unitmax_matched = mean_pivot_matched.max(axis=1) # FIXME, this is not actually max. it is max of mean
meannorm_pivot_matched = mean_pivot_matched / \
                         unitmax_matched.to_numpy()[:, None]
meannorm_pivot_matched.to_csv(Path(sumdir) / f"optimizer_unit_summary_matched_maxnorm.csv")

layermeannorm_pivot_matched = meannorm_pivot_matched\
    .groupby(by=["netname", "layer", "RFresize"]).mean()
layerorder = ['.layer1.Bottleneck1', '.layer2.Bottleneck3',
       '.layer3.Bottleneck5', '.layer4.Bottleneck2', '.Linearfc']
mapping = {layer: i for i, layer in enumerate(layerorder)}
key = layermeannorm_pivot_matched.index.get_level_values(1).map(mapping)
layermeannorm_pivot_matched = layermeannorm_pivot_matched.iloc[key.argsort()]

for netname in ["resnet50_linf8", "resnet50"]:
    msk = layermeannorm_pivot_matched.index.get_level_values("netname") == netname
    layermeannorm_pivot_matched.iloc[msk,:].T.plot(kind="bar", figsize=(6, 6), width=0.65)
    plt.xticks(rotation=30)
    plt.suptitle(f"Optimizer x GAN comparison (score normalized by unit max)\nVision model: {netname}")
    plt.tight_layout()
    saveallforms(sumdir, f"{netname}_optimizer_unit_summary_matched_layermean_maxnorm")
    plt.show()


#%%



#%%
# unitdirname = r"resnet50_.layer3.Bottleneck5_9_7_7_RFrsz"
# grad_scores_fn = Path(figdir) / f"{unitdirname}_proto_scores.pkl"
# evol_scores_fn = Path(figdir) / f"{unitdirname}_evolproto_info.pkl"
# grad_scores = pkl.load(open(grad_scores_fn, "rb"))
# evol_scores = pkl.load(open(evol_scores_fn, "rb"))
# #%%
# pd.DataFrame(evol_scores["CholCMA"])
