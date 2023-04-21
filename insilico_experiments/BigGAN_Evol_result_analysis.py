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
#%%
def sweep_dir(rootdir, unit_pattern, save_pattern):
    rootpath = Path(rootdir)
    unitdirs = list(rootpath.glob(unit_pattern))
    df_col = []
    for unitdir in tqdm(unitdirs):
        if ".SelectAdaptivePool2dglobal_pool" in unitdir.name:
            # this layername has _ in it so it will fail the regex below
            parts = unitdir.name.split("_"+".SelectAdaptivePool2dglobal_pool"+"_")
            netname = parts[0]
            layer = ".SelectAdaptivePool2dglobal_pool"
            RFresize = False
            unitstr = parts[1]
        else:
            unit_pat = re.compile("([^.]*)_([^_]*)_([\d_]*)(_RFrsz)?$")
            unit_match = unit_pat.findall(unitdir.name)
            assert len(unit_match) == 1
            unit_match = unit_match[0]
            netname = unit_match[0]
            layer = unit_match[1]
            RFresize = True if unit_match[3] == "_RFrsz" else False
            unitstr = unit_match[2]

        if "_" in unitstr:
            unit = unitstr.split("_")
            unitid = int(unit[0])
            x = int(unit[1])
            y = int(unit[2])
        else:
            unitid = int(unitstr)
            x = None
            y = None
        # print(unit_match)
        print(unitdir.name, "=", netname, layer, unitid, x, y, RFresize)
        unitdict = edict(netname=netname, layer=layer, unitid=unitid, x=x, y=y, RFresize=RFresize)

        savefiles = list(unitdir.glob("scores*.npz"))
        savefn_pat = re.compile("scores(.*)_(\d\d\d\d\d).npz$")
        for savefn in savefiles:
            savefn_pat_match = savefn_pat.findall(savefn.name)
            assert len(savefn_pat_match) == 1
            savefn_pat_match = savefn_pat_match[0]
            optimmethod = savefn_pat_match[0]
            RND = int(savefn_pat_match[-1])
            if optimmethod.endswith("_fc6"):
                GANname = "fc6"
            else:
                GANname = "BigGAN"
            # print(optimmethod, RND, GANname)
            optimdict = edict(optimmethod=optimmethod, RND=RND, GANname=GANname)
            data = np.load(savefn)
            scores_all = data["scores_all"]
            generations = data["generations"]
            endscores = scores_all[generations == generations.max()].mean()
            maxscores = scores_all.max(axis=0)
            maxstep = np.argmax(scores_all, axis=0)
            df_col.append({**unitdict, **optimdict, **dict(score=endscores, maxscore=maxscores, maxstep=maxstep)})
            # raise  Exception

        # break
    df_evol = pd.DataFrame(df_col)
    # change datatype of columns GANname, layer, optimmethod, netname as string
    # df_evol = df_evol.astype({"GANname": str, "layer": str, "optimmethod": str, "netname": str,
    #                           "score": float, "maxscore": float, "maxstep": int, "RFresize": bool})
    return df_evol
#%%
rootdir = r"F:\insilico_exps\GAN_Evol_cmp"
os.makedirs(join(rootdir, "summary"), exist_ok=True)
figdir = join(rootdir, "summary")
# datalist = glob.glob(join(rootdir, "*", "*.npz"))
#%% ResNet50
prefix = "resnet50_"
df_evol = sweep_dir(rootdir, unit_pattern="resnet50*", save_pattern="scores*.npz")
df_evol.to_csv(join(rootdir, "summary", f"{prefix}raw_summary.csv"), index=False)
#%%
maxactdf_evol = df_evol.groupby(["netname", "layer", "RFresize", "unitid",]).agg({"score": "max"})
df_evol_norm = df_evol.merge(maxactdf_evol, on=["netname", "layer", "RFresize", "unitid"], suffixes=("", "_max"))
df_evol_norm["score_norm"] = df_evol_norm["score"] / df_evol_norm["score_max"]
df_evol_norm["maxscore_norm"] = df_evol_norm["maxscore"] / df_evol_norm["score_max"]
df_evol_norm.to_csv(join(rootdir, "summary", f"{prefix}raw_summary_norm.csv"), index=False)
#%%
mask = ~df_evol_norm.optimmethod.isin(["CholCMA_class",])
meansumdf_norm = df_evol_norm[mask].groupby(["netname", "layer", "RFresize", "optimmethod", "GANname"])\
    .agg({"score_norm":"mean","maxscore_norm":"mean","maxstep":"mean"})#.to_csv(join(rootdir, "summary_mean.csv"))
meansum_norm_pivot = meansumdf_norm.pivot_table(index=["netname", "layer", ],
                        columns=["GANname", "optimmethod",], values="score_norm")
meansum_norm_pivot_T = meansumdf_norm.pivot_table(index=["optimmethod", "GANname"],
                        columns=["layer", "RFresize"], values="score_norm")
#%%
meansum_norm_pivot.plot.bar(figsize=(8, 5))
# xticks rotate
plt.ylabel("score normalized by unit max")
plt.xticks(rotation=40)
plt.tight_layout()
saveallforms(figdir, f"{prefix}score_norm_by_layer_net_RF_bar")
plt.show()
#%%
meansum_norm_pivot_T.plot.bar(figsize=(8, 5))
# xticks rotate
plt.ylabel("score normalized by unit max")
plt.xticks(rotation=40)
plt.tight_layout()
saveallforms(figdir, f"{prefix}score_norm_by_optim_GAN_bar")
plt.show()



#%% Efficient net
# rootdir = r"F:\insilico_exps\GAN_Evol_cmp"
prefix = "effnet_"
df_evol = sweep_dir(rootdir, unit_pattern="tf_efficientnet*", save_pattern="scores*.npz")
df_evol.to_csv(join(rootdir, "summary", f"{prefix}raw_summary.csv"), index=False)
#%%
maxactdf_evol = df_evol.groupby(["netname", "layer", "RFresize", "unitid",]).agg({"score": "max"})
df_evol_norm = df_evol.merge(maxactdf_evol, on=["netname", "layer", "RFresize", "unitid"], suffixes=("", "_max"))
df_evol_norm["score_norm"] = df_evol_norm["score"] / df_evol_norm["score_max"]
df_evol_norm["maxscore_norm"] = df_evol_norm["maxscore"] / df_evol_norm["score_max"]
df_evol_norm.to_csv(join(rootdir, "summary", f"{prefix}raw_summary_norm.csv"), index=False)
#%%
# change name of netname
df_evol_norm["netshort"] = df_evol_norm.netname.str.replace("tf_efficientnet_b6", "EffNet", )
df_evol_norm["layershort"] = df_evol_norm.layer.str.replace(".SelectAdaptivePool2dglobal_pool", "globalpool", ).\
    str.replace(".blocks", "blocks", ).str.replace(".Linearclassifier", "fc", )
#%%
mask = ~df_evol_norm.optimmethod.isin(["CholCMA_class",])
meansumdf_norm = df_evol_norm[mask].groupby(["netshort", "layershort", "RFresize", "optimmethod", "GANname"])\
    .agg({"score_norm":"mean","maxscore_norm":"mean","maxstep":"mean"})#.to_csv(join(rootdir, "summary_mean.csv"))
meansum_norm_pivot = meansumdf_norm.pivot_table(index=["netshort", "layershort", ],
                        columns=["GANname", "optimmethod",], values="score_norm")
meansum_norm_pivot_T = meansumdf_norm.pivot_table(index=["optimmethod", "GANname"],
                        columns=["netshort", "layershort", ], values="score_norm")
#%%
meansum_norm_pivot.plot.bar(figsize=(8, 5))
# xticks rotate
plt.ylabel("score normalized by unit max")
plt.xticks(rotation=40)
plt.tight_layout()
saveallforms(figdir, f"{prefix}score_norm_by_layer_net_RF_bar")
plt.show()
#%%
meansum_norm_pivot_T.plot.bar(figsize=(8, 5))
# xticks rotate
plt.ylabel("score normalized by unit max")
plt.xticks(rotation=40)
plt.tight_layout()
saveallforms(figdir, f"{prefix}score_norm_by_optim_GAN_bar")
plt.show()



#%% ResNet50_linf8 robust
rootdir = r"E:\Cluster_Backup\GAN_Evol_cmp"
os.makedirs(join(rootdir, "summary"), exist_ok=True)
figdir = join(rootdir, "summary")
# datalist = glob.glob(join(rootdir, "*", "*.npz"))
df_evol = sweep_dir(rootdir, unit_pattern="resnet50_linf8*", save_pattern="scores*.npz")
df_evol = df_evol.astype({"GANname": str, "layer": str, "optimmethod": str, "netname": str,
                          "score": float, "maxscore": float, "maxstep": int, "RFresize": bool})
df_evol.to_csv(join(rootdir, "summary", f"resnet50_linf8_raw_summary.csv"), index=False)
#%%
df_evol.to_csv(join(r"F:\insilico_exps\GAN_Evol_cmp", "summary", f"resnet50_linf8_raw_summary.csv"), index=False)

#%% ResNet50_linf8 robust
rootdir = r"E:\Cluster_Backup\GAN_Evol_cmp"
os.makedirs(join(rootdir, "summary"), exist_ok=True)
figdir = join(rootdir, "summary")
# datalist = glob.glob(join(rootdir, "*", "*.npz"))
df_evol2 = sweep_dir(rootdir, unit_pattern="res*", save_pattern="scores*.npz")
#%%
df_evol2 = df_evol2.astype({"GANname": str, "layer": str, "optimmethod": str, "netname": str,
                          "score": float, "maxscore": float, "maxstep": int, "RFresize": bool})
df_evol2.to_csv(join(rootdir, "summary", "raw_summary.csv"), index=False)
#%%

maxactdf_evol2 = df_evol2.groupby(["netname", "layer", "RFresize", "unitid",]).agg({"score": "max"})
df_evol_norm2 = df_evol2.merge(maxactdf_evol2, on=["netname", "layer", "RFresize", "unitid"],
                               suffixes=("", "_max"))
df_evol_norm2["score_norm"] = df_evol_norm2["score"] / df_evol_norm2["score_max"]
df_evol_norm2["maxscore_norm"] = df_evol_norm2["maxscore"] / df_evol_norm2["score_max"]
df_evol_norm2.to_csv(join(rootdir, "summary", "raw_summary_norm.csv"), index=False)
#%%
mask = ~df_evol_norm2.optimmethod.isin(["CholCMA_class",])
meansumdf_norm = df_evol_norm2[mask].groupby(["layer", "RFresize", "optimmethod", "GANname"])\
    .agg({"score_norm":"mean","maxscore_norm":"mean","maxstep":"mean"})#.to_csv(join(rootdir, "summary_mean.csv"))
meansum_norm_pivot = meansumdf_norm.pivot_table(index=["layer", "RFresize"],
                        columns=["optimmethod", "GANname"], values="score_norm")
meansum_norm_pivot_T = meansumdf_norm.pivot_table(index=["optimmethod", "GANname"],
                        columns=["layer", "RFresize"], values="score_norm")
#%%
meansumdf_norm.pivot_table(index=["layer", "RFresize"],
                        columns=["optimmethod", "GANname"], values="score_norm")\
    .to_csv(join(rootdir, "summary", "summary_pivot_layer_by_optimizer.csv"))

meansumdf_norm.pivot_table(index=["optimmethod", "GANname"],
                        columns=["layer", "RFresize"], values="score_norm")\
    .to_csv(join(rootdir, "summary", "summary_pivot_optimizer_by_layer.csv"))
#%%
meansum_norm_pivot.plot.bar(figsize=(8, 5))
# xticks rotate
plt.ylabel("score normalized by unit max")
plt.xticks(rotation=40)
plt.suptitle("Normalized score for ResNet50-robust")
plt.tight_layout()
saveallforms(figdir, "ResNet50-robust_score_norm_by_layer_RF_bar")
plt.show()
#%%
meansum_norm_pivot_T.plot.bar(figsize=(6, 5))
# xticks rotate
plt.ylabel("score normalized by unit max")
plt.xticks(rotation=40)
plt.suptitle("Normalized score for ResNet50-robust")
plt.tight_layout()
saveallforms(figdir, "ResNet50-robust_norm_by_optim_GAN_bar")
plt.show()
#%%
df_evol_norm2[mask].groupby(["layer", "RFresize", "optimmethod", "GANname"])\
    .plot.density(y="score_norm")
#%%
meansum_norm_pivot_T.plot.violin(figsize=(6, 5))
# xticks rotate
plt.ylabel("score normalized by unit max")
plt.xticks(rotation=40)
plt.suptitle("Normalized score for ResNet50-robust")
plt.tight_layout()
saveallforms(figdir, "ResNet50-robust_norm_by_optim_GAN_violin")
plt.show()