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
from insilico_analysis.insilico_analysis_lib import sweep_dir
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
            initscore = scores_all[generations <= generations.min() + 1].mean()
            # print((generations == generations.min()).sum(), (generations == generations.min()+1).sum(), )
            endscores = scores_all[generations == generations.max()].mean()
            maxscores = scores_all.max(axis=0)
            maxstep = np.argmax(scores_all, axis=0)
            df_col.append({**unitdict, **optimdict, **dict(score=endscores, maxscore=maxscores, maxstep=maxstep, initscore=initscore)})
            # raise  Exception

        # break
    df_evol = pd.DataFrame(df_col)
    # change datatype of columns GANname, layer, optimmethod, netname as string
    # df_evol = df_evol.astype({"GANname": str, "layer": str, "optimmethod": str, "netname": str,
    #                           "score": float, "maxscore": float, "maxstep": int, "RFresize": bool})
    return df_evol

# r"F:\insilico_exps\GAN_Evol_cmp\"

rootdir = r"F:\insilico_exps\GAN_Evol_cmp"
os.makedirs(join(rootdir, "summary"), exist_ok=True)
figdir = join(rootdir, "summary")
# datalist = glob.glob(join(rootdir, "*", "*.npz"))
df_evol = sweep_dir(rootdir, unit_pattern="resnet50_linf8*", save_pattern="scores*.npz")
df_evol = df_evol.astype({"GANname": str, "layer": str, "optimmethod": str, "netname": str,
                          "score": float, "maxscore": float, "maxstep": int, "RFresize": bool, "initscore": float, })
#%% Save the summary file
df_evol.to_csv(join(rootdir, "summary", f"resnet50_linf8_raw_init_summary.csv"), index=False)
#%%
df_evol.to_csv(join(rootdir, "summary", f"resnet50_linf8_raw_init_summary.csv"), index=False)
#%%
maxactdf_evol2 = df_evol.groupby(["netname", "layer", "RFresize", "unitid",]).agg({"score": "max"})
df_evol_norm2 = df_evol.merge(maxactdf_evol2, on=["netname", "layer", "RFresize", "unitid"],
                               suffixes=("", "_max"))
df_evol_norm2["score_norm"] = df_evol_norm2["score"] / df_evol_norm2["score_max"]
df_evol_norm2["maxscore_norm"] = df_evol_norm2["maxscore"] / df_evol_norm2["score_max"]
df_evol_norm2["initscore_norm"] = df_evol_norm2["initscore"] / df_evol_norm2["score_max"]
df_evol_norm2.to_csv(join(rootdir, "summary", "resnet50_linf8_raw_init_summary_norm.csv"), index=False)
#%%
# rename the layer names .Bottleneck to .B .layer to layer
df_evol_norm2["layershort"] = df_evol_norm2["layer"].str.replace(".Bottleneck", ".B", regex=False)
df_evol_norm2["layershort"] = df_evol_norm2["layershort"].str.replace(".layer", "layer", regex=False)
df_evol_norm2["layershort"] = df_evol_norm2["layershort"].str.replace(".Linearfc", "fc", regex=False)
#%%
import seaborn as sns
# split each layer into a panel
figh, axs = plt.subplots(1, 5, figsize=[13.5, 6])
for li, layername in enumerate(df_evol_norm2["layershort"].unique()):
    ax = axs[li]
    layerdf = df_evol_norm2[df_evol_norm2["layershort"] == layername]
    # sns.lineplot(data=layerdf, x="GANname", y="initscore_norm", hue="RFresize", ax=ax)
    sns.stripplot(data=layerdf, x="GANname", y="initscore_norm", hue="RFresize", ax=ax, dodge=True)
    # sns.violinplot(data=layerdf, x="GANname", y="initscore_norm", hue="RFresize", ax=ax, dodge=True)
    ax.set_title(layername)
    if li == 0:
        ax.set_ylabel("Initial Activation")
    else:
        ax.set_ylabel("")
# sns.lineplot(data=df_evol_norm2, x="GANname", y="initscore_norm", hue="layer")
plt.tight_layout()
plt.show()
#%%
