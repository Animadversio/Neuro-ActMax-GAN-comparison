
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
#%%
rootdir = r"F:\insilico_exps\GAN_gradEvol_cmp"
rootdir = r"F:\insilico_exps\GAN_Evol_cmp"
rootpath = Path(rootdir)
datalist = glob.glob(join(rootdir, "*", "*.pt"))
#%%

unitdirs = list(rootpath.glob("res*"))
df_col = []
for unitdir in tqdm(unitdirs):
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
    print(unit_match, "=", netname, layer, unitid, x, y, RFresize)
    unitdict = edict(netname=netname, layer=layer, unitid=unitid, x=x, y=y, RFresize=RFresize)
    #%
    savefiles = list(unitdir.glob("optimdata*.pt"))
    savefn_pat = re.compile("optimdata_([^_]*)(_fc6)?_(\d\d\d\d\d).pt$")
    for savefn in tqdm(savefiles):
        savefn_pat_match = savefn_pat.findall(savefn.name)
        assert len(savefn_pat_match) == 1
        savefn_pat_match = savefn_pat_match[0]
        optimmethod = savefn_pat_match[0]
        RND = int(savefn_pat_match[-1])
        if savefn_pat_match[1] == "_fc6":
            GANname = "fc6"
        else:
            GANname = "BigGAN"
        # print(optimmethod, RND, GANname)
        optimdict = edict(optimmethod=optimmethod, RND=RND, GANname=GANname)
        data = torch.load(savefn)
        score_traj = data["score_traj"]
        endscores = score_traj[-1, :]
        maxscores, maxsteps = score_traj.max(dim=0)
        for i in range(len(endscores)):
            df_col.append({**unitdict, **optimdict, **dict(batchi=i, score=endscores[i], maxscore=maxscores[i], maxstep=maxsteps[i])})

    # break
#%%
import pandas as pd
df = pd.DataFrame(df_col)
# change datatype of columns GANname, layer, optimmethod, netname as string
df = df.astype({"GANname": str, "layer": str, "optimmethod": str, "netname": str,
                "score": float, "maxscore": float, "maxstep": int, "batchi": int})
#%%
df.to_csv(join(rootdir, "summary", "raw_summary_grad.csv"))
#%%
# df.groupby(["netname", "layer", "RFresize", "optimmethod", "GANname"]).mean().to_csv(join(rootdir, "summary_mean.csv"))
meansumdf = df.groupby(["netname", "layer", "RFresize", "optimmethod", "GANname"])\
    .agg({"score": "mean","maxscore": "mean","maxstep": "mean"})#.to_csv(join(rootdir, "summary_mean.csv"))
meansum_pivot = meansumdf.pivot_table(index=["netname", "layer", "RFresize"],
                      columns=["optimmethod", "GANname"], values="score")

meansum_pivot.to_csv(join(rootdir, "summary_mean_pivot.csv"))
#%%
maxactdf = df.groupby(["netname", "layer", "RFresize", "unitid", ]).agg({"score": "max"})#.to_csv(join(rootdir, "summary_mean.csv"))
#%%
"""divide score and maxscore by max activation in each row by the max of corresponding unit"""
df_norm = df.merge(maxactdf, on=["netname", "layer", "RFresize", "unitid"], suffixes=("", "_max"))
#%%
df_norm["score_norm"] = df_norm["score"] / df_norm["score_max"]
df_norm["maxscore_norm"] = df_norm["maxscore"] / df_norm["score_max"]
#%%
meansumdf_norm = df_norm.groupby(["netname", "layer", "RFresize", "optimmethod", "GANname"])\
    .agg({"score_norm":"mean","maxscore_norm":"mean","maxstep":"mean"})#.to_csv(join(rootdir, "summary_mean.csv"))
meansum_norm_pivot = meansumdf_norm.pivot_table(index=["netname", "layer", "RFresize"],
                        columns=["optimmethod", "GANname"], values="score_norm")
meansum_norm_pivot_T = meansumdf_norm.pivot_table(index=["optimmethod", "GANname"],
                        columns=["netname", "layer", "RFresize"], values="score_norm")
#%%
meansum_norm_pivot.to_csv(join(rootdir, "summary_mean_norm_pivot.csv"))
#%%
import os
from core.utils.plot_utils import saveallforms
os.makedirs(join(rootdir, "summary"), exist_ok=True)
figdir = join(rootdir, "summary")
#%%
# mask = (df_norm["optimmethod"] == "Adam01") & (df_norm["GANname"] == "fc6"))
# plt.figure(figsize=(8, 5))
meansum_norm_pivot[[('Adam001', 'BigGAN'), ('Adam001Hess', 'BigGAN'), ('Adam01', 'fc6')]]\
    .plot.bar(figsize=(8, 5))
# xticks rotate
plt.ylabel("score normalized by unit max")
plt.xticks(rotation=40)
plt.tight_layout()
saveallforms(figdir, "score_norm_by_layer_net_RF_bar")
plt.show()
#%%
layerorder = ['.layer1', '.layer2', '.layer3', '.layer4', '.Linearfc']
mapping = {layer: i for i, layer in enumerate(layerorder)}
mapping_net = {"resnet50": 0, "resnet50_linf8": 100}
#%%
key = meansum_norm_pivot.index.get_level_values('layer').map(mapping) + \
      meansum_norm_pivot.index.get_level_values('netname').map(mapping_net)
meansum_norm_pivot = meansum_norm_pivot.iloc[key.argsort()]
meansum_norm_pivot[[('Adam001', 'BigGAN'), ('Adam001Hess', 'BigGAN'), ('Adam01', 'fc6')]]\
    .plot.bar(figsize=(8, 5))
plt.ylabel("score normalized by unit max")
plt.xticks(rotation=40)  # xticks rotate
plt.tight_layout()
saveallforms(figdir, "score_norm_by_layer_net_RF_bar")
plt.show()
#%%
# change box width to 0.5
meansum_norm_pivot[[
                    ('Adam001', 'BigGAN'), ('Adam001Hess', 'BigGAN'),
                    ('Adam0003', 'BigGAN'), ('Adam0003Hess', 'BigGAN'),
                    ('Adam0001', 'BigGAN'), ('Adam0001Hess', 'BigGAN'),
                    ('SGD001', 'BigGAN'), ('SGD001Hess', 'BigGAN'),
                    ('SGD0003', 'BigGAN'), ('SGD0003Hess', 'BigGAN'),
                    ('SGD0001', 'BigGAN'), ('SGD0001Hess', 'BigGAN'),
                    ]]\
    .plot.bar(figsize=(8, 5), width=0.75)
plt.ylabel("score normalized by unit max")
plt.xticks(rotation=40)  # xticks rotate
plt.tight_layout()
saveallforms(figdir, "BigGAN_gradOptimizer_cmp_score_norm_by_layer_net_RF")
plt.show()
#%%
# meansum_norm_pivot_T.loc[key.argsort()]
meansum_norm_pivot_T.plot.bar(figsize=(8, 8), width=0.75)
plt.ylabel("score normalized by unit max")
plt.xticks(rotation=75)  # xticks rotate
plt.tight_layout()
saveallforms(figdir, "BigGAN_gradOptimizer_cmp_score_norm_by_layer_net_RF_Trsps")
plt.show()
#%%
# select rows that has netname == "resnet50" from meansum_norm_pivot
resnet50_norm_mean_df = meansum_norm_pivot.loc[("resnet50", slice(None), slice(None)),]
key = resnet50_norm_mean_df.index.get_level_values('layer').map(mapping)
resnet50_norm_mean_df = resnet50_norm_mean_df.iloc[key.argsort()]
resnet50_norm_mean_df[[('Adam001', 'BigGAN'), ('Adam001Hess', 'BigGAN'), ('Adam01', 'fc6')]].plot.bar(figsize=(6, 5))
plt.ylabel("score normalized by unit max")
plt.xticks(rotation=40)
plt.tight_layout()
saveallforms(figdir, "resnet50_score_norm_by_layer_RF_bar")
plt.show()
#%%
# select rows that has netname == "resnet50" from meansum_norm_pivot
reslinf8_norm_mean_df = meansum_norm_pivot.loc[("resnet50_linf8", slice(None), slice(None)),]
key = reslinf8_norm_mean_df.index.get_level_values('layer').map(mapping)
reslinf8_norm_mean_df = reslinf8_norm_mean_df.iloc[key.argsort()]
reslinf8_norm_mean_df[[('Adam001', 'BigGAN'), ('Adam001Hess', 'BigGAN'), ('Adam01', 'fc6')]]\
    .plot.bar(figsize=(6, 5), rot=40)
plt.ylabel("score normalized by unit max")
# plt.xticks(rotation=40)
plt.tight_layout()
saveallforms(figdir, "resnet50_linf8_score_norm_by_layer_RF_bar")
plt.show()
#%%
# set the order of bars in the plot
resnet50_norm_mean_df[[('Adam001', 'BigGAN'), ('Adam001Hess', 'BigGAN'), ('Adam01', 'fc6')]]\
    .plot.bar(figsize=(6, 5), color=["C0", "C1", "C2"])
#%%
import matplotlib.pyplot as plt
import seaborn as sns
sns.barplot(x=("netname","layer"), y="score_norm", hue="optimmethod", data=df_norm)
plt.show()
#%%
optimorder = ['SGD0001', 'SGD0001Hess', 'SGD0003', 'SGD0003Hess',
              'SGD001', 'SGD001Hess', 'Adam0001', 'Adam0001Hess',
              'Adam0003', 'Adam0003Hess', 'Adam001', 'Adam001Hess',
              'Adam01', 'Adam01Hess', ]
rename_mapping = lambda x: (x + "_fc6") if x in ['Adam01', 'Adam01Hess'] else x
optim_remap = {x: rename_mapping(x) for x in optimorder}
#%%
sns.violinplot(x=("optimmethod"), y="score_norm", data=df_norm,
               order=optimorder, cut=0.0, width=0.9)
plt.xticks(rotation=20)
plt.gca().set_xticklabels([optim_remap[x] for x in optimorder])
plt.suptitle("Optimizer comparison\nPooled all network, layer, and unit")
plt.tight_layout()
saveallforms(figdir, "all_pooled_optimizer_cmp_score_norm_violin")
plt.show()
#%%
for netname in ["resnet50", "resnet50_linf8"]:
    mask = df_norm.netname == netname
    sns.violinplot(x=("optimmethod"), y="score_norm", data=df_norm[mask],
                   order=optimorder, cut=0.0, width=0.9)
    plt.xticks(rotation=20)
    plt.gca().set_xticklabels([optim_remap[x] for x in optimorder])
    plt.suptitle(f"Optimizer comparison\n{netname} Pooled all layer, and unit")
    plt.tight_layout()
    print(plt.gca().get_xticklabels())
    saveallforms(figdir, f"{netname}_pooled_optimizer_cmp_score_norm_violin")
    plt.show()

#%%
sns.swarmplot(x=("optimmethod"), y="score_norm", data=df_norm,
               order=optimorder, )
plt.xticks(rotation=20)
plt.gca().set_xticklabels([optim_remap[x] for x in optimorder])
plt.suptitle("Optimizer comparison\nPooled all network, layer, and unit")
plt.tight_layout()
saveallforms(figdir, "all_pooled_optimizer_cmp_score_norm_swarm")
plt.show()