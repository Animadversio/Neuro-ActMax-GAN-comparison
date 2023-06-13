import os
from os.path import join
import pandas as pd
import numpy as np
from insilico_analysis.insilico_analysis_lib import sweep_dir

rootdir = r"F:\insilico_exps\GAN_Evol_cmp"
os.makedirs(join(rootdir, "summary"), exist_ok=True)
figdir = join(rootdir, "summary")
# datalist = glob.glob(join(rootdir, "*", "*.npz"))
#%% ResNet50
prefix = "corner-s_"
df_evol = sweep_dir(rootdir, unit_pattern="corner-s*", save_pattern="scores*.npz")
# df_evol.to_csv(join(rootdir, "summary", f"{prefix}raw_summary.csv"), index=False)
df_evol["optimmethod_raw"] = df_evol["optimmethod"]
#%%
print(df_evol.optimmethod.unique())
#%%
df_evol["timestep"] = df_evol.optimmethod.map(lambda x: int(x.split("_")[1][1]))
assert np.all(df_evol.loc[df_evol.layer.str.contains('V2')]["timestep"].unique() == np.arange(2, ).astype(int))
assert np.all(df_evol.loc[df_evol.layer.str.contains('V4')]["timestep"].unique() == np.arange(4, ).astype(int))
assert np.all(df_evol.loc[df_evol.layer.str.contains('IT')]["timestep"].unique() == np.arange(2, ).astype(int))
#%%
df_evol["optimmethod"] = df_evol["optimmethod_raw"].map(lambda x: x[4:])
print(df_evol.optimmethod.unique())
assert len(df_evol.optimmethod.unique()) == 4
#%%
df_evol.to_csv(join(rootdir, "summary", f"{prefix}raw_summary.csv"), index=False)
#%%
maxactdf_evol = df_evol.groupby(["netname", "layer", "RFresize", "unitid",]).agg({"score": "max"})
df_evol_norm = df_evol.merge(maxactdf_evol, on=["netname", "layer", "RFresize", "unitid"], suffixes=("", "_max"))
df_evol_norm["score_norm"] = df_evol_norm["score"] / df_evol_norm["score_max"]
df_evol_norm["maxscore_norm"] = df_evol_norm["maxscore"] / df_evol_norm["score_max"]
df_evol_norm.to_csv(join(rootdir, "summary", f"{prefix}raw_summary_norm.csv"), index=False)
#%%
df_evol_norm.groupby(["layer", "timestep", "optimmethod"]).agg({"score_norm": ["mean", "std"]})
#%%
df_evol.groupby(["layer", "timestep", "optimmethod"]).agg({"maxscore": ["mean", "std"]})
#%%
df_evol_norm.groupby(["layer", "timestep", "optimmethod"]).agg({"maxscore_norm": ["mean", "std"]})
#%%
# plot the score
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()
# sns.set_context("talk")
plt.figure()
df_evol_norm.groupby(["layer", "timestep", "optimmethod"]).plot(y="maxscore", kind="bar", alpha=0.5)
plt.show()
#%%
meansumdf_norm = df_evol_norm.groupby(["netname", "layer", "timestep", "RFresize", "optimmethod", "GANname"])\
    .agg({"score_norm":"mean","maxscore_norm":"mean","maxstep":"mean"})#.to_csv(join(rootdir, "summary_mean.csv"))
meansum_norm_pivot = meansumdf_norm.pivot_table(index=["netname", "layer", "timestep"],
                        columns=["GANname", "optimmethod",], values="maxscore_norm")
meansum_norm_pivot_T = meansumdf_norm.pivot_table(index=["GANname", "optimmethod", ],
                        columns=["layer", "RFresize", "timestep"], values="maxscore_norm")
#%%
from core.utils.plot_utils import saveallforms
meansum_norm_pivot.plot.bar(figsize=(8, 5))
plt.title("Score of the best activation")
plt.ylabel("Score")
plt.xlabel("Layer")
plt.xticks(rotation=30)
plt.tight_layout()
saveallforms(figdir, f"{prefix}score_bar_norm_by_layer_net_RF_bar")
plt.show()
#%%
meansum_norm_pivot_T.plot.bar(figsize=(8, 5))
plt.title("Score of the best activation")
plt.ylabel("Score")
plt.xlabel("Layer")
plt.xticks(rotation=30)
plt.tight_layout()
saveallforms(figdir, f"{prefix}score_bar_norm_by_GANoptim_bar")
plt.show()
