import shutil
import os
import re
import glob
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from os.path import join
from easydict import EasyDict as edict
import numpy as np
import pandas as pd
from core.utils.plot_utils import saveallforms
from core.utils.stats_utils import ttest_rel_print_df, ttest_rel_df, ttest_ind_print_df, ttest_ind_df, ttest_ind_print
from contextlib import redirect_stdout
outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\insilico_Evol_activation_cmp"
#%%
rootdir = r"F:\insilico_exps\GAN_Evol_cmp"
os.makedirs(join(rootdir, "summary"), exist_ok=True)
figdir = join(rootdir, "summary")
netname_prefix = "resnet50_linf8_"
# df_evol = pd.read_csv(join(rootdir, "summary", f"resnet50_linf8_raw_summary.csv"))
df_evol = pd.read_csv(join(rootdir, "summary", f"resnet50_linf8_raw_init_summary.csv"))
df_evol["layershort"] = df_evol["layer"].apply(lambda x: x.replace(".Linearfc", "fc")\
                                          .replace('.Bottleneck', "B").replace(".layer", "block"))
#%%
maxactdf_evol = df_evol.groupby(["netname", "layer", "RFresize", "unitid",]).agg({"maxscore": "max"})
df_evol_norm = df_evol.merge(maxactdf_evol,
                            on=["netname", "layer", "RFresize", "unitid"], suffixes=("", "_max"))
df_evol_norm["score_norm"] = df_evol_norm["score"] / df_evol_norm["maxscore_max"]
df_evol_norm["maxscore_norm"] = df_evol_norm["maxscore"] / df_evol_norm["maxscore_max"]
df_evol_norm["initscore_norm"] = df_evol_norm["initscore"] / df_evol_norm["maxscore_max"]
#%%
FC6msk = df_evol_norm.optimmethod.isin(['CholCMA_fc6', 'HessCMA500_fc6'])
BGmsk  = df_evol_norm.optimmethod.isin(['CholCMA', 'HessCMA'])
FC6_evol_norm_pivot = df_evol_norm[FC6msk].pivot(index=["netname", "layershort", "RFresize", "unitid", "RND"],
                                        columns='optimmethod', values='score_norm')
BG_evol_norm_pivot = df_evol_norm[BGmsk].pivot(index=["netname", "layershort", "RFresize", "unitid", "RND"],
                                        columns='optimmethod', values='score_norm')
# assert FC6_evol_norm_pivot[["netname", "layershort", "RFresize", "unitid"]].equals(BG_evol_norm_pivot[["netname", "layershort", "RFresize", "unitid"]])
assert FC6_evol_norm_pivot.droplevel(4).index.equals(BG_evol_norm_pivot.droplevel(4).index)
#%%
BGFC6_evol_norm_pivot_sel = pd.concat([FC6_evol_norm_pivot.droplevel(4), BG_evol_norm_pivot.droplevel(4)], axis=1)
print(BGFC6_evol_norm_pivot_sel.shape)
BGFC6_evol_norm_pivot_sel.reset_index(inplace=True)
#%%
BGFC6_evol_norm_pivot_merge = pd.merge(FC6_evol_norm_pivot, BG_evol_norm_pivot,
                                       on=["netname", "layershort", "RFresize", "unitid"], how='inner')
print(BGFC6_evol_norm_pivot_merge.shape)
BGFC6_evol_norm_pivot_merge.reset_index(inplace=True)

#%%
optim2plot = ['CholCMA_fc6', 'CholCMA', 'HessCMA'] # 'HessCMA500_fc6',
xtick_annot = ["DeePSim", "BG CholCMA", "BG HessCMA"]
layername = "fc"
for layername in BGFC6_evol_norm_pivot_merge.layershort.unique():
    plotdata = BGFC6_evol_norm_pivot_merge[(BGFC6_evol_norm_pivot_merge["layershort"] == layername) &
                                       (BGFC6_evol_norm_pivot_merge["RFresize"] == False)]

    plt.figure(figsize=(4.5, 6))
    # each row plot as a line
    plt.plot(plotdata[optim2plot].to_numpy().T, alpha=0.01, color="black")
    plt.plot(plotdata[optim2plot].mean(axis=0).to_numpy(), "-o", color="red", linewidth=4, )
    plt.xticks([0, 1, 2], xtick_annot)
    plt.title(f"{layername} optim comparison")
    plt.ylabel("Max Normalized activation")
    plt.show()

#%% create combined plot, each panel is a layer
optim2plot = ['CholCMA_fc6', 'CholCMA', 'HessCMA'] # 'HessCMA500_fc6',
xtick_annot = ["DeePSim", "BigGAN\nCholCMA", "BigGAN\nHessCMA"]
RFresize = False
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, layername in enumerate(BGFC6_evol_norm_pivot_sel.layershort.unique()):
    print(f"Layer: {layername}")
    plotdata = BGFC6_evol_norm_pivot_sel[(BGFC6_evol_norm_pivot_sel["layershort"] == layername) &
                                       (BGFC6_evol_norm_pivot_sel["RFresize"] == RFresize)]
    axes[i].plot(plotdata[optim2plot].to_numpy().T, alpha=0.08, color="black")
    axes[i].plot(plotdata[optim2plot].mean(axis=0).to_numpy(), "-o", color="red", linewidth=4, )
    axes[i].set_xticks([0, 1, 2])
    axes[i].set_xticklabels(xtick_annot)
    axes[i].set_title(f"{layername}")
    axes[i].set_ylim([-0.05, 1.05])
    if i == 0:
        axes[i].set_ylabel("Max Normalized activation")


plt.suptitle(f"{netname_prefix}{'RFrsz' if RFresize else ''} optim comparison")
plt.tight_layout()
saveallforms(outdir, f"{netname_prefix}_alllayers_{'RFrsz' if RFresize else ''}_optim_cmp", fig,
             fmts=["png", "pdf", "svg"])
plt.show()
#%%
# create combined plot, each panel is a layer
optim2plot = ['CholCMA_fc6', 'CholCMA', 'HessCMA'] # 'HessCMA500_fc6',
xtick_annot = ["DeePSim", "BigGAN\nCholCMA", "BigGAN\nHessCMA"]
RFresize = False
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, layername in enumerate(BGFC6_evol_norm_pivot_sel.layershort.unique()):
    print(f"Layer: {layername}")
    plotdata = BGFC6_evol_norm_pivot_sel[(BGFC6_evol_norm_pivot_sel["layershort"] == layername) &
                                       (BGFC6_evol_norm_pivot_sel["RFresize"] == RFresize)]
    axes[i].plot(plotdata[optim2plot].to_numpy().T, alpha=0.08, color="black")
    axes[i].plot(plotdata[optim2plot].mean(axis=0).to_numpy(), "-o", color="magenta", linewidth=4, )
    axes[i].plot(0.0*np.ones(plotdata.shape[0]), plotdata[optim2plot[0]].to_numpy(), "o", color="blue", alpha=0.1)
    axes[i].plot(1.0*np.ones(plotdata.shape[0]), plotdata[optim2plot[1]].to_numpy(), "o", color="red", alpha=0.1)
    axes[i].plot(2.0*np.ones(plotdata.shape[0]), plotdata[optim2plot[2]].to_numpy(), "o", color="green", alpha=0.1)
    axes[i].set_xticks([0, 1, 2])
    axes[i].set_xticklabels(xtick_annot)
    axes[i].set_title(f"{layername}")
    axes[i].set_ylim([-0.05, 1.05])
    if i == 0:
        axes[i].set_ylabel("Max Normalized activation")

plt.suptitle(f"{netname_prefix}{'RFrsz' if RFresize else ''} optim comparison")
plt.tight_layout()
saveallforms(outdir, f"{netname_prefix}_alllayers_{'RFrsz' if RFresize else ''}_optim_cmp_wcolor", fig,
             fmts=["png", "pdf", "svg"])
plt.show()

#%%
# create combined plot, each panel is a layer
optim2plot = ['CholCMA_fc6', 'CholCMA', 'HessCMA'] # 'HessCMA500_fc6',
xtick_annot = ["DeePSim", "BigGAN\nCholCMA", "BigGAN\nHessCMA"]
RFresize = False
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, layername in enumerate(BGFC6_evol_norm_pivot_sel.layershort.unique()):
    print(f"Layer: {layername}")
    plotdata = BGFC6_evol_norm_pivot_sel[(BGFC6_evol_norm_pivot_sel["layershort"] == layername) &
                                       (BGFC6_evol_norm_pivot_sel["RFresize"] == RFresize)]
    axes[i].plot(plotdata[optim2plot].to_numpy().T, alpha=0.08, color="black")
    axes[i].plot(plotdata[optim2plot].mean(axis=0).to_numpy(), "-o", color="magenta", linewidth=4, )
    axes[i].plot(0.0*np.ones(plotdata.shape[0]), plotdata[optim2plot[0]].to_numpy(), "o", color="blue", alpha=0.1)
    axes[i].plot(1.0*np.ones(plotdata.shape[0]), plotdata[optim2plot[1]].to_numpy(), "o", color="red", alpha=0.1)
    axes[i].plot(2.0*np.ones(plotdata.shape[0]), plotdata[optim2plot[2]].to_numpy(), "o", color="green", alpha=0.1)
    axes[i].set_xticks([0, 1, 2])
    axes[i].set_xticklabels(xtick_annot)
    axes[i].set_title(f"{layername}")
    axes[i].set_ylim([-0.05, 1.05])
    if i == 0:
        axes[i].set_ylabel("Max Normalized activation")

plt.suptitle(f"{netname_prefix}{'RFrsz' if RFresize else ''} optim comparison")
plt.tight_layout()
saveallforms(outdir, f"{netname_prefix}_alllayers_{'RFrsz' if RFresize else ''}_optim_cmp_wcolor", fig,
             fmts=["png", "pdf", "svg"])
plt.show()
#%%
optim2plot = ['CholCMA_fc6', 'CholCMA', 'HessCMA'] # 'HessCMA500_fc6',
xtick_annot = ["DeePSim", "BigGAN CholCMA", "BigGAN HessCMA"]
layername = "fc"
RFresize = False
for layername in BGFC6_evol_norm_pivot_sel.layershort.unique():
    print(f"Layer: {layername}")
    plotdata = BGFC6_evol_norm_pivot_sel[(BGFC6_evol_norm_pivot_sel["layershort"] == layername) &
                                       (BGFC6_evol_norm_pivot_sel["RFresize"] == RFresize)]
    ttest_rel_print_df(plotdata, None, optim2plot[0], optim2plot[1])
    ttest_rel_print_df(plotdata, None, optim2plot[0], optim2plot[2])
    ttest_rel_print_df(plotdata, None, optim2plot[2], optim2plot[1])
    # fraction of time BG is better than FC6
    print(f"Fraction BG (CholCMA) > FC6: {np.mean(plotdata[optim2plot[1]] > plotdata[optim2plot[0]])}")
    print(f"Fraction of BG (HessCMA) > FC6: {np.mean(plotdata[optim2plot[2]] > plotdata[optim2plot[0]])}")
    print(f"Fraction of BG (HessCMA) > BG (CholCMA): {np.mean(plotdata[optim2plot[2]] > plotdata[optim2plot[1]])}")
    figh = plt.figure(figsize=(4.5, 6))
    # each row plot as a line
    plt.plot(plotdata[optim2plot].to_numpy().T, alpha=0.08, color="black")
    plt.plot(plotdata[optim2plot].mean(axis=0).to_numpy(), "-o", color="red", linewidth=4, )
    plt.xticks([0, 1, 2], xtick_annot)
    plt.title(f"{layername} optim comparison")
    plt.ylabel("Max Normalized activation")
    saveallforms(outdir, f"{netname_prefix}{layername}_{'RFrsz' if RFresize else ''}_optim_cmp", figh, fmts=["png", "pdf", "svg"])
    plt.show()
#%%
#%%
# print the ttest results into txt file in figdir
layerlist = BGFC6_evol_norm_pivot_sel.layershort.unique()
optim2plot = ['CholCMA_fc6', 'CholCMA', 'HessCMA'] # 'HessCMA500_fc6',
with redirect_stdout(open(join(outdir, f"stats_{netname_prefix}_optim_cmp_latex.txt"), 'w')):
    for RFresize in [False, True]:
        for layername in layerlist:
            print(f"Layer: {layername}  RF resize={RFresize}")
            plotdata = BGFC6_evol_norm_pivot_sel[(BGFC6_evol_norm_pivot_sel["layershort"] == layername) &
                                                 (BGFC6_evol_norm_pivot_sel["RFresize"] == RFresize)]
            ttest_rel_print_df(plotdata, None, optim2plot[0], optim2plot[1], sem=True, latex=True)
            ttest_rel_print_df(plotdata, None, optim2plot[0], optim2plot[2], sem=True, latex=True)
            ttest_rel_print_df(plotdata, None, optim2plot[2], optim2plot[1], sem=True, latex=True)
            # fraction of time BG is better than FC6
            print("Final activation comparison: ")
            print(f"Fraction BG (CholCMA) > FC6: {np.mean(plotdata[optim2plot[1]] > plotdata[optim2plot[0]])}")
            print(f"Fraction of BG (HessCMA) > FC6: {np.mean(plotdata[optim2plot[2]] > plotdata[optim2plot[0]])}")
            print(f"Fraction of BG (HessCMA) > BG (CholCMA): {np.mean(plotdata[optim2plot[2]] > plotdata[optim2plot[1]])}")
            print("\n")

with redirect_stdout(open(join(outdir, f"stats_{netname_prefix}_layer_cmp_latex.txt"), 'w')):
    print(f"Diff activation between {optim2plot[0]} and {optim2plot[1]}, independent t-test between layers:")
    print("")
    for RFresize in [False, True]:
        diffdata_col = {}
        for layername in layerlist:
            plotdata = BGFC6_evol_norm_pivot_sel[(BGFC6_evol_norm_pivot_sel["layershort"] == layername) &
                                                 (BGFC6_evol_norm_pivot_sel["RFresize"] == RFresize)]
            diffdata = plotdata[optim2plot[0]] - plotdata[optim2plot[1]]
            diffdata_col[layername] = diffdata

        for layername0 in layerlist[:-1]:
            print(f"Layer: {layername0} vs {layerlist[-1]}  RF resize={RFresize}", end="\n")
            ttest_ind_print(diffdata_col[layername0], diffdata_col[layerlist[-1]], sem=True, latex=True)

        for layername0 in layerlist[:-2]:
            print(f"Layer: {layername0} vs {layerlist[-2]}  RF resize={RFresize}", end="\n")
            ttest_ind_print(diffdata_col[layername0], diffdata_col[layerlist[-2]], sem=True, latex=True)


#%%
FC6msk = df_evol_norm.optimmethod.isin(['CholCMA_fc6', 'HessCMA500_fc6'])
BGmsk  = df_evol_norm.optimmethod.isin(['CholCMA', 'HessCMA'])
FC6_evol_norm_pivot_init = df_evol_norm[FC6msk].pivot(index=["netname", "layershort", "RFresize", "unitid", "RND"],
                                        columns='optimmethod', values='initscore_norm')
BG_evol_norm_pivot_init = df_evol_norm[BGmsk].pivot(index=["netname", "layershort", "RFresize", "unitid", "RND"],
                                        columns='optimmethod', values='initscore_norm')
# assert FC6_evol_norm_pivot[["netname", "layershort", "RFresize", "unitid"]].equals(BG_evol_norm_pivot[["netname", "layershort", "RFresize", "unitid"]])
assert FC6_evol_norm_pivot_init.droplevel(4).index.equals(BG_evol_norm_pivot_init.droplevel(4).index)
#%%
BGFC6_evol_norm_pivot_sel_init = pd.concat([FC6_evol_norm_pivot_init.droplevel(4), BG_evol_norm_pivot_init.droplevel(4)], axis=1)
print(BGFC6_evol_norm_pivot_sel_init.shape)
BGFC6_evol_norm_pivot_sel_init.reset_index(inplace=True)
#%%
BGFC6_evol_norm_pivot_merge_init = pd.merge(FC6_evol_norm_pivot_init, BG_evol_norm_pivot_init,
                                       on=["netname", "layershort", "RFresize", "unitid"], how='inner')
print(BGFC6_evol_norm_pivot_merge_init.shape)
BGFC6_evol_norm_pivot_merge_init.reset_index(inplace=True)
#%%
 # create combined plot, each panel is a layer
optim2plot = ['CholCMA_fc6', 'CholCMA', 'HessCMA'] # 'HessCMA500_fc6',
xtick_annot = ["DeePSim", "BigGAN\nCholCMA", "BigGAN\nHessCMA"]
RFresize = True
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, layername in enumerate(BGFC6_evol_norm_pivot_sel_init.layershort.unique()):
    print(f"Layer: {layername}")
    plotdata = BGFC6_evol_norm_pivot_sel_init[(BGFC6_evol_norm_pivot_sel_init["layershort"] == layername) &
                                       (BGFC6_evol_norm_pivot_sel_init["RFresize"] == RFresize)]
    axes[i].plot(plotdata[optim2plot].to_numpy().T, alpha=0.08, color="black")
    axes[i].plot(plotdata[optim2plot].mean(axis=0).to_numpy(), "-o", color="magenta", linewidth=4, )
    axes[i].plot(0.0*np.ones(plotdata.shape[0]), plotdata[optim2plot[0]].to_numpy(), "o", color="blue", alpha=0.1)
    axes[i].plot(1.0*np.ones(plotdata.shape[0]), plotdata[optim2plot[1]].to_numpy(), "o", color="red", alpha=0.1)
    axes[i].plot(2.0*np.ones(plotdata.shape[0]), plotdata[optim2plot[2]].to_numpy(), "o", color="green", alpha=0.1)
    axes[i].set_xticks([0, 1, 2])
    axes[i].set_xticklabels(xtick_annot)
    axes[i].set_title(f"{layername}")
    axes[i].set_ylim([-0.08, 1.05])
    if i == 0:
        axes[i].set_ylabel("Max Normalized activation")

plt.suptitle(f"{netname_prefix}{'RFrsz' if RFresize else ''} optim comparison")
plt.tight_layout()
saveallforms(outdir, f"{netname_prefix}_alllayers_{'RFrsz' if RFresize else ''}_optim_cmp_wcolor_init", fig,
             fmts=["png", "pdf", "svg"])
plt.show()
#%%
optim2plot = ['CholCMA_fc6', 'CholCMA', ] # 'HessCMA500_fc6',
xtick_annot = ["DeePSim", "BigGAN\nCholCMA", ]
RFresize = False
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, layername in enumerate(BGFC6_evol_norm_pivot_sel_init.layershort.unique()):
    print(f"Layer: {layername}")
    plotdata = BGFC6_evol_norm_pivot_sel_init[(BGFC6_evol_norm_pivot_sel_init["layershort"] == layername) &
                                       (BGFC6_evol_norm_pivot_sel_init["RFresize"] == RFresize)]
    axes[i].plot(plotdata[optim2plot].to_numpy().T, alpha=0.08, color="black")
    axes[i].plot(plotdata[optim2plot].mean(axis=0).to_numpy(), "-o", color="magenta", linewidth=4, )
    axes[i].plot(0.0*np.ones(plotdata.shape[0]), plotdata[optim2plot[0]].to_numpy(), "o", color="blue", alpha=0.1)
    axes[i].plot(1.0*np.ones(plotdata.shape[0]), plotdata[optim2plot[1]].to_numpy(), "o", color="red", alpha=0.1)
    # axes[i].plot(2.0*np.ones(plotdata.shape[0]), plotdata[optim2plot[2]].to_numpy(), "o", color="green", alpha=0.1)
    axes[i].set_xticks([0, 1,])
    axes[i].set_xticklabels(xtick_annot)
    axes[i].set_title(f"{layername}")
    axes[i].set_ylim([-0.08, 1.05])
    axes[i].set_xlim([-0.25, 1.25])
    if i == 0:
        axes[i].set_ylabel("Max Normalized activation")

plt.suptitle(f"{netname_prefix}{'RFrsz' if RFresize else ''} optim comparison")
plt.tight_layout()
saveallforms(outdir, f"{netname_prefix}_alllayers_{'RFrsz' if RFresize else ''}_optim_BGDP_cmp_wcolor_init", fig,
             fmts=["png", "pdf", "svg"])
plt.show()
#%%
optim2plot = ['CholCMA_fc6', 'CholCMA', 'HessCMA']  # 'HessCMA500_fc6',
xtick_annot = ["DeePSim", "BigGAN CholCMA", "BigGAN HessCMA"]
# layername = "fc"
RFresize = False
for layername in BGFC6_evol_norm_pivot_sel_init.layershort.unique():
    print(f"Layer: {layername}")
    plotdata = BGFC6_evol_norm_pivot_sel_init[(BGFC6_evol_norm_pivot_sel_init["layershort"] == layername) &
                                         (BGFC6_evol_norm_pivot_sel_init["RFresize"] == RFresize)]
    ttest_rel_print_df(plotdata, None, optim2plot[0], optim2plot[1])
    ttest_rel_print_df(plotdata, None, optim2plot[0], optim2plot[2])
    ttest_rel_print_df(plotdata, None, optim2plot[2], optim2plot[1])
    # fraction of time BG is better than FC6
    print(f"Fraction BG (CholCMA) > FC6: {np.mean(plotdata[optim2plot[1]] > plotdata[optim2plot[0]])}")
    print(f"Fraction of BG (HessCMA) > FC6: {np.mean(plotdata[optim2plot[2]] > plotdata[optim2plot[0]])}")
    print(f"Fraction of BG (HessCMA) > BG (CholCMA): {np.mean(plotdata[optim2plot[2]] > plotdata[optim2plot[1]])}")

#%% Resnet50
rootdir = r"F:\insilico_exps\GAN_Evol_cmp"
os.makedirs(join(rootdir, "summary"), exist_ok=True)
figdir = join(rootdir, "summary")
netname_prefix = "resnet50_"
df_evol = pd.read_csv(join(rootdir, "summary", f"raw_summary.csv"))
df_evol["layershort"] = df_evol["layer"].apply(lambda x: x.replace(".Linearfc", "fc")\
                                          .replace('.Bottleneck', "B").replace(".layer", "block"))
#%%
maxactdf_evol = df_evol.groupby(["netname", "layer", "RFresize", "unitid",]).agg({"maxscore": "max"})
df_evol_norm = df_evol.merge(maxactdf_evol,
                            on=["netname", "layer", "RFresize", "unitid"], suffixes=("", "_max"))
df_evol_norm["score_norm"] = df_evol_norm["score"] / df_evol_norm["maxscore_max"]
df_evol_norm["maxscore_norm"] = df_evol_norm["maxscore"] / df_evol_norm["maxscore_max"]
#%%
FC6msk = df_evol_norm.optimmethod.isin(['CholCMA_fc6', 'HessCMA500_fc6'])
BGmsk = df_evol_norm.optimmethod.isin(['CholCMA', 'HessCMA'])
FC6_evol_norm_pivot = df_evol_norm[FC6msk].pivot(index=["netname", "layershort", "RFresize", "unitid", "RND"],
                                        columns='optimmethod', values='score_norm')
BG_evol_norm_pivot = df_evol_norm[BGmsk].pivot(index=["netname", "layershort", "RFresize", "unitid", "RND"],
                                        columns='optimmethod', values='score_norm')
#%%
# assert FC6_evol_norm_pivot[["netname", "layershort", "RFresize", "unitid"]].equals(BG_evol_norm_pivot[["netname", "layershort", "RFresize", "unitid"]])
assert FC6_evol_norm_pivot.droplevel(4).index.equals(BG_evol_norm_pivot.droplevel(4).index)
#%%
# iterate over unique values of ["netname", "layershort", "RFresize", "unitid"]
align_df_col =[]
for partname, partdf in df_evol_norm.groupby(["netname", "layershort", "RFresize", "unitid"]):
    # print(partname)
    # print(partdf)
    # check all optimmethod has > 10 rows
    valid_unit = True
    for optimname, optimdf in partdf.groupby("optimmethod"):
        if optimdf.shape[0] < 10:
            print(f"Warning: {optimname} has only {optimdf.shape[0]} rows, @ {partname}")
            valid_unit = False
            break
    if not valid_unit:
        continue
    # add to the aligned col, each col is an optimmethod
    aligned_part = []
    for optimname, optimdf in partdf.groupby("optimmethod"):
        # keep the index of the first 10 rows
        aligned_part.append(pd.DataFrame(optimdf.score_norm.iloc[:10],)
                            .reset_index(drop=True)
                            .rename(columns={"score_norm": optimname}))
    aligned_part_df = pd.concat(aligned_part, axis=1)
    aligned_part_df["netname"] = partname[0]
    aligned_part_df["layershort"] = partname[1]
    aligned_part_df["RFresize"] = partname[2]
    aligned_part_df["unitid"] = partname[3]
    align_df_col.append(aligned_part_df)
    # raise Exception("Not finished")
#%%
aligned_df = pd.concat(align_df_col, axis=0)
aligned_df.reset_index(inplace=True, drop=True)
#%%
# common_idx = pd.Index.intersection(FC6_evol_norm_pivot.droplevel(4).index,
#                                    BG_evol_norm_pivot.droplevel(4).index)
# BGFC6_evol_norm_pivot_sel = pd.concat([FC6_evol_norm_pivot.droplevel(4).loc[common_idx],
#                                        BG_evol_norm_pivot.droplevel(4).loc[common_idx]], axis=1)
# print(BGFC6_evol_norm_pivot_sel.shape)
# BGFC6_evol_norm_pivot_sel.reset_index(inplace=True)
# #%%
# assert FC6_evol_norm_pivot.droplevel(4).loc[common_idx].index.equals(BG_evol_norm_pivot.droplevel(4).loc[common_idx].index)
#%%
optim2plot = ['HessCMA500_fc6', 'CholCMA', 'HessCMA'] # 'HessCMA500_fc6',
xtick_annot = ["DeePSim\nHess", "BigGAN\nCholCMA", "BigGAN\nHessCMA"]
fig, axes = plt.subplots(1, 10, figsize=(30, 5))
for i, layername in enumerate(aligned_df.layershort.unique()):
    print(f"Layer: {layername}")
    plotdata = aligned_df[(aligned_df["layershort"] == layername) &
                           (aligned_df["RFresize"] == RFresize)]
    axes[i].plot(plotdata[optim2plot].to_numpy().T, alpha=0.08, color="black")
    axes[i].plot(plotdata[optim2plot].mean(axis=0).to_numpy(), "-o", color="magenta", linewidth=4, )
    axes[i].plot(0.0*np.ones(plotdata.shape[0]), plotdata[optim2plot[0]].to_numpy(), "o", color="blue", alpha=0.1)
    axes[i].plot(1.0*np.ones(plotdata.shape[0]), plotdata[optim2plot[1]].to_numpy(), "o", color="red", alpha=0.1)
    axes[i].plot(2.0*np.ones(plotdata.shape[0]), plotdata[optim2plot[2]].to_numpy(), "o", color="green", alpha=0.1)
    axes[i].set_xticks([0, 1, 2])
    axes[i].set_xticklabels(xtick_annot)
    axes[i].set_title(f"{layername}")
    axes[i].set_ylim([-0.05, 1.05])
    if i == 0:
        axes[i].set_ylabel("Max Normalized activation")

plt.suptitle(f"{netname_prefix}{'RFrsz' if RFresize else ''} optim comparison")
plt.tight_layout()
saveallforms(outdir, f"{netname_prefix}_alllayers_{'RFrsz' if RFresize else ''}_optim_cmp_wcolor", fig,
             fmts=["png", "pdf", "svg"])
plt.show()
#%%
# print the ttest results into txt file in figdir
optim2plot = ['HessCMA500_fc6', 'CholCMA', 'HessCMA']  # 'HessCMA500_fc6',
RFresize = True
with open(join(outdir, f"stats_{netname_prefix}{'RFrsz' if RFresize else ''}_optim_cmp.txt"), 'w') as f:
    with redirect_stdout(f):
        for layername in aligned_df.layershort.unique():
            print(f"Layer: {layername}")
            plotdata = aligned_df[(aligned_df["layershort"] == layername) &
                                 (aligned_df["RFresize"] == RFresize)]
            ttest_rel_print_df(plotdata, None, optim2plot[0], optim2plot[1])
            ttest_rel_print_df(plotdata, None, optim2plot[0], optim2plot[2])
            ttest_rel_print_df(plotdata, None, optim2plot[2], optim2plot[1])
            # fraction of time BG is better than FC6
            print(f"Fraction BG (CholCMA) > FC6: {np.mean(plotdata[optim2plot[1]] > plotdata[optim2plot[0]])}")
            print(f"Fraction of BG (HessCMA) > FC6: {np.mean(plotdata[optim2plot[2]] > plotdata[optim2plot[0]])}")
            print(f"Fraction of BG (HessCMA) > BG (CholCMA): {np.mean(plotdata[optim2plot[2]] > plotdata[optim2plot[1]])}")
            print("\n")
