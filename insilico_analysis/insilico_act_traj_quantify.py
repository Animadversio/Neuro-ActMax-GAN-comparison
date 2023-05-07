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
import pickle as pkl
from collections import defaultdict
from core.utils.plot_utils import saveallforms
from pathlib import Path
import seaborn as sns
from core.utils.plot_utils import saveallforms

rootdir = r"F:\insilico_exps\GAN_Evol_cmp"
outdir = r"F:\insilico_exps\GAN_Evol_cmp\evol_traj_merge"
img_mtg_fps = [*Path(r"F:\insilico_exps\GAN_Evol_cmp\protoimgs_merge").glob("*.jpg")]
unitlist = [img_mtg_fp.name.replace("_optim_pool.jpg", "") for img_mtg_fp in img_mtg_fps]
#%%
sns.reset_defaults()
# remove the upper and right part of the frame
sns.set_style("white", {'axes.spines.right': False, 'axes.spines.top': False})
clrs = ["red", "green", "blue"]
#%%
# use the agg mode
# plt.switch_backend('agg')
# get current backend
plt.get_backend() # 'module://backend_interagg'
#%%
def _procrustes_array(arr, desired_len=10, axis=0):
    if arr.shape[axis] == desired_len:
        return arr
    elif arr.shape[axis] > desired_len:
        return arr[:desired_len]
    else:
        pad_shape = list(arr.shape)
        pad_shape[axis] = desired_len - arr.shape[axis]
        return np.concatenate([arr, np.full(pad_shape, np.nan)], axis=axis)
        # pad with the last value

#%%
optimnames = "CholCMA", "HessCMA", "HessCMA500_fc6"
score_mean1_col = []
score_mean2_col = []
score_mean3_col = []
score_sem1_col = []
score_sem2_col = []
score_sem3_col = []
FC_win_prob_col = []
BG_win_prob_col = []
FC_BG_tval_col = []
for unitname in tqdm(unitlist):
    # unitdir = join(r"F:\insilico_exps\GAN_Evol_cmp", unitname)
    score_traj_dict = pkl.load(open(join(outdir, f"{unitname}_score_traj_dict.pkl"), "rb"))
    score_mean1 = score_traj_dict['HessCMA500_fc6']['scores_mean']
    score_mean2 = score_traj_dict['CholCMA']['scores_mean']
    score_mean3 = score_traj_dict['HessCMA']['scores_mean']
    score_sem1 = score_traj_dict['HessCMA500_fc6']['scores_sem']
    score_sem2 = score_traj_dict['CholCMA']['scores_sem']
    score_sem3 = score_traj_dict['HessCMA']['scores_sem']
    # cross compare
    FC_win = score_mean1[:, None] - score_mean2[None, :] > 2 * np.sqrt(score_sem1[:, None]**2 + score_sem2[None, :]**2)
    BG_win = score_mean2[None, :] - score_mean1[:, None] > 2 * np.sqrt(score_sem1[:, None]**2 + score_sem2[None, :]**2)
    FC_BG_tval = (score_mean1[:, None] - score_mean2[None, :]) / np.sqrt(score_sem1[:, None]**2 + score_sem2[None, :]**2)
    FC_win_prob = np.mean(FC_win, axis=(0, 1))
    BG_win_prob = np.mean(BG_win, axis=(0, 1))
    FC_BG_tval_mean = np.mean(FC_BG_tval, axis=(0, 1))
    score_mean1_col.append(_procrustes_array(score_mean1))
    score_mean2_col.append(_procrustes_array(score_mean2))
    score_mean3_col.append(_procrustes_array(score_mean3))
    score_sem1_col.append(_procrustes_array(score_sem1))
    score_sem2_col.append(_procrustes_array(score_sem2))
    score_sem3_col.append(_procrustes_array(score_sem3))
    FC_win_prob_col.append(FC_win_prob)
    BG_win_prob_col.append(BG_win_prob)
    FC_BG_tval_col.append(FC_BG_tval_mean)

score_mean1_col = np.stack(score_mean1_col, axis=0)
score_mean2_col = np.stack(score_mean2_col, axis=0)
score_mean3_col = np.stack(score_mean3_col, axis=0)
score_sem1_col = np.stack(score_sem1_col, axis=0)
score_sem2_col = np.stack(score_sem2_col, axis=0)
score_sem3_col = np.stack(score_sem3_col, axis=0)
FC_win_prob_col = np.stack(FC_win_prob_col, axis=0)
BG_win_prob_col = np.stack(BG_win_prob_col, axis=0)
FC_BG_tval_col = np.stack(FC_BG_tval_col, axis=0)
#%%
figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\insilico_activation_dynamics"
np.savez(join(figdir, "insilico_act_traj_synopsis.npz"), score_mean1_col=score_mean1_col,
            score_mean2_col=score_mean2_col, score_mean3_col=score_mean3_col,
            score_sem1_col=score_sem1_col, score_sem2_col=score_sem2_col, score_sem3_col=score_sem3_col,
            FC_win_prob_col=FC_win_prob_col, BG_win_prob_col=BG_win_prob_col, FC_BG_tval_col=FC_BG_tval_col)


#%%
def parse_unitname(unitname):
    """parse unitname in directory name into netname layername, chan, x, y, RFrsz
    similar to the meta df format in the insilico experiment"""
    netname = unitname.split("_.")[0]
    RFrsz = unitname.endswith("_RFrsz")
    unitname_bare = unitname.replace(netname + "_", "").replace("_RFrsz", "")
    if ".SelectAdaptivePool2dglobal_pool" in unitname:
        layername = ".SelectAdaptivePool2dglobal_pool"
        chan = int(unitname_bare[len(layername):].split("_")[1])
        x = None
        y = None
    else:
        parts = unitname_bare.split("_")
        assert len(parts) == 4 or len(parts) == 2, f"unrecognized unitname {unitname}"
        layername = parts[0]
        chan = int(parts[1])
        if len(parts) == 4:
            x = int(parts[2])
            y = int(parts[3])
        elif len(parts) == 2:
            x = None
            y = None
        else:
            raise Exception(f"unrecognized unitname {unitname}")
    meta = edict(netname=netname, layername=layername, chan=chan, x=x, y=y, RFrsz=RFrsz, unitname=unitname)
    return meta

#%%
meta_info = []
for unitname in tqdm(unitlist):
    # parse unitname to get the layer and unit number 'resnet50_linf8_.layer4.Bottleneck2_29_4_4_RFrsz'
    # unitname = "resnet50_linf8_.layer4.Bottleneck2_29_4_4_RFrsz"
    meta = parse_unitname(unitname)
    meta_info.append(meta)

meta_info_df = pd.DataFrame(meta_info, columns=meta.keys())
meta_info_df.to_csv(join(figdir, "meta_info_df.csv"), index=False)
#%%
meta_info_df.groupby(by=["netname", "layername", "RFrsz"]).count()
#%%

def _shaded_errorbar(x, y, yerr, color, alpha, **kwargs):
    plt.plot(x, y, color=color, **kwargs)
    plt.fill_between(x, y - yerr, y + yerr, color=color, alpha=alpha)


def _shaded_errorbar_arr(arr, color="r", alpha=0.3, errtype="sem", **kwargs):
    x = np.arange(arr.shape[1])
    y = np.nanmean(arr, axis=0)
    if errtype == "sem":
        yerr = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
    elif errtype == "std":
        yerr = np.nanstd(arr, axis=0)
    plt.plot(x, y, color=color, **kwargs)
    plt.fill_between(x, y - yerr, y + yerr, color=color, alpha=alpha,
                     label="" if "label" not in kwargs else kwargs["label"]+"_"+errtype)
#%%
normalizer = np.nanquantile(np.concatenate([score_mean1_col, score_mean2_col, score_mean3_col], axis=1)
                            [:, :, -5:], 0.95, axis=(1,2))
# replace 0.0 as np.nan
normalizer[normalizer == 0.0] = np.nan
 #np.maximum(normalizer, 1e-5)
normscore_mean1_col = score_mean1_col / normalizer[:, None, None]
normscore_mean2_col = score_mean2_col / normalizer[:, None, None]
normscore_mean3_col = score_mean3_col / normalizer[:, None, None]

#%% Layer-wise plot
for (netname, layername, RFrsz), df in meta_info_df.groupby(by=["netname", "layername", "RFrsz"]):
    print(netname, layername, RFrsz)
    # print(df)

    plt.figure()
    plt.plot(normscore_mean1_col[df.index].reshape(-1,100).T, alpha=0.03, color="b", label=None)
    _shaded_errorbar_arr(normscore_mean1_col[df.index].reshape(-1,100), color="b", alpha=0.3, label="DeePSim",)
    plt.plot(normscore_mean2_col[df.index].reshape(-1,100).T, alpha=0.03, color="r", label=None)
    _shaded_errorbar_arr(normscore_mean2_col[df.index].reshape(-1,100), color="r", alpha=0.3, label="BG CholCMA")
    plt.plot(normscore_mean3_col[df.index].reshape(-1,100).T, alpha=0.03, color="g", label=None)
    _shaded_errorbar_arr(normscore_mean3_col[df.index].reshape(-1,100), color="g", alpha=0.3, label="BG HessCMA")
    plt.title(f"{netname} {layername} {'RF resize' if RFrsz else ''}")
    plt.ylabel("Normalized Activation")
    plt.xlabel("Blocks")
    plt.legend()#["DeePSim", "BG CholCMA", "BG HessCMA"])
    saveallforms(figdir, f"{netname}_{layername}{'_RFrsz' if RFrsz else ''}_normact_cmp")
    plt.show()

    plt.figure()
    _shaded_errorbar_arr(normscore_mean1_col[df.index].reshape(-1, 100), color="b", alpha=0.3, errtype="std", label="DeePSim",)
    _shaded_errorbar_arr(normscore_mean2_col[df.index].reshape(-1, 100), color="r", alpha=0.3, errtype="std", label="BG CholCMA")
    _shaded_errorbar_arr(normscore_mean3_col[df.index].reshape(-1, 100), color="g", alpha=0.3, errtype="std", label="BG HessCMA")
    plt.title(f"{netname} {layername} {'RF resize' if RFrsz else ''}")
    plt.ylabel("Normalized Activation")
    plt.xlabel("Blocks")
    plt.legend()#["DeePSim", "BG CholCMA", "BG HessCMA"])
    saveallforms(figdir, f"{netname}_{layername}{'_RFrsz' if RFrsz else ''}_normact_cmp_std")
    plt.show()
    # raise Exception
    # plt.figure()
    # plt.plot(BG_win_prob_col[df.index].T, alpha=0.3, color="k")
    # _shaded_errorbar_arr(BG_win_prob_col[df.index], color="r", alpha=0.3, )
    # plt.title(f"{netname} {layername} {'RF resize' if RFrsz else ''}")
    # plt.ylabel("BG win prob")
    # plt.xlabel("Blocks")
    # plt.ylim([0, 1])
    # saveallforms(figdir, f"{netname}_{layername}{'_RFrsz' if RFrsz else ''}_BG_win_prob")
    # plt.show()

    # plt.figure()
    # plt.plot(FC_BG_tval_col[df.index].T, alpha=0.3, color="k")
    # _shaded_errorbar_arr(FC_BG_tval_col[df.index], color="r", alpha=0.3, )
    # plt.title(f"{netname} {layername} {'RF resize' if RFrsz else ''}")
    # plt.ylabel("FC-BG tval")
    # plt.xlabel("Blocks")
    # saveallforms(figdir, f"{netname}_{layername}{'_RFrsz' if RFrsz else ''}_FC_BG_tval")
    # plt.show()
    # raise Exception

#%% Merged plot of all layers for each network
netname ='resnet50_linf8'# 'resnet50' # 'resnet50_linf8'
meta_info_df[meta_info_df.netname==netname].layername.unique()
layernames = ['.layer1.Bottleneck1', '.layer2.Bottleneck3',
       '.layer3.Bottleneck5', '.layer4.Bottleneck2', '.Linearfc']
RFrsz = False
figh, axs = plt.subplots(1, len(layernames), figsize=(len(layernames)*3.5, 4))
for li, layername in enumerate(layernames):
    df = meta_info_df[(meta_info_df.netname==netname) & (meta_info_df.layername==layername)
                      & (meta_info_df.RFrsz==False)]
    plt.sca(axs[li])
    _shaded_errorbar_arr(normscore_mean1_col[df.index].reshape(-1, 100), color="b", alpha=0.3, errtype="std",
                         label="DeePSim", )
    _shaded_errorbar_arr(normscore_mean2_col[df.index].reshape(-1, 100), color="r", alpha=0.3, errtype="std",
                         label="BG CholCMA")
    _shaded_errorbar_arr(normscore_mean3_col[df.index].reshape(-1, 100), color="g", alpha=0.3, errtype="std",
                         label="BG HessCMA")
    plt.title(f"{netname} {layername} {'RF resize' if RFrsz else ''}")
    plt.ylabel("Normalized Activation")
    plt.xlabel("Blocks")
    if li==len(layernames)-1:
        plt.legend()  # ["DeePSim", "BG CholCMA", "BG HessCMA"])
plt.suptitle(f"{netname} {'RF resize' if RFrsz else ''} Average Normalized Optimization trajectory")
plt.tight_layout()
saveallforms(figdir, f"{netname}_{'_RFrsz' if RFrsz else ''}_merged_normact_cmp_std", figh)
figh.show()
for ax in axs:
    ax.set_xlim([0, 40])
    # ax.autoscale(enable=True, axis='y', tight=True)
    ax.autoscale_view(tight=True, scalex=False, scaley=True)
saveallforms(figdir, f"{netname}_{'_RFrsz' if RFrsz else ''}_merged_normact_cmp_std_xlim40", figh)
figh.show()
#%%
netname = 'tf_efficientnet_b6'# "tf_efficientnet_b6_ap" # 'tf_efficientnet_b6'
meta_info_df[meta_info_df.netname==netname].layername.unique()
layernames = ['.blocks.0', '.blocks.1', '.blocks.2', '.blocks.3', '.blocks.4',
       '.blocks.5', '.blocks.6','.SelectAdaptivePool2dglobal_pool',
       '.Linearclassifier']
RFrsz = False

figh, axs = plt.subplots(1, len(layernames), figsize=(len(layernames)*3.5, 4))
for li, layername in enumerate(layernames):
    df = meta_info_df[(meta_info_df.netname==netname) & (meta_info_df.layername==layername)
                      & (meta_info_df.RFrsz==False)]
    plt.sca(axs[li])
    _shaded_errorbar_arr(normscore_mean1_col[df.index].reshape(-1, 100), color="b", alpha=0.3, errtype="std",
                         label="DeePSim", )
    _shaded_errorbar_arr(normscore_mean2_col[df.index].reshape(-1, 100), color="r", alpha=0.3, errtype="std",
                         label="BG CholCMA")
    _shaded_errorbar_arr(normscore_mean3_col[df.index].reshape(-1, 100), color="g", alpha=0.3, errtype="std",
                         label="BG HessCMA")
    plt.title(f"{netname} {layername} {'RF resize' if RFrsz else ''}")
    plt.ylabel("Normalized Activation")
    plt.xlabel("Blocks")
    if li==len(layernames)-1:
        plt.legend()  # ["DeePSim", "BG CholCMA", "BG HessCMA"])
plt.suptitle(f"{netname} {'RF resize' if RFrsz else ''} Average Normalized Optimization trajectory")
plt.tight_layout()
saveallforms(figdir, f"{netname}_{'_RFrsz' if RFrsz else ''}_merged_normact_cmp_std", figh)
figh.show()
for ax in axs:
    ax.set_xlim([0, 40])
    # ax.autoscale(enable=True, axis='y', tight=True)
    ax.autoscale_view(tight=True, scalex=False, scaley=True)
saveallforms(figdir, f"{netname}_{'_RFrsz' if RFrsz else ''}_merged_normact_cmp_std_xlim40", figh)
figh.show()
#%%
netname = 'resnet50_linf8' # 'resnet50_linf8'
for netname in [ 'resnet50_linf8', 'resnet50']:
    meta_info_df[meta_info_df.netname==netname].layername.unique()
    layernames = ['.layer1.Bottleneck1', '.layer2.Bottleneck3',
                  '.layer3.Bottleneck5', '.layer4.Bottleneck2', '.Linearfc']
    RFrsz = False
    figh, axs = plt.subplots(1, len(layernames), figsize=(len(layernames)*3.5, 4))
    for li, layername in enumerate(layernames):
        df = meta_info_df[(meta_info_df.netname==netname) & (meta_info_df.layername==layername)
                          & (meta_info_df.RFrsz==False)]
        plt.sca(axs[li])
        plt.plot(BG_win_prob_col[df.index].T, alpha=0.3, color="k")
        _shaded_errorbar_arr(BG_win_prob_col[df.index], color="r", alpha=0.3, )
        plt.title(f"{netname} {layername} {'RF resize' if RFrsz else ''}")
        plt.ylabel("BG win prob")
        plt.xlabel("Blocks")
        plt.ylim([0, 1])
        if li == len(layernames) - 1:
            plt.legend()  # ["DeePSim", "BG CholCMA", "BG HessCMA"])
    plt.suptitle(f"{netname} {'RF resize' if RFrsz else ''} BigGAN win probability")
    plt.tight_layout()
    saveallforms(figdir, f"{netname}_{'_RFrsz' if RFrsz else ''}_merged_BG_winprob", figh)
    figh.show()
    for ax in axs:
        ax.set_xlim([0, 40])
        # ax.autoscale(enable=True, axis='y', tight=True)
        ax.autoscale_view(tight=True, scalex=False, scaley=True)
    saveallforms(figdir, f"{netname}_{'_RFrsz' if RFrsz else ''}_merged_BG_winprob_xlim40", figh)
    figh.show()
#%%
for netname in ["tf_efficientnet_b6", 'tf_efficientnet_b6_ap']:
    meta_info_df[meta_info_df.netname==netname].layername.unique()
    layernames = ['.blocks.0', '.blocks.1', '.blocks.2', '.blocks.3', '.blocks.4',
                   '.blocks.5', '.blocks.6','.SelectAdaptivePool2dglobal_pool',
                   '.Linearclassifier']
    RFrsz = False
    figh, axs = plt.subplots(1, len(layernames), figsize=(len(layernames)*3.5, 4))
    for li, layername in enumerate(layernames):
        df = meta_info_df[(meta_info_df.netname==netname) & (meta_info_df.layername==layername)
                          & (meta_info_df.RFrsz==False)]
        plt.sca(axs[li])
        plt.plot(BG_win_prob_col[df.index].T, alpha=0.3, color="k")
        _shaded_errorbar_arr(BG_win_prob_col[df.index], color="r", alpha=0.3, )
        plt.title(f"{netname} {layername} {'RF resize' if RFrsz else ''}")
        plt.ylabel("BG win prob")
        plt.xlabel("Blocks")
        plt.ylim([0, 1])
        if li == len(layernames) - 1:
            plt.legend()  # ["DeePSim", "BG CholCMA", "BG HessCMA"])
    plt.suptitle(f"{netname} {'RF resize' if RFrsz else ''} BigGAN win probability")
    plt.tight_layout()
    saveallforms(figdir, f"{netname}_{'_RFrsz' if RFrsz else ''}_merged_BG_winprob", figh)
    figh.show()
    for ax in axs:
        ax.set_xlim([0, 40])
        # ax.autoscale(enable=True, axis='y', tight=True)
        ax.autoscale_view(tight=True, scalex=False, scaley=True)
    saveallforms(figdir, f"{netname}_{'_RFrsz' if RFrsz else ''}_merged_BG_winprob_xlim40", figh)
    figh.show()

#%%