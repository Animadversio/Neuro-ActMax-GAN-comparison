import os
import torch
import seaborn as sns
from matplotlib import cm
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, ttest_ind, ttest_1samp, ttest_rel
from core.utils.plot_utils import saveallforms, show_imgrid
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data, get_expstr, extract_evol_activation_array
from neuro_data_analysis.neural_data_utils import parse_meta, area_mapping
from os.path import join
from collections import OrderedDict
from easydict import EasyDict as edict
outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_activation_cmp"
os.makedirs(outdir, exist_ok=True)
#%%
_, BFEStats = load_neural_data()

#%%
# data structure to contain a collection of trajectories
# each trajectory is an 1D array of length n_blocks
resp_col = OrderedDict()
meta_col = OrderedDict()
#%%
for Expi in range(1, len(BFEStats) + 1):
    S = BFEStats[Expi - 1]
    if S["evol"] is None:
        continue
    expstr = get_expstr(BFEStats, Expi)
    print(expstr)
    Animal, expdate = parse_meta(S)
    ephysFN = S["meta"]['ephysFN']
    prefchan = int(S['evol']['pref_chan'][0])
    prefunit = int(S['evol']['unit_in_pref_chan'][0])
    visual_area = area_mapping(prefchan, Animal, expdate)
    spacenames = S['evol']['space_names']
    space1 = spacenames[0] if isinstance(spacenames[0], str) else spacenames[0][0]
    space2 = spacenames[1] if isinstance(spacenames[1], str) else spacenames[1][0]

    # load the evolution trajectory of each pair
    resp_arr0, bsl_arr0, gen_arr0, _, _, _ = extract_evol_activation_array(S, 0)
    resp_arr1, bsl_arr1, gen_arr1, _, _, _ = extract_evol_activation_array(S, 1)

    # if the lAST BLOCK has < 10 images, in either thread, then remove it
    if len(resp_arr0[-1]) < 10 or len(resp_arr1[-1]) < 10:
        resp_arr0 = resp_arr0[:-1]
        resp_arr1 = resp_arr1[:-1]
        bsl_arr0 = bsl_arr0[:-1]
        bsl_arr1 = bsl_arr1[:-1]
        gen_arr0 = gen_arr0[:-1]
        gen_arr1 = gen_arr1[:-1]

    resp_m_traj_0 = np.array([resp.mean() for resp in resp_arr0])
    resp_m_traj_1 = np.array([resp.mean() for resp in resp_arr1])
    resp_sem_traj_0 = np.array([sem(resp) for resp in resp_arr0])
    resp_sem_traj_1 = np.array([sem(resp) for resp in resp_arr1])
    bsl_m_traj_0 = np.array([bsl.mean() for bsl in bsl_arr0])
    bsl_m_traj_1 = np.array([bsl.mean() for bsl in bsl_arr1])

    # test the successfulness of the evolution
    # ttest between the last two blocks and the first two blocks
    t_endinit_0, p_endinit_0 = ttest_ind(np.concatenate(resp_arr0[-2:]), np.concatenate(resp_arr0[:2]))
    t_endinit_1, p_endinit_1 = ttest_ind(np.concatenate(resp_arr1[-2:]), np.concatenate(resp_arr1[:2]))
    # ttest between the max two blocks and the first two blocks
    max_id0 = np.argmax(resp_m_traj_0)
    max_id0 = max_id0 if max_id0 < len(resp_arr0) - 2 else len(resp_arr0) - 3
    t_maxinit_0, p_maxinit_0 = ttest_ind(np.concatenate(resp_arr0[max_id0:max_id0+2]), np.concatenate(resp_arr0[:2]))
    max_id1 = np.argmax(resp_m_traj_1)
    max_id1 = max_id1 if max_id1 < len(resp_arr1) - 2 else len(resp_arr1) - 3
    t_maxinit_1, p_maxinit_1 = ttest_ind(np.concatenate(resp_arr1[max_id1:max_id1+2]), np.concatenate(resp_arr1[:2]))

    t_FCBG_end_01, p_FCBG_end_01 = ttest_ind(np.concatenate(resp_arr0[-2:]), np.concatenate(resp_arr1[:2]))
    t_FCBG_max_01, p_FCBG_max_01 = ttest_ind(np.concatenate(resp_arr0[max_id0:max_id0+2]), np.concatenate(resp_arr1[max_id1:max_id1+2]))

    # save the meta data
    meta_dict = edict(Animal=Animal, expdate=expdate, ephysFN=ephysFN, prefchan=prefchan, prefunit=prefunit,
                      visual_area=visual_area, space1=space1, space2=space2, blockN=len(resp_arr0))
    stat_dict = edict(t_endinit_0=t_endinit_0, p_endinit_0=p_endinit_0,
                    t_endinit_1=t_endinit_1, p_endinit_1=p_endinit_1,
                    t_maxinit_0=t_maxinit_0, p_maxinit_0=p_maxinit_0,
                    t_maxinit_1=t_maxinit_1, p_maxinit_1=p_maxinit_1,
                    t_FCBG_end_01=t_FCBG_end_01, p_FCBG_end_01=p_FCBG_end_01,
                    t_FCBG_max_01=t_FCBG_max_01, p_FCBG_max_01=p_FCBG_max_01,)
    meta_dict.update(stat_dict)

    # stack the trajectories together
    resp_bunch = np.stack([resp_m_traj_0, resp_m_traj_1,
                           resp_sem_traj_0, resp_sem_traj_1,
                           bsl_m_traj_0, bsl_m_traj_1, ], axis=1)
    resp_col[Expi] = resp_bunch
    meta_col[Expi] = meta_dict
#%%
# get the longest trajectory
max_len = max([resp_bunch.shape[0] for resp_bunch in resp_col.values()])
# extrapolate the last block with the mean of last two blocks
resp_extrap_col = OrderedDict()
for Expi, resp_bunch in resp_col.items():
    n_blocks = resp_bunch.shape[0]
    if n_blocks < max_len:
        extrap_vals = resp_bunch[-2:, :].mean(axis=0)
        resp_bunch = np.concatenate([resp_bunch,
             np.tile(extrap_vals, (max_len - n_blocks, 1))], axis=0)
    resp_extrap_col[Expi] = resp_bunch

# concatenate all trajectories
resp_extrap_arr = np.stack([*resp_extrap_col.values()], axis=0)
#%%
meta_df = pd.DataFrame.from_dict(meta_col, orient="index")
#%%
meta_df.to_csv(join(outdir, "meta_stats.csv"))
np.save(join(outdir, "resp_traj_extrap_arr.npy"), resp_extrap_arr)
pkl.dump({"resp_col": resp_col, "meta_col": meta_col}, open(join(outdir, "resp_traj_col.pkl"), "wb"))

#%%
normresp_extrap_arr = resp_extrap_arr / resp_extrap_arr[:, :, 0:2].max(axis=(1,2), keepdims=True)
#%% masks
Amsk  = meta_df.Animal == "Alfa"
Bmsk  = meta_df.Animal == "Beto"
V1msk = meta_df.visual_area == "V1"
V4msk = meta_df.visual_area == "V4"
ITmsk = meta_df.visual_area == "IT"
length_msk = (meta_df.blockN > 14)
spc_msk = (meta_df.space1 == "fc6") & meta_df.space2.str.contains("BigGAN")

baseline_jump_list = ["Beto-18082020-002",
                      "Beto-07092020-006",
                      "Beto-14092020-002",
                      "Beto-27102020-003",
                      "Alfa-22092020-003",
                      "Alfa-04092020-003"]
bsl_unstable_msk = meta_df.ephysFN.str.contains("|".join(baseline_jump_list), case=True, regex=True)
assert bsl_unstable_msk.sum() == len(baseline_jump_list)
bsl_stable_msk = ~bsl_unstable_msk
validmsk = length_msk & bsl_stable_msk & spc_msk
sucsmsk = (meta_df.p_maxinit_0 < 0.05) | (meta_df.p_maxinit_1 < 0.05)
#%%
# print summary of the inclusion criteria
print("total number of experiments: %d" % len(meta_df))
print("total number of valid experiments: %d" % validmsk.sum())
print("total number of valid experiments with suc: %d" % (validmsk & sucsmsk).sum())
print("Exluded:")
print("  - short: %d" % (~length_msk).sum())
print("  - unstable baseline: %d" % bsl_unstable_msk.sum())
print("  - not fc6-BigGAN: %d" % (~spc_msk).sum())
#%%
# use pycharm backend
import matplotlib

matplotlib.use("module://backend_interagg")
#%%
meta_df.loc[validmsk, "t_FCBG_max_01"].plot.hist(bins=20)
plt.show()
#%%
import seaborn as sns
# sns.set_context("talk")
# use frequency instead of count
for msk, label in [(validmsk, "val"),
                   (validmsk & sucsmsk, "val_suc"),
                   (validmsk & Amsk, "val_monkA"),
                   (validmsk & Bmsk, "val_monkB"),]:
    for common_norm in [False, True]:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=meta_df.loc[msk, :], x="t_FCBG_end_01", hue="visual_area", bins=25,
                    stat="density", common_norm=common_norm, alpha=0.25)
        sns.kdeplot(data=meta_df.loc[msk, :], x="t_FCBG_end_01", hue="visual_area",
                    common_norm=common_norm, lw=3, )
        plt.xlabel(f"tval (FC end resp vs BG end resp)")
        plt.title(f"FC end resp vs BG end resp   mask: {label}")
        saveallforms(outdir, f"tval_FCBG_end_01_hist_{label}{'' if common_norm else '_norm'}")
        plt.show()
#%%
for msk, label in [(validmsk, "val"),
                   (validmsk & sucsmsk, "val_suc"),
                   (validmsk & Amsk, "val_monkA"),
                   (validmsk & Bmsk, "val_monkB"),]:
    for common_norm in [False, True]:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=meta_df.loc[msk, :], x="t_FCBG_max_01", hue="visual_area", bins=25,
                    stat="density", common_norm=common_norm, alpha=0.25)
        sns.kdeplot(data=meta_df.loc[msk, :], x="t_FCBG_max_01", hue="visual_area",
                    common_norm=common_norm, lw=3, )
        plt.xlabel("tval (FC max resp vs BG max resp)")
        plt.title(f"FC max resp vs BG max resp   mask: {label}")
        saveallforms(outdir, f"tval_FCBG_max_01_hist_{label}{'' if common_norm else '_norm'}")
        plt.show()
#%%
# plotting utilities
def add_identity(ax, *line_args, **line_kwargs):
    """Add a 1:1 line to an axes."""
    identity, = ax.plot([], [], *line_args, **line_kwargs)

    def callback_lim_changed(ax):
        low_x, high_x = ax.get_xlim()
        low_y, high_y = ax.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback_lim_changed(ax)
    ax.callbacks.connect('xlim_changed', callback_lim_changed)
    ax.callbacks.connect('ylim_changed', callback_lim_changed)


def paired_strip_plot_simple(col1, col2, msk=None, col1_err=None, col2_err=None, ax=None,
                             offset=0, jitter_std=0.1):
    if msk is None:
        msk = np.ones(len(col1), dtype=bool)
    vec1 = col1[msk]
    vec2 = col2[msk]
    xjitter = jitter_std * np.random.randn(len(vec1))
    if ax is None:
        figh, ax = plt.subplots(1,1,figsize=[4, 6])
    else:
        figh = ax.figure
    ax.scatter(offset + xjitter, vec1, color="blue", alpha=0.3)
    ax.scatter(offset + xjitter+1, vec2, color="red", alpha=0.3)
    if col1_err is not None:
        ax.errorbar(offset + xjitter, vec1, yerr=col1_err[msk],
                    fmt="none", color="blue", alpha=0.3)
    if col2_err is not None:
        ax.errorbar(offset + xjitter+1, vec2, yerr=col2_err[msk],
                    fmt="none", color="red", alpha=0.3)
    ax.plot(offset + np.arange(2)[:, None]+xjitter[None, :],
             np.stack((vec1, vec2)), color="k", alpha=0.1)
    # plt.xticks([0,1], [col1, col2])
    tval, pval = ttest_rel(vec1, vec2)
    # tval, pval = ttest_rel_df(df, msk, col1, col2)
    ax.set_title(f"tval={tval:.3f}, pval={pval:.1e} N={msk.sum()}")
    # annotate the string on top of the plot
    ax.text(offset+0.5, 1.1, f"t={tval:.3f}, P={pval:.1e}\nN={msk.sum()}",
            ha='center', va='center')
    figh.show()
    return figh
#%%
outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_activation_cmp"
# msk_general, label_general = validmsk, "valid"
msk_general, label_general = validmsk & sucsmsk, "valsucs"
#%%
"""Paired strip plot of initial block activation, seperated into the three visual areas"""
figh, ax = plt.subplots(1, 1, figsize=[8, 6])
paired_strip_plot_simple(normresp_extrap_arr[:, 0, 0], normresp_extrap_arr[:, 0, 1],
             col1_err=normresp_extrap_arr[:, 0, 2], col2_err=normresp_extrap_arr[:, 0, 3],
             msk=V1msk & msk_general, ax=ax, offset=0, jitter_std=0.15)
paired_strip_plot_simple(normresp_extrap_arr[:, 0, 0], normresp_extrap_arr[:, 0, 1],
             col1_err=normresp_extrap_arr[:, 0, 2], col2_err=normresp_extrap_arr[:, 0, 3],
             msk=V4msk & msk_general, ax=ax, offset=2, jitter_std=0.15)
paired_strip_plot_simple(normresp_extrap_arr[:, 0, 0], normresp_extrap_arr[:, 0, 1],
             col1_err=normresp_extrap_arr[:, 0, 2], col2_err=normresp_extrap_arr[:, 0, 3],
             msk=ITmsk & msk_general, ax=ax, offset=4, jitter_std=0.15)
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xticklabels(["V1 DeePSim", "V1 BG", "V4 DeePSim", "V4 BG", "IT DeePSim", "IT BG"])
ax.set_ylabel("Max Normalized response")
figh.suptitle("Initial response comparison between DeePSim and BG across areas")
saveallforms(outdir, f"maxnorm_initresp_cmp_{label_general}_areasep", figh)
figh.show()

#%%
"""Paired strip plot of end block activation, seperated into the three visual areas"""
figh, ax = plt.subplots(1, 1, figsize=[8, 6])
paired_strip_plot_simple(normresp_extrap_arr[:, -1, 0], normresp_extrap_arr[:, -1, 1],
            col1_err=normresp_extrap_arr[:, -1, 2], col2_err=normresp_extrap_arr[:, -1, 3],
            msk=V1msk & msk_general, ax=ax, offset=0, jitter_std=0.15)
paired_strip_plot_simple(normresp_extrap_arr[:, -1, 0], normresp_extrap_arr[:, -1, 1],
            col1_err=normresp_extrap_arr[:, -1, 2], col2_err=normresp_extrap_arr[:, -1, 3],
            msk=V4msk & msk_general, ax=ax, offset=2, jitter_std=0.15)
paired_strip_plot_simple(normresp_extrap_arr[:, -1, 0], normresp_extrap_arr[:, -1, 1],
            col1_err=normresp_extrap_arr[:, -1, 2], col2_err=normresp_extrap_arr[:, -1, 3],
            msk=ITmsk & msk_general, ax=ax, offset=4, jitter_std=0.15)
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xticklabels(["V1 DeePSim", "V1 BG", "V4 DeePSim", "V4 BG", "IT DeePSim", "IT BG"])
ax.set_ylabel("Max Normalized response")
figh.suptitle("Last block response comparison between DeePSim and BG across areas")
saveallforms(outdir, f"maxnorm_endresp_cmp_{label_general}_areasep", figh)
figh.show()

#%%
FC_maxresp = normresp_extrap_arr[:, :, 0].max(axis=1)
FC_maxblkid = np.argmax(normresp_extrap_arr[:, :, 0], axis=1)
FC_maxresp_sem = np.take_along_axis(normresp_extrap_arr[:, :, 2],
                        FC_maxblkid[:, None], axis=1).squeeze()

BG_maxresp = normresp_extrap_arr[:, :, 1].max(axis=1)
BG_maxblkid = np.argmax(normresp_extrap_arr[:, :, 1], axis=1)
BG_maxresp_sem = np.take_along_axis(normresp_extrap_arr[:, :, 3],
                        BG_maxblkid[:, None], axis=1).squeeze()

figh, ax = plt.subplots(1, 1, figsize=[8, 6])
paired_strip_plot_simple(FC_maxresp, BG_maxresp,
            col1_err=FC_maxresp_sem, col2_err=BG_maxresp_sem,
            msk=V1msk & msk_general, ax=ax, offset=0, jitter_std=0.15)
paired_strip_plot_simple(FC_maxresp, BG_maxresp,
            col1_err=FC_maxresp_sem, col2_err=BG_maxresp_sem,
            msk=V4msk & msk_general, ax=ax, offset=2, jitter_std=0.15)
paired_strip_plot_simple(FC_maxresp, BG_maxresp,
            col1_err=FC_maxresp_sem, col2_err=BG_maxresp_sem,
            msk=ITmsk & msk_general, ax=ax, offset=4, jitter_std=0.15)
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xticklabels(["V1 DeePSim", "V1 BG", "V4 DeePSim", "V4 BG", "IT DeePSim", "IT BG"])
ax.set_ylabel("Max Normalized response")
figh.suptitle("Max block response comparison between DeePSim and BG across areas")
saveallforms(outdir, f"maxnorm_maxresp_cmp_{label_general}_areasep", figh)
figh.show()






#%%
figh = paired_strip_plot_simple(normresp_extrap_arr[:, :, 0].max(axis=1), normresp_extrap_arr[:, :, 1].max(axis=1),
                                ITmsk & validmsk &sucsmsk,
                                normresp_extrap_arr[:, -1, 2], normresp_extrap_arr[:, -1, 3])
figh.gca().set_xticks([0, 1])
figh.gca().set_xticklabels(["FC", "BG"])
figh.gca().set_ylabel("Normalized response")
figh.show()


#%%
msk, label = (validmsk, "val")
plt.figure(figsize=(6, 6))
# plt.scatter(resp_extrap_arr[msk, 0, 0], resp_extrap_arr[msk, 0, 1], alpha=0.25)
plt.scatter(resp_extrap_arr[msk, -1, 0] / resp_extrap_arr[msk, :, 0:2].max(axis=(1,2)),
            resp_extrap_arr[msk, -1, 1] / resp_extrap_arr[msk, :, 0:2].max(axis=(1,2)), alpha=0.25)
add_identity(plt.gca(), color="k", ls="--")
plt.axis("image")
plt.xlabel("FC end resp")
plt.ylabel("BG end resp")
plt.show()
#%%
figh = paired_strip_plot_simple(normresp_extrap_arr[:, -1, 0], normresp_extrap_arr[:, -1, 1],
                                ITmsk & validmsk)
figh.gca().set_xticks([0, 1])
figh.gca().set_xticklabels(["FC", "BG"])
figh.gca().set_ylabel("Normalized response")
figh.show()

#%%

#%%
ttest_1samp(meta_df.loc[validmsk & V1msk, "t_FCBG_max_01"], 0)
#%%
ttest_ind(meta_df.loc[validmsk & V4msk, "t_FCBG_end_01"],
          meta_df.loc[validmsk & ITmsk, "t_FCBG_end_01"])
#%%
meta_df.loc[(meta_df.ephysFN == "Beto-16102020-003")].prefunit
