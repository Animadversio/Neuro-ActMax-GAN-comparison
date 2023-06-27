"""Attribute the increased activation in the Evolution to the different part of the psth."""

import os
from os.path import join
from pathlib import Path
from collections import OrderedDict
from easydict import EasyDict as edict
import torch
import seaborn as sns
from matplotlib import cm
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, ttest_ind, ttest_1samp, ttest_rel
from core.utils.plot_utils import saveallforms, show_imgrid
from neuro_data_analysis.neural_data_utils import parse_meta, area_mapping, get_all_masks
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data, get_expstr, \
    extract_evol_activation_array, extract_evol_psth_array, extract_all_evol_trajectory_psth, pad_psth_traj
from core.utils.stats_utils import ttest_rel_print, ttest_ind_print

#%%
tabdir = Path(r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables")
meta_df = pd.read_csv(tabdir / "meta_activation_stats.csv", index_col=0)
Amsk, Bmsk, V1msk, V4msk, ITmsk, length_msk, spc_msk, \
    sucsmsk, bsl_unstable_msk, bsl_stable_msk, validmsk = get_all_masks(meta_df)
#%%
bothsucmsk = (meta_df.p_maxinit_0 < 0.01) & (meta_df.p_maxinit_1 < 0.01)
FCsucsmsk = (meta_df.p_maxinit_0 < 0.01)
BGsucsmsk = (meta_df.p_maxinit_1 < 0.01)
#%%
_, BFEStats = load_neural_data()
psth_col, meta_df = extract_all_evol_trajectory_psth(BFEStats)
psth_extrap_arr, extrap_mask_arr, max_len = pad_psth_traj(psth_col)
# shape of psth_extrap_arr is (n_exp, n_blocks, 4, n_timepoint)
# shape of extrap_mask_arr is (n_exp, n_blocks)
#%%
figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_activation_psth_attribution"

#%%
def mean_time_bin(psth, bin_size=5):
    """Bin the psth in time axis and return the mean of each bin."""
    n_timepoint = psth.shape[-1]
    n_bin = n_timepoint // bin_size
    psth_bin = psth[..., :n_bin*bin_size].reshape(psth.shape[:-1] + (n_bin, bin_size))
    return psth_bin.mean(axis=-1)


def compute_center_of_mass(arr, time_vec, axis=-1):
    """Compute the center of mass of the array along the given axis.
    Args:
        arr: shape (n_exp, n_blocks, n_timepoint)
        time_vec: shape (n_timepoint,)
    return
        com: shape (n_exp, n_blocks)
    """
    if time_vec is None:
        time_vec = np.arange(arr.shape[axis])
    CoM_arr = (arr * time_vec).sum(axis=axis) / arr.sum(axis=axis)
    return CoM_arr

plt.switch_backend("agg")
plt.switch_backend("module://backend_interagg")
#%%
# get the block of max activation in the evolution and compare it to the first block
# mean_act_arr = psth_extrap_arr[:, :, :2, 50:].mean(axis=-1)
diff_attrib_tsr = []
diff_attrib_norm_tsr = []
diff_attrib_sem_tsr = []
diff_attrib_bin_tsr = []
diff_attrib_norm_bin_tsr = []
diff_attrib_sem_bin_tsr = []
diff_attrib_CoM_tsr = []
CoM_last_block_tsr = []
CoM_init_block_tsr = []
CoM_max_block_tsr = []
stat_dict = []
bin_size = 10
for thread in range(2):
    mean_act_arr_thread = psth_extrap_arr[:, :, thread, 50:].mean(axis=-1)
    max_block = mean_act_arr_thread.argmax(axis=1)
    # gather the psth of the max block, gather the psth of the first block
    max_psth_mean = np.take_along_axis(psth_extrap_arr[:, :, thread+0, :], max_block[:, None, None], axis=1).squeeze(axis=1)
    max_psth_sem  = np.take_along_axis(psth_extrap_arr[:, :, thread+2, :], max_block[:, None, None], axis=1).squeeze(axis=1)
    first_psth_mean = psth_extrap_arr[:, 0, thread+0, :]
    first_psth_sem  = psth_extrap_arr[:, 0, thread+2, :]
    diff_attrib = max_psth_mean - first_psth_mean
    diff_attrib_norm = diff_attrib / diff_attrib.sum(axis=1, keepdims=True)
    diff_attrib_sem = np.sqrt((max_psth_sem**2 + first_psth_sem**2) / 2)
    #TODO: better way to do the sem uncertainty.

    psth_arr_thread = psth_extrap_arr[:, :, thread, :]
    CoM_arr_thread = compute_center_of_mass(psth_arr_thread, time_vec=np.arange(200) + 0.5, axis=-1)
    CoM_last_block = CoM_arr_thread[:, -1]
    CoM_init_block = CoM_arr_thread[:, 0]
    CoM_max_block = np.take_along_axis(CoM_arr_thread, max_block[:, None], axis=1).squeeze(axis=1)

    max_psth_mean_bin = mean_time_bin(max_psth_mean, bin_size=bin_size)
    max_psth_sem_bin = mean_time_bin(max_psth_sem, bin_size=bin_size)
    first_psth_mean_bin = mean_time_bin(first_psth_mean, bin_size=bin_size)
    first_psth_sem_bin = mean_time_bin(first_psth_sem, bin_size=bin_size)
    diff_attrib_bin = max_psth_mean_bin - first_psth_mean_bin
    diff_attrib_norm_bin = diff_attrib_bin / diff_attrib_bin.sum(axis=1, keepdims=True)
    diff_attrib_sem_bin = np.sqrt((max_psth_sem_bin**2 + first_psth_mean_bin**2) / 2)
    diff_attrib_CoM = compute_center_of_mass(diff_attrib_bin[:,5:], time_vec=np.arange(0, 200, bin_size)[5:]+bin_size/2, axis=-1)

    diff_attrib_tsr.append(diff_attrib)
    diff_attrib_norm_tsr.append(diff_attrib_norm)
    diff_attrib_sem_tsr.append(diff_attrib_sem)
    diff_attrib_bin_tsr.append(diff_attrib_bin)
    diff_attrib_norm_bin_tsr.append(diff_attrib_norm_bin)
    diff_attrib_sem_bin_tsr.append(diff_attrib_sem_bin)
    diff_attrib_CoM_tsr.append(diff_attrib_CoM)
    CoM_last_block_tsr.append(CoM_last_block)
    CoM_init_block_tsr.append(CoM_init_block)
    CoM_max_block_tsr.append(CoM_max_block)
    # stat_dict.append({
    #     "diff_attrib": diff_attrib,
    #     "diff_attrib_norm": diff_attrib_norm,
    #     "diff_attrib_sem": diff_attrib_sem,
    #     "diff_attrib_bin": diff_attrib_bin,
    #     "diff_attrib_norm_bin": diff_attrib_norm_bin,
    #     "diff_attrib_sem_bin": diff_attrib_sem_bin,
    #     "diff_attrib_CoM": diff_attrib_CoM,
    #     "CoM_last_block": CoM_last_block,
    #     "CoM_init_block": CoM_init_block,
    #     "CoM_max_block": CoM_max_block,
    # })



diff_attrib_tsr = np.stack(diff_attrib_tsr, axis=-1)
diff_attrib_norm_tsr = np.stack(diff_attrib_norm_tsr, axis=-1)
diff_attrib_sem_tsr = np.stack(diff_attrib_sem_tsr, axis=-1)
diff_attrib_bin_tsr = np.stack(diff_attrib_bin_tsr, axis=-1)
diff_attrib_norm_bin_tsr = np.stack(diff_attrib_norm_bin_tsr, axis=-1)
diff_attrib_sem_bin_tsr = np.stack(diff_attrib_sem_bin_tsr, axis=-1)
diff_attrib_CoM_tsr = np.stack(diff_attrib_CoM_tsr, axis=-1)
CoM_last_block_tsr = np.stack(CoM_last_block_tsr, axis=-1)
CoM_init_block_tsr = np.stack(CoM_init_block_tsr, axis=-1)
CoM_max_block_tsr = np.stack(CoM_max_block_tsr, axis=-1)

#%% Plot the difference attribution
#%%
figh, axs = plt.subplots(1, 3, figsize=[12, 4], sharex=True, sharey=True, )
for i, (visual_area, mask1) in enumerate(zip(["V1", "V4", "IT"], [V1msk, V4msk, ITmsk])):
    ax = axs[i]
    mask = mask1 & validmsk & bothsucmsk
    diff_attrib = diff_attrib_norm_tsr[mask].mean(axis=0)
    diff_attrib_sem = diff_attrib_norm_tsr[mask].std(axis=0) / np.sqrt(mask.sum())
    ax.plot(diff_attrib[:, 0], color='b')
    ax.plot(diff_attrib[:, 1], color='r')
    # ax.fill_between(np.arange(0, 200), diff_attrib.mean(axis=0) - diff_attrib_sem, diff_attrib.mean(axis=0) + diff_attrib_sem, color='r', alpha=0.3)
    ax.set_title(visual_area + f" N={mask.sum()}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Activation difference")
    # ax.set_ylim([-0.1, 0.1])
    ax.set_xlim([0, 200])
plt.show()
#%%
# scatter plot of the difference attribution center of mass
# figh, axs = plt.subplots(1, 3, figsize=[12, 4], sharex=True, sharey=True)
# for i, (visual_area, mask1) in enumerate(zip(["V1", "V4", "IT"], [V1msk, V4msk, ITmsk])):
figh, axs = plt.subplots(1, 2, figsize=[8, 4.5], sharex=True, sharey=True)
for i, (visual_area, mask1) in enumerate(zip(["V4", "IT"], [V4msk, ITmsk])):
    ax = axs[i]
    mask = mask1 & validmsk & bothsucmsk
    ax.scatter(diff_attrib_CoM_tsr[mask, 0], diff_attrib_CoM_tsr[mask, 1], s=25)
    ax.set_xlabel("CoM of act diff DeePSim")
    ax.set_ylabel("CoM of act diff BigGAN")
    ax.axline([100, 100], slope=1)
    tval, pval, statstr = ttest_rel_print(diff_attrib_CoM_tsr[mask, 0],
                                  diff_attrib_CoM_tsr[mask, 1], sem=True)
    ax.set_title(visual_area + f" N={mask.sum()}\n" + statstr.replace("(N", "\n(N"))
    # ax.axis('equal')
    # ax.set_xlim([0, 200])
    # ax.set_ylim([0, 200])
    ax.set_aspect('equal', 'box')
plt.tight_layout()
saveallforms(figdir, "scatter_PSTH_Center_of_Mass_bothsuc", )
plt.show()
#%%
target_tsr = CoM_init_block_tsr
for target_tsr, label in zip([CoM_init_block_tsr, CoM_last_block_tsr, CoM_max_block_tsr],
                                ["init", "last", "max"]):
    for success_msk, success_str in zip([bothsucmsk, sucsmsk, ~sucsmsk],
                                        ["bothsuc", "anysuc", "nonsucs"]):
        figh, axs = plt.subplots(1, 2, figsize=[8, 4.5], sharex=True, sharey=True)
        for i, (visual_area, area_mask) in enumerate(zip(["V4", "IT"], [V4msk, ITmsk])):
            ax = axs[i]
            mask = area_mask & validmsk & success_msk # & bothsucmsk
            ax.scatter(target_tsr[mask, 0], target_tsr[mask, 1], s=25, alpha=0.6)
            ax.set_xlabel("CoM of PSTH DeePSim")
            ax.set_ylabel("CoM of PSTH BigGAN")
            ax.axline([100, 100], slope=1, color="k", linestyle="--")
            tval, pval, statstr = ttest_rel_print(target_tsr[mask, 0],
                                          target_tsr[mask, 1], sem=True)
            ax.set_title(visual_area + f" N={mask.sum()}\n" + statstr.replace("(N", "\n(N"))
            ax.set_aspect('equal', 'box')
        plt.suptitle(f"Center of Mass of PSTH {label} block\n[{success_str} sessions]")
        plt.tight_layout()
        saveallforms(figdir, f"scatter_PSTH_{label}_block_Center_of_Mass_{success_str}", )#_bothsuc
        plt.show()

#%%
figh, axs = plt.subplots(1, 2, figsize=[8, 3.5], sharex=True, sharey=True)
for i, (visual_area, mask1) in enumerate(zip(["V4", "IT"], [V4msk, ITmsk])):
    ax = axs[i]
    mask = mask1 & validmsk & bothsucmsk
    diff_attrib = diff_attrib_norm_tsr[mask].mean(axis=0)
    diff_attrib_sem = diff_attrib_norm_tsr[mask].std(axis=0) / np.sqrt(mask.sum())
    ax.plot(diff_attrib[:, 0], color='b')
    ax.fill_between(np.arange(0, 200),
                    diff_attrib[:, 0] - diff_attrib_sem[:, 0],
                    diff_attrib[:, 0] + diff_attrib_sem[:, 0], color='b', alpha=0.3)
    ax.plot(diff_attrib[:, 1], color='r')
    ax.fill_between(np.arange(0, 200),
                    diff_attrib[:, 1] - diff_attrib_sem[:, 1],
                    diff_attrib[:, 1] + diff_attrib_sem[:, 1], color='r', alpha=0.3)
    ax.axhline(0, color='k', ls='--', lw=1)
    # ax.fill_between(np.arange(0, 200), diff_attrib.mean(axis=0) - diff_attrib_sem, diff_attrib.mean(axis=0) + diff_attrib_sem, color='r', alpha=0.3)
    ax.set_title(visual_area + f" N={mask.sum()}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Fraction of activation difference")
    # ax.set_ylim([-0.1, 0.1])
    ax.set_xlim([0, 200])
plt.tight_layout()
plt.show()
#%%
bin_size = 10
figh, axs = plt.subplots(1, 2, figsize=[7, 3.5], sharex=True, sharey=True)
time_ticks = np.arange(0, 200, bin_size) + bin_size / 2
for i, (visual_area, area_mask) in enumerate(zip(["V4", "IT"], [V4msk, ITmsk])):
    ax = axs[i]
    mask = area_mask & validmsk & bothsucmsk
    diff_attrib = diff_attrib_norm_bin_tsr[mask].mean(axis=0)
    diff_attrib_sem = diff_attrib_norm_bin_tsr[mask].std(axis=0) / np.sqrt(mask.sum())
    ax.plot(time_ticks, diff_attrib[:, 0], color='b')
    ax.fill_between(time_ticks,
                    diff_attrib[:, 0] - diff_attrib_sem[:, 0],
                    diff_attrib[:, 0] + diff_attrib_sem[:, 0], color='b', alpha=0.3)
    ax.plot(time_ticks, diff_attrib[:, 1], color='r')
    ax.fill_between(time_ticks,
                    diff_attrib[:, 1] - diff_attrib_sem[:, 1],
                    diff_attrib[:, 1] + diff_attrib_sem[:, 1], color='r', alpha=0.3)
    ax.axhline(0, color='k', ls='--', lw=1)
    # ax.fill_between(np.arange(0, 200), diff_attrib.mean(axis=0) - diff_attrib_sem, diff_attrib.mean(axis=0) + diff_attrib_sem, color='r', alpha=0.3)
    ax.set_title(visual_area + f" N={mask.sum()}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Fraction of activation difference")
    # ax.set_ylim([-0.1, 0.1])
    ax.set_xlim([0, 200])
plt.suptitle(f"Activation increase attributed to different {bin_size}ms time windows")
plt.tight_layout()
saveallforms(figdir, "act_increase_attrib_bothsucs", figh)
plt.show()

#%%
thread_colors = ['b', 'r']
figh, axs = plt.subplots(1, 3, figsize=[9, 3.5], sharex=True, sharey=True)
time_ticks = np.arange(0, 200, bin_size) + bin_size / 2
for i, (visual_area, area_mask) in enumerate(zip(["V1", "V4", "IT"], [V1msk, V4msk, ITmsk])):
    ax = axs[i]
    for thread, GANname, thread_sucsmsk in zip([0, 1],
                                               ["DeePSim", "BigGAN"],
                                               [FCsucsmsk, BGsucsmsk]):
        mask = area_mask & validmsk & thread_sucsmsk
        diff_attrib = diff_attrib_norm_bin_tsr[mask].mean(axis=0)
        diff_attrib_sem = diff_attrib_norm_bin_tsr[mask].std(axis=0) / np.sqrt(mask.sum())
        ax.plot(time_ticks, diff_attrib[:, thread], color=thread_colors[thread],
                label=f"{GANname} N={mask.sum()}")
        ax.fill_between(time_ticks,
                        diff_attrib[:, thread] - diff_attrib_sem[:, thread],
                        diff_attrib[:, thread] + diff_attrib_sem[:, thread], color=thread_colors[thread], alpha=0.3)
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set_title(visual_area)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Fraction of activation difference")
    ax.set_xlim([0, 200])
    # ax.set_ylim([-0.02, 0.09])
    ax.legend()
plt.suptitle(f"Activation increase attributed to different {bin_size}ms time windows")
plt.tight_layout()
saveallforms(figdir, "act_increase_attrib_threadsucs_3area", figh)
plt.show()

#%%