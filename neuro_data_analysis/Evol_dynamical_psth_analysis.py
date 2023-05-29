
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
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data, get_expstr, extract_evol_activation_array, extract_evol_psth_array
from neuro_data_analysis.neural_data_utils import parse_meta, area_mapping, get_all_masks
from os.path import join
from collections import OrderedDict
from easydict import EasyDict as edict
#%%
_, BFEStats = load_neural_data()
#%%
# def extract_all_evol_psth_dyna(BFEStats, ):
"""Extract the evolution trajectory of all the experiments in the BFEStats list into
an dictionary of arrays. and a meta data dataframe
"""
psth_col = OrderedDict()
meta_col = OrderedDict()
#%
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
    psth_col0, gen_arr0, psth_arr0, _ = extract_evol_psth_array(S, 0, )
    psth_col1, gen_arr1, psth_arr1, _ = extract_evol_psth_array(S, 1, )
    # psth_arr0 / 1: time x trial
    # psth_col0 / 1: list with len = generation number, each element is a 2d array of time x trial
    # gen_arr0 / 1: list with len = generation number, each element is a 1d array of trial number
    # if the lAST BLOCK has < 10 images, in either thread, then remove it
    if len(gen_arr0[-1]) < 10 or len(gen_arr1[-1]) < 10:
        psth_col0 = psth_col0[:-1]
        psth_col1 = psth_col1[:-1]
        gen_arr0 = gen_arr0[:-1]
        gen_arr1 = gen_arr1[:-1]
    assert len(gen_arr0) == len(gen_arr1)

    psth_m_traj_0 = np.array([resp.mean(axis=-1) for resp in psth_col0]) # generation x time
    psth_m_traj_1 = np.array([resp.mean(axis=-1) for resp in psth_col1]) # generation x time
    psth_sem_traj_0 = np.array([sem(resp, axis=-1) for resp in psth_col0]) # generation x time
    psth_sem_traj_1 = np.array([sem(resp, axis=-1) for resp in psth_col1]) # generation x time

    # # test the successfulness of the evolution
    # # ttest between the last two blocks and the first two blocks
    # t_endinit_0, p_endinit_0 = ttest_ind(np.concatenate(resp_arr0[-2:]), np.concatenate(resp_arr0[:2]))
    # t_endinit_1, p_endinit_1 = ttest_ind(np.concatenate(resp_arr1[-2:]), np.concatenate(resp_arr1[:2]))
    # # ttest between the max two blocks and the first two blocks
    # max_id0 = np.argmax(resp_m_traj_0)
    # max_id0 = max_id0 if max_id0 < len(resp_arr0) - 2 else len(resp_arr0) - 3
    # t_maxinit_0, p_maxinit_0 = ttest_ind(np.concatenate(resp_arr0[max_id0:max_id0+2]), np.concatenate(resp_arr0[:2]))
    # max_id1 = np.argmax(resp_m_traj_1)
    # max_id1 = max_id1 if max_id1 < len(resp_arr1) - 2 else len(resp_arr1) - 3
    # t_maxinit_1, p_maxinit_1 = ttest_ind(np.concatenate(resp_arr1[max_id1:max_id1+2]), np.concatenate(resp_arr1[:2]))
    #
    # t_FCBG_end_01, p_FCBG_end_01 = ttest_ind(np.concatenate(resp_arr0[-2:]), np.concatenate(resp_arr1[:2]))
    # t_FCBG_max_01, p_FCBG_max_01 = ttest_ind(np.concatenate(resp_arr0[max_id0:max_id0+2]), np.concatenate(resp_arr1[max_id1:max_id1+2]))

    # save the meta data
    meta_dict = edict(Animal=Animal, expdate=expdate, ephysFN=ephysFN, prefchan=prefchan, prefunit=prefunit,
                      visual_area=visual_area, space1=space1, space2=space2, blockN=len(gen_arr0))
    # stat_dict = edict(t_endinit_0=t_endinit_0, p_endinit_0=p_endinit_0,
    #                 t_endinit_1=t_endinit_1, p_endinit_1=p_endinit_1,
    #                 t_maxinit_0=t_maxinit_0, p_maxinit_0=p_maxinit_0,
    #                 t_maxinit_1=t_maxinit_1, p_maxinit_1=p_maxinit_1,
    #                 t_FCBG_end_01=t_FCBG_end_01, p_FCBG_end_01=p_FCBG_end_01,
    #                 t_FCBG_max_01=t_FCBG_max_01, p_FCBG_max_01=p_FCBG_max_01,)
    # meta_dict.update(stat_dict)

    # stack the trajectories together
    psth_bunch = np.stack([psth_m_traj_0, psth_m_traj_1,
                           psth_sem_traj_0, psth_sem_traj_1], axis=1)
    # resp_bunch: generation x thread [mean, sem]  x time
    psth_col[Expi] = psth_bunch
    # list of 3d array, each array is a ( generation x thread [mean, sem]  x time )
    meta_col[Expi] = meta_dict
    # raise Exception("Not finished yet")

meta_df = pd.DataFrame.from_dict(meta_col, orient="index")
# return psth_col, meta_df
#%%
def pad_psth_traj(psth_col):
    """
    Pad the response trajectories to the same length by extrapolating the last block with the mean of last two blocks
    And then stack them together into a 3D array
    """
    # get the length of the longest trajectory
    max_len = max([resp_bunch.shape[0] for resp_bunch in psth_col.values()])
    # extrapolate the last block with the mean of last two blocks
    psth_extrap_col = OrderedDict()  # use OrderedDict instead of list to keep Expi as key
    extrap_mask_col = OrderedDict()
    for Expi, psth_bunch in psth_col.items():
        # resp_bunch: number of blocks x 6
        n_blocks = psth_bunch.shape[0]
        if n_blocks < max_len:
            extrap_vals = psth_bunch[-2:, :, :].mean(axis=0)
            psth_bunch = np.concatenate([psth_bunch,
                                         np.tile(extrap_vals, (max_len - n_blocks, 1, 1))], axis=0)
        psth_extrap_col[Expi] = psth_bunch
        extrap_mask_col[Expi] = np.concatenate([np.ones(n_blocks), np.zeros(max_len - n_blocks)]).astype(bool)

    # concatenate all trajectories
    psth_extrap_arr = np.stack([*psth_extrap_col.values()], axis=0)
    extrap_mask_arr = np.stack([*extrap_mask_col.values()], axis=0)
    # psth_extrap_arr: n_exp x n_blocks x 4 x n_time,
    #       values order: psth_m_traj_0, psth_m_traj_1, psth_sem_traj_0, psth_sem_traj_1
    # extrap_mask_arr: n_exp x n_blocks
    return psth_extrap_arr, extrap_mask_arr, max_len


psth_extrap_arr, extrap_mask_arr, max_len = pad_psth_traj(psth_col)
#%%
dfdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_activation_cmp"
meta_df = pd.read_csv(join(dfdir, "meta_stats.csv"))
# meta_df = pd.DataFrame.from_dict(meta_col, orient="index")
Amsk, Bmsk, V1msk, V4msk, ITmsk, \
        length_msk, spc_msk, sucsmsk, \
        bsl_unstable_msk, bsl_stable_msk, validmsk = get_all_masks(meta_df)
#%%
normalizer = psth_extrap_arr[:, :, 0:2, 50:200].mean(axis=-1).max(axis=(1, 2))
norm_psth_extrap_arr = psth_extrap_arr / normalizer[:, None, None, None]
#%%
figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_dynam_psth_cmp"
os.makedirs(figdir, exist_ok=True)
# plot the trajectories
for block in range(max_len):
    figh, axs = plt.subplots(2, 3, figsize=(9, 6))
    for rowi, (msk_major, label_major) in enumerate(zip([Amsk, Bmsk], ["A", "B"])):
        for colj, (msk_minor, lable_minor) in enumerate(zip([V1msk, V4msk, ITmsk],
                                                  ["V1", "V4", "IT"])):
            msk = msk_major & msk_minor & validmsk & sucsmsk
            axs[rowi, colj].plot(norm_psth_extrap_arr[msk, block, 0, :].T, color="blue", alpha=0.2, lw=0.7, label=None)
            axs[rowi, colj].plot(norm_psth_extrap_arr[msk, block, 1, :].T, color="red", alpha=0.2, lw=0.7, label=None)
            mean_psth_FC = norm_psth_extrap_arr[msk, block, 0, :].mean(axis=0)
            sem_psth_FC = norm_psth_extrap_arr[msk, block, 0, :].std(axis=0) / np.sqrt(msk.sum())
            mean_psth_BG = norm_psth_extrap_arr[msk, block, 1, :].mean(axis=0)
            sem_psth_BG =norm_psth_extrap_arr[msk, block, 1, :].std(axis=0) / np.sqrt(msk.sum())
            axs[rowi, colj].plot(mean_psth_FC, color="blue", lw=3, label="DeePSim")
            axs[rowi, colj].fill_between(np.arange(len(mean_psth_FC)),
                                            mean_psth_FC-sem_psth_FC,
                                            mean_psth_FC+sem_psth_FC,
                                            color="blue", alpha=0.25, label=None)
            axs[rowi, colj].plot(mean_psth_BG, color="red", lw=3, label="BigGAN")
            axs[rowi, colj].fill_between(np.arange(len(mean_psth_BG)),
                                            mean_psth_BG-sem_psth_BG,
                                            mean_psth_BG+sem_psth_BG,
                                            color="red", alpha=0.25, label=None)
            axs[rowi, colj].set_title(f"{label_major} {lable_minor} (N={msk.sum()})")

    for ax in axs.ravel():
        # ax.set_xlim([0, 40])
        ax.set_ylim([0, 2.0])

    axs[0, 0].legend(loc="lower right", frameon=False)
    plt.suptitle(f"Max Normalized PSTH blocks {block} [Valid & Succ Sessions]")

    # plt.suptitle(f"Max Normalized response ({rsp_wdw[0]}-{rsp_wdw[-1]+1}ms window) across blocks [Valid & Succ Sessions]")
    plt.tight_layout()
    saveallforms([figdir], f"maxnorm_psth_traj_val_succ_area_anim_sep_block{block:02d}", figh=figh)
    plt.show()

#%%
for block in range(max_len):
    figh, axs = plt.subplots(2, 3, figsize=(9, 6))
    for rowi, (msk_major, label_major) in enumerate(zip([Amsk, Bmsk], ["A", "B"])):
        for colj, (msk_minor, lable_minor) in enumerate(zip([V1msk, V4msk, ITmsk],
                                                            ["V1", "V4", "IT"])):
            msk = msk_major & msk_minor & validmsk
            axs[rowi, colj].plot(norm_psth_extrap_arr[msk, block, 0, :].T, color="blue", alpha=0.2, lw=0.7, label=None)
            axs[rowi, colj].plot(norm_psth_extrap_arr[msk, block, 1, :].T, color="red", alpha=0.2, lw=0.7, label=None)
            mean_psth_FC = norm_psth_extrap_arr[msk, block, 0, :].mean(axis=0)
            sem_psth_FC = norm_psth_extrap_arr[msk, block, 0, :].std(axis=0) / np.sqrt(msk.sum())
            mean_psth_BG = norm_psth_extrap_arr[msk, block, 1, :].mean(axis=0)
            sem_psth_BG = norm_psth_extrap_arr[msk, block, 1, :].std(axis=0) / np.sqrt(msk.sum())
            axs[rowi, colj].plot(mean_psth_FC, color="blue", lw=3, label="DeePSim")
            axs[rowi, colj].fill_between(np.arange(len(mean_psth_FC)),
                                         mean_psth_FC - sem_psth_FC,
                                         mean_psth_FC + sem_psth_FC,
                                         color="blue", alpha=0.25, label=None)
            axs[rowi, colj].plot(mean_psth_BG, color="red", lw=3, label="BigGAN")
            axs[rowi, colj].fill_between(np.arange(len(mean_psth_BG)),
                                         mean_psth_BG - sem_psth_BG,
                                         mean_psth_BG + sem_psth_BG,
                                         color="red", alpha=0.25, label=None)
            axs[rowi, colj].set_title(f"{label_major} {lable_minor} (N={msk.sum()})")

    for ax in axs.ravel():
        # ax.set_xlim([0, 40])
        ax.set_ylim([0, 2.0])

    axs[0, 0].legend(loc="lower right", frameon=False)
    plt.suptitle(f"Max Normalized PSTH blocks {block} [Valid Sessions]")
    plt.tight_layout()
    saveallforms([figdir], f"maxnorm_psth_traj_val_area_anim_sep_block{block:02d}", figh=figh)
    plt.show()
# %%
#%%
# export the animations into gif and mp4
import imageio
from pathlib import Path
from tqdm import tqdm

frames_succ = []
frames_val = []
for block in range(max_len):
    frames_succ.append(imageio.imread(join(figdir,f"maxnorm_psth_traj_val_succ_area_anim_sep_block{block:02d}.png")))
    frames_val.append(imageio.imread(join(figdir,f"maxnorm_psth_traj_val_area_anim_sep_block{block:02d}.png")))

imageio.mimsave(join(figdir, "maxnorm_psth_traj_val_succ_area_anim_sep.gif"), frames_succ, fps=1)
imageio.mimsave(join(figdir, "maxnorm_psth_traj_val_succ_area_anim_sep.mp4"), frames_succ, fps=1)

imageio.mimsave(join(figdir, "maxnorm_psth_traj_val_area_anim_sep.gif"), frames_val, fps=1)
imageio.mimsave(join(figdir, "maxnorm_psth_traj_val_area_anim_sep.mp4"), frames_val, fps=1)


