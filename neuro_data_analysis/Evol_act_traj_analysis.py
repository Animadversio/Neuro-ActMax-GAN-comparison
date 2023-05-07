"""Trajectory analysis
Devoted to compare the trajectory of BigGAN vs DeePSim, see how many blocks can BigGAN surpass DeePSim.
"""

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
#%%

def extract_all_evol_trajectory(BFEStats, ):
    """Extract the evolution trajectory of all the experiments in the BFEStats list into
    an dictionary of arrays. and a meta data dataframe
    """
    resp_col = OrderedDict()
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

    meta_df = pd.DataFrame.from_dict(meta_col, orient="index")
    return resp_col, meta_df


def pad_resp_traj(resp_col):
    """
    Pad the response trajectories to the same length by extrapolating the last block with the mean of last two blocks
    And then stack them together into a 3D array
    """
    # get the length of the longest trajectory
    max_len = max([resp_bunch.shape[0] for resp_bunch in resp_col.values()])
    # extrapolate the last block with the mean of last two blocks
    resp_extrap_col = OrderedDict()  # use OrderedDict instead of list to keep Expi as key
    extrap_mask_col = OrderedDict()
    for Expi, resp_bunch in resp_col.items():
        # resp_bunch: number of blocks x 6
        n_blocks = resp_bunch.shape[0]
        if n_blocks < max_len:
            extrap_vals = resp_bunch[-2:, :].mean(axis=0)
            resp_bunch = np.concatenate([resp_bunch,
                 np.tile(extrap_vals, (max_len - n_blocks, 1))], axis=0)
        resp_extrap_col[Expi] = resp_bunch
        extrap_mask_col[Expi] = np.concatenate([np.ones(n_blocks), np.zeros(max_len - n_blocks)]).astype(bool)

    # concatenate all trajectories
    resp_extrap_arr = np.stack([*resp_extrap_col.values()], axis=0)
    extrap_mask_arr = np.stack([*extrap_mask_col.values()], axis=0)
    # resp_extrap_arr: n_exp x n_blocks x 6,
    #       values order: resp_m_traj_0, resp_m_traj_1, resp_sem_traj_0, resp_sem_traj_1, bsl_m_traj_0, bsl_m_traj_1
    # extrap_mask_arr: n_exp x n_blocks
    return resp_extrap_arr, extrap_mask_arr, max_len


def get_all_masks(meta_df):
    """
    Get all the masks for different conditions in the analysis
    :param meta_df:
    :return:
    """
    # plot the FC and BG win block number as
    Amsk  = meta_df.Animal == "Alfa"
    Bmsk  = meta_df.Animal == "Beto"
    V1msk = meta_df.visual_area == "V1"
    V4msk = meta_df.visual_area == "V4"
    ITmsk = meta_df.visual_area == "IT"
    length_msk = (meta_df.blockN > 14)
    spc_msk = (meta_df.space1 == "fc6") & meta_df.space2.str.contains("BigGAN")
    sucsmsk = (meta_df.p_maxinit_0 < 0.05) | (meta_df.p_maxinit_1 < 0.05)
    baseline_jump_list = ["Beto-18082020-002",
                          "Beto-07092020-006",
                          "Beto-14092020-002",
                          "Beto-27102020-003",
                          "Alfa-22092020-003",
                          "Alfa-04092020-003"]
    bsl_unstable_msk = meta_df.ephysFN.str.contains("|".join(baseline_jump_list), case=True, regex=True)
    assert bsl_unstable_msk.sum() == len(baseline_jump_list)
    bsl_stable_msk = ~bsl_unstable_msk
    # valid experiments are those with enough blocks, stable baseline and correct fc6-BigGAN pairing
    validmsk = length_msk & bsl_stable_msk & spc_msk
    # print summary of the inclusion criteria
    print("total number of experiments: %d" % len(meta_df))
    print("total number of valid experiments: %d" % validmsk.sum())
    print("total number of valid experiments with suc: %d" % (validmsk & sucsmsk).sum())
    print("Exluded:")
    print("  - short: %d" % (~length_msk).sum())
    print("  - unstable baseline: %d" % bsl_unstable_msk.sum())
    print("  - not fc6-BigGAN: %d" % (~spc_msk).sum())
    return Amsk, Bmsk, V1msk, V4msk, ITmsk, length_msk, spc_msk, sucsmsk, bsl_unstable_msk, bsl_stable_msk, validmsk

#%%
_, BFEStats = load_neural_data()
resp_col, meta_df = extract_all_evol_trajectory(BFEStats, )
resp_extrap_arr, extrap_mask_arr, max_len = pad_resp_traj(resp_col)
#%%
Amsk, Bmsk, V1msk, V4msk, ITmsk, \
    length_msk, spc_msk, sucsmsk, \
    bsl_unstable_msk, bsl_stable_msk, validmsk = get_all_masks(meta_df)
#%%
# directory to save all the figures related to activation trajectory
figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_traj_synopsis_py"
#%%
""" Trajectory synopsis with all areas """
normresp_extrap_arr = resp_extrap_arr / resp_extrap_arr[:, :, 0:2].max(axis=(1, 2), keepdims=True)
#%%
figh, axs = plt.subplots(2, 3, figsize=(9, 6))
for rowi, (msk_major, label_major) in enumerate(zip([Amsk, Bmsk], ["A", "B"])):
    for colj, (msk_minor, lable_minor) in enumerate(zip([V1msk, V4msk, ITmsk],
                                              ["V1", "V4", "IT"])):
        msk = msk_major & msk_minor & validmsk & sucsmsk
        axs[rowi, colj].plot(normresp_extrap_arr[msk, :, 0].T, color="blue", alpha=0.2, lw=0.7)
        axs[rowi, colj].plot(normresp_extrap_arr[msk, :, 1].T, color="red", alpha=0.2, lw=0.7)
        mean_trace_FC = normresp_extrap_arr[msk, :, 0].mean(axis=0)
        sem_trace_FC = normresp_extrap_arr[msk, :, 0].std(axis=0) / np.sqrt(msk.sum())
        mean_trace_BG = normresp_extrap_arr[msk, :, 1].mean(axis=0)
        sem_trace_BG = normresp_extrap_arr[msk, :, 1].std(axis=0) / np.sqrt(msk.sum())
        axs[rowi, colj].plot(mean_trace_FC, color="blue", lw=3)
        axs[rowi, colj].fill_between(np.arange(len(mean_trace_FC)),
                                        mean_trace_FC-sem_trace_FC,
                                        mean_trace_FC+sem_trace_FC,
                                        color="blue", alpha=0.25)
        axs[rowi, colj].plot(mean_trace_BG, color="red", lw=3)
        axs[rowi, colj].fill_between(np.arange(len(mean_trace_BG)),
                                        mean_trace_BG-sem_trace_BG,
                                        mean_trace_BG+sem_trace_BG,
                                        color="red", alpha=0.25)
        axs[rowi, colj].set_title(f"{label_major} {lable_minor} (N={msk.sum()})")

for ax in axs.ravel():
    ax.set_xlim([0, 40])

plt.suptitle("Max Normalized response across blocks")
plt.tight_layout()
saveallforms(figdir, "maxnorm_resp_traj_val_succ_area_anim_sep", figh=figh)
plt.show()


#%%
"""Compute the fraction of blocks that FC is significantly larger than BG"""
blocknum_arr = np.arange(1, max_len+1)[None, :]
FC_win_blks = (resp_extrap_arr[:, :, 0] - resp_extrap_arr[:, :, 1]) > 2 * np.sqrt(resp_extrap_arr[:, :, 2]**2 + resp_extrap_arr[:, :, 3]**2)
BG_win_blks = (resp_extrap_arr[:, :, 1] - resp_extrap_arr[:, :, 0]) > 2 * np.sqrt(resp_extrap_arr[:, :, 2]**2 + resp_extrap_arr[:, :, 3]**2)
BG_win_blk_num = (BG_win_blks * extrap_mask_arr).sum(axis=1)
FC_win_blk_num = (FC_win_blks * extrap_mask_arr).sum(axis=1)
FC_win_avg_blk = (FC_win_blks * blocknum_arr * extrap_mask_arr).sum(axis=1) / (FC_win_blks * extrap_mask_arr).sum(axis=1)
BG_win_avg_blk = (BG_win_blks * blocknum_arr * extrap_mask_arr).sum(axis=1) / (BG_win_blks * extrap_mask_arr).sum(axis=1)
#%%
winblk_meta_df = meta_df.copy()
winblk_meta_df["FC_win_blk_num"] = FC_win_blk_num
winblk_meta_df["BG_win_blk_num"] = BG_win_blk_num
winblk_meta_df["FC_win_blk_prop"] = FC_win_blk_num / meta_df.blockN
winblk_meta_df["BG_win_blk_prop"] = BG_win_blk_num / meta_df.blockN
winblk_meta_df["FC_win_blk_prop"] = winblk_meta_df["FC_win_blk_prop"].astype(float)
winblk_meta_df["BG_win_blk_prop"] = winblk_meta_df["BG_win_blk_prop"].astype(float)
winblk_meta_df["FC_win_avg_blk"] = FC_win_avg_blk
winblk_meta_df["BG_win_avg_blk"] = BG_win_avg_blk
#%%
# plot the FC and BG win block number as a function of the visual area
# use different marker symbols for the two animals
fig, ax = plt.subplots(figsize=[4, 5])
sns.swarmplot(data=winblk_meta_df[validmsk & sucsmsk],
              x="visual_area", y="BG_win_blk_num", hue="Animal", ax=ax,
              alpha=0.6, order=["V1", "V4", "IT"], size=4)
ax.set_title("Number of blocks where\nBigGAN > DeePSim")
ax.set_ylabel("Number of blocks")
saveallforms(figdir, "BG_win_blk_num_swarm", fig, ["svg", "png", "pdf"])
fig.show()
#%%
# plot the FC and BG win block number as a function of the visual area
fig, ax = plt.subplots(figsize=[4.5, 5])
sns.swarmplot(data=winblk_meta_df[validmsk & sucsmsk],
              x="visual_area", y="BG_win_blk_prop", hue="Animal", ax=ax,
              alpha=0.6, order=["V1", "V4", "IT"], size=4)
ax.set_title("Proportion of blocks where\nBigGAN > DeePSim")
ax.set_ylabel("Proportion of blocks")
saveallforms(figdir, "BG_win_blk_prop_swarm", fig, ["svg", "png", "pdf"])
fig.show()
#%%
# plot the FC and BG win block number as a function of the visual area
fig, ax = plt.subplots(figsize=[4.5, 5])
sns.swarmplot(data=winblk_meta_df[validmsk & sucsmsk],
              x="visual_area", y="FC_win_blk_prop", hue="Animal", ax=ax,
              alpha=0.6, order=["V1", "V4", "IT"], size=4)
ax.set_title("Proportion of blocks where\nDeePSim > BigGAN")
ax.set_ylabel("Proportion of blocks")
saveallforms(figdir, "FC_win_blk_prop_swarm", fig, ["svg", "png", "pdf"])
fig.show()
#%%
fig, ax = plt.subplots(figsize=[4.5, 5])
sns.swarmplot(data=winblk_meta_df[validmsk & sucsmsk],
              x="visual_area", y="BG_win_avg_blk", hue="Animal", ax=ax,
              alpha=0.6, order=["V1", "V4", "IT"])
ax.set_title("Average block number where\nDeePSim > BigGAN")
ax.set_ylabel("Avg Block Number")
saveallforms(figdir, "BG_win_avg_blkid_swarm", fig, ["svg", "png", "pdf"])
fig.show()
#%%
from core.utils.stats_utils import ttest_ind_print_df
for msk_common in [validmsk & sucsmsk & Amsk, validmsk & sucsmsk & Bmsk]:
    ttest_ind_print_df(winblk_meta_df, msk_common & ITmsk, msk_common & V4msk, "BG_win_blk_num")
    ttest_ind_print_df(winblk_meta_df, msk_common & ITmsk, msk_common & V4msk, "BG_win_blk_prop")
    ttest_ind_print_df(winblk_meta_df, msk_common & V4msk, msk_common & V1msk, "BG_win_blk_num")
    ttest_ind_print_df(winblk_meta_df, msk_common & V4msk, msk_common & V1msk, "BG_win_blk_prop")

#%%
#TODO: integrate this beta function into the CI computation
# import numpy as np
from scipy.stats import beta

# Generate some random 0,1 data
data = np.random.randint(0, 2, size=100)

# Calculate the number of successes (1's) and total observations (n)
k = np.sum(data)
n = len(data)

# Calculate the 95% confidence interval using the Clopper-Pearson method
alpha = 0.05
lower_ci, upper_ci = beta.interval(1-alpha, k+1, n-k+1)
#%%
"""Plot the win rate of BG and FC as a function of the visual area and the block number"""
errtype = "sem"  # "beta_CI" #
figh, axh = plt.subplots(1, 3, figsize=[8, 3], )
for mi, msk in enumerate([validmsk & sucsmsk & V1msk,
                          validmsk & sucsmsk & V4msk,
                          validmsk & sucsmsk & ITmsk]):
    plt.sca(axh[mi])
    nExps, nblocks = BG_win_blks[msk, :].shape
    BG_winrate_traj = BG_win_blks[msk, :].mean(axis=0)
    BG_winrate_traj_std = BG_win_blks[msk, :].std(axis=0)
    BG_winrate_traj_sem = BG_win_blks[msk, :].std(axis=0) / np.sqrt(BG_win_blks[msk, :].shape[0])
    # better errorbars for a bunch of 0,1 obersvations using beta function
    BG_win_CI1, BG_win_CI2 = beta.interval(0.95, BG_win_blks[msk, :].sum(axis=0) + 1,
                      len(BG_win_blks[msk, :]) - BG_win_blks[msk, :].sum(axis=0) + 1)
    FC_winrate_traj = FC_win_blks[msk, :].mean(axis=0)
    FC_winrate_traj_std = FC_win_blks[msk, :].std(axis=0)
    FC_winrate_traj_sem = FC_win_blks[msk, :].std(axis=0) / np.sqrt(FC_win_blks[msk, :].shape[0])
    # better errorbars for a bunch of 0,1 obersvations using beta function
    FC_win_CI1, FC_win_CI2 = beta.interval(0.95, FC_win_blks[msk, :].sum(axis=0) + 1,
                      len(FC_win_blks[msk, :]) - FC_win_blks[msk, :].sum(axis=0) + 1)
    plt.plot(np.arange(1, nblocks + 1), BG_winrate_traj, label="BigGAN > DeePSiM", color="r")
    if errtype == "sem":
        plt.fill_between(np.arange(1, nblocks + 1), BG_winrate_traj - BG_winrate_traj_sem,
                            BG_winrate_traj + BG_winrate_traj_sem, alpha=0.3, color="r")
    elif errtype == "beta_CI":
        plt.fill_between(np.arange(1, nblocks + 1), BG_win_CI1, BG_win_CI2,
                         alpha=0.3, color="r")
    plt.plot(np.arange(1, nblocks + 1), FC_winrate_traj, label="DeePSiM > BigGAN", color="b")
    if errtype == "sem":
        plt.fill_between(np.arange(1, nblocks + 1), FC_winrate_traj - FC_winrate_traj_sem,
                            FC_winrate_traj + FC_winrate_traj_sem, alpha=0.3, color="b")
    elif errtype == "beta_CI":
        plt.fill_between(np.arange(1, nblocks + 1), FC_win_CI1, FC_win_CI2,
                     alpha=0.3, color="b")
    plt.title(meta_df.loc[msk, "visual_area"].unique()[0])
    plt.xlabel("Block Number")
    plt.ylim([0, 1])
    if mi == 2:
        plt.legend()
    if mi == 0:
        plt.ylabel("Fraction of experiments")
plt.tight_layout()
saveallforms(figdir, f"both_winrate_traj_area_sep_{errtype}", figh, ["svg", "png", "pdf"])
plt.show()

