import os

import numpy as np
import torch
import seaborn as sns
from scipy.stats import sem
from matplotlib import cm
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms, show_imgrid
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data, get_expstr, extract_evol_activation_array
from neuro_data_analysis.neural_data_utils import parse_meta, area_mapping
from os.path import join
outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_activation_cmp"
os.makedirs(outdir, exist_ok=True)
#%%
_, BFEStats = load_neural_data()

#%%
# data structure to contain a collection of trajectories
# each trajectory is an 1D array of length n_blocks
from collections import OrderedDict
from easydict import EasyDict as edict
from scipy.stats import ttest_ind, ttest_rel
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

#%% masks
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
validmsk = length_msk & bsl_stable_msk & spc_msk
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
from scipy.stats import ttest_ind, ttest_1samp
ttest_1samp(meta_df.loc[validmsk & V1msk, "t_FCBG_max_01"], 0)
#%%
ttest_ind(meta_df.loc[validmsk & V4msk, "t_FCBG_end_01"],
          meta_df.loc[validmsk & ITmsk, "t_FCBG_end_01"])
#%%
meta_df.loc[(meta_df.ephysFN == "Beto-16102020-003")].prefunit
