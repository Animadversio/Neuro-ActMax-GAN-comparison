
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

def compute_psth_stats(psth_arr, thresh=0.5):
    """
    psth_arr: N x 200
    """
    peak_latency = np.argmax(psth_arr, axis=1)
    resp_latency = np.argmax(psth_arr > thresh, axis=1)
    return peak_latency, resp_latency


def compute_center_of_mass(arr, time_vec=None, axis=-1, thresh=0):
    """Compute the center of mass of the array along the given axis.
    Args:
        arr: shape (n_exp, n_blocks, n_timepoint)
        time_vec: shape (n_timepoint,)
    return
        com: shape (n_exp, n_blocks)
    """
    if time_vec is None:
        time_vec = np.arange(arr.shape[axis])
    arr_trunc = np.clip(arr - thresh, 0, None)
    CoM_arr = (arr_trunc * time_vec).sum(axis=axis) / arr_trunc.sum(axis=axis)
    return CoM_arr
#%%
_, BFEStats = load_neural_data()
psth_col, _ = extract_all_evol_trajectory_psth(BFEStats)
psth_extrap_arr, extrap_mask_arr, max_len = pad_psth_traj(psth_col)
#%%
tabdir = Path(r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables")
meta_df = pd.read_csv(tabdir / "meta_activation_stats.csv", index_col=0)
Amsk, Bmsk, V1msk, V4msk, ITmsk, length_msk, spc_msk, \
    sucsmsk, bsl_unstable_msk, bsl_stable_msk, validmsk = get_all_masks(meta_df)
#%%
thresh = 0.05  # 0.01
anysucsmsk = (meta_df.p_maxinit_0 < thresh) | (meta_df.p_maxinit_1 < thresh)
bothsucmsk = (meta_df.p_maxinit_0 < thresh) & (meta_df.p_maxinit_1 < thresh)
FCsucsmsk = (meta_df.p_maxinit_0 < thresh)
BGsucsmsk = (meta_df.p_maxinit_1 < thresh)
#%%
latency_tab = []
for i, Expi in enumerate(meta_df.index):
    bsl_mean = psth_extrap_arr[i, :, 0, :50].mean()
    bsl_std = psth_extrap_arr[i, :, 0, :50].std()
    resp_thresh = bsl_mean + 2*bsl_std
    FC_peak_lat, FC_resp_lat = compute_psth_stats(psth_extrap_arr[i, :, 0, :], thresh=resp_thresh)
    BG_peak_lat, BG_resp_lat = compute_psth_stats(psth_extrap_arr[i, :, 1, :], thresh=resp_thresh)
    FC_CoM = compute_center_of_mass(psth_extrap_arr[i, :, 0, :], axis=-1, thresh=0)
    FC_CoM_bsl = compute_center_of_mass(psth_extrap_arr[i, :, 0, :], axis=-1,
                                        thresh=psth_extrap_arr[i, :, 0, :50].mean())#axis=-1))#[:,None]
    BG_CoM = compute_center_of_mass(psth_extrap_arr[i, :, 1, :], axis=-1, thresh=0)
    BG_CoM_bsl = compute_center_of_mass(psth_extrap_arr[i, :, 1, :], axis=-1,
                                        thresh=psth_extrap_arr[i, :, 1, :50].mean())#axis=-1))#[:,None]
    part_tab = pd.DataFrame({'FC_peak_lat': FC_peak_lat, 'FC_resp_lat': FC_resp_lat,
                             'BG_peak_lat': BG_peak_lat, 'BG_resp_lat': BG_resp_lat,
                             'FC_CoM': FC_CoM, 'BG_CoM': BG_CoM,
                             'FC_CoM_bsl': FC_CoM_bsl, 'BG_CoM_bsl': BG_CoM_bsl})
    part_tab['Expi'] = Expi
    part_tab["block"] = np.arange(psth_extrap_arr.shape[1])
    part_tab["non_extrap"] = extrap_mask_arr[i, :]
    latency_tab.append(part_tab)

latency_tab = pd.concat(latency_tab, axis=0)
latency_meta_tab = latency_tab.merge(meta_df, on='Expi')
#%%
latency_meta_tab.to_csv(tabdir / "Evol_psth_latency_stats.csv")
#%%
figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_psth_quantify"
#%%
figh, axs = plt.subplots(1,2,figsize=[7.5,4], sharey=True)
validmsk_full = latency_meta_tab.Expi.isin(meta_df[validmsk].index)
sns.lineplot(data=latency_meta_tab[validmsk_full], x='block', y='FC_peak_lat', hue='visual_area',
             errorbar="se", n_boot=0, linestyle="-", hue_order=['V4'], ax=axs[0], palette="Blues") # 'V1',
sns.lineplot(data=latency_meta_tab[validmsk_full], x='block', y='BG_peak_lat', hue='visual_area',
             errorbar="se", n_boot=0, linestyle="-", hue_order=['V4'], ax=axs[0], palette="Reds") # 'V1',
sns.lineplot(data=latency_meta_tab[validmsk_full], x='block', y='FC_peak_lat', hue='visual_area',
             errorbar="se", n_boot=0, linestyle="-", hue_order=['IT'], ax=axs[1], palette="Blues") # 'V1',
sns.lineplot(data=latency_meta_tab[validmsk_full], x='block', y='BG_peak_lat', hue='visual_area',
             errorbar="se", n_boot=0, linestyle="-", hue_order=['IT'], ax=axs[1], palette="Reds") # 'V1',
axs[0].set_title("V4")
axs[1].set_title("IT")
axs[0].set_ylabel("Peak Latency")
plt.suptitle("Peak Latency comparison between V4 and IT DeePSim vs BigGAN")
saveallforms(figdir, 'Evol_psth_peak_latency_area_sep', )
plt.show()

#%%
figh, axs = plt.subplots(1,2,figsize=[7.5,4], sharey=True)
validmsk_full = latency_meta_tab.Expi.isin(meta_df[validmsk].index)
sns.lineplot(data=latency_meta_tab[validmsk_full], x='block', y='FC_CoM', hue='visual_area',
             errorbar="se", n_boot=0, linestyle="-", hue_order=['V4'], ax=axs[0], palette="Blues") # 'V1',
sns.lineplot(data=latency_meta_tab[validmsk_full], x='block', y='BG_CoM', hue='visual_area',
             errorbar="se", n_boot=0, linestyle="-", hue_order=['V4'], ax=axs[0], palette="Reds") # 'V1',
sns.lineplot(data=latency_meta_tab[validmsk_full], x='block', y='FC_CoM', hue='visual_area',
             errorbar="se", n_boot=0, linestyle="-", hue_order=['IT'], ax=axs[1], palette="Blues") # 'V1',
sns.lineplot(data=latency_meta_tab[validmsk_full], x='block', y='BG_CoM', hue='visual_area',
             errorbar="se", n_boot=0, linestyle="-", hue_order=['IT'], ax=axs[1], palette="Reds") # 'V1',
axs[0].set_title("V4")
axs[1].set_title("IT")
axs[0].set_ylabel("Center of Mass")
plt.suptitle("PSTH Center of Mass comparison between V4 and IT DeePSim vs BigGAN")
saveallforms(figdir, 'Evol_psth_CoM_area_sep', )
plt.show()
#%%
figh, axs = plt.subplots(1,2,figsize=[7.5,4], sharey=True)
validmsk_full = latency_meta_tab.Expi.isin(meta_df[validmsk].index)
sns.lineplot(data=latency_meta_tab[validmsk_full], x='block', y='FC_CoM_bsl', hue='visual_area',
             errorbar="se", n_boot=0, linestyle="-", hue_order=['V4'], ax=axs[0], palette="Blues") # 'V1',
sns.lineplot(data=latency_meta_tab[validmsk_full], x='block', y='BG_CoM_bsl', hue='visual_area',
             errorbar="se", n_boot=0, linestyle="-", hue_order=['V4'], ax=axs[0], palette="Reds") # 'V1',
sns.lineplot(data=latency_meta_tab[validmsk_full], x='block', y='FC_CoM_bsl', hue='visual_area',
             errorbar="se", n_boot=0, linestyle="-", hue_order=['IT'], ax=axs[1], palette="Blues") # 'V1',
sns.lineplot(data=latency_meta_tab[validmsk_full], x='block', y='BG_CoM_bsl', hue='visual_area',
             errorbar="se", n_boot=0, linestyle="-", hue_order=['IT'], ax=axs[1], palette="Reds") # 'V1',
axs[0].set_title("V4")
axs[1].set_title("IT")
axs[0].set_ylabel("Center of Mass (Baseline subtracted)")
plt.suptitle("PSTH Center of Mass comparison between V4 and IT DeePSim vs BigGAN")
saveallforms(figdir, 'Evol_psth_CoM_bsl_area_sep', )
plt.show()
#%% statistics on the latency
from contextlib import redirect_stdout

with redirect_stdout((Path(tabdir)/'Evol_psth_quantify_stats.txt').open('w')):
    for stat_suffix in ["_CoM_bsl", "_peak_lat"]: #
        print(f"Statistics on {stat_suffix}")
        print("DeePSim vs BigGAN")
        print("[V4]")
        fullmsk = latency_meta_tab.Expi.isin(meta_df[validmsk & V4msk].index) & \
                  latency_meta_tab.non_extrap
        ttest_rel_print(latency_meta_tab[fullmsk]["FC"+stat_suffix],
                        latency_meta_tab[fullmsk]["BG"+stat_suffix], sem=True)
        print("[IT]")
        fullmsk = latency_meta_tab.Expi.isin(meta_df[validmsk & ITmsk].index) & \
                  latency_meta_tab.non_extrap
        ttest_rel_print(latency_meta_tab[fullmsk]["FC"+stat_suffix],
                        latency_meta_tab[fullmsk]["BG"+stat_suffix], sem=True)
        print("[IT] Any success only")
        fullmsk = latency_meta_tab.Expi.isin(meta_df[validmsk & ITmsk & anysucsmsk].index) & \
                  latency_meta_tab.non_extrap
        ttest_rel_print(latency_meta_tab[fullmsk]["FC"+stat_suffix],
                        latency_meta_tab[fullmsk]["BG"+stat_suffix], sem=True)
        print("[IT] Both success only")
        fullmsk = latency_meta_tab.Expi.isin(meta_df[validmsk & ITmsk & bothsucmsk].index) & \
                  latency_meta_tab.non_extrap
        ttest_rel_print(latency_meta_tab[fullmsk]["FC" + stat_suffix],
                        latency_meta_tab[fullmsk]["BG" + stat_suffix], sem=True)

        print("[IT] (block 0 only)")
        fullmsk = latency_meta_tab.Expi.isin(meta_df[validmsk & ITmsk].index) & \
                  (latency_meta_tab.block == 0)
        ttest_rel_print(latency_meta_tab[fullmsk]["FC" + stat_suffix],
                        latency_meta_tab[fullmsk]["BG" + stat_suffix], sem=True)
        print("[IT] (final block only)")
        fullmsk = latency_meta_tab.Expi.isin(meta_df[validmsk & ITmsk].index) & \
                  (latency_meta_tab.block == latency_meta_tab.blockN)
        ttest_rel_print(latency_meta_tab[fullmsk]["FC" + stat_suffix],
                        latency_meta_tab[fullmsk]["BG" + stat_suffix], sem=True)
        print("")
#%%
plt.figure(figsize=[5,5])
sns.lineplot(data=latency_meta_tab, x='block', y='FC_CoM', hue='visual_area',
             errorbar="se", n_boot=0, linestyle="-", hue_order=['V4', 'IT']) # 'V1',
sns.lineplot(data=latency_meta_tab, x='block', y='BG_CoM', hue='visual_area',
             errorbar="se", n_boot=0, linestyle=":", hue_order=['V4', 'IT']) # 'V1',
axs[0].set_ylabel("Center of Mass")
plt.title("Center of Mass")
plt.show()
#%%
"""This is not quite robust... """
plt.figure()
sns.lineplot(data=latency_meta_tab, x='block', y='FC_resp_lat', hue='visual_area',
             errorbar="se", n_boot=0, linestyle="-", hue_order=['V4', 'IT']) #'V1',
sns.lineplot(data=latency_meta_tab, x='block', y='BG_resp_lat', hue='visual_area',
             errorbar="se", n_boot=0, linestyle=":", hue_order=['V4', 'IT']) #'V1',
plt.ylabel("response latency")
plt.title("response latency")
plt.show()
#%%
bin_size = 10
for thread in range(2):
    psth_arr_thread = psth_extrap_arr[:, :, thread, :]
    CoM_arr_thread = compute_center_of_mass(psth_arr_thread, time_vec=np.arange(200) + 0.5, axis=-1)
    CoM_last_block = CoM_arr_thread[:, -1]
    CoM_init_block = CoM_arr_thread[:, 0]
    CoM_max_block = np.take_along_axis(CoM_arr_thread, max_block[:, None], axis=1).squeeze(axis=1)
