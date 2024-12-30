import os
from os.path import join
from pathlib import Path
import torch
import seaborn as sns
from tqdm import tqdm, trange
from matplotlib import cm
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, ttest_ind, ttest_1samp, ttest_rel
from core.utils.stats_utils import ttest_ind_print_df, ttest_rel_print_df
from core.utils.plot_utils import saveallforms, show_imgrid
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data, get_expstr, \
    extract_evol_activation_array, extract_all_evol_trajectory, pad_resp_traj
from neuro_data_analysis.neural_data_utils import parse_meta, area_mapping, get_all_masks
from collections import OrderedDict
from easydict import EasyDict as edict
outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_activation_cmp"
tabdir = (r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables")

meta_df = pd.read_csv(Path(tabdir) / "meta_activation_stats_w_optimizer.csv")
Amsk, Bmsk, V1msk, V4msk, ITmsk, \
    length_msk, spc_msk, sucsmsk, \
    bsl_unstable_msk, bsl_stable_msk, validmsk = get_all_masks(meta_df)
FCsucsmsk = meta_df.p_maxinit_0 < 0.01
BGsucsmsk = meta_df.p_maxinit_1 < 0.01
bothsucsmsk = FCsucsmsk & BGsucsmsk
anysucsmsk = FCsucsmsk | BGsucsmsk
#%%
gprtrajdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_traj_GPregress"
resp_gpr_col = OrderedDict()
for Expi in trange(1, 191):
    if os.path.exists(join(gprtrajdir, f"Exp{Expi:03d}_evol_traj_gpr_avg.npz")):
        data = np.load(join(gprtrajdir, f"Exp{Expi:03d}_evol_traj_gpr_avg.npz"))
        resp_tsr = np.stack([data["traj_pred_mean0"], data["traj_pred_mean1"],
                             data['traj_pred_std0'],  data['traj_pred_std1']], axis=-1)
        resp_gpr_col[Expi] = resp_tsr
#%%
_, BFEStats = load_neural_data()
resp_col, _ = extract_all_evol_trajectory(BFEStats, )
# resp_extrap_arr, extrap_mask_arr, max_len = pad_resp_traj(resp_col)
#%%
# compute time constant from traj for two trajs
def compute_time_constant(traj, bsl=0, thresh=0.632):
    traj = traj - bsl
    thresh_val = thresh * traj.max()
    indices = np.where(traj > thresh_val)[0]
    n_gens = (traj < thresh_val).sum()
    if len(indices) == 0:
        return 0, n_gens
    else:
        return indices[0], n_gens

#%%
# compute time constant for all neurons
thresh = 0.632
timeconst_col = OrderedDict()
for Expi, resp_arr in tqdm(resp_gpr_col.items()):
    print(f"Processing {Expi}")
    # original baseline
    bsl0 = resp_col[Expi][:, 4].mean(axis=0)
    bsl1 = resp_col[Expi][:, 5].mean(axis=0)
    """compute time constant, different statistics
    * tc0: time constant from the first time point
    * tc_cnt: time constant from the first time point, total count in number of generations, bigger than tc0 (with 0-50ms fr baseline subtraction)
    * tc0_bsl0: time constant from the first time point, with no baseline subtraction (with 0-50ms fr baseline subtraction)
    * tc0_bslinit: time constant from the first time point, with baseline as initial generation rate
    """
    FC_tc0, FC_tc_cnt = compute_time_constant(resp_arr[:, 0], bsl=bsl0, thresh=thresh)  # thresh=0.632)
    BG_tc0, BG_tc_cnt = compute_time_constant(resp_arr[:, 1], bsl=bsl1, thresh=thresh)  # thresh=0.632)
    FC_tc0bsl0, FC_tc_cnt_bsl0 = compute_time_constant(resp_arr[:, 0], bsl=0, thresh=thresh)  # thresh=0.632)
    BG_tc0bsl0, BG_tc_cnt_bsl0 = compute_time_constant(resp_arr[:, 1], bsl=0, thresh=thresh)  # thresh=0.632)
    FC_tc0bslinit, FC_tc_cnt_bslinit = compute_time_constant(resp_arr[:, 0], bsl=resp_arr[0, 0], thresh=thresh)  # thresh=0.632)
    BG_tc0bslinit, BG_tc_cnt_bslinit = compute_time_constant(resp_arr[:, 1], bsl=resp_arr[0, 1], thresh=thresh)  # thresh=0.632)
    col = {"FC_tc0": FC_tc0, "FC_tc_cnt": FC_tc_cnt,
           "BG_tc0": BG_tc0, "BG_tc_cnt": BG_tc_cnt,
           "FC_tc0_bsl0": FC_tc0bsl0, "FC_tc_cnt_bsl0": FC_tc_cnt_bsl0,
           "BG_tc0_bsl0": BG_tc0bsl0, "BG_tc_cnt_bsl0": BG_tc_cnt_bsl0,
           "FC_tc0_bslinit": FC_tc0bslinit, "FC_tc_cnt_bslinit": FC_tc_cnt_bslinit,
           "BG_tc0_bslinit": BG_tc0bslinit, "BG_tc_cnt_bslinit": BG_tc_cnt_bslinit,}
    timeconst_col[Expi] = col

timeconst_df = pd.DataFrame(timeconst_col).T
# timeconst_df.to_csv(join(tabdir, "Evol_traj_time_constant_GPregress.csv"))
#%%
timeconst_meta_df = pd.merge(meta_df, timeconst_df, left_on="Expi", right_index=True,)
#%%
"""independent comparison of time constants between areas"""
print("Time constant of optimization trajectory")
print("BigGAN V4 vs IT [BigGAN successs]")
ttest_ind_print_df(timeconst_meta_df, validmsk&BGsucsmsk&V4msk, validmsk&BGsucsmsk&ITmsk, "BG_tc0")
ttest_ind_print_df(timeconst_meta_df, validmsk&BGsucsmsk&V4msk, validmsk&BGsucsmsk&ITmsk, "BG_tc_cnt")
ttest_ind_print_df(timeconst_meta_df, validmsk&BGsucsmsk&V4msk, validmsk&BGsucsmsk&ITmsk, "BG_tc0_bslinit")
ttest_ind_print_df(timeconst_meta_df, validmsk&BGsucsmsk&V4msk, validmsk&BGsucsmsk&ITmsk, "BG_tc_cnt_bslinit")
print("DeePSiM V4 vs IT [DeePSim success]")
ttest_ind_print_df(timeconst_meta_df, validmsk&FCsucsmsk&V4msk, validmsk&FCsucsmsk&ITmsk, "FC_tc0")  # yes V4 < IT
ttest_ind_print_df(timeconst_meta_df, validmsk&FCsucsmsk&V4msk, validmsk&FCsucsmsk&ITmsk, "FC_tc_cnt")  # yes V4 < IT
ttest_ind_print_df(timeconst_meta_df, validmsk&FCsucsmsk&V4msk, validmsk&FCsucsmsk&ITmsk, "FC_tc0_bslinit")
ttest_ind_print_df(timeconst_meta_df, validmsk&FCsucsmsk&V4msk, validmsk&FCsucsmsk&ITmsk, "FC_tc_cnt_bslinit")
print("DeePSim V1 vs V4 [DeePSim success]")
ttest_ind_print_df(timeconst_meta_df, validmsk&FCsucsmsk&V1msk, validmsk&FCsucsmsk&V4msk, "FC_tc0")
ttest_ind_print_df(timeconst_meta_df, validmsk&FCsucsmsk&V1msk, validmsk&FCsucsmsk&V4msk, "FC_tc_cnt")
ttest_ind_print_df(timeconst_meta_df, validmsk&FCsucsmsk&V1msk, validmsk&FCsucsmsk&V4msk, "FC_tc0_bslinit")
ttest_ind_print_df(timeconst_meta_df, validmsk&FCsucsmsk&V1msk, validmsk&FCsucsmsk&V4msk, "FC_tc_cnt_bslinit")

"""paired Compare time constants of FC and BG"""
print("Time constant of optimization trajectory")
print("IT cortex, FC vs BG [Both success]")
ttest_rel_print_df(timeconst_meta_df, validmsk&bothsucsmsk&ITmsk, "FC_tc0", "BG_tc0")  # yes FC > BG
ttest_rel_print_df(timeconst_meta_df, validmsk&bothsucsmsk&ITmsk, "FC_tc_cnt", "BG_tc_cnt")  # yes FC > BG
ttest_rel_print_df(timeconst_meta_df, validmsk&bothsucsmsk&ITmsk, "FC_tc0_bslinit", "BG_tc0_bslinit")  # yes FC > BG
ttest_rel_print_df(timeconst_meta_df, validmsk&bothsucsmsk&ITmsk, "FC_tc_cnt_bslinit", "BG_tc_cnt_bslinit")  # yes FC > BG
print("V4 cortex, FC vs BG [Both success]")
ttest_rel_print_df(timeconst_meta_df, validmsk&bothsucsmsk&V4msk, "FC_tc0", "BG_tc0")
ttest_rel_print_df(timeconst_meta_df, validmsk&bothsucsmsk&V4msk, "FC_tc_cnt", "BG_tc_cnt")
ttest_rel_print_df(timeconst_meta_df, validmsk&bothsucsmsk&V4msk, "FC_tc0_bslinit", "BG_tc0_bslinit")
ttest_rel_print_df(timeconst_meta_df, validmsk&bothsucsmsk&V4msk, "FC_tc_cnt_bslinit", "BG_tc_cnt_bslinit")
#%%
plt.figure(figsize=[6, 6])
# sns.scatterplot(data=timeconst_meta_df[validmsk], x="FC_tc0", y="FC_tc_cnt",
#                 hue="visual_area", style="visual_area", s=100, alpha=0.7)
# sns.scatterplot(data=timeconst_meta_df[validmsk], x="FC_tc_cnt_bslinit", y="BG_tc_cnt_bslinit",
#                 hue="visual_area", style="visual_area", s=100, alpha=0.7)
sns.scatterplot(data=timeconst_meta_df[validmsk], x="FC_tc_bslinit", y="BG_tc_bslinit",
                hue="visual_area", style="visual_area", s=100, alpha=0.7)
# plt.plot([0, 1], [0, 1], 'k--')
plt.axline([0, 0], [1, 1], ls='--', c='k')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
plt.show()
#%%
plt.figure(figsize=[4, 6])
sns.stripplot(data=timeconst_meta_df[sucsmsk], x="visual_area", y="FC_tc0",
              color="blue", alpha=0.4, order=["V1", "V4", "IT"], jitter=0.35)
sns.stripplot(data=timeconst_meta_df[sucsmsk], x="visual_area", y="BG_tc0",
              color="red", alpha=0.4, order=["V1", "V4", "IT"], jitter=0.35)
plt.show()

#%%
plt.figure(figsize=[4, 6])
sns.stripplot(data=timeconst_meta_df[FCsucsmsk&validmsk], x="visual_area", y="FC_tc0", order=["V1", "V4", "IT"],
              color="blue", alpha=0.4, jitter=0.25, label="DeePSim")
sns.stripplot(data=timeconst_meta_df[BGsucsmsk&validmsk], x="visual_area", y="BG_tc0", order=["V1", "V4", "IT"],
              color="red", alpha=0.4, jitter=0.25, label="BigGAN", dodge=True)
plt.suptitle("Time constant of optimization trajectory\n[Each succeed]")
plt.legend()
plt.show()

#%%
plt.figure(figsize=[4, 6])
sns.stripplot(data=timeconst_meta_df[bothsucsmsk&validmsk], x="visual_area", y="FC_tc_cnt", order=["V1", "V4", "IT"],
              color="blue", alpha=0.4, jitter=0.25, label="DeePSim")
sns.stripplot(data=timeconst_meta_df[bothsucsmsk&validmsk], x="visual_area", y="BG_tc_cnt", order=["V1", "V4", "IT"],
              color="red", alpha=0.4, jitter=0.25, label="BigGAN")
plt.suptitle("Time constant of optimization trajectory\n[Both succeed]")

plt.legend()
plt.show()


#%%
timeconst_meta_df_thread0 = pd.merge(meta_df, timeconst_df[["FC_tc0", "FC_tc1", "FC_tc00", "FC_tc10"]], left_on="Expi", right_index=True,)
timeconst_meta_df_thread1 = pd.merge(meta_df, timeconst_df[["BG_tc0", "BG_tc1", "BG_tc00", "BG_tc10"]], left_on="Expi", right_index=True,)
timeconst_meta_df_thread0["GANthread"] = meta_df.space1
# timeconst_meta_df_thread0["optimthread"] = meta_df.space1 #TODO:?
timeconst_meta_df_thread0["thread"] = 0
timeconst_meta_df_thread1["GANthread"] = meta_df.space2
timeconst_meta_df_thread1["thread"] = 1
timeconst_meta_df_thread0.rename(columns={"FC_tc0": "tc0", "FC_tc1": "tc1", "FC_tc00": "tc00", "FC_tc10": "tc10"
                                          }, inplace=True)
timeconst_meta_df_thread1.rename(columns={"BG_tc0": "tc0", "BG_tc1": "tc1", "BG_tc00": "tc00", "BG_tc10": "tc10"
                                          }, inplace=True)
timeconst_meta_df_threads = pd.concat([timeconst_meta_df_thread0, timeconst_meta_df_thread1], ignore_index=True)

#%%
msk = bothsucsmsk & validmsk
plt.figure(figsize=[4, 6])
sns.stripplot(data=timeconst_meta_df_threads[pd.concat([msk, msk]).reset_index()[0]],
              x="visual_area", y="tc1", hue="thread", order=["V1", "V4", "IT"], dodge=True,
              alpha=0.4, jitter=0.25, palette=["blue", "red"])
sns.pointplot(data=timeconst_meta_df_threads[pd.concat([msk, msk]).reset_index()[0]],
                x="visual_area", y="tc1", hue="thread", order=["V1", "V4", "IT"], dodge=True,
                palette=["blue", "red"], errorbar="se", join=False, scale=1.0, errwidth=1, capsize=0.2)
plt.suptitle("Time constant of optimization trajectory\n[Both threads succeed]")
plt.legend()
plt.show()
#%%
msk = anysucsmsk&validmsk
plt.figure(figsize=[4, 6])
sns.stripplot(data=timeconst_meta_df_threads[pd.concat([msk, msk]).reset_index()[0]],
              x="visual_area", y="tc1", hue="thread", order=["V1", "V4", "IT"], dodge=True,
              alpha=0.4, jitter=0.25, palette=["blue", "red"])
sns.pointplot(data=timeconst_meta_df_threads[pd.concat([msk, msk]).reset_index()[0]],
                x="visual_area", y="tc1", hue="thread", order=["V1", "V4", "IT"], dodge=True,
                palette=["blue", "red"], errorbar="se", join=False, scale=1.0, errwidth=1, capsize=0.2)
plt.suptitle("Time constant of optimization trajectory\n[Any threads succeed]")
plt.legend()
plt.show()
#%%
msk = bothsucsmsk&validmsk
plt.figure(figsize=[4, 6])
sns.stripplot(data=timeconst_meta_df_threads[pd.concat([msk, msk]).reset_index()[0]],
              x="visual_area", y="tc0", hue="GANthread", order=["V1", "V4", "IT"], dodge=True,
              alpha=0.4, jitter=0.25, palette=["blue", "red", "magenta"])
sns.pointplot(data=timeconst_meta_df_threads[pd.concat([msk, msk]).reset_index()[0]],
                x="visual_area", y="tc0", hue="GANthread", order=["V1", "V4", "IT"], dodge=True,
                palette=["blue", "red", "magenta"], errorbar="se", join=False, scale=1.0, errwidth=1, capsize=0.2)
plt.suptitle("Time constant of optimization trajectory\n[Both success]") # ('ci', 68)
plt.legend()
plt.show()
#%%
msk = anysucsmsk&validmsk
plt.figure(figsize=[4, 6])
sns.stripplot(data=timeconst_meta_df_threads[pd.concat([msk, msk]).reset_index()[0]],
              x="visual_area", y="tc0", hue="GANthread", order=["V1", "V4", "IT"], dodge=True,
              alpha=0.4, jitter=0.25, palette=["blue", "red", "magenta"])
sns.pointplot(data=timeconst_meta_df_threads[pd.concat([msk, msk]).reset_index()[0]],
                x="visual_area", y="tc0", hue="GANthread", order=["V1", "V4", "IT"], dodge=True,
                palette=["blue", "red", "magenta"], join=False, errorbar="se", scale=1.0, errwidth=1, capsize=0.2)
plt.suptitle("Time constant of optimization trajectory\n[Any success]")  # ('ci', 68)
plt.legend()
plt.show()
#%%
# interactive pycharm
plt.switch_backend('module://backend_interagg')

#%%
msk = sucsmsk&validmsk
plt.figure(figsize=[4, 6])
sns.stripplot(data=timeconst_meta_df_threads[pd.concat([msk, msk]).reset_index()[0]],
              x="visual_area", y="tc1", hue="thread", order=["V1", "V4", "IT"], dodge=True,
              alpha=0.4, jitter=0.25, palette=["blue", "red"])
sns.pointplot(data=timeconst_meta_df_threads[pd.concat([msk, msk]).reset_index()[0]],
                x="visual_area", y="tc1", hue="thread", order=["V1", "V4", "IT"], dodge=True,
                palette=["blue", "red"], ci=68, join=False, scale=1.0, errwidth=1, capsize=0.2)
plt.suptitle("Time constant of optimization trajectory")
plt.legend()
plt.show()
#%%
msks = (FCsucsmsk&validmsk, BGsucsmsk&validmsk)
plt.figure(figsize=[4, 6])
sns.stripplot(data=timeconst_meta_df_threads[pd.concat(msks).reset_index()[0]],
              x="visual_area", y="tc1", hue="thread", order=["V1", "V4", "IT"], dodge=True,
              alpha=0.4, jitter=0.25, palette=["blue", "red"])
sns.pointplot(data=timeconst_meta_df_threads[pd.concat(msks).reset_index()[0]],
                x="visual_area", y="tc1", hue="thread", order=["V1", "V4", "IT"], dodge=True,
                palette=["blue", "red"], ci=68, join=False, scale=1.0, errwidth=1, capsize=0.2)
plt.suptitle("Time constant of optimization trajectory\n[Each succeed]")
plt.legend()
plt.show()
#%%
# show all columns
pd.set_option('display.max_columns', None)
timeconst_meta_df[validmsk&FCsucsmsk].groupby("visual_area", sort=False)[["FC_tc0", "FC_tc1", "BG_tc0", "BG_tc1"]].agg(["mean", "sem"],)
#%%
timeconst_meta_df[validmsk&BGsucsmsk].groupby("visual_area", sort=False)[["FC_tc0", "FC_tc1", "BG_tc0", "BG_tc1"]].agg(["mean", "sem"],)
#%%
#%%
plt.figure(figsize=[4, 6])
sns.boxplot(data=timeconst_meta_df[FCsucsmsk], x="visual_area", y="FC_tc0",
              color="blue", order=["V1", "V4", "IT"])
sns.boxplot(data=timeconst_meta_df[BGsucsmsk], x="visual_area", y="BG_tc0",
              color="red", order=["V1", "V4", "IT"])
plt.legend()
plt.show()


#%%
baseline_col = OrderedDict()
for Expi, resp_arr in tqdm(resp_col.items()):
    print(f"Processing {Expi}")
    bsl0 = resp_arr[:, 4].mean(axis=0)
    bsl1 = resp_arr[:, 5].mean(axis=0)
    col = {"FC_bsl": bsl0, "BG_bsl": bsl1, }
    baseline_col[Expi] = col
baseline_df = pd.DataFrame(baseline_col).T
baseline_meta_df = pd.merge(meta_df, baseline_df, left_on="Expi", right_index=True,)
#%%
ttest_rel_print_df(baseline_meta_df, validmsk&ITmsk&bothsucsmsk, "FC_bsl", "BG_bsl")