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
#%%
_, BFEStats = load_neural_data()
resp_col, _ = extract_all_evol_trajectory(BFEStats, )
resp_extrap_arr, extrap_mask_arr, max_len = pad_resp_traj(resp_col)
meta_df = pd.read_csv(Path(tabdir) / "meta_activation_stats_w_optimizer.csv")
Amsk, Bmsk, V1msk, V4msk, ITmsk, \
    length_msk, spc_msk, sucsmsk, \
    bsl_unstable_msk, bsl_stable_msk, validmsk = get_all_masks(meta_df)
#%%
FCsucsmsk = meta_df.p_maxinit_0 < 0.01
BGsucsmsk = meta_df.p_maxinit_1 < 0.01
bothsucsmsk = FCsucsmsk & BGsucsmsk
anysucsmsk = FCsucsmsk | BGsucsmsk
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
thresh = 0.8
timeconst_col = OrderedDict()
for Expi, resp_arr in tqdm(resp_col.items()):
    print(f"Processing {Expi}")
    bsl0 = resp_arr[:, 4].mean(axis=0)
    bsl1 = resp_arr[:, 5].mean(axis=0)
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
timeconst_df.to_csv(join(tabdir, "Evol_traj_time_constant.csv"))
#%%
timeconst_meta_df = pd.merge(meta_df, timeconst_df, left_on="Expi", right_index=True,)
#%%
figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_traj_time_constant"
timeconst_df = pd.read_csv(join(tabdir, "Evol_traj_time_constant.csv"), index_col=0)
timeconst_meta_df = pd.merge(meta_df, timeconst_df, left_on="Expi", right_index=True,)

#%%

def mean_sem_from_msk(df, col, msk, fmt=".1f", std_metric="sem"):
    N = msk.sum()
    mean_value = df[col][msk].mean()
    if std_metric == "sem":
        sem_value = sem(df[col][msk])
    elif std_metric == "std":
        sem_value = df[col][msk].std()
    else:
        raise ValueError("std_metric should be sem or std")
    latex_str = f"{mean_value:{fmt}} ± {sem_value:{fmt}} (N={N})"
    return mean_value, sem_value, N, latex_str


stats_sfx = "_tc0_bslinit"
df_syn_col = {}
for stats_sfx in ["_tc0_bslinit", "_tc0_bsl0", "_tc0", "_tc_cnt_bslinit", "_tc_cnt_bsl0", "_tc_cnt", ]:
    stats_row = {}
    print(f"Stats suffix: {stats_sfx}")
    for area in ["V1", "V4", "IT", ]:
        for GAName in ["FC", "BG"]:
            GAN_mean, GAN_sem, GAN_N, latex_str = mean_sem_from_msk(timeconst_meta_df, f"{GAName}{stats_sfx}",
                                                                    validmsk&eval(f"{GAName}sucsmsk")&eval(f"{area}msk"))
            print(f"Area {area} {GAName}: {latex_str}")
            stats_row[f"{area}-{GAName}"] = {"mean": GAN_mean, "sem": GAN_sem, "N": int(GAN_N), "latex_str": latex_str}
            # FCmean, FCsem, FC_N, FCstr = mean_sem_from_msk(timeconst_meta_df, "FC"+stats_sfx, validmsk&FCsucsmsk&eval(f"{area}msk"))
            # BGmean, BGsem, BG_N, BGstr = mean_sem_from_msk(timeconst_meta_df, "BG"+stats_sfx, validmsk&BGsucsmsk&eval(f"{area}msk"))
            # print(f"Area {area} FC: {FCstr} BG: {BGstr}")
            # stats_row[area] = {"FCmean": FCmean, "FCsem": FCsem, "FC_N": int(FC_N), "BGmean": BGmean, "BGsem": BGsem, "BG_N": int(BG_N)}
    tval, pval, dof, result_str = ttest_ind_print_df(timeconst_meta_df, validmsk&FCsucsmsk&ITmsk, validmsk&FCsucsmsk&V4msk, "FC"+stats_sfx, output_dof=True, sem=True)
    stats_row["IT vs V4-FC"] = {"tval": tval, "pval": pval, "dof":dof, "latex_str":result_str.split(") ")[-1]}
    tval, pval, dof, result_str = ttest_ind_print_df(timeconst_meta_df, validmsk&FCsucsmsk&ITmsk, validmsk&FCsucsmsk&V1msk, "FC"+stats_sfx, output_dof=True, sem=True)
    stats_row["IT vs V1-FC"] = {"tval": tval, "pval": pval, "dof":dof, "latex_str":result_str.split(") ")[-1]}#"result_str": result_str}
    tval, pval, dof, result_str = ttest_rel_print_df(timeconst_meta_df, validmsk&bothsucsmsk&ITmsk, "FC"+stats_sfx, "BG"+stats_sfx, output_dof=True, sem=True)
    stats_row["IT-FC vs BG"] = {"tval": tval, "pval": pval, "dof":dof, "latex_str":result_str.split(") ")[-1]}#"result_str": result_str}
    df_syn_col[stats_sfx] = stats_row

df_syn = pd.DataFrame(df_syn_col, )


# for area in ["IT", "V4", "V1"]:
#     FCmean, FCsem, FCstr = mean_sem_from_msk(timeconst_meta_df, "FC"+stats_sfx, validmsk&FCsucsmsk&eval(f"{area}msk"))
#     BGmean, BGsem, BGstr = mean_sem_from_msk(timeconst_meta_df, "BG"+stats_sfx, validmsk&BGsucsmsk&eval(f"{area}msk"))
#     print(f"Area {area} FC: {FCstr} BG: {BGstr}")
# mean_sem_from_msk(timeconst_meta_df, "FC"+stats_sfx, validmsk&FCsucsmsk&ITmsk)
# mean_sem_from_msk(timeconst_meta_df, "FC"+stats_sfx, validmsk&FCsucsmsk&V4msk)
# mean_sem_from_msk(timeconst_meta_df, "FC"+stats_sfx, validmsk&FCsucsmsk&V1msk)
#%%
data = []
for stats_sfx, areas in df_syn_col.items():
    for area, metrics in areas.items():
        for metric, value in metrics.items():
            if metric == "latex_str":
                continue
            data.append((stats_sfx, area, metric, value))
# Create a MultiIndex
index = pd.MultiIndex.from_tuples([(x[0], x[1], x[2]) for x in data], names=["Suffix", "Area", "Metric"])
# Create the DataFrame
df = pd.DataFrame([x[3] for x in data], index=index, columns=["Value"])
# Optionally, you can unstack the DataFrame for better readability or analysis
df_unstacked = df.unstack(level=["Area", "Metric"])
print(df_unstacked)
df_unstacked.to_csv(join(tabdir, "Evol_traj_time_constant_stats_synopsis_export.csv"))
df_unstacked.to_excel(join(tabdir, "Evol_traj_time_constant_stats_synopsis_export.xlsx"))

#%%

#%%
data = []
for stats_sfx, areas in df_syn_col.items():
    for area, metrics in areas.items():
        data.append((stats_sfx, area, metrics["latex_str"]))
        # for metric, value in metrics.items():
        #     data.append((stats_sfx, area, metric, value))
# Create a MultiIndex
index = pd.MultiIndex.from_tuples([(x[0], x[1], ) for x in data], names=["Suffix", "Area", ])
# Create the DataFrame
df = pd.DataFrame([x[2] for x in data], index=index, columns=["Value"])
# Optionally, you can unstack the DataFrame for better readability or analysis, no sorting
df_unstacked = df.unstack(level="Area", sort=False)
print(df_unstacked)
df_unstacked.to_csv(join(tabdir, "Evol_traj_time_constant_stats_synopsis_export_compact.csv"))
df_unstacked.to_excel(join(tabdir, "Evol_traj_time_constant_stats_synopsis_export_compact.xlsx"))


#%% plotting figures from this

#%%
exportdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Figure_OptimDynamic\src"
# msk = validmsk & bothsucsmsk
sucslabel = "eachsucs"
tclabel = "tc0_bslinit"
plt.figure(figsize=[4, 6])
sns.stripplot(data=timeconst_meta_df[validmsk&FCsucsmsk], x="visual_area", y="FC_tc0_bslinit", order=["V1", "V4", "IT"],
              color="blue", alpha=0.4, jitter=0.25, label=None,  dodge=True)
sns.stripplot(data=timeconst_meta_df[validmsk&BGsucsmsk], x="visual_area", y="BG_tc0_bslinit", order=["V1", "V4", "IT"],
              color="red", alpha=0.4, jitter=0.25, label=None, dodge=True)
sns.pointplot(data=timeconst_meta_df[validmsk&FCsucsmsk], x="visual_area", y="FC_tc0_bslinit", order=["V1", "V4", "IT"],
              errorbar=("ci", 95), color="blue", label="DeePSim",scale=1.0, errwidth=1, capsize=0.2)
sns.pointplot(data=timeconst_meta_df[validmsk&BGsucsmsk], x="visual_area", y="BG_tc0_bslinit", order=["V1", "V4", "IT"],
              errorbar=("ci", 95), color="red", label="BigGAN",scale=1.0, errwidth=1, capsize=0.2)
plt.suptitle("Time constant of optimization trajectory\n[Each succeed]")

plt.legend()
saveallforms([figdir, exportdir], f"Evol_traj_time_constant_{sucslabel}_{tclabel}_cmp", plt.gcf())
plt.show()
