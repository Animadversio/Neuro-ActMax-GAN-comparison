"""
Connect the feature similarity of the Evolution prototype vs the similarity of optimized psth.
"""
import os
from os.path import join
from tqdm.autonotebook import trange, tqdm
import pickle as pkl
import torch
import numpy as np
import pandas as pd
from lpips import LPIPS
import seaborn as sns
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms
from core.utils.montage_utils import crop_all_from_montage, make_grid_np, build_montages, crop_from_montage, make_grid
from neuro_data_analysis.neural_data_lib import load_neural_data, extract_all_evol_trajectory_psth, pad_psth_traj
from neuro_data_analysis.neural_data_utils import get_all_masks
#%%
tabdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables"
protosumdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoSummary"
meta_act_df = pd.read_csv(join(tabdir, "meta_activation_stats.csv"), )

_, BFEStats = load_neural_data()
psth_col, meta_df = extract_all_evol_trajectory_psth(BFEStats)
psth_extrap_arr, extrap_mask_arr, max_len = pad_psth_traj(psth_col)
Amsk, Bmsk, V1msk, V4msk, ITmsk, \
    length_msk, spc_msk, sucsmsk, \
    bsl_unstable_msk, bsl_stable_msk, validmsk = get_all_masks(meta_act_df)
bothsucmsk = (meta_act_df.p_maxinit_0 < 0.05) & (meta_act_df.p_maxinit_1 < 0.05)
#%% Load the data
def extract_max_act_block_psth(psth_arr):
    """

    :param psth_arr: (n_block, 4, n_time 200)
    :return:
    """
    max_id = np.argmax(psth_arr[:, 0, 50:200].mean(axis=-1), axis=0)
    max_psth_mean0 = psth_arr[max_id, 0, :]
    max_psth_sem0 = psth_arr[max_id, 2, :]
    max_id = np.argmax(psth_arr[:, 1, 50:200].mean(axis=-1), axis=0)
    max_psth_mean1 = psth_arr[max_id, 1, :]
    max_psth_sem1 = psth_arr[max_id, 3, :]
    return max_psth_mean0, max_psth_sem0, max_psth_mean1, max_psth_sem1

#%%
psth_dif_col = {}
for Expi in tqdm(list(psth_col.keys())):
    psth_bunch = psth_col[Expi]
    # normalizer = psth_bunch[:, 0:1, 50:200].mean(axis=-1).max(axis=(0, 1)) # buggy, fixed on June.5th,2023
    normalizer = psth_bunch[:, 0:2, 50:200].mean(axis=-1).max(axis=(0, 1))
    psth_bunch = psth_bunch / normalizer
    max_psth_mean0, max_psth_sem0, max_psth_mean1, max_psth_sem1 = extract_max_act_block_psth(psth_bunch)
    # ways to quantify the difference or similarity between the two psth
    psth_corr = np.corrcoef(max_psth_mean0[50:200], max_psth_mean1[50:200])[0, 1]
    psth_MAE = np.abs(max_psth_mean0 - max_psth_mean1)[50:200].mean()
    psth_MSE = ((max_psth_mean0 - max_psth_mean1)**2)[50:200].mean()
    # smooth the psth and then compute the MAE
    max_psth_mean0_smooth = np.convolve(max_psth_mean0[50:200], np.ones(25)/25, mode='valid')
    max_psth_mean1_smooth = np.convolve(max_psth_mean1[50:200], np.ones(25)/25, mode='valid')
    psth_MAE_smooth = np.abs(max_psth_mean0_smooth - max_psth_mean1_smooth).mean()
    psth_MSE_smooth = ((max_psth_mean0_smooth - max_psth_mean1_smooth)**2).mean()
    psth_corr_smooth = np.corrcoef(max_psth_mean0_smooth, max_psth_mean1_smooth)[0,1]
    act_MAE = np.abs(max_psth_mean0[50:200].mean() - max_psth_mean1[50:200].mean())
    S = edict()
    S['normalizer'] = normalizer
    S['psth_corr'] = psth_corr
    S['psth_MAE'] = psth_MAE
    S['psth_MSE'] = psth_MSE
    S['psth_corr_smooth'] = psth_corr_smooth
    S['psth_MAE_smooth'] = psth_MAE_smooth
    S['psth_MSE_smooth'] = psth_MSE_smooth
    S['act_MAE'] = act_MAE
    psth_dif_col[Expi] = S

psth_dif_df = pd.DataFrame(psth_dif_col).T
psth_dif_df['Expi'] = psth_dif_df.index
psth_dif_df.to_csv(join(tabdir, "Evol_psth_dif_df.csv"))
#%%

imgdist_df_orig = pd.read_csv(join(tabdir, "proto_imdist_df.csv"), index_col=0)
imgdist_df_rep = pd.read_csv(join(tabdir, "resnet50_imgdist_df_rep00_mskchange.csv"), )
covtsrdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Proto_covtsr_similarity"
cov_stat_df = pd.read_csv(join(covtsrdir, "covtsr_stat.csv"), index_col=0)
#%%
imgdist_df = imgdist_df_orig.copy()
imgdist_df = imgdist_df.merge(imgdist_df_rep, on='Expi')
imgdist_df = imgdist_df.merge(psth_dif_df, on='Expi')
imgdist_df = imgdist_df.merge(cov_stat_df, on='Expi')
imgdist_df = imgdist_df.merge(meta_act_df, on='Expi')
imgdist_df.to_csv(join(tabdir, "proto_imdist_psth_covstr_sim_df.csv"))

#%%
from core.utils.stats_utils import test_correlation_diff_df, scatter_corr
from scipy.stats import pearsonr, spearmanr
def scatter_corr(df, x, y, ax=None, corrtype="pearson", **kwargs):
    if ax is None:
        ax = plt.gca()
    # ax.scatter(df[x], df[y], **kwargs)
    sns.scatterplot(data=df, x=x, y=y, ax=ax, **kwargs)
    # scipy pearsonr
    validmsk = np.logical_and(np.isfinite(df[x]), np.isfinite(df[y]))
    if corrtype.lower() == "pearson":
        rho, pval = pearsonr(df[x][validmsk], df[y][validmsk], )
    elif corrtype.lower() == "spearman":
        rho, pval = spearmanr(df[x][validmsk], df[y][validmsk], )
    else:
        raise NotImplementedError
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs. {y}\ncorr={rho:.3f} p={pval:.1e} n={validmsk.sum()}")
    return ax, rho, pval

figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\PSTH_diff_vs_img_similarity"
#%%
# test `cosine_reevol_resnet_avgpool` correlate with `psth_MAE` better than `act_MAE`
test_correlation_diff_df(imgdist_df[validmsk & bothsucmsk],
             "cosine_reevol_resnet_avgpool", "psth_MAE", "act_MAE", verbose=True)
#%%
# test `cosine_reevol_resnet_avgpool` correlate with `psth_MAE` better than `act_MAE`
test_correlation_diff_df(imgdist_df[validmsk & bothsucmsk & ITmsk],
             "cosine_reevol_resnet_avgpool", "psth_MAE", "act_MAE", verbose=True)
#%%
fig = plt.figure(figsize=[4, 4])
ax, rho, pval = scatter_corr(imgdist_df[validmsk & bothsucmsk],
                   'cosine_reevol_resnet_avgpool', 'psth_MAE',
                   s=25, alpha=0.5, hue='visual_area')
plt.tight_layout()
saveallforms(figdir, 'psth_MAE_vs_img_sim_cosine_reevol_resnet_avgpool_bothsucs_val_areasep',)
plt.show()
#%%
fig = plt.figure(figsize=[4, 4])
ax, rho, pval = scatter_corr(imgdist_df[validmsk & bothsucmsk],
                   'cosine_reevol_resnet_avgpool', 'psth_MAE',
                   s=25, alpha=0.5, hue='visual_area')
plt.tight_layout()
saveallforms(figdir, 'psth_MAE_vs_img_sim_cosine_reevol_resnet_avgpool_bothsucs_val_areasep',)
plt.show()
#%%
fig = plt.figure(figsize=[4, 4])
ax, rho, pval = scatter_corr(imgdist_df[validmsk & bothsucmsk],
                   'cosine_reevol_resnet_avgpool', 'act_MAE',
                   s=25, alpha=0.5, hue='visual_area')
plt.tight_layout()
saveallforms(figdir, 'act_MAE_vs_img_sim_cosine_reevol_resnet_avgpool_bothsucs_val_areasep',)
plt.show()

# %%
fig = plt.figure(figsize=[4, 4])
ax, rho, pval = scatter_corr(imgdist_df[validmsk & bothsucmsk],
                             'psth_MAE', 'cosine_reevol_resnet_avgpool',
                             s=25, alpha=0.5, hue='visual_area')
plt.tight_layout()
saveallforms(figdir, 'psth_MAE_vs_img_sim_cosine_reevol_resnet_avgpool_bothsucs_val_areasep_T', )
plt.show()

#%%
fig = plt.figure(figsize=[4, 4])
ax,rho, pval = scatter_corr(imgdist_df[validmsk & sucsmsk & ITmsk],
                  'cosine_reevol_resnet_avgpool',
                  'psth_MAE', s=25, alpha=0.5)
plt.tight_layout()
plt.show()

#%%
fig = plt.figure(figsize=[4, 4])
ax, rho, pval = scatter_corr(imgdist_df[validmsk & bothsucmsk],
                   'cosine_reevol_resnet_avgpool',
                   'psth_MAE', s=25, alpha=0.5)
plt.tight_layout()
plt.show()
#%%
from core.utils.stats_utils import ttest_ind_print_df
ttest_ind_print_df(imgdist_df, validmsk & bothsucmsk&ITmsk,
                   validmsk & sucsmsk&(~bothsucmsk)&ITmsk,  'cosine_reevol_resnet_avgpool', )
#%%
fig = plt.figure(figsize=[4, 4])
ax, rho, pval = scatter_corr(imgdist_df[validmsk & bothsucmsk],
                   'cosine_reevol_resnet_avgpool',
                   't_FCBG_max_01', s=25, alpha=0.5)
plt.tight_layout()
plt.show()

#%%
fig = plt.figure(figsize=[4, 4])
ax, rho, pval = scatter_corr(imgdist_df[validmsk & bothsucmsk],
                   'layer4_cos_thr_crop_mean',
                   'psth_MAE', s=25, alpha=0.5)
plt.tight_layout()
plt.show()
#%%
fig = plt.figure(figsize=[4, 4])
ax, rho, pval = scatter_corr(imgdist_df[validmsk & sucsmsk ],
                   'reevol_pix_resnet_L4',
                  'psth_MAE', s=25, alpha=0.5)
plt.tight_layout()
plt.show()
#%%
fig = plt.figure(figsize=[4, 4])
ax = scatter_corr(imgdist_df[validmsk & ITmsk],
                  'cosine_reevol_alexnet_fc6',
                  'psth_MAE', s=25, alpha=0.5)
plt.tight_layout()
plt.show()
#%%
fig = plt.figure(figsize=[4, 4])
sns.scatterplot(data=imgdist_df[validmsk & sucsmsk],
                x='cosine_reevol_resnet_avgpool', y='psth_MAE',
                hue='visual_area', alpha=0.5)
rho, pval = pearsonr(imgdist_df[validmsk & sucsmsk]['cosine_reevol_resnet_avgpool'],
                        imgdist_df[validmsk & sucsmsk]['psth_MAE'])
plt.title(f"PSTH difference ~ Image similarity\nSpearman corr={rho:.3f} p={pval:.1e}")
plt.tight_layout()
saveallforms(figdir, 'psth_MAE_vs_img_sim_cosine_reevol_resnet_avgpool', fig, ['png', 'svg', 'pdf'])
plt.show()
#%%
fig = plt.figure(figsize=[4, 4])
sns.scatterplot(data=imgdist_df[validmsk & sucsmsk],
                x='psth_MAE', y='cosine_reevol_resnet_avgpool',
                hue='visual_area', alpha=0.5)
rho, pval = pearsonr(imgdist_df[validmsk & sucsmsk]['psth_MAE'],
                        imgdist_df[validmsk & sucsmsk]['cosine_reevol_resnet_avgpool'])
plt.title(f"PSTH difference ~ Image similarity\nSpearman corr={rho:.3f} p={pval:.1e}")
plt.tight_layout()
saveallforms(figdir, 'psth_MAE_vs_img_sim_cosine_reevol_resnet_avgpool_T', fig, ['png', 'svg', 'pdf'])
plt.show()
#%%
x_label = 'maxblk_resnet_L3'
y_label = 'psth_MAE'
fig = plt.figure(figsize=[4, 4])
sns.scatterplot(data=imgdist_df[validmsk & sucsmsk],
                x=x_label, y=y_label,
                hue='visual_area', alpha=0.5)
rho, pval = pearsonr(imgdist_df[validmsk & sucsmsk][x_label],
                        imgdist_df[validmsk & sucsmsk][y_label])
plt.title(f"PSTH difference ~ Image similarity\nSpearman corr={rho:.3f} p={pval:.1e} n={(validmsk & sucsmsk).sum()}")
plt.tight_layout()
saveallforms(figdir, f'{y_label}_vs_img_sim_{x_label}', fig, ['png', 'svg', 'pdf'])
plt.show()

