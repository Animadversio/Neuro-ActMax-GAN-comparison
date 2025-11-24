# %%
%load_ext autoreload
%autoreload 2

# %%
import os
from os.path import join
from tqdm.autonotebook import trange, tqdm
import pickle as pkl
import torch
import numpy as np
import pandas as pd
# from lpips import LPIPS÷
import seaborn as sns
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from scipy.stats import pearsonr, spearmanr

# import sys
# sys.path.append(r"/Users/binxuwang/Github/Neuro-ActMax-GAN-comparison/")
# from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms
# from core.utils.montage_utils import crop_all_from_montage, make_grid_np, build_montages, crop_from_montage, make_grid
# from neuro_data_analysis.neural_data_lib import load_neural_data, extract_all_evol_trajectory_psth, pad_psth_traj
# from neuro_data_analysis.neural_data_utils import get_all_masks

# %%
source_data_dir = r"/Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/Manuscript_BigGAN/Submissions/Manuscript_BigGAN - NatNeuro/2025-10-Accepted-In-Principle-Docs/SourceData/Fig3_Source_data"


def scatter_corr(df, x, y, ax=None, corrtype="pearson", **kwargs):
    """wrapper over sns.scatterplot to add correlation coefficient and p-value to annotation. """
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

# %%
imgdist_df = pd.read_csv(join(source_data_dir, "Fig3G_proto_imdist_psth_covstr_sim_df.csv"))
meta_act_df = pd.read_csv(join(source_data_dir, "Fig3_meta_df.csv"))

validmsk = meta_act_df["valid"]
bothsucmsk = (meta_act_df.p_maxinit_0 < 0.05) & (meta_act_df.p_maxinit_1 < 0.05)
anysucmsk = (meta_act_df.p_maxinit_0 < 0.05) | (meta_act_df.p_maxinit_1 < 0.05)

# %%
fig = plt.figure(figsize=[4, 4])
ax, rho, pval = scatter_corr(imgdist_df[validmsk & bothsucmsk ],
                   'cosine_reevol_resnet_avgpool',
                  'psth_MAE', s=25, alpha=0.5, hue='visual_area')
plt.tight_layout()
plt.show()

# %%
fig = plt.figure(figsize=[4, 4])
ax, rho, pval = scatter_corr(imgdist_df[validmsk & bothsucmsk ],
                   'reevol_pix_RNrobust_L4focus',
                  'psth_MAE', s=25, alpha=0.5, hue='visual_area')
plt.tight_layout()
plt.show()

# %% Control case, using activation instead of PSTh. 
fig = plt.figure(figsize=[4, 4])
ax, rho, pval = scatter_corr(imgdist_df[validmsk & bothsucmsk ],
                   'reevol_pix_RNrobust_L4focus',
                  'act_MAE', s=25, alpha=0.5, hue='visual_area')
plt.tight_layout()
plt.show()
#%%
fig = plt.figure(figsize=[4, 4])
ax, rho, pval = scatter_corr(imgdist_df[validmsk & bothsucmsk ],
                   'cosine_reevol_resnet_avgpool',
                  'act_MAE', s=25, alpha=0.5, hue='visual_area')
plt.tight_layout()
plt.show()



