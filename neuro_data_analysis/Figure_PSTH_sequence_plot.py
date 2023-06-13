import os
from os.path import join
from tqdm.autonotebook import trange, tqdm
import pickle as pkl
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from lpips import LPIPS
import seaborn as sns
import matplotlib.pyplot as plt
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms
from core.utils.montage_utils import crop_all_from_montage, make_grid_np, build_montages, crop_from_montage, make_grid
from neuro_data_analysis.neural_data_lib import load_neural_data, extract_all_evol_trajectory_psth, pad_psth_traj, get_expstr
from neuro_data_analysis.neural_data_utils import get_all_masks
protosumdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoSummary"
tabdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables"
meta_act_df = pd.read_csv(join(tabdir, "meta_activation_stats.csv"), )
#%%
_, BFEStats = load_neural_data()
psth_col, meta_df = extract_all_evol_trajectory_psth(BFEStats)
psth_extrap_arr, extrap_mask_arr, max_len = pad_psth_traj(psth_col)
#%%
Amsk, Bmsk, V1msk, V4msk, ITmsk, \
    length_msk, spc_msk, sucsmsk, \
    bsl_unstable_msk, bsl_stable_msk, validmsk = get_all_masks(meta_act_df)
#%%
Expi = 155
#%%
# (meta_act_df.index == Expi)
#%%
psth_col[Expi].shape
#%%
# plot the psths of Expi 155, as a sequence of time
def _shaded_errorbar(x, y, yerr, label=None, color=None, **kwargs):
    plt.fill_between(x, y-yerr, y+yerr, alpha=0.3, label=None, color=color)
    plt.plot(x, y, color=color, label=label, **kwargs)
# plt.figure(figsize=[12, 8])

def stack_psth_plot(psth_arr, offset=200, titlestr=""):
    """Plot a sequence of PSTHs stacked vertically with offset"""
    blockN = psth_arr.shape[0]
    fig = plt.figure(figsize=[4, 0.5 * blockN + 1])
    for block in range(blockN):
        _shaded_errorbar(np.arange(200), offset * block + psth_arr[block, 0, :], psth_col[Expi][block, 2, :],
                         color="blue")
        _shaded_errorbar(np.arange(200), offset * block + psth_arr[block, 1, :], psth_col[Expi][block, 3, :],
                         color="red")
        plt.axhline(offset * block, color="black", alpha=0.6, linestyle="--")
    plt.axhline(offset * blockN, color="black", alpha=0.3, linestyle=":", )
    plt.axhline(offset * (blockN + 1), color="black", alpha=0.3, linestyle=":", )
    plt.yticks(np.arange(0, offset * blockN, offset) + offset / 2, 1 + np.arange(0, blockN))
    plt.ylim(0, offset * blockN + np.max(psth_arr[-1, :2, :]))
    plt.xlabel("Time (ms)")
    plt.ylabel(f"Firing Rate each block (events/s)   ({offset} evt/s between dashed lines)")
    if titlestr != "":
        fig.suptitle(titlestr)
    plt.tight_layout()
    plt.show()
    return fig


def heatmap_psth_plot_horizontal(psth_arr, titlestr=""):
    vmin = psth_arr[:, 0:2, :].min()
    vmax = psth_arr[:, 0:2, :].max()
    fig, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, figsize=(10, 6.5),
                                            gridspec_kw={'width_ratios': [1, 1, 0.07], })
    sns.heatmap(psth_arr[:, 0, :], ax=ax1, cbar=False, vmin=vmin, vmax=vmax)
    ax1.set_title("DeePSim PSTHs")
    ax1.set_xticks([0, 50, 100, 150, 200])
    ax1.set_xticklabels([0, 50, 100, 150, 200], rotation=0)
    ax1.set_ylabel("Block #")
    ax1.set_xlabel("Time (ms)")
    sns.heatmap(psth_arr[:, 1, :], ax=ax2, cbar_ax=cbar_ax, vmin=vmin, vmax=vmax)
    ax2.set_title("BigGAN PSTHs")
    ax2.set_xticks([0, 50, 100, 150, 200])
    ax2.set_xticklabels([0, 50, 100, 150, 200], rotation=0)
    ax2.set_xlabel("Time (ms)")
    # Create the colorbar
    cbar = ax2.collections[0].colorbar
    cbar.ax.set_ylabel('Firing rate (events/s)')
    if titlestr != "":
        fig.suptitle(titlestr)
    plt.tight_layout()
    plt.show()
    return fig


def heatmap_psth_plot_vertical(psth_arr, titlestr=""):
    vmin = psth_arr[:, 0:2, :].min()
    vmax = psth_arr[:, 0:2, :].max()
    fig, axs = plt.subplots(2, 2, figsize=(5.5, 13),
                            gridspec_kw={'height_ratios': [1, 1], "width_ratios": [0.95, 0.05], })
    (ax1, ax2,) = axs[0, 0], axs[1, 0]
    cbar_ax = axs[1, 1]
    # combine axes for colorbar
    # cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    axs[0, 1].axis("off")
    # axs[1, 1].axis("off")
    sns.heatmap(psth_arr[:, 0, :], ax=ax1, cbar=False, vmin=vmin, vmax=vmax)
    ax1.set_title("DeePSim PSTHs")
    ax1.set_xticks([0, 50, 100, 150, 200])
    ax1.set_xticklabels([0, 50, 100, 150, 200], rotation=0)
    ax1.set_ylabel("Block #")
    # ax1.set_xlabel("Time (ms)")
    sns.heatmap(psth_arr[:, 1, :], ax=ax2, cbar_ax=cbar_ax, vmin=vmin, vmax=vmax)
    ax2.set_title("BigGAN PSTHs")
    ax2.set_xticks([0, 50, 100, 150, 200])
    ax2.set_xticklabels([0, 50, 100, 150, 200], rotation=0)
    ax2.set_ylabel("Block #")
    ax2.set_xlabel("Time (ms)")
    # Create the colorbar
    cbar = ax2.collections[0].colorbar
    cbar.ax.set_ylabel('Firing rate (events/s)')
    if titlestr != "":
        fig.suptitle(titlestr)
    plt.tight_layout()
    plt.show()
    return fig

from pathlib import Path
figdir = Path(r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_dynam_psth_raw_all")
# switch to agg backend to avoid showing figures
plt.switch_backend("agg")
for Expi in meta_act_df.Expi:
    expstr = get_expstr(BFEStats, Expi)
    figh1 = heatmap_psth_plot_vertical(psth_col[Expi], titlestr=expstr)
    figh2 = heatmap_psth_plot_horizontal(psth_col[Expi], titlestr=expstr)
    offset = psth_col[Expi][:, 0:2].max() / 2.2
    # round off to 10
    offset = np.ceil(offset / 10) * 10
    figh3 = stack_psth_plot(psth_col[Expi], offset=offset, titlestr=expstr)
    saveallforms(str(figdir), f"Exp{Expi}_psth_stack_trace", figh3, ["pdf", "png", ])
    saveallforms(str(figdir), f"Exp{Expi}_psth_heatmap_vert", figh1, ["pdf", "png", ])
    saveallforms(str(figdir), f"Exp{Expi}_psth_heatmap_horz", figh2, ["pdf", "png", ])
# 'module://backend_interagg'
plt.switch_backend("module://backend_interagg")
#%%
offset = 200
blockN = psth_col[Expi].shape[0]
fig = plt.figure(figsize=[4, 0.5*blockN])
for block in range(blockN):
    _shaded_errorbar(np.arange(200), offset*block + psth_col[Expi][block, 0, :], psth_col[Expi][block, 2, :], color="blue")
    _shaded_errorbar(np.arange(200), offset*block + psth_col[Expi][block, 1, :], psth_col[Expi][block, 3, :], color="red")
    plt.axhline(offset*block, color="black", alpha=0.6, linestyle="--")
plt.axhline(offset*blockN, color="black", alpha=0.3, linestyle=":", )
plt.axhline(offset*(blockN+1), color="black", alpha=0.3, linestyle=":", )
plt.yticks(np.arange(0, offset*blockN, offset)+offset/2, np.arange(0, blockN))
plt.ylim(0, offset*blockN+np.max(psth_col[Expi][-1, :2, :]))
plt.tight_layout()
plt.show()
# plt.legend()
#%%
vmin = np.min([psth_col[Expi][:, 0, :].min(), psth_col[Expi][:, 1, :].min()])
vmax = np.max([psth_col[Expi][:, 0, :].max(), psth_col[Expi][:, 1, :].max()])
fig, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, figsize=(10, 6),
                gridspec_kw={'width_ratios': [1, 1, 0.07], })
# Plot the first heatmap
sns.heatmap(psth_col[Expi][:, 0, :], ax=ax1, cbar=False, vmin=vmin, vmax=vmax)
ax1.set_title("DeePSim PSTHs")
ax1.set_xticks([0,50,100,150,200])
ax1.set_xticklabels([0,50,100,150,200], rotation=0)
ax1.set_ylabel("Block #")
ax1.set_xlabel("Time (ms)")
# Plot the second heatmap
sns.heatmap(psth_col[Expi][:, 1, :], ax=ax2, cbar_ax=cbar_ax, vmin=vmin, vmax=vmax)
ax2.set_title("BigGAN PSTHs")
ax2.set_xticks([0,50,100,150,200])
ax2.set_xticklabels([0,50,100,150,200], rotation=0)
ax2.set_xlabel("Time (ms)")
# Create the colorbar
cbar = ax2.collections[0].colorbar
cbar.ax.set_ylabel('Firing rate (events/s)')
plt.tight_layout()
plt.show()
#%%
# same as above but vertically stacked
vmin = np.min([psth_col[Expi][:, 0, :].min(), psth_col[Expi][:, 1, :].min()])
vmax = np.max([psth_col[Expi][:, 0, :].max(), psth_col[Expi][:, 1, :].max()])
fig, axs = plt.subplots(2, 2, figsize=(5.5, 12),
                gridspec_kw={'height_ratios': [1, 1], "width_ratios": [1, 0.07],})
(ax1, ax2, ) = axs[0, 0], axs[1, 0]
cbar_ax = axs[1, 1]
axs[0, 1].axis("off")
# Plot the first heatmap
sns.heatmap(psth_col[Expi][:, 0, :], ax=ax1, cbar=False, vmin=vmin, vmax=vmax)
ax1.set_title("DeePSim PSTHs")
ax1.set_xticks([0,50,100,150,200])
ax1.set_xticklabels([0,50,100,150,200], rotation=0)
ax1.set_ylabel("Block #")
# ax1.set_xlabel("Time (ms)")
# Plot the second heatmap
sns.heatmap(psth_col[Expi][:, 1, :], ax=ax2, cbar_ax=cbar_ax, vmin=vmin, vmax=vmax)
ax2.set_title("BigGAN PSTHs")
ax2.set_xticks([0,50,100,150,200])
ax2.set_xticklabels([0,50,100,150,200], rotation=0)
ax2.set_ylabel("Block #")
ax2.set_xlabel("Time (ms)")
# Create the colorbar
cbar = ax2.collections[0].colorbar
cbar.ax.set_ylabel('Firing rate (events/s)')
plt.tight_layout()
plt.show()


#%%
figh, axs = plt.subplots(1, 2, figsize=[10, 6])
sns.heatmap(psth_col[Expi][:, 0, :], ax=axs[0])
axs[0].set_title("DeePSim PSTHs")
axs[0].set_xticks([0,50,100,150,200])
axs[0].set_xticklabels([0,50,100,150,200], rotation=0)
# axs[0].set_xticks([0,50,100,150,200])
sns.heatmap(psth_col[Expi][:, 1, :], ax=axs[1])
axs[1].set_title("BigGAN PSTHs")
axs[1].set_xticks([0,50,100,150,200])
axs[1].set_xticklabels([0,50,100,150,200], rotation=0)
# axs[1].set_xticks([0,50,100,150,200])
# share colorbar
plt.tight_layout()
plt.show()