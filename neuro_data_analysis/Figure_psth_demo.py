"""
Save the psth pairs for some example sessions as montage.
"""
import os
from os.path import join
from tqdm.autonotebook import trange, tqdm
import pickle as pkl
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from lpips import LPIPS
import matplotlib.pyplot as plt
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms
from core.utils.montage_utils import crop_all_from_montage, make_grid_np, build_montages, crop_from_montage, make_grid
from neuro_data_analysis.neural_data_lib import load_neural_data, extract_all_evol_trajectory_psth, pad_psth_traj
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
example_list = [3, 64, 65, 66, 74, 79, 111, 113, 118, 155, 174, 175]  # 28
exemplar_col = []
exemplar_mtg_col = []
maxblk_mtg_col = []
for Expi in tqdm(example_list):
    mtg = plt.imread(join(protosumdir, f"Exp{Expi}_proto_attr_montage.jpg"))
#%%
from core.utils.stats_utils import shaded_errorbar
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
figsumdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoImage_cmp"
example_list = [3, 65, 66, 74, 111, 118, 174, 175, 113, 64, ] # [3, 64, 65, 66, 74, 79, 111, 113, 118, 155, 174, 175]  # 28
figh,axs = plt.subplots(1, len(example_list), figsize=[17, 2.35], sharey=True)  # ,sharex=True
for i, Expi in tqdm(enumerate(example_list)):
    titlestr = f"Exp{Expi} {meta_df.loc[Expi, 'Animal'][0]} {meta_df.loc[Expi, 'visual_area']} Ch{meta_df.loc[Expi,'prefchan']}"
    axs[i].set_title(titlestr)
    psth_bunch = psth_col[Expi]
    normalizer = psth_bunch[:, 0:1, 50:200].mean(axis=-1).max(axis=(0,1))
    psth_bunch = psth_bunch / normalizer
    max_psth_mean0, max_psth_sem0, max_psth_mean1, max_psth_sem1 = extract_max_act_block_psth(psth_bunch)
    shaded_errorbar(np.arange(200), max_psth_mean0, max_psth_sem0, ax=axs[i], color='b', label="DeePSim" if i==0 else "")
    shaded_errorbar(np.arange(200), max_psth_mean1, max_psth_sem1, ax=axs[i], color='r', label="BigGAN" if i==0 else "")
    if i == 0:
        # legend without box
        axs[i].legend(loc='upper right', frameon=False)
    axs[i].set_ylim([-0.05, 3.1])
plt.tight_layout()
saveallforms(figsumdir, f"Example_best_proto_psths_brief", figh, fmts=['png', 'pdf', 'svg'])
plt.show()

#%%
