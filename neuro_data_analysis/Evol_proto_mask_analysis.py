"""
Compare the similarity of the masks between different conditions
"""
import os
from os.path import join
import torch
import pickle as pkl
import numpy as np
import pandas as pd
from collections import OrderedDict
from easydict import EasyDict as edict
from tqdm.autonotebook import trange, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from neuro_data_analysis.neural_data_lib import parse_montage
from neuro_data_analysis.neural_data_utils import get_all_masks
from core.utils.plot_utils import saveallforms
cov_root = r"E:\Network_Data_Sync\corrFeatTsr_BigGAN"
attr_dir = r"E:\Network_Data_Sync\BigGAN_FeatAttribution"
tabdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables"
protosumdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoSummary"
alphamaskdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\AlphaMasks"
meta_df = pd.read_csv(join(tabdir, "meta_stats.csv"), index_col=0)
Amsk, Bmsk, V1msk, V4msk, ITmsk, \
    length_msk, spc_msk, sucsmsk, \
    bsl_unstable_msk, bsl_stable_msk, validmsk = \
    get_all_masks(meta_df)
bothsucsmsk = (meta_df.p_maxinit_0 < 0.05) & (meta_df.p_maxinit_1 < 0.05)
#%% implement some mask similarity function
def norm_mask_similarity(mask1, mask2):
    """mask1 and mask2 are two 2d masks of the same size"""
    assert mask1.shape == mask2.shape
    mask1 = mask1 / mask1.max()
    mask2 = mask2 / mask2.max()
    return np.sum(mask1 * mask2) / np.sqrt(np.sum(mask1 ** 2) * np.sum(mask2 ** 2))


def norm_mask_iou(mask1, mask2):
    """Compute intersection over union (IoU) metric between two masks."""
    assert mask1.shape == mask2.shape
    # Flatten masks to 1D arrays
    mask1 = np.clip(mask1.flatten() / mask1.max(), 0, None)
    mask2 = np.clip(mask2.flatten() / mask2.max(), 0, None)
    # Compute intersection and union
    intersection = np.sum(np.minimum(mask1, mask2))
    union = np.sum(np.maximum(mask1, mask2))
    # Compute IoU
    iou = intersection / union
    return iou


def mask_merge(mask1, mask2):
    """show mask1 in magenta and mask2 in cyan, normalizing to 1"""
    assert mask1.shape == mask2.shape
    mask1 = mask1 / mask1.max()
    mask2 = mask2 / mask2.max()
    mask_RGB = np.stack([mask1, mask2, mask1], axis=-1)
    return mask_RGB
#%%
outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\AlphaMask_proto_analysis"
thread = "_cmb"
alphamask_all = OrderedDict()
for Expi in trange(1, 191):
    if not os.path.exists(join(cov_root, f"Both_Exp{Expi:02d}_Evol_thr{thread}_res-robust_corrTsr.npz")):
        continue
    mtg = plt.imread(join(protosumdir, f"Exp{Expi}_proto_attr_montage.jpg"))
    S = parse_montage(mtg)
    alphamask_col = {}
    for thread in [0, 1, "_cmb"]:
        for layer in ["layer2", "layer3"]:
            # with open(join(alphamaskdir, f"Exp{Expi:02d}_{layer}_thr{thread}_Hmaps.pkl"), "wb") as f:
            #     pkl.dump(
            #         dict(Hmaps=Hmaps, Hmaps_pad=Hmaps_pad, ccfactor=ccfactor, FactStat=FactStat,
            #              alphamap_full=alphamap_full, alphamap=alphamap), f
            #     )
            with open(join(alphamaskdir, f"Exp{Expi:02d}_{layer}_thr{thread}_Hmaps.pkl"), "rb") as f:
                data = pkl.load(f)
                alphamask_col[(thread, layer)] = data["alphamap_full"]

    alphamask_all[Expi] = alphamask_col

    figh, axs = plt.subplots(3, 4, figsize=[12, 9.5])
    axs[0, 0].imshow(S["FC_reevol_pix"],)
    axs[0, 1].imshow(S["BG_reevol_pix"],)
    axs[0, 2].imshow(S["both_reevol_pix"],)
    axs[1, 0].imshow(alphamask_col[(0, "layer2")], cmap="gray")
    axs[1, 1].imshow(alphamask_col[(1, "layer2")], cmap="gray")
    axs[1, 2].imshow(alphamask_col[(0, "layer2")] * alphamask_col[(1, "layer2")], cmap="gray")
    axs[1, 3].imshow(mask_merge(alphamask_col[(0, "layer2")], alphamask_col[(1, "layer2")]))
    mask_sim = norm_mask_similarity(alphamask_col[(0, "layer2")], alphamask_col[(1, "layer2")])
    # axs[1, 3].text(0.5, 0.5, f"{mask_sim:.3f}", fontsize=20, ha="center", va="center")
    axs[1, 3].title.set_text(f"{mask_sim:.3f}")
    axs[2, 0].imshow(alphamask_col[(0, "layer3")], cmap="gray")
    axs[2, 1].imshow(alphamask_col[(1, "layer3")], cmap="gray")
    axs[2, 2].imshow(alphamask_col[(0, "layer3")] * alphamask_col[(1, "layer3")], cmap="gray")
    axs[2, 3].imshow(mask_merge(alphamask_col[(0, "layer3")], alphamask_col[(1, "layer3")]))
    mask_sim = norm_mask_similarity(alphamask_col[(0, "layer3")], alphamask_col[(1, "layer3")])
    # axs[2, 3].text(0.5, 0.5, f"{mask_sim:.3f}", fontsize=20, ha="center", va="center")
    axs[2, 3].title.set_text(f"{mask_sim:.3f}")
    for ax in axs.ravel():
        ax.axis("off")
    figh.suptitle(f"Exp {Expi}")
    figh.tight_layout()
    saveallforms(outdir, f"Exp{Expi}_alphamask_proto", figh, ["png", ]) #  "pdf"
    figh.show()
    # raise NotImplementedError
#%%
def centroid_standard_dist(mask):
    # Create an array of the same shape as your mask with the coordinates of each cell
    coord_array = np.indices(mask.shape).reshape(2, -1).T
    # Calculate the weighted centroid
    weights = mask.flatten()
    weighted_centroid = np.average(coord_array, weights=weights, axis=0)
    # Calculate the weighted standard distance (root mean square deviation)
    square_dist = np.sum((coord_array - weighted_centroid) ** 2, axis=1)
    distance = np.sqrt(square_dist)
    weighted_mean_sq_distance = np.sqrt(np.average(square_dist, weights=weights, axis=0))
    weighted_mean_distance = np.average(distance, weights=weights, axis=0)
    # print(f"Weighted Centroid: {weighted_centroid}")
    # print(f"Weighted Standard Distance: {weighted_std_distance}")
    return weighted_centroid, weighted_mean_sq_distance, weighted_mean_distance
#%% Compute the similarity matrix
outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\AlphaMask_proto_analysis"
thread = "_cmb"  #1  # 1 # "_cmb"
mask_sim_col = OrderedDict()
for Expi in trange(1, 191):
    if not os.path.exists(join(cov_root, f"Both_Exp{Expi:02d}_Evol_thr{thread}_res-robust_corrTsr.npz")):
        continue
    # mtg = plt.imread(join(protosumdir, f"Exp{Expi}_proto_attr_montage.jpg"))
    # S = parse_montage(mtg)
    alphamask_col = {}
    for thread in [0, 1, "_cmb"]:
        for layer in ["layer2", "layer3"]:
            with open(join(alphamaskdir, f"Exp{Expi:02d}_{layer}_thr{thread}_Hmaps.pkl"), "rb") as f:
                data = pkl.load(f)
                alphamask_col[(thread, layer)] = data["alphamap_full"]

    mask_sim = edict()
    # quantify properties of each mask.
    for thread in [0, 1, "_cmb"]:
        for layer in ["layer2", "layer3"]:
            w_centroid, w_m_sq_dist, w_m_dist = centroid_standard_dist(alphamask_col[(thread, layer)])
            mask_sim[f"{layer}_{thread}_centroid_i"] = w_centroid[0]
            mask_sim[f"{layer}_{thread}_centroid_j"] = w_centroid[1]
            mask_sim[f"{layer}_{thread}_m_sq_dist"] = w_m_sq_dist
            mask_sim[f"{layer}_{thread}_mean_dist"] = w_m_dist

    for layer in ["layer2", "layer3"]:
        prodmask = alphamask_col[(0, layer)] * alphamask_col[(1, layer)]
        w_centroid, w_m_sq_dist, w_m_dist = centroid_standard_dist(prodmask)
        mask_sim[f"{layer}_intersect_centroid_i"] = w_centroid[0]
        mask_sim[f"{layer}_intersect_centroid_j"] = w_centroid[1]
        mask_sim[f"{layer}_intersect_m_sq_dist"] = w_m_sq_dist
        mask_sim[f"{layer}_intersect_mean_dist"] = w_m_dist

    mask_sim.layer2_0_normmean = np.mean(alphamask_col[(0, "layer2")]) / np.max(alphamask_col[(0, "layer2")])
    mask_sim.layer2_1_normmean = np.mean(alphamask_col[(1, "layer2")]) / np.max(alphamask_col[(1, "layer2")])
    mask_sim.layer3_0_normmean = np.mean(alphamask_col[(0, "layer3")]) / np.max(alphamask_col[(0, "layer3")])
    mask_sim.layer3_1_normmean = np.mean(alphamask_col[(1, "layer3")]) / np.max(alphamask_col[(1, "layer3")])
    mask_sim.layer2_01_iou = norm_mask_iou(alphamask_col[(0, "layer2")], alphamask_col[(1, "layer2")])
    mask_sim.layer3_01_iou = norm_mask_iou(alphamask_col[(0, "layer3")], alphamask_col[(1, "layer3")])
    mask_sim.layer2_01_cov = norm_mask_similarity(alphamask_col[(0, "layer2")], alphamask_col[(1, "layer2")])
    mask_sim.layer3_01_cov = norm_mask_similarity(alphamask_col[(0, "layer3")], alphamask_col[(1, "layer3")])
    mask_sim_col[Expi] = mask_sim
#%%
mask_sim_df = pd.DataFrame(mask_sim_col, ).T
mask_sim_df.to_csv(join(tabdir, "Evol_proto_mask_similarity.csv"))
#%%
meta_masksim_df = meta_df.merge(mask_sim_df, left_index=True, right_index=True)
psth_dif_df = pd.read_csv(join(tabdir, "Evol_psth_dif_df.csv"), index_col=0)
meta_masksim_df = meta_masksim_df.merge(psth_dif_df, left_index=True, right_index=True)
#%%
from core.utils.stats_utils import scatter_corr
from core.utils.stats_utils import ttest_ind_print_df
plt.figure(figsize=[4, 4])
scatter_corr(meta_masksim_df[validmsk], "layer2_0_normmean", "layer2_01_iou", )
plt.tight_layout()
plt.show()
#%%
plt.figure(figsize=[4, 4])
# scatter_corr(meta_masksim_df[validmsk], "layer3_0_std_dist", "layer3_01_iou", )
# scatter_corr(meta_masksim_df[validmsk], "layer3_1_mean_dist", "t_maxinit_1", )
# scatter_corr(meta_masksim_df[validmsk], "layer2_0_mean_dist", "t_maxinit_0", )
# scatter_corr(meta_masksim_df[validmsk&sucsmsk&~bothsucsmsk], "t_FCBG_max_01", "layer3_01_iou", )
# scatter_corr(meta_masksim_df[validmsk&sucsmsk&~bothsucsmsk], "t_maxinit_0", "layer2_01_iou", )
scatter_corr(meta_masksim_df[validmsk&sucsmsk&~bothsucsmsk], "t_maxinit_0", "layer3_01_iou", )
# scatter_corr(meta_masksim_df[validmsk], "t_FCBG_max_01", "layer3_01_iou", )
# scatter_corr(meta_masksim_df[validmsk], "psth_MAE", "layer3_intersect_mean_dist", )
plt.tight_layout()
plt.show()
#%%
plt.figure(figsize=[4, 4])
# scatter_corr(meta_masksim_df[validmsk], "layer3_0_std_dist", "layer3_01_iou", )
# scatter_corr(meta_masksim_df[validmsk], "layer3_1_mean_dist", "t_maxinit_1", )
# scatter_corr(meta_masksim_df[validmsk], "layer2_0_mean_dist", "t_maxinit_0", )
# scatter_corr(meta_masksim_df[validmsk&sucsmsk&~bothsucsmsk], "t_FCBG_max_01", "layer3_01_iou", )
scatter_corr(meta_masksim_df[validmsk&sucsmsk&bothsucsmsk], "t_maxinit_0", "layer2_01_iou", )
# scatter_corr(meta_masksim_df[validmsk&sucsmsk&~bothsucsmsk], "layer3_01_iou", "layer3_1_normmean", )
# scatter_corr(meta_masksim_df[validmsk], "t_FCBG_max_01", "layer3_01_iou", )
# scatter_corr(meta_masksim_df[validmsk], "psth_MAE", "layer3_intersect_mean_dist", )
plt.tight_layout()
plt.show()
#%%
plt.figure(figsize=[4, 4])
scatter_corr(meta_masksim_df[validmsk], "layer2_0_normmean", "t_endinit_0", )
plt.tight_layout()
plt.show()
#%%
ttest_ind_print_df(meta_masksim_df, validmsk&(~bothsucsmsk)&sucsmsk, validmsk&~sucsmsk, "layer3_intersect_std_dist")
ttest_ind_print_df(meta_masksim_df, validmsk&bothsucsmsk&sucsmsk, validmsk&(~bothsucsmsk)&sucsmsk, "layer3_intersect_std_dist")
#%%
ttest_ind_print_df(meta_masksim_df, sucsmsk&ITmsk, ~sucsmsk&ITmsk, "layer3_01_iou")
#%%
ttest_ind_print_df(meta_masksim_df, validmsk&(~bothsucsmsk)&sucsmsk, validmsk&~sucsmsk, "layer2_01_iou")
ttest_ind_print_df(meta_masksim_df, validmsk&bothsucsmsk&sucsmsk, validmsk&(~bothsucsmsk)&sucsmsk, "layer2_01_iou")
ttest_ind_print_df(meta_masksim_df, validmsk&(~bothsucsmsk)&sucsmsk, validmsk&~sucsmsk, "layer3_01_iou")
ttest_ind_print_df(meta_masksim_df, validmsk&bothsucsmsk&sucsmsk, validmsk&(~bothsucsmsk)&sucsmsk, "layer3_01_iou")
#%%
ttest_ind_print_df(meta_masksim_df, (~bothsucsmsk)&sucsmsk, ~sucsmsk, "layer2_01_iou")
ttest_ind_print_df(meta_masksim_df, bothsucsmsk&sucsmsk, (~bothsucsmsk)&sucsmsk, "layer2_01_iou")
ttest_ind_print_df(meta_masksim_df, (~bothsucsmsk)&sucsmsk, ~sucsmsk, "layer3_01_iou")
ttest_ind_print_df(meta_masksim_df, bothsucsmsk&sucsmsk, (~bothsucsmsk)&sucsmsk, "layer3_01_iou")
#%%
ttest_ind_print_df(meta_masksim_df, sucsmsk&V4msk, sucsmsk&ITmsk, "layer2_01_iou")
ttest_ind_print_df(meta_masksim_df, bothsucsmsk&V4msk, bothsucsmsk&ITmsk, "layer2_01_iou")
ttest_ind_print_df(meta_masksim_df, sucsmsk&V4msk, sucsmsk&ITmsk, "layer3_01_iou")
ttest_ind_print_df(meta_masksim_df, bothsucsmsk&V4msk, bothsucsmsk&ITmsk, "layer3_01_iou")
#%%
#%%
figh = plt.figure(figsize=[4, 4])
scatter_corr(mask_sim_df[validmsk&bothsucsmsk&ITmsk],
             "layer3_01_iou", "psth_MAE", )
plt.tight_layout()
plt.show()
#%%
def masks_stripe_plot(df, varnm, msks, labels):
    dfplot = df.copy()
    dfplot["msk"] = ""
    for msk, label in zip(msks, labels):
        dfplot.loc[msk, "msk"] = label
    dfplot = dfplot[dfplot.msk != ""]
    figh = plt.figure(figsize=[3, 4])
    sns.stripplot(data=dfplot, x="msk", y=varnm, linewidth=1, jitter=0.3,
                   edgecolor=None, palette="Set2", order=labels, alpha=0.3)
    # plot points with error bars shifted slightly horizontally
    sns.pointplot(data=dfplot, x="msk", y=varnm, linewidth=1, errorbars="ci",
                     edgecolor="black", palette="Set2", order=labels, capsize=0.3, dodge=True)
    plt.tight_layout()
    plt.show()
    return figh

#%%
figsumdir = r'E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\AlphaMask_proto_analysis\summary'
# iou as a function of paired evol success
for varnm in ["layer2_01_iou", "layer3_01_iou", "layer2_01_cov", "layer3_01_cov"]:
    figh = masks_stripe_plot(meta_masksim_df, varnm,
             [bothsucsmsk&sucsmsk, (~bothsucsmsk)&sucsmsk, ~sucsmsk],
             ["both", "single", "none"])
    plt.sca(figh.gca())
    plt.title("Mask similarity ~ Paired Evol Success")
    plt.tight_layout()
    saveallforms(figsumdir, f"MaskSim_{varnm}_suc", figh, ["png", "pdf", "svg"])
    plt.show()
#%%
