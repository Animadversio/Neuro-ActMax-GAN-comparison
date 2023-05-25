import os
from os.path import join
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from core.utils.plot_utils import saveallforms
from core.utils.stats_utils import ttest_ind_print_df, ttest_rel_print_df, paired_strip_plot
from neuro_data_analysis.neural_data_lib import get_expstr, load_neural_data, parse_montage
from pathlib import Path
from easydict import EasyDict as edict

statdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_activation_cmp"
meta_df = pd.read_csv(join(statdir, "meta_stats.csv"), index_col=0)
#%%
_, BFEStats = load_neural_data()
#%%
import torch
from neuro_data_analysis.neural_data_lib import get_expstr

# def crop_center(corrmap, frac=0.1):
#     H, W = corrmap.shape
#     corrmap_crop = corrmap[int(H*frac):int(H*(1-frac)), int(W*frac):int(W*(1-frac))]
#     return corrmap_crop

layernames = ["layer1", "layer2", "layer3", "layer4"]
cov_root = r"E:\Network_Data_Sync\corrFeatTsr_BigGAN"
savedir = r"E:\Network_Data_Sync\corrFeatTsr_BigGAN\consensus"
cov_stat_col = []
for Expi in trange(1, 191):
    if Expi not in meta_df.index:
        print(f"Exp {Expi} not in the meta_df. Skip")
        continue
    expstr = get_expstr(BFEStats, Expi)
    ccdict1 = np.load(join(cov_root, f"Both_Exp{Expi:02d}_Evol_thr0_res-robust_corrTsr.npz"), allow_pickle=True)
    ccdict2 = np.load(join(cov_root, f"Both_Exp{Expi:02d}_Evol_thr1_res-robust_corrTsr.npz"), allow_pickle=True)
    covtsrs1 = {}
    covtsrs2 = {}
    for i, layer in enumerate(layernames):
        covtsrs1[layer] = ccdict1["cctsr"].item()[layer] * ccdict1["featStd"].item()[layer]
        np.nan_to_num(covtsrs1[layer], copy=False, nan=0.0)
        covtsrs1[layer] = torch.from_numpy(covtsrs1[layer])
        covtsrs2[layer] = ccdict2["cctsr"].item()[layer] * ccdict2["featStd"].item()[layer]
        np.nan_to_num(covtsrs2[layer], copy=False, nan=0.0)
        covtsrs2[layer] = torch.from_numpy(covtsrs2[layer])
    # #%%
    # figh, axs = plt.subplots(4, 10, figsize=[25, 12])
    # for i, layer in enumerate(layernames):
    #     axs[i, 0].imshow(covtsrs1[layer].norm(dim=0) ** 2)
    #     axs[i, 0].set_title(f"{layer} cov1")
    #     axs[i, 1].imshow(covtsrs2[layer].norm(dim=0) ** 2)
    #     axs[i, 1].set_title(f"{layer} cov2")
    #     axs[i, 2].imshow(torch.clamp(covtsrs1[layer], 0).norm(dim=0) ** 2)
    #     axs[i, 2].set_title(f"{layer} cov1 relu")
    #     axs[i, 3].imshow(torch.clamp(covtsrs2[layer], 0).norm(dim=0) ** 2)
    #     axs[i, 3].set_title(f"{layer} cov2 relu")
    #     prodcovtsr = covtsrs1[layer] * covtsrs2[layer]
    #     dotprodcovtsr = prodcovtsr.sum(dim=0)
    #     mincovtsr = torch.min(covtsrs1[layer], covtsrs2[layer])
    #     absmincovtsr = torch.min(covtsrs1[layer].abs(), covtsrs2[layer].abs())
    #     cosmap = torch.cosine_similarity(covtsrs1[layer], covtsrs2[layer], dim=0)
    #     axs[i, 4].imshow(prodcovtsr.norm(dim=0) ** 2)
    #     axs[i, 4].set_title(f"{layer} cov1 * cov2")
    #     axs[i, 5].imshow(torch.clamp(prodcovtsr, 0).norm(dim=0) ** 2)
    #     axs[i, 5].set_title(f"{layer} Relu (cov1 * cov2)")
    #     axs[i, 6].imshow(dotprodcovtsr ** 2)
    #     axs[i, 6].set_title(f"{layer} Dot(cov1 , cov2)")
    #     axs[i, 7].imshow(torch.clamp(mincovtsr, 0).norm(dim=0) ** 2)
    #     axs[i, 7].set_title(f"{layer} Relu min(cov1 , cov2)")
    #     axs[i, 8].imshow(absmincovtsr.norm(dim=0) ** 2)
    #     axs[i, 8].set_title(f"{layer} min(|cov1| , |cov2|)")
    #     axs[i, 9].imshow(cosmap)
    #     axs[i, 9].set_title(f"{layer} Cosine(cov1 , cov2)")
    # plt.suptitle(f"Exp{Expi}", fontsize=18)
    # plt.suptitle(expstr, fontsize=18)
    # plt.tight_layout()
    # saveallforms(savedir, f"Exp{Expi}_consensus_mask", figh, ["png", "pdf"])
    # plt.show()
    #%%
    Sall = edict()
    for layer in layernames:
        cov1 = covtsrs1[layer]
        cov2 = covtsrs2[layer]
        S = edict()
        S.cov1 = cov1.norm(dim=0) ** 2
        S.cov2 = cov2.norm(dim=0) ** 2
        S.cov1_relu = torch.clamp(cov1, 0).norm(dim=0) ** 2
        S.cov2_relu = torch.clamp(cov2, 0).norm(dim=0) ** 2
        prodcov = cov1 * cov2
        dotcov = prodcov.sum(dim=0)
        S.prodcov = prodcov.norm(dim=0) ** 2
        S.prodcov_relu = torch.clamp(prodcov, 0).norm(dim=0) ** 2
        S.dotcov = dotcov ** 2
        mincov = torch.min(cov1, cov2)
        absmincov = torch.min(cov1.abs(), cov2.abs())
        S.mincov_relu = torch.clamp(mincov, 0).norm(dim=0) ** 2
        S.absmincov = absmincov.norm(dim=0) ** 2
        cosmap = torch.cosine_similarity(cov1, cov2, dim=0)
        S.cosmap = cosmap
        Sall[layer] = S
    # %%
    torch.save(Sall, Path(savedir) / f"Exp{Expi}_consensus_maps.pt")
    # S["Expi"] = Expi
    # for layer in ["layer1", "layer2", "layer3", "layer4"]:
    #     corr_map = corr_map_dict[layer]
        # corr_map_crop = crop_center(corr_map)
        # S[layer] = corr_map_crop.mean()
    # for i, layer in enumerate(layernames):
#%%
