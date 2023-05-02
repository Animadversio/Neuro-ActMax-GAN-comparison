import os
import numpy as np
import torch
from os.path import join
from CorrFeatTsr_visualize_lib import tsr_posneg_factorize, visualize_cctsr_simple, \
    vis_feattsr, vis_feattsr_factor, vis_featvec, vis_featvec_wmaps, pad_factor_prod, rectify_tsr
from core.utils.CNN_scorers import load_featnet
from core.utils.GAN_utils import upconvGAN
from neuro_data_analysis.neural_data_lib import get_expstr, load_neural_data, parse_montage
from core.utils.plot_utils import saveallforms
from tqdm import trange, tqdm
import einops
import matplotlib.pyplot as plt

#%%
cov_root = r"E:\Network_Data_Sync\corrFeatTsr_BigGAN"
montage_dir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoSummary"
_, BFEStats = load_neural_data()
#%%
outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Proto_covtsr_similarity"
Animal = "Both"
Expi = 111
for Expi in trange(1, 191):  # range(160, 191):
    # thread =  "_cmb"
    try:
        explabel = get_expstr(BFEStats, Expi)
    except:
        continue
    print(explabel)
    showfig = True
    cctsr_dict = {}
    Ttsr_dict = {}
    stdtsr_dict = {}
    covtsr_dict = {}
    for thread in [0, 1]:  # , "_cmb"
        corrDict = np.load(join(cov_root, "Both_Exp%02d_Evol_thr%s_res-robust_corrTsr.npz" \
                                    % (Expi, thread)), allow_pickle=True)
        cctsr_dict[thread] = corrDict.get("cctsr").item()
        cctsr_dict[thread] = {layer: np.nan_to_num(tsr, nan=0.0) for layer, tsr in cctsr_dict[thread].items()}
        Ttsr_dict[thread] = corrDict.get("Ttsr").item()
        stdtsr_dict[thread] = corrDict.get("featStd").item()
        covtsr_dict[thread] = {layer: cctsr_dict[thread][layer] * stdtsr_dict[thread][layer]
                               for layer in cctsr_dict[thread]}

    #%%
    # einops.rearrange(covtsr_dict[0]["layer4"], "C H W -> C (H W)")
    # einops.rearrange(covtsr_dict[1]["layer4"], "C H W -> C (H W)")
    # layer_str = "layer4"
    #%%
    mtg_S = parse_montage(plt.imread(join(montage_dir, "Exp%d_proto_attr_montage.jpg" % Expi)))
    figh, axs = plt.subplots(2, 4, figsize=[10, 6])
    for i, key in enumerate(["FC_maxblk", "FC_reevol_G", "BG_maxblk", "BG_reevol_G"]):
        axs[0, i].imshow(mtg_S[key])
        axs[0, i].set_title(key)
        axs[0, i].axis("off")
    for i, layer_str in enumerate(["layer1", "layer2", "layer3", "layer4"]):
        C, H, W = covtsr_dict[0][layer_str].shape
        covtsr_corr = \
            np.corrcoef(einops.rearrange(covtsr_dict[0][layer_str], "C H W -> C (H W)"),
                        einops.rearrange(covtsr_dict[1][layer_str], "C H W -> C (H W)"),
                        rowvar=False)

        corr_map = covtsr_corr[range(H*W), range(H*W, 2*H*W)].reshape(H, W)
        im = axs[1, i].imshow(corr_map)
        axs[1, i].set_title(layer_str)
        # add colorbar to axis
        plt.colorbar(im, ax=axs[1, i])
    plt.suptitle(f"Correlation of Covariance Tensor\n{explabel}")
    plt.tight_layout()
    saveallforms(outdir, f"Exp{Expi:03d}_both_thread_covtsr_corr", figh=figh,)
    plt.show()
