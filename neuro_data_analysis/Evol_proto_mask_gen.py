"""
Code to generate alpha masks for the prototypes
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from os.path import join
from tqdm import tqdm
import pickle as pkl
#%%
cov_root = r"E:\Network_Data_Sync\corrFeatTsr_BigGAN"
attr_dir = r"E:\Network_Data_Sync\BigGAN_FeatAttribution"
protosumdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoSummary"
alphamaskdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\AlphaMasks"
tabdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables"
# os.makedirs(alphamaskdir, exist_ok=True)
#%%
from scipy import ndimage
from CorrFeatTsr_visualize_lib import pad_factor_prod, rectify_tsr, tsr_posneg_factorize, \
    visualize_cctsr_simple, vis_feattsr, vis_feattsr_factor, vis_featvec, vis_featvec_wmaps
from tqdm import trange, tqdm

thread = "_cmb" #1  # 1 # "_cmb"
for Expi in trange(1, 191):
    for thread in [0, 1, "_cmb"]:
        if not os.path.exists(join(cov_root, f"Both_Exp{Expi:02d}_Evol_thr{thread}_res-robust_corrTsr.npz")):
            continue
        #%% Load the data
        corrDict = np.load(join(cov_root, f"Both_Exp{Expi:02d}_Evol_thr{thread}_res-robust_corrTsr.npz"), \
                               allow_pickle=True)
        cctsr_dict = corrDict.get("cctsr").item()
        Ttsr_dict = corrDict.get("Ttsr").item()
        stdtsr_dict = corrDict.get("featStd").item()
        covtsr_dict = {layer: cctsr_dict[layer] * stdtsr_dict[layer] for layer in cctsr_dict}
        #%%
        for layer in ["layer2", "layer3"]:
            Ttsr = Ttsr_dict[layer]
            cctsr = cctsr_dict[layer]
            covtsr = covtsr_dict[layer]
            Ttsr = np.nan_to_num(Ttsr)
            cctsr = np.nan_to_num(cctsr)
            covtsr = np.nan_to_num(covtsr)
            #%%
            NF = 3
            bdr = 1
            nmf_cfgs = dict(
                init="nndsvda", solver="cd", l1_ratio=0, alpha=0, beta_loss="frobenius"
            )  # default
            # nmf_cfgs_alt = dict(
            #     init="nndsvd", solver="mu", l1_ratio=0.8, alpha=0.005, beta_loss="kullback-leibler"
            # ) #"frobenius" ##
            rect_mode = "Tthresh"; thresh = (None, 3)  # rect_mode = "pos"; thresh = (None, None)
            rectstr = rect_mode
            # Direct factorize
            Hmat, Hmaps, ccfactor, FactStat = tsr_posneg_factorize(rectify_tsr(covtsr, rect_mode, thresh, Ttsr=Ttsr),
                                           bdr=bdr, Nfactor=NF, **nmf_cfgs, do_plot=False, do_save=False,)
            DR_Wtsr = pad_factor_prod(Hmaps, ccfactor, bdr=bdr)
            Hmaps_pad = np.pad(Hmaps, ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant", constant_values=0)
            # turn the 3 channel Hmaps into a single channel alpha map
            # the exponent of 2 is to make the alpha map more "peaky", less "flat"
            factor_L2 = (ccfactor**2).sum(0)
            alphamap = ((Hmaps_pad**2) * factor_L2[None, None, ]).sum(axis=2)
            # resize the alpha map to 224 x 224
            alphamap_full = ndimage.zoom(alphamap, 224 / alphamap.shape[0], order=2)
            #%%
            plt.imsave(join(alphamaskdir, f"Exp{Expi:02d}_{layer}_thr{thread}_alpha.png"),
                       alphamap_full / np.max(alphamap_full))
            plt.imsave(join(alphamaskdir, f"Exp{Expi:02d}_{layer}_thr{thread}_Hmaps.png"),
                          Hmaps_pad / np.max(Hmaps_pad))

            with open(join(alphamaskdir, f"Exp{Expi:02d}_{layer}_thr{thread}_Hmaps.pkl"), "wb") as f:
                pkl.dump(
                    dict(Hmaps=Hmaps, Hmaps_pad=Hmaps_pad, ccfactor=ccfactor, FactStat=FactStat,
                         alphamap_full=alphamap_full, alphamap=alphamap), f
                )
#%%
# for all files inf alphamaskdir, rename pattern layerlayer2 to layer2 and layerlayer3 to layer3
import os
from os.path import join
alphamaskdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\AlphaMasks"
for fn in os.listdir(alphamaskdir):
    # if fn.endswith(".pkl"):
    os.rename(join(alphamaskdir, fn), join(alphamaskdir, fn.replace("layerlayer", "layer")))


#%%
plt.figure(figsize=(8, 8))
plt.imshow(Hmaps_pad / np.max(Hmaps_pad))
plt.show()
#%%
plt.figure(figsize=(8, 8))
plt.imshow(alphamap_full / np.max(alphamap_full))
plt.colorbar()
plt.show()



#%%
alphamap = (Hmaps_pad**2).sum(axis=2)
plt.figure(figsize=(8, 8))
plt.imshow(alphamap / np.max(alphamap))
plt.colorbar()
plt.show()
#%%
alphamap_full = ndimage.zoom(alphamap, 224 / alphamap.shape[0], order=3)
plt.figure(figsize=(8, 8))
plt.imshow(alphamap_full / np.max(alphamap_full))
plt.colorbar()
plt.show()

