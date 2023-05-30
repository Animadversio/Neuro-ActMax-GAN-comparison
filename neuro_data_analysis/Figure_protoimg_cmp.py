"""
Script to generate proto images comparison figure for the paper
"""
import os
from os.path import join
from tqdm import trange, tqdm
import pickle as pkl
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from lpips import LPIPS
import torch.nn.functional as F
import matplotlib.pyplot as plt
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms
from core.utils.montage_utils import crop_all_from_montage, make_grid_np, build_montages, crop_from_montage, make_grid
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from neuro_data_analysis.image_comparison_lib import compare_imgs_cnn, compare_imgs_cnn_featmsk, \
    compare_imgs_vit, compare_imgs_LPIPS

#%%
def parse_montage(mtg):
    mtg = mtg.astype(np.float32) / 255.0
    FC_maxblk = crop_from_montage(mtg, (0, 0), 224, 0)
    FC_maxblk_avg = crop_from_montage(mtg, (0, 1), 224, 0)
    FC_reevol_G = crop_from_montage(mtg, (0, 2), 224, 0)
    FC_reevol_pix = crop_from_montage(mtg, (0, 3), 224, 0)
    BG_maxblk = crop_from_montage(mtg, (1, 0), 224, 0)
    BG_maxblk_avg = crop_from_montage(mtg, (1, 1), 224, 0)
    BG_reevol_G = crop_from_montage(mtg, (1, 2), 224, 0)
    BG_reevol_pix = crop_from_montage(mtg, (1, 3), 224, 0)
    both_reevol_G = crop_from_montage(mtg, (2, 2), 224, 0)
    both_reevol_pix = crop_from_montage(mtg, (2, 3), 224, 0)
    return FC_maxblk, FC_maxblk_avg, FC_reevol_G, FC_reevol_pix, \
           BG_maxblk, BG_maxblk_avg, BG_reevol_G, BG_reevol_pix, \
           both_reevol_G, both_reevol_pix


def showimg(img, bar=False):
    plt.imshow(img)
    plt.axis("off")
    if bar:
        plt.colorbar()
    plt.show()

#%%
protosumdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoSummary"
alphamaskdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\AlphaMasks"
outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoImage_cmp\scratch"
figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoImage_cmp"
tabdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables"
meta_act_df = pd.read_csv(join(tabdir, "meta_activation_stats.csv"), )

"""Extract examples to make a montage figure for the paper"""

for Expi in trange(1, 191):
    if not os.path.exists(join(protosumdir, f"Exp{Expi}_proto_attr_montage.jpg")):
        # raise ValueError("Montage not found")
        continue
    mtg = plt.imread(join(protosumdir, f"Exp{Expi}_proto_attr_montage.jpg"))
    FC_maxblk, FC_maxblk_avg, FC_reevol_G, FC_reevol_pix, \
               BG_maxblk, BG_maxblk_avg, BG_reevol_G, BG_reevol_pix, \
               both_reevol_G, both_reevol_pix = parse_montage(mtg)
#%%
def example_proto_montage(example_list, figdir, protosumdir=protosumdir, suffix=""):
    """"""
    exemplar_col = []
    exemplar_mtg_col = []
    maxblk_mtg_col = []
    for Expi in tqdm(example_list):
        mtg = plt.imread(join(protosumdir, f"Exp{Expi}_proto_attr_montage.jpg"))
        FC_maxblk, FC_maxblk_avg, FC_reevol_G, FC_reevol_pix, \
            BG_maxblk, BG_maxblk_avg, BG_reevol_G, BG_reevol_pix, \
            both_reevol_G, both_reevol_pix = parse_montage(mtg)
        exemplar_col.append((FC_maxblk, BG_maxblk, FC_reevol_G, BG_reevol_G, FC_reevol_pix, BG_reevol_pix))
        exemplar_mtg_col.extend([FC_reevol_pix, BG_reevol_pix])
        maxblk_mtg_col.extend([FC_maxblk, BG_maxblk])

    mtg_pool = make_grid_np(exemplar_mtg_col, nrow=2, )
    plt.imsave(join(figdir, f"exemplar_mtg_pool{suffix}.jpg"), mtg_pool)
    maxblk_mtg_pool = make_grid_np(maxblk_mtg_col, nrow=2, )
    plt.imsave(join(figdir, f"maxblk_mtg_pool{suffix}.jpg"), maxblk_mtg_pool)

    mtg_pool_T = make_grid_np(exemplar_mtg_col, nrow=len(example_list), rowfirst=False)
    plt.imsave(join(figdir, f"exemplar_mtg_pool_T{suffix}.jpg"), mtg_pool_T)
    maxblk_mtg_pool_T = make_grid_np(maxblk_mtg_col, nrow=len(example_list), rowfirst=False)
    plt.imsave(join(figdir, f"maxblk_mtg_pool_T{suffix}.jpg"), maxblk_mtg_pool_T)
    return exemplar_col, exemplar_mtg_col, maxblk_mtg_col


def alpha_mask_montage(example_list, figdir, alphamaskdir=alphamaskdir, suffix="", resize_pix=256):
    mask1_col = []
    mask2_col = []
    consensus_col = []
    for Expi in tqdm(example_list):
        data = torch.load(join(alphamaskdir, f"Exp{Expi}_consensus_maps.pt"))
        mask1 = data['layer3']["cov1"]
        mask2 = data['layer3']["cov2"]
        consensus = data['layer3']["prodcov"]
        consensus = consensus / consensus.max()
        mask1 = mask1 / mask1.max()
        mask2 = mask2 / mask2.max()
        mask1_col.append(mask1)
        mask2_col.append(mask2)
        consensus_col.append(consensus)


    mask_stack = torch.stack(mask1_col + mask2_col)[:, None]
    mask_stack_rsz = F.interpolate(mask_stack, size=(resize_pix, resize_pix), mode='bilinear', align_corners=True)
    mask_pool12 = make_grid(mask_stack_rsz, nrow=len(example_list), padding=2)
    # create a png file with mask_pool12 as 1 - transparency alpha channel and the RGB as black
    mask_pool12_np = mask_pool12.permute(1, 2, 0).numpy()
    mask_pool12_RGBA = np.concatenate([np.zeros_like(mask_pool12_np), 1 - mask_pool12_np[:, :, :1]], axis=2)
    mask_pool12_RGBA = np.ascontiguousarray(mask_pool12_RGBA)
    plt.imsave(join(figdir, f"mask_pool12_transparent{suffix}.png"), mask_pool12_RGBA)
    # for consensus
    mask_stack = torch.stack(consensus_col)[:, None]
    mask_stack_rsz = F.interpolate(mask_stack, size=(resize_pix, resize_pix), mode='bilinear', align_corners=True)
    mask_consensus = make_grid(mask_stack_rsz, nrow=len(example_list), padding=2)
    mask_consensus_np = mask_consensus.permute(1, 2, 0).numpy()
    mask_consensus_RGBA = np.concatenate([np.zeros_like(mask_consensus_np), 1 - mask_consensus_np[:, :, :1]], axis=2)
    mask_consensus_RGBA = np.ascontiguousarray(mask_consensus_RGBA)
    plt.imsave(join(figdir, f"mask_consensus_transparent{suffix}.png"), mask_consensus_RGBA)
    return mask1_col, mask2_col, consensus_col


savedir = r"E:\Network_Data_Sync\corrFeatTsr_BigGAN\consensus"
example_list_brief = [3, 65, 66, 74, 111, 118, 174, 175, 113, 64, ]  # 28
exemplar_col, exemplar_mtg_col, maxblk_mtg_col = example_proto_montage(example_list_brief, figdir, suffix="_brief")
mask1_col, mask2_col, consensus_col = alpha_mask_montage(example_list_brief, figdir, alphamaskdir=savedir, suffix="_brief")
#%% Good in vivo examples
example_list = [3, 64, 65, 66, 74, 79, 111, 113, 118, 155, 174, 175]  # 28
exemplar_col = []
exemplar_mtg_col = []
maxblk_mtg_col = []
for Expi in tqdm(example_list):
    mtg = plt.imread(join(protosumdir, f"Exp{Expi}_proto_attr_montage.jpg"))
    FC_maxblk, FC_maxblk_avg, FC_reevol_G, FC_reevol_pix, \
               BG_maxblk, BG_maxblk_avg, BG_reevol_G, BG_reevol_pix, \
               both_reevol_G, both_reevol_pix = parse_montage(mtg)
    exemplar_col.append((FC_maxblk, BG_maxblk, FC_reevol_G, BG_reevol_G, FC_reevol_pix, BG_reevol_pix))
    exemplar_mtg_col.extend([FC_reevol_pix, BG_reevol_pix])
    maxblk_mtg_col.extend([FC_maxblk, BG_maxblk])

#%%
mtg_pool = make_grid_np(exemplar_mtg_col, nrow=2,)
plt.imsave(join(figdir, "exemplar_mtg_pool.jpg"), mtg_pool)
maxblk_mtg_pool = make_grid_np(maxblk_mtg_col, nrow=2, )
plt.imsave(join(figdir, "maxblk_mtg_pool.jpg"), maxblk_mtg_pool)

#%%
mtg_pool_T = make_grid_np(exemplar_mtg_col, nrow=len(example_list), rowfirst=False)
plt.imsave(join(figdir, "exemplar_mtg_pool_T.jpg"), mtg_pool_T)
maxblk_mtg_pool_T = make_grid_np(maxblk_mtg_col, nrow=len(example_list), rowfirst=False)
plt.imsave(join(figdir, "maxblk_mtg_pool_T.jpg"), maxblk_mtg_pool_T)

#%% Consensus masks
savedir = r"E:\Network_Data_Sync\corrFeatTsr_BigGAN\consensus"
example_list = [3, 64, 65, 66, 74, 79, 111, 113, 118, 155, 174, 175]  # 28
mask1_col = []
mask2_col = []
consensus_col = []
for Expi in tqdm(example_list):
    data = torch.load(join(savedir, f"Exp{Expi}_consensus_maps.pt"))
    mask1 = data['layer3']["cov1"]
    mask2 = data['layer3']["cov2"]
    consensus = data['layer3']["prodcov"]
    consensus = consensus / consensus.max()
    mask1 = mask1 / mask1.max()
    mask2 = mask2 / mask2.max()
    mask1_col.append(mask1)
    mask2_col.append(mask2)
    consensus_col.append(consensus)

#%%
mask1_pool = make_grid(torch.stack(mask1_col)[:,None], nrow=12,)
mask2_pool = make_grid(torch.stack(mask2_col)[:,None], nrow=12,)
consensus_pool = make_grid(torch.stack(consensus_col)[:,None], nrow=12,)
#%%
mask_stack = torch.stack(mask1_col+mask2_col+consensus_col)[:,None]
mask_stack_rsz = F.interpolate(mask_stack, size=(224,224), mode='bilinear', align_corners=True)
mask_pool = make_grid(mask_stack_rsz, nrow=12, padding=2)
#%%
mask_stack = torch.stack(mask1_col+mask2_col)[:,None]
mask_stack_rsz = F.interpolate(mask_stack, size=(256,256), mode='bilinear', align_corners=True)
mask_pool12 = make_grid(mask_stack_rsz, nrow=12, padding=2)
#%%
# create a png file with mask_pool12 as 1 - transparency alpha channel and the RGB as black
mask_pool12_np = mask_pool12.permute(1, 2, 0).numpy()
mask_pool12_RGBA = np.concatenate([np.zeros_like(mask_pool12_np), 1 - mask_pool12_np[:,:,:1]], axis=2)
mask_pool12_RGBA = np.ascontiguousarray(mask_pool12_RGBA)
plt.imsave(join(figdir, "mask_pool12_transparent.png"), mask_pool12_RGBA)
#%%
# for consensus
mask_stack = torch.stack(consensus_col)[:,None]
mask_stack_rsz = F.interpolate(mask_stack, size=(256,256), mode='bilinear', align_corners=True)
mask_consensus = make_grid(mask_stack_rsz, nrow=12, padding=2)
mask_consensus_np = mask_consensus.permute(1,2,0).numpy()
mask_consensus_RGBA = np.concatenate([np.zeros_like(mask_consensus_np), 1 - mask_consensus_np[:,:,:1]], axis=2)
mask_consensus_RGBA = np.ascontiguousarray(mask_consensus_RGBA)
plt.imsave(join(figdir, "mask_consensus_transparent.png"), mask_consensus_RGBA)
#%%
plt.imshow(mask_pool12_RGBA)
plt.axis("off")
plt.show()
#%%
figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Proto_covtsr_consensus"
save_imgrid(mask_pool, join(figdir, "consensus_mask_pool.png"), nrow=12, padding=2)
#%%
print(mask_pool.shape)
plt.imshow(mask_pool.permute(1,2,0))
plt.axis("off")
plt.show()
#%%

