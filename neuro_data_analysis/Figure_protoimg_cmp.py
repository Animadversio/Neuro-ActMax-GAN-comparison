"""
Script to generate proto images comparison figure for the paper
"""
import os
import torch
import numpy as np
from os.path import join
from tqdm import tqdm
import pandas as pd
from lpips import LPIPS
import matplotlib.pyplot as plt
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms
from core.utils.montage_utils import crop_all_from_montage, make_grid_np, build_montages, crop_from_montage
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

import os
from os.path import join
from tqdm import trange, tqdm
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

#%%

