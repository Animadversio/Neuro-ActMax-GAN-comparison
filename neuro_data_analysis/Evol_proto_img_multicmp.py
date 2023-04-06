#%%
"""
Compute various distances between images in the evolution and prototype summary figures
predecessor of image_metric_development
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
from core.utils.dataset_utils import ImagePathDataset, ImageFolder
from timm import list_models, create_model
from torchvision.models import resnet50, alexnet
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
#%%
Dist = LPIPS(net='squeeze', spatial=True,)
Dist = Dist.cuda().eval()
Dist.requires_grad_(False)
#%%
protosumdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoSummary"
tabdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables"
meta_act_df = pd.read_csv(join(tabdir, "meta_activation_stats.csv"), )
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

from neuro_data_analysis.image_comparison_lib import compare_imgs_cnn, compare_imgs_cnn_featmsk, \
    compare_imgs_vit, compare_imgs_LPIPS
#%%
#%%
from core.utils.CNN_scorers import load_featnet
# cnnmodel = resnet50(pretrained=True)
cnnmodel, _ = load_featnet("resnet50_linf8",)
# get_graph_node_names(cnnmodel)
fetcher_cnn = create_feature_extractor(cnnmodel, ['layer3', "layer4", "avgpool"])
fetcher_cnn = fetcher_cnn.cuda().eval()
#%%
alexnetmodel = alexnet(pretrained=True)
get_graph_node_names(alexnetmodel)
fetcher_alexnet = create_feature_extractor(alexnetmodel, ['classifier.2', "classifier.4",])
fetcher_alexnet = fetcher_alexnet.cuda().eval()
#%% VIT-DINO
from timm.models.vision_transformer import VisionTransformer
vitmodel = create_model('vit_base_patch16_224_dino', pretrained=True, )
# get_graph_node_names(model)
fetcher_vit = create_feature_extractor(vitmodel, ['blocks', 'norm']) # ['blocks', 'norm']
fetcher_vit = fetcher_vit.cuda().eval()
#%% CLIP
clipmodel = create_model('vit_large_patch14_224_clip_laion2b', pretrained=True, )
# get_graph_node_names(model)
fetcher_clip = create_feature_extractor(clipmodel, ['blocks', 'norm']) # ['blocks', 'norm']
fetcher_clip = fetcher_clip.cuda().eval()

#%%
from tqdm import trange, tqdm
Expi = 66
imdist_col = []
for Expi in trange(1, 191): # [118]: #
    if not os.path.exists(join(protosumdir, f"Exp{Expi}_proto_attr_montage.jpg")):
        continue
    mtg = plt.imread(join(protosumdir, f"Exp{Expi}_proto_attr_montage.jpg"))
    FC_maxblk, FC_maxblk_avg, FC_reevol_G, FC_reevol_pix, \
               BG_maxblk, BG_maxblk_avg, BG_reevol_G, BG_reevol_pix, \
               both_reevol_G, both_reevol_pix = parse_montage(mtg)

    #%%
    # original images
    maxblk_cossim_fc6 = compare_imgs_cnn(FC_maxblk, BG_maxblk, fetcher_alexnet,
                                         featkey='classifier.2', metric="cosine")
    maxblk_cossim_fc7 = compare_imgs_cnn(FC_maxblk, BG_maxblk, fetcher_alexnet,
                                         featkey='classifier.4', metric="cosine")
    maxblk_cossim_tsr_L3 = compare_imgs_cnn(FC_maxblk, BG_maxblk, fetcher_cnn, featkey='layer3', metric="cosine")
    maxblk_cossim_tsr_L4 = compare_imgs_cnn(FC_maxblk, BG_maxblk, fetcher_cnn, featkey='layer4', metric="cosine")
    maxblk_cossim_scalr = compare_imgs_cnn(FC_maxblk, BG_maxblk, fetcher_cnn, featkey='avgpool', metric="cosine")
    maxblk_cossim_vitcls, maxblk_cossim_vitmap = compare_imgs_vit(FC_maxblk, BG_maxblk, fetcher_vit, featkey='blocks',
                                                    metric="cosine")

    # reevol images
    reevol_cossim_fc6 = compare_imgs_cnn(FC_reevol_G, BG_reevol_G, fetcher_alexnet,
                                         featkey='classifier.2', metric="cosine")
    reevol_cossim_fc7 = compare_imgs_cnn(FC_reevol_G, BG_reevol_G, fetcher_alexnet,
                                         featkey='classifier.4', metric="cosine")
    reevol_cossim_tsr_L3 = compare_imgs_cnn(FC_reevol_G, BG_reevol_G, fetcher_cnn, featkey='layer3', metric="cosine")
    reevol_cossim_tsr_L4 = compare_imgs_cnn(FC_reevol_G, BG_reevol_G, fetcher_cnn, featkey='layer4', metric="cosine")
    reevol_cossim_scalr = compare_imgs_cnn(FC_reevol_G, BG_reevol_G, fetcher_cnn, featkey='avgpool', metric="cosine")
    reevol_cossim_vitcls, reevol_cossim_vitmap = compare_imgs_vit(FC_reevol_G, BG_reevol_G, fetcher_vit, featkey='blocks', metric="cosine")

    #%%
    # TODO: visualize the cossim maps / dist maps. Find ways to get rid of border artifacts
    #%%
    stats = {"Expi" : Expi,
            "cosine_reevol_resnet_L3_m": reevol_cossim_tsr_L3.mean().item(),
            "cosine_reevol_resnet_L4_m": reevol_cossim_tsr_L4.mean().item(),
            "cosine_reevol_resnet_avgpool": reevol_cossim_scalr.mean().item(),
            "cosine_reevol_alexnet_fc6": reevol_cossim_fc6.mean().item(),
            "cosine_reevol_alexnet_fc7": reevol_cossim_fc7.mean().item(),
            "cosine_reevol_vit_cls": reevol_cossim_vitcls,
            "cosine_reevol_vit_token_m": reevol_cossim_vitmap.mean().item(),
            "cosine_maxblk_resnet_L3_m": maxblk_cossim_tsr_L3.mean().item(),
            "cosine_maxblk_resnet_L4_m": maxblk_cossim_tsr_L4.mean().item(),
            "cosine_maxblk_resnet_avgpool": maxblk_cossim_scalr.mean().item(),
            "cosine_maxblk_alexnet_fc6": maxblk_cossim_fc6.mean().item(),
            "cosine_maxblk_alexnet_fc7": maxblk_cossim_fc7.mean().item(),
            "cosine_maxblk_vit_cls": maxblk_cossim_vitcls,
            "cosine_maxblk_vit_token_m": maxblk_cossim_vitmap.mean().item(),
             }
    imdist_col.append(stats)

#%%
imdist_df = pd.DataFrame(imdist_col)
imdist_df.to_csv(join(tabdir, "proto_imdist_df.csv"))
#%%

meta_act_df = pd.read_csv(join(tabdir, "meta_activation_stats.csv"), )
#%%
meta_imdist_df = pd.merge(meta_act_df, imdist_df, on="Expi")
#%%
succmsk = (meta_act_df.p_maxinit_0 < 0.01) & \
      (meta_act_df.p_maxinit_1 < 0.01)
nonemsk = (meta_act_df.p_maxinit_0 > 0.01) & \
        (meta_act_df.p_maxinit_1 > 0.01)
V1msk = meta_act_df.visual_area == "V1"
V4msk = meta_act_df.visual_area == "V4"
ITmsk = meta_act_df.visual_area == "IT"
cmpmsk = (meta_act_df.maxrsp_1_mean - meta_act_df.maxrsp_0_mean).abs() \
         < (meta_act_df.maxrsp_0_sem + meta_act_df.maxrsp_1_sem)
#%%
"""without any masks, the mean cosine similarity doesn't have effect / different
Esp. the non success ones the two threads are even more similar! 
"""
pd.concat([imdist_df[succmsk].mean(),
           imdist_df[~succmsk].mean()], axis=1)
#%%


def normalize_alphamask(img):
    img = img.astype(np.float32)
    img = img / img.max()
    return img
#%% add alpha mask to the features. or add to the images
import pickle as pkl
from easydict import EasyDict as edict
from tqdm import trange, tqdm
import scipy.ndimage as ndimage
alphamaskdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\AlphaMasks"
imdist_col = []
for Expi in trange(1, 191): # [118]: #
    if not os.path.exists(join(protosumdir, f"Exp{Expi}_proto_attr_montage.jpg")):
        continue
    mtg = plt.imread(join(protosumdir, f"Exp{Expi}_proto_attr_montage.jpg"))
    FC_maxblk, FC_maxblk_avg, FC_reevol_G, FC_reevol_pix, \
               BG_maxblk, BG_maxblk_avg, BG_reevol_G, BG_reevol_pix, \
               both_reevol_G, both_reevol_pix = parse_montage(mtg)

    data = pkl.load(open(join(alphamaskdir, f"Exp{Expi:02d}_layer3_thr0_Hmaps.pkl"), "rb"))
    alphamap0 = data["alphamap"]
    alphamap_full0 = data["alphamap_full"]
    data = pkl.load(open(join(alphamaskdir, f"Exp{Expi:02d}_layer3_thr1_Hmaps.pkl"), "rb"))
    alphamap1 = data["alphamap"]
    alphamap_full1 = data["alphamap_full"]
    # raise NotImplementedError("TODO: add alpha mask to the images")
    alphamap0_rsz = ndimage.zoom(alphamap0, 16 / 14, order=1)
    alphamap1_rsz = ndimage.zoom(alphamap1, 16 / 14, order=1)
    #%%
    # original images
    maxblk_cossim_tsr_L3 = compare_imgs_cnn(FC_maxblk, BG_maxblk, fetcher_cnn, featkey='layer3', metric="cosine")
    maxblk_cossim_vitcls, maxblk_cossim_vitmap = compare_imgs_vit(FC_maxblk, BG_maxblk, fetcher_vit,
                                                  featkey='norm', metric="cosine") # blocks
    maxblk_cossim_clipcls, maxblk_cossim_cliptoken = compare_imgs_vit(FC_maxblk, BG_maxblk, fetcher_clip,
                                                    featkey='norm', metric="cosine") # blocks
    # reevol images
    reevol_cossim_tsr_L3 = compare_imgs_cnn(FC_reevol_G, BG_reevol_G, fetcher_cnn, featkey='layer3', metric="cosine")
    reevol_cossim_vitcls, reevol_cossim_vitmap = compare_imgs_vit(FC_reevol_G, BG_reevol_G, fetcher_vit,
                                                  featkey='norm', metric="cosine") # blocks
    reevol_cossim_clipcls, reevol_cossim_cliptoken = compare_imgs_vit(FC_reevol_G, BG_reevol_G, fetcher_clip,
                                                    featkey='norm', metric="cosine") # blocks

    maxblk_cossim_tsr_L3_alpha0mean = maxblk_cossim_tsr_L3 * alphamap0 / alphamap0.mean()
    reevol_cossim_tsr_L3_alpha0mean = reevol_cossim_tsr_L3 * alphamap0 / alphamap0.mean()
    maxblk_cossim_vitmap_alpha0mean = maxblk_cossim_vitmap * alphamap0 / alphamap0.mean()
    reevol_cossim_vitmap_alpha0mean = reevol_cossim_vitmap * alphamap0 / alphamap0.mean()
    maxblk_cossim_clipmap_alpha0mean = maxblk_cossim_cliptoken * alphamap0_rsz / alphamap0_rsz.mean()
    reevol_cossim_clipmap_alpha0mean = reevol_cossim_cliptoken * alphamap0_rsz / alphamap0_rsz.mean()
    maxblk_cossim_tsr_L3_alpha1mean = maxblk_cossim_tsr_L3 * alphamap1 / alphamap1.mean()
    reevol_cossim_tsr_L3_alpha1mean = reevol_cossim_tsr_L3 * alphamap1 / alphamap1.mean()
    maxblk_cossim_vitmap_alpha1mean = maxblk_cossim_vitmap * alphamap1 / alphamap1.mean()
    reevol_cossim_vitmap_alpha1mean = reevol_cossim_vitmap * alphamap1 / alphamap1.mean()
    maxblk_cossim_clipmap_alpha1mean = maxblk_cossim_cliptoken * alphamap1_rsz / alphamap1_rsz.mean()
    reevol_cossim_clipmap_alpha1mean = reevol_cossim_cliptoken * alphamap1_rsz / alphamap1_rsz.mean()
    #%%
    # TODO: visualize the cossim maps / dist maps. Find ways to get rid of border artifacts
    #%%
    stats = edict(Expi=Expi,)
    stats.update({
        "reevol_resnet_L3_m": reevol_cossim_tsr_L3.mean().item(),
        "reevol_vit_cls": reevol_cossim_vitcls,
        "reevol_vit_token_m": reevol_cossim_vitmap.mean().item(),
        "maxblk_resnet_L3_m": maxblk_cossim_tsr_L3.mean().item(),
        "maxblk_vit_cls": maxblk_cossim_vitcls,
        "maxblk_vit_token_m": maxblk_cossim_vitmap.mean().item(),
        "reevol_clip_cls": reevol_cossim_clipcls,
        "reevol_clip_token_m": reevol_cossim_cliptoken.mean().item(),
        "maxblk_clip_cls": maxblk_cossim_clipcls,
        "maxblk_clip_token_m": maxblk_cossim_cliptoken.mean().item(),
        "reevol_resnet_L3_m_alpha0mean": reevol_cossim_tsr_L3_alpha0mean.mean().item(),
        "reevol_resnet_L3_m_alpha1mean": reevol_cossim_tsr_L3_alpha1mean.mean().item(),
        "maxblk_resnet_L3_m_alpha0mean": maxblk_cossim_tsr_L3_alpha0mean.mean().item(),
        "maxblk_resnet_L3_m_alpha1mean": maxblk_cossim_tsr_L3_alpha1mean.mean().item(),
        "reevol_vit_token_m_alpha0mean": reevol_cossim_vitmap_alpha0mean.mean().item(),
        "reevol_vit_token_m_alpha1mean": reevol_cossim_vitmap_alpha1mean.mean().item(),
        "maxblk_vit_token_m_alpha0mean": maxblk_cossim_vitmap_alpha0mean.mean().item(),
        "maxblk_vit_token_m_alpha1mean": maxblk_cossim_vitmap_alpha1mean.mean().item(),
        "reevol_clip_token_m_alpha0mean": reevol_cossim_clipmap_alpha0mean.mean().item(),
        "reevol_clip_token_m_alpha1mean": reevol_cossim_clipmap_alpha1mean.mean().item(),
        "maxblk_clip_token_m_alpha0mean": maxblk_cossim_clipmap_alpha0mean.mean().item(),
        "maxblk_clip_token_m_alpha1mean": maxblk_cossim_clipmap_alpha1mean.mean().item(),
         })
    imdist_col.append(stats)
#%%
imdist_alpha_df = pd.DataFrame(imdist_col)
#%%
showimg(FC_reevol_G * alphamap_full0[..., None] / alphamap_full0.max())
#%%

# pd.concat([imdist_df[succmsk].mean(),
#            imdist_df[~succmsk].mean()], axis=1)
pd.concat([imdist_alpha_df[succmsk].mean(),
           imdist_alpha_df[nonemsk].mean()], axis=1)
#%%
pd.concat([imdist_alpha_df[V1msk].mean(),
           imdist_alpha_df[nonemsk&V4msk].mean(),
           imdist_alpha_df[succmsk&V4msk].mean(),
           imdist_alpha_df[nonemsk&ITmsk].mean(),
           imdist_alpha_df[succmsk&ITmsk].mean()], axis=1)
#%%
# change pandas display options, to show more columns
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#%%
plt.imshow(cossim_tsr3[0])
plt.colorbar()
plt.title(f"cosine similarity map")
plt.show()

#%%




#%% Scratch zone
img1 = FC_reevol_G
img2 = BG_reevol_G
featkey = 'blocks'
img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0)
img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)
with torch.no_grad():
    feat1 = fetcher(img1.cuda())[featkey]
    feat2 = fetcher(img2.cuda())[featkey]
#%%
cossim_vec = torch.cosine_similarity(feat1, feat2, dim=-1).cpu()
cossim_cls = cossim_vec[0, 0].item()
cossim_map = cossim_vec[0, 1:].reshape(14, 14).numpy()
#%%
plt.imshow(cossim_map)
plt.colorbar()
plt.title(f"cosine similarity map, cls={cossim_cls:.3f}")
plt.show()
#%%

