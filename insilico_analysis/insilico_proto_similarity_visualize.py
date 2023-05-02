"""visualize the similarity map / mask of the protoimgs
Including LPIPS and CNN (VIT ommitted)
"""

import re
import os
import timm
import torch
from easydict import EasyDict as edict
from tqdm import trange, tqdm
from scipy.stats import sem
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from pathlib import Path
from os.path import join
from core.utils.CNN_scorers import load_featnet
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from core.utils.montage_utils import crop_from_montage, crop_all_from_montage
from core.utils.montage_utils import make_grid, make_grid_np, make_grid_T
from neuro_data_analysis.image_comparison_lib import compare_imgs_cnn, compare_imgs_cnn_featmsk, \
    compare_imgs_LPIPS, naive_featmsk
import pandas as pd
import seaborn as sns
import pickle as pkl
from core.utils.plot_utils import saveallforms
# directories
protosumdir = r"F:\insilico_exps\GAN_Evol_cmp\protoimgs"
figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoImage_cmp_insilico_vis"
#%%
# from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
def format_img(img_np):
    if isinstance(img_np, list):
        img_np = np.stack(img_np)
    img_np = img_np.astype("float32") / 255.0
    zoom_factor = 224 / 256
    if len(img_np.shape) == 3:
        img_np = zoom(img_np, (zoom_factor, zoom_factor, 1), order=1)
    elif len(img_np.shape) == 4:
        img_np = zoom(img_np, (1, zoom_factor, zoom_factor, 1), order=1)
    return img_np




#%%
from neuro_data_analysis.image_comparison_lib import compare_imgs_cnn, compare_imgs_cnn_featmsk, \
    compare_imgs_vit, compare_imgs_LPIPS
from lpips import LPIPS
from timm import list_models, create_model
from torchvision.models import resnet50, alexnet
from core.utils.CNN_scorers import load_featnet
#%%
# cnnmodel = resnet50(pretrained=True)
cnnmodel, _ = load_featnet("resnet50_linf8",)
# get_graph_node_names(cnnmodel)
fetcher_cnn = create_feature_extractor(cnnmodel, ['layer3', "layer4", ])
fetcher_cnn = fetcher_cnn.cuda().eval()
#%%
alexnetmodel = alexnet(pretrained=True)
get_graph_node_names(alexnetmodel)
fetcher_alexnet = create_feature_extractor(alexnetmodel, ['classifier.2', "classifier.4",])
fetcher_alexnet = fetcher_alexnet.cuda().eval()
#%% VIT-DINO
# from timm.models.vision_transformer import VisionTransformer
# vitmodel = create_model('vit_base_patch16_224_dino', pretrained=True, )
# # get_graph_node_names(model)
# fetcher_vit = create_feature_extractor(vitmodel, ['blocks', 'norm']) # ['blocks', 'norm']
# fetcher_vit = fetcher_vit.cuda().eval()
# #%% CLIP
# clipmodel = create_model('vit_large_patch14_224_clip_laion2b', pretrained=True, )
# # get_graph_node_names(model)
# fetcher_clip = create_feature_extractor(clipmodel, ['blocks', 'norm']) # ['blocks', 'norm']
# fetcher_clip = fetcher_clip.cuda().eval()

#%%
Dist = LPIPS(net='squeeze', spatial=True,)
Dist = Dist.cuda().eval()
Dist.requires_grad_(False)
Dist2 = LPIPS(net='vgg', spatial=True,)
Dist2 = Dist2.cuda().eval()
Dist2.requires_grad_(False)
#%%
def showimg(img, bar=False, cmap=None, ax=None):
    if ax is None:
        ax = plt.gca()
    plt.sca(ax)
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    if bar:
        plt.colorbar()
    plt.show()


def showimg_ax(img, bar=False, cmap=None, ax=None, title=None):
    if ax is None:
        ax = plt.gca()
    plt.sca(ax)
    if img is not None:
        plt.imshow(img, cmap=cmap)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    if bar:
        plt.colorbar()


def img_cmp_plot(img1, img2, suptitle=None):
    """Overall image comparison plotting function
    Comparing LPIPS, CNN L3 L4, VIT, CLIP
    """
    figh, axs = plt.subplots(2, 4, figsize=(16, 8 if suptitle is None else 10), squeeze=False)
    showimg_ax(img1, ax=axs[0,0],)
    showimg_ax(img2, ax=axs[0,1],)
    lpips_scalr, lpips_map = compare_imgs_LPIPS(img1, img2, Dist,)
    lpips_scalr_vgg, lpips_map_vgg = compare_imgs_LPIPS(img1, img2, Dist2,)
    cnn_map_3 = compare_imgs_cnn(img1, img2, fetcher_cnn, featkey='layer3', metric='cosine')
    cnn_map_4 = compare_imgs_cnn(img1, img2, fetcher_cnn, featkey='layer4', metric='cosine')
    # vit_cls, vit_map = compare_imgs_vit(img1, img2, fetcher_vit, featkey='norm', metric='cosine')
    # clip_cls, clip_map = compare_imgs_vit(img1, img2, fetcher_clip, featkey='norm', metric='cosine')
    showimg_ax(1 - lpips_map[0,0], ax=axs[0,2], bar=True, title=f"LPIPS squeeze\ndist: {lpips_scalr:.3f}")
    showimg_ax(1 - lpips_map_vgg[0,0], ax=axs[0,3], bar=False, title=f"LPIPS vgg\ndist: {lpips_scalr_vgg:.3f}")
    showimg_ax(cnn_map_3[0], ax=axs[1, 0], bar=True, title=f"Resnet L3\ncos: {cnn_map_3.mean():.3f}")
    showimg_ax(cnn_map_4[0], ax=axs[1, 1], bar=True, title=f"Resnet L4\ncos: {cnn_map_4.mean():.3f}")
    # showimg_ax(vit_map, ax=axs[1, 2], bar=True, title=f"VIT\ncos: {vit_map.mean():.3f}")
    # showimg_ax(clip_map, ax=axs[1, 3], bar=True, title=f"CLIP\ncos: {clip_map.mean():.3f}")
    if suptitle is not None:
        figh.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    figh.show()
    return figh
#%%
import matplotlib
matplotlib.use('Agg')
# use pycharm interactive backend
# matplotlib.use('module://backend_interagg')
#%%
optimname2cmp = ['CholCMA', 'HessCMA', 'HessCMA500_fc6']  #
suffix = ""
# optimname2cmp = ['RFrsz_CholCMA', 'RFrsz_HessCMA', 'RFrsz_HessCMA500_fc6']  #
# suffix = "_RFrsz"
# go through prototypes
# chan_rng = range(20)
chan_rng = range(50)
for layerstr, layer_pattern in [
                                # ("resnet_layer1B1", "resnet50_.layer1.Bottleneck1_%d_28_28_"),
                                # ("resnet_layer2B3", "resnet50_.layer2.Bottleneck3_%d_14_14_"),
                                # ("resnet_layer3B5", "resnet50_.layer3.Bottleneck5_%d_7_7_"),
                                # ("resnet_layer4B2", "resnet50_.layer4.Bottleneck2_%d_4_4_"),
                                # ("resnet_fc", "resnet50_.Linearfc_%d_"),
                                ("resnet_linf8_layer1B1", "resnet50_linf8_.layer1.Bottleneck1_%d_28_28_"),
                                # ("resnet_linf8_layer2B3", "resnet50_linf8_.layer2.Bottleneck3_%d_14_14_"),
                                # ("resnet_linf8_layer3B5", "resnet50_linf8_.layer3.Bottleneck5_%d_7_7_"),
                                # ("resnet_linf8_layer4B2", "resnet50_linf8_.layer4.Bottleneck2_%d_4_4_"),
                                # ("resnet_linf8_fc", "resnet50_linf8_.Linearfc_%d_"),
                                ]:
    img_col_all = {}
    img_stack_all = {}
    for iChan in chan_rng:
        img_col = {}
        unitstr = layer_pattern % iChan
        img_stack_all[iChan] = {}
        for optimnm in optimname2cmp:
            imgfps = [*Path(protosumdir).glob(f"{unitstr}{optimnm}.jpg")]
            mtg = plt.imread(imgfps[0])
            # crop from montage
            imgs = crop_all_from_montage(mtg, imgsize=256, )
            img_col[optimnm] = imgs
            print(mtg.shape, len(imgs))
        img_col_all[iChan] = img_col

        for k, v in img_col_all[iChan].items():
            img_stack_all[iChan][k] = format_img(v)

    for iChan in tqdm(chan_rng):
        img1 = img_stack_all[iChan][optimname2cmp[2]][0]
        img2 = img_stack_all[iChan][optimname2cmp[0]][0]
        img_cmp_plot(img1, img2, f"{layerstr} {iChan} FC6-BG(CholCMA)")
        saveallforms(figdir, f"{layerstr}_{iChan}_FC6BG_RND0{suffix}")
        plt.close()

        img1 = img_stack_all[iChan][optimname2cmp[2]][0]
        img2 = img_stack_all[iChan][optimname2cmp[2]][1]
        img_cmp_plot(img1, img2, f"{layerstr} {iChan} FC6-FC6")
        saveallforms(figdir, f"{layerstr}_{iChan}_FC6FC6_RND0{suffix}")
        plt.close()

        img1 = img_stack_all[iChan][optimname2cmp[0]][0]
        img2 = img_stack_all[iChan][optimname2cmp[0]][1]
        img_cmp_plot(img1, img2, f"{layerstr} {iChan} BG-BG (CholCMA)")
        saveallforms(figdir, f"{layerstr}_{iChan}_BGBG_RND0{suffix}")
        plt.close()