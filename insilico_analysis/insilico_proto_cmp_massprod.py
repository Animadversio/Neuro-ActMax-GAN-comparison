import re
import os
import timm
import torch
import tqdm
from easydict import EasyDict as edict
from tqdm import trange
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
#%%
# directories
protosumdir = r"F:\insilico_exps\GAN_Evol_cmp\protoimgs"
#%%

# cnnmodel = resnet50(pretrained=True)
cnnmodel, _ = load_featnet("resnet50_linf8",)
# get_graph_node_names(cnnmodel)
fetcher_cnn = create_feature_extractor(cnnmodel, ['layer3', "layer4", ])
fetcher_cnn = fetcher_cnn.cuda().eval()

# timm.list_models("*clip*", pretrained=True)
#%%



#%%
# get the names of the optimizers, initial sweep
# querystr = "resnet50_.layer3.Bottleneck5_"
# querystr = "resnet50_.layer2.Bottleneck3_"
querystr = "resnet50_linf8_.layer3.Bottleneck5_"
querystr = r"resnet50_linf8_.layer1.Bottleneck1_"
for iChan in range(20):
    imgfps = [*Path(protosumdir).glob(f"{querystr}{iChan}_*.jpg")]
    print(len(imgfps))
    # extract the string at * of _4_4_* and store in list
    optimnames = [re.findall(f"{querystr}{iChan}_(\d*)_(\d*)_(.*).jpg", imgfp.name, )[0][2]
                        for imgfp in imgfps]
    print(optimnames)

#%%
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


naive_featmask_L3, naive_featmask_L4 = naive_featmsk()
#%% RFmsk and pixel masks



#%%
import pickle as pkl
from core.utils.plot_utils import saveallforms
figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoImage_cmp_insilico"

# layerstr = "resnet_layer4B2"
# layerstr = "resnet_layer3B5"
# layerstr = "resnet_layer2B3"
# layerstr = "resnet_layer1B1"
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
                                ("resnet_linf8_layer2B3", "resnet50_linf8_.layer2.Bottleneck3_%d_14_14_"),
                                ("resnet_linf8_layer3B5", "resnet50_linf8_.layer3.Bottleneck5_%d_7_7_"),
                                ("resnet_linf8_layer4B2", "resnet50_linf8_.layer4.Bottleneck2_%d_4_4_"),
                                ("resnet_linf8_fc", "resnet50_linf8_.Linearfc_%d_"),
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


    #%% compute distance matrices
    dist_col = {}
    stat_col = []
    for iChan in tqdm(chan_rng):
        # image stack for this channel
        # TODO: may subsample to match the number of reps.
        imgstack0 = img_stack_all[iChan][optimname2cmp[0]]
        imgstack1 = img_stack_all[iChan][optimname2cmp[1]]
        imgstack2 = img_stack_all[iChan][optimname2cmp[2]]
        # random sample from other channels from 0,20 as the control
        while True:
            iChan_alt = np.random.choice(chan_rng)
            if iChan_alt != iChan:
                break
        imgstack0_alt = img_stack_all[iChan_alt][optimname2cmp[0]]
        imgstack1_alt = img_stack_all[iChan_alt][optimname2cmp[1]]
        imgstack2_alt = img_stack_all[iChan_alt][optimname2cmp[2]]
        D = edict()
        D.iChan = iChan
        D.iChan_alt = iChan_alt
        D.optimnames = optimname2cmp
        # design the metric
        def img_metric_L4(imgs1, imgs2):
            return compare_imgs_cnn_featmsk(imgs1, imgs2, fetcher_cnn,
                featmsk1=naive_featmask_L4, featkey="layer4", metric="cosine", )

        def img_metric_L3(imgs1, imgs2):
            return compare_imgs_cnn_featmsk(imgs1, imgs2, fetcher_cnn,
                featmsk1=naive_featmask_L3, featkey="layer3", metric="cosine", )

        def img_metric_L4_RFfeatmask(imgs1, imgs2):
            return compare_imgs_cnn_featmsk(imgs1, imgs2, fetcher_cnn,
                featmsk1=RF_featmask_L4, featkey="layer4", metric="cosine", )

        def img_metric_L3_RFfeatmask(imgs1, imgs2):
            return compare_imgs_cnn_featmsk(imgs1, imgs2, fetcher_cnn,
                featmsk1=RF_featmask_L3, featkey="layer3", metric="cosine", )

        def img_metric_L4_RFpixmask(imgs1, imgs2):
            return compare_imgs_cnn_featmsk(imgs1 * RFpixmask, imgs2 * RFpixmask, fetcher_cnn,
                featmsk1=RF_featmask_L4, featkey="layer4", metric="cosine", )

        def img_metric_L3_RFpixmask(imgs1, imgs2):
            return compare_imgs_cnn_featmsk(imgs1 * RFpixmask, imgs2 * RFpixmask, fetcher_cnn,
                featmsk1=RF_featmask_L3, featkey="layer3", metric="cosine", )

        # list of image metrics
        for metric, metric_sfx in [(img_metric_L3, "_L3"),
                                   (img_metric_L4, "_L4")]:
            # list of image stacks to compare between each other
            for entry, imgs1, imgs2 in [
                                        ("distmats00", imgstack0, imgstack0),
                                        ("distmats11", imgstack1, imgstack1),
                                        ("distmats22", imgstack2, imgstack2),
                                        ("distmats02", imgstack0, imgstack2),
                                        ("distmats02_FCalt", imgstack0, imgstack2_alt),
                                        ("distmats02_BGalt", imgstack0_alt, imgstack2),
                                        ("distmats12", imgstack1, imgstack2),
                                        ("distmats12_FCalt", imgstack1, imgstack2_alt),
                                        ("distmats12_BGalt", imgstack1_alt, imgstack2),
                                        ]:
                D[entry + metric_sfx] = metric(imgs1, imgs2)

        # compute summarizing stats for each distance metric. for sns plotting.
        S = edict()
        S.iChan = iChan
        S.iChan_alt = iChan_alt
        for k, v in D.items():
            if k.startswith("distmats"):
                S[k] = v.mean().item()
                S[k+"_std"] = v.std().item()
                S[k+"_sem"] = v.std().item() / np.sqrt(v.nelement())
        stat_col.append(S)
        dist_col[iChan] = D
    #%
    stat_df = pd.DataFrame(stat_col)
    #%%
    stat_df.to_csv(join(figdir, f"{layerstr}_imgdist_cmp_stats{suffix}.csv"))
    pkl.dump(dist_col, open(join(figdir, f"{layerstr}_imgdist_cmp_stats{suffix}.pkl"), "wb"))

#%% merging all layers
network_prefix = "resnet_linf8_"
suffix = ""
stat_all_df = pd.DataFrame()
for layerstr in [
                 # "resnet_layer1B1",
                 # "resnet_layer2B3",
                 # "resnet_layer3B5",
                 # "resnet_layer4B2",
                 # "resnet_fc",
                 "resnet_linf8_layer1B1",
                 "resnet_linf8_layer2B3",
                 "resnet_linf8_layer3B5",
                 "resnet_linf8_layer4B2",
                 "resnet_linf8_fc",
                 ]:
    stat_df = pd.read_csv(join(figdir, f"{layerstr}_imgdist_cmp_stats{suffix}.csv"))
    stat_df["layer"] = layerstr
    stat_df["layershort"] = layerstr[7:]
    stat_all_df = pd.concat([stat_all_df, stat_df], axis=0)

stat_all_df.to_csv(join(figdir, f"{network_prefix}alllayer_imgdist_cmp_stats{suffix}.csv"))
#%%
figh = plt.figure(figsize=[6,6])
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats02_L4", color="k", alpha=0.6, capsize=0.2)
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats00_L4", color="magenta", alpha=0.6, capsize=0.2, )
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats11_L4", color="green", alpha=0.6, capsize=0.2, )
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats22_L4", color="cyan", alpha=0.6, capsize=0.2, )
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats02_FCalt_L4", color="r", alpha=0.6, capsize=0.2)
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats02_BGalt_L4", color="b", alpha=0.6, capsize=0.2)
plt.legend(handles=plt.gca().lines[::16], labels=["FC-BGChol",
                                       "BGChol", "BGHess", "FC",
                                       "FC'-BGChol", "FC-BGChol'"])
plt.ylabel("Cosine Similarity - (resenet_linf8 L4)")
plt.title(f"Image Similarity among prototypes\nLayer 4 cosine\n{network_prefix[:-1]}", fontsize=14)
saveallforms(figdir, f"{network_prefix}alllayers_imgdist_FCBG_CholCMABG_L4{suffix}", figh, )
plt.show()
#%%
# alternatives of sns.pointplot using matplotlib errorbar and plot
# def df_errorbar_plot(data, x, y, color, alpha, capsize, **kwargs):
#     xvals = data[x].unique()
#     yvals = data[y].values
#     ystds = data[y+"_std"].values
#     ysems = data[y+"_sem"].values
#     plt.errorbar(xvals, yvals, yerr=ystds, color=color, alpha=alpha, capsize=capsize, **kwargs)
#     plt.plot(xvals, yvals, color=color, alpha=alpha, **kwargs)
#     return plt.gca()

#%%
figh = plt.figure(figsize=[6,6])
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats02_L3", color="k", alpha=0.6, capsize=0.2)
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats00_L3", color="magenta", alpha=0.6, capsize=0.2, )
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats11_L3", color="green", alpha=0.6, capsize=0.2, )
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats22_L3", color="cyan", alpha=0.6, capsize=0.2, )
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats02_FCalt_L3", color="r", alpha=0.6, capsize=0.2)
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats02_BGalt_L3", color="b", alpha=0.6, capsize=0.2)
plt.legend(handles=plt.gca().lines[::16], labels=["FC-BGChol",
                                       "BGChol", "BGHess", "FC",
                                       "FC'-BGChol", "FC-BGChol'"])
plt.ylabel("Cosine Similarity - (resenet_linf8 L3)")
plt.title(f"Image Similarity among prototypes\nLayer 3 cosine\n{network_prefix[:-1]}", fontsize=14)
saveallforms(figdir, f"{network_prefix}alllayers_imgdist_FCBG_CholCMABG_L3{suffix}", figh, )
plt.show()

#%%
from core.utils.stats_utils import ttest_ind_print_df, ttest_rel_print_df, paired_strip_plot
ttest_rel_print_df(stat_df, None, "distmats12_L3", "distmats12_FCalt_L3")
#%%
ttest_rel_print_df(stat_df, None, "distmats12_L3", "distmats12_BGalt_L3")
#%%
ttest_rel_print_df(stat_df, None, "distmats02_L3", "distmats02_FCalt_L3")
#%%
ttest_rel_print_df(stat_df, None, "distmats02_L4", "distmats02_FCalt_L4")
#%%
figh = paired_strip_plot(stat_df, None, "distmats02_L4", "distmats02_FCalt_L4")
figh.suptitle(f"{layerstr}\nimg dist FC-BG vs FC'-BG, CholCMA BG")
figh.gca().set_ylabel("cosine dist (layer4)")
saveallforms(figdir, f"{layerstr}_imgdist_FCBG_vs_FC'BG_CholCMABG_L4{suffix}", figh, )
figh.show()
#%%
figh = paired_strip_plot(stat_df, None, "distmats02_L4", "distmats02_BGalt_L4")
figh.suptitle(f"{layerstr}\nimg dist FC-BG vs FC-BG', CholCMA BG")
figh.gca().set_ylabel("cosine dist (layer4)")
saveallforms(figdir, f"{layerstr}_imgdist_FCBG_vs_FCBG'_CholCMABG_L4{suffix}", figh, )
figh.show()
#%%
figh = paired_strip_plot(stat_df, None, "distmats12_L4", "distmats12_FCalt_L4")
figh.suptitle(f"{layerstr}\nimg dist FC-BG vs FC'-BG, HessCMA BG")
figh.gca().set_ylabel("cosine dist (layer4)")
saveallforms(figdir, f"{layerstr}_imgdist_FCBG_vs_FC'BG_HessCMABG_L4{suffix}", figh, )
figh.show()
#%%
figh = paired_strip_plot(stat_df, None, "distmats12_L4", "distmats12_BGalt_L4")
figh.suptitle(f"{layerstr}\nimg dist FC-BG vs FC-BG', Hess BG")
figh.gca().set_ylabel("cosine dist (layer4)")
saveallforms(figdir, f"{layerstr}_imgdist_FCBG_vs_FCBG'_HessCMABG_L4{suffix}", figh, )
figh.show()
#%%
figh = paired_strip_plot(stat_df, None, "distmats02_L3", "distmats02_FCalt_L3")
figh.suptitle(f"{layerstr}\nimg dist FC-BG vs FC'-BG, CholCMA BG")
figh.gca().set_ylabel("cosine dist (layer3)")
saveallforms(figdir, f"{layerstr}_imgdist_FCBG_vs_FC'BG_CholCMABG_L3{suffix}", figh, )
figh.show()
#%%
figh = paired_strip_plot(stat_df, None, "distmats02_L3", "distmats02_BGalt_L3")
figh.suptitle(f"{layerstr}\nimg dist FC-BG vs FC-BG', CholCMA BG")
figh.gca().set_ylabel("cosine dist (layer3)")
saveallforms(figdir, f"{layerstr}_imgdist_FCBG_vs_FCBG'_CholCMABG_L3{suffix}", figh, )
figh.show()
