import re
import os
import timm
import torch
from tqdm import tqdm
from easydict import EasyDict as edict
import pickle as pkl
from tqdm import trange
from scipy.stats import sem
from scipy.ndimage import zoom
from pathlib import Path
from os.path import join
from core.utils.CNN_scorers import load_featnet
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from core.utils.montage_utils import crop_from_montage, crop_all_from_montage
from core.utils.montage_utils import make_grid, make_grid_np, make_grid_T
from neuro_data_analysis.image_comparison_lib import compare_imgs_cnn, compare_imgs_cnn_featmsk, \
    compare_imgs_LPIPS, naive_featmsk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms

#%%
# directories
protosumdir = r"F:\insilico_exps\GAN_Evol_cmp\protoimgs"
#%% load the image encoders
# cnnmodel = resnet50(pretrained=True)
cnnmodel, _ = load_featnet("resnet50_linf8",)
# get_graph_node_names(cnnmodel)
fetcher_cnn = create_feature_extractor(cnnmodel, ['layer3', "layer4", ])
fetcher_cnn = fetcher_cnn.cuda().eval()

# timm.list_models("*clip*", pretrained=True)

#%%
# get the names of the optimizers, initial sweep
# querystr = "resnet50_.layer3.Bottleneck5_"
# querystr = "resnet50_.layer2.Bottleneck3_"
# querystr = "resnet50_linf8_.layer3.Bottleneck5_"
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
naive_pixmask = np.ones((224, 224, 1), dtype="float32")
#%% RFmsk and pixel masks
RFdir = r"F:\insilico_exps\GAN_Evol_cmp\RFmaps"
fitRFdict = pkl.load(open(join(RFdir, "fitmaps_dict.pkl"), "rb"))
#%%
fitRFtorchdict = {}
for key, (pixmask, featmask_L3, featmask_L4) in fitRFdict.items():
    fitRFtorchdict[key] = pixmask[:,:,None].astype("float32"), \
                          torch.from_numpy(featmask_L3).float().cuda(), \
                          torch.from_numpy(featmask_L4).float().cuda()

fitRFtorchdict["resnet50_linf8_fc"] = naive_pixmask, naive_featmask_L3, naive_featmask_L4
fitRFtorchdict["resnet50_fc"] = naive_pixmask, naive_featmask_L3, naive_featmask_L4
fitRFtorchdict["tf_efficientnet_b6_ap_fc"] = naive_pixmask, naive_featmask_L3, naive_featmask_L4
fitRFtorchdict["tf_efficientnet_b6_fc"] = naive_pixmask, naive_featmask_L3, naive_featmask_L4
fitRFtorchdict["tf_efficientnet_b6_ap_globalpool"] = naive_pixmask, naive_featmask_L3, naive_featmask_L4
fitRFtorchdict["tf_efficientnet_b6_globalpool"] = naive_pixmask, naive_featmask_L3, naive_featmask_L4

#%%
figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoImage_cmp_insilico"
datadir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoImage_cmp_insilico\data"
optimname2cmp = ['CholCMA', 'HessCMA', 'HessCMA500_fc6']  #
suffix = ""
# optimname2cmp = ['RFrsz_CholCMA', 'RFrsz_HessCMA', 'RFrsz_HessCMA500_fc6']  #
# suffix = "_RFrsz"
# go through prototypes
# chan_rng = range(20)
chan_rng = range(50)
for layerstr, layer_pattern in [
                                # ("resnet50_layer1B1", "resnet50_.layer1.Bottleneck1_%d_28_28_"),
                                # ("resnet50_layer2B3", "resnet50_.layer2.Bottleneck3_%d_14_14_"),
                                # ("resnet50_layer3B5", "resnet50_.layer3.Bottleneck5_%d_7_7_"),
                                # ("resnet50_layer4B2", "resnet50_.layer4.Bottleneck2_%d_4_4_"),
                                # ("resnet50_fc", "resnet50_.Linearfc_%d_"),
                                # ("resnet50_linf8_layer1B1", "resnet50_linf8_.layer1.Bottleneck1_%d_28_28_"),
                                # ("resnet50_linf8_layer2B3", "resnet50_linf8_.layer2.Bottleneck3_%d_14_14_"),
                                # ("resnet50_linf8_layer3B5", "resnet50_linf8_.layer3.Bottleneck5_%d_7_7_"),
                                # ("resnet50_linf8_layer4B2", "resnet50_linf8_.layer4.Bottleneck2_%d_4_4_"),
                                # ("resnet50_linf8_fc", "resnet50_linf8_.Linearfc_%d_"),
                                # ("tf_efficientnet_b6_ap_blocks.0", "tf_efficientnet_b6_ap_.blocks.0_%d_57_57_"),
                                # ("tf_efficientnet_b6_ap_blocks.1", "tf_efficientnet_b6_ap_.blocks.1_%d_28_28_"),
                                ("tf_efficientnet_b6_ap_blocks.2", "tf_efficientnet_b6_ap_.blocks.2_%d_14_14_"),
                                ("tf_efficientnet_b6_ap_blocks.3", "tf_efficientnet_b6_ap_.blocks.3_%d_7_7_"),
                                ("tf_efficientnet_b6_ap_blocks.4", "tf_efficientnet_b6_ap_.blocks.4_%d_7_7_"),
                                ("tf_efficientnet_b6_ap_blocks.5", "tf_efficientnet_b6_ap_.blocks.5_%d_4_4_"),
                                ("tf_efficientnet_b6_ap_blocks.6", "tf_efficientnet_b6_ap_.blocks.6_%d_4_4_"),
                                ("tf_efficientnet_b6_ap_globalpool", "tf_efficientnet_b6_ap_.SelectAdaptivePool2dglobal_pool_%d_"),
                                ("tf_efficientnet_b6_ap_fc", "tf_efficientnet_b6_ap_.Linearclassifier_%d_"),

                                ("tf_efficientnet_b6_blocks.0", "tf_efficientnet_b6_.blocks.0_%d_57_57_"),
                                ("tf_efficientnet_b6_blocks.1", "tf_efficientnet_b6_.blocks.1_%d_28_28_"),
                                ("tf_efficientnet_b6_blocks.2", "tf_efficientnet_b6_.blocks.2_%d_14_14_"),
                                ("tf_efficientnet_b6_blocks.3", "tf_efficientnet_b6_.blocks.3_%d_7_7_"),
                                ("tf_efficientnet_b6_blocks.4", "tf_efficientnet_b6_.blocks.4_%d_7_7_"),
                                ("tf_efficientnet_b6_blocks.5", "tf_efficientnet_b6_.blocks.5_%d_4_4_"),
                                ("tf_efficientnet_b6_blocks.6", "tf_efficientnet_b6_.blocks.6_%d_4_4_"),
                                ("tf_efficientnet_b6_globalpool", "tf_efficientnet_b6_.SelectAdaptivePool2dglobal_pool_%d_"),
                                ("tf_efficientnet_b6_fc", "tf_efficientnet_b6_.Linearclassifier_%d_"),
                                ]:
    # sepidx = layerstr.rfind("_")
    # netname = layerstr[:sepidx]
    # layerkey = layerstr.split("_")[-1]
    RFpixmask, RF_featmask_L3, RF_featmask_L4 = fitRFtorchdict[layerstr]
    # continue
    img_col_all = {}
    img_stack_all = {}
    for iChan in tqdm(chan_rng):
        img_col = {}
        unitstr = layer_pattern % iChan
        for optimnm in optimname2cmp:
            imgfps = [*Path(protosumdir).glob(f"{unitstr}{optimnm}.jpg")]
            if len(imgfps) == 0:
                continue
            mtg = plt.imread(imgfps[0])
            # crop from montage
            imgs = crop_all_from_montage(mtg, imgsize=256, autostop=False)
            img_col[optimnm] = imgs
            print(mtg.shape, len(imgs))
        if len(img_col) == 0:
            print(f"no images for {unitstr}, skip")
            continue
        img_col_all[iChan] = img_col
        img_stack_all[iChan] = {}
        for k, v in img_col_all[iChan].items():
            img_stack_all[iChan][k] = format_img(v)

    #%% compute distance matrices
    dist_col = {}
    stat_col = []
    exist_chans = list(img_stack_all.keys())
    for iChan in tqdm(exist_chans):
        # image stack for this channel
        # TODO: may subsample to match the number of reps.
        imgstack0 = img_stack_all[iChan][optimname2cmp[0]]
        imgstack1 = img_stack_all[iChan][optimname2cmp[1]]
        imgstack2 = img_stack_all[iChan][optimname2cmp[2]]
        # random sample from other channels from 0,20 as the control
        while True:
            iChan_alt = np.random.choice(exist_chans)
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
                featmsk1=naive_featmask_L4, featkey="layer4", metric="cosine", )

        def img_metric_L3_RFpixmask(imgs1, imgs2):
            return compare_imgs_cnn_featmsk(imgs1 * RFpixmask, imgs2 * RFpixmask, fetcher_cnn,
                featmsk1=naive_featmask_L3, featkey="layer3", metric="cosine", )

        def img_metric_L4_RFpixfeatmask(imgs1, imgs2):
            return compare_imgs_cnn_featmsk(imgs1 * RFpixmask, imgs2 * RFpixmask, fetcher_cnn,
                featmsk1=RF_featmask_L4, featkey="layer4", metric="cosine", )

        def img_metric_L3_RFpixfeatmask(imgs1, imgs2):
            return compare_imgs_cnn_featmsk(imgs1 * RFpixmask, imgs2 * RFpixmask, fetcher_cnn,
                featmsk1=RF_featmask_L3, featkey="layer3", metric="cosine", )

        # list of image metrics
        for metric, metric_sfx in [(img_metric_L3, "_L3"),
                                   (img_metric_L4, "_L4"),
                                   (img_metric_L3_RFfeatmask, "_L3_RFftmsk"),
                                   (img_metric_L4_RFfeatmask, "_L4_RFftmsk"),
                                   (img_metric_L3_RFpixmask, "_L3_RFpxmsk"),
                                   (img_metric_L4_RFpixmask, "_L4_RFpxmsk"),
                                   (img_metric_L3_RFpixfeatmask, "_L3_RFpxftmsk"),
                                   (img_metric_L4_RFpixfeatmask, "_L4_RFpxftmsk"),]:
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
    stat_df.to_csv(join(datadir, f"{layerstr}_imgdist_cmp_stats{suffix}.csv"))
    pkl.dump(dist_col, open(join(datadir, f"{layerstr}_imgdist_cmp_stats{suffix}.pkl"), "wb"))

#%% merging all layers
network_prefix = "effnet_" #"effnet_ap_"#"resnet50_" #"resnet50_linf8_" #
suffix = ""
stat_all_df = pd.DataFrame()
for layerstr in [
                 # "resnet50_layer1B1",
                 # "resnet50_layer2B3",
                 # "resnet50_layer3B5",
                 # "resnet50_layer4B2",
                 # "resnet50_fc",
                 # "resnet50_linf8_layer1B1",
                 # "resnet50_linf8_layer2B3",
                 # "resnet50_linf8_layer3B5",
                 # "resnet50_linf8_layer4B2",
                 # "resnet50_linf8_fc",
                # "tf_efficientnet_b6_ap_blocks.0",
                # "tf_efficientnet_b6_ap_blocks.1",
                # "tf_efficientnet_b6_ap_blocks.2",
                # "tf_efficientnet_b6_ap_blocks.3",
                # "tf_efficientnet_b6_ap_blocks.4",
                # "tf_efficientnet_b6_ap_blocks.5",
                # "tf_efficientnet_b6_ap_blocks.6",
                # "tf_efficientnet_b6_ap_globalpool",
                # "tf_efficientnet_b6_ap_fc",

                "tf_efficientnet_b6_blocks.0",
                "tf_efficientnet_b6_blocks.1",
                "tf_efficientnet_b6_blocks.2",
                "tf_efficientnet_b6_blocks.3",
                "tf_efficientnet_b6_blocks.4",
                "tf_efficientnet_b6_blocks.5",
                "tf_efficientnet_b6_blocks.6",
                "tf_efficientnet_b6_globalpool",
                "tf_efficientnet_b6_fc",
]:
    stat_df = pd.read_csv(join(datadir, f"{layerstr}_imgdist_cmp_stats{suffix}.csv"))
    stat_df["layer"] = layerstr
    stat_df["layershort"] = layerstr.split("_")[-1]  #layerstr[len(network_prefix):]
    stat_all_df = pd.concat([stat_all_df, stat_df], axis=0)

stat_all_df.to_csv(join(datadir, f"{network_prefix}alllayer_imgdist_cmp_stats{suffix}.csv"))

#%%
from core.utils.stats_utils import ttest_ind_print_df, ttest_rel_print_df, paired_strip_plot
network_prefix = "effnet_" # "effnet_ap_" ##"resnet50_" #"resnet50_linf8_" #
stat_all_df = pd.read_csv(join(datadir, f"{network_prefix}alllayer_imgdist_cmp_stats{suffix}.csv"))
"""Compare the distance metrics between the different layers
Loop through all the metrics and plot the results
"""
metric_sfx = "_L4"
nlayer = len(stat_all_df.layershort.unique())
for metric_sfx in ["_L4",
                   "_L4_RFftmsk",
                   "_L4_RFpxmsk",
                   "_L4_RFpxftmsk",
                   "_L3",
                   "_L3_RFftmsk",
                   "_L3_RFpxmsk",
                   "_L3_RFpxftmsk",
                   ]:
    figh = plt.figure(figsize=[6,6])
    sns.pointplot(data=stat_all_df, x="layershort",
                  y="distmats02"+metric_sfx, color="k", alpha=0.6, capsize=0.2)
    sns.pointplot(data=stat_all_df, x="layershort",
                  y="distmats00"+metric_sfx, color="magenta", alpha=0.6, capsize=0.2, )
    sns.pointplot(data=stat_all_df, x="layershort",
                  y="distmats11"+metric_sfx, color="green", alpha=0.6, capsize=0.2, )
    sns.pointplot(data=stat_all_df, x="layershort",
                  y="distmats22"+metric_sfx, color="cyan", alpha=0.6, capsize=0.2, )
    sns.pointplot(data=stat_all_df, x="layershort",
                  y="distmats02_FCalt"+metric_sfx, color="r", alpha=0.6, capsize=0.2)
    sns.pointplot(data=stat_all_df, x="layershort",
                  y="distmats02_BGalt"+metric_sfx, color="b", alpha=0.6, capsize=0.2)
    plt.legend(handles=plt.gca().lines[::(3*nlayer+1)], labels=["FC-BGChol",
                                           "BGChol", "BGHess", "FC",
                                           "FC'-BGChol", "FC-BGChol'"])
    plt.ylabel(f"Cosine Similarity - (resenet_linf8 {metric_sfx.replace('_', ' ')})")
    plt.title(f"Image Similarity among prototypes\n{metric_sfx.replace('_', ' ')} cosine\n{network_prefix[:-1]}", fontsize=14)
    saveallforms(figdir, f"{network_prefix}alllayers_imgdist_FCBG_CholCMABG{metric_sfx}{suffix}", figh, )
    plt.show()

#%%
"""Compare the distance metrics between the different layers
Loop through all the metrics and plot the results
"""

#%% Scratch
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
#%%
# pandas display options with wider full columns
pd.set_option('display.max_columns', 500)
#%%
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
