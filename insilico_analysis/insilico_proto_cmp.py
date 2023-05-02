
import re
import os
import timm
import torch
from pathlib import Path
from os.path import join
import pandas as pd
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.stats import sem
from easydict import EasyDict as edict
from core.utils.CNN_scorers import load_featnet
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from core.utils.montage_utils import crop_from_montage, crop_all_from_montage
from core.utils.montage_utils import make_grid, make_grid_np, make_grid_T
from neuro_data_analysis.image_comparison_lib import compare_imgs_cnn, compare_imgs_cnn_featmsk, \
    compare_imgs_LPIPS, naive_featmsk
#%%
# directories
protosumdir = r"F:\insilico_exps\GAN_Evol_cmp\protoimgs"
#%%

# cnnmodel = resnet50(pretrained=True)
cnnmodel, _ = load_featnet("resnet50_linf8",)
# get_graph_node_names(cnnmodel)
fetcher_cnn = create_feature_extractor(cnnmodel, ['layer3', "layer4", "avgpool"])
fetcher_cnn = fetcher_cnn.cuda().eval()

# timm.list_models("*clip*", pretrained=True)

#%%
# get the names of the optimizers, initial sweep
querystr = "resnet50_.layer3.Bottleneck5_"
querystr = "resnet50_.layer2.Bottleneck3_"
querystr = "resnet50_.layer1.Bottleneck1_"
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
#%%
import pickle as pkl
from core.utils.plot_utils import saveallforms
figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoImage_cmp_insilico"

layerstr = "resnet_layer4B2"
# layerstr = "resnet_layer3B5"
# layerstr = "resnet_layer2B3"
# layerstr = "resnet_layer1B1"
# optimname2cmp = ['CholCMA', 'HessCMA', 'HessCMA500_fc6']  #
optimname2cmp = ['RFrsz_CholCMA', 'RFrsz_HessCMA', 'RFrsz_HessCMA500_fc6']  #
suffix = "_RFrsz"
# go through prototypes
for layerstr, layer_pattern in [("resnet_layer1B1", "resnet50_.layer1.Bottleneck1_%d_28_28_"),
                                ("resnet_layer2B3", "resnet50_.layer2.Bottleneck3_%d_14_14_"),
                                ("resnet_layer3B5", "resnet50_.layer3.Bottleneck5_%d_7_7_"),
                                ("resnet_layer4B2", "resnet50_.layer4.Bottleneck2_%d_4_4_"),]:
    img_col_all = {}
    img_stack_all = {}
    for iChan in range(20):
        img_col = {}
        unitstr = layer_pattern % iChan
        # unitstr = f"resnet50_.layer4.Bottleneck2_{iChan}_4_4_"
        # unitstr = f"resnet50_.layer3.Bottleneck5_{iChan}_7_7_"
        # unitstr = f"resnet50_.layer2.Bottleneck3_{iChan}_14_14_"
        # unitstr = f"resnet50_.layer1.Bottleneck1_{iChan}_28_28_"
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


    #%%
    dist_col = {}
    stat_col = []
    for iChan in trange(20):
        imgstack0 = img_stack_all[iChan][optimname2cmp[0]]
        imgstack1 = img_stack_all[iChan][optimname2cmp[1]]
        imgstack2 = img_stack_all[iChan][optimname2cmp[2]]
        # random sample from other channels from 0,20
        while True:
            iChan_alt = np.random.choice(20)
            if iChan_alt != iChan:
                break
        imgstack0_alt = img_stack_all[iChan_alt][optimname2cmp[0]]
        imgstack1_alt = img_stack_all[iChan_alt][optimname2cmp[1]]
        imgstack2_alt = img_stack_all[iChan_alt][optimname2cmp[2]]
        D = edict()
        D.iChan = iChan
        D.iChan_alt = iChan_alt
        D.distmats12_L4 = compare_imgs_cnn_featmsk(imgstack1, imgstack2, fetcher_cnn,
                                    featmsk1=naive_featmask_L4, featkey="layer4", metric="cosine", )
        D.distmats12_FCalt_L4 = compare_imgs_cnn_featmsk(imgstack1, imgstack2_alt, fetcher_cnn,
                                    featmsk1=naive_featmask_L4, featkey="layer4", metric="cosine", )
        D.distmats12_BGalt_L4 = compare_imgs_cnn_featmsk(imgstack1_alt, imgstack2, fetcher_cnn,
                                    featmsk1=naive_featmask_L4, featkey="layer4", metric="cosine", )
        D.distmats02_L4 = compare_imgs_cnn_featmsk(imgstack0, imgstack2, fetcher_cnn,
                                    featmsk1=naive_featmask_L4, featkey="layer4", metric="cosine", )
        D.distmats02_FCalt_L4 = compare_imgs_cnn_featmsk(imgstack0, imgstack2_alt, fetcher_cnn,
                                    featmsk1=naive_featmask_L4, featkey="layer4", metric="cosine", )
        D.distmats02_BGalt_L4 = compare_imgs_cnn_featmsk(imgstack0_alt, imgstack2, fetcher_cnn,
                                    featmsk1=naive_featmask_L4, featkey="layer4", metric="cosine", )
        D.distmats12_L3 = compare_imgs_cnn_featmsk(imgstack1, imgstack2, fetcher_cnn,
                                    featmsk1=naive_featmask_L3, featkey="layer3", metric="cosine", )
        D.distmats12_FCalt_L3 = compare_imgs_cnn_featmsk(imgstack1, imgstack2_alt, fetcher_cnn,
                                    featmsk1=naive_featmask_L3, featkey="layer3", metric="cosine", )
        D.distmats12_BGalt_L3 = compare_imgs_cnn_featmsk(imgstack1_alt, imgstack2, fetcher_cnn,
                                    featmsk1=naive_featmask_L3, featkey="layer3", metric="cosine", )
        D.distmats02_L3 = compare_imgs_cnn_featmsk(imgstack0, imgstack2, fetcher_cnn,
                                    featmsk1=naive_featmask_L3, featkey="layer3", metric="cosine", )
        D.distmats02_FCalt_L3 = compare_imgs_cnn_featmsk(imgstack0, imgstack2_alt, fetcher_cnn,
                                    featmsk1=naive_featmask_L3, featkey="layer3", metric="cosine", )
        D.distmats02_BGalt_L3 = compare_imgs_cnn_featmsk(imgstack0_alt, imgstack2, fetcher_cnn,
                                    featmsk1=naive_featmask_L3, featkey="layer3", metric="cosine", )
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
    #%%
    stat_df = pd.DataFrame(stat_col)
    #%%

    stat_df.to_csv(join(figdir, f"{layerstr}_imgdist_cmp_stats_{suffix}.csv"))
    pkl.dump(dist_col, open(join(figdir, f"{layerstr}_imgdist_cmp_stats_{suffix}.pkl"), "wb"))
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
saveallforms(figdir, f"{layerstr}_imgdist_FCBG_vs_FC'BG_CholCMABG_L4", figh, )
figh.show()
#%%
figh = paired_strip_plot(stat_df, None, "distmats02_L4", "distmats02_BGalt_L4")
figh.suptitle(f"{layerstr}\nimg dist FC-BG vs FC-BG', CholCMA BG")
figh.gca().set_ylabel("cosine dist (layer4)")
saveallforms(figdir, f"{layerstr}_imgdist_FCBG_vs_FCBG'_CholCMABG_L4", figh, )
figh.show()
#%%
figh = paired_strip_plot(stat_df, None, "distmats12_L4", "distmats12_FCalt_L4")
figh.suptitle(f"{layerstr}\nimg dist FC-BG vs FC'-BG, HessCMA BG")
figh.gca().set_ylabel("cosine dist (layer4)")
saveallforms(figdir, f"{layerstr}_imgdist_FCBG_vs_FC'BG_HessCMABG_L4", figh, )
figh.show()
#%%
figh = paired_strip_plot(stat_df, None, "distmats12_L4", "distmats12_BGalt_L4")
figh.suptitle(f"{layerstr}\nimg dist FC-BG vs FC-BG', Hess BG")
figh.gca().set_ylabel("cosine dist (layer4)")
saveallforms(figdir, f"{layerstr}_imgdist_FCBG_vs_FCBG'_HessCMABG_L4", figh, )
figh.show()
#%%
figh = paired_strip_plot(stat_df, None, "distmats02_L3", "distmats02_FCalt_L3")
figh.suptitle(f"{layerstr}\nimg dist FC-BG vs FC'-BG, CholCMA BG")
figh.gca().set_ylabel("cosine dist (layer3)")
saveallforms(figdir, f"{layerstr}_imgdist_FCBG_vs_FC'BG_CholCMABG_L3", figh, )
figh.show()
#%%
figh = paired_strip_plot(stat_df, None, "distmats02_L3", "distmats02_BGalt_L3")
figh.suptitle(f"{layerstr}\nimg dist FC-BG vs FC-BG', CholCMA BG")
figh.gca().set_ylabel("cosine dist (layer3)")
saveallforms(figdir, f"{layerstr}_imgdist_FCBG_vs_FCBG'_CholCMABG_L3", figh, )
figh.show()



#%%
import pandas as pd
import seaborn as sns
stat_all_df = pd.DataFrame()
for layerstr in ["resnet_layer1B1",
                 "resnet_layer2B3",
                 "resnet_layer3B5",
                 "resnet_layer4B2",
                 ]:
    stat_df = pd.read_csv(join(figdir, f"{layerstr}_imgdist_cmp_stats.csv"))
    stat_df["layer"] = layerstr
    stat_df["layershort"] = layerstr[7:-2]
    stat_all_df = pd.concat([stat_all_df, stat_df], axis=0)
#%%
plt.figure(figsize=[6,6])
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats02_L4", color="k", alpha=0.6, capsize=0.2)
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats02_FCalt_L4", color="r", alpha=0.6, capsize=0.2)
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats02_BGalt_L4", color="b", alpha=0.6, capsize=0.2)
plt.show()
#%%
plt.figure(figsize=[6,6])
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats02_L3", color="k", alpha=0.6, capsize=0.2)
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats02_FCalt_L3", color="r", alpha=0.6, capsize=0.2)
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats02_BGalt_L3", color="b", alpha=0.6, capsize=0.2)
plt.show()
#%%
# # formate list of images
# imgtsr1 = img_col[optimpair[0]][1].astype("float32") / 255.0
# imgtsr2 = img_col[optimpair[1]][1].astype("float32") / 255.0
# zoom_factor = 224/ 256
# imgtsr1 = zoom(imgtsr1, (zoom_factor, zoom_factor, 1))
# imgtsr2 = zoom(imgtsr2, (zoom_factor, zoom_factor, 1))
# img_col1 = [format_img(img) for img in img_col[optimpair[0]]]
# img_col2 = [format_img(img) for img in img_col[optimpair[1]]]
#%% compute two image sets
# TODO: consider image scores

# Compute image similarity scores based on resnet50?

# Collect in pandas dataframe

#%% Scratch zone
#%%
# inspect the images
plt.subplots(1, 3, figsize=(10,4))
for iOptim, optimnm in enumerate(optimname2cmp):
    imgs = img_col[optimnm]
    plt.subplot(1, 3, iOptim+1)
    plt.imshow(imgs[0])
    plt.axis("off")
    plt.title(f"{optimnm} {0}")
plt.show()
#%%
imgstack0 = format_img(img_col[optimname2cmp[0]])
imgstack1 = format_img(img_col[optimname2cmp[1]])
imgstack2 = format_img(img_col[optimname2cmp[2]])
distmats12 = compare_imgs_cnn_featmsk(imgstack1, imgstack2,
                         fetcher_cnn, featmsk1=naive_featmask_L4,
                         featkey="layer4", metric="cosine", )
distmats01 = compare_imgs_cnn_featmsk(imgstack0, imgstack1,
                            fetcher_cnn, featmsk1=naive_featmask_L4,
                            featkey="layer4", metric="cosine", )
distmats02 = compare_imgs_cnn_featmsk(imgstack0, imgstack2,
                            fetcher_cnn, featmsk1=naive_featmask_L4,
                            featkey="layer4", metric="cosine", )
print(f"dist between {optimname2cmp[0]} and {optimname2cmp[1]}: {distmats01.mean():.3f}+-{distmats01.std():.3f}")
print(f"dist between {optimname2cmp[0]} and {optimname2cmp[2]}: {distmats02.mean():.3f}+-{distmats02.std():.3f}")
print(f"dist between {optimname2cmp[1]} and {optimname2cmp[2]}: {distmats12.mean():.3f}+-{distmats12.std():.3f}")
#%%
chan, chan_alt = 10, 0
imgstack0 = format_img(img_col_all[chan_alt][optimname2cmp[0]])
imgstack1 = format_img(img_col_all[chan_alt][optimname2cmp[1]])
imgstack2 = format_img(img_col_all[chan][optimname2cmp[2]])
distmats12 = compare_imgs_cnn_featmsk(imgstack1, imgstack2, fetcher_cnn, featmsk1=naive_featmask_L4,
                         featkey="layer4", metric="cosine", )
distmats01 = compare_imgs_cnn_featmsk(imgstack0, imgstack1, fetcher_cnn, featmsk1=naive_featmask_L4,
                            featkey="layer4", metric="cosine", )
distmats02 = compare_imgs_cnn_featmsk(imgstack0, imgstack2, fetcher_cnn, featmsk1=naive_featmask_L4,
                            featkey="layer4", metric="cosine", )
# print(f"dist between {optimname2cmp[0]} and {optimname2cmp[1]}: {distmats01.mean():.3f}+-{distmats01.std():.3f}")
print(f"dist between {optimname2cmp[0]} and {optimname2cmp[2]}: {distmats02.mean():.3f}+-{distmats02.std():.3f}")
print(f"dist between {optimname2cmp[1]} and {optimname2cmp[2]}: {distmats12.mean():.3f}+-{distmats12.std():.3f}")

#%%
# show images in imgstack1
plt.figure()
plt.imshow(make_grid_np(list(imgstack2), nrow=8))
plt.axis("off")
plt.show()

