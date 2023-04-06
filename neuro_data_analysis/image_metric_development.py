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
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data, get_expstr
from core.utils.dataset_utils import ImagePathDataset, ImageFolder
from timm import list_models, create_model
from torchvision.models import resnet50, alexnet
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from neuro_data_analysis.neural_data_lib import get_expstr

_, BFEStats = load_neural_data()
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


from neuro_data_analysis.image_comparison_lib import compare_imgs_cnn, compare_imgs_cnn_featmsk, \
    compare_imgs_vit, compare_imgs_LPIPS
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
Dist = LPIPS(net='squeeze', spatial=True,)
Dist = Dist.cuda().eval()
Dist.requires_grad_(False)
#%%
def img_cmp_all(img1, img2, suptitle=None):
    """Overall image comparison plotting function"""
    figh, axs = plt.subplots(2, 4, figsize=(16, 8 if suptitle is None else 10), squeeze=False)
    showimg_ax(img1, ax=axs[0,0],)
    showimg_ax(img2, ax=axs[0,1],)
    lpips_scalr, lpips_map = compare_imgs_LPIPS(img1, img2, Dist,)
    cnn_map_3 = compare_imgs_cnn(img1, img2, fetcher_cnn, featkey='layer3', metric='cosine')
    cnn_map_4 = compare_imgs_cnn(img1, img2, fetcher_cnn, featkey='layer4', metric='cosine')
    vit_cls, vit_map = compare_imgs_vit(img1, img2, fetcher_vit, featkey='norm', metric='cosine')
    clip_cls, clip_map = compare_imgs_vit(img1, img2, fetcher_clip, featkey='norm', metric='cosine')
    showimg_ax(1 - lpips_map[0,0], ax=axs[0,2], bar=True, title=f"LPIPS\ndist: {lpips_scalr:.3f}")
    showimg_ax(None, ax=axs[0,3], bar=False,)
    showimg_ax(cnn_map_3[0], ax=axs[1, 0], bar=True, title=f"Resnet L3\ncos: {cnn_map_3.mean():.3f}")
    showimg_ax(cnn_map_4[0], ax=axs[1, 1], bar=True, title=f"Resnet L4\ncos: {cnn_map_4.mean():.3f}")
    showimg_ax(vit_map, ax=axs[1, 2], bar=True, title=f"VIT\ncos: {vit_map.mean():.3f}")
    showimg_ax(clip_map, ax=axs[1, 3], bar=True, title=f"CLIP\ncos: {clip_map.mean():.3f}")
    if suptitle is not None:
        figh.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    figh.show()
    return figh
#%%
from tqdm import trange, tqdm
import pickle as pkl
alphamaskdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\AlphaMasks"
outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoImage_cmp\scratch"
tabdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables"
meta_act_df = pd.read_csv(join(tabdir, "meta_activation_stats.csv"), )
#%%
import matplotlib
matplotlib.use("Agg")
#%%
Expi = 113
for Expi in trange(1, 191):
    if not os.path.exists(join(protosumdir, f"Exp{Expi}_proto_attr_montage.jpg")):
        # raise ValueError("Montage not found")
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
    expstr = get_expstr(BFEStats, Expi)
    exp_row = meta_act_df[meta_act_df.Expi == Expi].iloc[0]
    exptitle = f"{expstr}\nThread0 init {exp_row.initrsp_0_mean:.1f}+-{exp_row.initrsp_0_std:.1f}" \
               f" max {exp_row.maxrsp_0_mean:.1f}+-{exp_row.maxrsp_0_std:.1f}" \
               f"\nThread1 init {exp_row.initrsp_1_mean:.1f}+-{exp_row.initrsp_1_std:.1f}" \
               f" max {exp_row.maxrsp_1_mean:.1f}+-{exp_row.maxrsp_1_std:.1f}"

    #%%
    figh1 = img_cmp_all(FC_reevol_pix, BG_reevol_pix, suptitle=exptitle)
    saveallforms(outdir, f"Exp{Expi:02d}_FC_BG_reevol_pix", figh1)
    #%%
    figh2 = img_cmp_all(FC_reevol_G, BG_reevol_G, suptitle=exptitle)
    saveallforms(outdir, f"Exp{Expi:02d}_FC_BG_reevol_G", figh2)
    #%%
    figh3 = img_cmp_all(FC_maxblk, BG_maxblk, suptitle=exptitle)
    saveallforms(outdir, f"Exp{Expi:02d}_FC_BG_maxblk", figh3)




#%%
img_cmp_all(img1 * alphamap_full0[..., None] / alphamap_full0.max(),
            img2 * alphamap_full1[..., None] / alphamap_full1.max())
#%%
# matplotlib switch to pycharm interactive backend 
import matplotlib
matplotlib.use("module://backend_interagg")

#%%
import scipy.ndimage as ndimage
# create a mask center is 1, surreounded by 0.5 and then 0
naive_featmask_L4 = np.zeros((7, 7))
naive_featmask_L4[1:-1, 1:-1] = 0.5
naive_featmask_L4[2:-2, 2:-2] = 1
naive_featmask_L3 = ndimage.zoom(naive_featmask_L4, 2, order=0)
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(naive_featmask_L4)
plt.subplot(122)
plt.imshow(naive_featmask_L3)
plt.show()
naive_featmask_L3 = torch.from_numpy(naive_featmask_L3).float().to("cuda")
naive_featmask_L4 = torch.from_numpy(naive_featmask_L4).float().to("cuda")
#%%
from easydict import EasyDict as edict
# naive_featmask[alphamap_full0 > 0.5] = 0.5
# compute similarity with masks in ResNet50 layer4 layer3
# Expi = 118
for rep in trange(50):
    imgdist_col = []
    for Expi in trange(1, 191):
        if not os.path.exists(join(protosumdir, f"Exp{Expi}_proto_attr_montage.jpg")):
            # raise ValueError("Montage not found")
            continue
        stat = edict()
        exp_row = meta_act_df[meta_act_df.Expi == Expi].iloc[0]
        mtg = plt.imread(join(protosumdir, f"Exp{Expi}_proto_attr_montage.jpg"))
        FC_maxblk, FC_maxblk_avg, FC_reevol_G, FC_reevol_pix, \
                       BG_maxblk, BG_maxblk_avg, BG_reevol_G, BG_reevol_pix, \
                       both_reevol_G, both_reevol_pix = parse_montage(mtg)
        # real pair
        cnn_L3_msk_scl = compare_imgs_cnn_featmsk(FC_reevol_G, BG_reevol_G, fetcher_cnn,
                            featmsk1=naive_featmask_L3, featkey='layer3', metric='cosine')
        cnn_L4_msk_scl = compare_imgs_cnn_featmsk(FC_reevol_G, BG_reevol_G, fetcher_cnn,
                            featmsk1=naive_featmask_L4, featkey='layer4', metric='cosine')
        print(f"FC-BG L3 {cnn_L3_msk_scl.item():.3f} L4 {cnn_L4_msk_scl.item():.3f}")
        # control shuffled pair
        shflmsk = ~((meta_act_df.prefchan == exp_row.prefchan) &
          (meta_act_df.Animal == exp_row.Animal))
        shfl_row = meta_act_df[shflmsk].sample(1).iloc[0]
        ## load alternative experiment
        Expi_alt = shfl_row.Expi
        mtg_alt = plt.imread(join(protosumdir, f"Exp{Expi_alt}_proto_attr_montage.jpg"))
        FC_maxblk_alt, FC_maxblk_avg_alt, FC_reevol_G_alt, FC_reevol_pix_alt, \
            BG_maxblk_alt, BG_maxblk_avg_alt, BG_reevol_G_alt, BG_reevol_pix_alt, \
            both_reevol_G_alt, both_reevol_pix_alt = parse_montage(mtg_alt)
        ## compute similarity with shuffled.
        cnn_L3_msk_scl_BGalt = compare_imgs_cnn_featmsk(FC_reevol_G, BG_reevol_G_alt, fetcher_cnn,
                                    featmsk1=naive_featmask_L3, featkey='layer3', metric='cosine')
        cnn_L3_msk_scl_FCalt = compare_imgs_cnn_featmsk(FC_reevol_G_alt, BG_reevol_G, fetcher_cnn,
                                    featmsk1=naive_featmask_L3, featkey='layer3', metric='cosine')
        cnn_L4_msk_scl_BGalt = compare_imgs_cnn_featmsk(FC_reevol_G, BG_reevol_G_alt, fetcher_cnn,
                                    featmsk1=naive_featmask_L4, featkey='layer4', metric='cosine')
        cnn_L4_msk_scl_FCalt = compare_imgs_cnn_featmsk(FC_reevol_G_alt, BG_reevol_G, fetcher_cnn,
                                    featmsk1=naive_featmask_L4, featkey='layer4', metric='cosine')
        print(f"Exp {Expi} vs {Expi_alt}")
        print(f"FC-BG L3 {cnn_L3_msk_scl.item():.3f} L4 {cnn_L4_msk_scl.item():.3f}")
        print(f"FC-BG' L3 {cnn_L3_msk_scl_BGalt.item():.3f} L4 {cnn_L4_msk_scl_BGalt.item():.3f}")
        print(f"FC'-BG L3 {cnn_L3_msk_scl_FCalt.item():.3f} L4 {cnn_L4_msk_scl_FCalt.item():.3f}")
        stat.Expi = Expi
        stat.resnet_L3 = cnn_L3_msk_scl.item()
        stat.resnet_L4 = cnn_L4_msk_scl.item()
        stat.Expi_alt = Expi_alt
        stat.resnet_L3_BGalt = cnn_L3_msk_scl_BGalt.item()
        stat.resnet_L4_BGalt = cnn_L4_msk_scl_BGalt.item()
        stat.resnet_L3_FCalt = cnn_L3_msk_scl_FCalt.item()
        stat.resnet_L4_FCalt = cnn_L4_msk_scl_FCalt.item()
        imgdist_col.append(stat)

    imgdist_df = pd.DataFrame(imgdist_col)
    imgdist_df.to_csv(join(tabdir, f"resnet50_imgdist_df_rep{rep:02d}.csv"), index=False)
#%%
imgdist_df_cat = pd.concat([pd.read_csv(join(tabdir, f"resnet50_imgdist_df_rep{rep:02d}.csv"))
                        for rep in range(50)])
#%%
tmpdf = pd.read_csv(join(tabdir, f"resnet50_imgdist_df_rep{rep:02d}.csv"))
#%% average over reps
imgdist_df = imgdist_df_cat.groupby(['Expi', ]).mean().reset_index()
#%%
meta_imgdist_df = pd.merge(meta_act_df, imgdist_df, on="Expi")
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
pd.concat([imgdist_df[succmsk & ITmsk].mean(),
           imgdist_df[succmsk & V4msk].mean()], axis=1)

#%%
plt.subplot(121)
plt.imshow(FC_reevol_G)
plt.subplot(122)
plt.imshow(BG_reevol_G_alt)
plt.show()
#%%
from scipy.stats import pearsonr, ttest_rel, ttest_ind, ttest_1samp
def ttest_rel_df(df, msk, col1, col2):
    return ttest_rel(df[msk][col1], df[msk][col2])


def ttest_ind_df(df, msk1, msk2, col):
    return ttest_ind(df[msk1][col], df[msk2][col])


ttest_rel_df(imgdist_df, cmpmsk & ITmsk, 'resnet_L4', 'resnet_L4_BGalt')

#%%
ttest_rel_df(imgdist_df, succmsk & ITmsk, 'resnet_L4', 'resnet_L4_FCalt')

#%%
ttest_ind_df(imgdist_df, ITmsk, V4msk, 'resnet_L3')
#%%
def paired_strip_plot(df, msk, col1, col2):
    vec1 = df[msk][col1]
    vec2 = df[msk][col2]
    xjitter = 0.1 * np.random.randn(len(vec1))
    plt.figure(figsize=[4, 6])
    plt.scatter(xjitter, vec1)
    plt.scatter(xjitter+1, vec2)
    plt.plot(np.arange(2)[:,None]+xjitter[None,:],
             np.stack((vec1, vec2)), color="k", alpha=0.1)
    plt.xticks([0,1], [col1, col2])

paired_strip_plot(imgdist_df, succmsk & ITmsk, "resnet_L4", "resnet_L4_FCalt")
plt.title("both succeed, IT units\n Resnet robust Layer 4 center cosine")
plt.show()
#%%
paired_strip_plot(imgdist_df, ~succmsk & ITmsk, "resnet_L4", "resnet_L4_FCalt")
plt.title("At least one failed, IT units\n Resnet robust Layer 4 center cosine")
plt.show()
#%%
paired_strip_plot(imgdist_df, nonemsk & ITmsk, "resnet_L4", "resnet_L4_FCalt")
plt.title("None succeed, IT units\n Resnet robust Layer 4 center cosine")
plt.show()
#%%
paired_strip_plot(imgdist_df, succmsk & V4msk, "resnet_L4", "resnet_L4_FCalt")
plt.title("both succeed, V4 units\n Resnet robust Layer 4 center cosine")
plt.show()
#%%
paired_strip_plot(imgdist_df, ~succmsk & V4msk, "resnet_L4", "resnet_L4_FCalt")
plt.title("At least one failed, V4 units\n Resnet robust Layer 4 center cosine")
plt.show()
#%%
paired_strip_plot(imgdist_df, nonemsk & V4msk, "resnet_L4", "resnet_L4_FCalt")
plt.title("None succeed, V4 units\n Resnet robust Layer 4 center cosine")
plt.show()
#%%
ttest_rel(imgdist_df[succmsk & ITmsk].resnet_L4,
          imgdist_df[succmsk & ITmsk].resnet_L4_FCalt)
#%%
#%%
ttest_rel(imgdist_df[nonemsk & ITmsk].resnet_L4,
          imgdist_df[nonemsk & ITmsk].resnet_L4_FCalt)
#%%
ttest_rel(imgdist_df[~succmsk & ITmsk].resnet_L4,
          imgdist_df[~succmsk & ITmsk].resnet_L4_FCalt)
#%%
ttest_rel(imgdist_df[succmsk & V4msk].resnet_L4,
          imgdist_df[succmsk & V4msk].resnet_L4_FCalt)

#%%
ttest_rel(imgdist_df[succmsk & V4msk].resnet_L3,
          imgdist_df[succmsk & V4msk].resnet_L3_FCalt)
#%%
ttest_ind(imgdist_df[succmsk & V4msk].resnet_L3,
          imgdist_df[succmsk & ITmsk].resnet_L3)
#%%
ttest_ind(imgdist_df[succmsk & ITmsk].resnet_L4,
          imgdist_df[succmsk & V4msk].resnet_L4,)
#%%
ttest_ind(imgdist_df[succmsk & ITmsk].resnet_L4,
          imgdist_df[~succmsk & ITmsk].resnet_L4)

#%%
for _, row in imgdist_df.iterrows():
    print(int(row.Expi))

#%%
#%%
def ttest_1samp_print(seq, scalar):
    tval, pval = ttest_1samp(seq, scalar)
    print(f"{seq.mean():.3f}+-{seq.std():.3f} ~ {scalar:.3f} tval: {tval:.2f}, pval: {pval:.1e}")
    return tval, pval
#%%

# imgdist_Exp.resnet_L3.iloc[0]
# imgdist_Exp.resnet_L4.iloc[0]
# imgdist_Exp.resnet_L3_FCalt
# imgdist_Exp.resnet_L3_BGalt
# imgdist_Exp.resnet_L4_FCalt
# imgdist_Exp.resnet_L4_BGalt
imgdist_Exp = imgdist_df_cat[imgdist_df_cat.Expi == 177]# int(row.Expi)
print("L4 FC'-BG vs FC-BG(orig)", end="\t")
ttest_1samp_print(imgdist_Exp.resnet_L4_FCalt, imgdist_Exp.resnet_L4.iloc[0])
print("L4 FC-BG' vs FC-BG(orig)", end="\t")
ttest_1samp_print(imgdist_Exp.resnet_L4_BGalt, imgdist_Exp.resnet_L4.iloc[0])
print("L3 FC'-BG vs FC-BG(orig)", end="\t")
ttest_1samp_print(imgdist_Exp.resnet_L3_FCalt, imgdist_Exp.resnet_L3.iloc[0])
print("L3 FC-BG' vs FC-BG(orig)", end="\t")
ttest_1samp_print(imgdist_Exp.resnet_L3_BGalt, imgdist_Exp.resnet_L3.iloc[0])

#%%

