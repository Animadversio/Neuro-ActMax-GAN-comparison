"""Compare a paired evolved image with some non-paired images

Updated version, doing independent test insteado paired test. fetching non pairs from the whole matrix.

"""
import os
import torch
import numpy as np
from os.path import join
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms
from core.utils.montage_utils import crop_all_from_montage, make_grid_np, build_montages, crop_from_montage
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data, get_expstr
from core.utils.dataset_utils import ImagePathDataset, ImageFolder
from timm import list_models, create_model
from lpips import LPIPS
from torchvision.models import resnet50, alexnet
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from neuro_data_analysis.neural_data_lib import get_expstr
from neuro_data_analysis.neural_data_lib import parse_montage
# from neuro_data_analysis.image_comparison_lib import compare_imgs_cnn, compare_imgs_cnn_featmsk, \
#     compare_imgs_vit, compare_imgs_LPIPS
from core.utils.CNN_scorers import load_featnet
from tqdm import trange, tqdm
import pickle as pkl
from easydict import EasyDict as edict
from scipy.stats import pearsonr, ttest_rel, ttest_ind, ttest_1samp
from core.utils.stats_utils import ttest_rel_df, ttest_ind_df, ttest_ind_print, ttest_1samp_print, \
    ttest_rel_print, ttest_ind_print_df, ttest_rel_print_df
from neuro_data_analysis.image_comparison_lib import compare_imgs_cnn, compare_imgs_cnn_featmsk, \
    compare_imgs_LPIPS, naive_featmsk, extract_featvec_cnn_featmsk
from neuro_data_analysis.neural_data_utils import get_all_masks
#%%
protosumdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoSummary"
tabdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables"
meta_act_df = pd.read_csv(join(tabdir, "meta_activation_stats_w_optimizer.csv"), )  # meta_activation_stats.csv
#%%
Amsk, Bmsk, V1msk, V4msk, ITmsk, length_msk, spc_msk, \
    sucsmsk, bsl_unstable_msk, bsl_stable_msk, validmsk = get_all_masks(meta_act_df)
pthresh = 0.01
bothsucmsk = (meta_act_df.p_maxinit_0 < pthresh) & (meta_act_df.p_maxinit_1 < pthresh)
FCsucsmsk = (meta_act_df.p_maxinit_0 < pthresh)
BGsucsmsk = (meta_act_df.p_maxinit_1 < pthresh)
anysucsmsk = (meta_act_df.p_maxinit_0 < pthresh) | (meta_act_df.p_maxinit_1 < pthresh)
nonemsk = (meta_act_df.p_maxinit_0 > pthresh) & (meta_act_df.p_maxinit_1 > pthresh)
#%%
cnnmodel, _ = load_featnet("resnet50_linf8",)
# get_graph_node_names(cnnmodel)
fetcher_cnn = create_feature_extractor(cnnmodel, ['layer3', "layer4", "avgpool"])
fetcher_cnn = fetcher_cnn.cuda().eval()
#%%

#%%
from torch.utils.data import Dataset, DataLoader
class ListImageDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Convert NumPy array to Torch tensor
        sample = torch.from_numpy(sample.transpose((2, 0, 1))).float()
        return sample

def extract_feat(dataset, featnet, device="cuda", batch_size=10):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    featnet.eval().to(device)
    feat_col = []
    for imgtsr in tqdm(dl):
        with torch.no_grad():
            feat = featnet(imgtsr.to(device))
        feat_col.append(feat.cpu())
    feattsr = torch.cat(feat_col, dim=0)
    return feattsr

# dinov2 model
featnet = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
#%%
from collections import defaultdict
# cmp_sfx = "reevol_G"
FC_img_col = defaultdict(list)
BG_img_col = defaultdict(list)
meta_col = []
for Expi in trange(1, 191):
    if not os.path.exists(join(protosumdir, f"Exp{Expi}_proto_attr_montage.jpg")):
        continue
    exp_row = meta_act_df[meta_act_df.Expi == Expi].iloc[0]
    mtg = plt.imread(join(protosumdir, f"Exp{Expi}_proto_attr_montage.jpg"))
    Imgs = parse_montage(mtg)
    for cmp_sfx in ["reevol_G", "reevol_pix", "maxblk"]:
        FC_img_col[cmp_sfx].append(Imgs["FC_"+cmp_sfx])
        BG_img_col[cmp_sfx].append(Imgs["BG_"+cmp_sfx])
    meta_col.append(exp_row)
#%%
# turn a list of 224 224 3 numpy images into a dataset and dataloader
# use default transform for the image, including RGB norm and resize
img = torch.from_numpy(FC_img_col["reevol_G"][0]).permute(2, 0, 1).unsqueeze(0)
featnet(img).shape
#%%
import scipy.ndimage as ndimage
"""create a mask center is 1, surreounded by 0.5 and then 0"""
naive_featmask_L4 = np.zeros((7, 7))
cent_featmask_L4 = np.zeros((7, 7))
all_featmask_L4 = np.ones((7, 7))
naive_featmask_L4[1:-1, 1:-1] = 0.25  # 0.5
naive_featmask_L4[2:-2, 2:-2] = 0.5
naive_featmask_L4[3:-3, 3:-3] = 1
cent_featmask_L4[3, 3] = 1
naive_featmask_L3 = ndimage.zoom(naive_featmask_L4, 2, order=0)
cent_featmask_L3 = ndimage.zoom(cent_featmask_L4, 2, order=0)
all_featmask_L3 = np.ones((14, 14))
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(naive_featmask_L4)
plt.subplot(122)
plt.imshow(naive_featmask_L3)
plt.show()
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(cent_featmask_L4)
plt.subplot(122)
plt.imshow(cent_featmask_L3)
plt.show()
naive_featmask_L3 = torch.from_numpy(naive_featmask_L3).float().to("cuda")
naive_featmask_L4 = torch.from_numpy(naive_featmask_L4).float().to("cuda")
cent_featmask_L4 = torch.from_numpy(cent_featmask_L4).float().to("cuda")
cent_featmask_L3 = torch.from_numpy(cent_featmask_L3).float().to("cuda")
all_featmask_L4 = torch.from_numpy(all_featmask_L4).float().to("cuda")
all_featmask_L3 = torch.from_numpy(all_featmask_L3).float().to("cuda")
#%%
"""Obtain feature collection for each image set"""
FC_featvec_col = {}
BG_featvec_col = {}
for cmp_sfx in ["reevol_G", "reevol_pix", "maxblk"]:
    FC_featvec_col[cmp_sfx+"_RNrobust_L3focus"] = extract_featvec_cnn_featmsk(FC_img_col[cmp_sfx],
                                  fetcher_cnn, featmsk1=naive_featmask_L3, featkey='layer3')
    FC_featvec_col[cmp_sfx+"_RNrobust_L4focus"] = extract_featvec_cnn_featmsk(FC_img_col[cmp_sfx],
                                    fetcher_cnn, featmsk1=naive_featmask_L4, featkey='layer4')
    FC_featvec_col[cmp_sfx+"_RNrobust_L3cent"] = extract_featvec_cnn_featmsk(FC_img_col[cmp_sfx],
                                    fetcher_cnn, featmsk1=cent_featmask_L3, featkey='layer3')
    FC_featvec_col[cmp_sfx+"_RNrobust_L4cent"] = extract_featvec_cnn_featmsk(FC_img_col[cmp_sfx],
                                    fetcher_cnn, featmsk1=cent_featmask_L4, featkey='layer4')
    FC_featvec_col[cmp_sfx+"_RNrobust_L3all"] = extract_featvec_cnn_featmsk(FC_img_col[cmp_sfx],
                                    fetcher_cnn, featmsk1=all_featmask_L3, featkey='layer3')
    FC_featvec_col[cmp_sfx+"_RNrobust_L4all"] = extract_featvec_cnn_featmsk(FC_img_col[cmp_sfx],
                                    fetcher_cnn, featmsk1=all_featmask_L4, featkey='layer4')
    BG_featvec_col[cmp_sfx+"_RNrobust_L3focus"] = extract_featvec_cnn_featmsk(BG_img_col[cmp_sfx],
                                    fetcher_cnn, featmsk1=naive_featmask_L3, featkey='layer3')
    BG_featvec_col[cmp_sfx+"_RNrobust_L4focus"] = extract_featvec_cnn_featmsk(BG_img_col[cmp_sfx],
                                    fetcher_cnn, featmsk1=naive_featmask_L4, featkey='layer4')
    BG_featvec_col[cmp_sfx+"_RNrobust_L3cent"] = extract_featvec_cnn_featmsk(BG_img_col[cmp_sfx],
                                    fetcher_cnn, featmsk1=cent_featmask_L3, featkey='layer3')
    BG_featvec_col[cmp_sfx+"_RNrobust_L4cent"] = extract_featvec_cnn_featmsk(BG_img_col[cmp_sfx],
                                    fetcher_cnn, featmsk1=cent_featmask_L4, featkey='layer4')
    BG_featvec_col[cmp_sfx+"_RNrobust_L3all"] = extract_featvec_cnn_featmsk(BG_img_col[cmp_sfx],
                                    fetcher_cnn, featmsk1=all_featmask_L3, featkey='layer3')
    BG_featvec_col[cmp_sfx+"_RNrobust_L4all"] = extract_featvec_cnn_featmsk(BG_img_col[cmp_sfx],
                                    fetcher_cnn, featmsk1=all_featmask_L4, featkey='layer4')
    FC_featvec_col[cmp_sfx+"_dinov2"] = extract_feat(ListImageDataset(FC_img_col[cmp_sfx]),
                                     featnet, device="cuda", batch_size=16)
    BG_featvec_col[cmp_sfx+"_dinov2"] = extract_feat(ListImageDataset(BG_img_col[cmp_sfx]),
                                     featnet, device="cuda", batch_size=16)
#%%
import seaborn as sns
from torchmetrics.functional import pairwise_cosine_similarity
def non_diagonal_values(matrix):
    """ all off-diagonal values, (assuming matrix is not symmetric)
    For sym metrix, need to do only subdiagonal ..

    Example:
        # extract diagonal values from this matrix
        paired_xsim = np.diag(xsimmat)
        # collect all off-diagonal values
        unpaired_xsim = non_diagonal_values(xsimmat)

    :param matrix:
    :return:
    """
    n = matrix.shape[0]
    non_diagonal = matrix[~np.eye(n, dtype=bool)]
    return non_diagonal

imglabel = "reevol_pix"  #"maxblk"# "reevol_pix"  #"maxblk" # reevol_pix
featspace = "RNrobust_L4cent"  # L4 dinov2
# cross similarity matrix
# xsimmat = pairwise_cosine_similarity(FC_featvec_col["reevol_G_L4"],
#                                      BG_featvec_col["reevol_G_L4"]).cpu().numpy()
xsimmat = pairwise_cosine_similarity(FC_featvec_col[imglabel+"_"+featspace],
                                     BG_featvec_col[imglabel+"_"+featspace]).cpu().numpy()
# test on different masks.
for areamsk, areaname in zip([V1msk, V4msk, ITmsk, True], ["V1", "V4", "IT", "All area"]):
    for sucsmsk, sucname in zip([bothsucmsk, anysucsmsk, nonemsk, True],
                                ["Both success", "Any success", "None success", "All"]):
        expmsk = validmsk & areamsk & sucsmsk
        label = f"{areaname} {sucname}"
        paired_xsim = np.diag(xsimmat[expmsk, :][:, expmsk])
        unpaired_xsim = non_diagonal_values(xsimmat[expmsk, :][:, expmsk])
        print(label, end="\t")
        tval, pval, tstr = ttest_ind_print(paired_xsim, unpaired_xsim, sem=False)
        if pval < 0.05:
            print("**")
# extract_featvec_cnn_featmsk

#%%
"""export to a txt file"""
from contextlib import redirect_stdout

with open(join(tabdir, "proto_img_similarity_shuffle_cmp_synop.txt"), 'w') as f:
    with redirect_stdout(f):
        for featspace in ["RNrobust_L3focus", "RNrobust_L4focus",
                              "RNrobust_L3cent", "RNrobust_L4cent",
                              "RNrobust_L3all", "RNrobust_L4all", "dinov2"]:
            for imglabel in ["reevol_G", "reevol_pix", "maxblk"]:
                print("Image similarity comparison between paired and unpaired images.")
                print("Image set: ", imglabel)
                print("Feature space: ", featspace)
                xsimmat = pairwise_cosine_similarity(FC_featvec_col[imglabel+"_"+featspace],
                                                     BG_featvec_col[imglabel+"_"+featspace])
                xsimmat = xsimmat.cpu().numpy()
                for areamsk, areaname in zip([V1msk, V4msk, ITmsk, True], ["V1", "V4", "IT", "All area"]):
                    for sucsmsk, sucname in zip([bothsucmsk, anysucsmsk, nonemsk, True],
                                                ["Both success", "Any success", "None success", "All"]):
                        expmsk = validmsk & areamsk & sucsmsk
                        label = f"{areaname} {sucname}"
                        paired_xsim = np.diag(xsimmat[expmsk, :][:, expmsk])
                        unpaired_xsim = non_diagonal_values(xsimmat[expmsk, :][:, expmsk])
                        print(label, end="\t")
                        tval, pval, tstr = ttest_ind_print(paired_xsim, unpaired_xsim, sem=False)
                        if pval < 0.05:
                            print("**")
                print("\n"+"-"*80)

#%%
paired_xsim_all = np.diag(xsimmat)
ttest_ind(paired_xsim_all[validmsk&bothsucmsk],
          paired_xsim_all[validmsk&~bothsucmsk],)
#%%
plt.figure(figsize=[10, 10])
sns.heatmap(xsimmat)
plt.axis("equal")
plt.show()


#%%
# naive_featmask[alphamap_full0 > 0.5] = 0.5
# compute similarity with masks in ResNet50 layer4 layer3
#
# for rep in trange(50):
#     # repeat 50 times sample distribution of unpaired evolution
#     imgdist_col = []
#     for Expi in trange(1, 191):
#         if not os.path.exists(join(protosumdir, f"Exp{Expi}_proto_attr_montage.jpg")):
#             # raise ValueError("Montage not found")
#             continue
#         stat = edict()
#         exp_row = meta_act_df[meta_act_df.Expi == Expi].iloc[0]
#         mtg = plt.imread(join(protosumdir, f"Exp{Expi}_proto_attr_montage.jpg"))
#         Imgs = parse_montage(mtg)
#         # control shuffled pair
#         shflmsk = ~((meta_act_df.prefchan == exp_row.prefchan) &
#                     (meta_act_df.Animal == exp_row.Animal))
#         shfl_row = meta_act_df[shflmsk].sample(1).iloc[0]
#         ## load alternative experiment
#         Expi_alt = shfl_row.Expi
#         mtg_alt = plt.imread(join(protosumdir, f"Exp{Expi_alt}_proto_attr_montage.jpg"))
#         Imgs_alt = parse_montage(mtg_alt)
#         cmp_sfx = "reevol_G"  # "maxblk"
#         for cmp_sfx in ["reevol_G", "reevol_pix", "maxblk"]:
#             # real pair
#             cnn_L3_msk_scl = compare_imgs_cnn_featmsk(Imgs["FC_"+cmp_sfx], Imgs["BG_"+cmp_sfx], fetcher_cnn,
#                                 featmsk1=naive_featmask_L3, featkey='layer3', metric='cosine')
#             cnn_L4_msk_scl = compare_imgs_cnn_featmsk(Imgs["FC_"+cmp_sfx], Imgs["BG_"+cmp_sfx], fetcher_cnn,
#                                 featmsk1=naive_featmask_L4, featkey='layer4', metric='cosine')
#             print(f"FC-BG L3 {cnn_L3_msk_scl.item():.3f} L4 {cnn_L4_msk_scl.item():.3f}")
#             # compute similarity with shuffled.
#             cnn_L3_msk_scl_BGalt = compare_imgs_cnn_featmsk(Imgs["FC_"+cmp_sfx], Imgs_alt["BG_"+cmp_sfx], fetcher_cnn,
#                                 featmsk1=naive_featmask_L3, featkey='layer3', metric='cosine')
#             cnn_L3_msk_scl_FCalt = compare_imgs_cnn_featmsk(Imgs_alt["FC_"+cmp_sfx], Imgs["BG_"+cmp_sfx], fetcher_cnn,
#                                 featmsk1=naive_featmask_L3, featkey='layer3', metric='cosine')
#             cnn_L4_msk_scl_BGalt = compare_imgs_cnn_featmsk(Imgs["FC_"+cmp_sfx], Imgs_alt["BG_"+cmp_sfx], fetcher_cnn,
#                                 featmsk1=naive_featmask_L4, featkey='layer4', metric='cosine')
#             cnn_L4_msk_scl_FCalt = compare_imgs_cnn_featmsk(Imgs_alt["FC_"+cmp_sfx], Imgs["BG_"+cmp_sfx], fetcher_cnn,
#                                 featmsk1=naive_featmask_L4, featkey='layer4', metric='cosine')
#             print(f"Exp {Expi} vs {Expi_alt}")
#             print(f"FC-BG L3 {cnn_L3_msk_scl.item():.3f} L4 {cnn_L4_msk_scl.item():.3f}")
#             print(f"FC-BG' L3 {cnn_L3_msk_scl_BGalt.item():.3f} L4 {cnn_L4_msk_scl_BGalt.item():.3f}")
#             print(f"FC'-BG L3 {cnn_L3_msk_scl_FCalt.item():.3f} L4 {cnn_L4_msk_scl_FCalt.item():.3f}")
#             stat.Expi = Expi
#             stat[cmp_sfx+"_resnet_L3"] = cnn_L3_msk_scl.item()
#             stat[cmp_sfx+"_resnet_L4"] = cnn_L4_msk_scl.item()
#             stat.Expi_alt = Expi_alt
#             stat[cmp_sfx+"_resnet_L3_BGalt"] = cnn_L3_msk_scl_BGalt.item()
#             stat[cmp_sfx+"_resnet_L4_BGalt"] = cnn_L4_msk_scl_BGalt.item()
#             stat[cmp_sfx+"_resnet_L3_FCalt"] = cnn_L3_msk_scl_FCalt.item()
#             stat[cmp_sfx+"_resnet_L4_FCalt"] = cnn_L4_msk_scl_FCalt.item()
#         imgdist_col.append(stat)
#
#     imgdist_df = pd.DataFrame(imgdist_col)
#     imgdist_df.to_csv(join(tabdir, f"resnet50_imgdist_df_rep{rep:02d}_mskchange.csv"), index=False)
# #%%
# # tmpdf = pd.read_csv(join(tabdir, f"resnet50_imgdist_df_rep{rep:02d}.csv"))
# #%%
# # imgdist_df_cat = pd.concat([pd.read_csv(join(tabdir, f"resnet50_imgdist_df_rep{rep:02d}.csv"))
# #                         for rep in range(50)])
# imgdist_df_cat = pd.concat([pd.read_csv(join(tabdir, f"resnet50_imgdist_df_rep{rep:02d}_mskchange.csv"))
#                         for rep in range(5)])
# #%% average over reps
# imgdist_df_avg = imgdist_df_cat.groupby(['Expi', ]).mean().reset_index()
# #%%
# meta_imgdist_df = pd.merge(meta_act_df, imgdist_df_avg, on="Expi")
# #%%
# V1msk = meta_act_df.visual_area == "V1"
# V4msk = meta_act_df.visual_area == "V4"
# ITmsk = meta_act_df.visual_area == "IT"
# cmpmsk = (meta_act_df.maxrsp_1_mean - meta_act_df.maxrsp_0_mean).abs() \
#          < (meta_act_df.maxrsp_0_sem + meta_act_df.maxrsp_1_sem)
# spc_msk = (meta_act_df.space1 == "fc6") & meta_act_df.space2.str.contains("BigGAN")
# length_msk = (meta_act_df.blockN > 14)
# baseline_jump_list = ["Beto-18082020-002",
#                       "Beto-07092020-006",
#                       "Beto-14092020-002",
#                       "Beto-27102020-003",
#                       "Alfa-22092020-003",
#                       "Alfa-04092020-003"]
# bsl_unstable_msk = meta_act_df.ephysFN.str.contains("|".join(baseline_jump_list), case=True, regex=True)
# assert bsl_unstable_msk.sum() == len(baseline_jump_list)
# bsl_stable_msk = ~bsl_unstable_msk
# validmsk = length_msk & bsl_stable_msk & spc_msk
# p_thresh = 0.05
# bothsucsmsk = (meta_act_df.p_maxinit_0 < p_thresh) & \
#           (meta_act_df.p_maxinit_1 < p_thresh)
# anysucsmsk = (meta_act_df.p_maxinit_0 < p_thresh) | \
#              (meta_act_df.p_maxinit_1 < p_thresh)
# nonemsk = (meta_act_df.p_maxinit_0 > p_thresh) & \
#           (meta_act_df.p_maxinit_1 > p_thresh)
# #%%
# pd.concat([imgdist_df[bothsucsmsk& ITmsk & validmsk].mean(),
#            imgdist_df[bothsucsmsk& V4msk & validmsk].mean()], axis=1)
#
# #%%
# plt.subplot(121)
# plt.imshow(FC_reevol_G)
# plt.subplot(122)
# plt.imshow(BG_reevol_G_alt)
# plt.show()
#
# #%%
# for areamsk, areaname in zip([V1msk, V4msk, ITmsk], ["V1", "V4", "IT"]):
#     for commonmsk, mskstr in zip([cmpmsk, bothsucsmsk, anysucsmsk, nonemsk],
#                                  ["Comparable activation", "Both success", "Any success", "None success"]):
#         for metric_layer in ["L3", "L4"]:
#             print(f"[{areaname} {mskstr} thr{p_thresh}  Metric layer: {metric_layer}]")
#             for sfx in ["maxblk", "reevol_pix", "reevol_G"]:
#                 ttest_rel_print_df(imgdist_df_avg, commonmsk & areamsk & validmsk,
#                              sfx+f'_resnet_{metric_layer}', sfx+f'_resnet_{metric_layer}_FCalt', sem=True)
#         print("")
# #%%
# # redirect stdout to file
# import sys
# from contextlib import redirect_stdout
# with open(join(tabdir, f"proto_imgsim_shufl_cmp_p{p_thresh}.txt"), 'w') as f:
#     with redirect_stdout(f):
#         for areamsk, areaname in zip([V1msk, V4msk, ITmsk], ["V1", "V4", "IT"]):
#             for commonmsk, mskstr in zip([cmpmsk, bothsucsmsk, anysucsmsk, nonemsk],
#                                          ["Comparable activation", "Both success", "Any success", "None success"]):
#                 for metric_layer in ["L3", "L4"]:
#                     print(f"[{areaname} {mskstr} thr{p_thresh}  Metric layer: {metric_layer}]")
#                     for sfx in ["maxblk", "reevol_pix", "reevol_G"]:
#                         ttest_rel_print_df(imgdist_df_avg, commonmsk & areamsk & validmsk,
#                                            sfx + f'_resnet_{metric_layer}', sfx + f'_resnet_{metric_layer}_FCalt', sem=True)
#                 print("")
#
# #%%
# for sfx in ["maxblk", "reevol_pix", "reevol_G"]:
#     ttest_rel_print_df(imgdist_df_avg, bothsucsmsk & ITmsk & validmsk, sfx+'_resnet_L4', sfx+'_resnet_L4_FCalt')
#
# for sfx in ["maxblk", "reevol_pix", "reevol_G"]:
#     ttest_rel_print_df(imgdist_df_avg, cmpmsk & ITmsk & validmsk, sfx+'_resnet_L4', sfx+'_resnet_L4_FCalt')
# #%%
# for sfx in ["maxblk", "reevol_pix", "reevol_G"]:
#     ttest_rel_print_df(imgdist_df_avg, bothsucsmsk& V4msk & validmsk, sfx+'_resnet_L3', sfx+'_resnet_L3_FCalt')
# #%%
# for sfx in ["maxblk", "reevol_pix", "reevol_G"]:
#     ttest_ind_print_df(imgdist_df_avg, cmpmsk & ITmsk & validmsk, nonemsk & ITmsk & validmsk, sfx+'_resnet_L4',)
# # for sfx in ["maxblk", "reevol_pix", "reevol_G"]:
# #     ttest_rel_print_df(imgdist_df_avg, bothsucsmsk& ITmsk & validmsk, sfx+'_resnet_L3', sfx+'_resnet_L3_FCalt')
# #%%
# for sfx in ["maxblk", "reevol_pix", "reevol_G"]:
#     ttest_rel_print_df(imgdist_df_avg, bothsucsmsk& ITmsk & validmsk, sfx+'_resnet_L4', sfx+'_resnet_L4_FCalt')
# for sfx in ["maxblk", "reevol_pix", "reevol_G"]:
#     ttest_rel_print_df(imgdist_df_avg, bothsucsmsk& ITmsk & validmsk, sfx+'_resnet_L3', sfx+'_resnet_L3_FCalt')
# #%%
# for sfx in ["maxblk", "reevol_pix", "reevol_G"]:
#     ttest_rel_print_df(imgdist_df_avg, bothsucsmsk& ITmsk & validmsk, sfx+'_resnet_L4', sfx+'_resnet_L4_BGalt')
# for sfx in ["maxblk", "reevol_pix", "reevol_G"]:
#     ttest_rel_print_df(imgdist_df_avg, bothsucsmsk& ITmsk & validmsk, sfx+'_resnet_L3', sfx+'_resnet_L3_BGalt')
# #%%
# for sfx in ["maxblk", "reevol_pix", "reevol_G"]:
#     ttest_ind_print_df(imgdist_df_avg, bothsucsmsk& ITmsk & validmsk,
#                        bothsucsmsk& V4msk & validmsk, sfx+'_resnet_L4', )
# #%%
# for sfx in ["maxblk", "reevol_pix", "reevol_G"]:
#     ttest_ind_print_df(imgdist_df_avg, bothsucsmsk& ITmsk & validmsk,
#                        ~nonemsk & ITmsk & validmsk, sfx+'_resnet_L4', )
# #%%
# msk = validmsk & ITmsk #& succmsk
# # vec1 = (meta_imgdist_df.maxrsp_1_mean - meta_imgdist_df.initrsp_1_mean) / meta_imgdist_df.maxrsp_1_mean + \
# #          (meta_imgdist_df.maxrsp_0_mean - meta_imgdist_df.initrsp_0_mean) / meta_imgdist_df.maxrsp_0_mean
# # vec1 = (meta_imgdist_df.maxrsp_1_mean - meta_imgdist_df.initrsp_1_mean + \
# #         meta_imgdist_df.maxrsp_0_mean - meta_imgdist_df.initrsp_0_mean) / \
# #        np.maximum(meta_imgdist_df.maxrsp_1_mean, meta_imgdist_df.maxrsp_0_mean) / 2
# # vec1 = (meta_imgdist_df.maxrsp_1_mean + meta_imgdist_df.maxrsp_0_mean) / \
# #        np.maximum(meta_imgdist_df.maxrsp_1_mean, meta_imgdist_df.maxrsp_0_mean) / 2
# vec1 = np.abs(meta_imgdist_df.maxrsp_1_mean - meta_imgdist_df.maxrsp_0_mean) / \
#        np.maximum(meta_imgdist_df.maxrsp_1_mean, meta_imgdist_df.maxrsp_0_mean)
#
#
#
# vec1 = np.abs(meta_imgdist_df.maxrsp_1_mean - meta_imgdist_df.maxrsp_0_mean) / \
#        np.maximum(meta_imgdist_df.maxrsp_1_mean, meta_imgdist_df.maxrsp_0_mean)
# # vec1 = np.abs(meta_imgdist_df.maxrsp_1_mean - meta_imgdist_df.maxrsp_0_mean) / \
# #        (meta_imgdist_df.maxrsp_1_std + meta_imgdist_df.maxrsp_0_std) * 2
# # vec1 = np.abs(meta_imgdist_df.maxrsp_1_mean - meta_imgdist_df.maxrsp_0_mean) / \
# #        (np.maximum(meta_imgdist_df.maxrsp_1_mean, meta_imgdist_df.maxrsp_0_mean) - \
# #         np.minimum(meta_imgdist_df.initrsp_1_mean, meta_imgdist_df.initrsp_0_mean))
# vec2 = meta_imgdist_df.reevol_pix_resnet_L4
# plt.figure(figsize=(6, 6))
# plt.scatter(vec1[msk], vec2[msk])
# rho, pval = pearsonr(vec1[msk], vec2[msk])
# plt.title(f"corr: {rho:.3f} P={pval:.1e} n={msk.sum()}")
# plt.ylabel("prototype Similarity")
# plt.xlabel("change in response")
# plt.show()
# #%%
# ttest_ind_df(imgdist_df, bothsucsmsk& ITmsk & validmsk,
#                          bothsucsmsk& V4msk & validmsk, 'resnet_L4')
#
# #%%
# ttest_ind_df(imgdist_df, ITmsk, V4msk, 'resnet_L3')
# #%%
# ttest_ind_df(imgdist_df, ITmsk, V4msk, 'resnet_L3')
# #%%
# ttest_ind_print(imgdist_df[bothsucsmsk & ITmsk & validmsk]['resnet_L4'],
#                 imgdist_df[bothsucsmsk & V4msk & validmsk]['resnet_L4'])
# #%%
# ttest_rel_print(imgdist_df_avg[bothsucsmsk& V4msk & validmsk]['resnet_L3'],
#                 imgdist_df_avg[bothsucsmsk& V4msk & validmsk]['resnet_L3_BGalt'])
# ttest_rel_print(imgdist_df_avg[bothsucsmsk& V4msk & validmsk]['resnet_L3'],
#                 imgdist_df_avg[bothsucsmsk& V4msk & validmsk]['resnet_L3_FCalt'])
# #%%
# ttest_rel_print(imgdist_df_avg[bothsucsmsk& ITmsk & validmsk]['resnet_L4'],
#                 imgdist_df_avg[bothsucsmsk& ITmsk & validmsk]['resnet_L4_BGalt'])  # not significant
# ttest_rel_print(imgdist_df_avg[bothsucsmsk& ITmsk & validmsk]['resnet_L4'],
#                 imgdist_df_avg[bothsucsmsk& ITmsk & validmsk]['resnet_L4_FCalt'])  # significant
# #%%
# def paired_strip_plot(df, msk, col1, col2):
#     if msk is None:
#         msk = np.ones(len(df), dtype=bool)
#     vec1 = df[msk][col1]
#     vec2 = df[msk][col2]
#     xjitter = 0.1 * np.random.randn(len(vec1))
#     plt.figure(figsize=[4, 6])
#     plt.scatter(xjitter, vec1)
#     plt.scatter(xjitter+1, vec2)
#     plt.plot(np.arange(2)[:,None]+xjitter[None,:],
#              np.stack((vec1, vec2)), color="k", alpha=0.1)
#     plt.xticks([0,1], [col1, col2])
#
# sfx = "reevol_G"
# paired_strip_plot(imgdist_df_avg, bothsucsmsk& ITmsk & validmsk, sfx+"_resnet_L4", sfx+"_resnet_L4_FCalt")
# plt.title("both succeed, IT units\n Resnet robust Layer 4 center cosine")
# plt.show()
# #%%
# paired_strip_plot(imgdist_df_avg, ~bothsucsmsk& ITmsk, "resnet_L4", "resnet_L4_FCalt")
# plt.title("At least one failed, IT units\n Resnet robust Layer 4 center cosine")
# plt.show()
# #%%
# paired_strip_plot(imgdist_df_avg, nonemsk & ITmsk, "resnet_L4", "resnet_L4_FCalt")
# plt.title("None succeed, IT units\n Resnet robust Layer 4 center cosine")
# plt.show()
#
# #%%
# paired_strip_plot(imgdist_df_avg, bothsucsmsk& V4msk, "resnet_L4", "resnet_L4_FCalt")
# plt.title("both succeed, V4 units\n Resnet robust Layer 4 center cosine")
# plt.show()
# #%%
# paired_strip_plot(imgdist_df_avg, ~bothsucsmsk& V4msk, "resnet_L4", "resnet_L4_FCalt")
# plt.title("At least one failed, V4 units\n Resnet robust Layer 4 center cosine")
# plt.show()
# #%%
# paired_strip_plot(imgdist_df, nonemsk & V4msk, "resnet_L4", "resnet_L4_FCalt")
# plt.title("None succeed, V4 units\n Resnet robust Layer 4 center cosine")
# plt.show()
# #%%
# ttest_rel(imgdist_df[bothsucsmsk& ITmsk].resnet_L4,
#           imgdist_df[bothsucsmsk& ITmsk].resnet_L4_FCalt)
# #%%
# #%%
# ttest_rel(imgdist_df[nonemsk & ITmsk].resnet_L4,
#           imgdist_df[nonemsk & ITmsk].resnet_L4_FCalt)
# #%%
# ttest_rel(imgdist_df[~bothsucsmsk& ITmsk].resnet_L4,
#           imgdist_df[~bothsucsmsk& ITmsk].resnet_L4_FCalt)
# #%%
# ttest_rel(imgdist_df[bothsucsmsk& V4msk].resnet_L4,
#           imgdist_df[bothsucsmsk& V4msk].resnet_L4_FCalt)
#
# #%%
# ttest_rel(imgdist_df[bothsucsmsk& V4msk].resnet_L3,
#           imgdist_df[bothsucsmsk& V4msk].resnet_L3_FCalt)
# #%%
# ttest_ind(imgdist_df[bothsucsmsk& V4msk].resnet_L3,
#           imgdist_df[bothsucsmsk& ITmsk].resnet_L3)
# #%%
# ttest_ind(imgdist_df[bothsucsmsk& ITmsk].resnet_L4,
#           imgdist_df[bothsucsmsk& V4msk].resnet_L4,)
# #%%
# ttest_ind(imgdist_df[bothsucsmsk& ITmsk].resnet_L4,
#           imgdist_df[~bothsucsmsk& ITmsk].resnet_L4)
#
# #%%
# for _, row in imgdist_df.iterrows():
#     print(int(row.Expi))
#
# #%%
#
# # imgdist_Exp.resnet_L3.iloc[0]
# # imgdist_Exp.resnet_L4.iloc[0]
# # imgdist_Exp.resnet_L3_FCalt
# # imgdist_Exp.resnet_L3_BGalt
# # imgdist_Exp.resnet_L4_FCalt
# # imgdist_Exp.resnet_L4_BGalt
# imgdist_Exp = imgdist_df_cat[imgdist_df_cat.Expi == 177]# int(row.Expi)
# print("L4 FC'-BG vs FC-BG(orig)", end="\t")
# ttest_1samp_print(imgdist_Exp.resnet_L4_FCalt, imgdist_Exp.resnet_L4.iloc[0])
# print("L4 FC-BG' vs FC-BG(orig)", end="\t")
# ttest_1samp_print(imgdist_Exp.resnet_L4_BGalt, imgdist_Exp.resnet_L4.iloc[0])
# print("L3 FC'-BG vs FC-BG(orig)", end="\t")
# ttest_1samp_print(imgdist_Exp.resnet_L3_FCalt, imgdist_Exp.resnet_L3.iloc[0])
# print("L3 FC-BG' vs FC-BG(orig)", end="\t")
# ttest_1samp_print(imgdist_Exp.resnet_L3_BGalt, imgdist_Exp.resnet_L3.iloc[0])

#%%

