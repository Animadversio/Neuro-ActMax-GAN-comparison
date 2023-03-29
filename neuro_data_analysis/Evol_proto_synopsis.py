"""
Collect the prototypes and their reevolutions into a montage and save them
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lpips import LPIPS
import matplotlib.pyplot as plt
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms
from core.utils.dataset_utils import ImagePathDataset, ImageFolder
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data, get_expstr
_, BFEStats = load_neural_data()
#%%
# Dist = LPIPS(net='squeeze', spatial=True,)
# Dist = Dist.cuda().eval()
# Dist.requires_grad_(False)
#
# #%%
# from timm import list_models, create_model
# # from timm.data import resolve_data_config
# from timm.data.transforms_factory import create_transform
# from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
#
# # list_models('*dino*')
# model = create_model('vit_base_patch16_224_dino', pretrained=True, )
# get_graph_node_names(model)

#%%
# choose backend agg
import matplotlib
matplotlib.use('agg')
#%%
attr_root = r"E:\Network_Data_Sync\BigGAN_FeatAttribution"
protosumdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoSummary"
#%%
# Exp 66 fixed.
#%%
from core.utils.montage_utils import crop_all_from_montage, make_grid_np, build_montages
from os.path import join
from tqdm import tqdm
for Expi in tqdm(range(1, len(BFEStats)+1)):  # 66 is not good
    try:
        explabel = get_expstr(BFEStats, Expi)
    except:
        continue
    if BFEStats[Expi-1]["evol"] is None:
        continue
    imgfps_arr0, resp_arr0, bsl_arr0, gen_arr0 = \
        load_img_resp_pairs(BFEStats, Expi, "Evol", thread=0, output_fmt="arr")
    imgfps_arr1, resp_arr1, bsl_arr1, gen_arr1 = \
        load_img_resp_pairs(BFEStats, Expi, "Evol", thread=1, output_fmt="arr")
    # imgfps_vec0, resp_vec0, bsl_vec0, gen_vec0 = \
    #     load_img_resp_pairs(BFEStats, Expi, "Evol", thread=0, output_fmt="vec")
    # imgfps_vec1, resp_vec1, bsl_vec1, gen_vec1 = \
    #     load_img_resp_pairs(BFEStats, Expi, "Evol", thread=1, output_fmt="vec")
    # raise NotImplementedError
    # imgfps_arr0, resp_arr0, bsl_arr0, gen_arr0 = \
    #     load_img_resp_pairs(BFEStats, Expi, "Evol", thread=0, output_fmt="arr")
    # imgfps_arr1, resp_arr1, bsl_arr1, gen_arr1 = \
    #     load_img_resp_pairs(BFEStats, Expi, "Evol", thread=1, output_fmt="arr")
    #%%
    # if the lAST BLOCK has < 10 images, in either thread, then remove it
    if len(resp_arr0[-1]) < 10:
        resp_arr0 = resp_arr0[:-1]
        bsl_arr0 = bsl_arr0[:-1]
        gen_arr0 = gen_arr0[:-1]
    if len(resp_arr1[-1]) < 10:
        resp_arr1 = resp_arr1[:-1]
        bsl_arr1 = bsl_arr1[:-1]
        gen_arr1 = gen_arr1[:-1]
    #%%
    blck_m_0 = np.array([arr.mean() for arr in resp_arr0])  # np.mean(resp_arr0, axis=1)
    blck_m_1 = np.array([arr.mean() for arr in resp_arr1])  # np.mean(resp_arr1, axis=1)
    #%% max block mean response for each thread and their std. dev.
    maxrsp_blkidx_0 = np.argmax(blck_m_0, axis=0)
    maxrsp_blkidx_1 = np.argmax(blck_m_1, axis=0)
    maxrsp_blk_resps_0 = resp_arr0[maxrsp_blkidx_0]
    maxrsp_blk_resps_1 = resp_arr1[maxrsp_blkidx_1]
    maxrsp_blk_images_0 = [plt.imread(imgfp).astype(np.float32)/255. for imgfp in imgfps_arr0[maxrsp_blkidx_0]]
    maxrsp_blk_images_1 = [plt.imread(imgfp).astype(np.float32)/255. for imgfp in imgfps_arr1[maxrsp_blkidx_1]]

    proto_avg_img_0 = np.mean(maxrsp_blk_images_0, axis=0)
    proto_avg_img_1 = np.mean(maxrsp_blk_images_1, axis=0)
    proto_max_img_0 = maxrsp_blk_images_0[np.argmax(maxrsp_blk_resps_0)]
    proto_max_img_1 = maxrsp_blk_images_1[np.argmax(maxrsp_blk_resps_1)]

    thread_ids = 0, 1, "_cmb"
    # thread_id  = thread_ids[2]
    reevol_img_col = {}
    mtg_fns = [("G", "tsr_resnet50_linf8-layer3_G_layer3.png"),
                ("pix", "tsr_resnet50_linf8-layer3_pix_layer3.png"),]
    for thread_id in thread_ids:
        for label, fn in mtg_fns:
            imgpix = 256 if label == "G" else 224
            mtg = plt.imread(join(attr_root, f"Both_Exp{Expi:02d}_thr{thread_id}", fn))
            imgs = crop_all_from_montage(mtg, imgsize=imgpix, pad=2)
            reevol_img_col[(thread_id, label)] = imgs[0]
    #%%
    white_img = np.ones_like(proto_max_img_0)
    reevol_img_all = [proto_max_img_0, proto_avg_img_0, reevol_img_col[0, "G"], reevol_img_col[0, "pix"],
                      proto_max_img_1, proto_avg_img_1, reevol_img_col[1, "G"], reevol_img_col[1, "pix"],
                            white_img,       white_img, reevol_img_col["_cmb", "G"], reevol_img_col["_cmb", "pix"]]
    # mtg = make_grid_np(reevol_img_col, nrow=2)
    mtg = build_montages(reevol_img_all, (224, 224), (4, 3), transpose=True)[0]
    plt.imsave(join(protosumdir, f"Exp{Expi}_proto_attr_montage.jpg"), mtg)
    plt.figure(figsize=(12, 10.5))
    plt.imshow(mtg)
    plt.suptitle(f"{explabel}\n"
             f" thread 1 resp {maxrsp_blk_resps_0.mean():.1f}+- {maxrsp_blk_resps_0.std():.1f}  vs "
             f"thread 2 resp {maxrsp_blk_resps_1.mean():.1f}+- {maxrsp_blk_resps_1.std():.1f}", fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(join(protosumdir, f"Exp{Expi}_proto_attr_summary.png"))
    plt.show()
#%%
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(proto_max_img_0)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(proto_max_img_1)
plt.axis('off')
plt.tight_layout()
plt.show()

#%%
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(np.mean(maxrsp_blk_images_0, axis=0))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(np.mean(maxrsp_blk_images_1, axis=0))
plt.axis('off')
plt.tight_layout()
plt.show()

#%%
