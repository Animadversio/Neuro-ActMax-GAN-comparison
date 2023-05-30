#%%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lpips import LPIPS
import matplotlib.pyplot as plt
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data
from core.utils.dataset_utils import ImagePathDataset, ImageFolder
#%%
Dist = LPIPS(net='squeeze', spatial=True,)
Dist = Dist.cuda().eval()
Dist.requires_grad_(False)
#%%

_, BFEStats = load_neural_data()
#%%

import os
from os.path import join
import pandas as pd
import seaborn as sns
from easydict import EasyDict as edict
from core.utils.montage_utils import make_grid, make_grid_T, make_grid_np
from neuro_data_analysis.neural_data_lib import get_expstr, load_neural_data
def _shaded_errorbar(x, y, yerr, color, alpha, **kwargs):
    plt.plot(x, y, color=color, **kwargs)
    plt.fill_between(x, y - yerr, y + yerr, color=color, alpha=alpha)


# extract upper triangular part of the square matrix, function
def extract_upper_triangular(x):
    return x[np.triu_indices(x.shape[0], k=1)]


outroot = r"E:\Network_Data_Sync\BigGAN_ImgSim"
#%%
from tqdm import tqdm
for Expi in range(190, len(BFEStats)+1):
    try:
        explabel = get_expstr(BFEStats, Expi)
    except:
        continue
    if BFEStats[Expi-1]["evol"] is None:
        continue
    expdir = join(outroot, f"Both_Exp{Expi}")
    os.makedirs(expdir, exist_ok=True)

    imgfps_arr0, resp_arr0, bsl_arr0, gen_arr0 = \
        load_img_resp_pairs(BFEStats, Expi, "Evol", thread=0, output_fmt="arr")
    imgfps_arr1, resp_arr1, bsl_arr1, gen_arr1 = \
        load_img_resp_pairs(BFEStats, Expi, "Evol", thread=1, output_fmt="arr")
    #%%
    # geni = 22
    imgtsr_avg_FC6 = []
    imgtsr_avg_BG  = []
    for geni in tqdm(range(len(gen_arr0))):
        resp_blk0 = resp_arr0[geni]
        resp_blk1 = resp_arr1[geni]
        if len(resp_blk0) == 0 or len(resp_blk1) == 0:
            continue
        imgs_blk0 = [plt.imread(imgfp) for imgfp in imgfps_arr0[geni]]
        imgs_blk1 = [plt.imread(imgfp) for imgfp in imgfps_arr1[geni]]
        imgtsr_blk0 = torch.tensor(np.stack(imgs_blk0)).permute(0, 3, 1, 2).float() / 255.0
        imgtsr_blk1 = torch.tensor(np.stack(imgs_blk1)).permute(0, 3, 1, 2).float() / 255.0
        #%%
        distmats_crs = Dist.forward_distmat(imgtsr_blk0.cuda(), imgtsr_blk1.cuda()).cpu()
        distmats_FC6 = Dist.forward_distmat(imgtsr_blk0.cuda(), ).cpu()
        distmats_BG  = Dist.forward_distmat(imgtsr_blk1.cuda(), ).cpu()
        #%%
        dist_img_m_FC6 = distmats_FC6.mean(dim=(0, 1, 2))
        dist_img_m_BG  = distmats_BG.mean(dim=(0, 1, 2))
        dist_img_m_crs = distmats_crs.mean(dim=(0, 1, 2))
        dist_img_s_FC6 = distmats_FC6.std(dim=(0, 1, 2))
        dist_img_s_BG  = distmats_BG.std(dim=(0, 1, 2))
        dist_img_s_crs = distmats_crs.std(dim=(0, 1, 2))
        distmat_FC6 = distmats_FC6.mean(dim=(2, 3, 4))
        distmat_BG  = distmats_BG.mean(dim=(2, 3, 4))
        distmat_crs = distmats_crs.mean(dim=(2, 3, 4))

        torch.save({"resp_FC6": resp_blk0,
                    "resp_BG": resp_blk1,
                    "dist_img_m_BG" : dist_img_m_BG,
                    "dist_img_m_FC6": dist_img_m_FC6,
                    "dist_img_m_crs": dist_img_m_crs,
                    "distmat_BG" : distmat_BG,
                    "distmat_FC6": distmat_FC6,
                    "distmat_crs": distmat_crs,
        }, join(expdir, f"distmat_gen{geni}.pt"))
        imgtsr_avg_FC6.append(imgtsr_blk0.mean(dim=0))
        imgtsr_avg_BG.append(imgtsr_blk1.mean(dim=0))

    imgtsr_avg_FC6 = torch.stack(imgtsr_avg_FC6)
    imgtsr_avg_BG  = torch.stack(imgtsr_avg_BG)
    save_imgrid(imgtsr_avg_FC6, join(expdir, "imgtsr_avg_FC6.png"))
    save_imgrid(imgtsr_avg_BG, join(expdir, "imgtsr_avg_BG.png"))

    dist_col = []
    dist_img_m_BG_col = []
    dist_img_m_FC6_col = []
    dist_img_m_crs_col = []
    for geni in range(len(gen_arr0)):
        resp_blk0 = resp_arr0[geni]
        resp_blk1 = resp_arr1[geni]
        if len(resp_blk0) == 0 or len(resp_blk1) == 0:
            continue
        data = torch.load(join(expdir, f"distmat_gen{geni}.pt"))
        S = edict()
        S.dist_BG_m = extract_upper_triangular(data["distmat_BG"].detach().numpy()).mean()
        S.dist_BG_s = extract_upper_triangular(data["distmat_BG"].detach().numpy()).std()
        S.dist_FC6_m = extract_upper_triangular(data["distmat_FC6"].detach().numpy()).mean()
        S.dist_FC6_s = extract_upper_triangular(data["distmat_FC6"].detach().numpy()).std()
        S.dist_crs_m = data["distmat_crs"].detach().numpy().mean()
        S.dist_crs_s = data["distmat_crs"].detach().numpy().std()
        S.resp_BG_m = data["resp_BG"].mean()
        S.resp_BG_s = data["resp_FC6"].std()
        S.resp_FC6_m = data["resp_FC6"].mean()
        S.resp_FC6_s = data["resp_FC6"].std()
        dist_col.append(S)
        dist_img_m_BG = data["dist_img_m_BG"].numpy()
        dist_img_m_FC6 = data["dist_img_m_FC6"].numpy()
        dist_img_m_crs = data["dist_img_m_crs"].numpy()
        dist_img_m_BG_col.append(dist_img_m_BG)
        dist_img_m_FC6_col.append(dist_img_m_FC6)
        dist_img_m_crs_col.append(dist_img_m_crs)
        # plt.imsave(join(expdir, f"dist_img_m_BG_gen{geni}.png"), dist_img_m_BG)
        # plt.imsave(join(expdir, f"dist_img_m_FC6_gen{geni}.png"), dist_img_m_FC6)
        # plt.imsave(join(expdir, f"dist_img_m_crs_gen{geni}.png"), dist_img_m_crs)
    dist_df = pd.DataFrame(dist_col)
    dist_df.to_csv(join(expdir, "distance_tab.csv"))
    #%
    dist_img_tsr = torch.tensor(np.stack(dist_img_m_BG_col)[:,None])
    save_imgrid(dist_img_tsr, join(expdir, "dist_img_m_BG_seq.png"))
    dist_img_tsr = torch.tensor(np.stack(dist_img_m_FC6_col)[:,None])
    save_imgrid(dist_img_tsr, join(expdir, "dist_img_m_FC6_seq.png"))
    dist_img_tsr = torch.tensor(np.stack(dist_img_m_crs_col)[:,None])
    save_imgrid(dist_img_tsr, join(expdir, "dist_img_m_crs_seq.png"))


#%%
    plt.figure(figsize=(6, 6))
    _shaded_errorbar(range(len(dist_df["dist_BG_m"])), dist_df["dist_BG_m"],
                     dist_df["dist_BG_s"], "red", 0.2, label="between BigGAN")
    _shaded_errorbar(range(len(dist_df["dist_FC6_m"])), dist_df["dist_FC6_m"],
                     dist_df["dist_FC6_s"], "blue", 0.2, label="between FC6")
    _shaded_errorbar(range(len(dist_df["dist_crs_m"])), dist_df["dist_crs_m"],
                     dist_df["dist_crs_s"], "black", 0.2, label="across")
    plt.legend(fontsize=14)
    plt.ylabel("LPIPS distance", fontsize=14)
    plt.xlabel("Generation", fontsize=14)
    plt.title(f"Distance between samples\n{explabel}", fontsize=14)
    saveallforms(expdir, "dist_curve_LPIPS")
    plt.show()


#%%
plt.figure()
dist_df.plot(y=["dist_BG_m", "dist_FC6_m", "dist_crs_m"], kind="line")
plt.show()
#%%
# shaded errorbar
plt.figure()
sns.lineplot(data=dist_df, y="dist_BG_m", x=dist_df.index, ci="sd", )
dist_df.plot(y=["dist_BG_m", "dist_FC6_m", "dist_crs_m"],
             kind="line", capsize=3)
plt.show()
#%%
plt.figure()
plt.imshow(dist_img_m_FC6.detach().numpy())
plt.colorbar()
plt.axis("image")
plt.show()
#%%


#%%
plt.figure()
plt.imshow(distmats.mean([0,1,2]).detach().numpy())
plt.colorbar()
plt.axis("image")
plt.show()
#%%
show_imgrid(imgtsr_blk0, nrow=8, figsize=(10,10))
show_imgrid(imgtsr_blk1, nrow=8, figsize=(10,10))
#%%
from os.path import join
outdir = r"D:\Github\CrossAttention-Similarity-Map\sample_data"
save_imgrid(imgtsr_blk0, join(outdir, "imgs_FC6.png"), nrow=8, figsize=(10,10))
save_imgrid(imgtsr_blk1, join(outdir, "imgs_BigGAN.png"), nrow=8, figsize=(10,10))
#%%
import timm
from timm import create_model
vit_model = create_model("vit_base_patch16_224", pretrained=True)
vit_model.eval()
#%%
img_rsz0 = F.interpolate(imgtsr_blk0, (224, 224), mode="bilinear", align_corners=True)
#%%
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
train_nodes, _ = get_graph_node_names(vit_model)
#%%
vit_model.cuda().requires_grad_(False)
feature_extractor = create_feature_extractor(vit_model, return_nodes={"blocks": "out"})
#%%
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

hugfc_preprocessing = ViTFeatureExtractor.from_pretrained('facebook/dino-vitb8')
model = ViTModel.from_pretrained('facebook/dino-vitb8')
#%%
model.cuda().requires_grad_(False)
#%%
inputs = hugfc_preprocessing(images=list(imgtsr_blk1), return_tensors="pt")["pixel_values"]
with torch.no_grad():
    outputs = model(inputs.cuda()[:10], output_attentions=True, output_hidden_states=False)
last_hidden_states = outputs.last_hidden_state
#%%
attn_weights = torch.stack([attnmat.cpu() for attnmat in outputs.attentions])
del outputs
#%%
head_avg_attn = attn_weights.mean([2])

# head_avg_attn, _ = attn_weights.max(dim=2)
#%%
identity_mat = torch.eye(head_avg_attn.shape[-1]).repeat(head_avg_attn.shape[1], 1, 1)
rollback_mat = identity_mat.clone()  # torch.eye(head_avg_attn.shape[-1]).repeat(head_avg_attn.shape[1], 1, 1)
flow_mat     = identity_mat.clone()  # torch.eye(head_avg_attn.shape[-1]).repeat(head_avg_attn.shape[1], 1, 1)
for layeri in range(12):
    rollback_mat  = torch.bmm((head_avg_attn[layeri] + identity_mat), rollback_mat, )
    flow_mat  = torch.bmm((head_avg_attn[layeri]), flow_mat, )
#%%
cls_rollback_attn_mats = rollback_mat[:, 0, 1:]
cls_rollback_attn_mats = cls_rollback_attn_mats.reshape(-1, 28, 28)
cls_flow_attn_mats = flow_mat[:, 0, 1:]
cls_flow_attn_mats = cls_flow_attn_mats.reshape(-1, 28, 28)
#%%
cls_rollback_attn_img = F.interpolate(cls_rollback_attn_mats[:, None], (224, 224), mode="bilinear", align_corners=True)
#%%
# normalize min, max to 0,1 for visualization
val_min, _ = cls_rollback_attn_img.flatten(1).min(-1)
val_max, _ = cls_rollback_attn_img.flatten(1).max(-1)
cls_rollback_attn_img_norm = (cls_rollback_attn_img - val_min[:,None,None,None]) / \
                             (val_max[:,None,None,None] - val_min[:,None,None,None])

#%%
# cls_rollback_attn_img = (cls_rollback_attn_img - cls_rollback_attn_img.min(dim=(1,2,3), keepdim=True)) / \
#                         (cls_rollback_attn_img.max(dim=(1,2,3), keepdim=True) - cls_rollback_attn_img.min(dim=(1,2,3), keepdim=True))
#%%
show_imgrid(cls_rollback_attn_img_norm, nrow=5, figsize=(10, 10))
#%%
show_imgrid(cls_rollback_attn_img_norm, nrow=4, figsize=(10, 10))
#%%
F.interpolate(cls_rollback_attn_mats, (224, 224), mode="bilinear", align_corners=True).shape
#%%
torch.cuda.empty_cache()
#%%
img_rsz0 = F.interpolate(imgtsr_blk0, (224, 224), mode="bilinear", align_corners=True)
img_rsz1 = F.interpolate(imgtsr_blk1, (224, 224), mode="bilinear", align_corners=True)
with torch.no_grad():
    img_feat0 = feature_extractor(img_rsz0.cuda())["out"].cpu()
    img_feat1 = feature_extractor(img_rsz1.cuda())["out"].cpu()
#%%
with torch.no_grad():
    img_feat0 = vit_model.forward_features(img_rsz0.cuda()).cpu()
    img_feat1 = vit_model.forward_features(img_rsz1.cuda()).cpu()
#%% first token is the cls token
img_feat0_cls = img_feat0[:, :1, :]
img_feat0_patch = img_feat0[:, 1:, :]
img_feat1_cls = img_feat1[:, :1, :]
img_feat1_patch = img_feat1[:, 1:, :]
#%%
mean_feat0_cls = img_feat0_cls.mean([0, ])
mean_feat0_patch = img_feat0_patch.mean([0, ])
mean_feat1_cls = img_feat1_cls.mean([0, ])
mean_feat1_patch = img_feat1_patch.mean([0, ])
#%%
clspatch_dotprod00 = (mean_feat0_patch @ mean_feat0_cls.T).reshape(14, 14)
clspatch_dotprod01 = (mean_feat1_patch @ mean_feat0_cls.T).reshape(14, 14)
clspatch_dotprod10 = (mean_feat0_patch @ mean_feat1_cls.T).reshape(14, 14)
clspatch_dotprod11 = (mean_feat1_patch @ mean_feat1_cls.T).reshape(14, 14)
#%%
figh, axs = plt.subplots(2,2)
axs[0, 0].imshow(clspatch_dotprod00.detach().numpy())
axs[0, 1].imshow(clspatch_dotprod01.detach().numpy())
axs[1, 0].imshow(clspatch_dotprod10.detach().numpy())
axs[1, 1].imshow(clspatch_dotprod11.detach().numpy())
plt.show()

#%%.shae
from timm.models.vision_transformer import VisionTransformer

