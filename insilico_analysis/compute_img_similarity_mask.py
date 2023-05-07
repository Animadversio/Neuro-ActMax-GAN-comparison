
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
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
from torchvision.transforms import Normalize

normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def compare_imgs_cnn_feat_simmask(img1, img2, fetcher, featmsk1=None, featmsk2=None, featkey='blocks', metric='cosine'):
    if featmsk1 is not None and featmsk2 is None:
        featmsk2 = featmsk1
    elif featmsk1 is None and featmsk2 is not None:
        featmsk1 = featmsk2
    if isinstance(img1, list):
        img1 = np.stack(img1, axis=0)
        img2 = np.stack(img2, axis=0)
    if isinstance(img1, np.ndarray):
        if len(img1.shape) == 3:
            img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0)
            img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)
        elif len(img1.shape) == 4:
            img1 = torch.from_numpy(img1).permute(0, 3, 1, 2)
            img2 = torch.from_numpy(img2).permute(0, 3, 1, 2)
    # TODO: add normalization????
    img1 = normalize(img1)
    img2 = normalize(img2)
    with torch.no_grad():
        feat1 = fetcher(img1.cuda())[featkey]
        feat2 = fetcher(img2.cuda())[featkey]

    if featmsk1 is not None:
        feat1vec = (feat1 * featmsk1[None, None]).mean(dim=(2, 3), keepdim=False)
    else:
        feat1vec = feat1.mean(dim=(2, 3), keepdim=False)
    if featmsk2 is not None:
        feat2vec = (feat2 * featmsk2[None, None]).mean(dim=(2, 3), keepdim=False)
    else:
        feat2vec = feat2.mean(dim=(2, 3), keepdim=False)

    if metric == 'cosine':
        # cossim_sclr = torch.cosine_similarity(feat1vec, feat2vec, dim=1).cpu()
        cossim_sclr = pairwise_cosine_similarity(feat1vec, feat2vec, ).cpu()  # UPDATED FOR BATCH COMPARING
    elif metric == 'mse':
        # cossim_sclr = F.mse_loss(feat1vec, feat2vec, reduction='none').mean(dim=1).cpu()
        cossim_sclr = ((feat1vec[:,None] - feat2vec[None,:]) ** 2).mean(dim=-1).cpu()  # UPDATED FOR BATCH COMPARING
    else:
        raise ValueError(f"metric {metric} not recognized")
    return cossim_sclr
#%%
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
cnn, _ = load_featnet("resnet50_linf8")

cnn_feat = create_feature_extractor(cnn, ["layer2", "layer3", "layer4"])
#%%
example_mtg = r"F:\insilico_exps\GAN_Evol_cmp\protoimgs_merge\resnet50_linf8_.layer4.Bottleneck2_43_4_4_RFrsz_optim_pool.jpg"
example_mtg = r"F:\insilico_exps\GAN_Evol_cmp\protoimgs_merge\resnet50_linf8_.layer4.Bottleneck2_43_4_4_optim_pool.jpg"
example_mtg = r"F:\insilico_exps\GAN_Evol_cmp\protoimgs_merge\tf_efficientnet_b6_ap_.blocks.6_34_4_4_optim_pool.jpg"
crops = crop_all_from_montage(plt.imread(example_mtg), totalnum=30, imgsize=256, pad=2)
#%%
outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Protoimg_examples"
for i, crop in enumerate(crops):
    plt.imsave(os.path.join(outdir, f"crop_{i}.jpg"), crop)
#%%
ilist = [4, 20,23,24]
plt.imsave(os.path.join(outdir, f"crop_{str(ilist)}.jpg"), np.concatenate([crops[i] for i in ilist], axis=1))
#%%
import torch
imgtsr = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float() / 255.0
imgtsr_pp = normalize(imgtsr)
#%%
cnn_feat.cuda().eval()
with torch.no_grad():
    feats = cnn_feat(imgtsr_pp.cuda())
#%%
BG_ft_avg = feats['layer4'][:10].mean(dim=(0, 2, 3), keepdim=False)
FC_ft_avg = feats['layer4'][20:].mean(dim=(0, 2, 3), keepdim=False)
#%%
from einops import rearrange
feattsr = feats['layer4'][:10]
B, C, H, W = feattsr.shape
featmat = rearrange(feattsr, 'b c h w -> (b h w) c')
cosmap_tsr = pairwise_cosine_similarity(featmat, FC_ft_avg[None])
MSEmap_tsr = ((featmat[:, None] - FC_ft_avg[None]) ** 2).mean(dim=-1)
cosmaps = rearrange(cosmap_tsr, '(b h w) () -> b () h w', b=B, h=H, w=W)
MSEmaps = rearrange(MSEmap_tsr, '(b h w) () -> b () h w', b=B, h=H, w=W)
#%%
feattsr1 = feats['layer4'][:10]
feattsr2 = feats['layer4'][20:]
B, C, H, W = feattsr1.shape
featmat1 = rearrange(feattsr1, 'b c h w -> (b h w) c')
B, C, H, W = feattsr2.shape
featmat2 = rearrange(feattsr2, 'b c h w -> (b h w) c')
cosmap_tsr2 = torch.cosine_similarity(featmat1, featmat2, dim=-1)
MSEmap_tsr2 = ((featmat1 - featmat2) ** 2).mean(dim=-1)
cosmaps2 = rearrange(cosmap_tsr2, '(b h w) -> b () h w', b=B, h=H, w=W)
MSEmaps2 = rearrange(MSEmap_tsr2, '(b h w) -> b () h w', b=B, h=H, w=W)
#%%
from core.utils.plot_utils import show_imgrid
show_imgrid(cosmaps, cmap='jet', nrow=10)
show_imgrid(MSEmaps, cmap='jet', nrow=10)
#%%
plt.imshow(make_grid(cosmaps.cpu(), nrow=10, )[0], cmap='jet') # normalize=True
plt.colorbar()
plt.show()
#%%
plt.imshow(make_grid(cosmaps2.cpu(), nrow=10, )[0], cmap='jet') # normalize=True
plt.colorbar()
plt.show()
#%%
plt.imshow(make_grid(MSEmaps.cpu(), nrow=10, )[0], cmap='jet') # normalize=True
plt.colorbar()
plt.show()
#%%
plt.imshow(make_grid(MSEmaps2.cpu(), nrow=10, )[0], cmap='jet') # normalize=True
plt.colorbar()
plt.show()
#%%
show_imgrid(cosmaps2[:,:,1:-1,1:-1], cmap='jet', nrow=10)

