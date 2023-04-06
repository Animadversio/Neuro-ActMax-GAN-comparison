"""
Compute image similarity based on the LPIPS or some neural network features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance

normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def compare_imgs_LPIPS(img1, img2, Dist):
    # compare two images using LPIPS
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)
    Dist.spatial = True
    with torch.no_grad():
        distmaps = Dist(img1.cuda(), img2.cuda()).cpu()
    distval = distmaps.mean(dim=(1, 2, 3))
    return distval.item(), distmaps


# compare two images using ViT features
def compare_imgs_vit(img1, img2, fetcher, featkey='blocks', metric='cosine'):
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)
    # TODO: add normalization????
    img1 = normalize(img1)
    img2 = normalize(img2)
    with torch.no_grad():
        feat1 = fetcher(img1.cuda())[featkey]
        feat2 = fetcher(img2.cuda())[featkey]

    tokenN = feat1.shape[1]
    mapsize = int(math.sqrt(tokenN - 1))

    if metric == 'cosine':
        cossim_vec = torch.cosine_similarity(feat1, feat2, dim=-1).cpu()
        cossim_cls = cossim_vec[0, 0].item()
        cossim_map = cossim_vec[0, 1:].reshape(mapsize, mapsize).numpy()
    elif metric == 'mse':
        cossim_vec = F.mse_loss(feat1, feat2, reduction='none').mean(dim=-1).cpu()
        cossim_cls = cossim_vec[0, 0].item()
        cossim_map = cossim_vec[0, 1:].reshape(mapsize, mapsize).numpy()
    else:
        raise ValueError(f"metric {metric} not recognized")
    return cossim_cls, cossim_map


def compare_imgs_cnn(img1, img2, fetcher, featkey='blocks', metric='cosine'):
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)
    # TODO: add normalization????
    img1 = normalize(img1)
    img2 = normalize(img2)
    with torch.no_grad():
        feat1 = fetcher(img1.cuda())[featkey]
        feat2 = fetcher(img2.cuda())[featkey]

    if metric == 'cosine':
        cossim_tsr = torch.cosine_similarity(feat1, feat2, dim=1).cpu()
    elif metric == 'mse':
        cossim_tsr = F.mse_loss(feat1, feat2, reduction='none').mean(dim=1).cpu()
    else:
        raise ValueError(f"metric {metric} not recognized")
    return cossim_tsr


def compare_imgs_cnn_featmsk(img1, img2, fetcher, featmsk1=None, featmsk2=None, featkey='blocks', metric='cosine'):
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


def compare_imgs_vit_featmsk(img1, img2, fetcher, featmsk1=None, featmsk2=None, featkey='blocks', metric='cosine'):
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)
    img1 = normalize(img1)
    img2 = normalize(img2)
    with torch.no_grad():
        feat1 = fetcher(img1.cuda())[featkey]
        feat2 = fetcher(img2.cuda())[featkey]

    tokenN = feat1.shape[1]
    mapsize = int(math.sqrt(tokenN - 1))
    feat1 = feat1.reshape(feat1.shape[0], feat1.shape[1], -1)
    feat2 = feat2.reshape(feat2.shape[0], feat2.shape[1], -1)
    raise NotImplementedError
    # if featmsk1 is not None:
    #     feat1vec = (feat1 * featmsk1[None, None]).mean(dim=2, keepdim=False)
    # else:
    #     feat1vec = feat1.mean(dim=2, keepdim=False)
    # if featmsk2 is not None:
    #     feat2vec = (feat2 * featmsk2[None, None]).mean(dim=2, keepdim=False)
    # else:
    #     feat2vec = feat2.mean(dim=2, keepdim=False)
    # if metric == 'cosine':


def naive_featmsk():
    import numpy as np
    from scipy import ndimage
    naive_featmask_L4 = np.zeros((7, 7))
    naive_featmask_L4[1:-1, 1:-1] = 1  # 0.5
    naive_featmask_L4[2:-2, 2:-2] = 1
    naive_featmask_L3 = ndimage.zoom(naive_featmask_L4, 2, order=0)
    naive_featmask_L3 = torch.from_numpy(naive_featmask_L3).float().to("cuda")
    naive_featmask_L4 = torch.from_numpy(naive_featmask_L4).float().to("cuda")
    # plt.figure(figsize=(10, 5))
    # plt.subplot(121)
    # plt.imshow(naive_featmask_L4)
    # plt.subplot(122)
    # plt.imshow(naive_featmask_L3)
    # plt.show()
    return naive_featmask_L3, naive_featmask_L4

