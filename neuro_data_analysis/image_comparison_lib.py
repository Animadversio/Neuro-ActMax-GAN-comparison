import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

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
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)
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
        cossim_sclr = torch.cosine_similarity(feat1vec, feat2vec, dim=1).cpu()
    elif metric == 'mse':
        cossim_sclr = F.mse_loss(feat1vec, feat2vec, reduction='none').mean(dim=1).cpu()
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
