from pathlib import Path
import glob
import os
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import trange, tqdm
from core.utils.montage_utils import crop_from_montage, crop_all_from_montage
from torchvision.models import resnet50, vgg16
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.transforms import Normalize, Compose, ToTensor
from core.utils.plot_utils import saveallforms
from core.utils.CNN_scorers import load_featnet


def compute_covs(scores, feattsr, EPS=1e-6):
    scores = scores - scores.mean()
    featmean = feattsr.mean(dim=0)
    feattsr = feattsr - featmean[None, ...]
    featstd = feattsr.std(dim=0)
    covtsr = torch.einsum("b,bchw->chw", scores, feattsr)
    corrtsr = covtsr / (scores.std() * featstd + EPS)
    return covtsr, corrtsr, featmean, featstd
#%%
RGBmean = torch.tensor([0.485, 0.456, 0.406])
RGBstd = torch.tensor([0.229, 0.224, 0.225])
normalizer = Normalize(RGBmean, RGBstd)
denormalizer = Normalize(-RGBmean / RGBstd, 1 / RGBstd)
# cnn = resnet50(pretrained=True)
#%%
cnn, _ = load_featnet("resnet50_linf8")
cnn.eval().cuda()
netname2 = "resnet50_linf8"
layernames2 = ["layer1", "layer2", "layer3", "layer4"] # 'features.3',
extractor_pretrain2 = create_feature_extractor(cnn, return_nodes=layernames2)
#%%
unitdir = r"F:\insilico_exps\GAN_Evol_Dissection\resnet50_.layer3.Bottleneck5_5_7_7"
unitdir_path = Path(unitdir)
(unitdir_path / "covtsrs").mkdir(exist_ok=True)
optimnames = ["HessCMA500_fc6", "CholCMA_fc6", "CholCMA", "HessCMA"]
optimname = "CholCMA_fc6"
optimname = "CholCMA"
# besteachgenCholCMA_fc6_58209.jpg use glob pattern to match the five digits
for optimname in tqdm(optimnames):
    bestfns = list(unitdir_path.glob(f"besteachgen{optimname}_"+"[0-9]"*5+".jpg"))
    #%%
    bestfn = bestfns[0]
    for bestfn in tqdm(bestfns):
        RND = int(bestfn.name[-9:-4])
        scorefn = unitdir_path / f"scores{optimname}_{RND:05d}.npz"
        score_dict = np.load(scorefn, allow_pickle=True)
        scores = score_dict['scores_all']
        generations = score_dict['generations']
        score_tsr = torch.tensor(scores).float()
        bestmtg = plt.imread(bestfn)
        bestimg_all = crop_all_from_montage(bestmtg, totalnum=100, imgsize=256, pad=2, autostop=False)
        bestimg = bestimg_all[-1]
        bestimg_avg = np.mean(np.stack(bestimg_all[-10:]).astype(float) / 255.0, axis=0)
        #%%
        imgfns = sorted(list(unitdir_path.glob(f"imggen{optimname}_{RND:05d}_block"+"[0-9]"*2+".jpg")))
        assert len(imgfns) == 100
        img_all = []
        for blocki in trange(100): # 8 sec
            imgfn = imgfns[blocki]
            imgn_in_block = (generations == blocki).sum()
            imgmtg = plt.imread(imgfn)
            if blocki == 0:
                imgs = [imgmtg]
            else:
                imgs = crop_all_from_montage(imgmtg, totalnum=imgn_in_block, imgsize=256, pad=2, autostop=True)
            img_all.extend(imgs)
        #%%
        assert len(img_all) == len(score_tsr)
        #%%
        batchsize = 128
        with torch.no_grad():  # 20 sec for 3K images
            out_feat = {}
            for i in trange(0, len(img_all), batchsize):
                imgtsrs = normalizer(torch.stack([ToTensor()(img) for img in img_all[i:i+batchsize]]))
                out_rn = extractor_pretrain2(imgtsrs.cuda())
                for k, v in out_rn.items():
                    if k not in out_feat:
                        out_feat[k] = []
                    out_feat[k].append(v.cpu())

            for k, v in out_feat.items():
                out_feat[k] = torch.cat(v, dim=0)
        del img_all
        #%%
        covtsrs = {}
        corrtsrs = {}
        featmeans = {}
        featstds = {}
        for layer, feat in tqdm(out_feat.items()):  # 1mins for 4 layers
            covtsr, corrtsr, featmean, featstd = compute_covs(score_tsr, feat)
            covtsrs[layer] = covtsr
            corrtsrs[layer] = corrtsr
            featmeans[layer] = featmean
            featstds[layer] = featstd

        torch.save(dict(covtsrs=covtsrs, corrtsrs=corrtsrs,
                        featmeans=featmeans, featstds=featstds,
                        score_tsr=score_tsr, generations=generations, ),
                   unitdir_path / "covtsrs" / f"covtsrs_{optimname}_{RND:05d}.pt", )
        #%%
        figh, axs = plt.subplots(2, 4, figsize=(16, 8))
        axs[0, 0].imshow(bestimg)
        axs[0, 0].set_title("best img")
        axs[0, 1].imshow(bestimg_avg)
        axs[0, 1].set_title("best img pix avg")
        axs[0, 2].scatter(generations, scores, alpha=0.4)
        axs[0, 2].set_title("scores")
        axs[0, 3].axis("off")
        for i, (layer, covtsr) in enumerate(covtsrs.items()):
            axs[1, i].imshow(covtsr.norm(dim=0)**2)
            axs[1, i].set_title(layer)

        figh.suptitle(f"{unitdir_path.name} {optimname} {RND:05d}", fontsize=16)
        saveallforms(str(unitdir_path / "covtsrs"), f"covtsrs_{optimname}_{RND:05d}_vis", figh)
        plt.show()
        raise Exception
#%%

#%%
