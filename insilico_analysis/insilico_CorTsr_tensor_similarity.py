import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
import math
from core.utils.montage_utils import crop_all_from_montage, crop_from_montage
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from torchvision.models import resnet50, vgg16
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

#%%
class ImageResp_Dataset(Dataset):

    def __init__(self, imgs, scores, transform=None):
        self.imgs = imgs
        self.scores = scores.astype(np.float32)
        self.transform = transform
        assert len(imgs) == len(scores)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]  # np uint8 array
        score = self.scores[idx]
        if self.transform:
            img = self.transform(img)
        return img, score


import einops
def compute_covariance_matrix(feattsr, scores):
    # tensor: (N, C, H, W)
    # scores: (N, )
    # return: (C, C)
    N, C, H, W = feattsr.shape
    feattsr = einops.rearrange(feattsr, 'n c h w -> n (c h w)')
    feattsr = feattsr - feattsr.mean(dim=0, keepdim=True)
    scores = scores - scores.mean(dim=0, keepdim=True)
    cov = torch.mm(feattsr.T, scores[:,None]) / (feattsr.shape[0] - 1)
    return cov.view(C, H, W)


def feattsr_point_cosine(feattsr1, feattsr2):
    # feature tensors: (C, H, W)
    # return: (H, W)
    C, H, W = feattsr1.shape
    feattsr1 = einops.rearrange(feattsr1, 'c h w -> c (h w)')
    feattsr2 = einops.rearrange(feattsr2, 'c h w -> c (h w)')
    # feattsr1 = feattsr1 - feattsr1.mean(dim=0, keepdim=True)
    # feattsr2 = feattsr2 - feattsr2.mean(dim=0, keepdim=True)
    cosinemap = torch.cosine_similarity(feattsr1, feattsr2, dim=0)
    return cosinemap.view(H, W)

#%%
RGBmean = torch.tensor([0.485, 0.456, 0.406])
RGBstd = torch.tensor([0.229, 0.224, 0.225])
normalizer = Normalize(RGBmean, RGBstd)
denormalizer = Normalize(-RGBmean / RGBstd, 1 / RGBstd)
#%%
cnn = vgg16(pretrained=True)
cnn.eval().cuda()
netname1 = "vgg16"
layernames1 = ['features.8', 'features.15', 'features.22', 'features.29'] # 'features.3',
extractor_pretrain1 = create_feature_extractor(cnn, return_nodes=layernames1)
cnn = resnet50(pretrained=True)
cnn.eval().cuda()
netname2 = "resnet50"
layernames2 = ["layer1", "layer2", "layer3", "layer4"] # 'features.3',
extractor_pretrain2 = create_feature_extractor(cnn, return_nodes=layernames2)
# get_graph_node_names(cnn, )
del cnn
#%%
from pathlib import Path
unitdir = r"F:\insilico_exps\GAN_Evol_cmp\resnet50_linf8_.layer4.Bottleneck2_49_4_4"
unitdir = r"F:\insilico_exps\GAN_Evol_cmp\resnet50_linf8_.layer3.Bottleneck5_45_7_7"
unitdir = r"F:\insilico_exps\GAN_Evol_cmp\tf_efficientnet_b6_.blocks.6_15_4_4"
unitdir = r"F:\insilico_exps\GAN_Evol_cmp\resnet50_.layer4.Bottleneck2_15_4_4"
unitdir = r"F:\insilico_exps\GAN_Evol_cmp\resnet50_.layer3.Bottleneck5_44_7_7"
unitdir = r"F:\insilico_exps\GAN_Evol_cmp\resnet50_linf8_.layer2.Bottleneck3_24_14_14"
unitdir = r"F:\insilico_exps\GAN_Evol_cmp\resnet50_linf8_.layer2.Bottleneck3_12_14_14"
unitdir = r"F:\insilico_exps\GAN_Evol_cmp\resnet50_.layer3.Bottleneck5_14_7_7"
# glob with re pattern
mtg1_list = list(Path(unitdir).glob("besteachgenHessCMA500_fc6_[0-9][0-9][0-9][0-9][0-9].jpg"))
mtg2_list = list(Path(unitdir).glob("besteachgenCholCMA_[0-9][0-9][0-9][0-9][0-9].jpg"))
# random choice from the list
mtg1 = np.random.choice(mtg1_list).name
mtg2 = np.random.choice(mtg2_list).name
RND1 = int(mtg1[-9:-4])
RND2 = int(mtg2[-9:-4])
# mtg1 = mtg1_list[0].name
# mtg2 = mtg2_list[0].name
#%%
npz1 = mtg1.replace(".jpg", ".npz").replace("besteachgen", "scores")
npz2 = mtg2.replace(".jpg", ".npz").replace("besteachgen", "scores")

generations1 = np.load(join(unitdir, npz1))['generations']
scores1 = np.load(join(unitdir, npz1))['scores_all']
imgs1 = crop_all_from_montage(plt.imread(join(unitdir, mtg1)), 100,
                              imgsize=256, pad=2, autostop=False)
generations2 = np.load(join(unitdir, npz2))['generations']
scores2 = np.load(join(unitdir, npz2))['scores_all']
imgs2 = crop_all_from_montage(plt.imread(join(unitdir, mtg2)), 100,
                              imgsize=256, pad=2, autostop=False)
scores_maxpergen1 = np.array([np.max(scores1[generations1 == gen]) for gen
                              in np.unique(generations1)])
scores_maxpergen2 = np.array([np.max(scores2[generations2 == gen]) for gen
                              in np.unique(generations2)])

scores1 = torch.tensor(scores_maxpergen1).float()
scores2 = torch.tensor(scores_maxpergen2).float()
imgtsrs1 = torch.stack([ToTensor()(img) for img in imgs1])
imgtsrs2 = torch.stack([ToTensor()(img) for img in imgs2])
imgtsrs1 = normalizer(imgtsrs1)
imgtsrs2 = normalizer(imgtsrs2)
# evol_dataset1 = ImageResp_Dataset(imgs1, scores_maxpergen1,
#               transform=Compose([ToTensor(), Normalize([0.5], [0.5])]))
# evol_dataset2 = ImageResp_Dataset(imgs2, scores_maxpergen2,
#                 transform=Compose([ToTensor(), Normalize([0.5], [0.5])]))
# evol_dataloader1 = DataLoader(evol_dataset1, batch_size=100, shuffle=False)
# evol_dataloader2 = DataLoader(evol_dataset2, batch_size=100, shuffle=False)
#%%
with torch.no_grad():
    # for i, (img, score1) in enumerate(evol_dataloader1):
    out_vgg1 = extractor_pretrain1(imgtsrs1.cuda())
    out_vgg1 = {k: v.cpu() for k, v in out_vgg1.items()}
    # for i, (img, score2) in enumerate(evol_dataloader2):
    out_vgg2 = extractor_pretrain1(imgtsrs2.cuda())
    out_vgg2 = {k: v.cpu() for k, v in out_vgg2.items()}
    # transport the tensors to cpu
    out_rn1 = extractor_pretrain2(imgtsrs1.cuda())
    out_rn1 = {k: v.cpu() for k, v in out_rn1.items()}
    out_rn2 = extractor_pretrain2(imgtsrs2.cuda())
    out_rn2 = {k: v.cpu() for k, v in out_rn2.items()}
#%%
# compute covariance matrix between feature tensor and scores
figh, axs = plt.subplots(3, 4, figsize=[12, 8])
for i, layer in enumerate(layernames1):
    cov1 = compute_covariance_matrix(out_vgg1[layer], scores1)
    cov2 = compute_covariance_matrix(out_vgg2[layer], scores2)
    cosine_map = feattsr_point_cosine(cov1, cov2)
    im = axs[1, i].imshow(cosine_map.numpy())
    axs[1, i].set_title(netname1+" "+layer)
    plt.colorbar(im, ax=axs[1, i])

for i, layer in enumerate(layernames2):
    cov1 = compute_covariance_matrix(out_rn1[layer], scores1)
    cov2 = compute_covariance_matrix(out_rn2[layer], scores2)
    cosine_map = feattsr_point_cosine(cov1, cov2)
    im = axs[2, i].imshow(cosine_map.numpy())
    axs[2, i].set_title(netname2+" "+layer)
    plt.colorbar(im, ax=axs[2, i])

axs[0, 0].imshow(imgs1[-1])
axs[0, 1].imshow(imgs2[-1])
axs[0, 2].plot(scores1.numpy(), alpha=0.5, lw=1.5)
axs[0, 2].plot(scores2.numpy(), alpha=0.5, lw=1.5)
axs[0, 2].set_title("Score best per gen")
plt.suptitle(f"{Path(unitdir).name}")
plt.tight_layout()
saveallforms(outdir, f"{Path(unitdir).name}_cosine_maps_FC{RND1:05d}_BG{RND2:05d}")
plt.show()


#%% Example units
# unitdir = r"F:\insilico_exps\GAN_Evol_cmp\resnet50_linf8_.layer2.Bottleneck3_9_14_14"
# mtg1 = "besteachgenHessCMA500_fc6_40771.jpg"
# mtg2 = "besteachgenHessCMA_18193.jpg"
# npz1 = mtg1.replace(".jpg", ".npz").replace("besteachgen", "scores")
# npz2 = mtg2.replace(".jpg", ".npz").replace("besteachgen", "scores")
# #%%
# unitdir = r"F:\insilico_exps\GAN_Evol_cmp\resnet50_linf8_.layer4.Bottleneck2_19_4_4"
# mtg1 = "besteachgenHessCMA500_fc6_09065.jpg"
# mtg2 = "besteachgenHessCMA_55157.jpg"
# npz1 = mtg1.replace(".jpg", ".npz").replace("besteachgen", "scores")
# npz2 = mtg2.replace(".jpg", ".npz").replace("besteachgen", "scores")
# #%%
# unitdir = r"F:\insilico_exps\GAN_Evol_cmp\resnet50_linf8_.layer4.Bottleneck2_5_4_4"
# mtg1 = "besteachgenHessCMA500_fc6_03762.jpg"
# mtg2 = "besteachgenHessCMA_47912.jpg"
# npz1 = mtg1.replace(".jpg", ".npz").replace("besteachgen", "scores")
# npz2 = mtg2.replace(".jpg", ".npz").replace("besteachgen", "scores")
# #%%
# unitdir = r"F:\insilico_exps\GAN_Evol_cmp\resnet50_linf8_.layer3.Bottleneck5_20_7_7"
# mtg1 = r"besteachgenHessCMA500_fc6_65372.jpg"
# mtg2 = r"besteachgenCholCMA_43041.jpg"
# npz1 = mtg1.replace(".jpg", ".npz").replace("besteachgen", "scores")
# npz2 = mtg2.replace(".jpg", ".npz").replace("besteachgen", "scores")
# #%%
# unitdir = r"F:\insilico_exps\GAN_Evol_cmp\resnet50_linf8_.layer3.Bottleneck5_10_7_7"
# mtg1 = r"besteachgenHessCMA500_fc6_41743.jpg"
# mtg2 = r"besteachgenCholCMA_81749.jpg"
# npz1 = mtg1.replace(".jpg", ".npz").replace("besteachgen", "scores")
# npz2 = mtg2.replace(".jpg", ".npz").replace("besteachgen", "scores")