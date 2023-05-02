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
#%%
unitdir = r"F:\insilico_exps\GAN_Evol_cmp\resnet50_linf8_.layer2.Bottleneck3_9_14_14"
mtg1 = "besteachgenHessCMA500_fc6_40771.jpg"
mtg2 = "besteachgenHessCMA_18193.jpg"
npz1 = mtg1.replace(".jpg", ".npz").replace("besteachgen", "scores")
npz2 = mtg2.replace(".jpg", ".npz").replace("besteachgen", "scores")
#%%
unitdir = r"F:\insilico_exps\GAN_Evol_cmp\resnet50_linf8_.layer4.Bottleneck2_19_4_4"
mtg1 = "besteachgenHessCMA500_fc6_09065.jpg"
mtg2 = "besteachgenHessCMA_55157.jpg"
npz1 = mtg1.replace(".jpg", ".npz").replace("besteachgen", "scores")
npz2 = mtg2.replace(".jpg", ".npz").replace("besteachgen", "scores")
#%%
unitdir = r"F:\insilico_exps\GAN_Evol_cmp\resnet50_linf8_.layer4.Bottleneck2_5_4_4"
mtg1 = "besteachgenHessCMA500_fc6_03762.jpg"
mtg2 = "besteachgenHessCMA_47912.jpg"
npz1 = mtg1.replace(".jpg", ".npz").replace("besteachgen", "scores")
npz2 = mtg2.replace(".jpg", ".npz").replace("besteachgen", "scores")
#%%
unitdir = r"F:\insilico_exps\GAN_Evol_cmp\resnet50_linf8_.layer3.Bottleneck5_20_7_7"
mtg1 = r"besteachgenHessCMA500_fc6_65372.jpg"
mtg2 = r"besteachgenCholCMA_43041.jpg"
npz1 = mtg1.replace(".jpg", ".npz").replace("besteachgen", "scores")
npz2 = mtg2.replace(".jpg", ".npz").replace("besteachgen", "scores")
#%%
generations1 = np.load(join(unitdir, npz1))['generations']
scores1 = np.load(join(unitdir, npz1))['scores_all']
imgs1 = crop_all_from_montage(plt.imread(join(unitdir, mtg1)), 100, imgsize=256, pad=2,
                             autostop=False)

generations2 = np.load(join(unitdir, npz2))['generations']
scores2 = np.load(join(unitdir, npz2))['scores_all']
imgs2 = crop_all_from_montage(plt.imread(join(unitdir, mtg2)), 100, imgsize=256, pad=2,
                             autostop=False)
scores_maxpergen1 = np.array([np.max(scores1[generations1 == gen]) for gen
                              in np.unique(generations1)])
scores_maxpergen2 = np.array([np.max(scores2[generations2 == gen]) for gen
                              in np.unique(generations2)])

evol_dataset1 = ImageResp_Dataset(imgs1, scores_maxpergen1,
              transform=Compose([ToTensor(), Normalize([0.5], [0.5])]))
evol_dataset2 = ImageResp_Dataset(imgs2, scores_maxpergen2,
                transform=Compose([ToTensor(), Normalize([0.5], [0.5])]))
evol_dataloader1 = DataLoader(evol_dataset1, batch_size=25, shuffle=True)
evol_dataloader2 = DataLoader(evol_dataset2, batch_size=25, shuffle=True)
#%%
next(iter(evol_dataloader1))
next(iter(evol_dataloader2))
#%%
cnn = vgg16(pretrained=True)
cnn.eval().cuda()
# get_graph_node_names(cnn)
layernames = ['features.15', 'features.22', 'features.29']
extractor_pretrain = create_feature_extractor(cnn, return_nodes=layernames)
#%%
imgtsr1, score1 = evol_dataset1[-1]
imgtsr2, score2 = evol_dataset2[-1]
#%%
with torch.no_grad():
    outs1 = extractor_pretrain(imgtsr1.unsqueeze(0).cuda())
    outs2 = extractor_pretrain(imgtsr2.unsqueeze(0).cuda())
#%%
# outs1["features.15"]
layer2plot = "features.29"
plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.imshow(outs1[layer2plot][0].norm(dim=0).cpu())
plt.subplot(222)
plt.imshow(outs2[layer2plot][0].norm(dim=0).cpu())
plt.subplot(223)
plt.imshow(imgtsr1.permute(1,2,0)/2+0.5)
plt.subplot(224)
plt.imshow(imgtsr2.permute(1,2,0)/2+0.5)
plt.show()
#%%
from tqdm import trange, tqdm
cnn_pred = vgg16(pretrained=True)
cnn_pred.classifier = nn.Sequential(nn.Linear(25088, 1))
cnn_pred.cuda()
#%%
# train on the first dataset
optimizer = torch.optim.Adam(cnn_pred.parameters(), lr=5e-6)
criterion = nn.MSELoss()
#%%
for epoch in trange(40):
    train_loss = 0
    for imgtsr, score in evol_dataloader1:
        optimizer.zero_grad()
        # with torch.no_grad():
        #     outs = extractor(imgtsr.cuda())
        # pred = cnn_pred(outs[layer2plot])
        pred = cnn_pred(imgtsr.cuda())
        loss = criterion(pred, score.cuda()[:, None])
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgtsr.size(0)
        # raise Exception
        # print(loss.item())

    train_loss = train_loss / len(evol_dataset1)
    print("Epoch: {} Loss: {}".format(epoch, train_loss))
#%%
layernames = ['features.15', 'features.22', 'features.29']
# train on the second dataset
extractor_trained = create_feature_extractor(cnn_pred, return_nodes=layernames)
extractor_trained.eval().cuda()
#%%
imgtsr1, score1 = evol_dataset1[-2]
imgtsr2, score2 = evol_dataset2[-2]
with torch.no_grad():
    outs1 = extractor_pretrain(imgtsr1.unsqueeze(0).cuda())
    outs2 = extractor_pretrain(imgtsr2.unsqueeze(0).cuda())
#%%
layer2plot = "features.29"
plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.imshow(outs1[layer2plot][0].norm(dim=0).cpu())
plt.subplot(222)
plt.imshow(outs2[layer2plot][0].norm(dim=0).cpu())
plt.subplot(223)
plt.imshow(imgtsr1.permute(1,2,0)/2+0.5)
plt.subplot(224)
plt.imshow(imgtsr2.permute(1,2,0)/2+0.5)
plt.show()
#%%
@torch.no_grad()
def get_features_from_imgs(imgs, extractor):
    outs = extractor(imgs.cuda())
    return outs