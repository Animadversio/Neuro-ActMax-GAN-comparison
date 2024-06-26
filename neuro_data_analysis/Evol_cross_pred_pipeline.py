import torch
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from core.utils.dataset_utils import ImagePathDataset, normalizer, denormalizer
from torchvision import transforms, utils
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs_multiwindow, load_neural_data # load_img_resp_pairs_multiwindow
import matplotlib.pyplot as plt
from os.path import join
# from CorrFeatTsr_lib import Corr_Feat_Machine, visualize_cctsr, loadimg_preprocess
# from core.utils.layer_hook_utils import featureFetcher_module
# from core.utils.CNN_scorers import TorchScorer, load_featnet
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
#%%
import platform
if platform.system() == "Windows":
    stim_rootdir = "S:"
    device = "cuda"
elif platform.system() == "Darwin":
    stim_rootdir = "/Users/binxuwang/Network_mapping"
    device = "mps"

# load the neural data
BFEStats_merge, BFEStats = load_neural_data()
#%%
# Define the response time windows
rsp_wdws = [range(50, 200), range(0, 50), range(50, 100), range(100, 150), range(150, 200)]
# rsp_wdws += [range(strt, strt+25) for strt in range(0, 200, 25)]
# get image sequence and response in different time windows
# image paths: (n_images, )
# response: (n_images, n_time)
Expi = 155
Expi = 165
imgfps0, resp_mat0, gen_vec0 = load_img_resp_pairs_multiwindow(BFEStats, Expi,
                         "Evol", thread=0, stimdrive=stim_rootdir,
                                                               output_fmt="vec", rsp_wdws=rsp_wdws)
imgfps1, resp_mat1, gen_vec1 = load_img_resp_pairs_multiwindow(BFEStats, Expi,
                         "Evol", thread=1, stimdrive=stim_rootdir,
                                                               output_fmt="vec", rsp_wdws=rsp_wdws)
#%%
# use default transform for the image, including RGB norm and resize
evol_ds0 = ImagePathDataset(imgfps0, resp_mat0, transform=None, img_dim=(224, 224))
evol_dl0 = DataLoader(evol_ds0, batch_size=10, shuffle=False, num_workers=0)
evol_ds1 = ImagePathDataset(imgfps1, resp_mat1, transform=None, img_dim=(224, 224))
evol_dl1 = DataLoader(evol_ds1, batch_size=10, shuffle=False, num_workers=0)
#%%
plt.imshow(denormalizer(evol_ds0[-4][0]). \
           permute(1, 2, 0))
plt.show()
#%%
import timm
# timms = timm.list_models(pretrained=True)
# featnet = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
#%%
featnet = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

#%%
def extract_feat(dataset, featnet, device="mps", batch_size=10):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    featnet.eval().to(device)
    feat_col = []
    for imgtsr, score in tqdm(dl):
        with torch.no_grad():
            feat = featnet(imgtsr.to(device))
        feat_col.append(feat.cpu())
    feattsr = torch.cat(feat_col, dim=0)
    return feattsr
# extract features from image sequence using some encoder / PC space
# features: (n_images, n_features)
feattsr0 = extract_feat(evol_ds0, featnet, device=device, batch_size=15)
feattsr1 = extract_feat(evol_ds1, featnet, device=device, batch_size=15)
#%%
# loop over all the window pairs
# build prediction / regression model for window 1; test on window 2
from sklearn.linear_model import RidgeCV, LassoCV, MultiTaskLassoCV
from sklearn.model_selection import cross_val_score
# split the data into train and test
from sklearn.model_selection import train_test_split
def R2score_per_bin(model, feattsr, resp_mat):
    predmat = model.predict(feattsr)
    residue_ss = np.mean((predmat - resp_mat) ** 2, axis=0)
    total_ss = np.mean((resp_mat.mean(axis=0) - resp_mat) ** 2, axis=0)
    R2_vec = 1 - residue_ss / total_ss
    return R2_vec


def corrscore_per_bin(model, feattsr, resp_mat):
    predmat = model.predict(feattsr)
    corr_vec = np.corrcoef(predmat.T, resp_mat.T)[np.arange(predmat.shape[1]),
                np.arange(predmat.shape[1], predmat.shape[1] + resp_mat.shape[1])]
    return corr_vec
#%%
(feattsr0_train, feattsr0_test,
 resp_mat0_train, resp_mat0_test,
 gen_vec0_train, gen_vec0_test) = train_test_split(
    feattsr0, resp_mat0, gen_vec0,
    test_size=0.2, random_state=41, shuffle=True,)
(feattsr1_train, feattsr1_test,
    resp_mat1_train, resp_mat1_test,
    gen_vec1_train, gen_vec1_test) = train_test_split(
        feattsr1, resp_mat1, gen_vec1,
    test_size=0.2, random_state=41, shuffle=True,)
#%%
regmodel0 = RidgeCV(alphas=np.logspace(-3, 3, 10), )
regmodel0.fit(feattsr0_train, resp_mat0_train)
print("DeePSim train (R2)", R2score_per_bin(regmodel0, feattsr0_train, resp_mat0_train))
print("DeePSim test (R2)", R2score_per_bin(regmodel0, feattsr0_test, resp_mat0_test))
print("BigGAN train (R2)", R2score_per_bin(regmodel0, feattsr1_train, resp_mat1_train))
print("BigGAN test (R2)", R2score_per_bin(regmodel0, feattsr1_test, resp_mat1_test))
print("DeePSim train (corr)", corrscore_per_bin(regmodel0, feattsr0_train, resp_mat0_train))
print("DeePSim test (corr)", corrscore_per_bin(regmodel0, feattsr0_test, resp_mat0_test))
print("BigGAN train (corr)", corrscore_per_bin(regmodel0, feattsr1_train, resp_mat1_train))
print("BigGAN test (corr)", corrscore_per_bin(regmodel0, feattsr1_test, resp_mat1_test))

#%%
regmodel0 = MultiTaskLassoCV(alphas=np.logspace(-3, 3, 10), )
regmodel0.fit(feattsr0_train, resp_mat0_train)
print("DeePSim train (R2)", R2score_per_bin(regmodel0, feattsr0_train, resp_mat0_train))
print("DeePSim test (R2)", R2score_per_bin(regmodel0, feattsr0_test, resp_mat0_test))
print("BigGAN train (R2)", R2score_per_bin(regmodel0, feattsr1_train, resp_mat1_train))
print("BigGAN test (R2)", R2score_per_bin(regmodel0, feattsr1_test, resp_mat1_test))
#%%
print("DeePSim train (corr)", corrscore_per_bin(regmodel0, feattsr0_train, resp_mat0_train))
print("DeePSim test (corr)", corrscore_per_bin(regmodel0, feattsr0_test, resp_mat0_test))
print("BigGAN train (corr)", corrscore_per_bin(regmodel0, feattsr1_train, resp_mat1_train))
print("BigGAN test (corr)", corrscore_per_bin(regmodel0, feattsr1_test, resp_mat1_test))

#%%
regmodel1 = RidgeCV(alphas=np.logspace(-3, 3, 10), )
regmodel1.fit(feattsr1_train, resp_mat1_train)
print(regmodel1.score(feattsr0_train, resp_mat0_train))
print(regmodel1.score(feattsr0_test, resp_mat0_test))
print(regmodel1.score(feattsr1_train, resp_mat1_train))
print(regmodel1.score(feattsr1_test, resp_mat1_test))
#%%
regmodel1 = MultiTaskLassoCV(alphas=np.logspace(-3, 3, 10), )
regmodel1.fit(feattsr1_train, resp_mat1_train)
print("DeePSim train (R2)", R2score_per_bin(regmodel1, feattsr0_train, resp_mat0_train))
print("DeePSim test (R2)", R2score_per_bin(regmodel1, feattsr0_test, resp_mat0_test))
print("BigGAN train (R2)", R2score_per_bin(regmodel1, feattsr1_train, resp_mat1_train))
print("BigGAN test (R2)", R2score_per_bin(regmodel1, feattsr1_test, resp_mat1_test))
#%%
print("DeePSim train (corr)", corrscore_per_bin(regmodel1, feattsr0_train, resp_mat0_train))
print("DeePSim test (corr)", corrscore_per_bin(regmodel1, feattsr0_test, resp_mat0_test))
print("BigGAN train (corr)", corrscore_per_bin(regmodel1, feattsr1_train, resp_mat1_train))
print("BigGAN test (corr)", corrscore_per_bin(regmodel1, feattsr1_test, resp_mat1_test))


#%%
regmodel01 = RidgeCV(alphas=np.logspace(-3, 3, 10), )
regmodel01.fit(torch.cat([feattsr0_train, feattsr1_train], dim=0),
                np.concatenate([resp_mat0_train, resp_mat1_train], axis=0))
print(regmodel01.score(feattsr0_train, resp_mat0_train))
print(regmodel01.score(feattsr0_test, resp_mat0_test))
print(regmodel01.score(feattsr1_train, resp_mat1_train))
print(regmodel01.score(feattsr1_test, resp_mat1_test))
#%%
regmodel01 = MultiTaskLassoCV(alphas=np.logspace(-3, 3, 10), )
regmodel01.fit(torch.cat([feattsr0_train, feattsr1_train], dim=0),
                np.concatenate([resp_mat0_train, resp_mat1_train], axis=0))
print(regmodel01.score(feattsr0_train, resp_mat0_train))
print(regmodel01.score(feattsr0_test, resp_mat0_test))
print(regmodel01.score(feattsr1_train, resp_mat1_train))
print(regmodel01.score(feattsr1_test, resp_mat1_test))
#%%
regmodel01.predict(feattsr0_test)
#%%
resp_mat0_test
#%%
#visualize the response mat
gen_sortidx = gen_vec0_test.argsort()
plt.figure(figsize=[5, 10.4])
plt.subplot(121)
plt.imshow(regmodel01.predict(feattsr0_test)[gen_sortidx,], aspect="auto")
plt.subplot(122)
plt.imshow(resp_mat0_test[gen_sortidx,], aspect="auto")
plt.show()
#%%
#visualize the response mat
gen_sortidx = gen_vec0_train.argsort()
plt.figure(figsize=[5, 10.4])
plt.subplot(121)
plt.imshow(regmodel01.predict(feattsr0_train)[gen_sortidx,], aspect="auto")
plt.subplot(122)
plt.imshow(resp_mat0_train[gen_sortidx,], aspect="auto")
plt.show()
#%%


print(R2score_per_bin(regmodel01, feattsr0_train, resp_mat0_train))
print(R2score_per_bin(regmodel01, feattsr0_test, resp_mat0_test))
print(R2score_per_bin(regmodel01, feattsr1_train, resp_mat1_train))
print(R2score_per_bin(regmodel01, feattsr1_test, resp_mat1_test))

#%%
regmodel0.predict(feattsr0_test).shape
#%%
np.corrcoef(regmodel01.coef_)
#%%
# get prediction tensor (n_images, n_time, n_time)

# get cross prediction accuracy matrix train (n_time, n_time)

# get cross prediction accuracy matrix test (n_time, n_time)

# cross prediction based on DeePSim -> BigGAN

# cross prediction based on BigGAN -> DeePSim

# cross prediction based on Both -> DeePSim / BigGAN

#%%
# plot the covariance across time
covmat0 = np.cov(resp_mat0.T)
corrmat0 = np.corrcoef(resp_mat0.T)
covmat1 = np.cov(resp_mat1.T)
corrmat1 = np.corrcoef(resp_mat1.T)
#%%
import seaborn as sns
plt.figure(figsize=[10, 5.4])
plt.subplot(121)
sns.heatmap(corrmat0, annot=True, fmt=".2f")
plt.axis("image")
plt.title("Correlation matrix for thread 1")
plt.subplot(122)
sns.heatmap(corrmat1, annot=True, fmt=".2f")
plt.axis("image")
plt.title("Correlation matrix for thread 2")
plt.show()

