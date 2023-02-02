# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 18:58:22 2021

@author: Binxu Wang

Code to build model of neurons by analyzing their image and score trajectory.
Depending on the `Visual_Neuron_Modelling` repository.
"""
# %load_ext autoreload
# %autoreload 2
# %%
# backup_dir = r"C:\Users\Ponce lab\Documents\ml2a-monk\generate_BigGAN\2021-07-23-12-23-21"
# backup_dir = r"C:\Users\Poncelab-ML2a\Documents\monkeylogic2\generate_integrated\2021-10-25-11-05-37"
backup_dir = r"C:\Users\Ponce lab\Documents\ml2a-monk\generate_BigGAN\2022-01-14-12-55-52"
# r"C:\Users\Ponce lab\Documents\ml2a-monk\generate_BigGAN\2021-06-28-12-34-03"
# r"C:\Users\Ponce lab\Documents\ml2a-monk\generate_BigGAN\2021-06-04-11-54-42"
# backup_dir = r"N:\Stimuli\2021-EvolDecomp\2021-04-27-Alfa-03\2021-04-27-13-07-55"
threadid = 2
threadlabel = "_threadBigGAN"  # "_threadBigGAN"  ## "_threadBigGAN"
Animal = "Beto"
exptime = backup_dir.split("\\")[-1]
# %%
from time import time
import os
import re
from os.path import join
import sys

if os.environ['COMPUTERNAME'] == 'PONCELAB-ML2B':
    Python_dir = r"C:\Users\Ponce lab\Documents\Python"
elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2A':
    Python_dir = r"C:\Users\Poncelab-ML2a\Documents\Python"
elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':
    Python_dir = r"E:\Github_Projects"
elif os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':
    Python_dir = r"D:\Github"
elif os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':
    Python_dir = r"D:\Github"
elif os.environ['COMPUTERNAME'] == 'PONCELAB-OFFICE':
    Python_dir = r"G:\My Drive\Python"

sys.path.append(join(Python_dir, "Visual_Neuro_InSilico_Exp"))
sys.path.append(join(Python_dir, "Visual_Neuron_Modelling"))
# sys.path.append(join(Python_dir,"PerceptualSimilarity"))
import numpy as np
import torch
from scipy.stats import sem
from scipy.io import loadmat
from skimage.io import imsave, imread
from torchvision.utils import make_grid
from core.utils.GAN_utils import upconvGAN
from core.utils.montage_utils import build_montages
import matplotlib.pylab as plt
# from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample


def visualize_cctsr_simple(featFetcher, layers2plot, imgcol, savestr="Evol", titstr="Alfa_Evol", figdir=""):
    """
    Demo
    ExpType = "EM_cmb"
    layers2plot = ['conv3_3', 'conv4_3', 'conv5_3']
    figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, ExpType, )
    figh.savefig(join("S:\corrFeatTsr","VGGsummary","%s_Exp%d_%s_corrTsr_vis.png"%(Animal,Expi,ExpType)))
    """
    nlayer = max(4, len(layers2plot))
    figh, axs = plt.subplots(3, nlayer, figsize=[10 / 3 * nlayer, 8])
    if imgcol is not None:
        for imgi in range(len(imgcol)):
            axs[0, imgi].imshow(imgcol[imgi])
            axs[0, imgi].set_title("Highest Score Evol Img")
            axs[0, imgi].axis("off")
    for li, layer in enumerate(layers2plot):
        chanN = featFetcher.cctsr[layer].shape[0]
        tmp = axs[1, li].matshow(np.nansum(featFetcher.cctsr[layer].abs().numpy(), axis=0) / chanN)
        plt.colorbar(tmp, ax=axs[1, li])
        axs[1, li].set_title(layer + " mean abs cc")
        tmp = axs[2, li].matshow(np.nanmax(featFetcher.cctsr[layer].abs().numpy(), axis=0))
        plt.colorbar(tmp, ax=axs[2, li])
        axs[2, li].set_title(layer + " max abs cc")
    figh.suptitle("%s Exp Corr Feat Tensor" % (titstr))
    plt.show()
    figh.savefig(join(figdir, "%s_corrTsr_vis.png" % (savestr)))
    figh.savefig(join(figdir, "%s_corrTsr_vis.pdf" % (savestr)))
    return figh


def ind2xy(ind, div_n, pH, pW):
    yi, xi = np.divmod(ind, div_n)
    return yi * pH, xi * pW


def roll_image(img, yroll, xroll):
    imggxshift = np.zeros(img.shape, img.dtype)
    imggyshift = np.zeros(img.shape, img.dtype)
    # assert xroll * yroll is not 0
    if xroll is not 0:
        imggxshift[xroll:, :] = img[:-xroll, :]
        imggxshift[:xroll, :] = img[-xroll:, :]  # roll the lower edge up to fill the blank!
    else:
        imggxshift = img.copy()
    if yroll is not 0:
        imggyshift[:, yroll:] = imggxshift[:, :-yroll]  # same for x axis.
        imggyshift[:, :yroll] = imggxshift[:, -yroll:]
    else:
        imggyshift = imggxshift
    return imggyshift


def patch_shuffle(img, div_n=8):
    """div_n, how many patch do you want along each axis"""
    # div_n = 16
    H, W = img.shape
    # assert (H%div_n is 0) and (W%div_n is 0), "`div_n` should divide both W and H of image, like 1,2,4,8,16"
    patch_n = div_n * div_n
    pH = int(H / div_n)
    pW = int(W / div_n)
    perm_p_id = np.random.permutation(patch_n)
    imgpshf = np.zeros(img.shape, img.dtype)
    imgpshf[:, :] = img[:, :]
    for ind in range(patch_n):
        targ_y, targ_x = ind2xy(ind, div_n, pH, pW)
        src_y, src_x = ind2xy(perm_p_id[ind], div_n, pH, pW)
        imgpshf[targ_y:targ_y + pH, targ_x:targ_x + pW] = img[src_y:src_y + pH, src_x:src_x + pW]

    return imgpshf


# %% Load basic information
data = loadmat(join(backup_dir, "Evol_ScoreImgTraj.mat"))
imgfp_col = data.get("imgfp_col")
score_col = data.get("score_col")
imgsize = data.get('imgsize').astype('float')
imgpos = data.get('imgpos').astype('float')
pref_chan = data.get('prefchan').astype('int')
imgsize = imgsize[threadid - 1, 0]
imgpos = imgpos[threadid - 1, :]
pref_chan = pref_chan[threadid - 1, 0]
scorevec_thread = score_col[0, threadid - 1][:, 0]
imgfp_thread = imgfp_col[0, threadid - 1]
imgpatt = re.compile("block(\d*)_thread")
blockvec_thread = np.array([int(imgpatt.findall(imgfn)[0]) for imgfn in imgfp_thread])
blockarr = range(min(blockvec_thread), max(blockvec_thread) + 1)
meanarr = np.array([np.mean(scorevec_thread[blockvec_thread == blocki]) for blocki in blockarr])
semarr = np.array([sem(scorevec_thread[blockvec_thread == blocki]) for blocki in blockarr])
# %
if os.environ['COMPUTERNAME'] in ['DESKTOP-9DDE2RH', 'PONCELAB-OFFICE']:
    backup_dir_old, _ = os.path.split(imgfp_thread[0])
    imgfp_thread = np.array([fp.replace(backup_dir_old, backup_dir) for fp in imgfp_thread])

figh = plt.figure(figsize=[6, 5]);
plt.scatter(blockvec_thread, scorevec_thread, alpha=0.5)
plt.plot(blockarr, meanarr, 'k-')
plt.fill_between(blockarr, meanarr - semarr, meanarr + semarr, alpha=0.4)
plt.ylabel("Spike rate");
plt.xlabel("Generations");
plt.title("Evolution Trajectory prefchan %02d, %.1f deg pos [%.1f %.1f], thread %d" % \
          (pref_chan, imgsize, imgpos[0], imgpos[1], threadid))
plt.show()
# %% Collect some best images
score_idx = np.argsort(-scorevec_thread)
score_examp = scorevec_thread[score_idx[:4]]
imgfp_examp = imgfp_thread[score_idx[:4]]
imgcol_examp = [imread(fp) for fp in imgfp_examp]
# %%
from torchvision import models
from CorrFeatTsr_lib import Corr_Feat_Machine, Corr_Feat_pipeline, loadimg_preprocess, visualize_cctsr
from CorrFeatTsr_predict_lib import score_images, softplus, fitnl_predscore
from featvis_lib import rectify_tsr, tsr_factorize, vis_featmap_corr, vis_feattsr, \
    vis_feattsr_factor, vis_featvec, vis_featvec_wmaps, vis_featvec_point, load_featnet, \
    tsr_posneg_factorize, posneg_sep

# %%
# netname = "alexnet";layers2plot = ["conv2", "conv3", "conv4", "conv5",]
# netname = "vgg16";layers2plot = ["conv2_2", "conv3_3", "conv4_3",  "conv5_3", ]
# netname = "resnet50";layers2plot = ["layer2", "layer3", "layer4", ]
netname = "resnet50_linf8";
layers2plot = ["layer2", "layer3", "layer4", ]
ccdir = join(backup_dir, "CCFactor_%s%s" % (netname, threadlabel))
# ccdir = "debug_tmp_%s"%netname
os.makedirs(join(ccdir, "img"), exist_ok=True)
figh.savefig(join(ccdir, "ExpEvolTraj.png"))
featnet, net = load_featnet(netname)
G = upconvGAN("fc6")
G.requires_grad_(False).cuda().eval();
# %% Create correlation online
imgpix = int(imgsize * 40)  # %224  #
# titstr = "Driver Chan %d, %.1f deg [%s]"%(pref_chan, imgsize, tuple(imgpos))
featFetcher = Corr_Feat_Machine()
featFetcher.register_hooks(net, layers2plot, netname=netname, )
featFetcher.init_corr()
Corr_Feat_pipeline(featnet, featFetcher, scorevec_thread, imgfp_thread,
                   lambda x: loadimg_preprocess(x, borderblur=True, imgpix=imgpix), online_compute=True,
                   batchsize=100, savedir=ccdir, savenm="Evol")  # % (Animal, Expi, expsuffix),
corrDict = np.load(join(ccdir, "%s_corrTsr.npz" % ("Evol")), allow_pickle=True)
figh = visualize_cctsr_simple(featFetcher, layers2plot, imgcol_examp,
                              savestr="%s_Evol%s_%s" % (Animal, exptime, netname),
                              titstr="%s_Evol%s_%s" % (Animal, exptime, netname), figdir=ccdir)

cctsr_dict = corrDict.get("cctsr").item()
Ttsr_dict = corrDict.get("Ttsr").item()
stdtsr_dict = corrDict.get("featStd").item()
featFetcher.clear_hook()
# %% OK starts decompostion.
# layer = "conv4"; bdr = 1;
# layer = "conv3_3"; bdr = 2;
layer = "layer3";
bdr = 1;
vis_score_mode = "cosine"  # "corr"

ccdir = join(backup_dir, "CCFactor_%s-%s%s" % (netname, layer, threadlabel))
os.makedirs(join(ccdir, "img"), exist_ok=True)
NF = 3;
rect_mode = "Tthresh";
thresh = (None, 3)  # "pos"
Ttsr = Ttsr_dict[layer]
cctsr = cctsr_dict[layer]
stdtsr = stdtsr_dict[layer]
covtsr = cctsr * stdtsr
covtsr = np.nan_to_num(covtsr)
cctsr = np.nan_to_num(cctsr)
# Ttsr_pp = rectify_tsr(Ttsr, rect_mode)  #
covtsr_pp = rectify_tsr(covtsr, mode=rect_mode, thr=thresh, Ttsr=Ttsr)  # add thresholding to T tsr
# Hmat, Hmaps, Tcomponents, ccfactor = tsr_factorize(Ttsr_pp, cctsr, bdr=bdr, Nfactor=NF, figdir=ccdir, savestr="%s-%s"%(netname, layer))
Hmat, Hmaps, ccfactor, FactStat = tsr_posneg_factorize(covtsr_pp, bdr=bdr, Nfactor=NF,
                                                       figdir=ccdir, savestr="%s-%s" % (netname, layer))
Tcomponents = None
# %%
torchseed = int(time())
torch.manual_seed(torchseed)
print("Visualizing featuer vectors with differeent spatial masks (using %s scoring)" % vis_score_mode)
finimgs, mtg, score_traj = vis_feattsr(cctsr, net, G, layer, netname=netname, score_mode=vis_score_mode,
                                       featnet=featnet, Bsize=5, figdir=ccdir, savestr="corr", saveimg=True)
# finimgs, mtg, score_traj = vis_feattsr(covtsr_pp, net, G, layer, netname=netname, score_mode=vis_score_mode,
#             featnet=featnet, Bsize=5, figdir=ccdir, savestr="cov_pp", saveimg=True)
# finimgs, mtg, score_traj = vis_feattsr(covtsr, net, G, layer, netname=netname, score_mode=vis_score_mode,
#             featnet=featnet, Bsize=5, figdir=ccdir, savestr="cov", saveimg=True)
finimgs, mtg, score_traj = vis_feattsr_factor(ccfactor, Hmaps, net, G, layer, netname=netname,
                                              score_mode=vis_score_mode,
                                              featnet=featnet, Bsize=5, bdr=bdr, figdir=ccdir, savestr="corr",
                                              saveimg=True)
finimgs_col, mtg_col, score_traj_col = vis_featvec_wmaps(ccfactor, Hmaps, net, G, layer, netname=netname,
                                                         score_mode=vis_score_mode, \
                                                         featnet=featnet, bdr=bdr, Bsize=10, saveImgN=5, figdir=ccdir,
                                                         savestr="corr", imshow=False, saveimg=True)
finimgs_col, mtg_col, score_traj_col = vis_featvec(ccfactor, net, G, layer, netname=netname, score_mode=vis_score_mode,
                                                   featnet=featnet, Bsize=10, saveImgN=5, figdir=ccdir, savestr="corr",
                                                   imshow=False, saveimg=True)
finimgs_col, mtg_col, score_traj_col = vis_featvec_point(ccfactor, Hmaps, net, G, layer, netname=netname,
                                                         score_mode=vis_score_mode, \
                                                         featnet=featnet, bdr=bdr, Bsize=10, saveImgN=5, figdir=ccdir,
                                                         savestr="corr", imshow=False, saveimg=True, pntsize=4)
# %%
print("Saving the Evolved images with highest scores")
score_examp = scorevec_thread[score_idx[:5]]
imgfp_examp = imgfp_thread[score_idx[:5]]
imgcol_examp = [imread(fp) for fp in imgfp_examp]
for i, img in enumerate(imgcol_examp):
    imgid = imgfp_examp[i].split("\\")[-1].split(".")[0]
    imsave(join(ccdir, "img", "evol_best_%02d_%s.png" % (i, imgid)), img)
# %%
print("Saving record for the factorization method")
np.savez(join(ccdir, "factor_record.npz"), Hmat=Hmat, Hmaps=Hmaps, Tcomponents=Tcomponents, ccfactor=ccfactor,
         netname=netname, layer=layer, bdr=bdr, NF=NF, rect_mode=rect_mode, torchseed=torchseed,
         vis_score_mode=vis_score_mode)
# %% Scramble the feature vectors
print("Shuffling feature vectors")
ccfactor_shfl = np.concatenate(tuple([ccfactor[np.random.permutation(ccfactor.shape[0]), ci:ci + 1]
                                      for ci in range(ccfactor.shape[1])]), axis=1)
# %%
print("Visualizing models with shuffled feature vectors. (using %s scoring)" % vis_score_mode)
finimgs_col, mtg_col, score_traj_col = vis_featvec(ccfactor_shfl, net, G, layer, netname=netname,
                                                   score_mode=vis_score_mode,
                                                   featnet=featnet, Bsize=10, saveImgN=5, figdir=ccdir,
                                                   savestr="shuffle", imshow=False, saveimg=True)
finimgs_col, mtg_col, score_traj_col = vis_featvec_point(ccfactor_shfl, Hmaps, net, G, layer, netname=netname,
                                                         score_mode=vis_score_mode, \
                                                         featnet=featnet, bdr=bdr, Bsize=10, saveImgN=5, figdir=ccdir,
                                                         savestr="shuffle", imshow=False, saveimg=True)
finimgs_col, mtg_col, score_traj_col = vis_featvec_wmaps(ccfactor_shfl, Hmaps, net, G, layer, netname=netname,
                                                         score_mode=vis_score_mode, \
                                                         featnet=featnet, bdr=bdr, Bsize=10, saveImgN=5, figdir=ccdir,
                                                         savestr="shuffle", imshow=False, saveimg=True)

# %% Patch shuffling + rolling to scramble the spatial masks for factors.
print("Shuffling spatial masks")
PatchN = 6
H_H, H_W = Hmaps.shape[0], Hmaps.shape[1]
Hmaps_patchshffule = np.concatenate(tuple(roll_image(patch_shuffle(Hmaps[:, :, ci], div_n=PatchN), \
                                                     np.random.randint(H_H), np.random.randint(H_W))[:, :, np.newaxis]
                                          for ci in range(Hmaps.shape[2])), axis=2)
# %%
print("Visualizing models with shuffled spatial masks (using %s scoring)" % vis_score_mode)
finimgs_col, mtg_col, score_traj_col = vis_featvec_wmaps(ccfactor, Hmaps_patchshffule, net, G, layer, netname=netname,
                                                         score_mode=vis_score_mode, \
                                                         featnet=featnet, bdr=bdr, Bsize=10, saveImgN=5, figdir=ccdir,
                                                         savestr="maponly_patchshuffle", imshow=False, saveimg=True)
finimgs_col, mtg_col, score_traj_col = vis_featvec_wmaps(ccfactor_shfl, Hmaps_patchshffule, net, G, layer,
                                                         netname=netname, score_mode=vis_score_mode, \
                                                         featnet=featnet, bdr=bdr, Bsize=10, saveImgN=5, figdir=ccdir,
                                                         savestr="map_patchshuffle", imshow=False, saveimg=True)
finimgs_col, mtg_col, score_traj_col = vis_feattsr_factor(ccfactor, Hmaps_patchshffule, net, G, layer, netname=netname,
                                                          score_mode=vis_score_mode, \
                                                          featnet=featnet, bdr=bdr, Bsize=10, saveImgN=5, figdir=ccdir,
                                                          savestr="maponly_patchshuffle", imshow=False, saveimg=True)
finimgs_col, mtg_col, score_traj_col = vis_feattsr_factor(ccfactor_shfl, Hmaps_patchshffule, net, G, layer,
                                                          netname=netname, score_mode=vis_score_mode, \
                                                          featnet=featnet, bdr=bdr, Bsize=10, saveImgN=5, figdir=ccdir,
                                                          savestr="map_patchshuffle", imshow=False, saveimg=True)
# %% Save the relevant information for mask shuffling, for reproducibility.
np.savez(join(ccdir, "factor_record_shuffle.npz"), Hmat=Hmat, Hmaps=Hmaps, Tcomponents=Tcomponents,
         ccfactor_shfl=ccfactor_shfl,
         Hmaps_patchshfl=Hmaps_patchshffule, netname=netname, layer=layer, bdr=bdr, NF=NF, rect_mode=rect_mode,
         thresh=thresh, torchseed=torchseed, vis_score_mode=vis_score_mode)
# %%
# Hmats_shfl = np.concatenate(tuple([Hmat[np.random.permutation(Hmat.shape[0]),ci:ci+1]
#                                       for ci in range(Hmat.shape[1])]),axis=1)
# Hmaps_shfl = Hmats_shfl.reshape(Hmaps.shape)
# finimgs_col, mtg_col, score_traj_col = vis_featvec_wmaps(ccfactor_shfl, Hmaps_shfl, net, G, layer, netname=netname, score_mode=vis_score_mode,\
#                      featnet=featnet, bdr=bdr, Bsize=10, figdir=ccdir, savestr="map_shuffle", imshow=False, saveimg=True)
os.system('cd "' + join(ccdir, 'img') + '" & start .')