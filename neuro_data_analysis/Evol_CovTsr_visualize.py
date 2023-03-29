import os
import numpy as np
from os.path import join
from CorrFeatTsr_visualize_lib import tsr_posneg_factorize, visualize_cctsr_simple, \
    vis_feattsr, vis_feattsr_factor, vis_featvec, vis_featvec_wmaps, pad_factor_prod, rectify_tsr
from core.utils.CNN_scorers import load_featnet
from core.utils.GAN_utils import upconvGAN
from neuro_data_analysis.neural_data_lib import get_expstr, load_neural_data
import torch
_, BFEStats = load_neural_data()
#%%
# G = None
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
#%%
# use aggbackend to avoid plotting
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pylab as plt
# switch to the interactive backend for pycharm
#%%
cov_root = r"E:\Network_Data_Sync\corrFeatTsr_BigGAN"
Animal = "Both"
Expi = 111
for Expi in  [66]:#range(160, 191):
    # thread =  "_cmb"
    try:
        explabel = get_expstr(BFEStats, Expi)
    except:
        continue
    showfig = True
    for thread in [0, 1, "_cmb"]:
        try:
            corrDict = np.load(join(cov_root, "Both_Exp%02d_Evol_thr%s_res-robust_corrTsr.npz" \
                                    % (Expi, thread)), allow_pickle=True)
        except:
            continue
        cctsr_dict = corrDict.get("cctsr").item()
        Ttsr_dict = corrDict.get("Ttsr").item()
        stdtsr_dict = corrDict.get("featStd").item()
        covtsr_dict = {layer: cctsr_dict[layer] * stdtsr_dict[layer] for layer in cctsr_dict}
        #%%
        figroot = r"E:\Network_Data_Sync\BigGAN_FeatAttribution"
        layer = "layer3"
        # show_img(ReprStats[Expi - 1].Manif.BestImg)
        figdir = join(figroot, "%s_Exp%02d_thr%s" % (Animal, Expi, thread))
        os.makedirs(figdir, exist_ok=True)
        Ttsr = Ttsr_dict[layer]
        cctsr = cctsr_dict[layer]
        covtsr = covtsr_dict[layer]
        Ttsr = np.nan_to_num(Ttsr)
        cctsr = np.nan_to_num(cctsr)
        covtsr = np.nan_to_num(covtsr)
        #%%
        netname = "resnet50_linf8";layer = "layer3";bdr = 1;exp_suffix = "_nobdr_res-robust"
        NF = 3
        init = "nndsvda"; solver="cd"; l1_ratio=0; alpha=0; beta_loss="frobenius" # default
        # init="nndsvd"; solver="mu"; l1_ratio=0.8; alpha=0.005; beta_loss="kullback-leibler"#"frobenius"##
        # rect_mode = "pos"; thresh = (None, None)
        rect_mode = "Tthresh"; thresh = (None, 3)
        batchsize = 41
        # Record hyper parameters in name string
        rectstr = rect_mode
        featnet, net = load_featnet(netname)
        #%%
        # imgsize = EStats[Expi - 1].evol.imgsize
        # imgpos = EStats[Expi - 1].evol.imgpos
        # pref_chan = EStats[Expi - 1].evol.pref_chan
        # area = area_mapping(pref_chan)
        # imgpix = int(imgsize * 40)
        # explabel = "%s Exp%02d Driver Chan %d, %.1f deg [%s]\nCCtsr %s-%s sfx:%s bdr%d rect %s Fact %d" % (
        #                             Animal, Expi, pref_chan, imgsize, tuple(imgpos),
        #                             netname, layer, exp_suffix, bdr, rect_mode, NF)

        #%%
        # Indirect factorize
        # Ttsr_pp = rectify_tsr(Ttsr, "pos")  # mode="thresh", thr=(-5, 5))  #  #
        # Hmat, Hmaps, Tcomponents, ccfactor, Stat = tsr_factorize(Ttsr_pp, covtsr, bdr=bdr, Nfactor=NF,
        #                                 figdir=figdir, savestr="%s-%scov" % (netname, layer))
        # Direct factorize
        Hmat, Hmaps, ccfactor, FactStat = tsr_posneg_factorize(rectify_tsr(covtsr, rect_mode, thresh, Ttsr=Ttsr),
                                                               bdr=bdr, Nfactor=NF, init=init, solver=solver, l1_ratio=l1_ratio, alpha=alpha, beta_loss=beta_loss,
                                                               figdir=figdir, savestr="%s-%scov" % (netname, layer), suptit=explabel, show=showfig,)
        DR_Wtsr = pad_factor_prod(Hmaps, ccfactor, bdr=bdr)
        #%%
        save_cfg = dict(Bsize=5, figdir=figdir, savestr="", imshow=False)
        # Visualize the preferred features optimizing in the GAN space
        GAN_optim_cfg = dict(lr=0.05, MAXSTEP=150, use_adam=True, langevin_eps=0,)
        finimgs, mtg, score_traj = vis_feattsr(covtsr, net, G, layer, netname=netname, featnet=featnet,
                                               **save_cfg, **GAN_optim_cfg)
        finimgs, mtg, score_traj = vis_feattsr_factor(ccfactor, Hmaps, net, G, layer, netname=netname, featnet=featnet,
                                              bdr=bdr, **save_cfg, **GAN_optim_cfg)
        finimgs_col, mtg_col, score_traj_col = vis_featvec(ccfactor, net, G, layer, netname=netname, featnet=featnet,
                                               **save_cfg, **GAN_optim_cfg)
        finimgs_col, mtg_col, score_traj_col = vis_featvec_wmaps(ccfactor, Hmaps, net, G, layer, netname=netname, featnet=featnet,
                                             bdr=bdr, **save_cfg, **GAN_optim_cfg)

        # Visualize the preferred features optimizing in the pixel space
        pix_optim_cfg = dict(lr=0.02, MAXSTEP=150, use_adam=False, langevin_eps=0.00)
        finimgs, mtg, score_traj = vis_feattsr(covtsr, net, None, layer, netname=netname, featnet=featnet,
                                               **pix_optim_cfg, **save_cfg)
        finimgs, mtg, score_traj = vis_feattsr_factor(ccfactor, Hmaps, net, None, layer, netname=netname, featnet=featnet,
                                          bdr=bdr, **pix_optim_cfg, **save_cfg)
        finimgs_col, mtg_col, score_traj_col = vis_featvec(ccfactor, net, None, layer, netname=netname, featnet=featnet,
                                               **pix_optim_cfg, **save_cfg)
        finimgs_col, mtg_col, score_traj_col = vis_featvec_wmaps(ccfactor, Hmaps, net, None, layer, netname=netname, featnet=featnet,
                                         bdr=bdr, **pix_optim_cfg, **save_cfg)
    torch.cuda.empty_cache()
