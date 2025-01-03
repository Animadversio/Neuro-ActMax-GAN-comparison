import torch
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from core.utils.dataset_utils import ImagePathDataset
from torchvision import transforms, utils
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs_multiwindow, load_neural_data # load_img_resp_pairs_multiwindow
import matplotlib.pyplot as plt
from os.path import join
from CorrFeatTsr_lib import Corr_Feat_Machine, visualize_cctsr, loadimg_preprocess
# from core.utils.layer_hook_utils import featureFetcher_module
from core.utils.CNN_scorers import TorchScorer, load_featnet
#%%
BFEStats_merge, BFEStats = load_neural_data()
#%%
def _get_ref_img(BFEStats, Expi,):
    S = BFEStats[Expi - 1]  # Expi follows matlab convention, starts from 1
    imglist = S['imageName']
    refimgfn_set = set([imgfnl[0] for imgfnl in imglist if imgfnl[0].endswith("_nat")])
    return refimgfn_set


def _get_comments(BFEStats, Expi):
    S = BFEStats[Expi - 1]  # Expi follows matlab convention, starts from 1
    return S["meta"]["comments"]


def get_expstr(BFEStats, Expi):
    S = BFEStats[Expi - 1]  # Expi follows matlab convention, starts from 1
    expstr = f"Exp {Expi:03d} {S['meta']['ephysFN']} Pref chan{int(S['evol']['pref_chan'][0])} U{int(S['evol']['unit_in_pref_chan'][0])}" \
             f"\nimage size {S['evol']['imgsize']} deg  pos {S['evol']['imgpos'][0]}" \
             f"\nEvol thr0: {S['evol']['space_names'][0][0]}" \
             f"   thr1: {S['evol']['space_names'][1][0]}"
    return expstr
#%%
figroot = r"E:\Network_Data_Sync\corrFeatTsr_BigGAN\fig_summary"
figdynam_root = r"E:\Network_Data_Sync\corrFeatTsr_BigGAN\fig_dynamic_summary"
saveroot = r"E:\Network_Data_Sync\corrFeatTsr_BigGAN"
#%%
model, _ = load_featnet("resnet50_linf8")
recmodule_dict = {"layer1": model.layer1,
                   "layer2": model.layer2,
                   "layer3": model.layer3,
                   "layer4": model.layer4}
#%%
rsp_wdws = [range(50, 200), range(0, 50), range(50, 100), range(100, 150), range(150, 200)]
rsp_wdws += [range(strt, strt+25) for strt in range(0, 200, 25)]
#%%
def visualize_cctsr_dynamics(featFetcher: Corr_Feat_Machine, rsp_wdws, layers2plot: list, ReprStats, Expi, Animal, ExpType, Titstr, figdir="",
                             show=False):
    """ Given a `Corr_Feat_Machine` show the tensors in the different layers of it.
    Example:
        ExpType = "EM_cmb"
        layers2plot = ['conv3_3', 'conv4_3', 'conv5_3']
        figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, ExpType, )
        figh.savefig(join("S:\corrFeatTsr","VGGsummary","%s_Exp%d_%s_corrTsr_vis.png"%(Animal,Expi,ExpType)))
    :param featFetcher:
    :param layers2plot:
    :param ReprStats:
    :param Expi:
    :param Animal:
    :param Titstr:
    :return:
    """
    nlayer = len(layers2plot)
    for Ti, rsp_wdw in enumerate(rsp_wdws):
        figh, axs = plt.subplots(3, nlayer, figsize=[10/3*nlayer,10])
        if ReprStats is not None:
            axs[0,0].imshow(ReprStats[Expi-1].Evol.BestImg)
            axs[0,0].set_title("Best Evol Img")
            axs[0,0].axis("off")
            axs[0,1].imshow(ReprStats[Expi-1].Evol.BestBlockAvgImg)
            axs[0,1].set_title("Best BlockAvg Img")
            axs[0,1].axis("off")
            axs[0,2].imshow(ReprStats[Expi-1].Manif.BestImg)
            axs[0,2].set_title("Best Manif Img")
            axs[0,2].axis("off")
        for li, layer in enumerate(layers2plot):
            chanN = featFetcher.cctsr[layer].shape[0]
            tmp=axs[1,li].matshow(np.nansum(featFetcher.cctsr[layer][Ti].abs().numpy(), axis=0) / chanN)
            plt.colorbar(tmp, ax=axs[1,li])
            axs[1,li].set_title(layer+" mean abs cc")
            tmp=axs[2,li].matshow(np.nanmax(featFetcher.cctsr[layer][Ti].abs().numpy(), axis=0))
            plt.colorbar(tmp, ax=axs[2,li])
            axs[2,li].set_title(layer+" max abs cc")
        figh.suptitle("%s Exp%d Corr Tensor %s T:[%d,%d] %s"%(Animal, Expi, ExpType, rsp_wdw[0], rsp_wdw[-1]+1, Titstr))
        figh.savefig(join(figdir, "%s_Exp%d_%s_T%d_%d-%d_corrTsr_vis.png" % (Animal, Expi, ExpType, Ti, rsp_wdw[0], rsp_wdw[-1]+1)))
        figh.savefig(join(figdir, "%s_Exp%d_%s_T%d_%d-%d_corrTsr_vis.pdf" % (Animal, Expi, ExpType, Ti, rsp_wdw[0], rsp_wdw[-1]+1)))
        if show:
            plt.show()
        else:
            plt.close(figh)
    return figh
#%%
import matplotlib
matplotlib.use('Agg')
#%%
plot_err_dict = {}
Animal = "Both"
for Expi in tqdm(range(110, 191)): # [66]: #:
    S = BFEStats[Expi - 1]
    if S["evol"] is None:
        continue
    for thread in range(2):
        try:
            imgfps, resp_mat, gen_vec = load_img_resp_pairs_multiwindow(BFEStats, Expi,
                         "Evol", thread=thread, stimdrive="S:", output_fmt="vec",
                         rsp_wdws=rsp_wdws)
        except Exception as e:
            print(f"Exp {Expi} thread {thread} failed to load, try Network version")
            try:
                imgfps, resp_mat, gen_vec = load_img_resp_pairs_multiwindow(BFEStats, Expi,
                         "Evol", thread=thread, stimdrive="N:", output_fmt="vec",
                         rsp_wdws=rsp_wdws)
            except Exception as e2:
                print(f"Exp {Expi} thread {thread} failed to load")
            plot_err_dict[(Expi, thread)] = e
            continue
        if len(imgfps) == 0:
            continue
        # create a dataloader for the image response pairs
        # use default transform for the image, including RGB norm and resize
        evol_ds = ImagePathDataset(imgfps, resp_mat, transform=None, img_dim=(224, 224))
        evol_dl = DataLoader(evol_ds, batch_size=60, shuffle=False, num_workers=0)

        fetcher = Corr_Feat_Machine()
        fetcher.register_module_hooks(recmodule_dict, verbose=False)
        fetcher.init_corr()
        for i, (imgtsr, resps) in tqdm(enumerate(evol_dl)):
            with torch.no_grad():
                model(imgtsr.cuda())
            fetcher.update_corr_multi(resps.float())

        fetcher.calc_corr_multi()
        fetcher.clear_hook()
        savedict = fetcher.make_savedict()

        np.savez(join(saveroot, f"{Animal}_Exp{Expi:02d}_Evol_thr{thread}_res-robust_corrTsr_dynamics.npz"),
                 **savedict)

        titstr = get_expstr(BFEStats, Expi)
        figh = visualize_cctsr_dynamics(fetcher, rsp_wdws, ["layer1", "layer2", "layer3", "layer4"], None, Expi,
                       "Both", f"BigGAN_Evol_thr{thread}_res-robust_dynamics", titstr, figdir=figdynam_root)
        plt.close(figh)
#%%

plot_err_dict_cmb = {}
Animal = "Both"
for Expi in tqdm(range(1, 191)): # [66]
    S = BFEStats[Expi - 1]
    if S["evol"] is None:
        continue
    try:
        imgfps1, resp_mat1, gen_vec1 = load_img_resp_pairs_multiwindow(BFEStats, Expi,
                    "Evol", thread=0, stimdrive="S:", output_fmt="vec", rsp_wdws=rsp_wdws)
        imgfps2, resp_mat2, gen_vec2 = load_img_resp_pairs_multiwindow(BFEStats, Expi,
                    "Evol", thread=1, stimdrive="S:", output_fmt="vec", rsp_wdws=rsp_wdws)
        # imgfps1, resp_vec1, bsl_vec1, gen_vec1 = load_img_resp_pairs(BFEStats, Expi,
        #              "Evol", thread=0, stimdrive="S:", output_fmt="vec")
        # imgfps2, resp_vec2, bsl_vec2, gen_vec2 = load_img_resp_pairs(BFEStats, Expi,
        #              "Evol", thread=1, stimdrive="S:", output_fmt="vec")
        imgfps = imgfps1 + imgfps2
        resp_mat = np.concatenate((resp_mat1, resp_mat2), axis=0)
        gen_vec = np.concatenate((gen_vec1, gen_vec2), axis=0)
    except Exception as e:
        print(f"Exp {Expi} failed to load")
        plot_err_dict_cmb[Expi] = e
        continue
    if len(imgfps) == 0:
        continue
    # create a dataloader for the image response pairs
    evol_ds = ImagePathDataset(imgfps, resp_mat, transform=None, img_dim=(224, 224))
    evol_dl = DataLoader(evol_ds, batch_size=60, shuffle=False, num_workers=0)

    fetcher = Corr_Feat_Machine()
    fetcher.register_module_hooks(recmodule_dict, verbose=False)
    fetcher.init_corr()
    for i, (imgtsr, resps) in tqdm(enumerate(evol_dl)):
        with torch.no_grad():
            model(imgtsr.cuda())
        fetcher.update_corr_multi(resps.float())

    fetcher.calc_corr_multi()
    fetcher.clear_hook()
    savedict = fetcher.make_savedict()

    np.savez(join(saveroot, f"{Animal}_Exp{Expi:02d}_Evol_thr{'_cmb'}_res-robust_corrTsr_dynamics.npz"),
             **savedict)

    titstr = get_expstr(BFEStats, Expi)
    figh = visualize_cctsr_dynamics(fetcher, rsp_wdws, ["layer1", "layer2", "layer3", "layer4"], None, Expi,
                       "Both", f"BigGAN_Evol_thr{'_cmb'}_res-robust_dynamics", titstr, figdir=figdynam_root)
    plt.close(figh)


#%%
Expi = 12
thread = 0
imgfps, resp_vec, bsl_vec, gen_vec = load_img_resp_pairs(BFEStats, Expi,
                 "Evol", thread=thread, stimdrive="S:", output_fmt="vec")
#%%
plt.plot(gen_vec, resp_vec, "o", alpha=0.5)
plt.plot(gen_vec, bsl_vec, "o", alpha=0.5)
plt.show()


#%% bug fixing
# fetcher.init_corr()
print(BFEStats[-9]["meta"]["stimuli"])
print(BFEStats[-8]["meta"]["stimuli"])

BFEStats[-9]["meta"]["stimuli"] = r"N:\Stimuli\2020-BigGAN\2021-05-28-Beto-01\2021-05-28-10-12-47"
BFEStats[-8]["meta"]["stimuli"] = r"N:\Stimuli\2020-BigGAN\2021-05-28-Beto-02\2021-05-28-10-28-01"
BFEStats[5 - 1]["meta"]["stimuli"] = r"N:\Stimuli\2020-BigGAN\2020-07-23-Beto-01\2020-07-23-15-59-38"
BFEStats[92 - 1]["meta"]["stimuli"] = r"N:\Stimuli\2020-BigGAN\2020-07-27-Alfa-01\2020-07-27-09-47-40"
BFEStats_merge["meta"][-9]["stimuli"] = r"N:\Stimuli\2020-BigGAN\2021-05-28-Beto-01\2021-05-28-10-12-47"
BFEStats_merge["meta"][-8]["stimuli"] = r"N:\Stimuli\2020-BigGAN\2021-05-28-Beto-02\2021-05-28-10-28-01"
BFEStats_merge["meta"][5 - 1]["stimuli"] = r"N:\Stimuli\2020-BigGAN\2020-07-23-Beto-01\2020-07-23-15-59-38"
BFEStats_merge["meta"][92 - 1]["stimuli"] = r"N:\Stimuli\2020-BigGAN\2020-07-27-Alfa-01\2020-07-27-09-47-40"
#%%
import pickle
matroot = "E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
# pickle.dump(BFEStats, open(join(matroot, "Both_BigGAN_FC6_Evol_Stats_expsep.pkl"), "wb"))
pickle.dump(BFEStats_merge, open(join(matroot, "Both_BigGAN_FC6_Evol_Stats.pkl"), "wb"))
#%% Check the images are loadable
error_dict = {}
for Expi in tqdm(range(1, 191)):
    for thread in range(2):
        try:
            imgfps, resp_vec, bsl_vec, gen_vec = load_img_resp_pairs(BFEStats, Expi,
                         "Evol", thread=thread, stimdrive="S:", output_fmt="vec")
        except Exception as e:
            print(f"Exp{Expi} thread{thread}", e)
            error_dict[(Expi, thread)] = e

#%%
# print(_get_comments(BFEStats, 110))
# _get_ref_img(BFEStats, 110)
print(get_expstr(BFEStats, 12))
#%%
import re
import glob
import shutil
from pathlib import Path
for Expi in [66, 78]: #range(107, 111):
    print(Expi, )
    refimgfns = _get_ref_img(BFEStats, Expi)
    imgfn_strips = [re.sub("_thread\d\d\d_nat", "", refimgfn) for refimgfn in refimgfns]
    imgfn_strips = set(imgfn_strips)
    for imgfn_strip in imgfn_strips:
        # imgfn_strip = re.sub("_thread\d\d\d_nat", "", refimgfn)
        fplist = glob.glob("S:\\Stimuli\\2019-Selectivity\\*\\"+glob.escape(imgfn_strip)+".*")
        if len(fplist) == 0:
            print(imgfn_strip)
        else:
            shutil.copy2(fplist[0], Path(BFEStats[Expi-1]["meta"]["stimuli"]).parent)
            shutil.copy2(fplist[0], Path(BFEStats[Expi-1]["meta"]["stimuli"].replace("N:", "S:")).parent)

