"""
Evolution using specific time step from CorNet-S units.
Run from cluster win Command Line Inferface in large scale.
Binxu
Feb.6th, 2022
Add CLI interface for cluster.
Updated Jun.5th, 2023
"""
import os, sys, argparse, time, glob, pickle, subprocess, shlex, io, pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo
import torchvision
import cornet
from PIL import Image
from easydict import EasyDict
from collections import defaultdict
from os.path import join
import matplotlib.pylab as plt
from core.utils.GAN_utils import upconvGAN, BigGAN_wrapper
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample, one_hot_from_names, save_as_images)
from core.utils.Optimizers import CholeskyCMAES, ZOHA_Sphere_lr_euclid
from core.utils.Optimizers import fix_param_wrapper, concat_wrapper, HessCMAES
from core.utils.layer_hook_utils import featureFetcher, get_module_names, get_layer_names, featureFetcher_recurrent
from core.utils.plot_utils import make_grid, to_imgrid, show_imgrid #, show_tsrbatch, PIL_tsrbatch, ToPILImage
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# sys.path.append("E:\Github_Projects\ActMax-Optimizer-Dev")                 #Binxu local
# sys.path.append(r"D:\Github\ActMax-Optimizer-Dev")                           #Binxu office
# sys.path.append(r"D:\OneDrive - UC San Diego\GitHub\ActMax-Optimizer-Dev")   #Victoria local
#sys.path.append(r"\data\Victoria\UCSD_projects\ActMax-Optimizer-Dev")       #Victoria remote
if sys.platform == "linux":
    # rootdir = r"/scratch/binxu/BigGAN_Optim_Tune_new"
    # Hdir_BigGAN = r"/scratch/binxu/GAN_hessian/BigGAN/summary/H_avg_1000cls.npz"
    # Hdir_fc6 = r"/scratch/binxu/GAN_hessian/FC6GAN/summary/Evolution_Avg_Hess.npz"
    # O2 path interface
    scratchdir = "/n/scratch3/users/b/biw905"  # os.environ['SCRATCH1']
    rootdir = join(scratchdir, "GAN_Evol_cmp")
    Hdir_BigGAN = join("/home/biw905/Hessian", "H_avg_1000cls.npz")  #r"/scratch/binxu/GAN_hessian/BigGAN/summary/H_avg_1000cls.npz"
    Hdir_fc6 = join("/home/biw905/Hessian", "Evolution_Avg_Hess.npz")  #r"/scratch/binxu/GAN_hessian/FC6GAN/summary/Evolution_Avg_Hess.npz"
else:
    # rootdir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_Optim_Tune_tmp"
    rootdir = r"D:\Cluster_Backup\GAN_Evol_cmp" #r"E:\Monkey_Data\BigGAN_Optim_Tune_tmp"
    Hdir_BigGAN = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN\H_avg_1000cls.npz"
    Hdir_fc6 = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\fc6GAN\Evolution_Avg_Hess.npz"


def get_cornet_model(pretrained=False):
    map_location = 'cpu'
    model = getattr(cornet, 'cornet_s')
    model = model(pretrained=pretrained, map_location=map_location)
    model = model.module  # remove DataParallel
    return model


def load_GAN(name):
    if name == "BigGAN":
        BGAN = BigGAN.from_pretrained("biggan-deep-256")
        BGAN.eval().cuda()
        for param in BGAN.parameters():
            param.requires_grad_(False)
        G = BigGAN_wrapper(BGAN)
    elif name == "fc6":
        G = upconvGAN("fc6")
        G.eval().cuda()
        for param in G.parameters():
            param.requires_grad_(False)
    else:
        raise ValueError("Unknown GAN model")
    return G


def load_Hessian(name):
    # Select Hessian
    try:
        if name == "BigGAN":
            H = np.load(Hdir_BigGAN)
        elif name == "fc6":
            H = np.load(Hdir_fc6)
        else:
            raise ValueError("Unknown GAN model")
    except:
        print("Hessian not found for the specified GAN")
        H = None
    return H

#%%
def label2optimizer(methodlabel, init_code, GAN="BigGAN", ):  # TODO add default init_code
    """ Input a label output an grad-free optimizer """
    if GAN == "BigGAN":
        if methodlabel == "CholCMA":
            optim_cust = CholeskyCMAES(space_dimen=256, init_code=init_code, init_sigma=0.2,)  # FIXME: sigma may be too large
        elif methodlabel == "CholCMA_class":
            optim = CholeskyCMAES(space_dimen=128, init_code=init_code[:, 128:], init_sigma=0.06,)
            optim_cust = fix_param_wrapper(optim, init_code[:, :128], pre=True)
        elif methodlabel == "CholCMA_noise":
            optim = CholeskyCMAES(space_dimen=128, init_code=init_code[:, :128], init_sigma=0.3,)
            optim_cust = fix_param_wrapper(optim, init_code[:, 128:], pre=False)
        elif methodlabel == "CholCMA_prod":
            optim1 = CholeskyCMAES(space_dimen=128, init_code=init_code[:, :128], init_sigma=0.1,)
            optim2 = CholeskyCMAES(space_dimen=128, init_code=init_code[:, 128:], init_sigma=0.06,)
            optim_cust = concat_wrapper(optim1, optim2)
        elif methodlabel == "CholCMA_noA":
            optim_cust = CholeskyCMAES(space_dimen=256, init_code=init_code, init_sigma=0.2, Aupdate_freq=102)  # FIXME: sigma may be too large
        elif methodlabel == "HessCMA":
            eva = Hdata['eigvals_avg'][::-1]
            evc = Hdata['eigvects_avg'][:, ::-1]
            optim_cust = HessCMAES(space_dimen=256, init_code=init_code, init_sigma=0.2, )
            optim_cust.set_Hessian(eigvals=eva, eigvects=evc, expon=1 / 2.5)
        elif methodlabel == "HessCMA_noA":
            eva = Hdata['eigvals_avg'][::-1]
            evc = Hdata['eigvects_avg'][:, ::-1]
            optim_cust = HessCMAES(space_dimen=256, init_code=init_code, init_sigma=0.2, Aupdate_freq=102)
            optim_cust.set_Hessian(eigvals=eva, eigvects=evc, expon=1 / 2.5)
        elif methodlabel == "HessCMA_class":
            eva = Hdata['eigvals_clas_avg'][::-1]
            evc = Hdata['eigvects_clas_avg'][:, ::-1]
            optim_hess = HessCMAES(space_dimen=128, init_code=init_code[:, 128:], init_sigma=0.2, )
            optim_hess.set_Hessian(eigvals=eva, eigvects=evc, expon=1 / 2.5)
            optim_cust = fix_param_wrapper(optim_hess, init_code[:, :128], pre=True)
    elif GAN == "fc6":
        if methodlabel == "CholCMA":
            optim_cust = CholeskyCMAES(space_dimen=4096, init_code=init_code, init_sigma=3,)
        elif methodlabel == "CholCMA_noA":
            optim_cust = CholeskyCMAES(space_dimen=4096, init_code=init_code, init_sigma=3, Aupdate_freq=102)
        elif methodlabel == "HessCMA800":
            eva = Hdata['eigv_avg'][::-1]
            evc = Hdata['eigvect_avg'][:, ::-1]
            optim_cust = HessCMAES(space_dimen=4096, cutoff=800, init_code=init_code, init_sigma=0.8, )
            optim_cust.set_Hessian(eigvals=eva, eigvects=evc, cutoff=800, expon=1 / 5)
        elif methodlabel == "HessCMA500":
            eva = Hdata['eigv_avg'][::-1]
            evc = Hdata['eigvect_avg'][:, ::-1]
            optim_cust = HessCMAES(space_dimen=4096, cutoff=500, init_code=init_code, init_sigma=0.8, )
            optim_cust.set_Hessian(eigvals=eva, eigvects=evc, cutoff=500, expon=1 / 5)
        elif methodlabel == "HessCMA500_1":
            eva = Hdata['eigv_avg'][::-1]
            evc = Hdata['eigvect_avg'][:, ::-1]
            optim_cust = HessCMAES(space_dimen=4096, cutoff=500, init_code=init_code, init_sigma=0.4, )
            optim_cust.set_Hessian(eigvals=eva, eigvects=evc, cutoff=500, expon=1 / 4)
    return optim_cust


def resize_and_pad(imgs, corner, size):
    """ Resize and pad images to a square with a given corner and size
    Background is gray.
    Assume image is float, range [0, 1]
    """ # FIXME: this should depend on the input size of image, add canvas size parameter
    pad_img = torch.ones_like(imgs) * 0.5
    rsz_img = F.interpolate(imgs, size=size, align_corners=True, mode="bilinear")
    pad_img[:, :, corner[0]:corner[0]+size[0], corner[1]:corner[1]+size[1]] = rsz_img
    return pad_img
#%%
"""
Actually, if you use higher version of pytorch, the torch transform could work...
Lower version you need to manually write the preprocessing function. 
"""
RGBmean = torch.tensor([0.485, 0.456, 0.406]).reshape([1,3,1,1]).cuda()
RGBstd  = torch.tensor([0.229, 0.224, 0.225]).reshape([1,3,1,1]).cuda()
# imsize = 224
# normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                              std=[0.229, 0.224, 0.225])
# preprocess_fun = torchvision.transforms.Compose([
#                     torchvision.transforms.Resize((imsize, imsize)),
#                     normalize,
#                 ])
def preprocess_fun(imgtsr, imgsize=224, ):
    """Manually write some version of preprocessing"""
    imgtsr = nn.functional.interpolate(imgtsr, [imgsize,imgsize])
    return (imgtsr - RGBmean) / RGBstd

#%%
def visualize_trajectory(scores_all, generations, codes_arr=None, show=False, title_str=""):
    """ Visualize the Score Trajectory """
    gen_slice = np.arange(min(generations), max(generations) + 1)
    AvgScore = np.zeros_like(gen_slice)
    MaxScore = np.zeros_like(gen_slice)
    for i, geni in enumerate(gen_slice):
        AvgScore[i] = np.mean(scores_all[generations == geni])
        MaxScore[i] = np.max(scores_all[generations == geni])
    figh, ax1 = plt.subplots()
    ax1.scatter(generations, scores_all, s=16, alpha=0.6, label="all score")
    ax1.plot(gen_slice, AvgScore, color='black', label="Average score")
    ax1.plot(gen_slice, MaxScore, color='red', label="Max score")
    ax1.set_xlabel("generation #")
    ax1.set_ylabel("CNN unit score")
    plt.legend()
    if codes_arr is not None:
        ax2 = ax1.twinx()
        if codes_arr.shape[1] == 256:  # BigGAN
            nos_norm = np.linalg.norm(codes_arr[:, :128], axis=1)
            cls_norm = np.linalg.norm(codes_arr[:, 128:], axis=1)
            ax2.scatter(generations, nos_norm, s=5, color="orange", label="noise", alpha=0.2)
            ax2.scatter(generations, cls_norm, s=5, color="magenta", label="class", alpha=0.2)
        elif codes_arr.shape[1] == 4096:  # FC6GAN
            norms_all = np.linalg.norm(codes_arr[:, :], axis=1)
            ax2.scatter(generations, norms_all, s=5, color="magenta", label="all", alpha=0.2)
        ax2.set_ylabel("L2 Norm", color="red", fontsize=14)
        plt.legend()
    plt.title("Optimization Trajectory of Score\n" + title_str)
    plt.legend()
    if show:
        plt.show()
    else:
        plt.close(figh)
    return figh


#%%
from argparse import ArgumentParser
parser = ArgumentParser()
# parser.add_argument("--net", type=str, default="alexnet", help="Network model to use for Image distance computation")
parser.add_argument("--area", type=str, default="IT", help="module in cornet")
parser.add_argument("--sublayer", type=str, default="output", help="submodule in area in cornet")
parser.add_argument("--time_range", type=int, nargs='+', default=[0, 2], help="time steps in the recurrent computation")
parser.add_argument("--chans", type=int, nargs='+', default=[0, 25], help="")
parser.add_argument("--G", type=str, default="BigGAN", help="")
parser.add_argument("--optim", type=str, nargs='+', default=["HessCMA", "HessCMA_class", "CholCMA", "CholCMA_prod", "CholCMA_class"], help="")
parser.add_argument("--steps", type=int, default=100, help="")
parser.add_argument("--reps", type=int, default=2, help="")
parser.add_argument("--RFresize", type=bool, default=False, help="")
args = parser.parse_args()
# ["--G", "BigGAN", "--optim", "HessCMA", "CholCMA","--chans",'1','2','--steps','100',"--reps",'2']

#%% Prepare model
# G = upconvGAN("fc6")
# G.eval().cuda().requires_grad_(False)
G = load_GAN(args.G)
Hdata = load_Hessian(args.G)
model = get_cornet_model(pretrained=True)
model.eval().requires_grad_(False)
#%%
# area = "IT"
# sublayer = "output"  # None
# time_range = [0, 1]
# chan_range = [200, 250]
area = args.area
sublayer = args.sublayer
#% Select the Optimizer
method_col = args.optim
#%%
fetcher = featureFetcher_recurrent(model, print_module=False)
h = fetcher.record(area, sublayer, "target")
with torch.no_grad():
    model(preprocess_fun(torch.randn(1, 3, 256, 256).cuda()))
tsr = fetcher["target"][0]
_, C, H, W = tsr.shape
cent_pos = (H // 2, W // 2)
fetcher.remove_hook()
del fetcher
dataroot = r"F:\insilico_exps\CorNet-recurrent-evol_BigGAN_tmp"
#%%
for unit_id in range(args.chans[0], args.chans[1]):
    if "decoder" in area or cent_pos is None:
        savedir = join(rootdir, "corner-s_%s.%s_%d"%(area, sublayer, unit_id))
    else:
        savedir = join(rootdir, r"corner-s_%s.%s_%d_%d_%d" % (area, sublayer, unit_id, cent_pos[0], cent_pos[1]))

    # Save directory named after the unit. Add RFrsz as suffix if resized
    if args.RFresize:
        savedir += "_RFrsz"
    os.makedirs(savedir, exist_ok=True)

    # scorer.select_unit(unit, allow_grad=True)
    fetcher = featureFetcher_recurrent(model, print_module=False)
    h = fetcher.record(area, sublayer, "target")
    for triali in range(args.reps):
        # generate initial code.
        if args.G == "BigGAN":
            fixnoise = 0.7 * truncated_noise_sample(1, 128)
            init_code = np.concatenate((fixnoise, np.zeros((1, 128))), axis=1)
        elif args.G == "fc6":
            init_code = np.random.randn(1, 4096)
        # generate RND label save initial code
        RND = np.random.randint(1E5)
        np.save(join(savedir, "init_code_%05d.npy"%RND), init_code)

        # loop over time steps and methods
        for time_step in range(*args.time_range):
            optimizer_col = [label2optimizer(methodlabel, init_code, args.G) for methodlabel in method_col]
            for methodlab, optimizer in zip(method_col, optimizer_col):
                if args.G == "fc6":
                    # add space notation as suffix to optimizer
                    methodlab += "_fc6"
                # core evolution code
                new_codes = init_code
                scores_all = []
                generations = []
                codes_all = []
                best_imgs = []
                for i in range(args.steps,):
                    codes_all.append(new_codes.copy())
                    latent_code = torch.from_numpy(np.array(new_codes)).float()
                    imgs = G.visualize(latent_code.cuda()).cpu()
                    # if args.RFresize:
                    #       Bug: imgs are resized to 256x256 and it will be further resized in score_tsr
                    #     imgs = resize_and_pad(imgs, corner, imgsize)
                    # scores = scorer.score_tsr(imgs)
                    fetcher.activations["target"] = []
                    with torch.no_grad():
                        model(preprocess_fun(imgs.cuda()))
                    scores = np.array(fetcher["target"][time_step][:, unit_id, cent_pos[0], cent_pos[1]])
                    if args.G == "BigGAN":
                        print("step %d score %.3f (%.3f) (norm %.2f noise norm %.2f)" % (
                            i, scores.mean(), scores.std(), latent_code[:, 128:].norm(dim=1).mean(),
                            latent_code[:, :128].norm(dim=1).mean()))
                    else:
                        print("step %d score %.3f (%.3f) (norm %.2f )" % (
                            i, scores.mean(), scores.std(), latent_code.norm(dim=1).mean(),))
                    new_codes = optimizer.step_simple(scores, new_codes, )
                    scores_all.extend(list(scores))
                    generations.extend([i] * len(scores))
                    best_imgs.append(imgs[scores.argmax(),:,:,:])

                codes_all = np.concatenate(tuple(codes_all), axis=0)
                scores_all = np.array(scores_all)
                generations = np.array(generations)
                mtg_exp = to_imgrid(best_imgs, nrow=10)
                mtg_exp.save(join(savedir, "besteachgen_T%d_%s_%05d.jpg" % (time_step, methodlab, RND,)))
                mtg = to_imgrid(imgs, nrow=7)
                mtg.save(join(savedir, "lastgen_T%d_%s_%05d_score%.1f.jpg" % (time_step, methodlab, RND, scores.mean())))
                # save_imgrid(imgs, join(savedir, "lastgen%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())), nrow=7)
                # save_imgrid(best_imgs, join(savedir, "bestgen%s_%05d.jpg" % (methodlab, RND, )), nrow=10)
                if args.G == "fc6":
                    np.savez(join(savedir, "scores_T%d_%s_%05d.npz" % (time_step, methodlab, RND)),
                     generations=generations, scores_all=scores_all, codes_fin=codes_all[-80:, :])
                else:
                    np.savez(join(savedir, "scores_T%d_%s_%05d.npz" % (time_step, methodlab, RND)),
                     generations=generations, scores_all=scores_all, codes_all=codes_all)
                visualize_trajectory(scores_all, generations, codes_arr=codes_all, title_str=methodlab).savefig(
                    join(savedir, "traj_T%d_%s_%05d_score%.1f.jpg" % (time_step, methodlab, RND, scores.mean())))

fetcher.remove_hook()
del fetcher
#%%
# def visualize_image_trajectory(G, codes_all, generations, show=True):
#     meancodes = [np.mean(codes_all[generations == i, :], axis=0)
#                  for i in range(int(generations.min()), int(generations.max())+1)]
#     meancodes = np.array(meancodes)
#     imgtsrs = G.visualize_batch_np(meancodes)
#     mtg = to_imgrid(imgtsrs, nrow=10)
#     if show:mtg.show()
#     return mtg
#
#
# def visualize_best(G, codes_all, scores_all, show=True):
#     bestidx = np.argmax(scores_all)
#     bestcodes = np.array([codes_all[bestidx,:]])
#     bestimgtsrs = G.visualize_batch_np(bestcodes)
#     mtg = to_imgrid(bestimgtsrs, nrow=1)
#     if show:mtg.show()
#     return mtg
#
#
# def calc_meancodes(codes_all, generations):
#     meancodes = [np.mean(codes_all[generations == i, :], axis=0)
#                  for i in range(int(generations.min()), int(generations.max()) + 1)]
#     meancodes = np.array(meancodes)
#     return meancodes


#%% Evolution parameters and Optimzer
# def run_evolution(G, optim, model, area, sublayer, time_step, channum, pos="autocenter"):
#     if pos is "autocenter":
#         # find the center of the feature map
#         findcenter = True
#     else:
#         # predefine the position
#         findcenter = False
#         assert len(pos) == 2 and type(pos) in [list, tuple]
#     fetcher = featureFetcher_recurrent(model, print_module=False)
#     h = fetcher.record(area, sublayer, "target")
#     if findcenter:
#         with torch.no_grad():
#             model(preprocess_fun(torch.randn(1, 3, 256, 256).cuda()))
#         tsr = fetcher["target"][time_step]
#         _, C, H, W = tsr.shape
#         pos = (H // 2, W // 2)
#
#     print("Evolve from {}_{}_Time {}_ Channel {} Position {}".\
#           format(area, sublayer, str(time_step), str(channum), str(pos)))
#     # optim = CholeskyCMAES(4096, population_size=40, init_sigma=2.0, Aupdate_freq=10, init_code=np.zeros([1, 4096]))
#     # optim = ZOHA_Sphere_lr_euclid(4096, population_size=40, select_size=20,
#     #                                   lr=1.5, sphere_norm=300)
#     # optim.lr_schedule(n_gen=75, mode="exp", lim=(50, 7.33), )
#     codes_col = []
#     gen_col = []
#     scores_col = []
#     codes = optim.get_init_pop()
#     for i in range(100):
#         # get score
#         fetcher.activations["target"] = []
#         with torch.no_grad():
#             ppx = preprocess_fun(G.visualize(torch.tensor(codes, dtype=torch.float32, device="cuda")))
#             model(ppx)
#         scores = np.array(fetcher["target"][time_step][:, channum, pos[0], pos[1]])
#         # optimizer update
#         newcodes = optim.step_simple(scores, codes)
#         gen_col.append(i * np.ones_like(scores, dtype=np.int))
#         scores_col.append(scores)
#         codes_col.append(codes)
#         # print(f"Gen {i:d} {scores.mean():.3f}+-{scores.std():.3f}")
#         codes = newcodes
#         del newcodes
#     generations = np.concatenate(tuple(gen_col), axis=0)
#     scores_all = np.concatenate(tuple(scores_col), axis=0)
#     codes_all = np.concatenate(tuple(codes_col), axis=0)
#     fetcher.remove_hook()
#     del fetcher
#     return codes, scores, \
#            EasyDict(generations=generations,scores_all=scores_all,codes_all=codes_all,)

#%%
# area = "IT"
# sublayer = "output"  # None
# time_range = [0, 1]
# chan_range = [200, 250]
# repN = 5
# outdir = join(dataroot, "%s-%s"%(area, sublayer))
# os.makedirs(outdir, exist_ok=True)
# for runnum in range(repN):
#     for channum in range(*chan_range):
#         for time_step in time_range:
#             explabel = f"{area}-{sublayer}-Ch{channum:03d}-T{time_step:d}-run{runnum:02d}"
#             meta = EasyDict(area=area, sublayer=sublayer, channum=channum, time_step=time_step,
#                             runnum=runnum, explabel=explabel,)
#             t0 = time.time()
#             codes, scores, datadict = run_evolution(model, area, sublayer, time_step, channum, pos="autocenter")
#             t1 = time.time()
#             print(f"Final activation {scores.mean():.2f}+-{scores.std():.2f} time {t1 - t0:.3f} sec")
#             figh = visualize_trajectory(datadict.scores_all, datadict.generations, False)
#             figh.savefig(join(outdir, "score_traj_%s.png" % (explabel)))
#             plt.close(figh)
#             meancodes = calc_meancodes(datadict.codes_all, datadict.generations)
#             # mtg = visualize_image_trajectory(G, datadict.codes_all, datadict.generations, False)
#             mtg = to_imgrid(G.visualize_batch_np(meancodes), nrow=10)
#             mtg.save(join(outdir, "evol_img_traj_%s.jpg" % (explabel)))
#             bestmtg = visualize_best(G, datadict.codes_all, datadict.scores_all, False)
#             bestmtg.save(join(outdir, "bestimg_%s.jpg" % (explabel)))
#             np.savez(join(outdir, "exp_data_%s.png" % (explabel)),
#                    meancodes=meancodes, generations=datadict.generations,
#                    scores_all=datadict.scores_all, **meta,)
#             t2 = time.time()
#             print(f"Finish saving time {t2 - t0:.3f} sec")
#             # del mtg, bestmtg
#
# # ToPILImage()(make_grid(G.visualize(torch.tensor(codes, dtype=torch.float32, device="cuda")).cpu()))
# # Final activation 20.48+-1.08 time 64.520 sec
# # Final activation 83.59+-5.11 time 64.792 sec
# # Final activation 16.68+-0.97 time 55.456 sec
# # Final activation 96.97+-6.94 time 57.806 sec
# #%%
# area = "V4"
# sublayer = "output"  # None
# outdir = join(dataroot, "%s-%s"%(area, sublayer))
# os.makedirs(outdir, exist_ok=True)
# for runnum in range(5):
#     for channum in range(25, 50):
#         for time_step in [0, 1, 2, 3]:
#             explabel = f"{area}-{sublayer}-Ch{channum:03d}-T{time_step:d}-run{runnum:02d}"
#             meta = EasyDict(area=area, sublayer=sublayer, channum=channum, time_step=time_step,
#                             runnum=runnum, explabel=explabel,)
#             t0 = time.time()
#             codes, scores, datadict = run_evolution(model, area, sublayer, time_step, channum, pos="autocenter")
#             t1 = time.time()
#             print(f"Final activation {scores.mean():.2f}+-{scores.std():.2f} time {t1 - t0:.3f} sec")
#             meancodes = calc_meancodes(datadict.codes_all, datadict.generations)
#             figh = visualize_trajectory(datadict.scores_all, datadict.generations, False)
#             figh.savefig(join(outdir, "score_traj_%s.png" % (explabel)))
#             plt.close(figh)
#             # mtg = visualize_image_trajectory(G, datadict.codes_all, datadict.generations, False)
#             mtg = to_imgrid(G.visualize_batch_np(meancodes), nrow=10)
#             mtg.save(join(outdir, "evol_img_traj_%s.jpg" % (explabel)))
#             bestmtg = visualize_best(G, datadict.codes_all, datadict.scores_all, False)
#             bestmtg.save(join(outdir, "bestimg_%s.jpg" % (explabel)))
#             np.savez(join(outdir, "exp_data_%s.png" % (explabel)),
#                    meancodes=meancodes, generations=datadict.generations,
#                      scores_all=datadict.scores_all, **meta,)
#             t2 = time.time()
#             print(f"Finish saving time {t2 - t0:.3f} sec")
#             # del mtg, bestmtg
#
# #%%
# area = "V2"
# sublayer = "output"  # None
# outdir = join(dataroot, "%s-%s"%(area, sublayer))
# os.makedirs(outdir, exist_ok=True)
# for runnum in range(5):
#     for channum in range(25, 50):
#         for time_step in [0, 1]:
#             explabel = f"{area}-{sublayer}-Ch{channum:03d}-T{time_step:d}-run{runnum:02d}"
#             meta = EasyDict(area=area, sublayer=sublayer, channum=channum, time_step=time_step,
#                             runnum=runnum, explabel=explabel,)
#             t0 = time.time()
#             codes, scores, datadict = run_evolution(model, area, sublayer, time_step, channum, pos="autocenter")
#             t1 = time.time()
#             print(f"Final activation {scores.mean():.2f}+-{scores.std():.2f} time {t1 - t0:.3f} sec")
#             meancodes = calc_meancodes(datadict.codes_all, datadict.generations)
#             figh = visualize_trajectory(datadict.scores_all, datadict.generations, False)
#             figh.savefig(join(outdir, "score_traj_%s.png" % (explabel)))
#             plt.close(figh)
#             # mtg = visualize_image_trajectory(G, datadict.codes_all, datadict.generations, False)
#             mtg = to_imgrid(G.visualize_batch_np(meancodes), nrow=10)
#             mtg.save(join(outdir, "evol_img_traj_%s.jpg" % (explabel)))
#             bestmtg = visualize_best(G, datadict.codes_all, datadict.scores_all, False)
#             bestmtg.save(join(outdir, "bestimg_%s.jpg" % (explabel)))
#             np.savez(join(outdir, "exp_data_%s.png" % (explabel)),
#                    meancodes=meancodes, generations=datadict.generations,
#                      scores_all=datadict.scores_all, **meta,)
#             t2 = time.time()
#             print(f"Finish saving time {t2 - t0:.3f} sec")
#             # del mtg, bestmtg
#%%
# #%%
# time_steps = [0, 1]
# area = "IT"
# sublayer = "conv3"
#
# import random
# C = 512 # np.shape(fetcher["target"][time_step])[1]
# channums = random.sample(range(C), 200)
# for channum in channums:
#     for time_step in time_steps:
#         for i in range(3):
#             fetcher, codes, scores = run_evolution(model, area, sublayer, time_step, channum)
#             pil_image = ToPILImage()(make_grid(G.visualize(torch.tensor(codes, dtype=torch.float32, device="cuda")).cpu()))
#             # filename = "N:\\Users\\Victoria_data\\CORnet_evolution\\{}_{}_time_{}_chan_{}_trial_{}_score{}.png".format(
#             #     area, sublayer, str(time_step), str(channum), str(i), format(scores.mean(), ".2f"))
#
#             filename = "D:\\Ponce-Lab\\Victoria\\Victoria_data\\CORnet_evolution\\{}_{}_time_{}_chan_{}_trial_{}_score{}.png".format(area, sublayer, str(time_step), str(channum), str(i), format(scores.mean(),".2f"))
#             pil_image.save(filename)
#             del codes, pil_image



