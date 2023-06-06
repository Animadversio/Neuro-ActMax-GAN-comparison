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
import torch.utils.model_zoo
import torchvision
import cornet
from PIL import Image
from easydict import EasyDict
from collections import defaultdict
from os.path import join
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# sys.path.append("E:\Github_Projects\ActMax-Optimizer-Dev")                 #Binxu local
# sys.path.append(r"D:\Github\ActMax-Optimizer-Dev")                           #Binxu office
# sys.path.append(r"D:\OneDrive - UC San Diego\GitHub\ActMax-Optimizer-Dev")   #Victoria local
#sys.path.append(r"\data\Victoria\UCSD_projects\ActMax-Optimizer-Dev")       #Victoria remote
import matplotlib.pylab as plt
from core.utils.GAN_utils import upconvGAN
from core.utils.Optimizers import CholeskyCMAES, ZOHA_Sphere_lr_euclid
from core.utils.layer_hook_utils import featureFetcher, get_module_names, get_layer_names, featureFetcher_recurrent
from core.utils.plot_utils import make_grid, to_imgrid, show_imgrid #, show_tsrbatch, PIL_tsrbatch, ToPILImage
#%%
"""
Actually, if you use higher version of pytorch, the torch transform could work...
Lower version you need to manually write the preprocessing function. 
"""
def get_model(pretrained=False):
    map_location = 'cpu'
    model = getattr(cornet, 'cornet_s')
    model = model(pretrained=pretrained, map_location=map_location)
    model = model.module  # remove DataParallel
    return model

# imsize = 224
# normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                              std=[0.229, 0.224, 0.225])
# preprocess_fun = torchvision.transforms.Compose([
#                     torchvision.transforms.Resize((imsize, imsize)),
#                     normalize,
#                 ])
RGBmean = torch.tensor([0.485, 0.456, 0.406]).reshape([1,3,1,1]).cuda()
RGBstd  = torch.tensor([0.229, 0.224, 0.225]).reshape([1,3,1,1]).cuda()
def preprocess_fun(imgtsr, imgsize=224, ):
    """Manually write some version of preprocessing"""
    imgtsr = nn.functional.interpolate(imgtsr, [imgsize,imgsize])
    return (imgtsr - RGBmean) / RGBstd
#%%
def visualize_trajectory(scores_all, generations, show=True):
    gen_slice = np.arange(min(generations), max(generations)+1)
    AvgScore = np.zeros_like(gen_slice).astype("float64")
    MaxScore = np.zeros_like(gen_slice).astype("float64")
    for i, geni in enumerate(gen_slice):
        AvgScore[i] = np.mean(scores_all[generations == geni])
        MaxScore[i] = np.max(scores_all[generations == geni])
    figh = plt.figure(figsize=[6,5])
    plt.scatter(generations, scores_all, s=16, alpha=0.6, label="all score")
    plt.plot(gen_slice, AvgScore, color='black', label="Average score")
    plt.plot(gen_slice, MaxScore, color='red', label="Max score")
    plt.xlabel("generation #")
    plt.ylabel("CNN unit score")
    plt.title("Optimization Trajectory of Score\n")# + title_str)
    plt.legend()
    if show:
        plt.show()
    return figh


def visualize_image_trajectory(G, codes_all, generations, show=True):
    meancodes = [np.mean(codes_all[generations == i, :], axis=0)
                 for i in range(int(generations.min()), int(generations.max())+1)]
    meancodes = np.array(meancodes)
    imgtsrs = G.visualize_batch_np(meancodes)
    mtg = to_imgrid(imgtsrs, nrow=10)
    if show:mtg.show()
    return mtg


def visualize_best(G, codes_all, scores_all, show=True):
    bestidx = np.argmax(scores_all)
    bestcodes = np.array([codes_all[bestidx,:]])
    bestimgtsrs = G.visualize_batch_np(bestcodes)
    mtg = to_imgrid(bestimgtsrs, nrow=1)
    if show:mtg.show()
    return mtg


def calc_meancodes(codes_all, generations):
    meancodes = [np.mean(codes_all[generations == i, :], axis=0)
                 for i in range(int(generations.min()), int(generations.max()) + 1)]
    meancodes = np.array(meancodes)
    return meancodes


#%% Prepare model
G = upconvGAN("fc6")
G.eval().cuda().requires_grad_(False)

model = get_model(pretrained=True)
model.eval().requires_grad_(False)
#%% Evolution parameters and Optimzer
def run_evolution(model, area, sublayer, time_step, channum, pos="autocenter"):
    if pos is "autocenter":
        # find the center of the feature map
        findcenter = True
    else:
        # predefine the position
        findcenter = False
        assert len(pos) == 2 and type(pos) in [list, tuple]
    fetcher = featureFetcher_recurrent(model, print_module=False)
    h = fetcher.record(area, sublayer, "target")
    if findcenter:
        with torch.no_grad():
            model(preprocess_fun(torch.randn(1, 3, 256, 256).cuda()))
        tsr = fetcher["target"][time_step]
        _, C, H, W = tsr.shape
        pos = (H // 2, W // 2)

    print("Evolve from {}_{}_Time {}_ Channel {} Position {}".\
          format(area, sublayer, str(time_step), str(channum), str(pos)))
    optim = CholeskyCMAES(4096, population_size=40, init_sigma=2.0, Aupdate_freq=10, init_code=np.zeros([1, 4096]))
    # optim = ZOHA_Sphere_lr_euclid(4096, population_size=40, select_size=20,
    #                                   lr=1.5, sphere_norm=300)
    # optim.lr_schedule(n_gen=75, mode="exp", lim=(50, 7.33), )
    codes_col = []
    gen_col = []
    scores_col = []
    codes = optim.get_init_pop()
    for i in range(100):
        # get score
        fetcher.activations["target"] = []
        with torch.no_grad():
            ppx = preprocess_fun(G.visualize(torch.tensor(codes, dtype=torch.float32, device="cuda")))
            model(ppx)
        scores = np.array(fetcher["target"][time_step][:, channum, pos[0], pos[1]])
        # optimizer update
        newcodes = optim.step_simple(scores, codes)
        gen_col.append(i * np.ones_like(scores, dtype=np.int))
        scores_col.append(scores)
        codes_col.append(codes)
        # print(f"Gen {i:d} {scores.mean():.3f}+-{scores.std():.3f}")
        codes = newcodes
        del newcodes
    generations = np.concatenate(tuple(gen_col), axis=0)
    scores_all = np.concatenate(tuple(scores_col), axis=0)
    codes_all = np.concatenate(tuple(codes_col), axis=0)
    fetcher.remove_hook()
    del fetcher
    return codes, scores, \
           EasyDict(generations=generations,scores_all=scores_all,codes_all=codes_all,)


dataroot = r"F:\insilico_exps\CorNet-recurrent-evol_tmp"
#%%
for unit_id in range(args.chans[0], args.chans[1]):
    if "fc" in args.layer or cent_pos is None:
        unit = (args.net, args.layer, unit_id)
        savedir = join(rootdir, r"%s_%s_%d" % unit[:3])
    else:
        unit = (args.net, args.layer, unit_id, *cent_pos)
        savedir = join(rootdir, r"%s_%s_%d_%d_%d" % unit[:5])
    scorer.select_unit(unit, allow_grad=True)
    # Save directory named after the unit. Add RFrsz as suffix if resized
    if args.RFresize:
        savedir += "_RFrsz"

    os.makedirs(savedir, exist_ok=True)
    for triali in range(args.reps):
        # generate initial code.
        if args.G == "BigGAN":
            fixnoise = 0.7 * truncated_noise_sample(1, 128)
            init_code = np.concatenate((fixnoise, np.zeros((1, 128))), axis=1)
        elif args.G == "fc6":
            init_code = np.random.randn(1, 4096)
        RND = np.random.randint(1E5)
        np.save(join(savedir, "init_code_%05d.npy"%RND), init_code)
        optimizer_col = [label2optimizer(methodlabel, init_code, args.G) for methodlabel in method_col]
        for methodlab, optimizer in zip(method_col, optimizer_col):
            if args.G == "fc6":  methodlab += "_fc6"  # add space notation as suffix to optimizer
            # core evolution code
            new_codes = init_code
            # new_codes = init_code + np.random.randn(25, 256) * 0.06
            scores_all = []
            generations = []
            codes_all = []
            best_imgs = []
            for i in range(args.steps,):
                codes_all.append(new_codes.copy())
                latent_code = torch.from_numpy(np.array(new_codes)).float()
                # imgs = G.visualize_batch_np(new_codes) # B=1
                imgs = G.visualize(latent_code.cuda()).cpu()
                if args.RFresize:
                    imgs = resize_and_pad(imgs, corner, imgsize)  #  Bug: imgs are resized to 256x256 and it will be further resized in score_tsr
                scores = scorer.score_tsr(imgs)
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
            mtg_exp.save(join(savedir, "besteachgen%s_%05d.jpg" % (methodlab, RND,)))
            mtg = to_imgrid(imgs, nrow=7)
            mtg.save(join(savedir, "lastgen%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))
            # save_imgrid(imgs, join(savedir, "lastgen%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())), nrow=7)
            # save_imgrid(best_imgs, join(savedir, "bestgen%s_%05d.jpg" % (methodlab, RND, )), nrow=10)
            if args.G == "fc6":
                np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)),
                 generations=generations, scores_all=scores_all, codes_fin=codes_all[-80:, :])
            else:
                np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)),
                 generations=generations, scores_all=scores_all, codes_all=codes_all)
            visualize_trajectory(scores_all, generations, codes_arr=codes_all, title_str=methodlab).savefig(
                join(savedir, "traj%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))

#%%
area = "IT"
sublayer = "output"  # None
time_range = [0, 1]
chan_range = [200, 250]
repN = 5
outdir = join(dataroot, "%s-%s"%(area, sublayer))
os.makedirs(outdir, exist_ok=True)
for runnum in range(repN):
    for channum in range(*chan_range):
        for time_step in time_range:
            explabel = f"{area}-{sublayer}-Ch{channum:03d}-T{time_step:d}-run{runnum:02d}"
            meta = EasyDict(area=area, sublayer=sublayer, channum=channum, time_step=time_step,
                            runnum=runnum, explabel=explabel,)
            t0 = time.time()
            codes, scores, datadict = run_evolution(model, area, sublayer, time_step, channum, pos="autocenter")
            t1 = time.time()
            print(f"Final activation {scores.mean():.2f}+-{scores.std():.2f} time {t1 - t0:.3f} sec")
            figh = visualize_trajectory(datadict.scores_all, datadict.generations, False)
            figh.savefig(join(outdir, "score_traj_%s.png" % (explabel)))
            plt.close(figh)
            meancodes = calc_meancodes(datadict.codes_all, datadict.generations)
            # mtg = visualize_image_trajectory(G, datadict.codes_all, datadict.generations, False)
            mtg = to_imgrid(G.visualize_batch_np(meancodes), nrow=10)
            mtg.save(join(outdir, "evol_img_traj_%s.jpg" % (explabel)))
            bestmtg = visualize_best(G, datadict.codes_all, datadict.scores_all, False)
            bestmtg.save(join(outdir, "bestimg_%s.jpg" % (explabel)))
            np.savez(join(outdir, "exp_data_%s.png" % (explabel)),
                   meancodes=meancodes, generations=datadict.generations,
                   scores_all=datadict.scores_all, **meta,)
            t2 = time.time()
            print(f"Finish saving time {t2 - t0:.3f} sec")
            # del mtg, bestmtg

# ToPILImage()(make_grid(G.visualize(torch.tensor(codes, dtype=torch.float32, device="cuda")).cpu()))
# Final activation 20.48+-1.08 time 64.520 sec
# Final activation 83.59+-5.11 time 64.792 sec
# Final activation 16.68+-0.97 time 55.456 sec
# Final activation 96.97+-6.94 time 57.806 sec
#%%
area = "V4"
sublayer = "output"  # None
outdir = join(dataroot, "%s-%s"%(area, sublayer))
os.makedirs(outdir, exist_ok=True)
for runnum in range(5):
    for channum in range(25, 50):
        for time_step in [0, 1, 2, 3]:
            explabel = f"{area}-{sublayer}-Ch{channum:03d}-T{time_step:d}-run{runnum:02d}"
            meta = EasyDict(area=area, sublayer=sublayer, channum=channum, time_step=time_step,
                            runnum=runnum, explabel=explabel,)
            t0 = time.time()
            codes, scores, datadict = run_evolution(model, area, sublayer, time_step, channum, pos="autocenter")
            t1 = time.time()
            print(f"Final activation {scores.mean():.2f}+-{scores.std():.2f} time {t1 - t0:.3f} sec")
            meancodes = calc_meancodes(datadict.codes_all, datadict.generations)
            figh = visualize_trajectory(datadict.scores_all, datadict.generations, False)
            figh.savefig(join(outdir, "score_traj_%s.png" % (explabel)))
            plt.close(figh)
            # mtg = visualize_image_trajectory(G, datadict.codes_all, datadict.generations, False)
            mtg = to_imgrid(G.visualize_batch_np(meancodes), nrow=10)
            mtg.save(join(outdir, "evol_img_traj_%s.jpg" % (explabel)))
            bestmtg = visualize_best(G, datadict.codes_all, datadict.scores_all, False)
            bestmtg.save(join(outdir, "bestimg_%s.jpg" % (explabel)))
            np.savez(join(outdir, "exp_data_%s.png" % (explabel)),
                   meancodes=meancodes, generations=datadict.generations,
                     scores_all=datadict.scores_all, **meta,)
            t2 = time.time()
            print(f"Finish saving time {t2 - t0:.3f} sec")
            # del mtg, bestmtg

#%%
area = "V2"
sublayer = "output"  # None
outdir = join(dataroot, "%s-%s"%(area, sublayer))
os.makedirs(outdir, exist_ok=True)
for runnum in range(5):
    for channum in range(25, 50):
        for time_step in [0, 1]:
            explabel = f"{area}-{sublayer}-Ch{channum:03d}-T{time_step:d}-run{runnum:02d}"
            meta = EasyDict(area=area, sublayer=sublayer, channum=channum, time_step=time_step,
                            runnum=runnum, explabel=explabel,)
            t0 = time.time()
            codes, scores, datadict = run_evolution(model, area, sublayer, time_step, channum, pos="autocenter")
            t1 = time.time()
            print(f"Final activation {scores.mean():.2f}+-{scores.std():.2f} time {t1 - t0:.3f} sec")
            meancodes = calc_meancodes(datadict.codes_all, datadict.generations)
            figh = visualize_trajectory(datadict.scores_all, datadict.generations, False)
            figh.savefig(join(outdir, "score_traj_%s.png" % (explabel)))
            plt.close(figh)
            # mtg = visualize_image_trajectory(G, datadict.codes_all, datadict.generations, False)
            mtg = to_imgrid(G.visualize_batch_np(meancodes), nrow=10)
            mtg.save(join(outdir, "evol_img_traj_%s.jpg" % (explabel)))
            bestmtg = visualize_best(G, datadict.codes_all, datadict.scores_all, False)
            bestmtg.save(join(outdir, "bestimg_%s.jpg" % (explabel)))
            np.savez(join(outdir, "exp_data_%s.png" % (explabel)),
                   meancodes=meancodes, generations=datadict.generations,
                     scores_all=datadict.scores_all, **meta,)
            t2 = time.time()
            print(f"Finish saving time {t2 - t0:.3f} sec")
            # del mtg, bestmtg
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



