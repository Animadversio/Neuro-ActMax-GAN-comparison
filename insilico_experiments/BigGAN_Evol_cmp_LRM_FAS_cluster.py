""" Cluster version of BigGAN Evol """
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append("/n/home12/binxuwang/Github/Neuro-ActMax-GAN-comparison")
import tqdm
import numpy as np
from os.path import join
import seaborn as sns
import matplotlib.pylab as plt
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample, one_hot_from_names, save_as_images)
from core.utils.CNN_scorers import TorchScorer
from core.utils.GAN_utils import BigGAN_wrapper, upconvGAN, loadBigGAN
from core.utils.grad_RF_estim import grad_RF_estimate, gradmap2RF_square
from core.utils.layer_hook_utils import get_module_names, layername_dict, register_hook_by_module_names, get_module_name_shapes
from core.utils.layer_hook_utils import featureFetcher, featureFetcher_module, featureFetcher_recurrent
from core.utils.Optimizers import CholeskyCMAES, HessCMAES, ZOHA_Sphere_lr_euclid
from core.utils.Optimizers import label2optimizer

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


def visualize_trajectory(scores_all, generations, codes_arr=None, show=False, title_str=""):
    """ Visualize the Score Trajectory """
    gen_slice = np.arange(min(generations), max(generations) + 1)
    AvgScore = np.zeros(gen_slice.shape)
    MaxScore = np.zeros(gen_slice.shape)
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


def resize_and_pad(imgs, corner, size):
    """ Resize and pad images to a square with a given corner and size
    Background is gray.
    Assume image is float, range [0, 1]
    """ # FIXME: this should depend on the input size of image, add canvas size parameter
    pad_img = torch.ones_like(imgs) * 0.5
    rsz_img = F.interpolate(imgs, size=size, align_corners=True, mode="bilinear")
    pad_img[:, :, corner[0]:corner[0]+size[0], corner[1]:corner[1]+size[1]] = rsz_img
    return pad_img


def get_center_pos_and_rf(model, layer, input_size=(3, 227, 227), device="cuda"):
    module_names, module_types, module_spec = get_module_names(model, input_size=input_size, device=device)
    layer_key = [k for k, v in module_names.items() if v == layer][0]
    # FIXME note this may not work when multiple layers have the same name.
    # This will only get the first one. Need to add a check for this.
    feat_outshape = module_spec[layer_key]['outshape']
    if len(feat_outshape) == 3:
        cent_pos = (feat_outshape[1]//2, feat_outshape[2]//2)
    elif len(feat_outshape) == 1:
        cent_pos = None
    else:
        raise ValueError(f"Unknown layer shape {feat_outshape} for layer {layer}")

    # rf Mapping,
    if len(feat_outshape) == 3: # fixit
        print("Computing RF by direct backprop: ")
        gradAmpmap = grad_RF_estimate(model, layer, (slice(None), *cent_pos), input_size=input_size,
                                      device=device, show=False, reps=30, batch=1)
        Xlim, Ylim = gradmap2RF_square(gradAmpmap, absthresh=1E-8, relthresh=0.01, square=True)
        corner = (Xlim[0], Ylim[0])
        imgsize = (Xlim[1] - Xlim[0], Ylim[1] - Ylim[0])
    elif len(feat_outshape) == 1:
        imgsize = input_size[-2:]
        corner = (0, 0)
        Xlim = (corner[0], corner[0] + imgsize[0])
        Ylim = (corner[1], corner[1] + imgsize[1])
    else:
        raise ValueError(f"Unknown layer shape {feat_outshape} for layer {layer}")

    return cent_pos, corner, imgsize, Xlim, Ylim



layername_map = {".feedforward.features.Conv2d0": "conv1",
                ".feedforward.features.ReLU1": "conv1_relu",
                ".feedforward.features.MaxPool2d2": "pool1",
                ".feedforward.features.Conv2d3": "conv2",
                ".feedforward.features.ReLU4": "conv2_relu",
                ".feedforward.features.MaxPool2d5": "pool2",
                ".feedforward.features.Conv2d6": "conv3",
                ".feedforward.features.ReLU7": "conv3_relu",
                ".feedforward.features.Conv2d8": "conv4",
                ".feedforward.features.ReLU9": "conv4_relu",
                ".feedforward.features.Conv2d10": "conv5",
                ".feedforward.features.ReLU11": "conv5_relu",
                ".feedforward.features.MaxPool2d12": "pool5",
                ".feedforward.AdaptiveAvgPool2davgpool": "avgpool",
                ".feedforward.classifier.Dropout0": "avgpool_dropout",
                ".feedforward.classifier.Linear1": "fc6",
                ".feedforward.classifier.ReLU2": "fc6_relu",
                ".feedforward.classifier.Dropout3": 'fc6_dropout',
                ".feedforward.classifier.Linear4": "fc7",
                ".feedforward.classifier.ReLU5": "fc7_relu",
                ".feedforward.classifier.Linear6": "fc8",}
layername_inv_map = {v: k for k, v in layername_map.items()}

import argparse
parser = argparse.ArgumentParser(description='Evolvability Comparison')
parser.add_argument('--model', type=str, default='alexnet_lrm3', help='Model name')
parser.add_argument('--layershort', type=str, default='fc6_relu', help='Layer name')
parser.add_argument('--channel_rng', type=int, default=(0, 25), nargs=2, help='Channel range')
parser.add_argument('--reps', type=int, default=5, help='Number of repetitions')
parser.add_argument('--steps', type=int, default=100, help='Number of steps')
parser.add_argument('--RFresize', action='store_true', help='Resize images to RF size')
parser.add_argument('--drytmp', action='store_true', help='Dry run to temporary folder')
args = parser.parse_args()

if args.drytmp:
    print ("Dry run to temporary folder")
    saveroot = r"/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Projects/Evol_lrm_GAN_cmp/trashc"
else:
    saveroot = r"/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Projects/Evol_lrm_GAN_cmp"
    

model, transforms = torch.hub.load('harvard-visionlab/lrm-steering', 'alexnet_lrm3', pretrained=True, steering=True, force_reload=True)
input_size = (3, 224, 224)
layerkey = args.layershort
layername = layername_inv_map[layerkey]
model.to("cuda")
model.forward_passes = 1
cent_pos, corner, imgsize, Xlim, Ylim = get_center_pos_and_rf(model, layername,
                                          input_size=input_size, device="cuda")
print("Target setting network %s layer %s, center pos" % (args.model, layername), cent_pos)
print("Xlim %s Ylim %s \n imgsize %s corner %s" % (Xlim, Ylim, imgsize, corner))
model.forward_passes = 2
fetcher = featureFetcher(model, input_size=input_size, device="cuda", 
                         print_module=False, store_device="cpu", )
fetcher.record(layername, store_name=layerkey)
max_forward = 4 
for iChannel in range(args.channel_rng[0], args.channel_rng[1]):
    savedir = join(saveroot, f"{args.model}-{layerkey}-Ch{iChannel:04d}")
    os.makedirs(savedir, exist_ok=True)
    for GANname in ["fc6", "BigGAN"]:
        G = load_GAN(GANname)
        # use the same initial code for all trials
        if GANname == "BigGAN":
            fixnoise = 0.7 * truncated_noise_sample(1, 128)
            init_code = np.concatenate((fixnoise, np.zeros((1, 128))), axis=1)
        elif GANname == "fc6":
            init_code = np.random.randn(1, 4096)
        else:
            raise ValueError("Unknown GAN model")
        for trial in range(args.reps):
            # Use the same random seed / seq of random vector for time points
            RND = np.random.randint(100000)
            for iT in range(max_forward):
                print (f"{layerkey} Ch{iChannel} T{iT} | GAN {GANname} CMAES Trial | {trial}")
                methodlab = f"{GANname}_CMAES_T{iT}" # alex-lrm3-{layerkey}_Ch{iChannel}_T{iT}_
                if GANname == "BigGAN":
                    optimizer = CholeskyCMAES(256, init_sigma=0.2,)
                elif GANname == "fc6":
                    optimizer = CholeskyCMAES(init_code.shape[1], init_sigma=3,)
                else:
                    raise ValueError
                np.random.seed(RND)
                new_codes = init_code.copy()
                # new_codes = init_code + np.random.randn(25, 256) * 0.06
                scores_all = []
                scores_dyn_all = []
                generations = []
                codes_all = []
                best_imgs = []
                for i in range(args.steps,):
                    codes_all.append(new_codes.copy())
                    latent_code = torch.from_numpy(np.array(new_codes)).float()
                    imgs = G.visualize(latent_code.cuda()) # (B, 3, 256, 256)
                    if args.RFresize:
                        #  Bug: imgs are resized to 256x256 and it will be further resized in score_tsr
                        imgs = resize_and_pad(imgs, corner, imgsize)  
                    imgs.to("cuda")
                    # scores = scorer.score_tsr(imgs)
                    scores_dyn = []
                    with torch.no_grad():
                        # first pass . drop state 
                        model(imgs, drop_state=True, forward_passes=1)
                        if cent_pos is None:
                            scores_dyn.append(fetcher[layerkey][:, iChannel].cpu().detach().numpy())
                        else:
                            scores_dyn.append(fetcher[layerkey][:, iChannel, cent_pos[0], cent_pos[1]].cpu().detach().numpy())
                        for iforward in range(0, max_forward):
                            # subsequent passes . keep state to enable recurrence. 
                            model(imgs, drop_state=True if iforward == 0 else False, forward_passes=1)
                            if cent_pos is None:
                                scores_dyn.append(fetcher[layerkey][:, iChannel].cpu().detach().numpy())
                            else:
                                scores_dyn.append(fetcher[layerkey][:, iChannel, cent_pos[0], cent_pos[1]].cpu().detach().numpy())
                            
                    scores = scores_dyn[iT]
                    scores_dyn = np.stack(scores_dyn, axis=1) # (Batch, Time)
                    # main optimization step                    
                    new_codes = optimizer.step_simple(scores, new_codes, verbosity=0)
                    # save step results
                    scores_all.extend(list(scores))
                    scores_dyn_all.extend(list(scores_dyn))
                    generations.extend([i] * len(scores))
                    best_imgs.append(imgs[scores.argmax(),:,:,:])
                
                if GANname == "BigGAN":
                    print("step %d score %.3f (%.3f) (norm %.2f noise norm %.2f)" % (
                        i, scores.mean(), scores.std(), latent_code[:, 128:].norm(dim=1).mean(),
                        latent_code[:, :128].norm(dim=1).mean()))
                elif GANname == "fc6":
                    print("step %d score %.3f (%.3f) (norm %.2f )" % (
                        i, scores.mean(), scores.std(), latent_code.norm(dim=1).mean(),))
                else:
                    raise ValueError("Unknown GAN model")
                codes_all = np.concatenate(tuple(codes_all), axis=0)
                scores_all = np.array(scores_all)
                scores_dyn_all = np.array(scores_dyn_all)
                generations = np.array(generations)
                mtg_exp = ToPILImage()(make_grid(best_imgs, nrow=10))
                mtg_exp.save(join(savedir, "besteachgen%s_%05d.jpg" % (methodlab, RND,)))
                mtg = ToPILImage()(make_grid(imgs, nrow=7))
                mtg.save(join(savedir, "lastgen%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))
                if GANname == "fc6":
                    np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)),
                        generations=generations, scores_all=scores_all, scores_dyn_all=scores_dyn_all, codes_fin=codes_all[-80:, :])
                elif GANname == "BigGAN":
                    np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)),
                        generations=generations, scores_all=scores_all, scores_dyn_all=scores_dyn_all, codes_all=codes_all)
                else:
                    raise ValueError("Unknown GAN model")
                figh = visualize_trajectory(scores_all, generations, codes_arr=codes_all, title_str=methodlab, show=False)
                figh.savefig(join(savedir, "traj%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())), )
                plt.close("all")

        