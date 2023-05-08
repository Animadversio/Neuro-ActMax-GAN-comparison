""" Local version of BigGAN Evol """
import sys
sys.path.append(r"/home/biw905/Github/Neuro-ActMax-GAN-comparison")
import os
import tqdm
import numpy as np
from os.path import join
import matplotlib.pylab as plt
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample, one_hot_from_names, save_as_images)
from core.utils.CNN_scorers import TorchScorer
from core.utils.GAN_utils import BigGAN_wrapper, upconvGAN, loadBigGAN
from core.utils.grad_RF_estim import grad_RF_estimate, gradmap2RF_square
from core.utils.layer_hook_utils import get_module_names, layername_dict, register_hook_by_module_names
from core.utils.Optimizers import CholeskyCMAES, HessCMAES, ZOHA_Sphere_lr_euclid
from core.utils.plot_utils import saveallforms, save_imgrid
#%%
if sys.platform == "linux":
    # rootdir = r"/scratch/binxu/BigGAN_Optim_Tune_new"
    # Hdir_BigGAN = r"/scratch/binxu/GAN_hessian/BigGAN/summary/H_avg_1000cls.npz"
    # Hdir_fc6 = r"/scratch/binxu/GAN_hessian/FC6GAN/summary/Evolution_Avg_Hess.npz"
    # O2 path interface
    scratchdir = "/n/scratch3/users/b/biw905"  # os.environ['SCRATCH1']
    # rootdir = join(scratchdir, "GAN_Evol_cmp")
    rootdir = join(scratchdir, "GAN_Evol_Dissection")
    Hdir_BigGAN = join("/home/biw905/Hessian", "H_avg_1000cls.npz")  #r"/scratch/binxu/GAN_hessian/BigGAN/summary/H_avg_1000cls.npz"
    Hdir_fc6 = join("/home/biw905/Hessian", "Evolution_Avg_Hess.npz")  #r"/scratch/binxu/GAN_hessian/FC6GAN/summary/Evolution_Avg_Hess.npz"
else:
    rootdir = r"F:\insilico_exps\GAN_Evol_Dissection"
    Hdir_BigGAN = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN\H_avg_1000cls.npz"
    Hdir_fc6 = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\fc6GAN\Evolution_Avg_Hess.npz"


from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--net", type=str, default="alexnet", help="Network model to use for Image distance computation")
parser.add_argument("--layer", type=str, default="fc6", help="Network model to use for Image distance computation")
parser.add_argument("--chans", type=int, nargs='+', default=[0, 25], help="")
parser.add_argument("--G", type=str, default="BigGAN", help="")
parser.add_argument("--optim", type=str, nargs='+', default=["HessCMA", "HessCMA_class", "CholCMA", "CholCMA_prod", "CholCMA_class"], help="")
parser.add_argument("--steps", type=int, default=100, help="")
parser.add_argument("--reps", type=int, default=2, help="")
parser.add_argument("--RFresize", type=bool, default=False, help="")
args = parser.parse_args() # ["--G", "BigGAN", "--optim", "HessCMA", "CholCMA","--chans",'1','2','--steps','100',"--reps",'2']
#%%
"""with a correct cmaes or initialization, BigGAN can match FC6 activation."""

#%% Select GAN
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


def get_center_pos_and_rf(model, layer, input_size=(3, 227, 227), device="cuda"):
    module_names, module_types, module_spec = get_module_names(model, input_size=input_size, device=device)
    layer_key = [k for k, v in module_names.items() if v == layer][0]
    feat_outshape = module_spec[layer_key]['outshape']
    # assert len(feat_outshape) == 3  # fc layer will fail
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

#%% Wrap the optimizers, like concatenate 2 or concatenate one with a fixed code
class concat_wrapper:
    """ Concatenate 2 gradient free optimizers
    each optimize a different part of the latent code
    latent code space dim = self.optim1.space_dim + self.optim2.space_dim

    optim1: gradient free optimizer
    optim2: gradient free optimizer
    """
    def __init__(self, optim1, optim2):
        self.optim1 = optim1
        self.optim2 = optim2
        self.sep = self.optim1.space_dimen
        self.space_dimen = self.optim1.space_dimen + self.optim2.space_dimen

    def step_simple(self, scores, codes):
        new_codes1 = self.optim1.step_simple(scores, codes[:,:self.sep])
        new_codes2 = self.optim2.step_simple(scores, codes[:,self.sep:])
        return np.concatenate((new_codes1, new_codes2), axis=1)


class fix_param_wrapper:
    """ Fixe part of parameters, and optimize the rest
    latent code space dim = self.optim.space_dim + self.fix_code.shape[1]

    optim: gradient free optimizer
    fix_code: fixed part of the latent code
    pre: whether fix_code is before or after the optimizable part
    """
    def __init__(self, optim, fixed_code, pre=True):
        self.optim = optim
        self.fix_code = fixed_code
        self.pre = pre  # if the fix code is in the first part
        self.sep = fixed_code.shape[1]
        self.space_dimen = self.optim.space_dimen + self.sep

    def step_simple(self, scores, codes):
        if self.pre:
            """fix the first part of code, optimize the latter part"""
            new_codes1 = self.optim.step_simple(scores, codes[:, self.sep:])
            freezed_codes = np.repeat(self.fix_code, new_codes1.shape[0], axis=0)
            return np.concatenate((freezed_codes, new_codes1), axis=1)
        else:
            """fix the last part of code, optimize the first part"""
            new_codes1 = self.optim.step_simple(scores, codes[:, :-self.sep])
            freezed_codes = np.repeat(self.fix_code, new_codes1.shape[0], axis=0)
            return np.concatenate((new_codes1, freezed_codes), axis=1)


#%% Optimizer from label, Use this to translate string labels to optimizer
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



G = load_GAN(args.G)
Hdata = load_Hessian(args.G)
scorer = TorchScorer(args.net)
#%% Select vision model as scorer
# net = tv.alexnet(pretrained=True)
# scorer.select_unit(("alexnet", "fc6", 2))
# imgs = G.visualize(torch.randn(3, 256).cuda()).cpu()
# scores = scorer.score_tsr(imgs)
#%% Select the Optimizer
method_col = args.optim
# optimizer_col = [label2optimizer(methodlabel, np.random.randn(1, 256), GAN=args.G) for methodlabel in method_col]
#%% Set recording location and image size and position.
pos_dict = {"conv5": (7, 7), "conv4": (7, 7), "conv3": (7, 7), "conv2": (14, 14), "conv1": (28, 28)}
input_size = (3, 227, 227)
cent_pos, corner, imgsize, Xlim, Ylim = get_center_pos_and_rf(scorer.model, args.layer,
                                          input_size=input_size, device="cuda")
print("Target setting network %s layer %s, center pos" % (args.net, args.layer), cent_pos)
print("Xlim %s Ylim %s \n imgsize %s corner %s" % (Xlim, Ylim, imgsize, corner))
#%% Start iterating through channels.
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
                mtg = ToPILImage()(make_grid(imgs, nrow=8))
                mtg.save(join(savedir, "imggen%s_%05d_block%02d.jpg" % (methodlab, RND, i, )))

            codes_all = np.concatenate(tuple(codes_all), axis=0)
            scores_all = np.array(scores_all)
            generations = np.array(generations)
            mtg_exp = ToPILImage()(make_grid(best_imgs, nrow=10))
            mtg_exp.save(join(savedir, "besteachgen%s_%05d.jpg" % (methodlab, RND,)))
            mtg = ToPILImage()(make_grid(imgs, nrow=7))
            mtg.save(join(savedir, "lastgen%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))
            if args.G == "fc6":
                np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)),
                 generations=generations, scores_all=scores_all, codes_fin=codes_all)
            else:
                np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)),
                 generations=generations, scores_all=scores_all, codes_all=codes_all)
            visualize_trajectory(scores_all, generations, codes_arr=codes_all, title_str=methodlab).savefig(
                join(savedir, "traj%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))


