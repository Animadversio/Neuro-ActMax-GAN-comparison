""" Cluster version of BigGAN Evol """
import math
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
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms

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
    if not "fc" in layer:
        module_names, module_types, module_spec = get_module_names(model, input_size=input_size, device=device)
        layer_key = [k for k, v in module_names.items() if v == layer][0]
        feat_outshape = module_spec[layer_key]['outshape']
        assert len(feat_outshape) == 3  # fc layer will fail
        cent_pos = (feat_outshape[1]//2, feat_outshape[2]//2)
    else:
        cent_pos = None

    # rf Mapping,
    if not "fc" in layer:
        print("Computing RF by direct backprop: ")
        gradAmpmap = grad_RF_estimate(model, layer, (slice(None), *cent_pos), input_size=input_size,
                                      device=device, show=False, reps=30, batch=1)
        Xlim, Ylim = gradmap2RF_square(gradAmpmap, absthresh=1E-8, relthresh=0.01, square=True)
        corner = (Xlim[0], Ylim[0])
        imgsize = (Xlim[1] - Xlim[0], Ylim[1] - Ylim[0])
    else:
        imgsize = input_size[-2:]
        corner = (0, 0)
        Xlim = (corner[0], corner[0] + imgsize[0])
        Ylim = (corner[1], corner[1] + imgsize[1])

    return cent_pos, corner, imgsize, Xlim, Ylim
#%%
if sys.platform == "linux":
    # rootdir = r"/scratch/binxu/BigGAN_Optim_Tune_new"
    # Hdir_BigGAN = r"/scratch/binxu/GAN_hessian/BigGAN/summary/H_avg_1000cls.npz"
    # Hdir_fc6 = r"/scratch/binxu/GAN_hessian/FC6GAN/summary/Evolution_Avg_Hess.npz"
    # O2 path interface
    scratchdir = "/n/scratch3/users/b/biw905"  # os.environ['SCRATCH1']
    rootdir = join(scratchdir, "GAN_gradEvol_cmp")
    Hdir_BigGAN = join("/home/biw905/Hessian", "H_avg_1000cls.npz")  #r"/scratch/binxu/GAN_hessian/BigGAN/summary/H_avg_1000cls.npz"
    Hdir_fc6 = join("/home/biw905/Hessian", "Evolution_Avg_Hess.npz")  #r"/scratch/binxu/GAN_hessian/FC6GAN/summary/Evolution_Avg_Hess.npz"
    import matplotlib as mpl
    mpl.use('agg')
else:
    # rootdir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_Optim_Tune_tmp"
    # rootdir = r"D:\Cluster_Backup\GAN_gradEvol_cmp" #r"E:\Monkey_Data\BigGAN_Optim_Tune_tmp"
    rootdir = r"F:\insilico_exps\GAN_gradEvol_cmp" #r"E:\Monkey_Data\BigGAN_Optim_Tune_tmp"
    Hdir_BigGAN = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN\H_avg_1000cls.npz"
    Hdir_fc6 = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\fc6GAN\Evolution_Avg_Hess.npz"



#%%
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--net", type=str, default="alexnet", help="Network model to use for Image distance computation")
parser.add_argument("--layer", type=str, default="fc6", help="Network model to use for Image distance computation")
parser.add_argument("--chans", type=int, nargs='+', default=[0, 25], help="")
parser.add_argument("--G", type=str, default="BigGAN", help="")
parser.add_argument("--optim", type=str, nargs='+', default=["Adam001Hess", "Adam0003Hess", "Adam0001"], help="")
parser.add_argument("--steps", type=int, default=100, help="")
parser.add_argument("--reps", type=int, default=10, help="")
parser.add_argument("--batch", type=int, default=4, help="")
parser.add_argument("--RFresize", type=bool, default=False, help="")
args = parser.parse_args() # ["--G", "BigGAN", "--optim", "HessCMA", "CholCMA","--chans",'1','2','--steps','100',"--reps",'2']

# from easydict import EasyDict as edict
# args = edict()
# args.net = "resnet50"
# args.layer = ".layer3"
# args.G = "BigGAN"
# args.batch = 4
# args.steps = 100
# args.reps = 2
# args.optim = ["Adam001", "Adam001Hess", "Adam0003", "Adam0003Hess",
#               "Adam0001", "Adam0001Hess", "SGD001", "SGD001Hess",
#               "SGD0003", "SGD0003Hess", "SGD0001", "SGD0001Hess", ]
# args.RFresize = True
#%%
def resize_and_pad_canvas(imgs, corner, size, input_size):
    """ Resize and pad images to a square with a given corner and size
    Background is gray.
    Assume image is float, range [0, 1]
    """  # FIXED: this should depend on the input size of image
    pad_img = torch.ones((imgs.shape[0], *input_size), dtype=imgs.dtype, device=imgs.device) * 0.5
    rsz_img = F.interpolate(imgs, size=size, align_corners=True, mode="bilinear")
    pad_img[:, :, corner[0]:corner[0]+size[0], corner[1]:corner[1]+size[1]] += rsz_img - 0.5
    return pad_img


def grad_evolution(scorer, optim_constructor, z_init, hess_param:bool, evc=None, steps=100,
                   RFresize=False, corner=None, imgsize=None):
    if RFresize:
        assert corner is not None and imgsize is not None
    if hess_param:
        assert evc is not None
        evc = evc.cuda()

    z_init = z_init.cuda()
    w = z_init @ evc if hess_param else z_init.detach().clone()
    w.requires_grad_(True)
    # optimizer = optimCls([w], **optimCfg)
    optimizer = optim_constructor([w])
    score_traj = []
    z_traj = []
    img_traj = []
    for i in range(steps):
        optimizer.zero_grad()
        z = (w @ evc.t()) if hess_param else w
        img = G.visualize(z)
        if RFresize:
            img = resize_and_pad_canvas(img, corner, imgsize, scorer.inputsize)
        score = scorer.score_tsr_wgrad(img)
        score_traj.append(score.detach().cpu())
        z_traj.append(z.detach().cpu())
        loss = - score.sum()
        loss.backward()
        optimizer.step()
        zero_mask = (score == 0)
        if zero_mask.sum() > 0:
            new_z = G.sample_vector(zero_mask.sum())
            w.data[zero_mask] = new_z @ evc if hess_param else new_z
        print("  ".join(["%.2f" % s for s in score.detach()]))
        img_traj.append(img.detach().cpu().clone())

    idx = torch.argsort(score.detach().cpu(), descending=True)
    score_traj = torch.stack(score_traj)
    z_traj = torch.stack(z_traj)
    img = img.detach()[idx]
    img_traj = torch.stack(img_traj)[:, idx, :, :, :]
    z_traj = z_traj[:, idx, :]  # sort the sample
    score_traj = score_traj[:, idx]  # sort the sample
    return img, img_traj, z_traj, score_traj


def visualize_gradevol(score_traj, z_traj, savedir, savestr="", titlestr=""):
    if z_traj.shape[-1] == 256:
        noise_norm = z_traj[:, :, :128].norm(dim=-1)
        class_norm = z_traj[:, :, 128:].norm(dim=-1)
        figh, axs = plt.subplots(1, 3, figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.plot(score_traj)
        plt.title("score traj")
        plt.subplot(1, 3, 2)
        plt.plot(noise_norm)
        plt.title("noise norm")
        plt.subplot(1, 3, 3)
        plt.plot(class_norm)
        plt.title("class norm")
    elif z_traj.shape[-1] == 4096:
        noise_norm = z_traj.norm(dim=-1)
        figh, axs = plt.subplots(1, 2, figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(score_traj)
        plt.title("score traj")
        plt.subplot(1, 2, 2)
        plt.plot(noise_norm)
        plt.title("noise norm")
    plt.suptitle(titlestr)
    plt.tight_layout()
    saveallforms(savedir, savestr + "score_traj", figh, ["png", "pdf"])
    plt.show()


def get_optimizer_constructor(optim_name):
    if optim_name.endswith("Hess"):
        hess_param = True
    else:
        hess_param = False
    if optim_name in ["Adam01", "Adam01Hess"]:
        def optim_constructor(params):
            return Adam(params, lr=0.1)
    elif optim_name in ["Adam001", "Adam001Hess"]:
        def optim_constructor(params):
            return Adam(params, lr=0.01)
    elif optim_name in ["Adam0003", "Adam0003Hess"]:
        def optim_constructor(params):
            return Adam(params, lr=0.003)
    elif optim_name in ["Adam0001", "Adam0001Hess"]:
        def optim_constructor(params):
            return Adam(params, lr=0.001)
    elif optim_name in ["SGD001", "SGD001Hess"]:
        def optim_constructor(params):
            return SGD(params, lr=0.01)
    elif optim_name in ["SGD0003", "SGD0003Hess"]:
        def optim_constructor(params):
            return SGD(params, lr=0.003)
    elif optim_name in ["SGD0001", "SGD0001Hess"]:
        def optim_constructor(params):
            return SGD(params, lr=0.001)
    else:
        raise ValueError("Unknown optimizer %s" % optim_name)

    return optim_constructor, hess_param
#%%
#%%
from torch.optim import SGD, Adam
input_size = (3, 227, 227)
G = load_GAN(args.G)
Hdata = load_Hessian(args.G)
scorer = TorchScorer(args.net, imgpix=227)
if args.G == "BigGAN":
    evc = torch.tensor(Hdata["eigvects_avg"]).cuda()
elif args.G == "fc6":
    evc = torch.tensor(Hdata["eigvect_avg"]).float().cuda()

cent_pos, corner, imgsize, Xlim, Ylim = get_center_pos_and_rf(scorer.model, args.layer,
                                          input_size=input_size, device="cuda")
print("Target setting network %s layer %s, center pos" % (args.net, args.layer), cent_pos)
print("Xlim %s Ylim %s \n imgsize %s corner %s" % (Xlim, Ylim, imgsize, corner))
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
    for repi in range(args.reps):
        z_init = G.sample_vector(args.batch).cuda()
        RND = np.random.randint(1E5)
        np.save(join(savedir, "init_code_%05d.npy"%RND), z_init.cpu().numpy())
        for methodlab in args.optim: # "Adam0001", "Adam0001Hess","SGD0001", "SGD0001Hess"]:
            optim_constructor, hess_param = get_optimizer_constructor(methodlab)
            if args.G == "fc6":
                methodlab += "_fc6"
            img, img_traj, z_traj, score_traj = grad_evolution(scorer, optim_constructor, z_init,
                                                               hess_param=hess_param, evc=evc, steps=args.steps,
                                                               RFresize=args.RFresize, corner=corner, imgsize=imgsize)
            visualize_gradevol(score_traj, z_traj, savedir, savestr="traj%s_%05d"%(methodlab, RND, ),
                               titlestr=f"{unit} - {methodlab}\nRND{RND}, rep{repi}")
            save_imgrid(img, join(savedir, "imglastgen%s_%05d.jpg" % (methodlab, RND, )), nrow=5)
            for tri in range(z_init.shape[0]):
                img_traj_trial = img_traj[:, tri, ]  # (steps, 3, 227, 227)
                save_imgrid(img_traj_trial, join(savedir, "imgtraj%s_%05d_%d.jpg" % (methodlab, RND, tri, )), nrow=10)
            torch.save({"z_traj": z_traj, "score_traj": score_traj,},
                       join(savedir, "optimdata_%s_%05d.pt"%(methodlab, RND)))

    scorer.cleanup()
#%%





