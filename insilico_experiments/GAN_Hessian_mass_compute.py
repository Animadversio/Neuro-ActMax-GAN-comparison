import sys
sys.path.append("/n/home12/binxuwang/Github/GAN-Geometry")
from core import get_full_hessian, hessian_compute, save_imgrid, show_imgrid, plot_spectra
import pickle as pkl
from os.path import join
import torch
import numpy as np
import matplotlib.pylab as plt
from tqdm.auto import tqdm, trange
import argparse
# from core import get_full_hessian, hessian_compute, save_imgrid, show_imgrid, plot_spectra
from circuit_toolkit.GAN_utils import BigGAN_wrapper, loadBigGAN, upconvGAN
from pytorch_pretrained_biggan import BigGAN




# Set up command line argument parser
parser = argparse.ArgumentParser(description='Compute Hessian for range of random seeds')
parser.add_argument('--start', type=int, default=0, help='Starting random seed')
parser.add_argument('--end', type=int, default=100, help='Ending random seed (exclusive)')
parser.add_argument('--dist', type=str, default="MSE", help='Distance function')
parser.add_argument('--GAN', type=str, default="fc6", help='GAN model name')
args = parser.parse_args()

# L2 / MSE
def MSE(im1, im2):
    return (im1 - im2).pow(2).mean(dim=[1,2,3])


if args.dist == "MSE":
    dist_func = MSE
elif args.dist == "SSIM":
    import pytorch_msssim
    
    D = pytorch_msssim.SSIM()  # note SSIM, higher the score the more similar they are. So to confirm to the distance convention, we use 1 - SSIM as a proxy to distance.
    
    def SSIM_Dist(im1, im2):
        return 1 - D(im1, im2)
    
    dist_func = SSIM_Dist
elif args.dist == "LPIPS":
    import lpips
    
    ImDist = lpips.LPIPS(net="squeeze").cuda()
    dist_func = ImDist
else:
    raise ValueError(f"Distance function {args.dist} not supported")

if args.GAN == "fc6":
    UC_G = upconvGAN("fc6")
elif args.GAN == "fc7":
    UC_G = upconvGAN("fc7")
else:
    raise ValueError(f"GAN model {args.GAN} not supported")

UC_G.cuda().eval()
UC_G.requires_grad_(False)

savedir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/GAN_Hessian/DeePSim_FC6"
for rnd in trange(args.start, args.end):
    refvec = UC_G.sample_vector(device="cuda", seed=rnd)
    eigvals_MSE, eigvects_MSE, H_MSE = hessian_compute(UC_G, refvec, dist_func, hessian_method="BP", symmetrize=True) # 28 sec
    print("num of small eigvals: ", np.sum(eigvals_MSE < 1E-8))
    pkl.dump({f"eigvals_{args.dist}": eigvals_MSE, f"eigvects_{args.dist}": eigvects_MSE, f"H_{args.dist}": H_MSE, "vector": refvec, "seed": rnd}, 
             open(join(savedir, f"Hessian_{args.dist}_rnd{rnd}.pkl"), "wb"))
