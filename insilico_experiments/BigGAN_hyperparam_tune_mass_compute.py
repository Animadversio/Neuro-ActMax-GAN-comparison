import os
import pickle as pkl
import time
import torch
import torch.nn as nn
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from circuit_toolkit.layer_hook_utils import featureFetcher, featureFetcher_module
from circuit_toolkit.dataset_utils import create_imagenet_valid_dataset
from circuit_toolkit.plot_utils import to_imgrid, show_imgrid, saveallforms, save_imgrid
from circuit_toolkit.GAN_utils import upconvGAN, Caffenet, RGB_mean
from circuit_toolkit.Optimizers import CholeskyCMAES, CholeskyCMAES_torch, CholeskyCMAES_torch_noCMA
from circuit_toolkit.CNN_scorers import TorchScorer, resize_and_pad, resize_and_pad_tsr
import torchvision

RGB_mean = torch.tensor([0.485, 0.456, 0.406]) #.view(1,-1,1,1).cuda()
RGB_std  = torch.tensor([0.229, 0.224, 0.225]) #.view(1,-1,1,1).cuda()
IN_transform = torchvision.transforms.Compose([
                                            # torchvision.transforms.Resize(256, ),
                                            # torchvision.transforms.CenterCrop((256, 256), ),
                                            torchvision.transforms.Normalize(RGB_mean, RGB_std)])


def optimize_gan_codes(G, CNN, fetcher, unit_idx, imgsize=(256, 256), corner=(0, 0), 
                      total_steps=100, print_freq=10, RFresize=True,
                      init_code_std=0.01, init_sigma=0.06, optimizer=None):
    """
    Optimize GAN codes to maximize activation of a specific CNN unit
    
    Args:
        G: GAN model wrapper with visualize() method
        CNN: CNN model to optimize against
        unit_idx: Index of unit to optimize
        layerkey: Layer name to extract features from
        imgsize: Size to resize images to
        corner: Corner position for resizing
        total_steps: Number of optimization steps
        print_freq: How often to print progress
        RFresize: Whether to resize images
        init_code_std: Standard deviation for initial codes
        init_sigma: Initial sigma for optimizer
        optimizer: Optional pre-configured optimizer
        
    Returns:
        dict containing optimization results
    """
    code_len = G.codelen  # Fixed for BigGAN
    latent_shape = G.latent_shape
    
    assert not np.isnan(init_sigma)
    if optimizer is None:
        optimizer = CholeskyCMAES_torch_noCMA(code_len, init_sigma=init_sigma, 
                                            Aupdate_freq=1000, device='cuda')
    
    new_codes = init_code_std * torch.randn(1, code_len, device='cuda')
    scores_all = []
    generations = []
    codes_all = []
    best_imgs = []
    
    with torch.no_grad():
        for i in range(total_steps,):
            codes_all.append(new_codes.cpu().numpy())
            latent_code = new_codes.view(-1, *latent_shape)
            imgs = G.visualize(latent_code)
            
            if RFresize:
                imgs = resize_and_pad_tsr(imgs, imgsize, corner, canvas_size=imgsize)
                
            imgs_pp = IN_transform(imgs)
            CNN.model(imgs_pp)
            activations = fetcher["score"]
            
            if activations.ndim == 2:
                scores = activations[:, unit_idx]
            elif activations.ndim == 4:
                center_idx = tuple(dim // 2 for dim in activations.shape[-2:])
                scores = activations[:, unit_idx, center_idx[0], center_idx[1]]
            else:
                raise ValueError(f"Unsupported activation dimension: {activations.ndim}")
                
            if i % print_freq == 0 or i == total_steps - 1:
                print("step %d score %.3f (%.3f) (norm %.2f )" % (
                    i, scores.mean().cpu(), scores.std().cpu(), 
                    latent_code.view(-1, code_len).norm(dim=1).mean().cpu(),))
                    
            new_codes = optimizer.step_simple(scores, new_codes, verbosity=False)
            scores_all.extend(list(scores.cpu().numpy()))
            generations.extend([i] * len(scores))
            best_imgs.append(imgs[scores.argmax(),:,:,:].cpu())
    
    scores_all = np.array(scores_all)
    generations = np.array(generations)
    codes_all = np.concatenate(codes_all, axis=0)
    return {
        'scores': scores_all,
        'generations': generations, 
        'codes': codes_all,
        'best_imgs': best_imgs
    }
    

def visualize_best_images_traj(results):
    # Visualize best images grid
    mtg = to_imgrid(results['best_imgs'], nrow=10)
    figh = plt.figure(figsize=(10, 10))
    plt.imshow(mtg)
    plt.axis('off')
    plt.show()
    return figh, mtg


def plot_evolution_traj(results):
    # Plot evolution trajectory
    figh = plt.figure(figsize=(6, 6))
    plt.scatter(results['generations'], results['scores'], s=25, alpha=0.5)
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.show()
    return figh


class BigGAN_EcoSet_Wrapper(nn.Module):
    def __init__(self, G, ):
        super().__init__()
        self.G = G
        self.latent_shape = (140 + 128, )
        self.codelen = 140 + 128
    
    def visualize(self, latent_codes):
        ys = latent_codes[:, :128]
        zs = latent_codes[:, 128:]
        imgs = self.G.forward(zs, ys)
        imgs = (imgs + 1) / 2
        return imgs


import sys
# sys.path.append("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/biggan-pytorch-ecoset/code")
# from BigGAN_nodist import Generator
from circuit_toolkit.GAN_utils import upconvGAN, BigGAN_wrapper
from pytorch_pretrained_biggan import BigGAN

# BGEco_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/biggan-pytorch-ecoset"
# suffix = "best2"
# config = torch.load(join(BGEco_root, "weights", f"state_dict_{suffix}.pth"))['config']
# weights_dict = torch.load(join(BGEco_root, "weights", f"G_{suffix}.pth"))
# G = Generator(**config)
# G.load_state_dict(weights_dict, strict=True)
# G.to("cuda").eval()
# G.requires_grad_(False);
# G_Eco = BigGAN_EcoSet_Wrapper(G,)

DP_G = upconvGAN("fc6")
DP_G.to("cuda").eval()
DP_G.requires_grad_(False);

BG = BigGAN.from_pretrained("biggan-deep-256")
BG.to("cuda").eval()
BG.requires_grad_(False);
G_IN = BigGAN_wrapper(BG)


saveroot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/BigGAN_hyperparam_tune"
savedir = join(saveroot, "alexnet")
os.makedirs(savedir, exist_ok=True)
CNN = TorchScorer("alexnet", )
for rep in range(3):
    for target_module, layername in [
        (CNN.model.classifier[2], "fc6"), 
        (CNN.model.classifier[5], "fc7"),
        (CNN.model.features[11], "conv5"),
        (CNN.model.features[8],  "conv4"),
        # (CNN.model.features[5], "conv3"),
        # (CNN.model.features[2], "conv2"),
        # (CNN.model.features[0], "conv1"),

    ]:
        fetcher = featureFetcher_module()
        fetcher.record_module(target_module, target_name="score")
        for unit_i in range(10):
            init_sigma = 3.0
            T0 = time.time()
            Evol_results_INet = optimize_gan_codes(DP_G, CNN, fetcher, unit_idx=unit_i, init_code_std=0.01, init_sigma=init_sigma, print_freq=50)
            T1 = time.time()
            print(f"DP x alexnet {layername} ch{unit_i} rep{rep} CMA init sigma {init_sigma}: Act {Evol_results_INet['scores'][-25:].mean():.2f} time {T1-T0:.2f} sec")
            pkl.dump(Evol_results_INet, open(join(savedir, f"Evol_results_DeePSim_fc6_INet_sigma{init_sigma}_resnet50_linf8_{layername}_ch{unit_i}_rep{rep}.pkl"), "wb"))
            for init_sigma in [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 3.0]:
                T0 = time.time()
                Evol_results_INet = optimize_gan_codes(G_IN, CNN, fetcher, unit_idx=unit_i, init_code_std=0.01, init_sigma=init_sigma, print_freq=50)
                T1 = time.time()
                print(f"INet x alexnet {layername} ch{unit_i} rep{rep} CMA init sigma {init_sigma}: Act {Evol_results_INet['scores'][-25:].mean():.2f} time {T1-T0:.2f} sec")
                # Evol_results_eco = optimize_gan_codes(G_Eco, CNN, fetcher, unit_idx=unit_i, init_code_std=0.01, init_sigma=0.06, print_freq=100)
                # T2 = time.time()
                # print(f"EcoSet x ResNet50-linf8 {layername} ch{unit_i} rep{rep} : Act {Evol_results_eco['scores'][-25:].mean():.2f} time {T2-T1:.2f} sec")
                # plot_evolution_traj(Evol_results_INet)
                # plot_evolution_traj(Evol_results_eco)
                pkl.dump(Evol_results_INet, open(join(savedir, f"Evol_results_BigGAN_INet_sigma{init_sigma}_resnet50_linf8_{layername}_ch{unit_i}_rep{rep}.pkl"), "wb"))
                # pkl.dump(Evol_results_eco, open(join(savedir, f"Evol_results_BigGAN_EcoSet_resnet50_linf8_{layername}_ch{unit_i}_rep{rep}.pkl"), "wb"))
                print("")
        fetcher.cleanup()
        del fetcher
