import torch
import numpy as np
import torch as th
import pickle as pkl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from IPython.display import display
from circuit_toolkit.dataset_utils import create_imagenet_valid_dataset
from circuit_toolkit.plot_utils import to_imgrid, show_imgrid, saveallforms, save_imgrid
from circuit_toolkit.GAN_utils import upconvGAN, Caffenet,RGB_mean
from circuit_toolkit.Optimizers import CholeskyCMAES, CholeskyCMAES_torch, CholeskyCMAES_torch_noCMA
from circuit_toolkit.CNN_scorers import TorchScorer, resize_and_pad, resize_and_pad_tsr
from circuit_toolkit.layer_hook_utils import featureFetcher, featureFetcher_module

CNN = Caffenet(pretrained=True)
CNN = CNN.cuda().eval()
CNN.requires_grad_(False)
G_dict = {}
for layer in ["norm1", "norm2", "conv3", "conv4", "pool5", "fc6", "fc6_eucl", "fc7", "fc8"]:
    G_dict[layer] = upconvGAN(name=layer, pretrained=True).cuda().eval()
    G_dict[layer].requires_grad_(False)
    
invers_layer_map = {
    "norm1": 3,
    "norm2": 7,
    "conv3": 9,
    "conv4": 11,
    "pool5": 14,
    "fc6": 17,
    "fc6_eucl": 17,
    "fc7": 19,
    "fc8": 20,
}
savedir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/ReprInvertNet"
evolsavedir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/ReprInvertNet_Evol"
total_steps = 100
print_freq = 100
RFresize = True
corner = (0, 0)
imgsize = (224, 224)
std_scaling_factor = 1.0

with open(f"{savedir}/caffenet_activ_stats.pkl", "rb") as f:
    stats = pkl.load(f)
    activ_mean, activ_std = stats["mean"], stats["std"]

results_collection = []
for unit_idx in range(50, 100):
    for layerkey in ["fc8", "relu7", "relu6", "pool5", "relu4", "relu3", "norm2", "norm1"]:
        for GAN_key in ["fc8", "fc7", "fc6", "pool5", "conv4", "conv3", "norm2", "norm1"]:
            print(f"CNN layer: {layerkey}, GAN layer: {GAN_key}")
            G = G_dict[GAN_key]
            fetcher = featureFetcher_module()
            fetcher.record_module(CNN.net.__getattr__(layerkey), target_name=layerkey)
            code_len = np.prod(G.latent_shape)
            # use the mean of the layer activation as initialization
            new_codes = activ_mean[GAN_key][None, ...].flatten(start_dim=1) ##[unit_idx]
            init_sigma = activ_std[GAN_key].mean().item() * std_scaling_factor
            assert not np.isnan(init_sigma)
            optimizer = CholeskyCMAES_torch_noCMA(code_len, init_sigma=init_sigma, Aupdate_freq=1000, device='cuda')
            # new_codes = torch.randn(1, code_len, device='cuda') #np.random.randn(1, code_len)
            scores_all = []
            generations = []
            codes_all = []
            best_imgs = []
            with torch.no_grad():
                for i in range(total_steps,):
                    codes_all.append(new_codes.cpu().numpy())
                    latent_code = new_codes.view(-1, *G.latent_shape)
                    imgs = G.visualize(latent_code)#.cuda().cpu()
                    if RFresize:
                        imgs = resize_and_pad_tsr(imgs, imgsize, corner, canvas_size=imgsize, )  #  Bug: imgs are resized to 256x256 and it will be further resized in score_tsr
                    CNN(imgs, preproc=True)
                    activations = fetcher[layerkey]
                    if activations.ndim == 2:
                        scores = activations[:, unit_idx] #.cpu().numpy() #scorer.score_tsr(imgs)
                    elif activations.ndim == 4:
                        center_idx = tuple(dim // 2 for dim in activations.shape[-2:])
                        scores = activations[:, unit_idx, center_idx[0], center_idx[1]]
                    else:
                        raise ValueError(f"Unsupported activation dimension: {activations.ndim}")
                    if i % print_freq == 0 or i == total_steps - 1:
                        print("step %d score %.3f (%.3f) (norm %.2f )" % (
                            i, scores.mean().cpu(), scores.std().cpu(), latent_code.view(-1, code_len).norm(dim=1).mean().cpu(),))
                    new_codes = optimizer.step_simple(scores, new_codes, verbosity=False)
                    scores_all.extend(list(scores.cpu().numpy()))
                    generations.extend([i] * len(scores))
                    best_imgs.append(imgs[scores.argmax(),:,:,:].cpu())
            print(f"CNN {layerkey}-unit {unit_idx} | GAN {GAN_key}: last gen avg score: {scores.cpu().mean()}")
            fetcher.cleanup()
            results_collection.append({
                "CNN": layerkey,
                "GAN": GAN_key,
                "unit_idx": unit_idx,
                "final_score": scores.cpu().mean().item(),
                "final_best_img": best_imgs[-1].cpu(),
                "scores": scores_all,
                "generations": generations,
                # "best_imgs": best_imgs,
            })
            del optimizer
            del fetcher

pkl.dump(results_collection, open(f"{evolsavedir}/results_collection_noCMA_pilot6_preproc_optimtuned_std{std_scaling_factor}.pkl", "wb"))

df = pd.DataFrame(results_collection)
df.head()
df["final_score_"] = df["final_score"]#.map(lambda x:x.item())
# Create a figure with 6 subplots, one for each CNN layer
cnn_layers = df["CNN"].unique()
fig, axes = plt.subplots(1, len(cnn_layers), figsize=(20, 4.5))
axes = axes.flatten()
# Get unique CNN layers
# Create a subplot for each CNN layer
for i, cnn_layer in enumerate(cnn_layers):
    # Filter data for this CNN layer
    layer_data = df[df["CNN"] == cnn_layer]
    # Create swarm plot
    # sns.stripplot(x="GAN", y="final_score_", data=layer_data, ax=axes[i], dodge=True, hue="GAN", alpha=0.5)
    sns.barplot(x="GAN", y="final_score_", data=layer_data, ax=axes[i], hue="GAN",  )
    axes[i].set_title(f'CNN Layer: {cnn_layer}')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
saveallforms(evolsavedir, f"caffenet_invert_alignment_exp_pilot6_preproc_optimtuned_std{std_scaling_factor}_alllayers_barplot")
plt.show()

# Create a figure with 6 subplots, one for each CNN layer
cnn_layers = df["CNN"].unique()
fig, axes = plt.subplots(1, len(cnn_layers), figsize=(16, 4.5))
axes = axes.flatten()
# Get unique CNN layers
# Create a subplot for each CNN layer
for i, cnn_layer in enumerate(cnn_layers):
    # Filter data for this CNN layer
    layer_data = df[df["CNN"] == cnn_layer]
    # Create swarm plot
    sns.stripplot(x="GAN", y="final_score_", data=layer_data, ax=axes[i], dodge=True, hue="GAN", alpha=0.5)
    # sns.barplot(x="GAN", y="final_score_", data=layer_data, ax=axes[i], hue="GAN",  )
    axes[i].set_title(f'CNN Layer: {cnn_layer}')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
saveallforms(evolsavedir, f"caffenet_invert_alignment_exp_pilot6_preproc_optimtuned_std{std_scaling_factor}_alllayers_stripplot")
plt.show()
