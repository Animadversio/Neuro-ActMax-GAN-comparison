#%%
import sys
sys.path.append("/n/home12/binxuwang/Github/Neuro-ActMax-GAN-comparison")
import torch
import numpy as np
import os
from os.path import join
from tqdm import tqdm, trange
from PIL import Image
import pickle as pkl
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from core.utils.dataset_utils import ImagePathDataset, normalizer, denormalizer
from torchvision import transforms
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs_multiwindow, load_neural_data # load_img_resp_pairs_multiwindow
# from CorrFeatTsr_lib import Corr_Feat_Machine, visualize_cctsr, loadimg_preprocess
# from core.utils.layer_hook_utils import featureFetcher_module
# from core.utils.CNN_scorers import TorchScorer, load_featnet
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
#%%
import platform
if platform.system() == "Windows":
    stim_rootdir = "S:"
    device = "cuda"
elif platform.system() == "Darwin":
    stim_rootdir = "/Users/binxuwang/Network_mapping"
    device = "mps"
elif platform.system() == "Linux":
    stim_rootdir = "/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Datasets"
    device = "cuda"

# load the neural data
BFEStats_merge, BFEStats = load_neural_data()
#%%
rsp_wdws = [range(50, 200), range(0, 50), range(50, 100), range(100, 150), range(150, 200)]

#%%
# Define the response time windows
# rsp_wdws += [range(strt, strt+25) for strt in range(0, 200, 25)]
# get image sequence and response in different time windows
# image paths: (n_images, )
# response: (n_images, n_time)
Expi = 155
Expi = 165
imgfps0, resp_mat0, gen_vec0 = load_img_resp_pairs_multiwindow(BFEStats, Expi,
                         "Evol", thread=0, stimdrive=stim_rootdir,
                        output_fmt="vec", rsp_wdws=rsp_wdws)
imgfps1, resp_mat1, gen_vec1 = load_img_resp_pairs_multiwindow(BFEStats, Expi,
                         "Evol", thread=1, stimdrive=stim_rootdir,
                         output_fmt="vec", rsp_wdws=rsp_wdws)
#%%
embed_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

#%%
def embed_imgs(imgfps, embed_model, batch_size=100, size=224, device="cuda"):
    dataset = ImagePathDataset(imgfps, transform=transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalizer,
    ]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embed_model = embed_model.to(device)
    embed_model.eval()
    embeddings = []
    for batch, _ in tqdm(dataloader):
        with torch.no_grad():
            embed_vec = embed_model(batch.to(device))
        embeddings.append(embed_vec)
    embeddings = torch.cat(embeddings, dim=0).detach().cpu()
    return embeddings, dataset
#%%

embeddings0, dataset0 = embed_imgs(imgfps0, embed_model, batch_size=100, size=224, device=device)
embeddings1, dataset1 = embed_imgs(imgfps1, embed_model, batch_size=100, size=224, device=device)
#%%
BFEStats[1]["meta"]
#%%
embed_root = "/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Projects/BigGAN_img_embedding"
embed_dir = join(embed_root, "dinov2_vitb14")
os.makedirs(embed_dir, exist_ok=True)
for Expi in trange(63, 1 + 190):
    if BFEStats[Expi-1]["evol"] is None:
        print(f"Exp {Expi} is None")
        continue
    try:
        imgfps0, resp_mat0, gen_vec0 = load_img_resp_pairs_multiwindow(BFEStats, Expi,
                            "Evol", thread=0, stimdrive=stim_rootdir,
                            output_fmt="vec", rsp_wdws=rsp_wdws)
        imgfps1, resp_mat1, gen_vec1 = load_img_resp_pairs_multiwindow(BFEStats, Expi,
                            "Evol", thread=1, stimdrive=stim_rootdir,
                            output_fmt="vec", rsp_wdws=rsp_wdws)
    except IndexError or FileNotFoundError:
        print(f"File missing for Exp {Expi}")
        continue
    if len(imgfps0) == 0 or len(imgfps1) == 0:
        continue
    embeddings0, dataset0 = embed_imgs(imgfps0, embed_model, batch_size=100, size=224, device=device)
    embeddings1, dataset1 = embed_imgs(imgfps1, embed_model, batch_size=100, size=224, device=device)
    embed_act_dict = {
        "embeddings0": embeddings0,
        "embeddings1": embeddings1,
        "imgfps0": imgfps0,
        "imgfps1": imgfps1,
        "resp_mat0": resp_mat0,
        "resp_mat1": resp_mat1,
        "gen_vec0": gen_vec0,
        "gen_vec1": gen_vec1,
    }
    pkl.dump(embed_act_dict, open(join(embed_dir, f"Exp{Expi:03d}_embed_act_data.pkl"), "wb"))
    print(f"Exp {Expi} done. saved to {embed_dir}")

#%%

embed_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
#%%
from core.utils.dataset_utils import create_imagenet_valid_dataset
rootdir = r"/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Datasets/imagenet-valid"
INdataset = create_imagenet_valid_dataset(imgpix=224,rootdir=rootdir)
#%%
# show the image
img, score = INdataset[1]
plt.imshow(denormalizer(img).permute(1, 2, 0))
plt.title(score)
#%%
dataloaders = DataLoader(INdataset, batch_size=256, shuffle=False, num_workers=8)
#%%
embed_model.cuda().eval()
embedding_col = []
for batch, _ in tqdm(dataloaders):
    with torch.no_grad():
        embed_vec = embed_model(batch.cuda())
    embedding_col.append(embed_vec.detach().cpu())

embedding_mat = torch.cat(embedding_col, dim=0)


#%% Playground 
dataset = ImagePathDataset(imgfps0, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalizer,
]))
dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)

#%%
dataset[1]
#%%
embed_model = embed_model.cuda()
embed_model.eval()
embeddings = []
for batch, _ in tqdm(dataloader):
    batch = batch.cuda()
    with torch.no_grad():
        embed_vec = embed_model(batch)
    embeddings.append(embed_vec)
embeddings = torch.cat(embeddings, dim=0)

