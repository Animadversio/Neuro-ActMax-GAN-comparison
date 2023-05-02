import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
from os.path import join
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.functional import interpolate
# set path according to user name
# if on windows machine
import platform
if platform.system() == 'Windows':
    if os.environ['COMPUTERNAME'] == 'PONCELAB-OFF6':
        ckpt_root = r"D:\Github\segment-anything\ckpts"
        protomtg_dir = r"F:\insilico_exps\GAN_Evol_cmp\protoimgs_merge"
        outdir = r"F:\insilico_exps\GAN_Evol_cmp\SAM_embeds"
        maskdir = r"F:\insilico_exps\GAN_Evol_cmp\SAM_masks"
elif platform.system() == 'Linux':
    # TODO: check it's on Odin
    ckpt_root = r"/home/binxu/Github/segment-anything/ckpts"
    protomtg_dir = r"/home/binxu/Datasets/GAN_Evol_cmp/protoimgs_merge"
    outdir = r"/home/binxu/Datasets/GAN_Evol_cmp/SAM_embeds"
    maskdir = r"/home/binxu/Datasets/GAN_Evol_cmp/SAM_masks"

os.makedirs(outdir, exist_ok=True)
os.makedirs(maskdir, exist_ok=True)
#%%
sam_checkpoint = join(ckpt_root, "sam_vit_h_4b8939.pth") #"sam_vit_l_0b3195.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)
predictor = SamPredictor(sam, )
#%%
from pathlib import Path
protomtgs = list(Path(protomtg_dir).glob("*.jpg"))
#%%
import pickle as pkl
for protomtg_path in tqdm(protomtgs[:]):
    mtg = plt.imread(protomtg_path)
    # for ipatch, patch_slice in enumerate([slice(1548, 2324)]): # slice(0, 776), slice(774, 1550),  slice(1548, 2324)
    ipatch, patch_slice = 2, slice(1548, 2324)
    image = mtg[:, patch_slice, :]
    predictor.set_image(image)
    # masks = mask_generator.generate(image)
    img_embed = predictor.get_image_embedding()
    torch.save(img_embed.cpu(), join(outdir, f"{protomtg_path.stem}_part{ipatch}.npy"),)
    # pkl.dump(masks, open(join(maskdir, f"{protomtg_path.stem}_part{ipatch}_masks.pkl"), "wb"),)
#%%
from core.utils.montage_utils import crop_all_from_montage
crops = crop_all_from_montage(mtg, totalnum=30, imgsize=256, pad=2, autostop=True)

#%%
img_embeds = []
for crop in tqdm(crops):
    predictor.set_image(crop)
    # masks = mask_generator.generate(image)
    img_embed = predictor.get_image_embedding()
    img_embeds.append(img_embed.cpu())
#%%
