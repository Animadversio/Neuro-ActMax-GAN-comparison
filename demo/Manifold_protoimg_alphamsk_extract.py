# load in mat files
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
matdir = r"E:\OneDrive - Harvard University\Mat_Statistics"

Amat = loadmat(join(matdir, "Alfa_ImageRepr.mat"))['ReprStats']
Bmat = loadmat(join(matdir, "Beto_ImageRepr.mat"))['ReprStats']
#%%
# AStats = loadmat(join(matdir, "Alfa_ImageRepr.mat"))['ReprStats']
# Alfa_Evol_stats.mat
import pandas as pd
metaA = pd.read_csv(join(matdir, "Alfa_EvolTrajStats.csv"))
metaB = pd.read_csv(join(matdir, "Beto_EvolTrajStats.csv"))
meta_df = pd.concat([metaA, metaB], axis=0)
#%%
Amat[0]["Evol"][45]["BestBlockAvgImg"][0][0].shape
#%%
# collect these images into a dict
proto_dict = {}
for i in range(46):
    proto_dict[i] = Amat[0]["Evol"][i]["BestBlockAvgImg"][0][0]

for i in range(45):
    proto_dict[i+46] = Bmat[0]["Evol"][i]["BestBlockAvgImg"][0][0]
#%%
def areamapping(chan):          
    if (chan <= 48 and chan >= 33):
        area = "V1"
    if (chan > 48):
        area = "V4"
    if (chan < 33):
        area = "IT"
    return area
#%%
meta_df["area"] = meta_df["pref_chan"].apply(areamapping)
ITmsk = meta_df["area"] == "IT"
V4msk = meta_df["area"] == "V4"
V1msk = meta_df["area"] == "V1"
#%%
# find the index in each mask 
ITidx = np.where(ITmsk)[0]
#%%
# show the images for each mask as a montage
from core.utils.montage_utils import make_grid_np
figdir = r"E:\OneDrive - Harvard University\PhDDefense_Talk\Figures\prototype_montage"
for msk, label in zip([ITmsk, V4msk, V1msk], ["IT", "V4", "V1"]):
    mskidx = np.where(msk)[0]
    mtg = make_grid_np([proto_dict[idx] for idx in mskidx], 7, 10)
    plt.imsave(join(figdir, f"{label}_proto_montage.png"), mtg)
    plt.imshow(mtg)
    plt.axis("off")
    plt.title(label)
    plt.tight_layout()
    plt.show()
    # save the meta data as json
    meta_df.loc[msk, :].to_json(join(figdir, f"{label}_meta.json"), orient="records")

# mtg = make_grid_np([proto_dict[idx] for idx in ITidx], 6, 10)
#%%
plt.imshow(mtg)
plt.axis("off")
plt.tight_layout()
plt.show()
#%%
import pickle as pkl
from PIL import Image
from scipy.ndimage import zoom
from scipy.interpolate import interp2d, RectBivariateSpline
model_msk_dir = r"E:\OneDrive - Washington University in St. Louis\corrFeatTsr_FactorVis\models\resnet50-layer3_NF3_bdr1_Tthresh_3__nobdr_resnet_CV"
#%%
def Hmaps2alphamap(Hmaps):
    alphamap = (Hmaps**2).sum(axis=-1)
    alphamap_full = np.pad(alphamap, ((1, 1), (1, 1)),
                           mode="constant", constant_values=0.0)
    alphamap_full_interp = zoom(alphamap_full, 256/14, order=1)
    alphamap_full_interp = alphamap_full_interp / alphamap_full_interp.max()
    return alphamap_full_interp

alphamap_dict = {}
for i in range(46):
    expname = f"Alfa_Exp{i+1:02d}"
    data = pkl.load(open(join(model_msk_dir, 
                              expname+"_factors.pkl"), "rb"))
    Hmaps = data['Hmaps']
    alphamap_dict[i] = Hmaps2alphamap(Hmaps)[:,:,None]

for i in range(45):
    expname = f"Beto_Exp{i+1:02d}"
    data = pkl.load(open(join(model_msk_dir, 
                              expname+"_factors.pkl"), "rb"))
    Hmaps = data['Hmaps']
    alphamap_dict[i + 46] = Hmaps2alphamap(Hmaps)[:,:,None]

#%%
def create_rgba_image(alpha_array, filename='image.png'):
    # Create the RGB channels with the given values
    rgb_channel = np.zeros_like(alpha_array,)
    # Combine the RGB channels and alpha channel into an RGBA image
    rgba_image = np.stack((rgb_channel, rgb_channel,
                           rgb_channel, alpha_array), axis=-1)
    # Convert to a PIL Image object and save as PNG
    image = Image.fromarray((rgba_image * 255.0).astype('uint8'), 'RGBA')
    image.save(filename)
    return image

#%%
for msk, label in zip([ITmsk, V4msk, V1msk], ["IT", "V4", "V1"]):
    mskidx = np.where(msk)[0]
    mtg = make_grid_np([alphamap_dict[idx] for idx in mskidx], 7, 10)
    create_rgba_image(1-mtg[:,:,0], join(figdir, f"{label}_alphamap_montage_rgbamsk.png"))
    plt.imsave(join(figdir, f"{label}_alphamap_montage.png"), mtg)
    plt.imshow(mtg)
    plt.axis("off")
    plt.title(label)
    plt.tight_layout()
    plt.show()
    # save the meta data as json
    # meta_df.loc[msk, :].to_json(join(figdir, f"{label}_meta.json"), orient="records")

#%%
mskdir = r"E:\OneDrive - Harvard University\Manifold_attrb_mask"
alphamap_dict = {}
for i in range(46):
    expname = f"Alfa_Exp{i+1:02d}"
    data  = plt.imread(join(mskdir,expname+"_mask_L2.png"))
    alphamap_dict[i] = data[:,:,0:1]

for i in range(45):
    expname = f"Beto_Exp{i+1:02d}"
    data  = plt.imread(join(mskdir,expname+"_mask_L2.png"))
    alphamap_dict[i + 46] = data[:,:,0:1]
#%%
plt.figure(figsize=(10, 10))
plt.imshow(alphamap_full)
plt.axis("off")
plt.tight_layout()
plt.show()
#%%
# interpolate the alphamap_full to 256x256 with image resize
# f = RectBivariateSpline(np.arange(0, 14), np.arange(0, 14),
#              alphamap_full, )
# alphamap_full_interp = f(np.linspace(0, 13, 256), 
#                          np.linspace(0, 13, 256))
#%%
plt.figure(figsize=(10, 10))
plt.imshow(alphamap_full_interp)
plt.axis("off")
plt.tight_layout()
plt.show()
#%%
# render this map as a gray scale RGBA png with 1 - the map as the alpha channel,
# with RGB channels as 0
alphamap_full_interp = alphamap_full_interp / alphamap_full_interp.max()
alphamap_full_interp = 1 - alphamap_full_interp
RGBAchannel = np.stack([np.zeros_like(alphamap_full_interp)]*3+\
                      [alphamap_full_interp], axis=-1)
RGBA_array = (RGBAchannel * 255).astype(np.uint8)
# save png
Image.fromarray(RGBA_array).save(join(figdir, "model_mask.png"))


#%%
Bmat[0]["Evol"][44]["BestBlockAvgImg"][0][0].shape

