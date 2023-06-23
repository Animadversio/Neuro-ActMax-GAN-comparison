import torch
import re
from pathlib import Path
import pickle as pkl
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms
from core.utils.montage_utils import make_grid_np
from neuro_data_analysis.neural_data_utils import get_all_masks
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data, get_expstr
from core.utils.stats_utils import ttest_ind_print, ttest_rel_print, ttest_ind_print_df
from core.utils.GAN_utils import upconvGAN, BigGAN_wrapper, loadBigGAN
G = upconvGAN("fc6")
G.cuda().eval().requires_grad_(False)
#%%
BG = BigGAN_wrapper(loadBigGAN())
#%%
# Model
yolomodel = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
plt.switch_backend('module://backend_interagg')
#%%
outdir = Path(r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Figure_Evol_objectness\GAN_ref_samples")
#%%
# set random seed
for seed in range(100):
    torch.manual_seed(seed)
    gen_img = 255*G.visualize(4*torch.randn(1, 4096).cuda()).cpu().permute(0,2,3,1).numpy()
    deepsim_result = yolomodel(gen_img[0], size=256)
    plt.imsave(outdir / f"GAN_fc6_{seed:03d}.png", gen_img[0].astype(np.uint8))
    plt.imsave(outdir / f"GAN_fc6_{seed:03d}_render.png", deepsim_result.render()[0])

    # plt.figure(figsize=[4.5, 4.5])
    # plt.imshow(deepsim_result.render()[0])
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()
#%%
for seed in range(100):
    torch.manual_seed(seed)
    gen_img = BG.visualize(BG.sample_vector(1)).cpu().permute(0,2,3,1).numpy()*255
    deepsim_result = yolomodel(gen_img[0], size=256)
    plt.imsave(outdir / f"BigGAN_{seed:03d}.png", gen_img[0].astype(np.uint8))
    plt.imsave(outdir / f"BigGAN_{seed:03d}_render.png", deepsim_result.render()[0])
    # plt.figure(figsize=[4.5, 4.5])
    # plt.imshow(deepsim_result.render()[0])
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()
#%%
imgdir = r"E:\Datasets\imagenet-valid\valid"
imgpathlist = sorted(list(Path(imgdir).glob("*.JPEG")))
result = yolomodel(imgpathlist[:100], size=256)
annot_imgs = result.render()
for i in range(100):
    plt.imsave(outdir / f"imagenet_valid_{i:03d}.png", annot_imgs[i])
    img = plt.imread(imgpathlist[i])
    plt.imsave(outdir / f"imagenet_valid_{i:03d}_orig.png", img)
#%%

