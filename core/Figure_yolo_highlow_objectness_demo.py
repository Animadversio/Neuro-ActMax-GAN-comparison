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
outdir = Path(r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Figure_Evol_objectness\topbot_samples_src")
yolomodel = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
plt.switch_backend('module://backend_interagg')
#%%
def print_confidence_range(df, conf_col="confidence",):
    print(f"confidence range: {df[conf_col].min():.3f} ~ {df[conf_col].max():.3f}")
#%%
def yolo_detect_render_col(img_col):
    deepsim_result = yolomodel(img_col, size=256)
    return deepsim_result.render()[0]
#%%
tabdir = Path(r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables")
meta_df = pd.read_csv(tabdir / "meta_activation_stats.csv", index_col=0)
Amsk, Bmsk, V1msk, V4msk, ITmsk, length_msk, spc_msk, \
    sucsmsk, bsl_unstable_msk, bsl_stable_msk, validmsk = get_all_masks(meta_df)
bothsucmsk = (meta_df.p_maxinit_0 < 0.05) & (meta_df.p_maxinit_1 < 0.05)
FCsucsmsk = (meta_df.p_maxinit_0 < 0.05)
BGsucsmsk = (meta_df.p_maxinit_1 < 0.05)
#%%
all_df = pd.read_csv(tabdir / f"Evol_invivo_all_yolo_v5_stats.csv", index_col=0)
all_df["confidence_fill0"] = all_df.confidence.fillna(0)
GANimgtab = pd.read_csv(tabdir / 'GAN_samples_all_yolo_stats.csv', index_col=0)
GANimgtab["confidence_fill0"] = GANimgtab.confidence.fillna(0)
#%%
img_renders = yolomodel(top_imgs, size=256).render()
#%%
valid_df = all_df[(all_df.thread == 0) & all_df.Expi.isin(meta_df[validmsk].index)]
sorted_valid_df = valid_df.sort_values('confidence', ascending=False)
# sorted_valid_df.head(9).img_path.values
# sorted_valid_df.head(9).confidence.values
top_imgs = []
for img_path in sorted_valid_df.head(16).img_path.values:
    img = plt.imread(img_path)
    top_imgs.append(img)

sorted_valid_df.head(256).to_csv(outdir / "DeePSim_top256_imgs.csv")
plt.imsave(outdir / "DeePSim_top16_imgs.png", make_grid_np(top_imgs, nrow=4, padding=2))
print_confidence_range(sorted_valid_df.head(16))
plt.imsave(outdir / "DeePSim_top9_imgs.png", make_grid_np(top_imgs[:9], nrow=3, padding=2))
print_confidence_range(sorted_valid_df.head(9))
img_yolo_renders = yolomodel(top_imgs, size=256).render()
plt.imsave(outdir / "DeePSim_top16_imgs_yolo.png", make_grid_np(img_yolo_renders, nrow=4, padding=2))
plt.imsave(outdir / "DeePSim_top9_imgs_yolo.png", make_grid_np(img_yolo_renders[:9], nrow=3, padding=2))

plt.figure(figsize=[6, 6])
plt.imshow(make_grid_np(top_imgs, nrow=4, padding=2))
plt.axis('off')
plt.tight_layout()
plt.show()
#%%
sorted_valid_df = valid_df.sort_values('confidence', ascending=True) # , na_position='last'
# sorted_valid_df.head(9).img_path.values
# sorted_valid_df.head(9).confidence.values
bot_imgs = []
for img_path in sorted_valid_df.head(16).img_path.values:
    img = plt.imread(img_path)
    bot_imgs.append(img)
sorted_valid_df.head(256).to_csv(outdir / "DeePSim_bot256_imgs.csv")
plt.imsave(outdir / "DeePSim_bot16_imgs.png", make_grid_np(bot_imgs, nrow=4, padding=2))
print_confidence_range(sorted_valid_df.head(16))
plt.imsave(outdir / "DeePSim_bot9_imgs.png", make_grid_np(bot_imgs[:9], nrow=3, padding=2))
print_confidence_range(sorted_valid_df.head(9))

img_yolo_renders = yolomodel(bot_imgs, size=256).render()
plt.imsave(outdir / "DeePSim_bot16_imgs_yolo.png", make_grid_np(img_yolo_renders, nrow=4, padding=2))
plt.imsave(outdir / "DeePSim_bot9_imgs_yolo.png", make_grid_np(img_yolo_renders[:9], nrow=3, padding=2))

plt.figure(figsize=[6, 6])
plt.imshow(make_grid_np(bot_imgs, nrow=4, padding=2))
plt.axis('off')
plt.tight_layout()
plt.show()
#%%
valid_df = all_df[(all_df.thread == 1) & all_df.Expi.isin(meta_df[validmsk].index)]
sorted_valid_df = valid_df.sort_values('confidence', ascending=False)
top_imgs = []
for img_path in sorted_valid_df.head(16).img_path.values:
    img = plt.imread(img_path)
    top_imgs.append(img)
# sorted_valid_df.head(9).img_path.values
# sorted_valid_df.head(9).confidence.values
plt.imsave(outdir / "BigGAN_top16_imgs.png", make_grid_np(top_imgs, nrow=4, padding=2))
print_confidence_range(sorted_valid_df.head(16))
plt.imsave(outdir / "BigGAN_top9_imgs.png", make_grid_np(top_imgs[:9], nrow=3, padding=2))
print_confidence_range(sorted_valid_df.head(9))
sorted_valid_df.head(256).to_csv(outdir / "BigGAN_top256_imgs.csv")

img_yolo_renders = yolomodel(top_imgs, size=256).render()
plt.imsave(outdir / "BigGAN_top16_imgs_yolo.png", make_grid_np(img_yolo_renders, nrow=4, padding=2))
plt.imsave(outdir / "BigGAN_top9_imgs_yolo.png", make_grid_np(img_yolo_renders[:9], nrow=3, padding=2))

plt.figure(figsize=[6, 6])
plt.imshow(make_grid_np(top_imgs, nrow=4, padding=2))
plt.axis('off')
plt.tight_layout()
plt.show()
#%%
sorted_valid_df = valid_df.sort_values('confidence', ascending=True) # , na_position='last'
bot_imgs = []
for img_path in sorted_valid_df.head(16).img_path.values:
    img = plt.imread(img_path)
    bot_imgs.append(img)
# sorted_valid_df.head(9).img_path.values
# sorted_valid_df.head(9).confidence.values
plt.imsave(outdir / "BigGAN_bot16_imgs.png", make_grid_np(bot_imgs, nrow=4, padding=2))
print_confidence_range(sorted_valid_df.head(16))
plt.imsave(outdir / "BigGAN_bot9_imgs.png", make_grid_np(bot_imgs[:9], nrow=3, padding=2))
print_confidence_range(sorted_valid_df.head(9))
sorted_valid_df.head(256).to_csv(outdir / "BigGAN_bot256_imgs.csv")

img_yolo_renders = yolomodel(bot_imgs, size=256).render()
plt.imsave(outdir / "BigGAN_bot16_imgs_yolo.png", make_grid_np(img_yolo_renders, nrow=4, padding=2))
plt.imsave(outdir / "BigGAN_bot9_imgs_yolo.png", make_grid_np(img_yolo_renders[:9], nrow=3, padding=2))

plt.figure(figsize=[6, 6])
plt.imshow(make_grid_np(bot_imgs, nrow=4, padding=2))
plt.axis('off')
plt.tight_layout()
plt.show()
#%%
GANroot = Path(r"F:\insilico_exps\GAN_sample_fid")
#%%
sorted_df = GANimgtab[GANimgtab.imgdir_name=='DeePSim_4std'].sort_values('confidence', ascending=False)
#%%
top_imgs = []
for img_path in sorted_df.head(9).img_path:
    img = plt.imread(GANroot / "DeePSim_4std" / img_path.split('/')[-1])
    top_imgs.append(img)
#%%
sorted_df = GANimgtab[GANimgtab.imgdir_name=='BigGAN_std_008'].sort_values('confidence', ascending=False).head(9)

