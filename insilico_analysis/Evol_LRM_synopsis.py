# %%
import sys
sys.path.append("/n/home12/binxuwang/Github/Neuro-ActMax-GAN-comparison")

# %%
""" Cluster version of BigGAN Evol """
import re
import glob
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pylab as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample, one_hot_from_names, save_as_images)
from core.utils.CNN_scorers import TorchScorer
from core.utils.GAN_utils import BigGAN_wrapper, upconvGAN, loadBigGAN
from core.utils.grad_RF_estim import grad_RF_estimate, gradmap2RF_square
from core.utils.layer_hook_utils import get_module_names, layername_dict, register_hook_by_module_names, get_module_name_shapes
from core.utils.Optimizers import CholeskyCMAES, HessCMAES, ZOHA_Sphere_lr_euclid
import scipy.stats as stats
from scipy.stats import sem
from core.utils.stats_utils import shaded_errorbar

def group_stats_by_gen(generations, value, funcs=[np.mean, sem]):
    gen_slice = np.arange(min(generations), max(generations) + 1)
    mean_score = np.zeros((gen_slice.shape[0], value.shape[1]))
    sem_score = np.zeros((gen_slice.shape[0], value.shape[1]))
    max_score = np.zeros((gen_slice.shape[0], value.shape[1]))
    for i, geni in enumerate(gen_slice):
        mean_score[i, :] = np.mean(value[generations == geni], axis=0)
        sem_score[i, :] = sem(value[generations == geni], axis=0)
        max_score[i, :] = np.max(value[generations == geni])
    return gen_slice, mean_score, sem_score, max_score



# %% [markdown]
# ### Fetch All Table

# %%
from core.utils.montage_utils import crop_from_montage

# %%
saveroot = r"/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Projects/Evol_lrm_GAN_cmp"
syndir = join(saveroot, "synopsis")
os.makedirs(syndir, exist_ok=True)

# %%
!echo $STORE_DIR/Projects/Evol_lrm_GAN_cmp

# %%
!ls {saveroot}/alexnet_lrm3* -d

# %% [markdown]
# #### LRM1

# %%
saveroot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/Evol_lrm_GAN_cmp"
# saveroot = r"/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Projects/Evol_lrm_GAN_cmp"
syndir = join( r"/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Projects/Evol_lrm_GAN_cmp", "synopsis")
os.makedirs(syndir, exist_ok=True)

# %%
!ls $STORE_DIR/Projects/Evol_lrm_GAN_cmp

# %%
df_col = []
netname = r"alexnet_lrm2"
# "conv1_relu","conv2_relu","conv3_relu",
for layerkey in ["fc6_relu", "fc7_relu", ]:#"conv2_relu","conv3_relu","conv4_relu","conv5_relu","fc8", 
    for iChannel in trange(25): #range(25):
        if layerkey in ["conv1_relu", "conv2_relu", "conv3_relu",]:
            savedir = join(saveroot, f"{netname}-{layerkey}-Ch{iChannel:04d}_RFrsz")
        else:
            savedir = join(saveroot, f"{netname}-{layerkey}-Ch{iChannel:04d}")
        jpgfiles = glob.glob(join(savedir, "besteachgen*.jpg"))
        for iT in [0, 1, 2, 3]:
            for GANname in ["BigGAN", "fc6", ]:
                methodlab = f"{GANname}_CMAES_T{iT}"
                # search for files with the naming "scores%s_%05d.npz" % (methodlab, RND)
                pattern = re.compile(r"besteachgen%s_(\d+).jpg" % (methodlab,))
                jpgfile_GAN = [f for f in jpgfiles if pattern.search(f)]
                for jpgfile in jpgfile_GAN:
                    RND = int(pattern.search(jpgfile).group(1))
                    title_str = f"{netname} {layerkey} Ch{iChannel} {methodlab} RND {RND}"
                    mtg = plt.imread(jpgfile)
                    finalimg = crop_from_montage(mtg, -1)
                    npzfile = join(savedir, f"scores{methodlab}_{RND:05d}.npz")
                    data = np.load(npzfile)
                    generations = data["generations"]
                    scores_all = data["scores_all"]
                    scores_dyn_all = data["scores_dyn_all"]
                    gen_slice, mean_score, sem_score, max_score = group_stats_by_gen(generations, scores_dyn_all)
                    finalscore = mean_score[-1, iT]
                    df_col.append({"network": netname, "layerkey": layerkey, "iChannel": iChannel, "iT": iT, "GANname": GANname,
                                "methodlab": methodlab, "RND": RND, "finalimg": finalimg, "finalscore": finalscore, 
                                "gen_slice": gen_slice, "mean_score": mean_score, "sem_score": sem_score, "max_score": max_score})
df_img = pd.DataFrame(df_col)


# %%
df_img.to_pickle(join(syndir, f"{netname}_df_img.pkl"))

# %%
df_col = []
netname = r"alexnet_lrm1"
# "conv1_relu","conv2_relu","conv3_relu",
for layerkey in ["fc7_relu", ]:#"fc6_relu", "conv2_relu","conv3_relu","conv4_relu","conv5_relu","fc8", 
    for iChannel in trange(25): #range(25):
        if layerkey in ["conv1_relu", "conv2_relu", "conv3_relu",]:
            savedir = join(saveroot, f"{netname}-{layerkey}-Ch{iChannel:04d}_RFrsz")
        else:
            savedir = join(saveroot, f"{netname}-{layerkey}-Ch{iChannel:04d}")
        jpgfiles = glob.glob(join(savedir, "besteachgen*.jpg"))
        for iT in [0, 1, 2, 3]:
            for GANname in ["BigGAN", "fc6", ]:
                methodlab = f"{GANname}_CMAES_T{iT}"
                # search for files with the naming "scores%s_%05d.npz" % (methodlab, RND)
                pattern = re.compile(r"besteachgen%s_(\d+).jpg" % (methodlab,))
                jpgfile_GAN = [f for f in jpgfiles if pattern.search(f)]
                for jpgfile in jpgfile_GAN:
                    RND = int(pattern.search(jpgfile).group(1))
                    title_str = f"{netname} {layerkey} Ch{iChannel} {methodlab} RND {RND}"
                    mtg = plt.imread(jpgfile)
                    finalimg = crop_from_montage(mtg, -1)
                    npzfile = join(savedir, f"scores{methodlab}_{RND:05d}.npz")
                    data = np.load(npzfile)
                    generations = data["generations"]
                    scores_all = data["scores_all"]
                    scores_dyn_all = data["scores_dyn_all"]
                    gen_slice, mean_score, sem_score, max_score = group_stats_by_gen(generations, scores_dyn_all)
                    finalscore = mean_score[-1, iT]
                    df_col.append({"network": netname, "layerkey": layerkey, "iChannel": iChannel, "iT": iT, "GANname": GANname,
                                "methodlab": methodlab, "RND": RND, "finalimg": finalimg, "finalscore": finalscore, 
                                "gen_slice": gen_slice, "mean_score": mean_score, "sem_score": sem_score, "max_score": max_score})
df_img = pd.DataFrame(df_col)
df_img.to_pickle(join(syndir, f"{netname}_df_img.pkl"))

# %%
!ls {syndir}

# %%
!du -sh {syndir}/*

# %% [markdown]
# #### LRM3

# %%
df_col = []
# "conv1_relu","conv2_relu","conv3_relu",
for layerkey in ["conv1_relu","conv2_relu","conv3_relu","conv4_relu","conv5_relu","fc6_relu","fc7_relu", "fc8"]:
    for iChannel in trange(25): #range(25):
        if layerkey in ["conv1_relu","conv2_relu","conv3_relu",]:
            savedir = join(saveroot, f"alexnet_lrm3-{layerkey}-Ch{iChannel:04d}_RFrsz")
        else:
            savedir = join(saveroot, f"alexnet_lrm3-{layerkey}-Ch{iChannel:04d}")
        jpgfiles = glob.glob(join(savedir, "besteachgen*.jpg"))
        for iT in [0, 1, 2, 3]:
            for GANname in ["BigGAN", "fc6", ]:
                methodlab = f"{GANname}_CMAES_T{iT}"
                # search for files with the naming "scores%s_%05d.npz" % (methodlab, RND)
                pattern = re.compile(r"besteachgen%s_(\d+).jpg" % (methodlab,))
                jpgfile_GAN = [f for f in jpgfiles if pattern.search(f)]
                for jpgfile in jpgfile_GAN:
                    RND = int(pattern.search(jpgfile).group(1))
                    title_str = f"alexnet_lrm3 {layerkey} Ch{iChannel} {methodlab} RND {RND}"
                    mtg = plt.imread(jpgfile)
                    finalimg = crop_from_montage(mtg, -1)
                    npzfile = join(savedir, f"scores{methodlab}_{RND:05d}.npz")
                    data = np.load(npzfile)
                    generations = data["generations"]
                    scores_all = data["scores_all"]
                    scores_dyn_all = data["scores_dyn_all"]
                    gen_slice, mean_score, sem_score, max_score = group_stats_by_gen(generations, scores_dyn_all)
                    finalscore = mean_score[-1, iT]
                    df_col.append({"layerkey": layerkey, "iChannel": iChannel, "iT": iT, "GANname": GANname,
                                "methodlab": methodlab, "RND": RND, "finalimg": finalimg, "finalscore": finalscore, 
                                "gen_slice": gen_slice, "mean_score": mean_score, "sem_score": sem_score, "max_score": max_score})
df_img = pd.DataFrame(df_col)
df_img.to_pickle(join(syndir, "df_img.pkl"))



# %%


# %%
df_col = []
# "conv1_relu","conv2_relu","conv3_relu",
for layerkey in ["conv2_relu","conv3_relu","conv4_relu","conv5_relu","fc6_relu","fc7_relu", "fc8"]:
    for iChannel in trange(25): #range(25):
        if layerkey in ["conv1_relu","conv2_relu","conv3_relu",]:
            savedir = join(saveroot, f"alexnet_lrm3-{layerkey}-Ch{iChannel:04d}_RFrsz")
        else:
            savedir = join(saveroot, f"alexnet_lrm3-{layerkey}-Ch{iChannel:04d}")
        jpgfiles = glob.glob(join(savedir, "besteachgen*.jpg"))
        for iT in [0, 1, 2, 3]:
            for GANname in ["BigGAN", "fc6", ]:
                methodlab = f"{GANname}_CMAES_T{iT}"
                # search for files with the naming "scores%s_%05d.npz" % (methodlab, RND)
                pattern = re.compile(r"besteachgen%s_(\d+).jpg" % (methodlab,))
                jpgfile_GAN = [f for f in jpgfiles if pattern.search(f)]
                for jpgfile in jpgfile_GAN:
                    RND = int(pattern.search(jpgfile).group(1))
                    title_str = f"alexnet_lrm3 {layerkey} Ch{iChannel} {methodlab} RND {RND}"
                    mtg = plt.imread(jpgfile)
                    finalimg = crop_from_montage(mtg, -1)
                    npzfile = join(savedir, f"scores{methodlab}_{RND:05d}.npz")
                    data = np.load(npzfile)
                    generations = data["generations"]
                    scores_all = data["scores_all"]
                    scores_dyn_all = data["scores_dyn_all"]
                    gen_slice, mean_score, sem_score, max_score = group_stats_by_gen(generations, scores_dyn_all)
                    finalscore = mean_score[-1, iT]
                    df_col.append({"layerkey": layerkey, "iChannel": iChannel, "iT": iT, "GANname": GANname,
                                "methodlab": methodlab, "RND": RND, "finalimg": finalimg, "finalscore": finalscore, 
                                "gen_slice": gen_slice, "mean_score": mean_score, "sem_score": sem_score, "max_score": max_score})
df_img = pd.DataFrame(df_col)


# %%
df_img.to_pickle(join(syndir, "df_img.pkl"))

# %% [markdown]
# ### Figure synopsis

# %%
!du -sh /n/holylabs/LABS/kempner_fellows/Users/binxuwang/Projects/Evol_lrm_GAN_cmp/synopsis

# %%
import torch
model, transforms = torch.hub.load('harvard-visionlab/lrm-steering', 'alexnet_lrm3', pretrained=True, steering=True, force_reload=True)
print(model)

# %% [markdown]
# ### Activation Trajectory Analysis

# %%
df.norm_score = df.mean_score / df.mean_score.max(axis=1)[:, None]

# %%
avg_traj = df.groupby(["iT", "GANname"]).agg({"mean_score":"mean"})

# %%
avg_traj

# %%
plt.figure()
for iT in [0, 1, 2, 3]:
    plt.plot(avg_traj.loc[iT, "fc6"]["mean_score"][:, iT], label=f"T{iT} DeePSim")
for iT in [0, 1, 2, 3]:
    plt.plot(avg_traj.loc[iT, "BigGAN"]["mean_score"][:, iT], label=f"T{iT} BigGAN")
plt.legend()
plt.show()

# %%
resp_dyn_mat = np.zeros((2, 4, 4, ))
for iT in [0, 1, 2, 3]:
    resp_dyn_mat[0, iT, :] = avg_traj.loc[iT, "fc6"]["mean_score"][-1, :]
    resp_dyn_mat[1, iT, :] = avg_traj.loc[iT, "BigGAN"]["mean_score"][-1, :]

figh, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(resp_dyn_mat[0], ax=axs[0], annot=True, fmt=".1f", cmap="inferno", vmin=13, vmax=19)#center=0)
axs[0].set_title("fc6")
sns.heatmap(resp_dyn_mat[1], ax=axs[1], annot=True, fmt=".1f", cmap="inferno", vmin=13, vmax=19)#center=0)
axs[1].set_title("BigGAN")
plt.show()

# %% [markdown]
# ### Evol Image Analysis for LRM

# %%
# make a montage of the final images
iCh = 24
layerkey = "fc6_relu"
layerkey, iCh = "fc6_relu", 2
layerkey, iCh = "conv4_relu", 5
layerkey, iCh = "fc7_relu", 10
for iRND in range(5):
    figh, axs = plt.subplots(2, 4, figsize=(16, 8))
    figh2, axs2  = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    for j, GANname in enumerate(["fc6", "BigGAN"]):
        fix_RND = df_img[(df_img.layerkey == layerkey) & (df_img.iChannel == iCh) & \
                         (df_img.iT == 0) & (df_img.GANname == GANname)].RND.values[iRND]
        for i, iT in enumerate([0, 1, 2, 3]):
            df_unit = df_img[(df_img.layerkey == layerkey) & (df_img.iChannel == iCh) & \
                         (df_img.iT == iT) & (df_img.GANname == GANname)]
            df_part = df_unit[df_unit.RND == fix_RND].iloc[0]
            final_score = df_part.mean_score[-1,iT]
            axs[j, i].imshow(df_part.finalimg)
            axs[j, i].set_title(f"{GANname} T{iT} score {final_score:.02f}")
            axs[j, i].axis("off")
            shaded_errorbar(df_part.gen_slice, df_part.mean_score[:,iT], df_part.sem_score[:,iT], 
                            color=f"k", ax=axs2[j,i], lw=2)
            axs2[j, i].set_title(f"{GANname} T{iT} score {final_score:.02f}")
            for it in range(4):
                shaded_errorbar(df_part.gen_slice, df_part.mean_score[:,it], df_part.sem_score[:,it], 
                            color=f"C{it}", ax=axs2[j,i], lw=0.3)
    plt.tight_layout()
    plt.show()

# %%
from core.utils.plot_utils import saveallforms
os.makedirs(join(syndir,"figsummary"),exist_ok=True)
figdir = join(syndir,"figsummary")

# %%
plt.get_backend() # 'module://matplotlib_inline.backend_inline'

# %%
layernames = df_img.layerkey.unique()

# %% [markdown]
# ### Mass produce panels

# %%
plt.switch_backend("agg")
model_name = "alexnet_lrm3"
layernames = df_img.layerkey.unique()
for layerkey in layernames:
    for iCh in trange(25):
        for iRND in range(5):
            figh, axs = plt.subplots(2, 4, figsize=(16, 8))
            figh2, axs2  = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
            titlestr = f"{model_name} {layerkey} Ch{iCh:04d} trial{iRND}\n"
            for j, GANname in enumerate(["fc6", "BigGAN"]):
                fix_RND = df_img[(df_img.layerkey == layerkey) & (df_img.iChannel == iCh) & \
                                 (df_img.iT == 0) & (df_img.GANname == GANname)].RND.values[iRND]
                titlestr += f"\t{GANname}: {fix_RND}"
                for i, iT in enumerate([0, 1, 2, 3]):
                    df_unit = df_img[(df_img.layerkey == layerkey) & (df_img.iChannel == iCh) & \
                                 (df_img.iT == iT) & (df_img.GANname == GANname)]
                    df_part = df_unit[df_unit.RND == fix_RND].iloc[0]
                    final_score = df_part.mean_score[-1,iT]
                    axs[j, i].imshow(df_part.finalimg)
                    axs[j, i].set_title(f"{GANname} T{iT} score {final_score:.02f}")
                    axs[j, i].axis("off")
                    shaded_errorbar(df_part.gen_slice, df_part.mean_score[:,iT], df_part.sem_score[:,iT], 
                                    color=f"k", ax=axs2[j,i], lw=2)
                    axs2[j, i].set_title(f"{GANname} T{iT} score {final_score:.02f}")
                    for it in range(4):
                        shaded_errorbar(df_part.gen_slice, df_part.mean_score[:,it], df_part.sem_score[:,it], 
                                    color=f"C{it}", ax=axs2[j,i], lw=0.3)
            figh.suptitle(titlestr)
            figh2.suptitle(titlestr)
            figh.tight_layout()
            figh2.tight_layout()
            plt.close(figh)
            plt.close(figh2)
            saveallforms(figdir, f"{model_name}-{layerkey}-Ch{iCh:04d}_trial{iRND}_final_image_montage", figh)
            saveallforms(figdir, f"{model_name}-{layerkey}-Ch{iCh:04d}_trial{iRND}_score_traj_montage", figh2)
            

# %% [markdown]
# ### Score Synopsis & Success Rate paired plot

# %%
df_img.shape

# %%
df_img[(df_img.layerkey == layerkey) & (df_img.iChannel == iCh) & \
                                (df_img.GANname == GANname)].RND.unique()

# %%
df_img[(df_img.layerkey == layerkey) & (df_img.iChannel == iCh) & \
                                (df_img.iT == 0) & (df_img.GANname == GANname)].RND

# %%
df_img[(df_img.layerkey == layerkey) & (df_img.iChannel == iCh) & \
        (df_img.iT == 0) & (df_img.GANname == GANname)].RND.unique()

# %%
df_img.layerkey.unique()

# %%
score_df_col = []
for layerkey in df_img.layerkey.unique():
    for iCh in range(25):
        df_unit_all = df_img[(df_img.layerkey == layerkey) & (df_img.iChannel == iCh) & \
                                (df_img.iT == iT)]
        for j, GANname in enumerate(["fc6", "BigGAN"]):
            fix_RND_list = df_img[(df_img.layerkey == layerkey) & (df_img.iChannel == iCh) & \
                                (df_img.iT == 0) & (df_img.GANname == GANname)].RND
            # if len(fix_RND_list) > 5:
            #     continue
            for iRND in range(5):
                row_dict = {"layer":layerkey, "iCh":iCh, "iRND":iRND, "GANname": GANname}
                fix_RND = df_img[(df_img.layerkey == layerkey) & (df_img.iChannel == iCh) & \
                                (df_img.iT == 0) & (df_img.GANname == GANname)].RND.values[iRND]
                for i, iT in enumerate([0, 1, 2, 3]):
                    df_unit = df_img[(df_img.layerkey == layerkey) & (df_img.iChannel == iCh) & \
                                (df_img.iT == iT) & (df_img.GANname == GANname)]
                    df_part = df_unit[df_unit.RND == fix_RND].iloc[0]
                    final_score = df_part.mean_score[-1,iT]
                    row_dict[f"score_T{iT}"] = final_score
                score_df_col.append(row_dict)

score_df = pd.DataFrame(score_df_col)

# %%
score_df.shape

# %%
pivot_table = score_df.pivot(index=['layer',"iCh","iRND"], columns='GANname', values='score_T0')

# %%
pivot_table

# %%
import scipy.stats as stats
plt.figure()
layer_names = [layerkey for layerkey in pivot_table.index.levels[0]]  # Collect layer names
for i, layerkey in enumerate(pivot_table.index.levels[0]):
    part_df = pivot_table[pivot_table.index.get_level_values(0) == layerkey]
    x_jitter = i * np.ones(part_df.shape[0]) + np.random.rand(part_df.shape[0]) * 0.2
    plt.scatter(x_jitter, part_df.fc6, c="C0", alpha=0.3, label="DeePSim" if i==0 else None)
    plt.scatter(x_jitter+0.5, part_df.BigGAN, c="C1", alpha=0.3, label="BigGAN" if i==0 else None)
    t_statistic, p_value = stats.ttest_rel(part_df.fc6, part_df.BigGAN, nan_policy="omit")
    x_jitter_arr = np.stack([x_jitter, x_jitter+0.5], axis=1)
    plt.plot(x_jitter_arr.T, part_df[["fc6", "BigGAN"]].values.T, c="k", lw=0.1)
    y_position = np.nanmax(part_df[['fc6', 'BigGAN']].values) + 0.1  # Adjust y_position based on your data range
    print(y_position)
    plt.text(i + 0.25, y_position, f'T-stat: {t_statistic:.2f}\nP-value: {p_value:.2f}', fontsize=9, ha='center')

plt.xticks(np.arange(len(layer_names)) + 0.25, labels=layer_names, rotation=45)  # Adjust rotation for better readability
plt.legend()
plt.xlabel('Layers')
plt.ylabel('Final Score')
plt.title('Comparison of fc6 and BigGAN across layers')
plt.tight_layout()
plt.show()

# %%
(pivot_table.fc6 < 0.1).mean()

# %%
(pivot_table.fc6 < 0.1).values.mean()

# %%
import scipy.stats as stats
for iT in range(4):
    pivot_table = score_df.pivot(index=['layer',"iCh","iRND"], columns='GANname', values=f'score_T{iT}')
    
    plt.figure(figsize=[7,4.5])
    layer_names = [layerkey for layerkey in pivot_table.index.levels[0]]  # Collect layer names
    for i, layerkey in enumerate(pivot_table.index.levels[0]):
        part_df = pivot_table[pivot_table.index.get_level_values(0) == layerkey]
        BigGAN_sucs_rate = (part_df.BigGAN < 0.1).mean()
        DeePSim_sucs_rate = (part_df.fc6 < 0.1).mean()
        x_jitter = i * np.ones(part_df.shape[0]) + np.random.rand(part_df.shape[0]) * 0.2
        plt.scatter(x_jitter, part_df.fc6, c="C0", alpha=0.3, label="DeePSim" if i==0 else None)
        plt.scatter(x_jitter+0.5, part_df.BigGAN, c="C1", alpha=0.3, label="BigGAN" if i==0 else None)
        t_statistic, p_value = stats.ttest_rel(part_df.fc6, part_df.BigGAN, nan_policy="omit")
        x_jitter_arr = np.stack([x_jitter, x_jitter+0.5], axis=1)
        plt.plot(x_jitter_arr.T, part_df[["fc6", "BigGAN"]].values.T, c="k", lw=0.1)
        y_position = np.nanmax(part_df[['fc6', 'BigGAN']].values) + 0.1  # Adjust y_position based on your data range
        # print(y_position)
        plt.text(i + 0.25, y_position, f'T-stat: {t_statistic:.2f}\nP-value: {p_value:.2f}\nFC rate {DeePSim_sucs_rate:.3f}\nBG rate {BigGAN_sucs_rate:.3f}', fontsize=9, ha='center')
    
    plt.xticks(np.arange(len(layer_names)) + 0.25, labels=layer_names, rotation=30)  # Adjust rotation for better readability
    plt.legend()
    plt.xlabel('Layers')
    plt.ylabel('Final Score')
    plt.title(f'Comparison of fc6 and BigGAN across layers forward pass {iT}')
    plt.tight_layout()
    plt.show()
        

# %%
sns.stripplot(data=score_df, x="layer", y="score_T3", hue="GANname", dodge=True, alpha=0.6)

# %% [markdown]
# #### 

# %%
!ls {syndir}

# %%
!ls

# %%
from core.utils.plot_utils import saveallforms
figdir = join(syndir, "figsummary")

# %%
import scipy.stats as stats
netname = "alexnet_lrm3"
for netname in ["alexnet_lrm1", "alexnet_lrm2", "alexnet_lrm3"]:
    df_img = pd.read_pickle(join(syndir, f"{netname}_df_img.pkl"))

    score_df_col = []
    for layerkey in df_img.layerkey.unique():
        for iCh in range(25):
            df_unit_all = df_img[(df_img.layerkey == layerkey) & (df_img.iChannel == iCh) & \
                                    (df_img.iT == iT)]
            for j, GANname in enumerate(["fc6", "BigGAN"]):
                fix_RND_list = df_img[(df_img.layerkey == layerkey) & (df_img.iChannel == iCh) & \
                                    (df_img.iT == 0) & (df_img.GANname == GANname)].RND
                # if len(fix_RND_list) > 5:
                #     continue
                for iRND in range(5):
                    row_dict = {"layer":layerkey, "iCh":iCh, "iRND":iRND, "GANname": GANname}
                    fix_RND = df_img[(df_img.layerkey == layerkey) & (df_img.iChannel == iCh) & \
                                    (df_img.iT == 0) & (df_img.GANname == GANname)].RND.values[iRND]
                    for i, iT in enumerate([0, 1, 2, 3]):
                        df_unit = df_img[(df_img.layerkey == layerkey) & (df_img.iChannel == iCh) & \
                                    (df_img.iT == iT) & (df_img.GANname == GANname)]
                        df_part = df_unit[df_unit.RND == fix_RND].iloc[0]
                        final_score = df_part.mean_score[-1,iT]
                        row_dict[f"score_T{iT}"] = final_score
                    score_df_col.append(row_dict)

    score_df = pd.DataFrame(score_df_col)

    for iT in range(4):
        pivot_table = score_df.pivot(index=['layer',"iCh","iRND"], columns='GANname', values=f'score_T{iT}')
        plt.figure(figsize=[7,4.5])
        layer_names = [layerkey for layerkey in pivot_table.index.levels[0]]  # Collect layer names
        for i, layerkey in enumerate(pivot_table.index.levels[0]):
            part_df = pivot_table[pivot_table.index.get_level_values(0) == layerkey]
            BigGAN_sucs_rate = (part_df.BigGAN < 0.1).mean()
            DeePSim_sucs_rate = (part_df.fc6 < 0.1).mean()
            x_jitter = i * np.ones(part_df.shape[0]) + np.random.rand(part_df.shape[0]) * 0.2
            plt.scatter(x_jitter, part_df.fc6, c="C0", alpha=0.3, label="DeePSim" if i==0 else None)
            plt.scatter(x_jitter+0.5, part_df.BigGAN, c="C1", alpha=0.3, label="BigGAN" if i==0 else None)
            t_statistic, p_value = stats.ttest_rel(part_df.fc6, part_df.BigGAN, nan_policy="omit")
            x_jitter_arr = np.stack([x_jitter, x_jitter+0.5], axis=1)
            plt.plot(x_jitter_arr.T, part_df[["fc6", "BigGAN"]].values.T, c="k", lw=0.1)
            y_position = np.nanmax(part_df[['fc6', 'BigGAN']].values) + 0.1  # Adjust y_position based on your data range
            # print(y_position)
            plt.text(i + 0.25, y_position, f'T-stat: {t_statistic:.2f}\nP-value: {p_value:.2f}\nFC rate {DeePSim_sucs_rate:.3f}\nBG rate {BigGAN_sucs_rate:.3f}', fontsize=9, ha='center')
        
        plt.xticks(np.arange(len(layer_names)) + 0.25, labels=layer_names, rotation=30)  # Adjust rotation for better readability
        plt.legend()
        plt.xlabel('Layers')
        plt.ylabel('Final Score')
        plt.title(f'Comparison of fc6 and BigGAN across layers forward pass {iT}\n{netname}')
        plt.tight_layout()
        saveallforms(figdir, f"{netname}-fc6_BigGAN_forward_pass_cmp_T{iT}", plt.gcf())
        plt.show()
        

# %% [markdown]
# ### File Management 

# %%
if False:
    from pathlib import Path
    import datetime
    saveroot = r"/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Projects/Evol_lrm_GAN_cmp"
    today = datetime.date.today()
    for subdir, dirs, files in os.walk(saveroot):
        cnt = 0
        # Check if the current subdir has the desired prefix
        if Path(subdir).name.startswith("alexnet_lrm3-fc6_relu"):
            for file in files:
                file_path = Path(subdir) / file
                creation_time = datetime.date.fromtimestamp(os.path.getctime(file_path))
                if creation_time == today:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    cnt +=1
            print(subdir, cnt)


