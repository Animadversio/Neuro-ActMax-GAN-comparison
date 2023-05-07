"""Merge the evol trajectory for in silico experiments into one
one figure for each unit.
"""
import shutil
import os
import re
import glob
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from os.path import join
from easydict import EasyDict as edict
import numpy as np
import pandas as pd
import pickle as pkl
from collections import defaultdict
from core.utils.plot_utils import saveallforms

rootdir = r"F:\insilico_exps\GAN_Evol_cmp"
outdir = r"F:\insilico_exps\GAN_Evol_cmp\evol_traj_merge"
# os.makedirs(outdir, exist_ok=True)
#%%

img_mtg_fps = [*Path(r"F:\insilico_exps\GAN_Evol_cmp\protoimgs_merge").glob("*.jpg")]
unitlist = [img_mtg_fp.name.replace("_optim_pool.jpg", "") for img_mtg_fp in img_mtg_fps]
#%%
from pathlib import Path
import seaborn as sns
from core.utils.plot_utils import saveallforms
# sns.set() # how to reset seaborn settings?
# sns.set_style("white")
sns.reset_defaults()
# remove the upper and right part of the frame
sns.set_style("white", {'axes.spines.right': False, 'axes.spines.top': False})

clrs = ["red", "green", "blue"]
#%%
# use the agg mode
plt.switch_backend('agg')
# get current backend
# plt.get_backend() # 'module://backend_interagg'
#%%
optimnames = "CholCMA", "HessCMA", "HessCMA500_fc6"
for unitname in tqdm(unitlist):
    unitdir = join(r"F:\insilico_exps\GAN_Evol_cmp", unitname)
    score_traj_dict = defaultdict(dict)
    for opti, optimname in enumerate(optimnames):
        # glob pattern for 5 digit number like scoresHessCMA_04910.npz
        score_npzs = list(Path(unitdir).glob(f"scores{optimname}_[0-9][0-9][0-9][0-9][0-9].npz"))
        # %%
        score_traj_dict[optimname]["scores_mean"] = []
        score_traj_dict[optimname]["scores_std"] = []
        score_traj_dict[optimname]["scores_sem"] = []
        # use a datastructure to store the score stats for each run
        for runi, score_npz in enumerate(score_npzs):
            print(score_npz)
            scoredata = np.load(join(unitdir, score_npz))
            scores_vec = scoredata["scores_all"]
            gen_vec = scoredata["generations"]
            generation = np.unique(gen_vec)
            #%%
            # mean and std of scores for each value in generations
            scores_mean = np.array([scores_vec[gen_vec == gen].mean() for gen in np.unique(generation)])
            scores_std = np.array([scores_vec[gen_vec == gen].std() for gen in np.unique(generation)])
            scores_sem = np.array([scores_vec[gen_vec == gen].std() / np.sqrt(len(scores_vec[gen_vec == gen])) for gen in np.unique(generation)])
            scores_q25 = np.array([np.quantile(scores_vec[gen_vec == gen], 0.25) for gen in np.unique(generation)])
            scores_q75 = np.array([np.quantile(scores_vec[gen_vec == gen], 0.75) for gen in np.unique(generation)])
            score_traj_dict[optimname]["scores_mean"].append(scores_mean)
            score_traj_dict[optimname]["scores_std"].append(scores_std)
            score_traj_dict[optimname]["scores_sem"].append(scores_sem)
        #%%
        score_traj_dict[optimname]["scores_mean"] = np.array(score_traj_dict[optimname]["scores_mean"])
        score_traj_dict[optimname]["scores_std"] = np.array(score_traj_dict[optimname]["scores_std"])
        score_traj_dict[optimname]["scores_sem"] = np.array(score_traj_dict[optimname]["scores_sem"])
    #%%
    pkl.dump(score_traj_dict, open(join(outdir, f"{unitname}_score_traj_dict.pkl"), "wb"))
    #%%
    figh = plt.figure()
    for opti, optimname in enumerate(optimnames):
        for runi, (scores_mean, scores_sem) in enumerate(zip(score_traj_dict[optimname]["scores_mean"],
                                                            score_traj_dict[optimname]["scores_sem"])):
            #% shaded errorbar
            generation = np.arange(len(scores_mean))
            plt.plot(generation, scores_mean, color=clrs[opti], alpha=0.5, lw=1.8,
                     label=optimname if runi == 0 else None)
            plt.fill_between(generation, scores_mean - scores_sem,
                             scores_mean + scores_sem, alpha=0.2, color=clrs[opti])
    # plt.fill_between(generation, scores_q25, scores_q75, alpha=0.3)
    plt.ylabel("Activation", fontsize=14)
    plt.xlabel("Generation", fontsize=14)
    plt.legend(fontsize=14)
    plt.title(unitname+" Evol traj comparison", fontsize=14)
    saveallforms(outdir, f"{unitname}_traj", figh)
    plt.show()

#%%
# get the first 3 colors in seaborn default color palette
# clrs = sns.color_palette("colorblind", 3)
# clrs = sns.color_palette("Set2", 3)
#% plot the clrs
# figh = plt.figure()
# for i, clr in enumerate(clrs):
#     plt.plot([0, 1], [i, i], color=clr, lw=5)
# plt.yticks([0, 1, 2], ["CholCMA", "HessCMA", "HessCMA500_fc6"])
# plt.savefig(join(outdir, "color_legend.png"))
# saveallforms(join(outdir, "color_legend.png"), figh, dpi=300)
# plt.show()