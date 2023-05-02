"""
Plot the neural activation as a function of evolution generation.
    on the 2d phase plane of the two spaces BigGAN DeePSim .
"""

import torch
import seaborn as sns
from scipy.stats import sem
from matplotlib import cm
from os.path import join
import numpy as np
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms, show_imgrid
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data, get_expstr, extract_evol_activation_array
from neuro_data_analysis.neural_data_utils import parse_meta, area_mapping
outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_activation_dynamics"
#%%
_, BFEStats = load_neural_data()

#%%
# data structure to contain a collection of trajectories
# each trajectory is an 1D array of length n_blocks
from collections import OrderedDict
from easydict import EasyDict as edict
from scipy.stats import ttest_ind, ttest_rel
resp_col = OrderedDict()
meta_col = OrderedDict()
#%%
for Expi in range(1, len(BFEStats) + 1):
    S = BFEStats[Expi - 1]
    if S["evol"] is None:
        continue
    expstr = get_expstr(BFEStats, Expi)
    print(expstr)
    Animal, expdate = parse_meta(S)
    ephysFN = S["meta"]['ephysFN']
    prefchan = int(S['evol']['pref_chan'][0])
    prefunit = int(S['evol']['unit_in_pref_chan'][0])
    visual_area = area_mapping(prefchan, Animal, expdate)
    spacenames = S['evol']['space_names']
    space1 = spacenames[0] if isinstance(spacenames[0], str) else spacenames[0][0]
    space2 = spacenames[1] if isinstance(spacenames[1], str) else spacenames[1][0]

    # load the evolution trajectory of each pair
    resp_arr0, bsl_arr0, gen_arr0, _, _, _ = extract_evol_activation_array(S, 0)
    resp_arr1, bsl_arr1, gen_arr1, _, _, _ = extract_evol_activation_array(S, 1)

    # if the lAST BLOCK has < 10 images, in either thread, then remove it
    if len(resp_arr0[-1]) < 10 or len(resp_arr1[-1]) < 10:
        resp_arr0 = resp_arr0[:-1]
        resp_arr1 = resp_arr1[:-1]
        bsl_arr0 = bsl_arr0[:-1]
        bsl_arr1 = bsl_arr1[:-1]
        gen_arr0 = gen_arr0[:-1]
        gen_arr1 = gen_arr1[:-1]

    resp_m_traj_0 = np.array([resp.mean() for resp in resp_arr0])
    resp_m_traj_1 = np.array([resp.mean() for resp in resp_arr1])
    resp_sem_traj_0 = np.array([sem(resp) for resp in resp_arr0])
    resp_sem_traj_1 = np.array([sem(resp) for resp in resp_arr1])
    bsl_m_traj_0 = np.array([bsl.mean() for bsl in bsl_arr0])
    bsl_m_traj_1 = np.array([bsl.mean() for bsl in bsl_arr1])

    # test the successfulness of the evolution
    # ttest between the last two blocks and the first two blocks
    t_endinit_0, p_endinit_0 = ttest_ind(np.concatenate(resp_arr0[-2:]), np.concatenate(resp_arr0[:2]))
    t_endinit_1, p_endinit_1 = ttest_ind(np.concatenate(resp_arr1[-2:]), np.concatenate(resp_arr1[:2]))
    # ttest between the max two blocks and the first two blocks
    max_id = np.argmax(resp_m_traj_0)
    max_id = max_id if max_id < len(resp_arr0) - 2 else len(resp_arr0) - 3
    t_maxinit_0, p_maxinit_0 = ttest_ind(np.concatenate(resp_arr0[max_id:max_id+2]), np.concatenate(resp_arr0[:2]))
    max_id = np.argmax(resp_m_traj_1)
    max_id = max_id if max_id < len(resp_arr1) - 2 else len(resp_arr1) - 3
    t_maxinit_1, p_maxinit_1 = ttest_ind(np.concatenate(resp_arr1[max_id:max_id+2]), np.concatenate(resp_arr1[:2]))

    meta_dict = edict(Animal=Animal, expdate=expdate, ephysFN=ephysFN, prefchan=prefchan, prefunit=prefunit,
                      visual_area=visual_area, space1=space1, space2=space2, blockN=len(resp_arr0))
    stat_dict = edict(t_endinit_0=t_endinit_0, p_endinit_0=p_endinit_0,
                    t_endinit_1=t_endinit_1, p_endinit_1=p_endinit_1,
                    t_maxinit_0=t_maxinit_0, p_maxinit_0=p_maxinit_0,
                    t_maxinit_1=t_maxinit_1, p_maxinit_1=p_maxinit_1)
    meta_dict.update(stat_dict)

    # stack the trajectories together
    resp_bunch = np.stack([resp_m_traj_0, resp_m_traj_1,
                           resp_sem_traj_0, resp_sem_traj_1,
                           bsl_m_traj_0, bsl_m_traj_1, ], axis=1)
    resp_col[Expi] = resp_bunch
    meta_col[Expi] = meta_dict
#%%
# get the longest trajectory
max_len = max([resp_bunch.shape[0] for resp_bunch in resp_col.values()])
# extrapolate the last block with the mean of last two blocks
resp_extrap_col = OrderedDict()
for Expi, resp_bunch in resp_col.items():
    n_blocks = resp_bunch.shape[0]
    if n_blocks < max_len:
        extrap_vals = resp_bunch[-2:, :].mean(axis=0)
        resp_bunch = np.concatenate([resp_bunch,
             np.tile(extrap_vals, (max_len - n_blocks, 1))], axis=0)
    resp_extrap_col[Expi] = resp_bunch

# concatenate all trajectories
resp_extrap_arr = np.stack([*resp_extrap_col.values()], axis=0)
#%%
meta_df = pd.DataFrame.from_dict(meta_col, orient="index")
#%%
meta_df.to_csv(join(outdir, "meta_stats.csv"))
np.save(join(outdir, "resp_traj_extrap_arr.npy"), resp_extrap_arr)
pkl.dump({"resp_col": resp_col, "meta_col": meta_col}, open(join(outdir, "resp_traj_col.pkl"), "wb"))
#%%
# plot an animaion of the evolution of the trajectories
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from matplotlib import rc
# export to mp4
#%%
def animate_activation_dynamics(resp_mean_tsr, resp_sem_tsr, animname, outdir, interval=500, xylim=(-25, 375), alpha=0.5, linetrace=False):
    max_len = resp_mean_tsr.shape[1]
    figh = plt.figure(figsize=(8, 8))
    def animation_func(i):
        plt.cla()
        plt.plot(resp_mean_tsr[:, i, 0], resp_mean_tsr[:, i, 1], 'o', label=str(i), color="k", alpha=alpha)
        # add errorbar x, y
        plt.errorbar(resp_mean_tsr[:, i, 0], resp_mean_tsr[:, i, 1],
                     xerr=resp_sem_tsr[:, i, 0], yerr=resp_sem_tsr[:, i, 1], fmt="o", color="k", alpha=alpha)
        if linetrace:
            plt.plot(resp_mean_tsr[:, [0, i], 0].T, resp_mean_tsr[:, [0, i], 1].T, color="k", alpha=0.1)
        # add diagonal
        plt.gca().axline([0, 0], slope=1, color="black", linestyle="--")
        plt.xlim(xylim)
        plt.ylim(xylim)
        plt.title(f"gen {i}", fontsize=16)
        plt.xlabel("DeePSim", fontsize=16)
        plt.ylabel("BigGAN", fontsize=16)

    anim = FuncAnimation(figh, animation_func, frames=max_len, interval=interval)
    anim.save(join(outdir, f"{animname}.gif"))
    anim.save(join(outdir, f"{animname}.mp4"))
    plt.close(figh)
    return anim


def animate_activation_dynamics_with_masks(resp_mean_tsr, resp_sem_tsr, masktuples, animname, outdir, interval=500,
                                             xylim=(-25, 375), alpha=0.5, linetrace=False):
    """ Advanced version of animate_activation_dynamics, with masks and labels
    masktuples: list of (mask, color, label_str)
    """
    max_len = resp_mean_tsr.shape[1]
    figh = plt.figure(figsize=(8, 8))

    def animation_func(i):
        plt.cla()
        for masktuple in masktuples:
            mask, color, label = masktuple
            plt.plot(resp_mean_tsr[mask, i, 0], resp_mean_tsr[mask, i, 1], 'o', label=label, color=color, alpha=alpha)
            # add errorbar x, y
            plt.errorbar(resp_mean_tsr[mask, i, 0], resp_mean_tsr[mask, i, 1],
                         xerr=resp_sem_tsr[mask, i, 0], yerr=resp_sem_tsr[mask, i, 1], fmt="o", color=color, alpha=alpha)
            if linetrace:
                plt.plot(resp_mean_tsr[mask, :, 0][:, [0, i]].T, resp_mean_tsr[mask, :, 1][:, [0, i]].T, color=color, alpha=0.1)
        # add diagonal
        plt.gca().axline([0, 0], slope=1, color="black", linestyle="--")
        plt.xlim(xylim)
        plt.ylim(xylim)
        plt.title(f"gen {i}", fontsize=16)
        plt.xlabel("DeePSim", fontsize=16)
        plt.ylabel("BigGAN", fontsize=16)

    anim = FuncAnimation(figh, animation_func, frames=max_len, interval=interval)
    anim.save(join(outdir, f"{animname}.gif"))
    anim.save(join(outdir, f"{animname}.mp4"))
    plt.close(figh)
    return anim


def plot_activation_dynamics(resp_mean_tsr, resp_sem_tsr, masktuples, animname, outdir,
                             xylim=(-25, 375), alpha=0.5, endonly=False, errorbar=False):
    """ static plot of the activation dynamics

    masktuples: list of (mask, color, label_str)
    endonly:  if True, only plot the first and last point;
              else, plot all points during the evolution
    errorbar: if True, plot errorbar at the end point. (only works when endonly=True)

    animname: name of the figure to be saved
    """
    figh = plt.figure(figsize=(8, 8))
    for masktuple in masktuples:
        mask, color, label = masktuple
        if endonly:
            plt.plot(resp_mean_tsr[mask, :, 0][:, [0, -1]].T, resp_mean_tsr[mask, :, 1][:, [0, -1]].T,
                     color=color, alpha=alpha)
            if errorbar:
                plt.errorbar(resp_mean_tsr[mask, -1, 0], resp_mean_tsr[mask, -1, 1],
                        xerr=resp_sem_tsr[mask, -1, 0], yerr=resp_sem_tsr[mask, -1, 1], fmt="o", color=color, alpha=alpha)
                plt.errorbar(resp_mean_tsr[mask, 0, 0], resp_mean_tsr[mask, 0, 1],
                        xerr=resp_sem_tsr[mask, 0, 0], yerr=resp_sem_tsr[mask, 0, 1], fmt="o", color=color, alpha=alpha)
        else:
            plt.plot(resp_mean_tsr[mask, :, 0].T, resp_mean_tsr[mask, :, 1].T, color=color, alpha=alpha)

    # add diagonal
    plt.gca().axline([0, 0], slope=1, color="black", linestyle="--")
    plt.xlim(xylim)
    plt.ylim(xylim)
    plt.xlabel("DeePSim", fontsize=16)
    plt.ylabel("BigGAN", fontsize=16)
    saveallforms(outdir, f"{animname}", figh)
    # plt.close(figh)
    return figh
#%%
anim = animate_activation_dynamics(resp_extrap_arr[:, :, :2], resp_extrap_arr[:, :, 2:4], "resp_traj", outdir,
                                   interval=500, xylim=(-25, 375))
#%%
norm_max = resp_extrap_arr[:, :, :2].max(axis=(1,2))[:, None, None]
norm_min = 0.0  # resp_extrap_arr[:, :, -2:].mean(axis=(1,2))[:, None, None]
scaling = (norm_max - norm_min)
normresp_extrap_arr = np.concatenate([(resp_extrap_arr[:, :, :2] - norm_min) / scaling,
                                      (resp_extrap_arr[:, :, 2:4]) / scaling,], axis=-1)
anim = animate_activation_dynamics(normresp_extrap_arr[:, :, :2], normresp_extrap_arr[:, :, 2:4], "maxnorm_resp_traj",
                                   outdir, interval=500, xylim=(-0.2, 1.0))
#%%
norm_max = resp_extrap_arr[:, :, :2].max(axis=(1,2))[:, None, None]
norm_min = resp_extrap_arr[:, :, -2:].mean(axis=(1,2))[:, None, None]
scaling = (norm_max - norm_min)
normresp_extrap_arr = np.concatenate([(resp_extrap_arr[:, :, :2] - norm_min) / scaling,
                                      (resp_extrap_arr[:, :, 2:4]) / scaling,], axis=-1)
anim = animate_activation_dynamics(normresp_extrap_arr[:, :, :2], normresp_extrap_arr[:, :, 2:4],
                                   "maxbslnorm_resp_traj", outdir, interval=500, xylim=(-0.25, 1.0))
#%%
#%% Visualizing dynamics with masks
Amsk  = meta_df.Animal == "Alfa"
Bmsk  = meta_df.Animal == "Beto"
V1msk = meta_df.visual_area == "V1"
V4msk = meta_df.visual_area == "V4"
ITmsk = meta_df.visual_area == "IT"
validmsk = (meta_df.blockN > 14)
sucsmsk = (meta_df.p_maxinit_0 < 0.05) | (meta_df.p_maxinit_1 < 0.05)
#%%
baseline_jump_list = ["Beto-18082020-002",
                      "Beto-07092020-006",
                      "Beto-14092020-002",
                      "Beto-27102020-003",
                      "Alfa-22092020-003",
                      "Alfa-04092020-003"]
bsl_unstable_msk = meta_df.ephysFN.str.contains("|".join(baseline_jump_list), case=True, regex=True)
bsl_stable_msk = ~bsl_unstable_msk
validmsk = (meta_df.blockN > 14) & bsl_stable_msk

#%%
masktuples = [(V1msk & validmsk & sucsmsk, "red", "V1"),
              (V4msk & validmsk & sucsmsk, "green", "V4"),
              (ITmsk & validmsk & sucsmsk, "blue", "IT")]
#%%
anim = animate_activation_dynamics_with_masks(resp_extrap_arr[:, :, :2], resp_extrap_arr[:, :, 2:4],
                                              [(validmsk & sucsmsk, "k", "all")], "resp_traj_allsucs", outdir,
                                              interval=500, xylim=(-25, 375), alpha=0.3, )
anim = animate_activation_dynamics_with_masks(resp_extrap_arr[:, :, :2], resp_extrap_arr[:, :, 2:4],
                                              [(validmsk & sucsmsk, "k", "all")], "resp_traj_allsucs_trace", outdir,
                                              interval=500, xylim=(-25, 375), alpha=0.3, linetrace=True)
#%%
anim = animate_activation_dynamics_with_masks(resp_extrap_arr[:, :, :2], resp_extrap_arr[:, :, 2:4], masktuples,
                                              "resp_traj_area_sep_sucs", outdir, interval=500, xylim=(-25, 375),
                                              alpha=0.3)
anim = animate_activation_dynamics_with_masks(resp_extrap_arr[:, :, :2], resp_extrap_arr[:, :, 2:4], masktuples,
                                              "resp_traj_area_sep_sucs_trace", outdir, interval=500, xylim=(-25, 375),
                                              alpha=0.3, linetrace=True)
for masktuple in masktuples:
    _, _, label = masktuple
    plot_activation_dynamics(resp_extrap_arr[:, :, :2], resp_extrap_arr[:, :, 2:4], [masktuple],
                                f"resp_traj_{label}_sucs", outdir,
                             xylim=(-25, 375), alpha=0.3, endonly=True, errorbar=True)

plot_activation_dynamics(resp_extrap_arr[:, :, :2], resp_extrap_arr[:, :, 2:4], masktuples,
                                f"resp_traj_area_sep_sucs", outdir,
                             xylim=(-25, 375), alpha=0.3, endonly=True, errorbar=False)
plot_activation_dynamics(resp_extrap_arr[:, :, :2], resp_extrap_arr[:, :, 2:4], masktuples,
                                f"resp_traj_area_sep_sucserr", outdir,
                             xylim=(-25, 375), alpha=0.3, endonly=True, errorbar=True)
#%%
norm_max = resp_extrap_arr[:, :, :2].max(axis=(1,2))[:, None, None]
norm_min = 0.0  # resp_extrap_arr[:, :, -2:].mean(axis=(1,2))[:, None, None]
scaling = (norm_max - norm_min)
normresp_extrap_arr = np.concatenate([(resp_extrap_arr[:, :, :2] - norm_min) / scaling,
                                      (resp_extrap_arr[:, :, 2:4]) / scaling, ], axis=-1)
anim = animate_activation_dynamics_with_masks(normresp_extrap_arr[:, :, :2], normresp_extrap_arr[:, :, 2:4], masktuples,
                                              "maxnorm_resp_traj_area_sep_sucs", outdir, interval=500,
                                              xylim=(-0.2, 1.0), alpha=0.3)
anim = animate_activation_dynamics_with_masks(normresp_extrap_arr[:, :, :2], normresp_extrap_arr[:, :, 2:4], masktuples,
                                              "maxnorm_resp_traj_area_sep_sucs_trace", outdir, interval=500,
                                              xylim=(-0.2, 1.0), alpha=0.3, linetrace=True)

# plot_activation_dynamics(normresp_extrap_arr[:, :, :2], normresp_extrap_arr[:, :, 2:4], masktuples,
#                          "maxnorm_resp_traj_area_sep_sucs_curv", outdir, xylim=(-0.2, 1.0), alpha=0.3)
plot_activation_dynamics(normresp_extrap_arr[:, :, :2], normresp_extrap_arr[:, :, 2:4], masktuples,
                         "maxnorm_resp_traj_area_sep_sucs_initend", outdir, xylim=(-0.2, 1.0), alpha=0.3, endonly=True)
plot_activation_dynamics(normresp_extrap_arr[:, :, :2], normresp_extrap_arr[:, :, 2:4], masktuples,
                         "maxnorm_resp_traj_area_sep_sucs_initenderr", outdir, xylim=(-0.2, 1.0), alpha=0.3, endonly=True, errorbar=True)
for masktuple in masktuples:
    _, _, label = masktuple
    plot_activation_dynamics(normresp_extrap_arr[:, :, :2], normresp_extrap_arr[:, :, 2:4], [masktuple],
                             f"maxnorm_resp_traj_{label}_sucs_initend", outdir,
                             xylim=(-0.2, 1.0), alpha=0.3, endonly=True, errorbar=True)
#%%
norm_max = resp_extrap_arr[:, :, :2].max(axis=(1, 2))[:, None, None]
norm_min = resp_extrap_arr[:, :, -2:].mean(axis=(1, 2))[:, None, None]
scaling = (norm_max - norm_min)
normresp_extrap_arr = np.concatenate([(resp_extrap_arr[:, :, :2] - norm_min) / scaling,
                                        (resp_extrap_arr[:, :, 2:4]) / scaling, ], axis=-1)
anim = animate_activation_dynamics_with_masks(normresp_extrap_arr[:, :, :2], normresp_extrap_arr[:, :, 2:4],
                masktuples, "maxbslnorm_resp_traj_area_sep_sucs", outdir, interval=500, xylim=(-0.25, 1.0), alpha=0.3)
anim = animate_activation_dynamics_with_masks(normresp_extrap_arr[:, :, :2], normresp_extrap_arr[:, :, 2:4], masktuples,
                                              "maxbslnorm_resp_traj_area_sep_sucs_trace", outdir, interval=500,
                                              xylim=(-0.25, 1.0), alpha=0.3, linetrace=True)
for masktuple in masktuples:
    _, _, label = masktuple
    plot_activation_dynamics(normresp_extrap_arr[:, :, :2], normresp_extrap_arr[:, :, 2:4], [masktuple],
                                f"maxbslnorm_resp_traj_{label}_sucs_initend", outdir,
                                xylim=(-0.25, 1.0), alpha=0.3, endonly=True, errorbar=True)
#%%
figh = plt.figure(figsize=(8, 8))
plt.plot(resp_extrap_arr[:, [0,-1], 0].T, resp_extrap_arr[:, [0,-1], 1].T, color="k", alpha=0.1)
plt.gca().axline([0, 0], slope=1, color="black", linestyle="--")
plt.gca().set_aspect("equal")
plt.axis("image")
saveallforms(outdir, "resp_traj_lines", figh, ["png"])
plt.show()
#%%
plt.figure(figsize=(8, 8))
plt.plot(resp_m_traj_0, resp_m_traj_1, label="0")
# add errorbar x, y
plt.errorbar(resp_m_traj_0, resp_m_traj_1,
             xerr=resp_sem_traj_0, yerr=resp_sem_traj_1, fmt="o", color="k", alpha=0.5)
# add diagonal
plt.gca().axline([0, 0], slope=1, color="black", linestyle="--")
# plt.gca().set_aspect("equal")
plt.axis("image")
plt.xlabel("fc6")
plt.ylabel("BigGAN")
plt.show()




#%% Scratch space
for geni in range(max_len):
    plt.figure(figsize=(8, 8))
    plt.plot(resp_extrap_arr[:, geni, 0], resp_extrap_arr[:, geni, 1], 'o', label=str(geni), color="k", alpha=0.5)
    # add errorbar x, y
    plt.errorbar(resp_extrap_arr[:, geni, 0], resp_extrap_arr[:, geni, 1],
            xerr=resp_extrap_arr[:, geni, 2], yerr=resp_extrap_arr[:, geni, 3], fmt="o", color="k", alpha=0.5)
    # add diagonal
    plt.gca().axline([0, 0], slope=1, color="black", linestyle="--")
    plt.xlim([-25, 375])
    plt.ylim([-25, 375])
    plt.show()


#%%
rc('animation', html='jshtml')
figh = plt.figure(figsize=(8, 8))
plt.plot(resp_extrap_arr[:, 0, 0], resp_extrap_arr[:, 0, 1], 'o', label=str(0), color="k", alpha=0.5)
# add errorbar x, y
plt.errorbar(resp_extrap_arr[:, 0, 0], resp_extrap_arr[:, 0, 1],
        xerr=resp_extrap_arr[:, 0, 2], yerr=resp_extrap_arr[:, 0, 3], fmt="o", color="k", alpha=0.5)
# add diagonal
plt.gca().axline([0, 0], slope=1, color="black", linestyle="--")
plt.xlim([-25, 375])
plt.ylim([-25, 375])
plt.xlabel("DeePSim", fontsize=16)
plt.ylabel("BigGAN", fontsize=16)
# plt.show()
def animation_func(i):
    plt.cla()
    plt.plot(resp_extrap_arr[:, i, 0], resp_extrap_arr[:, i, 1], 'o', label=str(i), color="k", alpha=0.5)
    # add errorbar x, y
    plt.errorbar(resp_extrap_arr[:, i, 0], resp_extrap_arr[:, i, 1],
            xerr=resp_extrap_arr[:, i, 2], yerr=resp_extrap_arr[:, i, 3], fmt="o", color="k", alpha=0.5)
    # add diagonal
    plt.gca().axline([0, 0], slope=1, color="black", linestyle="--")
    plt.xlim([-25, 375])
    plt.ylim([-25, 375])
    plt.title(f"gen {i}", fontsize=16)
    plt.xlabel("DeePSim", fontsize=16)
    plt.ylabel("BigGAN", fontsize=16)

anim = FuncAnimation(figh, animation_func, frames=max_len, interval=500)
figh.close()
anim.save(join(outdir, "resp_traj.gif"))
anim.save(join(outdir, "resp_traj.mp4"))