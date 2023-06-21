import torch
import re
from pathlib import Path
import pickle as pkl
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms
from neuro_data_analysis.neural_data_utils import get_all_masks
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data, get_expstr
from core.utils.stats_utils import ttest_ind_print, ttest_rel_print, ttest_ind_print_df
_, BFEStats = load_neural_data()
saveroot = Path(r"E:\Network_Data_Sync\BigGAN_Evol_yolo")

tabdir = Path(r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables")
meta_df = pd.read_csv(tabdir / "meta_activation_stats.csv", index_col=0)
Amsk, Bmsk, V1msk, V4msk, ITmsk, length_msk, spc_msk, \
    sucsmsk, bsl_unstable_msk, bsl_stable_msk, validmsk = get_all_masks(meta_df)
bothsucmsk = (meta_df.p_maxinit_0 < 0.05) & (meta_df.p_maxinit_1 < 0.05)
FCsucsmsk = (meta_df.p_maxinit_0 < 0.05)
BGsucsmsk = (meta_df.p_maxinit_1 < 0.05)
#%%

GANimgtab = pd.read_csv(tabdir / 'GAN_samples_all_yolo_stats.csv', index_col=0)
GANimgtab["confidence_fill0"] = GANimgtab.confidence.fillna(0)
#%% Load in the yolo stats for evolution images
sumdir = (saveroot / "yolo_v5_summary")
all_df_col = []
for Expi in trange(1, 190+1):
    if BFEStats[Expi-1]["evol"] is None:
        continue

    df0 = pd.read_csv(sumdir / f"Exp{Expi:03d}_thread0_yolo_stats.csv", index_col=0)
    df1 = pd.read_csv(sumdir / f"Exp{Expi:03d}_thread1_yolo_stats.csv", index_col=0)
    df0["thread"] = 0
    df1["thread"] = 1
    df0["Expi"] = Expi
    df1["Expi"] = Expi
    all_df_col.append(df0)
    all_df_col.append(df1)

all_df = pd.concat(all_df_col)
# split all_df into name
all_df["img_name"] = all_df.img_path.apply(lambda x: x.split("\\")[-1])
# use re to match different part of the string "block001_thread000_gen_gen000_000009.bmp"
all_df["block"] = all_df.img_name.apply(lambda x: int(re.findall(r"block(\d+)", x)[0]))
all_df["imgid"] = all_df.img_name.apply(lambda x: int(re.findall(r"gen\d+_(\d+)", x)[0]))
# fill in zeros for nan confidence (which means no object detected)
all_df["confidence_fill0"] = all_df.confidence.fillna(0)
# use the visual area of the corresponding Expi in metadf in all_df
all_df["visual_area"] = all_df.Expi.apply(lambda x: meta_df.loc[x, "visual_area"])

EXCLUSION_MINBLOCKSIZE = 10
all_df["included"] = True
all_df["maxblock"] = all_df.groupby("Expi")["block"].transform(max)
for Expi in tqdm(meta_df.index):
    part_df0 = all_df[(all_df.Expi == Expi) & (all_df.thread == 0)]
    part_df1 = all_df[(all_df.Expi == Expi) & (all_df.thread == 1)]
    maxblock = max(part_df0.block.max(), part_df1.block.max())
    if (part_df0.block == maxblock).sum() < EXCLUSION_MINBLOCKSIZE or \
       (part_df1.block == maxblock).sum() < EXCLUSION_MINBLOCKSIZE:
        print(f"Expi {Expi} last block {maxblock} has less than 10 images")
        last_block_msk = (all_df.Expi == Expi) & (all_df.block == maxblock)
        # all_df[last_block_msk]["included"] = False
        all_df.loc[last_block_msk, "included"] = False
        print(f"excluded {last_block_msk.sum()} images from analysis")
        maxblock = maxblock - 1

    all_df[all_df.Expi == Expi]["maxblock"] = maxblock


#%%
all_df.to_csv(sumdir / f"Evol_invivo_all_yolo_v5_stats.csv")
all_df.to_csv(tabdir / f"Evol_invivo_all_yolo_v5_stats.csv")
#%%
all_df = pd.read_csv(tabdir / f"Evol_invivo_all_yolo_v5_stats.csv", index_col=0)
#%%
figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_proto_Yolo_stats"
#%%\
for area in ["V4", "IT"]:
    for GANname, thread in zip(["DeePSim", "BigGAN"], [0, 1]):
        for success_str, success_mask in zip(["Both", "Any", "None"], [bothsucmsk, sucsmsk, ~sucsmsk]):
            titlestr = f"{area} {GANname} {success_str} Success"
            commonmsk = (all_df.visual_area == area) & \
                        (all_df.included) & (all_df.thread == thread) & \
                        (all_df.Expi.isin(meta_df[validmsk & success_mask].index))
            fig, ax = plt.subplots(figsize=[5, 4])
            ax.hist(all_df.confidence_fill0[(all_df.block < 4) & commonmsk], bins=100, alpha=0.5, density=True)
            ax.hist(all_df.confidence_fill0[(all_df.block > all_df.maxblock - 4) & commonmsk], bins=100, alpha=0.5, density=True)
            tval, pval, result_str = ttest_ind_print_df(all_df, (all_df.block < 4) & commonmsk,
                               (all_df.block > all_df.maxblock - 4) & commonmsk, "confidence_fill0", )
            ax.set_xlabel("Confidence score")
            ax.set_ylabel("Density")
            plt.legend(["First 4 blocks", "Last 4 blocks"])
            # "Confidence score distribution of YOLOv5 detections"
            ax.set_title(f"YOLOv5 confidence score through Evolution\n[{titlestr}]\n"+result_str.replace('tval','\ntval')) # t={tval:.3f}, p={pval:.1e}
            plt.tight_layout()
            saveallforms(figdir, f"yolo_confidence_score_dist_init_end_{area}_{GANname}_{success_str}_suc", fig, fmts=["png", "pdf"])
            # fig.savefig(tabdir / "yolo_confidence_score_dist.png")
            plt.show()
#%%
commonmsk = (all_df.visual_area == "V4") & (all_df.thread == 0) & \
            (all_df.included) & \
            (all_df.Expi.isin(meta_df[validmsk & ~sucsmsk].index))
# ttest_ind_print_df(all_df, (all_df.block < 3) & commonmsk, (all_df.block > 20) & commonmsk, "confidence_fill0",)
ttest_ind_print_df(all_df, (all_df.block < 4) & commonmsk,
                   (all_df.block > all_df.maxblock - 4) & commonmsk, "confidence_fill0", )

#%%
#%%
for area in ["V4", "IT"]:
    for GANname, thread in zip(["DeePSim", "BigGAN"], [0, 1]):
        for success_str, success_mask in zip(["Both", "Any", "None"], [bothsucmsk, sucsmsk, ~sucsmsk]):
            titlestr = f"{area} {GANname} {success_str} Success"
            commonmsk = (all_df.visual_area == area) & \
                        (all_df.included) & (all_df.thread == thread) & \
                        (all_df.Expi.isin(meta_df[validmsk & success_mask].index))
            fig, ax = plt.subplots(figsize=[5, 4])
            ax.hist(all_df.confidence_fill0[(all_df.block < 4) & commonmsk], bins=100, alpha=0.5, density=True)
            ax.hist(all_df.confidence_fill0[(all_df.block > all_df.maxblock - 4) & commonmsk], bins=100, alpha=0.5, density=True)
            if GANname == "DeePSim":
                ax.hist(GANimgtab.confidence_fill0[GANimgtab.imgdir_name == "DeePSim_4std"], bins=100, alpha=0.5, density=True)
            elif GANname == "BigGAN":
                ax.hist(GANimgtab.confidence_fill0[GANimgtab.imgdir_name == 'BigGAN_std_008'], bins=100, alpha=0.5, density=True)
            tval, pval, result_str = ttest_ind_print_df(all_df, (all_df.block < 4) & commonmsk,
                               (all_df.block > all_df.maxblock - 4) & commonmsk, "confidence_fill0", )
            ax.set_xlabel("Confidence score")
            ax.set_ylabel("Density")
            plt.legend(["First 4 blocks", "Last 4 blocks", f"{GANname} samples"])
            # "Confidence score distribution of YOLOv5 detections"
            ax.set_title(f"YOLOv5 confidence score through Evolution\n[{titlestr}]\n"+result_str.replace('tval','\ntval')) # t={tval:.3f}, p={pval:.1e}
            plt.tight_layout()
            saveallforms(figdir, f"yolo_confidence_score_dist_init_end_{area}_{GANname}_{success_str}_suc_with_GANref", fig, fmts=["png", "pdf"])
            # fig.savefig(tabdir / "yolo_confidence_score_dist.png")
            plt.show()
#%%
for GANname, thread in zip(["DeePSim", "BigGAN"], [0, 1]):
    for success_str, success_mask in zip(["Both", "Any", "None"], [bothsucmsk, sucsmsk, ~sucsmsk]):
        for value_str in ["confidence_fill0", "confidence"]:
            titlestr = f"V4 vs IT {GANname} {success_str} Success"
            commonmsk1 = (all_df.visual_area == "V4") & \
                        (all_df.included) & (all_df.thread == thread) & \
                        (all_df.Expi.isin(meta_df[validmsk & success_mask].index))
            commonmsk2 = (all_df.visual_area == "IT") & \
                        (all_df.included) & (all_df.thread == thread) & \
                        (all_df.Expi.isin(meta_df[validmsk & success_mask].index))
            fig, ax = plt.subplots(figsize=[5, 4])
            ax.hist(all_df[value_str][(all_df.block > all_df.maxblock - 4) & commonmsk1], bins=100, alpha=0.5, density=True)
            ax.hist(all_df[value_str][(all_df.block > all_df.maxblock - 4) & commonmsk2], bins=100, alpha=0.5, density=True)
            if GANname == "DeePSim":
                ax.hist(GANimgtab[value_str][GANimgtab.imgdir_name == "DeePSim_4std"], bins=100, alpha=0.5, density=True)
            elif GANname == "BigGAN":
                ax.hist(GANimgtab[value_str][GANimgtab.imgdir_name == 'BigGAN_std_008'], bins=100, alpha=0.5, density=True)
            tval, pval, result_str = ttest_ind_print_df(all_df, (all_df.block > all_df.maxblock - 4) & commonmsk1,
                               (all_df.block > all_df.maxblock - 4) & commonmsk2, value_str, )
            ax.set_xlabel("Confidence score")
            ax.set_ylabel("Density")
            plt.legend(["V4", "IT", f"{GANname} samples"])
            # "Confidence score distribution of YOLOv5 detections"
            ax.set_title(f"YOLOv5 confidence score through Evolution\n[{titlestr}]\n"+result_str.replace('tval','\ntval')) # t={tval:.3f}, p={pval:.1e}
            plt.tight_layout()
            saveallforms(figdir, f"yolo_{value_str}_score_dist_V4_vs_IT_{GANname}_{success_str}_suc_with_GANref", fig, fmts=["png", "pdf"])
            # fig.savefig(tabdir / "yolo_confidence_score_dist.png")
            plt.show()

#%%
for GANname, thread in zip(["DeePSim", "BigGAN"], [0, 1]):
    for success_str, success_mask in zip(["Both", "Any", "None"], [bothsucmsk, sucsmsk, ~sucsmsk]):
        for value_str in ["confidence_fill0", "confidence"]:
            titlestr = f"V4 vs IT {GANname} {success_str} Success"
            commonmsk1 = (all_df.visual_area == "V4") & \
                         (all_df.included) & (all_df.thread == thread) & \
                         (all_df.Expi.isin(meta_df[validmsk & success_mask].index))
            commonmsk2 = (all_df.visual_area == "IT") & \
                         (all_df.included) & (all_df.thread == thread) & \
                         (all_df.Expi.isin(meta_df[validmsk & success_mask].index))
            fig, ax = plt.subplots(figsize=[5, 4])
            ax.hist(all_df[value_str][(all_df.block > all_df.maxblock - 4) & commonmsk1], bins=100, alpha=0.5, density=True)
            ax.hist(all_df[value_str][(all_df.block > all_df.maxblock - 4) & commonmsk2], bins=100, alpha=0.5, density=True)
            if GANname == "DeePSim":
                ax.hist(GANimgtab[value_str][GANimgtab.imgdir_name == "DeePSim_4std"], bins=100, alpha=0.5, density=True)
            elif GANname == "BigGAN":
                ax.hist(GANimgtab[value_str][GANimgtab.imgdir_name == 'BigGAN_std_008'], bins=100, alpha=0.5, density=True)
            tval, pval, result_str = ttest_ind_print_df(all_df, (all_df.block > all_df.maxblock - 4) & commonmsk1,
                               (all_df.block > all_df.maxblock - 4) & commonmsk2, value_str, )
            ax.set_xlabel("Confidence score")
            ax.set_ylabel("Density")
            plt.legend(["V4", "IT", f"{GANname} samples"])
            # "Confidence score distribution of YOLOv5 detections"
            ax.set_title(f"YOLOv5 confidence score through Evolution\n[{titlestr}]\n"+result_str.replace('tval','\ntval')) # t={tval:.3f}, p={pval:.1e}
            plt.tight_layout()
            saveallforms(figdir, f"yolo_{value_str}_score_dist_V4_vs_IT_{GANname}_{success_str}_suc_with_GANref", fig, fmts=["png", "pdf"])
            # fig.savefig(tabdir / "yolo_confidence_score_dist.png")
            plt.show()
#%%
import seaborn as sns
all_df["block_split"] = np.nan
all_df.loc[all_df.block <= 5, "block_split"] = "init 5"
all_df.loc[all_df.block >= all_df.maxblock - 4, "block_split"] = "final 5"
all_df["detected"] = ~ all_df["confidence"].isna()
all_df["non_detected"] = all_df["confidence"].isna()
#%%
# value_str = "confidence_fill0"
""" DeePSim GAN and BigGAN with both success mask """
success_mask = bothsucmsk #sucsmsk
success_str = "Both"
for with_bar in [True, False]:
    for value_str in ["confidence_fill0", "confidence"]:
        for thread, GANname in zip([0, 1], ["DeePSim", "BigGAN"]):
            plt.figure(figsize=[4, 5])
            if GANname == "DeePSim":
                ref_dist = GANimgtab[value_str][GANimgtab.imgdir_name == "DeePSim_4std"]
            elif GANname == "BigGAN":
                ref_dist = GANimgtab[value_str][GANimgtab.imgdir_name == 'BigGAN_std_008']
            else:
                raise ValueError("GANname not recognized")
            plt.axhline(y=ref_dist.median(), color='r', linestyle='-', label=GANname + " dist")
            plt.axhline(y=ref_dist.quantile(0.25), color='r', linestyle=':', label=GANname + " 25%")
            plt.axhline(y=ref_dist.quantile(0.75), color='r', linestyle=':', label=GANname + " 75%")
            # plt.axhline(y=ref_dist.mean(), color='r', linestyle='-', label=GANname+" dist")
            # plt.axhline(y=ref_dist.mean() + ref_dist.sem(), color='r', linestyle='--', label="25%")
            # plt.axhline(y=ref_dist.mean() - ref_dist.sem(), color='r', linestyle='--', label="75%")

            commonmsk = (all_df.included) & (all_df.thread == thread) &\
                        (all_df.Expi.isin(meta_df[validmsk & success_mask].index))
            if with_bar:
                sns.barplot(data=all_df[commonmsk], x="visual_area", y="non_detected",
                                hue="block_split", order=["V4", "IT"], alpha=0.45, capsize=.2,
                                errwidth=1, )
            ax = sns.violinplot(data=all_df[commonmsk], x="visual_area", y=value_str,
                                hue="block_split", cut=0, bw=0.05,
                                order=["V4", "IT"])
            for violin in (ax.collections[::2]):
                violin.set_alpha(0.5)
            # sns.pointplot(data=all_df[commonmsk], x="visual_area", y=value_str,
            #                     hue="block_split", linestyles="none",
            #                     order=["V4", "IT"])
            plt.suptitle(f"Objectness distribution of\nEvolved Image for V4 and IT\n"
                         f"[{GANname}, {success_str} Success]")
            plt.gca().get_legend().set_title("block")
            plt.ylim([0, 1])
            saveallforms(figdir, f"yolo_{value_str}_score_dist_V4_vs_IT_{GANname}_{success_str}_suc_violin{'bar' if with_bar else ''}", plt.gcf(), fmts=["png", "pdf"])
            plt.show()
#%%
""" DeePSim GAN with DeePSim success mask
BigGAN with BigGAN success mask """
for with_bar in [True, False]:
    for value_str in ["confidence_fill0", "confidence"]:
        for thread, GANname, success_mask in zip([0, 1], ["DeePSim", "BigGAN"],
                                                 [FCsucsmsk, BGsucsmsk]):
            success_str = GANname
            plt.figure(figsize=[4, 5])
            if GANname == "DeePSim":
                ref_dist = GANimgtab[value_str][GANimgtab.imgdir_name == "DeePSim_4std"]
            elif GANname == "BigGAN":
                ref_dist = GANimgtab[value_str][GANimgtab.imgdir_name == 'BigGAN_std_008']
            else:
                raise ValueError("GANname not recognized")
            plt.axhline(y=ref_dist.median(), color='r', linestyle='-', label=GANname + " dist")
            plt.axhline(y=ref_dist.quantile(0.25), color='r', linestyle=':', label=GANname + " 25%")
            plt.axhline(y=ref_dist.quantile(0.75), color='r', linestyle=':', label=GANname + " 75%")

            commonmsk = (all_df.included) & (all_df.thread == thread) &\
                        (all_df.Expi.isin(meta_df[validmsk & success_mask].index))
            if with_bar:
                sns.barplot(data=all_df[commonmsk], x="visual_area", y="non_detected",
                                hue="block_split", order=["V4", "IT"], alpha=0.45, capsize=.2,
                                errwidth=1, )
            ax = sns.violinplot(data=all_df[commonmsk], x="visual_area", y=value_str,
                                hue="block_split", cut=0, bw=0.05,
                                order=["V4", "IT"])
            for violin in (ax.collections[::2]):
                violin.set_alpha(0.5)
            plt.suptitle(f"Objectness distribution of\nEvolved Image for V4 and IT\n"
                         f"[{GANname}, {success_str} Success]")
            plt.gca().get_legend().set_title("block")
            plt.ylim([0, 1])
            saveallforms(figdir, f"yolo_{value_str}_score_dist_V4_vs_IT_{GANname}_{success_str}_suc_violin{'bar' if with_bar else ''}", plt.gcf(), fmts=["png", "pdf"])
            plt.show()
#%%
""" DeePSim GAN with DeePSim fail mask
BigGAN with BigGAN fail mask """
for with_bar in [True, False]:
    for value_str in ["confidence_fill0", "confidence"]:
        for thread, GANname, success_mask in zip([0, 1], ["DeePSim", "BigGAN"],
                                                 [~FCsucsmsk, ~BGsucsmsk]):
            success_str = GANname
            plt.figure(figsize=[4, 5])
            if GANname == "DeePSim":
                ref_dist = GANimgtab[value_str][GANimgtab.imgdir_name == "DeePSim_4std"]
            elif GANname == "BigGAN":
                ref_dist = GANimgtab[value_str][GANimgtab.imgdir_name == 'BigGAN_std_008']
            else:
                raise ValueError("GANname not recognized")
            plt.axhline(y=ref_dist.median(), color='r', linestyle='-', label=GANname + " dist")
            plt.axhline(y=ref_dist.quantile(0.25), color='r', linestyle=':', label=GANname + " 25%")
            plt.axhline(y=ref_dist.quantile(0.75), color='r', linestyle=':', label=GANname + " 75%")

            commonmsk = (all_df.included) & (all_df.thread == thread) &\
                        (all_df.Expi.isin(meta_df[validmsk & success_mask].index))
            if with_bar:
                sns.barplot(data=all_df[commonmsk], x="visual_area", y="non_detected",
                                hue="block_split", order=["V4", "IT"], alpha=0.45, capsize=.2,
                                errwidth=1, )
            ax = sns.violinplot(data=all_df[commonmsk], x="visual_area", y=value_str,
                                hue="block_split", cut=0, bw=0.05,
                                order=["V4", "IT"])
            for violin in (ax.collections[::2]):
                violin.set_alpha(0.5)
            plt.suptitle(f"Objectness distribution of\nEvolved Image for V4 and IT\n"
                         f"[{GANname}, {success_str} Failed]")
            plt.gca().get_legend().set_title("block")
            plt.ylim([0, 1])
            saveallforms(figdir, f"yolo_{value_str}_score_dist_V4_vs_IT_{GANname}_{success_str}_fail_violin{'bar' if with_bar else ''}", plt.gcf(), fmts=["png", "pdf"])
            plt.show()

#%%
# redirect output to a text file.
import sys

sys.stdout = open(tabdir / "yolo_Evol_reference_stats.txt", "w")
for area in ["V4", "IT"]:
    for GANname, thread in zip(["DeePSim", "BigGAN"], [0, 1]):
        for success_str, success_mask in zip(["Both", "Any", "None"], [bothsucmsk, sucsmsk, ~sucsmsk]):
            titlestr = f"{area} {GANname} {success_str} Success"
            print(f"[{titlestr}]")
            commonmsk = (all_df.visual_area == area) & \
                        (all_df.included) & (all_df.thread == thread) & \
                        (all_df.Expi.isin(meta_df[validmsk & success_mask].index))
            if GANname == "DeePSim":
                GAN_ref_data = GANimgtab.confidence_fill0[GANimgtab.imgdir_name == "DeePSim_4std"]
            elif GANname == "BigGAN":
                GAN_ref_data = GANimgtab.confidence_fill0[GANimgtab.imgdir_name == 'BigGAN_std_008']
            else:
                raise NotImplementedError
            ttest_ind_print_df(all_df, (all_df.block < 5) & commonmsk,
                                 (all_df.block > all_df.maxblock - 5) & commonmsk, "confidence_fill0", )
            print(f"Initial 5 gen vs {GANname} reference", end=" ")
            ttest_ind_print(all_df.confidence_fill0[(all_df.block < 5) & commonmsk], GAN_ref_data, )
            print(f"Last 5 gen vs {GANname} reference", end=" ")
            ttest_ind_print(all_df.confidence_fill0[(all_df.block > all_df.maxblock - 5) & commonmsk], GAN_ref_data, )

# redirect output back to console
sys.stdout = sys.__stdout__

#%% form a confidence score tensor for experiments
confscore_col = []
for Expi in tqdm(meta_df.index):
    part_df0 = all_df[(all_df.Expi == Expi) & (all_df.thread == 0) & all_df.included]
    part_df1 = all_df[(all_df.Expi == Expi) & (all_df.thread == 1) & all_df.included]
    conf_mean0 = part_df0.groupby("block").confidence_fill0.mean()
    conf_mean1 = part_df1.groupby("block").confidence_fill0.mean()
    conf_sem0 = part_df0.groupby("block").confidence_fill0.sem()
    conf_sem1 = part_df1.groupby("block").confidence_fill0.sem()
    if any(conf_sem1.isna()):
        raise ValueError(f"Expi {Expi} thread 1 has nan sem")
    if any(conf_sem0.isna()):
        raise ValueError(f"Expi {Expi} thread 0 has nan sem")
    traj_df = pd.concat([conf_mean0, conf_mean1, conf_sem0, conf_sem1], axis=1)
    confscore_col.append(traj_df.to_numpy())
#%%
max_traj_len = max([len(traj) for traj in confscore_col])
extrapconfscore_col = []
extrap_mask_col = []
for i, traj_arr in enumerate(confscore_col):
    nblocks = traj_arr.shape[0]
    if nblocks < max_traj_len:
        # extrap_arr = np.zeros((max_traj_len - traj_arr.shape[0], 4))
        extrap_arr = np.repeat(traj_arr[-1:], max_traj_len - nblocks, axis=0)
        extrap_full_arr = np.concatenate([traj_arr, extrap_arr], axis=0)
        extrap_msk = np.concatenate([np.ones(nblocks), np.zeros(max_traj_len - nblocks)]).astype(bool)
        extrapconfscore_col.append(extrap_full_arr)
        extrap_mask_col.append(extrap_msk)
    else:
        extrapconfscore_col.append(traj_arr)
        extrap_mask_col.append(np.ones(max_traj_len).astype(bool))

extrapconfscore_tsr = np.stack(extrapconfscore_col, axis=0)
extrap_mask_tsr = np.stack(extrap_mask_col, axis=0)
#%%
# combine multiple mean and sem into one sem and mean, sem of the all means
def _combine_mean_sem(mean_arr, sem_arr):
    n = mean_arr.shape[0]
    mean = mean_arr.mean(axis=0)
    sem = np.sqrt((sem_arr**2).sum(axis=0)) / n
    return mean, sem


def _shaded_errorbar(x, y, yerr, label=None, color=None, **kwargs):
    plt.fill_between(x, y-yerr, y+yerr, alpha=0.3, label=None, color=color)
    plt.plot(x, y, color=color, label=label, **kwargs)

trajdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_proto_Yolo_traj"
#%%
commonmsk = validmsk & bothsucmsk#(~sucsmsk)
plt.figure(figsize=[5, 5])
for i, (label, msk) in enumerate(zip(["V1", "V4", "IT"],
        [V1msk, V4msk, ITmsk])):
    msk = msk & commonmsk
    popmean0 = extrapconfscore_tsr[msk].mean(axis=0)[:, 0]
    popsem0 = extrapconfscore_tsr[msk].std(axis=0)[:, 0] / np.sqrt(msk.sum())
    popmean1 = extrapconfscore_tsr[msk].mean(axis=0)[:, 1]
    popsem1 = extrapconfscore_tsr[msk].std(axis=0)[:, 1] / np.sqrt(msk.sum())
    _shaded_errorbar(np.arange(max_traj_len), popmean0, yerr=popsem0,
        label=label+"_DeePSim", linestyle="-", color=plt.cm.tab10(i))
    _shaded_errorbar(np.arange(max_traj_len), popmean1, yerr=popsem1,
        label=label+"_BigGAN", linestyle="-.", color=plt.cm.tab10(i))
    #
    # plt.errorbar(np.arange(max_traj_len), extrapconfscore_tsr[msk].mean(axis=0)[:, 0],
    #     yerr=extrapconfscore_tsr[msk].mean(axis=0)[:, 2], label=label+"DeePSim")
    # plt.errorbar(np.arange(max_traj_len), extrapconfscore_tsr[msk].mean(axis=0)[:, 1],
    #     yerr=extrapconfscore_tsr[msk].mean(axis=0)[:, 3], label=label+"BigGAN")
    # plt.plot(extrapconfscore_tsr[msk].mean(axis=0)[:, 0], label=label+"DeePSim")
    # plt.plot(extrapconfscore_tsr[msk].mean(axis=0)[:, 1], label=label+"BigGAN")
plt.legend()
plt.show()
#%%
#%%
commonmsk = validmsk & bothsucmsk#(~sucsmsk)
figh, axs = plt.subplots(1, 3, figsize=[9, 3.5], sharex=True, sharey=True)
for i, (label, msk) in enumerate(zip(["V1", "V4", "IT"],
                                    [V1msk, V4msk, ITmsk])):
    msk = msk & commonmsk
    popmean0 = extrapconfscore_tsr[msk].mean(axis=0)[:, 0]
    popsem0 = extrapconfscore_tsr[msk].std(axis=0)[:, 0] / np.sqrt(msk.sum())
    popmean1 = extrapconfscore_tsr[msk].mean(axis=0)[:, 1]
    popsem1 = extrapconfscore_tsr[msk].std(axis=0)[:, 1] / np.sqrt(msk.sum())
    plt.sca(axs[i])
    plt.plot(1+np.arange(max_traj_len), extrapconfscore_tsr[msk][:, :, 0].T,
             linestyle="-", color=plt.cm.tab10(i), alpha=0.2)
    plt.plot(1+np.arange(max_traj_len), extrapconfscore_tsr[msk][:, :, 1].T,
             linestyle="-.", color=plt.cm.tab10(i), alpha=0.2)
    _shaded_errorbar(1+np.arange(max_traj_len), popmean0, yerr=popsem0,
                        label=label+"_DeePSim", linestyle="-", color=plt.cm.tab10(i))
    _shaded_errorbar(1+np.arange(max_traj_len), popmean1, yerr=popsem1,
                        label=label+"_BigGAN", linestyle="-", color=plt.cm.tab10(i))
    plt.title(label)
    plt.ylim([0.2, 0.9])
    plt.xlim([0, 45])
    plt.xticks([1, 15, 30, 45])
    plt.yticks([0.2, 0.4, 0.6, 0.8])
    plt.xlabel("Block")
    plt.ylabel("Confidence")
    # if i == 0:
    plt.legend()
plt.tight_layout()
plt.show()
#%%
commonmsk = validmsk & bothsucmsk#(~sucsmsk)
figh, axs = plt.subplots(1, 2, figsize=[6, 3.5], sharex=True, sharey=True)
for i, (label, msk) in enumerate(zip(["V4", "IT"],
                                    [V4msk, ITmsk])):
    msk = msk & commonmsk
    popmean0 = extrapconfscore_tsr[msk].mean(axis=0)[:, 0]
    popsem0 = extrapconfscore_tsr[msk].std(axis=0)[:, 0] / np.sqrt(msk.sum())
    popmean1 = extrapconfscore_tsr[msk].mean(axis=0)[:, 1]
    popsem1 = extrapconfscore_tsr[msk].std(axis=0)[:, 1] / np.sqrt(msk.sum())
    plt.sca(axs[i])
    # plt.plot(1+np.arange(max_traj_len), extrapconfscore_tsr[msk][:, :, 0].T
    #          , linestyle="-", color="blue", alpha=0.2)
    # plt.plot(1+np.arange(max_traj_len), extrapconfscore_tsr[msk][:, :, 1].T
    #          , linestyle="-.", color="red", alpha=0.2)
    _shaded_errorbar(1+np.arange(max_traj_len), popmean0, yerr=popsem0,
                        label="DeePSim", linestyle="-", color="blue")
    _shaded_errorbar(1+np.arange(max_traj_len), popmean1, yerr=popsem1,
                        label="BigGAN", linestyle="-.", color="red")
    plt.title(label + " (n={})".format(msk.sum()))
    plt.ylim([0.2, 0.82])
    plt.xlim([0, 55])
    plt.xticks([1, 15, 30, 45])
    plt.yticks([0.2, 0.4, 0.6, 0.8])
    plt.xlabel("Block")
    if i == 0:
        plt.ylabel("Confidence")
# if i == 0:
plt.legend()
plt.tight_layout()
saveallforms(trajdir, "extrap_confid_bothsuc_area_sep", figh, ["png", "pdf"])
plt.show()
#%%
commonmsk = validmsk & bothsucmsk#(~sucsmsk)
figh, axs = plt.subplots(2, 2, figsize=[6, 6.5], sharex=True, sharey=True)
for i, (label1, msk1) in enumerate(zip(["V4", "IT"], [V4msk, ITmsk])):
    for j, (label2, msk2) in enumerate(zip(["A", "B"], [Amsk, Bmsk])):
        msk = msk1 & msk2 & commonmsk
        popmean0 = extrapconfscore_tsr[msk].mean(axis=0)[:, 0]
        popsem0 = extrapconfscore_tsr[msk].std(axis=0)[:, 0] / np.sqrt(msk.sum())
        popmean1 = extrapconfscore_tsr[msk].mean(axis=0)[:, 1]
        popsem1 = extrapconfscore_tsr[msk].std(axis=0)[:, 1] / np.sqrt(msk.sum())
        plt.sca(axs[j, i])
        plt.plot(1+np.arange(max_traj_len), extrapconfscore_tsr[msk][:, :, 0].T
                 , linestyle="-", color="blue", alpha=0.2)
        plt.plot(1+np.arange(max_traj_len), extrapconfscore_tsr[msk][:, :, 1].T
                 , linestyle="-", color="red", alpha=0.2)
        _shaded_errorbar(1+np.arange(max_traj_len), popmean0, yerr=popsem0,
                            label="DeePSim", linestyle="-", color="blue")
        _shaded_errorbar(1+np.arange(max_traj_len), popmean1, yerr=popsem1,
                            label="BigGAN", linestyle="-", color="red")
        plt.title(f"{label2} {label1} (n={msk.sum()})")
        plt.ylim([0.2, 0.9])
        plt.xlim([0, 55])
        plt.xticks([1, 15, 30, 45])
        plt.yticks([0.2, 0.4, 0.6, 0.8])
        if j == 1:
            plt.xlabel("Block")
        if i == 0:
            plt.ylabel("Confidence")
# if i == 0:
plt.legend()
plt.tight_layout()
saveallforms(trajdir, "extrap_confid_bothsuc_area_anim_sep_windiv", figh, ["png", "pdf"])
plt.show()



#%%
baseline_df = pd.read_csv(tabdir / "GAN_samples_all_yolo_stats.csv", index_col=0)
#%%
ttest_ind_print(baseline_df[baseline_df.imgdir_name=='DeePSim_4std'].confidence,
                all_df[(all_df.visual_area == 'V4')&(all_df.thread==0)&(all_df.block <20)].confidence)
#%%
ttest_ind_print(baseline_df[baseline_df.imgdir_name=='resnet50_linf8_gradevol_avgpool'].confidence.fillna(0),
                all_df[(all_df.visual_area == 'IT')&(all_df.thread==0)&(all_df.block >30)].confidence.fillna(0))
#%%
ttest_ind_print(all_df[(all_df.visual_area == 'IT')&(all_df.thread==0)&(all_df.block <10)].confidence.fillna(0),
                all_df[(all_df.visual_area == 'IT')&(all_df.thread==0)&(all_df.block >30)].confidence.fillna(0))
#%%
ttest_ind_print(all_df[(all_df.visual_area == 'IT')&(all_df.thread==1)&(all_df.block <10)].confidence.fillna(0),
                all_df[(all_df.visual_area == 'IT')&(all_df.thread==1)&(all_df.block >30)].confidence.fillna(0))
#%%
ttest_ind_print(all_df[(all_df.visual_area == 'V4')&(all_df.thread==1)&(all_df.block <10)].confidence.fillna(0),
                all_df[(all_df.visual_area == 'V4')&(all_df.thread==1)&(all_df.block >30)].confidence.fillna(0))



