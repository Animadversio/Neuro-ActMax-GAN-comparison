import torch
import re
from pathlib import Path
import pickle as pkl
import pandas as pd
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms
from neuro_data_analysis.neural_data_utils import get_all_masks
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data, get_expstr
# from ultralytics import YOLO
# model_new = YOLO("yolov8x.pt")
#%%
_, BFEStats = load_neural_data()
saveroot = Path(r"E:\Network_Data_Sync\BigGAN_Evol_yolo")
sumdir = (saveroot / "yolo_v5_summary")
sumdir.mkdir(exist_ok=True)

figdir = saveroot / "figsummary"
figdir.mkdir(exist_ok=True)
#%%
def yolo_process(imgpathlist, batch_size=100, size=256, savename=None, sumdir=sumdir):
    """Process images with yolo model and return results in a list of dataframes"""
    results_dfs = []
    for i in trange(0, len(imgpathlist), batch_size):
        results = yolomodel(imgpathlist[i:i+batch_size], size=size)
        results_dfs.extend(results.pandas().xyxy)
        # yolo_results[i] = results

    yolo_stats = {}
    for i, single_df in tqdm(enumerate(results_dfs)):
        yolo_stats[i] = {"confidence": single_df.confidence.max(),
                        "class": single_df["class"][single_df.confidence.argmax()] if len(single_df) > 0 else None,
                        "n_objs": len(single_df),
                        "img_path": imgpathlist[i]}
    yolo_stats_df = pd.DataFrame(yolo_stats).T
    if savename is not None:
        yolo_stats_df.to_csv(sumdir / f"{savename}_yolo_stats.csv")
        pkl.dump(results_dfs, open(sumdir / f"{savename}_dfs.pkl", "wb"))
        print(f"Saved to {sumdir / f'{savename}_dfs.pkl'}")
        print(f"Saved to {sumdir / f'{savename}_yolo_stats.csv'}")
    print("Fraction of images with objects", (yolo_stats_df.n_objs > 0).mean())
    print("confidence", yolo_stats_df.confidence.mean(), "confidence with 0 filled",
          yolo_stats_df.confidence.fillna(0).mean())
    print("most common class", yolo_stats_df["class"].value_counts().index[0])
    print("n_objs", yolo_stats_df.n_objs.mean(), )
    return results_dfs, yolo_stats_df


# Model
yolomodel = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
# plt.switch_backend('module://backend_interagg')
#%%
for Expi in trange(1, 190+1):
    if BFEStats[Expi-1]["evol"] is None:
        continue
    expdir = saveroot / f"Both_Exp{Expi}"
    expdir.mkdir(exist_ok=True)
    imgfps_col0, resp_vec0, bsl_vec0, gen_vec0 = \
        load_img_resp_pairs(BFEStats, Expi, "Evol", thread=0, output_fmt="vec")
    imgfps_col1, resp_vec1, bsl_vec1, gen_vec1 = \
        load_img_resp_pairs(BFEStats, Expi, "Evol", thread=1, output_fmt="vec")
    results_dfs0, yolo_stats_df0 = yolo_process(imgfps_col0, batch_size=100, size=256,
                                    savename=f"Exp{Expi:03d}_thread0", sumdir=sumdir)
    results_dfs1, yolo_stats_df1 = yolo_process(imgfps_col1, batch_size=100, size=256,
                                    savename=f"Exp{Expi:03d}_thread1", sumdir=sumdir)


#%% Load all the yolo results and combine them into a single dataframe
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
all_df["confidence_fill0"] = all_df.confidence.fillna(0)
#%%
#%%
tabdir = Path(r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables")
all_df.to_csv(sumdir / f"Evol_invivo_all_yolo_stats.csv")
all_df.to_csv(tabdir / f"Evol_invivo_all_yolo_stats.csv")
#%%
meta_df = pd.read_csv(tabdir / "meta_activation_stats.csv", index_col=0)
#%%
all_df["included"] = True
for Expi in meta_df.index:
    part_df0 = all_df[(all_df.Expi == Expi) & (all_df.thread == 0)]
    part_df1 = all_df[(all_df.Expi == Expi) & (all_df.thread == 1)]
    maxblock = max(part_df0.block.max(), part_df1.block.max())
    if (part_df0.block == maxblock).sum() < 10 or (part_df1.block == maxblock).sum() < 10:
        print(f"Expi {Expi} last block {maxblock} has less than 10 images")
        last_block_msk = (all_df.Expi == Expi) & (all_df.block == maxblock)
        all_df.loc[last_block_msk, "included"] = False
        print(f"excluded {last_block_msk.sum()} images from analysis")
        continue
    # raise NotImplementedError
    # all_df.loc[all_df.Expi == Expi, "invalid_last_block"]
    # all_df.loc[all_df.Expi == Expi, "lastblock"] = meta_df.loc[Expi, "last_block"]
    # all_df.loc[all_df.Expi==Expi, "visual_area"] = meta_df.loc[Expi, "visual_area"]
    #
#%%
# use the visual area of the corresponding Expi in metadf in all_df
all_df["visual_area"] = all_df.Expi.apply(lambda x: meta_df.loc[x, "visual_area"])
#%%
Amsk, Bmsk, V1msk, V4msk, ITmsk, length_msk, spc_msk, \
    sucsmsk, bsl_unstable_msk, bsl_stable_msk, validmsk = get_all_masks(meta_df)
bothsucmsk = (meta_df.p_maxinit_0 < 0.05) & (meta_df.p_maxinit_1 < 0.05)
#%%
plt.figure(figsize=[12, 8])
plt.scatter(all_df.block, all_df.confidence, s=1, alpha=0.5)
plt.xlabel("block")
plt.ylabel("confidence")
plt.title("confidence vs block")
# plt.savefig(sumdir / "confidence_vs_block.png")
plt.show()
#%%
mean_conf0 = all_df[all_df.thread == 0].groupby(["block"]).confidence_fill0.mean()
mean_conf1 = all_df[all_df.thread == 1].groupby(["block"]).confidence_fill0.mean()
sem_conf0 = all_df[all_df.thread == 0].groupby(["block"]).confidence_fill0.sem()
sem_conf1 = all_df[all_df.thread == 1].groupby(["block"]).confidence_fill0.sem()
#%%
plt.figure(figsize=[12, 8])
plt.errorbar(mean_conf0.index, mean_conf0, yerr=sem_conf0, label="thread0")
plt.errorbar(mean_conf1.index, mean_conf1, yerr=sem_conf1, label="thread1")
plt.xlabel("block")
plt.ylabel("confidence")
plt.title("confidence vs block")
plt.legend()
# plt.savefig(sumdir / "confidence_vs_block.png")
plt.show()
#%%
all_df[all_df.thread==0].groupby(["block"]).confidence_fill0.mean().plot()
all_df[all_df.thread==1].groupby(["block"]).confidence_fill0.mean().plot()
plt.xlabel("block")
plt.ylabel("confidence")
plt.title("confidence vs block")
# plt.savefig(sumdir / "confidence_vs_block.png")
plt.show()
#%%
import seaborn as sns
# mask some Expi
sel_Expi = meta_df[validmsk & ITmsk & bothsucmsk].index
#%%
plt.figure(figsize=[5, 5])
sns.lineplot(data=all_df[all_df.Expi.isin(sel_Expi)],
             x="block", y="confidence_fill0", hue="thread",
             errorbar="se", estimator="mean", n_boot=0)
plt.xlabel("block")
plt.ylabel("confidence")
plt.title("confidence vs block")
# plt.savefig(sumdir / "confidence_vs_block.png")
plt.show()
#%%
plt.figure(figsize=[5, 5])
for areamsk in [V1msk, V4msk, ITmsk]:
    sel_Expi = meta_df[validmsk & areamsk].index
    sns.lineplot(data=all_df[all_df.Expi.isin(sel_Expi)],
                 x="block", y="confidence_fill0", hue="thread",
                 errorbar="se", estimator="mean", n_boot=0)
plt.xlabel("block")
plt.ylabel("confidence")
plt.title("confidence vs block")
# plt.savefig(sumdir / "confidence_vs_block.png")
plt.show()
#%%
plt.figure(figsize=[5, 5])
sns.lineplot(x="block", y="confidence_fill0", data=all_df[all_df.Expi == 69], hue="thread",
             errorbar="se", estimator="mean", n_boot=0)
plt.xlabel("block")
plt.ylabel("confidence")
plt.title("confidence vs block")
# plt.savefig(sumdir / "confidence_vs_block.png")
plt.show()
#%%
msk_sfx = "validbothsuc"
sel_Expi = meta_df[validmsk & bothsucmsk].index
plt.figure(figsize=[5, 5])
sns.lineplot(data=all_df[all_df["included"] & all_df.Expi.isin(sel_Expi)],
             x="block", y="confidence_fill0", hue="visual_area", style="thread",
             errorbar="se", estimator="mean", n_boot=0)
plt.xlabel("block")
plt.ylabel("confidence")
plt.title("confidence vs block [Valid & Both Success]")
saveallforms(str(figdir), f"confid_vs_block_traj_{msk_sfx}")
plt.show()
#%%
msk_sfx = "allvalid"
sel_Expi = meta_df[validmsk].index
plt.figure(figsize=[5, 5])
sns.lineplot(data=all_df[all_df["included"] & all_df.Expi.isin(sel_Expi)],
             x="block", y="confidence_fill0", hue="visual_area", style="thread",
             errorbar="se", estimator="mean", n_boot=0)
plt.xlabel("block")
plt.ylabel("confidence")
plt.title("confidence vs block [Valid]")
saveallforms(str(figdir), f"confid_vs_block_traj_{msk_sfx}")
plt.show()
#%%
def plot_confidence_curve(all_df, meta_df, title_mask_str,
                          all_msk=None, sel_Expi=None, msk_sfx="",
                          hue_var="visual_area", style_var="thread"):
    if all_msk is None:
        all_msk = all_df["included"]
    if sel_Expi is None:
        sel_Expi = meta_df.index
    fig = plt.figure(figsize=[5, 5])
    sns.lineplot(data=all_df[all_msk & all_df.Expi.isin(sel_Expi)],
                 x="block", y="confidence_fill0", hue=hue_var, style=style_var,
                 errorbar="se", estimator="mean", n_boot=0)
    plt.xlabel("block")
    plt.ylabel("confidence")
    plt.title(f"confidence vs block [{title_mask_str}]")
    saveallforms(str(figdir), f"confid_vs_block_traj_{msk_sfx}")
    plt.show()
    return fig
#%%


plot_confidence_curve(all_df, meta_df, "Valid", msk_sfx="allvalid",
                sel_Expi=meta_df[validmsk].index, )
plot_confidence_curve(all_df, meta_df, "Valid & Both Success", msk_sfx="validbothsuc",
              sel_Expi=meta_df[bothsucmsk & validmsk].index)
plot_confidence_curve(all_df, meta_df, "Valid & Any Success", msk_sfx="validanysuc",
              sel_Expi=meta_df[sucsmsk & validmsk].index)
plot_confidence_curve(all_df, meta_df, "Valid & None Success", msk_sfx="validnonesuc",
              sel_Expi=meta_df[~sucsmsk & validmsk].index)
#%%
for Expi in meta_df.index:
    expstr = get_expstr(BFEStats, Expi)
    plot_confidence_curve(all_df, meta_df, f"Exp {Expi}\n{expstr}", msk_sfx=f"Exp{Expi:03d}",
                          sel_Expi=[Expi], hue_var="thread", )


