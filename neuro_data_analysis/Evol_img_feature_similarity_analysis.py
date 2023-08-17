"""
Compute similarity between images via pre-extracted features from all Evolution images
in E:\Network_Data_Sync\BigGAN_Evol_feat_extract

"""
import re
from pathlib import Path
import pickle as pkl
import pandas as pd
from tqdm import trange, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from timm.models import create_model, list_models
from core.utils.CNN_scorers import load_featnet
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from core.utils.dataset_utils import ImagePathDataset, ImagePathDataset_pure
from torchvision import transforms
from core.utils.plot_utils import saveallforms
from neuro_data_analysis.neural_data_utils import get_all_masks
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data, get_expstr
# from ultralytics import YOLO
# model_new = YOLO("yolov8x.pt")
tabdir = Path(r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables")
meta_df = pd.read_csv(tabdir / "meta_activation_stats_w_optimizer.csv", index_col=0)
#%%
_, BFEStats = load_neural_data()
#%%
Amsk, Bmsk, V1msk, V4msk, ITmsk, \
    length_msk, spc_msk, sucsmsk, bsl_unstable_msk, \
    bsl_stable_msk, validmsk = get_all_masks(meta_df)
#%%
saveroot = Path(r"E:\Network_Data_Sync\BigGAN_Evol_feat_extract")

figdir = saveroot / "figsummary"
figdir.mkdir(exist_ok=True, parents=True)
#%%
def cosine_simmat(feat_arr):
    """Compute the cosine similarity matrix of a feature array
    """
    feat_arr = feat_arr / feat_arr.norm(dim=1, keepdim=True)
    simmat = feat_arr @ feat_arr.T
    # get mean and std of the upper triangle
    index = torch.triu_indices(simmat.shape[0], simmat.shape[1], offset=1)
    simmat_upper = simmat[index[0], index[1]]
    sim_mean = simmat_upper.mean()
    sim_std = simmat_upper.std()
    sim_sem = sim_std / simmat_upper.shape[0] ** 0.5
    return simmat.cpu(), sim_mean.item(), sim_std.item(), sim_sem.item()


def cosine_simmat_pair(feat_arr1, feat_arr2):
    feat_arr1 = feat_arr1 / feat_arr1.norm(dim=1, keepdim=True)
    feat_arr2 = feat_arr2 / feat_arr2.norm(dim=1, keepdim=True)
    simmat = feat_arr1 @ feat_arr2.T
    sim_mean = simmat.mean()
    sim_std = simmat.std()
    sim_sem = sim_std / simmat.flatten().shape[0] ** 0.5
    return simmat.cpu(), sim_mean.item(), sim_std.item(), sim_sem.item()


#%%
# netname = "resnet50_linf8"
# sumdir = (saveroot / "resnet50_linf8")
netname = "vit_base_patch8_224_dino"
sumdir = (saveroot / "vit_base_patch8_224_dino")
for Expi in trange(1, 190+1):
    if BFEStats[Expi-1]["evol"] is None:
        continue
    imgfps_col0, resp_vec0, bsl_vec0, gen_vec0 = \
        load_img_resp_pairs(BFEStats, Expi, "Evol", thread=0, output_fmt="vec")
    imgfps_col1, resp_vec1, bsl_vec1, gen_vec1 = \
        load_img_resp_pairs(BFEStats, Expi, "Evol", thread=1, output_fmt="vec")
    feat_tsr1 = torch.load(sumdir / f"Evol_Exp{Expi:03d}_thread0_features.pt")
    meta_data_df0 = pd.read_csv(sumdir / f"Evol_Exp{Expi:03d}_thread0_meta.csv", index_col=0)
    feat_tsr2 = torch.load(sumdir / f"Evol_Exp{Expi:03d}_thread1_features.pt")
    meta_data_df1 = pd.read_csv(sumdir / f"Evol_Exp{Expi:03d}_thread1_meta.csv", index_col=0)
    feat_tsr1.to("cuda")
    feat_tsr2.to("cuda")
    # assert gen_vec0.max() == gen_vec1.max()
    if (gen_vec0 == gen_vec0.max()).sum() < 10:
        maxblock1 = gen_vec0.max() - 1
    else:
        maxblock1 = gen_vec0.max()
    if (gen_vec0 == gen_vec1.max()).sum() < 10:
        maxblock2 = gen_vec1.max() - 1
    else:
        maxblock2 = gen_vec1.max()
    maxblock = min(maxblock1, maxblock2)
    stat_col = {}
    raw_simmat_col = {}
    for blocki in range(1, maxblock + 1):
        nimgs1 = (gen_vec0 == blocki).sum()
        nimgs2 = (gen_vec1 == blocki).sum()
        resp_m1 = resp_vec0[gen_vec0 == blocki].mean()
        resp_m2 = resp_vec1[gen_vec1 == blocki].mean()
        resp_std1 = resp_vec0[gen_vec0 == blocki].std()
        resp_std2 = resp_vec1[gen_vec1 == blocki].std()
        resp_sem1 = resp_vec0[gen_vec0 == blocki].std() / nimgs1 ** 0.5
        resp_sem2 = resp_vec0[gen_vec0 == blocki].std() / nimgs1 ** 0.5
        simmat1, sim_mean1, sim_std1, sim_sem1 = cosine_simmat(feat_tsr1[gen_vec0 == blocki])
        simmat2, sim_mean2, sim_std2, sim_sem2 = cosine_simmat(feat_tsr2[gen_vec1 == blocki])
        simmat12, sim_mean12, sim_std12, sim_sem12 = cosine_simmat_pair(
            feat_tsr1[gen_vec0 == blocki], feat_tsr2[gen_vec1 == blocki])
        sim_mean12_avgvec = torch.cosine_similarity(feat_tsr1[gen_vec0 == blocki].mean(dim=0, keepdim=True),
                                                    feat_tsr2[gen_vec1 == blocki].mean(dim=0, keepdim=True)).item()
        stat_col[blocki] = {"block": blocki, "nimgs1": nimgs1, "nimgs2": nimgs2,
                            "resp_m1": resp_m1, "resp_std1": resp_std1, "resp_sem1": resp_sem1,
                            "resp_m2": resp_m2, "resp_std2": resp_std2, "resp_sem2": resp_sem2,
                            "sim_mean1": sim_mean1, "sim_std1": sim_std1, "sim_sem1": sim_sem1,
                            "sim_mean2": sim_mean2, "sim_std2": sim_std2, "sim_sem2": sim_sem2,
                            "sim_mean12": sim_mean12, "sim_std12": sim_std12, "sim_sem12": sim_sem12,
                            "sim_mean12_avgvec": sim_mean12_avgvec
                            }
        raw_simmat_col[blocki] = {"simmat1": simmat1, "simmat2": simmat2, "simmat12": simmat12}
    stat_df = pd.DataFrame.from_dict(stat_col, orient="index")
    stat_df.to_csv(sumdir / f"Evol_Exp{Expi:03d}_block_stats.csv")
    torch.save(raw_simmat_col, sumdir / f"Evol_Exp{Expi:03d}_block_simmat.pt")
    # raise NotImplementedError
#%%
stat_all_col = []
for Expi in trange(1, 190+1):
    if BFEStats[Expi-1]["evol"] is None:
        continue
    stat_df = pd.read_csv(sumdir / f"Evol_Exp{Expi:03d}_block_stats.csv", index_col=0)
    stat_df["Expi"] = Expi
    stat_df["block_rev"] = stat_df["block"].max() - stat_df["block"]
    stat_all_col.append(stat_df)

stat_all_df = pd.concat(stat_all_col, ignore_index=True)
stat_all_df.to_csv(figdir / f"Evol_block_stats_all_{netname}.csv")
stat_all_df=stat_all_df.merge(meta_df, on="Expi")
stat_all_df.to_csv(figdir / f"Evol_block_stats_all_{netname}.csv")
#%%
bothsucsmsk = (meta_df.p_maxinit_0 < 0.01) & (meta_df.p_maxinit_1 < 0.01)
msk = stat_all_df["Expi"].isin(meta_df[(validmsk & bothsucsmsk & ITmsk)].index)
plt.figure(figsize=[5,5])
sns.lineplot(data=stat_all_df[msk], x="block", y="sim_mean1", n_boot=0, errorbar="se")# hue="Expi"
sns.lineplot(data=stat_all_df[msk], x="block", y="sim_mean2", n_boot=0, errorbar="se")# hue="Expi"
sns.lineplot(data=stat_all_df[msk], x="block", y="sim_mean12", n_boot=0, errorbar="se")# hue="Expi"
sns.lineplot(data=stat_all_df[msk], x="block", y="sim_mean12_avgvec", n_boot=0, errorbar="se")# hue="Expi"
plt.legend(["sim_mean1", "sim_mean2", "sim_mean12", "sim_mean12_avgvec"])
plt.show()
#%%
bothsucsmsk = (meta_df.p_maxinit_0 < 0.01) & (meta_df.p_maxinit_1 < 0.01)
msk = stat_all_df["Expi"].isin(meta_df[(validmsk & bothsucsmsk & ITmsk)].index)
plt.figure(figsize=[5,5])
sns.lineplot(data=stat_all_df[msk], x="block_rev", y="sim_mean1", n_boot=0, errorbar="se")# hue="Expi"
sns.lineplot(data=stat_all_df[msk], x="block_rev", y="sim_mean2", n_boot=0, errorbar="se")# hue="Expi"
sns.lineplot(data=stat_all_df[msk], x="block_rev", y="sim_mean12", n_boot=0, errorbar="se")# hue="Expi"
sns.lineplot(data=stat_all_df[msk], x="block_rev", y="sim_mean12_avgvec", n_boot=0, errorbar="se")# hue="Expi"
plt.legend(["sim_mean1", "sim_mean2", "sim_mean12", "sim_mean12_avgvec"])
plt.show()

#%%
from core.utils.stats_utils import ttest_ind_print_df, ttest_rel_print_df, ttest_rel_print
bothsucsmsk = (meta_df.p_maxinit_0 < 0.01) & (meta_df.p_maxinit_1 < 0.01)
nonesucsmsk = (meta_df.p_maxinit_0 > 0.05) & (meta_df.p_maxinit_1 > 0.01)

# msk = stat_all_df["Expi"].isin(meta_df[(validmsk & bothsucsmsk & ITmsk)].index)
msk = stat_all_df["Expi"].isin(meta_df[(validmsk & nonesucsmsk & ITmsk)].index)
# stat_all_df[msk & (stat_all_df.block == 1)]
# stat_all_df[msk & (stat_all_df.block_rev == 0)]
ttest_rel_print(stat_all_df[msk & (stat_all_df.block == 1)].sim_mean1,
                stat_all_df[msk & (stat_all_df.block_rev == 0)].sim_mean1, sem=True)
ttest_rel_print(stat_all_df[msk & (stat_all_df.block == 1)].sim_mean2,
                stat_all_df[msk & (stat_all_df.block_rev == 0)].sim_mean2, sem=True)
ttest_rel_print(stat_all_df[msk & (stat_all_df.block == 1)].sim_mean12,
                stat_all_df[msk & (stat_all_df.block_rev == 0)].sim_mean12, sem=True)
# sim_mean12_avgvec
ttest_rel_print(stat_all_df[msk & (stat_all_df.block == 1)].sim_mean12_avgvec,
                stat_all_df[msk & (stat_all_df.block_rev == 0)].sim_mean12_avgvec, sem=True)
# ttest_ind_print_df(stat_all_df, msk & (stat_all_df.block == 2),
#                    msk & (stat_all_df.block_rev == 0), "sim_mean1")
# ttest_ind_print_df(stat_all_df, msk & (stat_all_df.block == 2),
#                    msk & (stat_all_df.block_rev == 0), "sim_mean2")
# ttest_ind_print_df(stat_all_df, msk & (stat_all_df.block == 2),
#                    msk & (stat_all_df.block_rev == 0), "sim_mean12")
