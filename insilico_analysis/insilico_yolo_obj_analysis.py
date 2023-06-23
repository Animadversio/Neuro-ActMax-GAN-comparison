
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms
def _shortenname(layername):
    return layername.replace("Bottleneck","B").replace(".layer", "layer").replace(".Linear", "")

figdir = r'E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\insilico_yolo_objectness'
rootdir = r"F:/insilico_exps/GAN_Evol_cmp/yolo_summary"
finalsumdir = Path(rootdir) / "final_summary"
finalsumdir.mkdir(exist_ok=True)
#%%
# all_yolo_stats = pd.read_csv(finalsumdir / "efficientnet_yolo_stats_df_all.csv", index_col=0)
all_yolo_stats = pd.read_csv(finalsumdir / "resnet50_linf8_yolo_stats_df_all.csv", index_col=0)
#%%
all_yolo_stats["confidence_fill0"] = all_yolo_stats.confidence.fillna(0)
#%%
netname = "resnet50_linf8"
layernames = all_yolo_stats.layer.unique()
for layer in layernames:
    for RFresize in [True, False]:
        figh = plt.figure(figsize=[5, 5])
        sns.lineplot(x="imgnum", y="confidence_fill0", hue="optimmethod",
                     hue_order=['CholCMA', 'HessCMA', 'CholCMA_fc6', 'HessCMA500_fc6'],
                     data=all_yolo_stats[(all_yolo_stats.layer == layer) &
                                         (all_yolo_stats.RFresize == RFresize)],
                     errorbar="se", estimator="mean", n_boot=0)
        plt.xlabel("block", fontsize=18)
        plt.ylabel("confidence", fontsize=18)
        plt.title(f"confidence vs block net={netname}\nlayer={_shortenname(layer)}, RFresize={RFresize}", fontsize=18)
        saveallforms(figdir, f"confidence_vs_block_{netname}_{layer}{'_RFrsz' if RFresize else '_full'}", figh,)
        plt.tight_layout()
        plt.show()
        # raise Exception

#%%
layernames = all_yolo_stats.layer.unique()
figh, axs = plt.subplots(2, 5, figsize=[25, 10], sharex=True, sharey=True)
for ci, layer in enumerate(layernames):
    for rj, RFresize in enumerate([True, False]):
        sns.lineplot(x="imgnum", y="confidence_fill0", hue="optimmethod",
                     hue_order=['CholCMA', 'HessCMA', 'CholCMA_fc6', 'HessCMA500_fc6'],
                     data=all_yolo_stats[(all_yolo_stats.layer == layer) &
                                         (all_yolo_stats.RFresize == RFresize)],
                     errorbar="se", estimator="mean", n_boot=0, ax=axs[rj, ci],
                     legend=True if (rj == 0) and (ci == 0) else False)
        axs[rj, ci].set_xlabel("block", fontsize=18)
        axs[rj, ci].set_ylabel("confidence", fontsize=18)
        axs[rj, ci].set_title(f"{_shortenname(layer)}, RFresize={RFresize}", fontsize=18)
plt.suptitle(f"confidence vs block net={netname}", fontsize=24)
plt.tight_layout()
saveallforms(figdir, f"confidence_vs_block_{netname}_merge", figh, )
plt.show()

#%%
figh, axs = plt.subplots(2, 5, figsize=[25, 10], sharex=True, sharey=True)
for ci, layer in enumerate(layernames):
    for rj, RFresize in enumerate([True, False]):
        sns.lineplot(x="imgnum", y="n_objs", hue="optimmethod",
                     hue_order=['CholCMA', 'HessCMA', 'CholCMA_fc6', 'HessCMA500_fc6'],
                     data=all_yolo_stats[(all_yolo_stats.layer == layer) &
                                         (all_yolo_stats.RFresize == RFresize)],
                     errorbar="se", estimator="mean", n_boot=0, ax=axs[rj, ci],
                     legend=True if (rj == 0) and (ci == 0) else False)
        axs[rj, ci].set_xlabel("block", fontsize=18)
        axs[rj, ci].set_ylabel("detected object #", fontsize=18)
        axs[rj, ci].set_title(f"{_shortenname(layer)}, RFresize={RFresize}", fontsize=18)
plt.suptitle(f"object number vs block net={netname}", fontsize=24)
plt.tight_layout()
saveallforms(figdir, f"objnum_vs_block_{netname}_merge", figh, )
plt.show()

#%%
layernames = all_yolo_stats.layer.unique()
for netname in all_yolo_stats.netname.unique():
    figh, axs = plt.subplots(2, len(layernames), figsize=[5 * len(layernames), 10],
                             sharex=True, sharey=True)
    for ci, layer in enumerate(layernames):
        for rj, RFresize in enumerate([True, False]):
            sns.lineplot(x="imgnum", y="confidence_fill0", hue="optimmethod",
                         hue_order=['CholCMA', 'HessCMA', 'CholCMA_fc6', 'HessCMA500_fc6'],
                         data=all_yolo_stats[(all_yolo_stats.netname == netname) &
                                             (all_yolo_stats.layer == layer) &
                                             (all_yolo_stats.RFresize == RFresize)],
                         errorbar="se", estimator="mean", n_boot=0, ax=axs[rj, ci],
                         legend=True if (rj == 0) and (ci == 0) else False)
            axs[rj, ci].set_xlabel("block", fontsize=18)
            axs[rj, ci].set_ylabel("confidence", fontsize=18)
            axs[rj, ci].set_title(f"{_shortenname(layer)}, RFresize={RFresize}", fontsize=18)
    plt.suptitle(f"confidence vs block net={netname}", fontsize=24)
    plt.tight_layout()
    saveallforms(figdir, f"confidence_vs_block_{netname}_merge", figh, )
    plt.show()

    figh, axs = plt.subplots(2, len(layernames), figsize=[5 * len(layernames), 10],
                             sharex=True, sharey=True)
    for ci, layer in enumerate(layernames):
        for rj, RFresize in enumerate([True, False]):
            sns.lineplot(x="imgnum", y="n_objs", hue="optimmethod",
                         hue_order=['CholCMA', 'HessCMA', 'CholCMA_fc6', 'HessCMA500_fc6'],
                         data=all_yolo_stats[(all_yolo_stats.netname == netname) &
                                             (all_yolo_stats.layer == layer) &
                                             (all_yolo_stats.RFresize == RFresize)],
                         errorbar="se", estimator="mean", n_boot=0, ax=axs[rj, ci],
                         legend=True if (rj == 0) and (ci == 0) else False)
            axs[rj, ci].set_xlabel("block", fontsize=18)
            axs[rj, ci].set_ylabel("detected object #", fontsize=18)
            axs[rj, ci].set_title(f"{_shortenname(layer)}, RFresize={RFresize}", fontsize=18)
    plt.suptitle(f"object number vs block net={netname}", fontsize=24)
    plt.tight_layout()
    saveallforms(figdir, f"objnum_vs_block_{netname}_merge", figh, )
    plt.show()

    for layer in layernames:
        for RFresize in [True, False]:
            figh = plt.figure(figsize=[5, 5])
            sns.lineplot(x="imgnum", y="confidence_fill0", hue="optimmethod",
                         hue_order=['CholCMA', 'HessCMA', 'CholCMA_fc6', 'HessCMA500_fc6'],
                         data=all_yolo_stats[(all_yolo_stats.netname == netname) &
                                             (all_yolo_stats.layer == layer) &
                                             (all_yolo_stats.RFresize == RFresize)],
                         errorbar="se", estimator="mean", n_boot=0)
            plt.xlabel("block", fontsize=18)
            plt.ylabel("confidence", fontsize=18)
            plt.title(f"confidence vs block net={netname}\nlayer={_shortenname(layer)}, RFresize={RFresize}",
                      fontsize=18)
            saveallforms(figdir, f"confidence_vs_block_{netname}_{layer}{'_RFrsz' if RFresize else '_full'}", figh, )
            plt.tight_layout()
            plt.show()

