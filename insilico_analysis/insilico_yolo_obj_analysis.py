
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms


def _shortenname(layername):
    return layername.replace("Bottleneck","B").replace(".layer", "layer").replace(".Linear", "")


figdir = r'E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\insilico_yolo_objectness'
# rootdir = r"F:/insilico_exps/GAN_Evol_cmp/yolo_summary"
# finalsumdir = Path(rootdir) / "final_summary"
# finalsumdir.mkdir(exist_ok=True)
#%%
finalsumdir = Path(r"F:\insilico_exps\GAN_Evol_cmp\final_summary")
# all_yolo_stats = pd.read_csv(finalsumdir / "efficientnet_yolo_stats_df_all.csv", index_col=0)
# all_yolo_stats = pd.read_csv(finalsumdir / "resnet50_linf8_yolo_stats_df_all.csv", index_col=0)
all_yolo_stats = pd.read_csv(finalsumdir / "resnet50_linf8_yolo_objconf_stats_df_all.csv", index_col=0)
all_yolo_stats["confidence_fill0"] = all_yolo_stats.confidence.fillna(0)
all_yolo_stats["obj_confidence_fill0"] = all_yolo_stats.obj_confidence.fillna(0)
all_yolo_stats["cls_confidence_fill0"] = all_yolo_stats.cls_confidence.fillna(0)
#%%
tabdir = Path(r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables")
GANimgtab = pd.read_csv(tabdir / 'GAN_samples_all_yolo_objconf_stats.csv', index_col=0)
GANimgtab["confidence_fill0"] = GANimgtab.confidence.fillna(0)
GANimgtab["obj_confidence_fill0"] = GANimgtab.obj_confidence.fillna(0)
GANimgtab["cls_confidence_fill0"] = GANimgtab.cls_confidence.fillna(0)
#%%
netname = "resnet50_linf8"
layernames = all_yolo_stats.layer.unique()
for yvalue in ["confidence_fill0", "obj_confidence_fill0", "cls_confidence_fill0"]:
    # "confidence", "obj_confidence", "cls_confidence",
    figh, axs = plt.subplots(2, 5, figsize=[25, 10], sharex=True, sharey=True)
    for ci, layer in enumerate(layernames):
        for rj, RFresize in enumerate([True, False]):
            sns.lineplot(x="imgnum", y=yvalue, hue="optimmethod",
                         hue_order=['CholCMA', 'HessCMA', 'CholCMA_fc6', 'HessCMA500_fc6'],
                         data=all_yolo_stats[(all_yolo_stats.layer == layer) &
                                             (all_yolo_stats.RFresize == RFresize)],
                         errorbar="se", estimator="mean", n_boot=0, ax=axs[rj, ci],
                         legend=True if (rj == 0) and (ci == 0) else False)
            axs[rj, ci].set_xlabel("block", fontsize=18)
            axs[rj, ci].set_ylabel(f"{yvalue}", fontsize=18)
            axs[rj, ci].set_title(f"{_shortenname(layer)}, RFresize={RFresize}", fontsize=18)
    plt.suptitle(f"{yvalue} vs block net={netname}", fontsize=24)
    plt.tight_layout()
    saveallforms(figdir, f"{yvalue}_vs_block_{netname}_merge", figh, )
    plt.show()

    figh, axs = plt.subplots(1, 5, figsize=[25, 5], sharex=True, sharey=True, squeeze=False)
    rj, RFresize = 0, False
    for ci, layer in enumerate(layernames):
        print(layer)
        sns.lineplot(x="imgnum", y=yvalue, hue="optimmethod",
                     hue_order=['CholCMA', 'HessCMA', 'CholCMA_fc6', 'HessCMA500_fc6'],
                     data=all_yolo_stats[(all_yolo_stats.layer == layer) &
                                         (all_yolo_stats.RFresize == RFresize)],
                     errorbar="se", estimator="mean", n_boot=0, ax=axs[rj, ci],
                     legend=True if (rj == 0) and (ci == 0) else False, )
        print(layer, "complete")
        axs[rj, ci].set_xlabel("block", fontsize=18)
        axs[rj, ci].set_ylabel(f"{yvalue}", fontsize=18)
        axs[rj, ci].set_title(f"{_shortenname(layer)}, RFresize={RFresize}", fontsize=18)
    plt.suptitle(f"{yvalue} vs block net={netname}", fontsize=24)
    plt.tight_layout()
    saveallforms(figdir, f"{yvalue}_vs_block_{netname}_merge_noRF", figh, )
    plt.show()

#%%
""" Plot violin plots for each layer last five blocks """
for yvalue in ["confidence_fill0", "obj_confidence_fill0", "cls_confidence_fill0"]:
    # "confidence", "obj_confidence", "cls_confidence",
    pass
#%%
""" Violin plots comparing each layer last five blocks """
# "confidence_fill0", "obj_confidence_fill0", "cls_confidence_fill0",
netname = "resnet50_linf8"
layernames = all_yolo_stats.layer.unique()
#hue_order=['CholCMA', 'HessCMA', 'CholCMA_fc6', 'HessCMA500_fc6'],)
for optimname in ["CholCMA_fc6", "CholCMA", 'HessCMA', 'HessCMA500_fc6']:
    for yvalue in ["confidence", "obj_confidence", "cls_confidence"]:
        GANname = "DeePSim" if "fc6" in optimname else "BigGAN"
        themecolor = "blue" if GANname == "DeePSim" else "red"
        plt.figure(figsize=[4.5, 6])
        if GANname == "DeePSim":
            ref_dist = GANimgtab[GANimgtab.imgdir_name == 'DeePSim_4std'][yvalue]
        elif GANname == "BigGAN":
            ref_dist = GANimgtab[GANimgtab.imgdir_name == 'BigGAN_trunc07'][yvalue]
        else:
            raise ValueError("GANname not recognized")
        plt.axhline(y=ref_dist.median(), color=themecolor, linestyle='-', label=GANname + " dist")
        plt.axhline(y=ref_dist.quantile(0.25), color=themecolor, linestyle=':', label=GANname + " 25%")
        plt.axhline(y=ref_dist.quantile(0.75), color=themecolor, linestyle=':', label=GANname + " 75%")

        ax = sns.violinplot(data=all_yolo_stats[(all_yolo_stats.imgnum >= 95) &
                                    (all_yolo_stats.RFresize == False) &
                                    (all_yolo_stats.optimmethod == optimname)],
                       x="layer", y=yvalue, cut=0, order=layernames,
                       color=themecolor)
        for violin, in zip(ax.collections[::2]):
            violin.set_alpha(0.5)
        plt.xlabel("layer", fontsize=16)
        plt.ylabel(f"{yvalue}", fontsize=16)
        plt.title(f"Last 5 block {yvalue} distribution\nnet={netname} optim={optimname}", fontsize=14)
        plt.xticks(labels=[_shortenname(layer) for layer in layernames], ticks=range(len(layernames)))
        saveallforms(figdir, f"{yvalue}_vs_layer_last5block_{netname}_{optimname}_noRF_violin", plt.gcf())
        plt.show()


#%%
"""Plot each layer separately"""
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
""" Plot each layer as a subplot, two rows for RFresize=True/False """
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

