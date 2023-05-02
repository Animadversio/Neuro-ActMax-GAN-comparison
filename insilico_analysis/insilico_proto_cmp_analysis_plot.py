
import os
from os.path import join
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms

protosumdir = r"F:\insilico_exps\GAN_Evol_cmp\protoimgs"
figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoImage_cmp_insilico"
datadir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoImage_cmp_insilico\data"
optimname2cmp = ['CholCMA', 'HessCMA', 'HessCMA500_fc6']  #
#%%
"""Read in the dataframe data for each network layer and then synthesize them into a plot """
network_prefix = "resnet50_linf8_" # "resnet50_" #
suffix = "" # "_RFrsz"
stat_all_df = pd.DataFrame()
for layerstr in [
                 # "resnet50_layer1B1",
                 # "resnet50_layer2B3",
                 # "resnet50_layer3B5",
                 # "resnet50_layer4B2",
                 # "resnet50_fc",
                 "resnet50_linf8_layer1B1",
                 "resnet50_linf8_layer2B3",
                 "resnet50_linf8_layer3B5",
                 "resnet50_linf8_layer4B2",
                 "resnet50_linf8_fc",
                 ]:
    stat_df = pd.read_csv(join(datadir, f"{layerstr}_imgdist_cmp_stats{suffix}.csv"))
    stat_df["layer"] = layerstr
    stat_df["layershort"] = layerstr[len(network_prefix):]
    stat_all_df = pd.concat([stat_all_df, stat_df], axis=0)

stat_all_df.to_csv(join(datadir, f"{network_prefix}alllayer_imgdist_cmp_stats{suffix}.csv"))

#%%
from core.utils.stats_utils import ttest_ind_print_df, ttest_rel_print_df, paired_strip_plot
"""Compare the distance metrics between the different layers
Loop through all the metrics and plot the results
"""
metric_sfx = "_L4"
for metric_sfx in ["_L4",
                   "_L4_RFftmsk",
                   "_L4_RFpxmsk",
                   "_L4_RFpxftmsk",
                   "_L3",
                   "_L3_RFftmsk",
                   "_L3_RFpxmsk",
                   "_L3_RFpxftmsk",
                   ]:
    figh = plt.figure(figsize=[6,6])
    sns.pointplot(data=stat_all_df, x="layershort",
                  y="distmats02"+metric_sfx, color="k", alpha=0.6, capsize=0.2)
    sns.pointplot(data=stat_all_df, x="layershort",
                  y="distmats00"+metric_sfx, color="magenta", alpha=0.6, capsize=0.2, )
    sns.pointplot(data=stat_all_df, x="layershort",
                  y="distmats11"+metric_sfx, color="green", alpha=0.6, capsize=0.2, )
    sns.pointplot(data=stat_all_df, x="layershort",
                  y="distmats22"+metric_sfx, color="cyan", alpha=0.6, capsize=0.2, )
    sns.pointplot(data=stat_all_df, x="layershort",
                  y="distmats02_FCalt"+metric_sfx, color="r", alpha=0.6, capsize=0.2)
    sns.pointplot(data=stat_all_df, x="layershort",
                  y="distmats02_BGalt"+metric_sfx, color="b", alpha=0.6, capsize=0.2)
    plt.legend(handles=plt.gca().lines[::16], labels=["FC-BGChol",
                                           "BGChol", "BGHess", "FC",
                                           "FC'-BGChol", "FC-BGChol'"])
    plt.ylabel(f"Cosine Similarity - (resenet_linf8 {metric_sfx.replace('_', ' ')})")
    plt.title(f"Image Similarity among prototypes\n{metric_sfx.replace('_', ' ')} cosine\n{network_prefix[:-1]}", fontsize=14)
    saveallforms(figdir, f"{network_prefix}alllayers_imgdist_FCBG_CholCMABG{metric_sfx}{suffix}", figh, )
    plt.show()


#%%
"""
Compare the distance between BG-FC vs FC'-BG and BG'-FC, paired t-test
"""
network_prefix = "resnet50_" #"resnet50_linf8_" #
suffix = "" # "_RFrsz"
stat_all_df = pd.read_csv(join(datadir, f"{network_prefix}alllayer_imgdist_cmp_stats{suffix}.csv"))

#%%
for layerstr in [
               #  'resnet50_linf8_layer1B1', 'resnet50_linf8_layer2B3',
               # 'resnet50_linf8_layer3B5', 'resnet50_linf8_layer4B2',
               # 'resnet50_linf8_fc'
               'resnet50_layer1B1', 'resnet50_layer2B3',
                 'resnet50_layer3B5', 'resnet50_layer4B2',
                 'resnet50_fc'
                 ]:
    stat_df = pd.read_csv(join(datadir, f"{layerstr}_imgdist_cmp_stats{suffix}.csv"))
    for metric_sfx in ["_L4", "_L3",
                       "_L4_RFftmsk", "_L3_RFftmsk", ]:
        figh = paired_strip_plot(stat_df, None, f"distmats02{metric_sfx}", f"distmats02_FCalt{metric_sfx}")
        figh.suptitle(f"{layerstr}\nimg dist FC-BG vs FC'-BG, CholCMA BG")
        figh.gca().set_ylabel(f"cosine dist ({metric_sfx})")
        saveallforms(figdir, f"{layerstr}_imgdist_FCBG_vs_FC'BG_CholCMABG{metric_sfx}{suffix}", figh, )
        figh.show()
        figh = paired_strip_plot(stat_df, None, f"distmats02{metric_sfx}", f"distmats02_BGalt{metric_sfx}")
        figh.suptitle(f"{layerstr}\nimg dist FC-BG vs FC-BG', CholCMA BG")
        figh.gca().set_ylabel(f"cosine dist ({metric_sfx})")
        saveallforms(figdir, f"{layerstr}_imgdist_FCBG_vs_FCBG'_CholCMABG{metric_sfx}{suffix}", figh, )
        figh.show()
        figh = paired_strip_plot(stat_df, None, f"distmats12{metric_sfx}", f"distmats12_FCalt{metric_sfx}")
        figh.suptitle(f"{layerstr}\nimg dist FC-BG vs FC'-BG, HessCMA BG")
        figh.gca().set_ylabel(f"cosine dist ({metric_sfx})")
        saveallforms(figdir, f"{layerstr}_imgdist_FCBG_vs_FC'BG_HessCMABG{metric_sfx}{suffix}", figh, )
        figh.show()
        figh = paired_strip_plot(stat_df, None, f"distmats12{metric_sfx}", f"distmats12_BGalt{metric_sfx}")
        figh.suptitle(f"{layerstr}\nimg dist FC-BG vs FC-BG', Hess BG")
        figh.gca().set_ylabel(f"cosine dist ({metric_sfx.replace('_', ' ')})")
        saveallforms(figdir, f"{layerstr}_imgdist_FCBG_vs_FCBG'_HessCMABG{metric_sfx}{suffix}", figh, )
        figh.show()



#%% Scratch space
figh = plt.figure(figsize=[6,6])
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats02_L3", color="k", alpha=0.6, capsize=0.2)
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats00_L3", color="magenta", alpha=0.6, capsize=0.2, )
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats11_L3", color="green", alpha=0.6, capsize=0.2, )
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats22_L3", color="cyan", alpha=0.6, capsize=0.2, )
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats02_FCalt_L3", color="r", alpha=0.6, capsize=0.2)
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats02_BGalt_L3", color="b", alpha=0.6, capsize=0.2)
plt.legend(handles=plt.gca().lines[::16], labels=["FC-BGChol",
                                       "BGChol", "BGHess", "FC",
                                       "FC'-BGChol", "FC-BGChol'"])
plt.ylabel("Cosine Similarity - (resenet_linf8 L3)")
plt.title(f"Image Similarity among prototypes\nLayer 3 cosine\n{network_prefix[:-1]}", fontsize=14)
saveallforms(figdir, f"{network_prefix}alllayers_imgdist_FCBG_CholCMABG_L3{suffix}", figh, )
plt.show()
#%%
figh = plt.figure(figsize=[6,6])
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats02_L4", color="k", alpha=0.6, capsize=0.2)
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats00_L4", color="magenta", alpha=0.6, capsize=0.2, )
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats11_L4", color="green", alpha=0.6, capsize=0.2, )
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats22_L4", color="cyan", alpha=0.6, capsize=0.2, )
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats02_FCalt_L4", color="r", alpha=0.6, capsize=0.2)
sns.pointplot(data=stat_all_df, x="layershort",
              y="distmats02_BGalt_L4", color="b", alpha=0.6, capsize=0.2)
plt.legend(handles=plt.gca().lines[::16], labels=["FC-BGChol",
                                       "BGChol", "BGHess", "FC",
                                       "FC'-BGChol", "FC-BGChol'"])
plt.ylabel("Cosine Similarity - (resenet_linf8 L4)")
plt.title(f"Image Similarity among prototypes\nLayer 4 cosine\n{network_prefix[:-1]}", fontsize=14)
saveallforms(figdir, f"{network_prefix}alllayers_imgdist_FCBG_CholCMABG_L4{suffix}", figh, )
plt.show()
#%%
ttest_rel_print_df(stat_df, None, "distmats12_L3", "distmats12_FCalt_L3")
#%%
ttest_rel_print_df(stat_df, None, "distmats12_L3", "distmats12_BGalt_L3")
#%%
ttest_rel_print_df(stat_df, None, "distmats02_L3", "distmats02_FCalt_L3")
#%%
ttest_rel_print_df(stat_df, None, "distmats02_L4", "distmats02_FCalt_L4")

#%%
figh = paired_strip_plot(stat_df, None, "distmats02_L3", "distmats02_FCalt_L3")
figh.suptitle(f"{layerstr}\nimg dist FC-BG vs FC'-BG, CholCMA BG")
figh.gca().set_ylabel("cosine dist (layer3)")
saveallforms(figdir, f"{layerstr}_imgdist_FCBG_vs_FC'BG_CholCMABG_L3{suffix}", figh, )
figh.show()
#%%
figh = paired_strip_plot(stat_df, None, "distmats02_L3", "distmats02_BGalt_L3")
figh.suptitle(f"{layerstr}\nimg dist FC-BG vs FC-BG', CholCMA BG")
figh.gca().set_ylabel("cosine dist (layer3)")
saveallforms(figdir, f"{layerstr}_imgdist_FCBG_vs_FCBG'_CholCMABG_L3{suffix}", figh, )
figh.show()