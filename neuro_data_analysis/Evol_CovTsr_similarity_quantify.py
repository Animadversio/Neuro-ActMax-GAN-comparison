import os
from os.path import join
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms
from core.utils.stats_utils import ttest_ind_print_df, ttest_rel_print_df, paired_strip_plot
#%%
statdir =  r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_activation_cmp"
meta_df = pd.read_csv(join(statdir, "meta_stats.csv"), index_col=0)
#%%
from neuro_data_analysis.neural_data_utils import get_all_masks
Amsk, Bmsk, V1msk, V4msk, ITmsk, \
    length_msk, spc_msk, sucsmsk, \
    bsl_unstable_msk, bsl_stable_msk, validmsk = \
    get_all_masks(meta_df)
#%%
def crop_center(corrmap, frac=0.1):
    H, W = corrmap.shape
    corrmap_crop = corrmap[int(H*frac):int(H*(1-frac)), int(W*frac):int(W*(1-frac))]
    return corrmap_crop

for layer in ["layer1", "layer2", "layer3", "layer4"]:
    corrmap = corr_map_dict[layer]
    corrmap_crop = crop_center(corrmap)
    print(corrmap_crop.shape)
#%%
cov_root = r"E:\Network_Data_Sync\corrFeatTsr_BigGAN"
cov_stat_col = []
for Expi in range(1, 191):
    if Expi not in meta_df.index:
        print(f"Exp {Expi} not in the meta_df. Skip")
        continue
    corr_map_dict = np.load(join(cov_root, f"Both_Exp{Expi:02d}_covtsr_corr.npz"),)
    S = dict()
    S["Expi"] = Expi
    for layer in ["layer1", "layer2", "layer3", "layer4"]:
        S[layer+"_cov_corr"] = corr_map_dict[layer+"_cov_corr"]
        # S[layer+"_T_corr"] = corr_map_dict[layer+"_T_corr"]
        corrmap = corr_map_dict[layer]
        S[layer+"_mean"] = np.nanmean(corrmap)
        S[layer+"_max"] = np.nanmax(corrmap)
        S[layer+"_q9"] = np.quantile(corrmap, 0.9)
        corrmap_crop = crop_center(corrmap)
        S[layer+"_crop_mean"] = np.nanmean(corrmap_crop)
        S[layer+"_crop_max"] = np.nanmax(corrmap_crop)
        S[layer+"_crop_q9"] = np.quantile(corrmap_crop, 0.9)

        corrmap = corr_map_dict[layer+"_cosine"][0]
        S[layer+"_cos_mean"] = np.nanmean(corrmap)
        S[layer+"_cos_max"] = np.nanmax(corrmap)
        S[layer+"_cos_q9"] = np.quantile(corrmap, 0.9)
        corrmap_crop = crop_center(corrmap)
        S[layer+"_cos_crop_mean"] = np.nanmean(corrmap_crop)
        S[layer+"_cos_crop_max"] = np.nanmax(corrmap_crop)
        S[layer+"_cos_crop_q9"] = np.quantile(corrmap_crop, 0.9)

        corrmap_thr = corr_map_dict[layer+"_Tthr3"]
        S[layer+"_thr_mean"] = np.nanmean(corrmap_thr)
        S[layer+"_thr_max"] = np.nanmax(corrmap_thr)
        S[layer+"_thr_q9"] = np.quantile(corrmap_thr, 0.9)
        corrmap_crop_thr = crop_center(corrmap_thr)
        S[layer+"_thr_crop_mean"] = np.nanmean(corrmap_crop_thr)
        S[layer+"_thr_crop_max"] = np.nanmax(corrmap_crop_thr)
        S[layer+"_thr_crop_q9"] = np.quantile(corrmap_crop_thr, 0.9)

        corrmap_thr = corr_map_dict[layer+"_Tthr3_cosine"][0]
        S[layer+"_cos_thr_mean"] = np.nanmean(corrmap_thr)
        S[layer+"_cos_thr_max"] = np.nanmax(corrmap_thr)
        S[layer+"_cos_thr_q9"] = np.quantile(corrmap_thr, 0.9)
        corrmap_crop_thr = crop_center(corrmap_thr)
        S[layer+"_cos_thr_crop_mean"] = np.nanmean(corrmap_crop_thr)
        S[layer+"_cos_thr_crop_max"] = np.nanmax(corrmap_crop_thr)
        S[layer+"_cos_thr_crop_q9"] = np.quantile(corrmap_crop_thr, 0.9)

    cov_stat_col.append(S)
cov_stat_df = pd.DataFrame(cov_stat_col)
#%%
outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Proto_covtsr_similarity"
cov_stat_df.to_csv(join(outdir, "covtsr_stat.csv"))
#%%
meta_cov_stat_df = pd.merge(meta_df.reset_index(), cov_stat_df, left_on="index", right_on="Expi")
meta_cov_stat_df.set_index("index", inplace=True)
#%%
# set df printing options, to show all columns
pd.set_option('display.max_columns', None)
#%%
plt.figure(figsize=[6, 6])
sns.scatterplot(data=meta_cov_stat_df[validmsk], x="blockN", y="layer1_crop_q9",
                hue="visual_area", s=64)
# plt.plot([0, .5], [0, .5], 'k--')
# plt.xlim([0, .5])
# plt.ylim([0, .5])
plt.title("Correlation of Covariance Tensor")
# plt.savefig(join(outdir, "covtsr_corr_scatter.png"))
# plt.savefig(join(outdir, "covtsr_corr_scatter.pdf"))
plt.show()
#%%
sucsmsk_both = (meta_df.p_maxinit_0 < 0.001) & (meta_df.p_maxinit_1 < 0.001)
sucsmsk_any = (meta_df.p_maxinit_0 < 0.001) | (meta_df.p_maxinit_1 < 0.001)
#%%
meta_cov_stat_df[validmsk & sucsmsk_both].groupby("visual_area", sort=False).agg({
                                                   "layer1_cov_corr": ["mean", "sem"],
                                                   "layer2_cov_corr": ["mean", "sem"],
                                                   "layer3_cov_corr": ["mean", "sem"],
                                                   "layer4_cov_corr": ["mean", "sem","count"]})
#%%
meta_cov_stat_df[validmsk & sucsmsk_both].groupby("visual_area", sort=False).agg({
                                                   "layer1_cos_thr_crop_q9": ["mean","sem"],
                                                   "layer2_cos_thr_crop_q9": ["mean","sem"],
                                                   "layer3_cos_thr_crop_q9": ["mean","sem"],
                                                   "layer4_cos_thr_crop_q9": ["mean","sem","count"]})
#%%
meta_cov_stat_df[validmsk & sucsmsk_both].groupby("visual_area", sort=False).agg({
                                                   "layer1_thr_crop_q9": ["mean","sem"],
                                                   "layer2_thr_crop_q9": ["mean","sem"],
                                                   "layer3_thr_crop_q9": ["mean","sem"],
                                                   "layer4_thr_crop_q9": ["mean","sem","count"]})
#%%
meta_cov_stat_df[validmsk & sucsmsk_both].groupby("visual_area", sort=False).agg({
                                                   "layer1_crop_max": ["mean","sem"],
                                                   "layer2_crop_max": ["mean","sem"],
                                                   "layer3_crop_max": ["mean","sem"],
                                                   "layer4_crop_max": ["mean","sem","count"]})
#%%
meta_cov_stat_df[validmsk & sucsmsk_any].groupby("visual_area", sort=False).agg({
                                                       "layer1_crop_max": ["mean","sem"],
                                                       "layer2_crop_max": ["mean","sem"],
                                                       "layer3_crop_max": ["mean","sem"],
                                                       "layer4_crop_max": ["mean","sem","count"]})
#%%
meta_cov_stat_df[validmsk & sucsmsk_both].groupby("visual_area", sort=False).agg({
                                                       "layer1_crop_q9": ["mean","sem"],
                                                       "layer2_crop_q9": ["mean","sem"],
                                                       "layer3_crop_q9": ["mean","sem"],
                                                       "layer4_crop_q9": ["mean","sem","count"]})
#%%
meta_cov_stat_df[validmsk & sucsmsk].groupby("visual_area", sort=False).agg({
                                                       "layer1_crop_mean": ["mean","sem"],
                                                       "layer2_crop_mean": ["mean","sem"],
                                                       "layer3_crop_mean": ["mean","sem"],
                                                       "layer4_crop_mean": ["mean","sem","count"]})
#%%
# TODO: key results
for layer in ["layer1", "layer2", "layer3", "layer4"]:
    ttest_ind_print_df(meta_cov_stat_df,
                       validmsk & sucsmsk_any & V4msk,
                       validmsk & sucsmsk_any & ITmsk, f"{layer}_cos_thr_crop_q9")
# layer1_cos_thr_crop_q9 (N=37) ~ (N=75) 0.526+-0.135 (N=37) ~ 0.324+-0.171 (N=75) tval: 1.95, pval: 6.3e-02
# layer2_cos_thr_crop_q9 (N=37) ~ (N=75) 0.270+-0.086 (N=37) ~ 0.206+-0.104 (N=75) tval: 2.13, pval: 3.7e-02
# layer3_cos_thr_crop_q9 (N=37) ~ (N=75) 0.124+-0.047 (N=37) ~ 0.132+-0.068 (N=75) tval: -0.60, pval: 5.5e-01
# layer4_cos_thr_crop_q9 (N=37) ~ (N=75) 0.111+-0.069 (N=37) ~ 0.152+-0.099 (N=75) tval: -2.18, pval: 3.1e-02
#%%
for layer in ["layer1", "layer2", "layer3", "layer4"]:
    ttest_ind_print_df(meta_cov_stat_df,
                   validmsk & ~sucsmsk_both & V4msk,
                   validmsk & sucsmsk_both & V4msk, f"{layer}_cos_thr_crop_q9")
#%%
for layer in ["layer1", "layer2", "layer3", "layer4"]:
    ttest_ind_print_df(meta_cov_stat_df,
                   validmsk & ~sucsmsk_any & ITmsk,
                   validmsk & sucsmsk_any & ITmsk, f"{layer}_cos_thr_crop_q9")
#%%
for layer in ["layer1", "layer2", "layer3", "layer4"]:
    ttest_ind_print_df(meta_cov_stat_df,
                   validmsk & ~sucsmsk_both & ITmsk,
                   validmsk & sucsmsk_both & ITmsk, f"{layer}_cos_thr_crop_q9")
#%%
for layer in ["layer1", "layer2", "layer3", "layer4"]:
    ttest_ind_print_df(meta_cov_stat_df,
                   validmsk & ~sucsmsk_both & ITmsk,
                   validmsk &  sucsmsk_both & ITmsk, f"{layer}_cos_thr_crop_max")
#%%
for layer in ["layer1", "layer2", "layer3", "layer4"]:
    ttest_ind_print_df(meta_cov_stat_df,
                   validmsk & sucsmsk_any & V4msk,
                   validmsk & sucsmsk_any & ITmsk, f"{layer}_cos_thr_q9")
# layer1_cos_thr_q9 (N=37) ~ (N=75) 0.518+-0.158 (N=37) ~ 0.316+-0.138 (N=75) tval: 2.35, pval: 2.8e-02
# layer2_cos_thr_q9 (N=37) ~ (N=75) 0.258+-0.075 (N=37) ~ 0.216+-0.100 (N=75) tval: 1.48, pval: 1.5e-01
# layer3_cos_thr_q9 (N=37) ~ (N=75) 0.126+-0.036 (N=37) ~ 0.133+-0.064 (N=75) tval: -0.56, pval: 5.8e-01
# layer4_cos_thr_q9 (N=37) ~ (N=75) 0.110+-0.060 (N=37) ~ 0.147+-0.087 (N=75) tval: -2.28, pval: 2.4e-02
#%%
for layer in ["layer1", "layer2", "layer3", "layer4"]:
    ttest_ind_print_df(meta_cov_stat_df,
                   validmsk & sucsmsk_any & V4msk,
                   validmsk & sucsmsk_any & ITmsk, f"{layer}_cos_thr_max")
# layer1_cos_thr_max (N=37) ~ (N=75) 0.700+-0.156 (N=37) ~ 0.674+-0.150 (N=75) tval: 0.83, pval: 4.1e-01
# layer2_cos_thr_max (N=37) ~ (N=75) 0.492+-0.118 (N=37) ~ 0.447+-0.152 (N=75) tval: 1.60, pval: 1.1e-01
# layer3_cos_thr_max (N=37) ~ (N=75) 0.260+-0.072 (N=37) ~ 0.282+-0.106 (N=75) tval: -1.17, pval: 2.5e-01
# layer4_cos_thr_max (N=37) ~ (N=75) 0.181+-0.079 (N=37) ~ 0.234+-0.116 (N=75) tval: -2.50, pval: 1.4e-02
#%%
for layer in ["layer1", "layer2", "layer3", "layer4"]:
    ttest_ind_print_df(meta_cov_stat_df,
                   validmsk & sucsmsk_any & V4msk,
                   validmsk & sucsmsk_any & ITmsk, f"{layer}_thr_crop_q9")
# layer1_thr_crop_q9 (N=37) ~ (N=75) 0.378+-0.093 (N=37) ~ 0.314+-0.182 (N=75) tval: 0.76, pval: 4.6e-01
# layer2_thr_crop_q9 (N=37) ~ (N=75) 0.251+-0.096 (N=37) ~ 0.179+-0.111 (N=75) tval: 2.61, pval: 1.1e-02
# layer3_thr_crop_q9 (N=37) ~ (N=75) 0.128+-0.067 (N=37) ~ 0.116+-0.071 (N=75) tval: 0.84, pval: 4.0e-01
# layer4_thr_crop_q9 (N=37) ~ (N=75) 0.101+-0.072 (N=37) ~ 0.119+-0.102 (N=75) tval: -0.98, pval: 3.3e-01
#%%
for layer in ["layer1", "layer2", "layer3", "layer4"]:
    ttest_ind_print_df(meta_cov_stat_df,
                   validmsk & sucsmsk_both & V4msk,
                   validmsk & sucsmsk_both & ITmsk, f"{layer}_crop_q9")
# layer1_crop_q9 (N=14) ~ (N=45) 0.440+-0.103 (N=14) ~ 0.362+-0.163 (N=45) tval: 1.68, pval: 9.8e-02
# layer2_crop_q9 (N=14) ~ (N=45) 0.296+-0.092 (N=14) ~ 0.237+-0.119 (N=45) tval: 1.71, pval: 9.3e-02
# layer3_crop_q9 (N=14) ~ (N=45) 0.158+-0.085 (N=14) ~ 0.148+-0.082 (N=45) tval: 0.40, pval: 6.9e-01
# layer4_crop_q9 (N=14) ~ (N=45) 0.116+-0.098 (N=14) ~ 0.154+-0.105 (N=45) tval: -1.18, pval: 2.4e-01
#%%
for layer in ["layer1", "layer2", "layer3", "layer4"]:
    ttest_ind_print_df(meta_cov_stat_df,
                   validmsk & sucsmsk_any & V4msk,
                   validmsk & sucsmsk_any & ITmsk, f"{layer}_crop_q9")
# layer1_crop_q9 (N=37) ~ (N=75) 0.426+-0.093 (N=37) ~ 0.356+-0.142 (N=75) tval: 2.72, pval: 7.6e-03
# layer2_crop_q9 (N=37) ~ (N=75) 0.292+-0.081 (N=37) ~ 0.227+-0.105 (N=75) tval: 3.34, pval: 1.1e-03
# layer3_crop_q9 (N=37) ~ (N=75) 0.166+-0.072 (N=37) ~ 0.144+-0.075 (N=75) tval: 1.45, pval: 1.5e-01
# layer4_crop_q9 (N=37) ~ (N=75) 0.153+-0.098 (N=37) ~ 0.154+-0.106 (N=75) tval: -0.06, pval: 9.6e-01
#%%
for layer in ["layer1", "layer2", "layer3", "layer4"]:
    ttest_ind_print_df(meta_cov_stat_df,
                   validmsk & V4msk,
                   validmsk & ITmsk, f"{layer}_crop_q9")
# layer1_crop_q9 (N=38) ~ (N=106) 0.421+-0.096 (N=38) ~ 0.338+-0.131 (N=106) tval: 3.58, pval: 4.8e-04
# layer2_crop_q9 (N=38) ~ (N=106) 0.289+-0.082 (N=38) ~ 0.214+-0.094 (N=106) tval: 4.35, pval: 2.6e-05
# layer3_crop_q9 (N=38) ~ (N=106) 0.163+-0.074 (N=38) ~ 0.136+-0.070 (N=106) tval: 2.07, pval: 4.0e-02
# layer4_crop_q9 (N=38) ~ (N=106) 0.149+-0.101 (N=38) ~ 0.146+-0.099 (N=106) tval: 0.15, pval: 8.8e-01
# %%
for layer in ["layer1", "layer2", "layer3", "layer4"]:
    ttest_ind_print_df(meta_cov_stat_df,
                       validmsk & sucsmsk_both & V4msk,
                       validmsk & sucsmsk_both & ITmsk, f"{layer}_crop_q9")
# layer1_crop_q9 (N=14) ~ (N=45) 0.440+-0.103 (N=14) ~ 0.362+-0.163 (N=45) tval: 1.68, pval: 9.8e-02
# layer2_crop_q9 (N=14) ~ (N=45) 0.296+-0.092 (N=14) ~ 0.237+-0.119 (N=45) tval: 1.71, pval: 9.3e-02
# layer3_crop_q9 (N=14) ~ (N=45) 0.158+-0.085 (N=14) ~ 0.148+-0.082 (N=45) tval: 0.40, pval: 6.9e-01
# layer4_crop_q9 (N=14) ~ (N=45) 0.116+-0.098 (N=14) ~ 0.154+-0.105 (N=45) tval: -1.18, pval: 2.4e-01
# %%
for layer in ["layer2", "layer3", "layer4"]:
    ttest_ind_print_df(meta_cov_stat_df,
                       validmsk & sucsmsk_both & V4msk,
                       validmsk & sucsmsk_both & ITmsk, f"{layer}_cov_corr")
# %%
for layer in ["layer2", "layer3", "layer4"]:
    ttest_ind_print_df(meta_cov_stat_df,
                       validmsk & ~sucsmsk_any & ITmsk,
                       validmsk & sucsmsk_any & ITmsk, f"{layer}_cov_corr")
# %%
for layer in ["layer1", "layer2", "layer3", "layer4"]:
    ttest_ind_print_df(meta_cov_stat_df,
                       validmsk & ~sucsmsk_any & ITmsk,
                       validmsk & sucsmsk_any & ITmsk, f"{layer}_crop_q9")
# layer1_crop_q9 (N=31) ~ (N=75) 0.296+-0.085 (N=31) ~ 0.356+-0.142 (N=75) tval: -2.16, pval: 3.3e-02
# layer2_crop_q9 (N=31) ~ (N=75) 0.184+-0.048 (N=31) ~ 0.227+-0.105 (N=75) tval: -2.17, pval: 3.3e-02
# layer3_crop_q9 (N=31) ~ (N=75) 0.114+-0.048 (N=31) ~ 0.144+-0.075 (N=75) tval: -2.07, pval: 4.1e-02
# layer4_crop_q9 (N=31) ~ (N=75) 0.125+-0.078 (N=31) ~ 0.154+-0.106 (N=75) tval: -1.38, pval: 1.7e-01
# %%
for layer in ["layer1", "layer2", "layer3", "layer4"]:
    ttest_ind_print_df(meta_cov_stat_df,
                       validmsk & sucsmsk_both & V4msk,
                       validmsk & ~sucsmsk_both & V4msk, f"{layer}_crop_q9")
# layer1_crop_q9 (N=14) ~ (N=24) 0.440+-0.103 (N=14) ~ 0.410+-0.092 (N=24) tval: 0.93, pval: 3.6e-01
# layer2_crop_q9 (N=14) ~ (N=24) 0.296+-0.092 (N=14) ~ 0.285+-0.078 (N=24) tval: 0.41, pval: 6.8e-01
# layer3_crop_q9 (N=14) ~ (N=24) 0.158+-0.085 (N=14) ~ 0.166+-0.068 (N=24) tval: -0.32, pval: 7.5e-01
# layer4_crop_q9 (N=14) ~ (N=24) 0.116+-0.098 (N=14) ~ 0.168+-0.099 (N=24) tval: -1.54, pval: 1.3e-01
# %%
for layer in ["layer1", "layer2", "layer3", "layer4"]:
    ttest_ind_print_df(meta_cov_stat_df,
                       validmsk & sucsmsk_both & ITmsk,
                       validmsk & ~sucsmsk_both & ITmsk, f"{layer}_crop_q9")
# layer1_crop_q9 (N=45) ~ (N=61) 0.362+-0.163 (N=45) ~ 0.321+-0.099 (N=61) tval: 1.62, pval: 1.1e-01
# layer2_crop_q9 (N=45) ~ (N=61) 0.237+-0.119 (N=45) ~ 0.197+-0.067 (N=61) tval: 2.21, pval: 2.9e-02
# layer3_crop_q9 (N=45) ~ (N=61) 0.148+-0.082 (N=45) ~ 0.126+-0.058 (N=61) tval: 1.61, pval: 1.1e-01
# layer4_crop_q9 (N=45) ~ (N=61) 0.154+-0.105 (N=45) ~ 0.140+-0.095 (N=61) tval: 0.70, pval: 4.9e-01
 #%%
plt.figure()
plt.scatter(meta_cov_stat_df[validmsk & ITmsk].layer2_crop_q9,
            (meta_cov_stat_df[validmsk & ITmsk].layer2_cov_corr))
plt.show()
#%%
# plot correlograms for all pairs of variables
#%%
plt.figure()
plt.scatter(meta_cov_stat_df[validmsk & ITmsk].layer2_crop_q9,
            meta_cov_stat_df[validmsk & ITmsk].layer2_cov_corr)
plt.show()
 #%%
for layer in ["layer1", "layer2", "layer3", "layer4"]:
    ttest_ind_print_df(meta_cov_stat_df, validmsk & sucsmsk_both & V4msk,
                   validmsk & sucsmsk_both & ITmsk, f"{layer}_cov_corr")

#%%
