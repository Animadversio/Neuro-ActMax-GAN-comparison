# %%
import os
from os.path import join
from pathlib import Path
from collections import OrderedDict
from easydict import EasyDict as edict
import torch
import seaborn as sns
from matplotlib import cm
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, ttest_ind, ttest_1samp, ttest_rel
from statsmodels.stats.multitest import fdrcorrection
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
# %%
source_data_dir = r"/Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/Manuscript_BigGAN/Submissions/Manuscript_BigGAN - NatNeuro/2025-10-Accepted-In-Principle-Docs/SourceData/ExtendedFig4_Source_data"

# %% [markdown]
# ### Ext Figure 4B

# %%

def mean_time_bin(psth, bin_size=5):
    """Bin the psth in time axis and return the mean of each bin."""
    n_timepoint = psth.shape[-1]
    n_bin = n_timepoint // bin_size
    psth_bin = psth[..., :n_bin*bin_size].reshape(psth.shape[:-1] + (n_bin, bin_size))
    return psth_bin.mean(axis=-1)

# %%
meta_df = pd.read_csv(join(source_data_dir, f"FigureExt4_meta_df.csv"), index_col=0)
V1msk = meta_df.visual_area == "V1"
V4msk = meta_df.visual_area == "V4"
ITmsk = meta_df.visual_area == "IT"
validmsk = meta_df.valid
thresh = 0.01
bothsucsmsk = (meta_df.p_maxinit_0 < thresh) & (meta_df.p_maxinit_1 < thresh)
FCsucsmsk = (meta_df.p_maxinit_0 < thresh)
BGsucsmsk = (meta_df.p_maxinit_1 < thresh)
bin_size = 5 # 5 10 20 25 50 
for bin_size in [5, 10, 20, 25, 50]:
    DeePSim_df = pd.read_csv(join(source_data_dir, f"FigureExt4B_DeePSim_diff_attrib_norm_bin_tsr_{bin_size}ms.csv"), index_col=0)
    BigGAN_df = pd.read_csv(join(source_data_dir, f"FigureExt4B_BigGAN_diff_attrib_norm_bin_tsr_{bin_size}ms.csv"), index_col=0)
    diff_attrib_norm_bin_tsr = np.stack([DeePSim_df.values, BigGAN_df.values], axis=-1)

    thread_colors = ['b', 'r']
    figh, axs = plt.subplots(1, 3, figsize=[9, 3.5], sharex=True, sharey=True)
    time_ticks = np.arange(0, 200, bin_size) + bin_size / 2
    for i, (visual_area, area_mask) in enumerate(zip(["V1", "V4", "IT"], [V1msk, V4msk, ITmsk])):
        ax = axs[i]
        diff_attrib_both = np.zeros((2, diff_attrib_norm_bin_tsr.shape[1]))
        for thread, GANname, thread_sucsmsk in zip([0, 1],
                                                ["DeePSim", "BigGAN"],
                                                [FCsucsmsk, BGsucsmsk]):

            mask = area_mask & validmsk & thread_sucsmsk
            diff_attrib = diff_attrib_norm_bin_tsr[mask].mean(axis=0)
            diff_attrib_sem = diff_attrib_norm_bin_tsr[mask].std(axis=0) / np.sqrt(mask.sum())
            diff_attrib_both[thread, :] = diff_attrib[:, thread]
            ax.plot(time_ticks, diff_attrib[:, thread], color=thread_colors[thread],
                    label=f"{GANname} N={mask.sum()}")
            ax.fill_between(time_ticks,
                            diff_attrib[:, thread] - diff_attrib_sem[:, thread],
                            diff_attrib[:, thread] + diff_attrib_sem[:, thread], color=thread_colors[thread], alpha=0.3)
        pvals = []
        tvals = []
        for t in range(diff_attrib_norm_bin_tsr.shape[1]):
            tval, p = ttest_ind(diff_attrib_norm_bin_tsr[area_mask & validmsk & FCsucsmsk][:,t,0], 
                                diff_attrib_norm_bin_tsr[area_mask & validmsk & BGsucsmsk][:,t,1])
            pvals.append(p)
            tvals.append(tval)
        
        pvals = np.array(pvals)
        tvals = np.array(tvals)
        signif_orig = pvals < 0.05
        fdr_reject, pvals_fdr = fdrcorrection(pvals, alpha=0.05)
        signif_fdr = fdr_reject # pvals_fdr < 0.05
        for signif, signif_str, y_offset in zip([signif_orig, signif_fdr], ["orig", "fdr"], [0, 0.1]):
            annot_y = diff_attrib_both.max() * (1.15 + y_offset)
            ax.plot(time_ticks[(tvals > 0) & signif], np.ones(np.sum((tvals > 0) & signif))*annot_y, 'b.', markersize=4)
            ax.plot(time_ticks[(tvals < 0) & signif], np.ones(np.sum((tvals < 0) & signif))*annot_y, 'r.', markersize=4)
        print(visual_area, "orig", pvals[signif_orig], tvals[signif_orig])
        print(visual_area, "fdr", pvals_fdr[signif_fdr], tvals[signif_fdr])
        
        ax.axhline(0, color='k', ls='--', lw=1)
        ax.set_title(visual_area)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Fraction of activation difference")
        ax.set_xlim([0, 200])
        # ax.set_ylim([-0.05, 0.30])
        ax.legend(loc="lower right")
    for ax in axs:
        ax.relim()           # Re-compute the data limits
        ax.margins(y=0.15)
        ax.autoscale_view()  # Update the view to the new limits
    plt.suptitle(f"Activation increase attributed to different {bin_size}ms time windows [Evol succ criterion p < {thresh}]")
    plt.tight_layout()
    # saveallforms(figdir, f"act_increase_attrib_threadsucs_3area_thr{thresh}_{bin_str}", figh)
    plt.show()



# %% [markdown]
# ### Ext Figure 4C

# %%
def pad_normalize_multi_wdw_resp_trajs(rsp_wdws, resp_col_multi_wdw, full_normalizer, ):
    normresp_extrap_arr_univ_col = {}
    for colj, rsp_wdw in enumerate(rsp_wdws):
        # resp_col, _ = extract_all_evol_trajectory_dyna(BFEStats, rsp_wdw=rsp_wdw)
        resp_col = resp_col_multi_wdw[colj]
        resp_extrap_arr, extrap_mask_arr, max_len = pad_resp_traj(resp_col)
        normresp_extrap_arr_univ = resp_extrap_arr / full_normalizer
        normresp_extrap_arr_univ_col[rsp_wdw] = normresp_extrap_arr_univ
    return normresp_extrap_arr_univ_col


def plot_normalized_response_trajectories_from_precomputed_normresp(normresp_extrap_arr_univ_col, area_masks, area_labels,
                                          commonmsk, signif_test=False, signif_alpha=0.05, plot_individual_exp=True,
                                          mcc_corrections=["nomcc", "fdr", "bonf"],
                                          panel_width=3, panel_height=3):

    figh, axs = plt.subplots(len(area_masks), len(rsp_wdws), figsize=(panel_width * len(rsp_wdws), panel_height * len(area_masks) + .5), sharey="row", )
    for colj, (rsp_wdw, normresp_extrap_arr_univ) in enumerate(normresp_extrap_arr_univ_col.items()):
        for rowi, (msk_major, label_major) in enumerate(zip(area_masks, area_labels)):
            msk = msk_major & commonmsk
            if plot_individual_exp:
                axs[rowi, colj].plot(normresp_extrap_arr_univ[msk, :, 0].T, color="blue", alpha=0.2, lw=0.7, label=None)
                axs[rowi, colj].plot(normresp_extrap_arr_univ[msk, :, 1].T, color="red", alpha=0.2, lw=0.7, label=None)
            
            mean_trace_FC = normresp_extrap_arr_univ[msk, :, 0].mean(axis=0)
            sem_trace_FC = normresp_extrap_arr_univ[msk, :, 0].std(axis=0) / np.sqrt(msk.sum())
            mean_trace_BG = normresp_extrap_arr_univ[msk, :, 1].mean(axis=0)
            sem_trace_BG = normresp_extrap_arr_univ[msk, :, 1].std(axis=0) / np.sqrt(msk.sum())
            axs[rowi, colj].plot(mean_trace_FC, color="blue", lw=3, label="DeePSim")
            axs[rowi, colj].fill_between(np.arange(len(mean_trace_FC)),
                                         mean_trace_FC - sem_trace_FC,
                                         mean_trace_FC + sem_trace_FC,
                                         color="blue", alpha=0.25, label=None)
            axs[rowi, colj].plot(mean_trace_BG, color="red", lw=3, label="BigGAN")
            axs[rowi, colj].fill_between(np.arange(len(mean_trace_BG)),
                                         mean_trace_BG - sem_trace_BG,
                                         mean_trace_BG + sem_trace_BG,
                                         color="red", alpha=0.25, label=None)
            axs[rowi, colj].set_title(f"[{rsp_wdw[0]}, {rsp_wdw[-1] + 1}ms]")
            if colj == 0:
                axs[rowi, 0].set_ylabel(f"Max Normalized Response\n{label_major} (N={msk.sum()})")
            
            if signif_test:
                # We'll do a two-sample t-test at each time point comparing FC vs BG
                FC_arr = normresp_extrap_arr_univ[msk, :, 0]  # shape: (num_units, time)
                BG_arr = normresp_extrap_arr_univ[msk, :, 1]  # shape: (num_units, time)

                pvals = []
                tstat_signs = []  # Store which value is larger
                for t in range(FC_arr.shape[1]):
                    # paired t test
                    tstat, p = stats.ttest_rel(FC_arr[:, t], BG_arr[:, t])
                    pvals.append(p)
                    tstat_signs.append(tstat > 0)  # True if FC > BG
                pvals = np.array(pvals)
                tstat_signs = np.array(tstat_signs)
                sig_mask_nomcc = pvals < signif_alpha
                # Multiple comparison correction
                # FDR correction
                alpha = signif_alpha  # significance threshold
                reject, pvals_fdr, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
                sig_mask_fdr = reject  # Boolean array: True if significant at alpha
                # Bonferroni correction 
                breakpoint()
                pvals_bonf = pvals * len(pvals)  # Multiply by number of comparisons
                sig_mask_bonf = pvals_bonf < signif_alpha
                print(label_major, "nomcc", pvals[sig_mask_nomcc], tstat_signs[sig_mask_nomcc])
                print(label_major, "fdr", pvals_fdr[sig_mask_fdr], tstat_signs[sig_mask_fdr])
                print(label_major, "bonf", pvals_bonf[sig_mask_bonf], tstat_signs[sig_mask_bonf])
                # Plot both FDR and Bonferroni significant points at different heights
                for mcc_correction in mcc_corrections:
                    if mcc_correction == "nomcc":
                        # Split into FC>BG and BG>FC points
                        fc_higher = sig_mask_nomcc & tstat_signs
                        bg_higher = sig_mask_nomcc & ~tstat_signs
                        axs[rowi, colj].plot(np.where(fc_higher)[0],
                                    np.ones(np.sum(fc_higher))*1.3,
                                    'b.', markersize=4,
                                    label=f'p<{signif_alpha}' if colj==0 and rowi==0 else "")
                        axs[rowi, colj].plot(np.where(bg_higher)[0],
                                    np.ones(np.sum(bg_higher))*1.3,
                                    'r.', markersize=4,
                                    label=None)
                    elif mcc_correction == "fdr":
                        fc_higher = sig_mask_fdr & tstat_signs
                        bg_higher = sig_mask_fdr & ~tstat_signs
                        axs[rowi, colj].plot(np.where(fc_higher)[0],
                                    np.ones(np.sum(fc_higher))*1.35,
                                    'b.', markersize=4,
                                    label=f'FDR p<{signif_alpha}' if colj==0 and rowi==0 else "")
                        axs[rowi, colj].plot(np.where(bg_higher)[0],
                                    np.ones(np.sum(bg_higher))*1.35,
                                    'r.', markersize=4,
                                    label=None)
                    elif mcc_correction == "bonf":
                        fc_higher = sig_mask_bonf & tstat_signs
                        bg_higher = sig_mask_bonf & ~tstat_signs
                        axs[rowi, colj].plot(np.where(fc_higher)[0],
                                    np.ones(np.sum(fc_higher))*1.40,
                                    'b.', markersize=4,
                                    label=f'Bonf p<{signif_alpha}' if colj==0 and rowi==0 else "")
                        axs[rowi, colj].plot(np.where(bg_higher)[0],
                                    np.ones(np.sum(bg_higher))*1.40,
                                    'r.', markersize=4,
                                    label=None)

    for ax in axs.ravel():
        ax.set_xlim([-0.5, 45.5])
        ax.set_ylim([0, 1.5])

    axs[0, 0].legend(loc="upper right", frameon=False)
    plt.show()
    return figh


# %% [markdown]
# ### Reproduce figure

# %%
meta_df = pd.read_csv(join(source_data_dir, f"FigureExt4_meta_df.csv"), index_col=0)
V1msk = meta_df.visual_area == "V1"
V4msk = meta_df.visual_area == "V4"
ITmsk = meta_df.visual_area == "IT"
validmsk = meta_df.valid
thresh = 0.01
bothsucsmsk = (meta_df.p_maxinit_0 < thresh) & (meta_df.p_maxinit_1 < thresh)
FCsucsmsk = (meta_df.p_maxinit_0 < thresh)
BGsucsmsk = (meta_df.p_maxinit_1 < thresh)

# %%
# for wdw_col_str, rsp_wdws in window_configs:
slice_names = ["DeePSim_mean", "BigGAN_mean", "DeePSim_sem", "BigGAN_sem"]
for window_length  in (10, 20, 25, 50):
    wdw_col_str = f"{window_length}ms_wdw"
    rsp_wdws = [range(i*window_length, (i+1)*window_length) for i in range(200//window_length)]
    normresp_extrap_arr_univ_col = {}
    for rsp_wdw in rsp_wdws:
        slices = []
        for slice_name in slice_names:
            savepath = join(source_data_dir, f"FigureExtended4C_src_normresp_wdw_{rsp_wdw.start}-{rsp_wdw.stop}_{slice_name}.csv")
            df = pd.read_csv(savepath)
            slices.append(df.values)
        # Stack along the last axis and store in the dict
        normresp_extrap_arr_univ_col[rsp_wdw] = np.stack(slices, axis=-1)
    
    for commonmsk, commonmsk_title_str, commonmsk_str in [(validmsk & bothsucsmsk, "Valid & Both Success",  "valid_bothsucc"),
                                                        # (validmsk & sucsmsk, "Valid & Any Success", "valid_succ"),
                                                        # (validmsk, "Valid", "valid"),
                                                        ]:
        figh = plot_normalized_response_trajectories_from_precomputed_normresp(normresp_extrap_arr_univ_col, 
                                [V4msk, ITmsk], ["V4", "IT"], 
                                commonmsk, signif_alpha=0.05, signif_test=True, 
                                plot_individual_exp=False, mcc_corrections=["fdr"],
                                panel_width=3, panel_height=3)
        figh.suptitle(
            f"Universal Max Normalized response {wdw_col_str} across blocks [{commonmsk_title_str} Sessions]")
        figh.tight_layout()
        # saveallforms([outdir, figdir], f"univmaxnorm_resp_traj_{commonmsk_str}_area_sep_{wdw_col_str}_synopsis_annot_sigif_mcc_fdr_V4IT", figh=figh)
        figh.show()

# %%



