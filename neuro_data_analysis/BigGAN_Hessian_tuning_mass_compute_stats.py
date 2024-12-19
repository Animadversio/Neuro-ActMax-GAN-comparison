# %%
%load_ext autoreload
%autoreload 2
#%%
import os
from os.path import join
import re
import math
import pickle as pkl
import time
import glob
from PIL import Image
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
from neuro_data_analysis.mat_data_translate_lib import h5_to_dict_simplify, print_hdf5_info
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from core.utils.plot_utils import saveallforms
from core.utils.colormap_matlab import parula, viridis

# set the pandas display options width
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

mat_root = r"S:\Data-Ephys-MAT"
pkl_root = r"S:\Data-Ephys-PKL"
exp_record_pathdict = {"Alfa": r"S:\Exp_Record_Alfa.xlsx", 
                       "Beto": r"S:\ExpSpecTable_Augment.xlsx",
                       "Caos": r"S:\Exp_Record_Caos.xlsx",
                       "Diablito": r"S:\Exp_Record_Diablito.xlsx"}

#%%
def plot_heatmap(grouped, space, ax, CLIM):
    """Plot heatmap for a given space"""
    space_data = grouped[grouped['space_name'] == space]
    pivot_table = space_data.pivot(index='eig_id', columns='lin_dist', values='pref_unit_resp')
    pivot_table = pivot_table.astype(float)
    if pivot_table.empty:
        return
    plt.sca(ax)
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap=parula, 
                cbar_kws={'label': 'Preferred Unit Response'}, ax=ax, vmin=CLIM[0], vmax=CLIM[1])
    plt.title(f'Heatmap of Preferred Unit Response for Space: {space}')
    plt.xlabel('Linear Distance')
    plt.ylabel('Eigenvalue ID')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.axis('image')


def plot_tuning_curves(filtered_df, space, prefchan_bsl_mean, prefchan_bsl_sem, exprow, prefchan_str, compute_stats=True):
    """Plot tuning curves for each eigenvector"""
    if filtered_df.empty:
        return
        
    unique_eig_ids = sorted(filtered_df['eig_id'].unique())
    num_eig_ids = len(unique_eig_ids)
    max_cols = 4
    cols = min(max_cols, num_eig_ids)
    rows = math.ceil(num_eig_ids / cols)
    
    fig_width = cols * 4
    fig_height = rows * 3.5 + 0.5
    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), 
                           sharex=True, sharey=True,
                           squeeze=False)
    axs = axs.flatten()
    for i, eig_id in enumerate(unique_eig_ids):
        ax = axs[i]
        breakpoint()
        subset = filtered_df[filtered_df['eig_id'] == eig_id]
        F_value = None
        p_value = None
        stats_str = ""
        if compute_stats and len(subset.lin_dist.unique()) > 1:
            # Perform ANOVA, only if there are more than 1 data point
            try:
                model = ols('pref_unit_resp ~ C(lin_dist)', data=subset).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                F_value = anova_table.loc['C(lin_dist)', 'F']
                p_value = anova_table.loc['C(lin_dist)', 'PR(>F)']
                stats_str = f"F-val: {F_value:.2f} | p-val: {p_value:.1e}"
            except Exception as e:
                print(f"Error performing ANOVA for eig_id {eig_id}: {e}")
                stats_str = ""
        print(f"Eig ID: {eig_id} | {stats_str}")
        sns.lineplot(data=subset, x='lin_dist', y='pref_unit_resp', ax=ax, marker='o')
        ax.axhline(prefchan_bsl_mean, color='black', linestyle='--', label='Baseline Mean')
        ax.axhline(prefchan_bsl_mean + prefchan_bsl_sem, color='black', linestyle=':', label='Baseline SEM')
        ax.axhline(prefchan_bsl_mean - prefchan_bsl_sem, color='black', linestyle=':')
        ax.set_title(f'Eig ID: {eig_id} | {stats_str}')
        ax.set_xlabel('Linear Distance')
        ax.set_ylabel('Preferred Unit Response')

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.suptitle(f'Preferred Unit Response for Different Eigenvectors {space}\n {exprow.ephysFN} | Pref Channel {prefchan_str} ')
    # plt.legend(title='Space Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    return fig


def display_image_grid(noise_grid_img, class_grid_img, expstr):
    """
    Display image grids for noise and class spaces.

    Parameters:
        noise_grid_img (PIL.Image or None): Image grid for noise space.
        class_grid_img (PIL.Image or None): Image grid for class space.
        expstr (str): Experiment string.
    """
    figh = plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    if class_grid_img is not None:
        plt.imshow(class_grid_img)
    plt.axis('off')
    plt.title('Image Grid for class space')
    
    plt.subplot(1, 2, 2)
    if noise_grid_img is not None:
        plt.imshow(noise_grid_img)
    plt.axis('off')
    plt.title('Image Grid for noise space')
    
    plt.suptitle(f'BigGAN Hessian Frame Stimuli set \n {expstr}')
    plt.tight_layout()
    plt.show()
    return figh


def display_image_grid_with_scores(noise_grid_img_score, class_grid_img_score, expstr):
    """
    Display image grids with scores for noise and class spaces.

    Parameters:
        noise_grid_img_score (PIL.Image or None): Image grid with scores for noise space.
        class_grid_img_score (PIL.Image or None): Image grid with scores for class space.
        ephysFN (str): Ephys file name.
        prefchan_str (str): Preferred channel string.
    """
    figh = plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 2, 1)
    if class_grid_img_score is not None:
        plt.imshow(class_grid_img_score)
    plt.axis('off')
    plt.title('Image Grid for class space')
    
    plt.subplot(1, 2, 2)
    if noise_grid_img_score is not None:
        plt.imshow(noise_grid_img_score)
    plt.axis('off')
    plt.title('Image Grid for noise space')
    
    plt.suptitle(f'Preferred Unit Response for Different Spaces and Eigenvectors \n {expstr}')
    plt.tight_layout()
    plt.show()
    return figh

#%% Main analysis pipeline
from neuro_data_analysis.neural_tuning_analysis_lib import organize_unit_info, maybe_add_unit_id_to_meta, \
    calculate_neural_responses, parse_stim_info, find_full_image_paths, load_space_images
from core.utils.montage_utils import PIL_array_to_montage, PIL_array_to_montage_score_frame
from core.utils.dataset_utils import ImagePathDataset

#%% 
from tqdm.auto import tqdm
ExpRecord_Hessian_All = pd.read_csv(r"ExpRecord_BigGAN_Hessian_tuning_ABCD_w_meta.csv")
ExpRecord_Evol_All = pd.read_csv(r"ExpRecord_BigGAN_Hessian_Evol_ABCD_w_meta.csv")
figroot = f"E:\OneDrive - Harvard University\BigGAN_Hessian"
tuning_stats_synopsis = []
# ExpRecord_Hessian_All = ExpRecord_Hessian_All.sort_values(by=["Animal", "Expi"]).reset_index(drop=True)
for rowi, exprow in tqdm(ExpRecord_Hessian_All.iterrows()):
    print("Processing: row", rowi, exprow.ephysFN, exprow.Animal, exprow.Expi)
    figdir = join(figroot, exprow.ephysFN)
    os.makedirs(figdir, exist_ok=True)
    
    # Load data
    data = pkl.load(open(join(pkl_root, f"{exprow.ephysFN}.pkl"), "rb"))
    rasters = data["rasters"]
    meta = data["meta"]
    Trials = data["Trials"]
    stimuli_dir = exprow.stimuli
    imageName = np.squeeze(Trials.imageName)
    # Process unit information
    meta = maybe_add_unit_id_to_meta(meta, rasters,) # for older experiments, unit_id is not in the meta file
    unit_info = organize_unit_info(meta, exprow)
    prefchan_id = unit_info["prefchan_id"]
    prefchan_str = unit_info["prefchan_str"]
    expstr = f"{exprow.ephysFN} | Pref Channel {prefchan_str}"
    
    # Process image names
    unique_imgnames = np.unique(imageName)
    stim_info_df = parse_stim_info(unique_imgnames)
    indices_per_name = {name: np.where(imageName == name)[0] for name in unique_imgnames}
    stim_info_df["trial_ids"] = stim_info_df.apply(lambda row: indices_per_name[row["img_name"]], axis=1)
    uniq_img_fps = find_full_image_paths(stimuli_dir, unique_imgnames)
    
    # make the image dataset
    # stimuli_dataset = ImagePathDataset(list(uniq_img_fps.values()), scores=None, img_dim=(256, 256))

    # Calculate responses
    resp_info = calculate_neural_responses(rasters, prefchan_id)
    prefchan_resp_sgtr = resp_info["prefchan_resp_sgtr"]
    prefchan_bsl_mean = resp_info["prefchan_bsl_mean"]
    prefchan_bsl_sem = resp_info["prefchan_bsl_sem"]
    
    # Create response dataframe
    sgtr_resp_df = pd.DataFrame({"img_name": imageName, "pref_unit_resp": prefchan_resp_sgtr[:, 0]})
    # annotate the response dataframe with the stimulus information by merging on the image name
    sgtr_resp_df = sgtr_resp_df.merge(stim_info_df.drop(columns=['trial_ids']), on="img_name")
    # compute the average response per space and eigenvector
    pref_avgresp_df = sgtr_resp_df.groupby(['space_name', 'eig_id', 'lin_dist']).agg({'pref_unit_resp': 'mean'}).reset_index()
    pref_avg_resp_class = pref_avgresp_df.query(f"space_name == 'class'")
    pref_avg_resp_noise = pref_avgresp_df.query(f"space_name == 'noise'")
    pref_avg_resp_noise_mat = pref_avg_resp_noise.pivot(index='eig_id', columns='lin_dist', values='pref_unit_resp')
    pref_avg_resp_class_mat = pref_avg_resp_class.pivot(index='eig_id', columns='lin_dist', values='pref_unit_resp')
    
    tuning_stats_col = []
    for space in ["class", "noise"]:
        sgtr_resp_per_space = sgtr_resp_df.query(f"space_name == '{space}'")
        unique_eig_ids = sorted(sgtr_resp_per_space['eig_id'].unique())
        num_eig_ids = len(unique_eig_ids)
        for i, eig_id in enumerate(unique_eig_ids):
            subset = sgtr_resp_per_space[sgtr_resp_per_space['eig_id'] == eig_id]
            unique_lin_dists = sorted(subset.lin_dist.unique())
            # TODO: if 0 is not in this set fetch the 0 from other eig_id
            F_value = None
            p_value = None
            stats_str = ""
            if len(unique_lin_dists) > 1:
                # Perform ANOVA, only if there are more than 1 data point
                try:
                    model = ols('pref_unit_resp ~ C(lin_dist)', data=subset).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    F_value = anova_table.loc['C(lin_dist)', 'F']
                    p_value = anova_table.loc['C(lin_dist)', 'PR(>F)']
                    stats_str = f"F-val: {F_value:.2f} | p-val: {p_value:.1e}"
                except Exception as e:
                    print(f"Error performing ANOVA for eig_id {eig_id}: {e}")
                    stats_str = ""
            # find avg resp for each lin_dist
            avg_resp_per_lin_dist = subset.groupby('lin_dist').agg({'pref_unit_resp': 'mean'}).reset_index().sort_values(by='lin_dist')
            # find lin_dist with max resp
            max_resp_lin_dist = avg_resp_per_lin_dist.loc[avg_resp_per_lin_dist['pref_unit_resp'].idxmax(), 'lin_dist']
            max_resp_val = avg_resp_per_lin_dist.loc[avg_resp_per_lin_dist['pref_unit_resp'].idxmax(), 'pref_unit_resp']
            stats_dict = {"space_name": space, "eig_id": eig_id, "F_value": F_value, "p_value": p_value, "stats_str": stats_str, 
                          "lin_dist_set": unique_lin_dists, "lin_dist_num": len(unique_lin_dists), 
                          "avg_resp_per_lin_dist": avg_resp_per_lin_dist["pref_unit_resp"].values.tolist(),
                          "max_resp_lin_dist": max_resp_lin_dist, "max_resp_val": max_resp_val}
            tuning_stats_col.append(stats_dict)
    
    tuning_stats_df = pd.DataFrame(tuning_stats_col)
    tuning_stats_df["Animal"] = exprow.Animal
    tuning_stats_df["Expi"] = exprow.Expi
    tuning_stats_df["ephysFN"] = exprow.ephysFN
    tuning_stats_df["stimuli"] = exprow.stimuli
    tuning_stats_df["prefchan"] = exprow.pref_chan
    tuning_stats_df["prefunit"] = exprow.pref_unit
    tuning_stats_df.to_csv(join(figdir, f"tuning_curves_stats_df.csv"), index=False)
    tuning_stats_synopsis.append(tuning_stats_df)
    # break
    # Group and plot heatmaps
    # CLIM = np.quantile(pref_avgresp_df['pref_unit_resp'], [0.02, 0.98])
    # figh, axs = plt.subplots(1, 2, figsize=(13, 6))
    # for ax, space in zip(axs, ['class', 'noise']):
    #     plot_heatmap(pref_avgresp_df, space, ax, CLIM)
    # plt.suptitle(f'Preferred Unit Response for Different Spaces and Eigenvectors \n {exprow.ephysFN} | Pref Channel {prefchan_str} ')
    # plt.tight_layout()
    # saveallforms(figdir, f"preferred_unit_response_heatmap_py")
    # plt.show()

    # Plot tuning curves
    # for space in ["class", "noise"]:
    #     sgtr_resp_per_space = sgtr_resp_df.query(f"space_name == '{space}'")
    #     fig = plot_tuning_curves(sgtr_resp_per_space, space, prefchan_bsl_mean, prefchan_bsl_sem, exprow, prefchan_str, compute_stats=True)
    #     saveallforms(figdir, f"preferred_unit_{space}_tuning_curve_ANOVA_py", fig)

    # # load the images for the noise and class spaces
    # noise_imgname_table, noise_image_array = load_space_images("noise", stim_info_df, uniq_img_fps)
    # class_imgname_table, class_image_array = load_space_images("class", stim_info_df, uniq_img_fps)
    # noise_grid_img = PIL_array_to_montage(noise_image_array)
    # noise_grid_img_score = PIL_array_to_montage_score_frame(noise_image_array, pref_avg_resp_noise_mat.values, colormap=parula, border_size=24, clim=CLIM)
    # class_grid_img = PIL_array_to_montage(class_image_array)
    # class_grid_img_score = PIL_array_to_montage_score_frame(class_image_array, pref_avg_resp_class_mat.values, colormap=parula, border_size=24, clim=CLIM)
    # save the images
    # fig_grid = display_image_grid(noise_grid_img, class_grid_img, expstr)
    # saveallforms(figdir, f"stimuli_grid_plot_py", fig_grid)
    # fig_grid_score = display_image_grid_with_scores(noise_grid_img_score, class_grid_img_score, expstr)
    # saveallforms(figdir, f"stimuli_grid_pref_resp_colorframe_py", fig_grid_score)
#%%
syndir = join(figroot, "synopsis")
os.makedirs(syndir, exist_ok=True)
tuning_stats_synopsis_df = pd.concat(tuning_stats_synopsis, ignore_index=True)
tuning_stats_synopsis_df.to_csv(join(syndir, f"ABCD_tuning_stats_synopsis.csv"), index=False)
tuning_stats_synopsis_df.to_pickle(join(syndir, f"ABCD_tuning_stats_synopsis.pkl"))
