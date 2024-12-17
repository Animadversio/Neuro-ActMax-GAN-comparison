# %%
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

ExpRecord_CD = pd.concat([pd.read_excel(exp_record_pathdict[Animal]) for Animal in ("Caos", "Diablito")])
exp_mask = ExpRecord_CD.Exp_collection.str.contains('BigGAN_Hessian', na=False) & ~ ExpRecord_CD.Expi.isna()
ExpRecord_Hessian = ExpRecord_CD.loc[exp_mask, :]
exp_mask = ExpRecord_CD.Exp_collection.str.contains('BigGAN_FC6', na=False) & ~ ExpRecord_CD.Expi.isna()
ExpRecord_Evol = ExpRecord_CD.loc[exp_mask, :]
print(ExpRecord_Hessian)
print(ExpRecord_Evol)
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
    max_cols = 3
    cols = min(max_cols, num_eig_ids)
    rows = math.ceil(num_eig_ids / cols)
    
    fig_width = cols * 4.5
    fig_height = rows * 3.5 + 0.5
    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), 
                           sharex=True, sharey=True,
                           squeeze=False)
    axs = axs.flatten()
    for i, eig_id in enumerate(unique_eig_ids):
        ax = axs[i]
        subset = filtered_df[filtered_df['eig_id'] == eig_id]
        if compute_stats:
            # Perform ANOVA
            model = ols('pref_unit_resp ~ C(lin_dist)', data=subset).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            F_value = anova_table.loc['C(lin_dist)', 'F']
            p_value = anova_table.loc['C(lin_dist)', 'PR(>F)']
        else:
            F_value = None
            p_value = None
        print(f"Eig ID: {eig_id} | F-value: {F_value:.4f} | p-value: {p_value:.4e}")
        sns.lineplot(data=subset, x='lin_dist', y='pref_unit_resp', ax=ax, marker='o')
        ax.axhline(prefchan_bsl_mean, color='black', linestyle='--', label='Baseline Mean')
        ax.axhline(prefchan_bsl_mean + prefchan_bsl_sem, color='black', linestyle=':', label='Baseline SEM')
        ax.axhline(prefchan_bsl_mean - prefchan_bsl_sem, color='black', linestyle=':')
        ax.set_title(f'Eig ID: {eig_id} | F-val: {F_value:.2f} | p-val: {p_value:.1e}')
        ax.set_xlabel('Linear Distance')
        ax.set_ylabel('Preferred Unit Response')

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.suptitle(f'Preferred Unit Response for Different Eigenvectors {space}\n {exprow.ephysFN} | Pref Channel {prefchan_str} ')
    plt.legend(title='Space Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    return fig

#%% Main analysis loop
from neuro_data_analysis.neural_tuning_analysis_lib import organize_unit_info, \
    calculate_neural_responses, parse_stim_info, find_full_image_paths
from core.utils.dataset_utils import ImagePathDataset
figroot = f"E:\OneDrive - Harvard University\BigGAN_Hessian"
for _, exprow in ExpRecord_Hessian.iterrows():
    print(exprow.ephysFN,exprow.Expi)
    figdir = join(figroot, exprow.ephysFN)
    os.makedirs(figdir, exist_ok=True)
    
    # Load data
    data = pkl.load(open(join(pkl_root, f"{exprow.ephysFN}.pkl"), "rb"))
    rasters = data["rasters"]
    meta = data["meta"]
    Trials = data["Trials"]
    imageName = np.squeeze(Trials.imageName)
    stimuli_dir = exprow.stimuli
    # Process unit information
    unit_info = organize_unit_info(meta, exprow)
    prefchan_id = unit_info["prefchan_id"]
    prefchan_str = unit_info["prefchan_str"]
    
    # Process image names
    unique_imgnames = np.unique(imageName)
    indices_per_name = {name: np.where(imageName == name)[0] for name in unique_imgnames}
    stim_info_df = parse_stim_info(unique_imgnames)
    uniq_img_fps = find_full_image_paths(stimuli_dir, unique_imgnames)
    
    # make the image dataset
    stimuli_dataset = ImagePathDataset(imgfps=list(uniq_img_fps.values()), scores=None, img_dim=(256, 256))
    
    # Calculate responses
    resp_info = calculate_neural_responses(rasters, prefchan_id)
    prefchan_resp_sgtr = resp_info["prefchan_resp_sgtr"]
    prefchan_bsl_mean = resp_info["prefchan_bsl_mean"]
    prefchan_bsl_sem = resp_info["prefchan_bsl_sem"]
    
    # Create response dataframe
    sgtr_resp_df = pd.DataFrame({"img_name": imageName, "pref_unit_resp": prefchan_resp_sgtr[:, 0]})
    sgtr_resp_df = sgtr_resp_df.merge(stim_info_df.drop(columns=['trial_ids']), on="img_name")
    
    # Group and plot heatmaps
    pref_avgresp_per_img = sgtr_resp_df.groupby(['space_name', 'eig_id', 'lin_dist']).agg({'pref_unit_resp': 'mean'}).reset_index()
    CLIM = np.quantile(pref_avgresp_per_img['pref_unit_resp'], [0.01, 0.99])
    figh, axs = plt.subplots(1, 2, figsize=(13, 6))
    for ax, space in zip(axs, ['class', 'noise']):
        plot_heatmap(pref_avgresp_per_img, space, ax, CLIM)
    plt.suptitle(f'Preferred Unit Response for Different Spaces and Eigenvectors \n {exprow.ephysFN} | Pref Channel {prefchan_str} ')
    plt.tight_layout()
    saveallforms(figdir, f"preferred_unit_response_heatmap_py")
    plt.show()

    # Plot tuning curves
    for space in ["class", "noise"]:
        sgtr_resp_per_space = sgtr_resp_df.query(f"space_name == '{space}'")
        fig = plot_tuning_curves(sgtr_resp_per_space, space, prefchan_bsl_mean, prefchan_bsl_sem, exprow, prefchan_str, compute_stats=True)
        saveallforms(figdir, f"preferred_unit_{space}_tuning_curve_ANOVA_py", fig)

# %%
space = "noise"
space_stim_df = stim_info_df.query(f"space_name == '{space}'")
if space_stim_df.empty:
    print(f"No {space} space data found")
else:
    pivot_table = space_stim_df.pivot(index='eig_id', columns='lin_dist', values='img_name')
    print(pivot_table)
    image_array = np.empty(pivot_table.shape, dtype=Image.Image)
    # for each entry in the pivot table, get the image path and load the image put in in a new array
    for ri, eig_id in enumerate(pivot_table.index):
        for ci, lin_dist in enumerate(pivot_table.columns):
            img_name = pivot_table.loc[eig_id, lin_dist]
            img_path = uniq_img_fps[img_name]
            img = Image.open(img_path)
            image_array[ri, ci] = img
    print(image_array.shape)

#%%
resp_df = sgtr_resp_df.query(f"space_name == '{space}'")
if resp_df.empty:
    print(f"No {space} space data found")
else:
    avgresp_df = resp_df.groupby(['eig_id', 'lin_dist']).agg({'pref_unit_resp': 'mean'}).reset_index()
    resp_pivot_table = avgresp_df.pivot(index='eig_id', columns='lin_dist', values='pref_unit_resp')
#%%
from core.utils.montage_utils import PIL_array_to_montage, PIL_array_to_montage_score_frame
grid_img = PIL_array_to_montage(image_array)
grid_img_score = PIL_array_to_montage_score_frame(image_array, resp_pivot_table.values, colormap=parula, border_size=24)
# Display the grid
plt.figure(figsize=(20,10))
plt.imshow(grid_img)
plt.axis('off')
plt.title(f'Image Grid for {space} space')
plt.show()
#%%
plt.figure(figsize=(20,10))
plt.imshow(grid_img_score)
plt.axis('off')
plt.title(f'Image Grid for {space} space')
plt.show()
# %%

