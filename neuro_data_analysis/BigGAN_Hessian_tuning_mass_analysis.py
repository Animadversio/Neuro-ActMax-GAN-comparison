# %%
import os
from os.path import join
import re
import math
import pickle as pkl
import time
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
import glob
def find_full_image_paths(folder_path, image_names):
    """
    Searches the specified folder for image files whose stem matches the given image names.

    Parameters:
        folder_path (str): Path to the folder containing the images.
        image_names (list of str): List of image name stems to search for.

    Returns:
        dict: A dictionary mapping each imageName to its full filename. If no matching file is found, the value is None.
    """
    files = glob.glob(os.path.join(folder_path, "*"))
    file_map = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem in image_names:
            file_map[stem] = f
    return {img_name: file_map.get(img_name) for img_name in image_names}


def parse_stim_info(image_names):
    stim_info = []
    re_pattern = r'(noise|class)_eig(\d+)_lin([+-]?\d+\.\d+)'
    for name in image_names: 
        match = re.match(re_pattern, name)
        if match:
            space_name = match.groups()[0]
            eig_value = int(match.groups()[1])
            lin_value = float(match.groups()[2])
            stim_info.append({"img_name": name, "space_name": space_name, "eig_id": eig_value, "lin_dist": lin_value, "hessian_img": True, "trial_ids": indices_per_name[name]})
        else:
            stim_info.append({"img_name": name, "space_name": None, "eig_id": None, "lin_dist": None, "hessian_img": False, "trial_ids": indices_per_name[name]})

    stim_info_df = pd.DataFrame(stim_info)
    return stim_info_df

#%%

# %%
figroot = f"E:\OneDrive - Harvard University\BigGAN_Hessian"
for _, exprow in ExpRecord_Hessian.iterrows():
    print(exprow.ephysFN,exprow.Expi)
    figdir = join(figroot, exprow.ephysFN)
    os.makedirs(figdir, exist_ok=True)
    data = pkl.load(open(join(pkl_root, f"{exprow.ephysFN}.pkl"), "rb"))
    rasters = data["rasters"]   # trials x time x units
    meta = data["meta"]
    Trials = data["Trials"]
    imageName = np.squeeze(Trials.imageName)
    stimuli_dir = exprow.stimuli
    # %
    # Organize the unit information
    spikeID = meta.spikeID[0].astype(int)
    channel_id = spikeID # set alias
    unit_id = meta.unitID[0].astype(int)
    char_map = {0:"U", 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H'}
    unit_str = [f"{channel_id}{char_map[unit_id]}" for channel_id, unit_id in zip(channel_id, unit_id)]
    prefchan = exprow.pref_chan
    prefunit = exprow.pref_unit   # the prefchan is the first unit???
    prefchan_id_allunits = np.where((channel_id == prefchan))[0]
    prefchan_id = np.where((channel_id == prefchan) & (unit_id == prefunit))[0]
    prefchan_str = unit_str[prefchan_id.item()]
    prefchan_id_allunits
    # %
    # parse the image name
    unique_imgnames = np.unique(imageName)
    indices_per_name = {name: np.where(imageName == name)[0] for name in unique_imgnames} # indices of the trials with the same image name
    # % parse the image name into stim info
    stim_info_df = parse_stim_info(unique_imgnames)
    uniq_img_fps = find_full_image_paths(figdir, unique_imgnames)
    
    # % neural responses for preferred channel
    wdw = slice(50, 200)
    bslwdw = slice(0, 45)
    respmat = rasters[:, wdw, :].mean(axis=1)
    bslmat = rasters[:, bslwdw, :].mean(axis=1)
    prefchan_resp_sgtr = respmat[:, prefchan_id] 
    prefchan_bsl_sgtr = bslmat[:, prefchan_id]
    prefchan_bsl_mean = prefchan_bsl_sgtr.mean()
    prefchan_bsl_sem = stats.sem(prefchan_bsl_sgtr)
    # %
    sgtr_resp_df = pd.DataFrame({"img_name": imageName, "pref_unit_resp": prefchan_resp_sgtr[:, 0]})
    sgtr_resp_df = sgtr_resp_df.merge(stim_info_df.drop(columns=['trial_ids']), on="img_name");
    # sgtr_resp_df.columns # ['img_name', 'pref_unit_resp', 'space_name', 'eig_id', 'lin_dist', 'hessian_img']
    # %
    # Group by and compute the mean of preferred unit responses
    grouped = sgtr_resp_df.groupby(['space_name', 'eig_id', 'lin_dist']).agg({'pref_unit_resp': 'mean'}).reset_index()
    # Pivot the data to create a matrix for each space_name
    space_names = grouped['space_name'].unique()
    CLIM = np.quantile(grouped['pref_unit_resp'],[0.01, 0.99]) #grouped['pref_unit_resp'].quantile([0.01, 0.99])
    figh, axs = plt.subplots(1, 2, figsize=(13, 6))
    for ax, space in zip(axs, ['class', 'noise',]):
        plt.sca(ax)
        space_data = grouped[grouped['space_name'] == space]
        pivot_table = space_data.pivot(index='eig_id', columns='lin_dist', values='pref_unit_resp') # set values as float
        pivot_table = pivot_table.astype(float)
        if pivot_table.empty:
            continue
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap=parula, 
                    cbar_kws={'label': 'Preferred Unit Response'}, ax=ax, vmin=CLIM[0], vmax=CLIM[1])
        plt.title(f'Heatmap of Preferred Unit Response for Space: {space}')
        plt.xlabel('Linear Distance')
        plt.ylabel('Eigenvalue ID')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.axis('image')
    plt.suptitle(f'Preferred Unit Response for Different Spaces and Eigenvectors \n {exprow.ephysFN} | Pref Channel {prefchan_str} ')
    plt.tight_layout()
    saveallforms(figdir, f"preferred_unit_response_heatmap_py", )
    plt.show()

    # %
    space = "class"
    for space in ["class", "noise"]:
        filtered_df = sgtr_resp_df.query(f"space_name == '{space}'")
        if filtered_df.empty:
            continue
        unique_eig_ids = sorted(filtered_df['eig_id'].unique())
        num_eig_ids = len(unique_eig_ids)
        # Determine grid size
        # Calculate grid dimensions for subplots
        max_cols = 3  # Maximum number of columns
        cols = min(max_cols, num_eig_ids)  # Use fewer columns if fewer eigenvectors
        rows = math.ceil(num_eig_ids / cols)
        # Create figure with appropriate size scaling
        fig_width = cols * 4.5  # Slightly wider to accommodate labels
        fig_height = rows * 3.5 + 0.5  # Extra space for suptitle
        fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), 
                               sharex=True, sharey=True,
                               squeeze=False)  # Always return 2D array of axes
        axs = axs.flatten()
        for i, eig_id in enumerate(unique_eig_ids):
            ax = axs[i]
            subset = filtered_df[filtered_df['eig_id'] == eig_id]
            # Perform ANOVA
            model = ols('pref_unit_resp ~ C(lin_dist)', data=subset).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            # Extract F and p values
            F_value = anova_table.loc['C(lin_dist)', 'F']  # Get F-value
            p_value = anova_table.loc['C(lin_dist)', 'PR(>F)']  # Get p-value
            # Optional: Print results
            print(f" Eig ID: {eig_id} | F-value: {F_value:.4f} | p-value: {p_value:.4e}")
            sns.lineplot(data=subset, x='lin_dist', y='pref_unit_resp', ax=ax, marker='o') # default to plot 95% confidence interval around the mean
            ax.axhline(prefchan_bsl_mean, color='black', linestyle='--', label='Baseline Mean')
            ax.axhline(prefchan_bsl_mean + prefchan_bsl_sem, color='black', linestyle=':', label='Baseline SEM')
            ax.axhline(prefchan_bsl_mean - prefchan_bsl_sem, color='black', linestyle=':')
            ax.set_title(f'Eig ID: {eig_id} | F-val: {F_value:.2f} | p-val: {p_value:.1e}')
            ax.set_xlabel('Linear Distance')
            ax.set_ylabel('Preferred Unit Response')

        # Remove any unused subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.suptitle(f'Preferred Unit Response for Different Eigenvectors {space}\n {exprow.ephysFN} | Pref Channel {prefchan_str} ')
        plt.legend(title='Space Name', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        saveallforms(figdir, f"preferred_unit_{space}_tuning_curve_ANOVA_py", )
        plt.show()

# %%



