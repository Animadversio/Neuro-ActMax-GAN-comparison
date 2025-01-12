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
    tuning_stats_df["prefchan_str"] = prefchan_str
    tuning_stats_df["prefchan_bsl_mean"] = prefchan_bsl_mean.item()
    tuning_stats_df["prefchan_bsl_sem"] = prefchan_bsl_sem.item()
    tuning_stats_df.to_csv(join(figdir, f"tuning_curves_stats_df.csv"), index=False)
    tuning_stats_synopsis.append(tuning_stats_df)
#%%
syndir = join(figroot, "synopsis")
os.makedirs(syndir, exist_ok=True)
tuning_stats_synopsis_df = pd.concat(tuning_stats_synopsis, ignore_index=True)
tuning_stats_synopsis_df.to_csv(join(syndir, f"ABCD_tuning_stats_synopsis.csv"), index=False)
tuning_stats_synopsis_df.to_pickle(join(syndir, f"ABCD_tuning_stats_synopsis.pkl"))
