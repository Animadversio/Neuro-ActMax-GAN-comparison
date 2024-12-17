import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from easydict import EasyDict as edict
from scipy import stats

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
            stim_info.append({"img_name": name, "space_name": space_name, "eig_id": eig_value, "lin_dist": lin_value, "hessian_img": True, })
        else:
            stim_info.append({"img_name": name, "space_name": None, "eig_id": None, "lin_dist": None, "hessian_img": False, })

    stim_info_df = pd.DataFrame(stim_info)
    return stim_info_df


def organize_unit_info(meta, exprow):
    """Extract and organize unit information from metadata"""
    spikeID = meta.spikeID[0].astype(int)
    channel_id = spikeID # set alias
    unit_id = meta.unitID[0].astype(int)
    char_map = {0:"U", 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H'}
    unit_str = [f"{channel_id}{char_map[unit_id]}" for channel_id, unit_id in zip(channel_id, unit_id)]
    prefchan = exprow.pref_chan
    prefunit = exprow.pref_unit
    prefchan_id_allunits = np.where((channel_id == prefchan))[0]
    prefchan_id = np.where((channel_id == prefchan) & (unit_id == prefunit))[0]
    assert len(prefchan_id) == 1, f"Multiple preferred units found for {exprow.ephysFN} | {prefchan} | {prefunit}"
    prefchan_str = unit_str[prefchan_id.item()]
    return edict({"prefchan_id": prefchan_id, "prefchan_str": prefchan_str})


def calculate_neural_responses(rasters, prefchan_id, resp_wdw=slice(50, 200), bsl_wdw=slice(0, 45)):
    """Calculate neural responses for preferred channel"""
    respmat = rasters[:, resp_wdw, :].mean(axis=1)
    bslmat = rasters[:, bsl_wdw, :].mean(axis=1)
    prefchan_resp_sgtr = respmat[:, prefchan_id] 
    prefchan_bsl_sgtr = bslmat[:, prefchan_id]
    prefchan_bsl_mean = prefchan_bsl_sgtr.mean()
    prefchan_bsl_sem = stats.sem(prefchan_bsl_sgtr)
    return edict({"prefchan_resp_sgtr": prefchan_resp_sgtr, 
            "prefchan_bsl_mean": prefchan_bsl_mean, 
            "prefchan_bsl_sem": prefchan_bsl_sem})

