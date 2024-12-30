import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from easydict import EasyDict as edict
from scipy import stats
from PIL import Image

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
    stim_info = []
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
    if stim_info_df.hessian_img.sum() == 0:
        print(f"Warning: No hessian images found for {image_names[:10]}..., try the older pattern")
        re_pattern = r'(noise|class)_eig(\d+)_exp([+-]?\d+\.\d+)_lin([+-]?\d+\.\d+)'
        stim_info = []
        for name in image_names: 
            match = re.match(re_pattern, name)
            if match:
                space_name = match.groups()[0]
                eig_value = int(match.groups()[1])
                exp_value = float(match.groups()[2])
                lin_value = float(match.groups()[3])
                stim_info.append({"img_name": name, "space_name": space_name, "eig_id": eig_value, "lin_dist": lin_value, "exp_value": exp_value, "hessian_img": True, })
            else:
                stim_info.append({"img_name": name, "space_name": None, "eig_id": None, "lin_dist": None, "exp_value": None, "hessian_img": False, })
        stim_info_df = pd.DataFrame(stim_info)
    return stim_info_df



import os
from typing import List, Optional
import numpy as np
def parse_unit_id_from_spikeID(
    spikeID: List[int],
    active_chan: Optional[List[bool]] = None,
) -> np.ndarray:
    """
    Generate unique unit IDs based on spike channel IDs.

    Parameters:
    - spikeID (List[int] or np.ndarray): Array of spike channel IDs.
    - active_chan (List[bool] or np.ndarray, optional): Array of active channels. Defaults to all channels.

    Returns:
    - unit_id (np.ndarray): Generated list of unit labels.
    """
    # Ensure spikeID is a NumPy array for efficient processing
    spikeID = np.array(spikeID)
    unit_id = np.zeros(len(spikeID), dtype=int)
    if active_chan is None:
        active_chan = np.ones(len(spikeID), dtype=bool)
    # Dictionary to keep track of the occurrence of each channel
    channel_counts = {}
    for i in range(len(spikeID)):
        cur_chan = spikeID[i]
        if ~ active_chan[i]:
            unit_id[i] = 0
        else:
            if cur_chan not in channel_counts:
                channel_counts[cur_chan] = 1
            else:
                channel_counts[cur_chan] += 1
            unit_id[i] = channel_counts[cur_chan]
    
    return unit_id


def maybe_add_unit_id_to_meta(meta, rasters, INACTIVE_THRESHOLD=1.25):
    """For older experimental meta, unit_id is not in the meta file. 
    We need to parse it from the spikeID and add it to the meta file.
    """
    if "unitID" in meta:
        return meta
    chan_mean_rate = rasters.mean(axis=(0,1))  # average across 200 ms and stimuli
    active_chan = chan_mean_rate > INACTIVE_THRESHOLD
    spikeID = meta["spikeID"][0].astype(int)
    if np.sum(~active_chan) > 0:
        # print the number of inactive channels
        print(f"Exist inactive channels: {spikeID[~active_chan]}\n firing rate {chan_mean_rate[~active_chan]}")
    unit_id = parse_unit_id_from_spikeID(spikeID, active_chan)
    meta["unitID"] = unit_id
    return meta


def organize_unit_info(meta, exprow):
    """Extract and organize unit information from metadata"""
    spikeID = np.squeeze(meta.spikeID).astype(int)
    channel_id = spikeID # set alias
    unit_id = np.squeeze(meta.unitID).astype(int)
    char_map = {0:"U", 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H'}
    unit_str = [f"{channel_id}{char_map[unit_id]}" for channel_id, unit_id in zip(channel_id, unit_id)]
    prefchan = exprow.pref_chan
    prefunit = exprow.pref_unit
    prefchan_id_allunits = np.where((channel_id == prefchan))[0]
    prefchan_id = np.where((channel_id == prefchan) & (unit_id == prefunit))[0]
    assert len(prefchan_id) == 1, f"Multiple preferred units found for {exprow.ephysFN} | {prefchan} | {prefunit}"
    prefchan_str = unit_str[prefchan_id.item()]
    return edict({"prefchan_id": prefchan_id, "prefchan_str": prefchan_str, 
                  "prefchan_id_allunits": prefchan_id_allunits,
                  "prefchan": prefchan, "prefunit": prefunit, 
                  "channel_id": channel_id, "unit_id": unit_id, "unit_str": unit_str, 
                  "spikeID": spikeID, 
                  })


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


def load_space_images(space, stim_info_df, uniq_img_fps):
    """
    Loads images for a given space and organizes them into an array.

    Parameters:
        space (str): The space name to filter the stimuli (e.g., "noise").
        stim_info_df (pd.DataFrame): DataFrame containing stimulus information.
        uniq_img_fps (dict): Dictionary mapping image names to their full file paths.

    Returns:
        tuple:
            imgname_table (pd.DataFrame or None): Pivot table of image names if available, else None.
            image_array (np.ndarray or None): Array of loaded images if available, else None.
    """
    space_stim_df = stim_info_df.query(f"space_name == '{space}'")
    if space_stim_df.empty:
        print(f"Warning: No {space} space data found")

    pivot_table = space_stim_df.pivot(index='eig_id', columns='lin_dist', values='img_name')
    print(pivot_table)
    image_array = np.empty(pivot_table.shape, dtype=object)
    # For each entry in the pivot table, get the image path and load the image into a new array
    for ri, eig_id in enumerate(pivot_table.index):
        for ci, lin_dist in enumerate(pivot_table.columns):
            img_name = pivot_table.loc[eig_id, lin_dist]
            img_path = uniq_img_fps.get(img_name)
            if img_path:
                img = Image.open(img_path)
                image_array[ri, ci] = img
            else:
                # print(f"Image path for {img_name} not found.")
                image_array[ri, ci] = None
    print(image_array.shape)
    return pivot_table, image_array


