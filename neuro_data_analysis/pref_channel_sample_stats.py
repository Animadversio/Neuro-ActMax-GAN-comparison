"""Trajectory analysis
Devoted to compare the trajectory of BigGAN vs DeePSim, see how many blocks can BigGAN surpass DeePSim.
"""

import os
import torch
import seaborn as sns
from matplotlib import cm
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, ttest_ind, ttest_1samp, ttest_rel
from core.utils.plot_utils import saveallforms, show_imgrid
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data, get_expstr, extract_evol_activation_array
from neuro_data_analysis.neural_data_utils import parse_meta, area_mapping
from os.path import join
from collections import OrderedDict
from easydict import EasyDict as edict
from neuro_data_analysis.neural_data_lib import extract_all_evol_trajectory, pad_resp_traj
from neuro_data_analysis.neural_data_utils import get_all_masks
outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_activation_cmp"
os.makedirs(outdir, exist_ok=True)
#%%
tabdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables"
# meta_df.to_csv(os.path.join(tabdir, "meta_stats.csv"))
meta_df = pd.read_csv(os.path.join(tabdir, "meta_stats_w_optimizer.csv"), index_col=0)
#%%
Amsk, Bmsk, V1msk, V4msk, ITmsk, \
    length_msk, spc_msk, sucsmsk, \
    bsl_unstable_msk, bsl_stable_msk, validmsk = get_all_masks(meta_df)
#%%
# create a column of expdate with datetime format
meta_df["expdate"] = pd.to_datetime(meta_df["expdate"])
#%%
import datetime
MFA_implant_date = datetime.datetime.fromisoformat("2021-08-01")
# create a column called MFA_id default to be Animal name
meta_df["MFA_id"] = meta_df["Animal"]
# when Animal is Beto and expdate > 2021-08-01, MFA_id is Beto_new
meta_df.loc[(meta_df["Animal"] == "Beto") & (meta_df["expdate"] > MFA_implant_date),\
    "MFA_id"] = "Beto_new"
#%%
# find the unique MFA_id, prefchan pairs, and find the count of each pair
MFA_prefchan_pairs = meta_df[validmsk].groupby(["MFA_id", "prefchan"]).size()
# save to new csv
MFA_prefchan_pairs.to_csv(os.path.join(tabdir, "MFA_prefchan_exp_count.csv"))
#%%
# find occurance of unique value and count in the MFA_prefchan_pairs
MFA_prefchan_pairs.value_counts()
