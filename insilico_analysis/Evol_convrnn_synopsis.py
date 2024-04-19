# %%
%load_ext autoreload
%autoreload 2

# %%
import sys
sys.path.append("/n/home12/binxuwang/Github/Neuro-ActMax-GAN-comparison")
from core.utils.plot_utils import saveallforms
from insilico_analysis.insilico_analysis_lib import sweep_dir
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from glob import glob
from tqdm import tqdm, trange

# %%
figdir = r"/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Projects/BigGAN_Project/CCN_dynamics_figs"

# %% [markdown]
# ### Convrnn

# %%
# rootdir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/convrnn_Evol"

# %%
!ls {rootdir}

# %%
def parse_npz_files_BigGAN(rootdir, netname, layername):
    layerdir = f"{netname}-{layername}_dyn_BigGAN"
    file_pattern = rf"scores_{layername}_chan(\d*)_T(\d*)_CholCMA-BigGAN_(\d*).npz"
    pattern = re.compile(file_pattern)
    files = sorted(glob(join(rootdir, layerdir, "*.npz")))
    meta = []
    for file in tqdm(files):
        matches = re.findall(pattern, file)
        if matches:
            chan, T, RND = matches[0]
            data = np.load(file)
            scores_dyn = data["scores_dyn"]
            generations = data["generations"]
            best_score = np.max(scores_dyn,axis=0)
            best_avg_score = np.mean(scores_dyn[generations==generations.max(),:],axis=0)
            meta.append({"file": file, "netname":netname, "layer":layername, 
                         "chan": int(chan), "T": int(T), "RND": int(RND), 
                         "best_avg_score": best_avg_score, "best_score": best_score})
    df = pd.DataFrame(meta)
    return df


def parse_npz_files(rootdir, netname, layername):
    layerdir = f"{netname}-{layername}_dyn"
    file_pattern = rf"scores_{layername}_chan(\d*)_T(\d*).npz"
    pattern = re.compile(file_pattern)
    files = sorted(glob(join(rootdir, layerdir, "*.npz")))
    meta = []
    for file in tqdm(files):
        matches = re.findall(pattern, file)
        if matches:
            chan, T = matches[0]
            data = np.load(file)
            scores_dyn = data["scores_dyn"]
            generations = data["generations"]
            best_score = np.max(scores_dyn,axis=0)
            best_avg_score = np.mean(scores_dyn[generations==generations.max(),:],axis=0)
            meta.append({"file": file, "netname":netname, "layer":layername, 
                         "chan": int(chan), "T": int(T), 
                         "best_avg_score": best_avg_score, "best_score": best_score})
    df = pd.DataFrame(meta)
    return df

# %%
def mean_lists(series):
    return np.mean([np.array(item) for item in series], axis=0)#.tolist()

def std_lists(series):
    return (np.std([np.array(item) for item in series], axis=0))#.tolist()

def sem_lists(series):
    N = len(series)
    return (np.std([np.array(item) for item in series], axis=0) / np.sqrt(N))#.tolist()

# %% [markdown]
# #### Mass compute and save

# %%
convrnn_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/convrnn_Evol"


# %%
!ls {convrnn_root}

# %%
convrnn_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/convrnn_Evol"
df_all = []
for layername in ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8", "conv9", "conv10", "imnetds"]:
    df_FC = parse_npz_files(convrnn_root, "rgc_intermediate", layername)
    df_FC["GANname"] = "DeePSim"
    df_BG = parse_npz_files_BigGAN(convrnn_root, "rgc_intermediate", layername)
    df_BG["GANname"] = "BigGAN"
    df_all.append(df_FC)
    df_all.append(df_BG)

# %%
df_convrnn = pd.concat(df_all, axis=0)

# %%
df_convrnn["T"] = df_convrnn["T"].astype(int)

# %%
def extract_Tth_element(row, key='best_avg_score'):
    try:
        return row[key][- (16 - row["T"] + 1)]
    except IndexError:
        return None

# T = 0  # replace with your desired index
# df_convrnn['Tth_element'] = 
df_convrnn["best_avg_score_scalar"] = df_convrnn.apply(extract_Tth_element, axis=1, args=('best_avg_score',)) 
df_convrnn["best_score_scalar"] = df_convrnn.apply(extract_Tth_element, axis=1, args=('best_score',)) 

# %%
df_convrnn.to_pickle(join(figdir, "df_convrnn_BigGAN_score_summary.pkl"))
df_convrnn.to_csv(join(figdir, "df_convrnn_BigGAN_score_summary.csv"))

# %% [markdown]
# ##### Visualize

# %%
df_convrnn.netname.unique()

# %%
# for optimizer in df_convrnn['optimizer'].unique():
figh, axs = plt.subplots(2, 5, figsize=(18, 6.5), sharex=True, sharey=False)
layers = df_convrnn.layer.unique()
for axi, layer in enumerate(layers):
    ax = axs.flatten()[axi]
    df_layer = df_convrnn[(df_convrnn['layer'] == layer)]
    sns.pointplot(data=df_layer, x='T', y='best_avg_score_scalar', hue='GANname', ax=ax, 
                    linestyles='-', palette='magma', hue_order=['DeePSim', 'BigGAN'],
                    legend=True if axi == 9 else False, alpha=0.7)
    # ax.plot(df_layer['unitid'], df_layer['score'], 'o', markersize=2)
    ax.set_title(layer, )
    ax.set_xlabel('Time Step', fontsize=12)
    if axi == 0 or axi == 5:
        ax.set_ylabel('score', )
    else:
        ax.set_ylabel('')
figh.suptitle(f'ConvRNN (rgc_intermediate) Evolutionary Scores', fontsize=16)
saveallforms(figdir, f'ConvRNN_rgc_intermediate_Evol_Scores_time_traj', figh)

# %%
# for optimizer in df_convrnn['optimizer'].unique():
figh, axs = plt.subplots(2, 5, figsize=(18, 6.5), sharex=True, sharey=False)
layers = df_convrnn.layer.unique()
for axi, layer in enumerate(layers):
    ax = axs.flatten()[axi]
    df_layer = df_convrnn[(df_convrnn['layer'] == layer)]
    sns.pointplot(data=df_layer, x='GANname', y='best_avg_score_scalar', hue='T', ax=ax, 
                    palette='viridis', order=['DeePSim', 'BigGAN'],
                    legend=True if axi == 9 else False, alpha=0.7)
    # ax.plot(df_layer['unitid'], df_layer['score'], 'o', markersize=2)
    ax.set_title(layer, )
    ax.set_xlabel('GAN name', fontsize=12)
    if axi == 0 or axi == 5:
        ax.set_ylabel('score', )
    else:
        ax.set_ylabel('')
figh.suptitle(f'ConvRNN (rgc_intermediate) Evolutionary Scores', fontsize=16)
saveallforms(figdir, f'ConvRNN_rgc_intermediate_Evol_Scores_line_cmp', figh)

# %% [markdown]
# ##### Individual layer figure

# %%
convrnn_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/convrnn_Evol"
df_FC = parse_npz_files(convrnn_root, "rgc_intermediate", "conv9")
df_BG = parse_npz_files_BigGAN(convrnn_root, "rgc_intermediate", "conv9")

# %%
FC_mean = df_FC.groupby("T").agg({"best_avg_score": "mean"})
FC_sem = df_FC.groupby("T").agg({"best_avg_score": sem_lists})
BG_mean = df_BG.groupby("T").agg({"best_avg_score": "mean"})
BG_sem = df_BG.groupby("T").agg({"best_avg_score": sem_lists})
Ts = df_FC["T"].unique()
figh, axh = plt.subplots(2,4,figsize=(12,5), sharex=True, ) # sharey=True
axs = axh.flatten()
Tlist = Ts
for axi, iT in enumerate(Ts):
    ax = axs[axi]
    ax.plot(Tlist, FC_mean.loc[iT].best_avg_score, label="DeePSim")
    ax.fill_between(Tlist, FC_mean.loc[iT].best_avg_score-FC_sem.loc[iT].best_avg_score,
                    FC_mean.loc[iT].best_avg_score+FC_sem.loc[iT].best_avg_score, alpha=0.3)
    ax.plot(Tlist, BG_mean.loc[iT].best_avg_score, label="BigGAN")
    ax.fill_between(Tlist, BG_mean.loc[iT].best_avg_score-BG_sem.loc[iT].best_avg_score,
                    BG_mean.loc[iT].best_avg_score+BG_sem.loc[iT].best_avg_score, alpha=0.3)
    # ax.set_xticks()
    ax.axvline(iT, color="k", linestyle="--", alpha=0.4)
    ax.set_title(f"T={iT}")
    if axi == 0:
        ax.legend()
    
plt.legend()

# %%
df_FC = parse_npz_files(convrnn_root, "rgc_intermediate", "conv7")
df_BG = parse_npz_files_BigGAN(convrnn_root, "rgc_intermediate", "conv7")

# %%
FC_mean = df_FC.groupby("T").agg({"best_avg_score": "mean"})
FC_sem = df_FC.groupby("T").agg({"best_avg_score": sem_lists})
BG_mean = df_BG.groupby("T").agg({"best_avg_score": "mean"})
BG_sem = df_BG.groupby("T").agg({"best_avg_score": sem_lists})
Ts = df_FC["T"].unique()
figh, axh = plt.subplots(2,5,figsize=(12,5), sharex=True, ) # sharey=True
axs = axh.flatten()
Tlist = Ts
for axi, iT in enumerate(Ts):
    ax = axs[axi]
    ax.plot(Tlist, FC_mean.loc[iT].best_avg_score, label="DeePSim")
    ax.fill_between(Tlist, FC_mean.loc[iT].best_avg_score-FC_sem.loc[iT].best_avg_score,
                    FC_mean.loc[iT].best_avg_score+FC_sem.loc[iT].best_avg_score, alpha=0.3)
    ax.plot(Tlist, BG_mean.loc[iT].best_avg_score, label="BigGAN")
    ax.fill_between(Tlist, BG_mean.loc[iT].best_avg_score-BG_sem.loc[iT].best_avg_score,
                    BG_mean.loc[iT].best_avg_score+BG_sem.loc[iT].best_avg_score, alpha=0.3)
    # ax.set_xticks()
    ax.axvline(iT, color="k", linestyle="--", alpha=0.4)
    ax.set_title(f"T={iT}")
    ax.legend()    
    
plt.legend()

# %%
iChan = 42
figh, axh = plt.subplots(2,4,figsize=(12,5), sharex=True, ) # sharey=True
axs = axh.flatten()
Ts = df_FC["T"].unique()
Tlist = Ts
for axi, iT in enumerate(Ts):
    ax = axs[axi]
    ax.plot(Tlist, df_FC[(df_FC["chan"]==iChan) & (df_FC["T"]==iT)].best_avg_score.iloc[0], label="DeePSim")
    ax.plot(Tlist, df_BG[(df_BG["chan"]==iChan) & (df_BG["T"]==iT)].best_avg_score.iloc[0], label="BigGAN")
    # ax.set_xticks()
    ax.axvline(iT, color="k", linestyle="--", alpha=0.4)
    ax.set_title(f"T={iT}")
    ax.legend()    
    
plt.legend()

# %%


# %% [markdown]
# #### Mass Compute: Window optimize Results 

# %%
def parse_npz_files_wdw(rootdir, netname, layername):
    layerdir = f"{netname}-{layername}_timewdw"
    file_pattern = rf"scores_{layername}_chan(\d*)_T(\d*)-T(\d*)_CholCMA-(BigGAN|fc6)_(\d*).npz"
    pattern = re.compile(file_pattern)
    files = sorted(glob(join(rootdir, layerdir, "*.npz")))
    meta = []
    for file in tqdm(files):
        matches = re.findall(pattern, file)
        if matches:
            chan, Tbeg, Tend, GANname, RND = matches[0]
            data = np.load(file)
            scores_dyn = data["scores_dyn"]
            generations = data["generations"]
            best_score = np.max(scores_dyn,axis=0)
            best_avg_score = np.mean(scores_dyn[generations==generations.max(),:],axis=0)
            meta.append({"file": file, "netname":netname, "layer":layername, 
                         "chan": int(chan), 
                         "Tbeg": int(Tbeg), "Tend": int(Tend), "GANname": GANname, "RND": int(RND),
                         "best_avg_score": best_avg_score, "best_score": best_score})
    df = pd.DataFrame(meta)
    return df

# %%
df_wdw_all = []
for layername in ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8", "conv9", "conv10", "imnetds"]:
    df_FC = parse_npz_files_wdw(convrnn_root, "rgc_intermediate", layername)
    df_wdw_all.append(df_FC)

# %%
df_wdw_syn = pd.concat(df_wdw_all, axis=0)

# %%
df_wdw_syn

# %%
df_wdw_syn.Tbeg.unique()

# %%
def extract_element_mean(row, key='best_avg_score'):
    try:
        return row[key].mean()
    except IndexError:
        return None

# T = 0  # replace with your desired index
# df_convrnn['Tth_element'] = 
df_wdw_syn["best_avg_score_scalar"] = df_wdw_syn.apply(extract_element_mean, axis=1, args=('best_avg_score',)) 
df_wdw_syn["best_score_scalar"] = df_wdw_syn.apply(extract_element_mean, axis=1, args=('best_score',)) 

# %%
df_wdw_syn.to_pickle(join(figdir, "df_convrnn_BigGAN_wdw_Evol_score_summary.pkl"))
df_wdw_syn.to_csv(join(figdir, "df_convrnn_BigGAN_wdw_Evol_score_summary.csv"))

# %% [markdown]
# ##### Visualize

# %%
rsp_mean = df_layer.groupby("GANname").agg({"best_avg_score": mean_lists})
rsp_sem = df_layer.groupby("GANname").agg({"best_avg_score": sem_lists})

# %%
rsp_mean.loc["fc6","best_avg_score"]

# %%
palette[2]

# %%
import seaborn as sns
# Get the magma palette
palette = sns.color_palette("magma")
# Get the color for 0 and 1
color_0 = palette[1]
color_1 = palette[4]

# %%
# for optimizer in df_convrnn['optimizer'].unique():
layers = df_wdw_syn.layer.unique()
figh, axs = plt.subplots(2, 5, figsize=(18, 6.5), sharex=True, sharey=False)
for axi, layer in enumerate(layers):
    ax = axs.flatten()[axi]
    df_layer = df_wdw_syn[(df_wdw_syn['layer'] == layer)]
    rsp_mean = df_layer.groupby("GANname").agg({"best_avg_score": mean_lists})
    rsp_sem = df_layer.groupby("GANname").agg({"best_avg_score": sem_lists})
    Tlist = range(df_layer["Tbeg"].min(), df_layer["Tend"].max()+1)
    plt.sca(ax)
    plt.plot(Tlist, rsp_mean.loc["fc6"].best_avg_score, label="DeePSim", color=color_0)
    plt.fill_between(Tlist, rsp_mean.loc["fc6"].best_avg_score-rsp_sem.loc["fc6"].best_avg_score,
                    rsp_mean.loc["fc6"].best_avg_score+rsp_sem.loc["fc6"].best_avg_score, alpha=0.3, color=color_0)
    plt.plot(Tlist, rsp_mean.loc["BigGAN"].best_avg_score, label="BigGAN", color=color_1)
    plt.fill_between(Tlist, rsp_mean.loc["BigGAN"].best_avg_score-rsp_sem.loc["BigGAN"].best_avg_score,
                    rsp_mean.loc["BigGAN"].best_avg_score+rsp_sem.loc["BigGAN"].best_avg_score, alpha=0.3, color=color_1)
    if axi == 9:
        plt.legend()
    # sns.pointplot(data=df_layer, x='Tbeg', y='best_avg_score_scalar', hue='GANname', ax=ax, 
    #                 linestyles='-', palette='magma', #hue_order=['DeePSim', 'fc6'],
    #                 legend=True if axi == 9 else False, alpha=0.7)
    # ax.plot(df_layer['unitid'], df_layer['score'], 'o', markersize=2)
    ax.set_title(layer, )
    ax.set_xlabel('Time Step', fontsize=12)
    if axi == 0 or axi == 5:
        ax.set_ylabel('score', )
    else:
        ax.set_ylabel('')
figh.suptitle(f'ConvRNN (rgc_intermediate) Evolutionary Scores (time window objective)', fontsize=16)
saveallforms(figdir, f'ConvRNN_rgc_intermediate_Window_Evol_Scores_time_traj', figh)

# %%
# for optimizer in df_convrnn['optimizer'].unique():
layers = df_wdw_syn.layer.unique()
figh, axs = plt.subplots(2, 5, figsize=(18, 6.5), sharex=True, sharey=False)
for axi, layer in enumerate(layers):
    ax = axs.flatten()[axi]
    df_layer = df_wdw_syn[(df_wdw_syn['layer'] == layer)]
    rsp_mean = df_layer.groupby("GANname").agg({"best_avg_score": mean_lists})
    rsp_sem = df_layer.groupby("GANname").agg({"best_avg_score": sem_lists})
    Tlist = range(df_layer["Tbeg"].min(), df_layer["Tend"].max()+1)
    plt.sca(ax)
    plt.errorbar(Tlist, rsp_mean.loc["fc6"].best_avg_score, yerr=rsp_sem.loc["fc6"].best_avg_score, label="DeePSim", color=color_0)
    plt.errorbar(Tlist, rsp_mean.loc["BigGAN"].best_avg_score, yerr=rsp_sem.loc["BigGAN"].best_avg_score, label="BigGAN", color=color_1)
    # plt.plot(Tlist, rsp_mean.loc["fc6"].best_avg_score, label="DeePSim", color=color_0)
    # plt.fill_between(Tlist, rsp_mean.loc["fc6"].best_avg_score-rsp_sem.loc["fc6"].best_avg_score,
    #                 rsp_mean.loc["fc6"].best_avg_score+rsp_sem.loc["fc6"].best_avg_score, alpha=0.3, color=color_0)
    # plt.plot(Tlist, rsp_mean.loc["BigGAN"].best_avg_score, label="BigGAN", color=color_1)
    # plt.fill_between(Tlist, rsp_mean.loc["BigGAN"].best_avg_score-rsp_sem.loc["BigGAN"].best_avg_score,
    #                 rsp_mean.loc["BigGAN"].best_avg_score+rsp_sem.loc["BigGAN"].best_avg_score, alpha=0.3, color=color_1)
    if axi == 9:
        plt.legend()
    # sns.pointplot(data=df_layer, x='Tbeg', y='best_avg_score_scalar', hue='GANname', ax=ax, 
    #                 linestyles='-', palette='magma', #hue_order=['DeePSim', 'fc6'],
    #                 legend=True if axi == 9 else False, alpha=0.7)
    # ax.plot(df_layer['unitid'], df_layer['score'], 'o', markersize=2)
    ax.set_title(layer, )
    ax.set_xlabel('Time Step', fontsize=12)
    if axi == 0 or axi == 5:
        ax.set_ylabel('score', )
    else:
        ax.set_ylabel('')
figh.suptitle(f'ConvRNN (rgc_intermediate) Evolutionary Scores (time window objective)', fontsize=16)
saveallforms(figdir, f'ConvRNN_rgc_intermediate_Window_Evol_Scores_time_traj_err', figh)

# %% [markdown]
# ##### Individual Evol Visualize

# %%
df_wdw = parse_npz_files_wdw(convrnn_root, "rgc_intermediate", "conv9")

# %%
rsp_mean = df_wdw.groupby(["Tbeg", "Tend", "GANname"]).agg({"best_avg_score": "mean"})
rsp_sem = df_wdw.groupby(["Tbeg", "Tend", "GANname"]).agg({"best_avg_score": sem_lists})

# %%
rsp_mean.loc[9,16,"BigGAN"].best_avg_score

# %%
df_wdw

# %%
unit_rsp_mean

# %%
unit_rsp_mean = df_wdw.groupby(["GANname", "chan"]).agg({"best_avg_score": "mean"})
unit_rsp_sem = df_wdw.groupby(["GANname", "chan"]).agg({"best_avg_score": sem_lists})

figh, axs = plt.subplots(2, 5, figsize=(12,5), sharex=True, sharey=False)
for chan in range(10):
    ax = axs.flatten()[chan]
    ax.plot(range(9,17), unit_rsp_mean.loc["fc6",chan].best_avg_score, label="fc6")
    ax.fill_between(range(9,17), unit_rsp_mean.loc["fc6",chan].best_avg_score-unit_rsp_sem.loc["fc6",chan].best_avg_score,
                    unit_rsp_mean.loc["fc6",chan].best_avg_score+unit_rsp_sem.loc["fc6",chan].best_avg_score, alpha=0.3)
    ax.plot(range(9,17), unit_rsp_mean.loc["BigGAN",chan].best_avg_score, label="BigGAN")
    ax.fill_between(range(9,17), unit_rsp_mean.loc["BigGAN",chan].best_avg_score-unit_rsp_sem.loc["BigGAN",chan].best_avg_score,
                    unit_rsp_mean.loc["BigGAN",chan].best_avg_score+unit_rsp_sem.loc["BigGAN",chan].best_avg_score, alpha=0.3)
    ax.set_title(f"chan{chan}")
# plt.plot(range(9,17), rsp_mean.loc[9,16,"fc6"].best_avg_score, label="fc6")
# plt.fill_between(range(9,17), rsp_mean.loc[9,16,"fc6"].best_avg_score-rsp_sem.loc[9,16,"fc6"].best_avg_score,
#                     rsp_mean.loc[9,16,"fc6"].best_avg_score+rsp_sem.loc[9,16,"fc6"].best_avg_score, alpha=0.3)
# plt.plot(range(9,17), rsp_mean.loc[9,16,"BigGAN"].best_avg_score, label="BigGAN")
# plt.fill_between(range(9,17), rsp_mean.loc[9,16,"BigGAN"].best_avg_score-rsp_sem.loc[9,16,"BigGAN"].best_avg_score,
#                     rsp_mean.loc[9,16,"BigGAN"].best_avg_score+rsp_sem.loc[9,16,"BigGAN"].best_avg_score, alpha=0.3)
plt.legend()


# %%
unit_rsp_mean = df_wdw.groupby(["Tbeg", "Tend", "GANname", "chan"]).agg({"best_avg_score": "mean"})
unit_rsp_sem = df_wdw.groupby(["Tbeg", "Tend", "GANname", "chan"]).agg({"best_avg_score": sem_lists})
# Tlist = range(df_wdw["Tbeg"].unique(), df_wdw["Tend"].unique()+1)
plt.figure(figsize=(12,5))
plt.plot(range(9,17), rsp_mean.loc[9,16,"fc6"].best_avg_score, label="fc6")
plt.fill_between(range(9,17), rsp_mean.loc[9,16,"fc6"].best_avg_score-rsp_sem.loc[9,16,"fc6"].best_avg_score,
                    rsp_mean.loc[9,16,"fc6"].best_avg_score+rsp_sem.loc[9,16,"fc6"].best_avg_score, alpha=0.3)
plt.plot(range(9,17), rsp_mean.loc[9,16,"BigGAN"].best_avg_score, label="BigGAN")
plt.fill_between(range(9,17), rsp_mean.loc[9,16,"BigGAN"].best_avg_score-rsp_sem.loc[9,16,"BigGAN"].best_avg_score,
                    rsp_mean.loc[9,16,"BigGAN"].best_avg_score+rsp_sem.loc[9,16,"BigGAN"].best_avg_score, alpha=0.3)
plt.legend()

# %%
rsp_mean = df_wdw.groupby(["Tbeg", "Tend", "GANname"]).agg({"best_avg_score": "mean"})
rsp_sem = df_wdw.groupby(["Tbeg", "Tend", "GANname"]).agg({"best_avg_score": sem_lists})
# Tlist = range(df_wdw["Tbeg"].unique(), df_wdw["Tend"].unique()+1)
plt.figure(figsize=(5,5))
plt.plot(range(9,17), rsp_mean.loc[9,16,"fc6"].best_avg_score, label="fc6")
plt.fill_between(range(9,17), rsp_mean.loc[9,16,"fc6"].best_avg_score-rsp_sem.loc[9,16,"fc6"].best_avg_score,
                    rsp_mean.loc[9,16,"fc6"].best_avg_score+rsp_sem.loc[9,16,"fc6"].best_avg_score, alpha=0.3)
plt.plot(range(9,17), rsp_mean.loc[9,16,"BigGAN"].best_avg_score, label="BigGAN")
plt.fill_between(range(9,17), rsp_mean.loc[9,16,"BigGAN"].best_avg_score-rsp_sem.loc[9,16,"BigGAN"].best_avg_score,
                    rsp_mean.loc[9,16,"BigGAN"].best_avg_score+rsp_sem.loc[9,16,"BigGAN"].best_avg_score, alpha=0.3)
plt.legend()

# %% [markdown]
# #### scratch zoen
import re
import numpy as np
from os.path import join
from glob import glob

rootdir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/convrnn_Evol"
netname = "rgc_intermediate"
layername = "conv8"
layerdir = f"{netname}-{layername}_dyn"
# glob all npz files 
file_pattern = rf"scores_{layername}_chan(\d*)_T(\d*).npz"
pattern = re.compile(file_pattern)
files = sorted(glob(join(rootdir, layerdir, "*.npz")))
# parse all npz files
meta = []
for file in files:
    matches = re.findall(pattern, file)
    if matches:
        chan, T = matches[0]
        data = np.load(file)
        scores_dyn = data["scores_dyn"]
        generations = data["generations"]
        best_score = np.max(scores_dyn, axis=0)
        best_avg_score = np.mean(scores_dyn[generations==generations.max(),:],axis=0)
        meta.append({"file": file, "chan": int(chan), "T": int(T), 
                     "best_avg_score": best_avg_score, "best_score": best_score})
df = pd.DataFrame(meta)
df

# %%

rootdir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/convrnn_Evol"
netname = "rgc_intermediate"
layername = "conv8"
layerdir = f"{netname}-{layername}_dyn_BigGAN"
# glob all npz files 
file_pattern = rf"scores_{layername}_chan(\d*)_T(\d*)_CholCMA-BigGAN_(\d*).npz"
pattern = re.compile(file_pattern)
files = sorted(glob(join(rootdir, layerdir, "*.npz")))
# parse all npz files
meta = []
for file in files:
    matches = re.findall(pattern, file)
    if matches:
        chan, T, RND = matches[0]
        data = np.load(file)
        scores_dyn = data["scores_dyn"]
        generations = data["generations"]
        best_score = np.max(scores_dyn,axis=0)
        best_avg_score = np.mean(scores_dyn[generations==generations.max(),:],axis=0)
        meta.append({"file": file, "chan": int(chan), "T": int(T), "RND": int(RND), 
                     "best_avg_score": best_avg_score, "best_score": best_score})
df = pd.DataFrame(meta)



# %%  CorNet-recurrent evolution
# %%
import shutil
import os
import re
import glob
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from os.path import join
from easydict import EasyDict as edict
import numpy as np
import pandas as pd
def sweep_dir(rootdir, unit_pattern, save_pattern):
    rootpath = Path(rootdir)
    unitdirs = sorted(list(rootpath.glob(unit_pattern)))
    df_col = []
    for unitdir in tqdm(unitdirs):
        if ".SelectAdaptivePool2dglobal_pool" in unitdir.name:
            # this layername has _ in it so it will fail the regex below
            parts = unitdir.name.split("_"+".SelectAdaptivePool2dglobal_pool"+"_")
            netname = parts[0]
            layer = ".SelectAdaptivePool2dglobal_pool"
            RFresize = False
            unitstr = parts[1]
        else:
            unit_pat = re.compile("([^.]*)_([^_]*)_([\d_]*)(_RFrsz)?$")
            unit_match = unit_pat.findall(unitdir.name)
            assert len(unit_match) == 1
            unit_match = unit_match[0]
            netname = unit_match[0]
            layer = unit_match[1]
            RFresize = True if unit_match[3] == "_RFrsz" else False
            unitstr = unit_match[2]

        if "_" in unitstr:
            unit = unitstr.split("_")
            unitid = int(unit[0])
            x = int(unit[1])
            y = int(unit[2])
        else:
            unitid = int(unitstr)
            x = None
            y = None
        # print(unit_match)
        print(unitdir.name, "=", netname, layer, unitid, x, y, RFresize)
        unitdict = edict(netname=netname, layer=layer, unitid=unitid, x=x, y=y, RFresize=RFresize)
        savefiles = sorted(list(unitdir.glob("scores*.npz")))
        savefn_pat = re.compile("scores(.*)_(\d\d\d\d\d).npz$")
        for savefn in savefiles:
            savefn_pat_match = savefn_pat.findall(savefn.name)
            assert len(savefn_pat_match) == 1
            savefn_pat_match = savefn_pat_match[0]
            optimmethod = savefn_pat_match[0]
            RND = int(savefn_pat_match[-1])
            if optimmethod.endswith("_fc6"):
                GANname = "fc6"
            else:
                GANname = "BigGAN"
            # print(optimmethod, RND, GANname)
            optimdict = edict(optimmethod=optimmethod, RND=RND, GANname=GANname)
            data = np.load(savefn)
            scores_all = data["scores_all"]
            generations = data["generations"]
            endscores = scores_all[generations == generations.max()].mean()
            maxscores = scores_all.max(axis=0)
            maxstep = np.argmax(scores_all, axis=0)
            df_col.append({**unitdict, **optimdict, **dict(score=endscores, maxscore=maxscores, maxstep=maxstep)})
            # raise  Exception

        # break
    df_evol = pd.DataFrame(df_col)
    # change datatype of columns GANname, layer, optimmethod, netname as string
    # df_evol = df_evol.astype({"GANname": str, "layer": str, "optimmethod": str, "netname": str,
    #                           "score": float, "maxscore": float, "maxstep": int, "RFresize": bool})
    return df_evol

# %%
CorNet_dir = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/GAN_Evol_cmp"

# %%
df_cornet = sweep_dir(CorNet_dir, "corner-s_*_*_*_*", "scores*.npz")
df_cornet

# %%
df_cornet.layer.unique()

# %%
df_cornet['time'] = df_cornet['optimmethod'].str.extract(r'_T(\d+)_', expand=False)
df_cornet['time'] = df_cornet['time'].fillna(0) # Replace NaN values with 0
df_cornet['time'] = df_cornet['time'].astype(int) # Convert the time column to integer type
df_cornet['optimizer'] = df_cornet['optimmethod'].str.extract(r'_T\d+_(.*CMA)', expand=False)

# %%
df_cornet.to_pickle(join(figdir, "df_cornet_BigGAN_score_summary.pkl"))
df_cornet.to_csv(join(figdir, "df_cornet_BigGAN_score_summary.csv"))

# %%
df_cornet['optimizer'].unique()

# %%
df_cornet.time.unique()

# %% [markdown]
# #### Visualize

# %%
for optimizer in df_cornet['optimizer'].unique():
    figh, axs = plt.subplots(1, 4, figsize=(15, 4), sharex=False, sharey=False)
    layers = ['V1.output', 'V2.output', 'V4.output', 'IT.output',]
    for axi, layer in enumerate(layers):
        ax = axs.flatten()[layers.index(layer)]
        df_layer = df_cornet[(df_cornet['layer'] == layer) & (df_cornet['optimizer'] == optimizer) ]
        sns.pointplot(data=df_layer, x='time', y='score', hue='GANname', ax=ax, 
                      linestyles='-', palette='magma', hue_order=['fc6', 'BigGAN'])
        # ax.plot(df_layer['unitid'], df_layer['score'], 'o', markersize=2)
        ax.set_title(layer, )
        ax.set_xlabel('Time Step', fontsize=12)
        if axi == 0:
            ax.set_ylabel('score', )
        else:
            ax.set_ylabel('')
    figh.suptitle(f'CorNet-S Evolutionary Scores {optimizer}', fontsize=16)
    saveallforms(figdir, f'CorNet-S_Evol_Scores_{optimizer}_time_traj', figh)

# %%
for optimizer in df_cornet['optimizer'].unique():
    figh, axs = plt.subplots(1, 4, figsize=(15, 4), sharex=True, sharey=False)
    layers = ['V1.output', 'V2.output', 'V4.output', 'IT.output',]
    for axi, layer in enumerate(layers):
        ax = axs.flatten()[layers.index(layer)]
        df_layer = df_cornet[(df_cornet['layer'] == layer) & (df_cornet['optimizer'] == optimizer) ]
        sns.pointplot(data=df_layer, x='GANname', y='score', hue='time', ax=ax, 
                      linestyles='-', palette='viridis', order=['fc6', 'BigGAN'])
        # ax.plot(df_layer['unitid'], df_layer['score'], 'o', markersize=2)
        ax.set_title(layer, )
        ax.set_xlabel('GAN', fontsize=12)
        if axi == 0:
            ax.set_ylabel('score', )
        else:
            ax.set_ylabel('')
    figh.suptitle(f'CorNet-S Evolutionary Scores {optimizer}', fontsize=16)
    saveallforms(figdir, f'CorNet-S_Evol_Scores_{optimizer}_line_cmp', figh)

# %%
figh, axs = plt.subplots(1, 4, figsize=(15, 3.5), sharex=True, sharey=False)
layers = ['V1.output', 'V2.output', 'V4.output', 'IT.output',]
for layer in layers:
    ax = axs.flatten()[layers.index(layer)]
    df_layer = df_cornet[df_cornet['layer'] == layer]
    sns.pointplot(data=df_layer, x='unitid', y='score', hue='GANname', ax=ax, linestyles='none')
    # ax.plot(df_layer['unitid'], df_layer['score'], 'o', markersize=2)
    ax.set_title(layer)
    ax.set_xlabel('unitid')
    ax.set_ylabel('score')

# %%
df_cornet[df_cornet['optimmethod'].str.contains('HessCMA')].groupby(["layer","GANname", "optimmethod"]).agg({"score": ["mean","std"]})

# %%
df_cornet[df_cornet['optimmethod'].str.contains('CholCMA')].groupby(["layer","GANname", "optimmethod"]).agg({"score": ["mean","sem"]})

# %%
rootdir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/CorNet-recurrent-evol"

# %% [markdown]
# ### LRM zone 

# %%
LRMroot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/Evol_lrm_GAN_cmp"
LRMroot = "/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Projects/Evol_lrm_GAN_cmp/" #synopsis/alexnet_lrm3_df_img.pkl
df_lrm3 = pd.read_pickle(join(LRMroot, "synopsis", "alexnet_lrm3_df_img.pkl"))

# %%
df_lrm3.loc[1, "mean_score"].shape

# %%
df_lrm3 = df_lrm3.rename(columns={'layerkey': 'layer'})

# %%
df_lrm3.methodlab.unique()

# %%
figh, axs = plt.subplots(2, 4, figsize=(15, 6.), sharex=False, sharey=False)
layers = df_lrm3.layer.unique()
for axi, layer in enumerate(layers):
    ax = axs.flatten()[axi]
    df_layer = df_lrm3[(df_lrm3['layer'] == layer)]
    sns.pointplot(data=df_layer, x='iT', y='finalscore', hue='GANname', ax=ax, 
                    linestyles='-', palette='magma', hue_order=['fc6', 'BigGAN'],
                    legend=True if axi == 7 else False, alpha=0.7)
    # ax.plot(df_layer['unitid'], df_layer['score'], 'o', markersize=2)
    ax.set_title(layer, )
    if axi > 3:
        ax.set_xlabel('Time Step', fontsize=12)
    else:
        ax.set_xlabel('')
    if axi == 0:
        ax.set_ylabel('score', )
    else:
        ax.set_ylabel('')
figh.suptitle(f'AlexNet-LRM3 Evolutionary Scores {"CholCMA"}', fontsize=16)
saveallforms(figdir, f'AlexNet-LRM3_Evol_Scores_{"CholCMA"}_time_traj', figh)

# %%
df_lrm3.layer.unique()

# %%
figh, axs = plt.subplots(2, 4, figsize=(15, 6.), sharex=False, sharey=False)
layers = df_lrm3.layer.unique()
for axi, layer in enumerate(layers):
    ax = axs.flatten()[axi]
    df_layer = df_lrm3[(df_lrm3['layer'] == layer)]
    sns.pointplot(data=df_layer, x='GANname', y='finalscore', hue='iT', ax=ax, 
                    linestyles='-', palette='viridis', order=['fc6', 'BigGAN'],
                    legend=True if axi == 6 else False, alpha=0.7)
    # ax.plot(df_layer['unitid'], df_layer['score'], 'o', markersize=2)
    ax.set_title(layer, )
    if axi > 3:
        ax.set_xlabel('Time Step', fontsize=12)
    else:
        ax.set_xlabel('')
    if axi == 0:
        ax.set_ylabel('score', )
    else:
        ax.set_ylabel('')
figh.suptitle(f'AlexNet-LRM3 Evolutionary Scores {"CholCMA"}', fontsize=16)
saveallforms(figdir, f'AlexNet-LRM3_Evol_Scores_{"CholCMA"}_line_cmp', figh)

# %%
df_lrm3

# %%



