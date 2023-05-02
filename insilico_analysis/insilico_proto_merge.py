"""Save the merges version of the proto images as montage
and their scores as heatmap
"""
import os
from os.path import join
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from core.utils.plot_utils import saveallforms
from core.utils.montage_utils import crop_from_montage, crop_all_from_montage, make_grid_np
protosumdir = r"F:\insilico_exps\GAN_Evol_cmp\protoimgs"
protooutdir = r"F:\insilico_exps\GAN_Evol_cmp\protoimgs_merge"
Path(protooutdir).mkdir(parents=True, exist_ok=True)
#%%
optim_names = ['CholCMA', 'HessCMA', 'HessCMA500_fc6']  #
# RFsfx = "_RFrsz"
# find common filenames with these optimnames as suffix
#%% Montage full file paths for each optimizer
montage_fp = []
for optim_name in optim_names:
    filepaths = sorted(list(Path(protosumdir).glob(f"*_{optim_name}.jpg")))
    montage_fp.append(filepaths)
#%%
# assert the lists are aligned
for i in trange(len(optim_names)):
    assert len(montage_fp[i]) == len(montage_fp[0])

unit_names = []
for i in trange(len(montage_fp[0])):
    # assert the matched montage filepaths have the same prefix
    montage_fps = [montage_fp[j][i] for j in range(len(optim_names))]
    montage_prefix = [fp.name.split(optim_name)[0] for fp, optim_name in zip(montage_fps, optim_names)]
    assert len(set(montage_prefix)) == 1
    unit_names.append(montage_prefix[0])
#%%
for i in trange(len(montage_fp[0])):
    # get the montage filepaths
    montage_fps = [montage_fp[j][i] for j, _ in enumerate(optim_names)]
    # get the montage prefix
    montage_prefix = [fp.name.split(optim_name)[0] for fp, optim_name in zip(montage_fps, optim_names)]
    # assert the montage prefix is the same acroos the optimizers
    assert len(set(montage_prefix)) == 1
    montage_prefix = montage_prefix[0]
    # load the montage
    montages_all = [plt.imread(fp) for fp in montage_fps]
    # crop the montage
    crops_all = [crop_all_from_montage(montage, imgsize=256, pad=2, autostop=False) for montage in montages_all]
    # merge the montage, only take the first 10 images
    montage_pool = sum([crops[:10] for crops in crops_all], [])
    # Pool the crops into a new montage
    mtg_pool = make_grid_np(montage_pool, nrow=10, pad_value=0, padding=2)
    # save the montage
    plt.imsave(join(protooutdir, f"{montage_prefix}optim_pool.jpg"), mtg_pool)
#%% Collect scores and plot as heatmap


import pickle as pkl
from os.path import join
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%%
# use the agg mode
plt.switch_backend('agg')
# get current backend
# plt.get_backend() # 'module://backend_interagg'
#%%
# load the montage
optim_names = ['CholCMA', 'HessCMA', "HessCMA500_fc6"]#"#'HessCMA500_fc6']  #
score_key = 'score_maxlast'
for unitname in unit_names:
    data = pkl.load(open(join(protosumdir,
                              f"{unitname}evolproto_info.pkl"), "rb"))
    scores = []
    for optim_name in optim_names:
        score_runs = [run[score_key] for run in data[optim_name]]
        if len(score_runs) < 10:
            score_runs += [np.nan] * (10 - len(score_runs))
        elif len(score_runs) > 10:
            score_runs = score_runs[:10]
        scores.append(score_runs)
    scores_arr = np.array(scores)
    print(scores_arr.shape)
    #%%
    plt.figure(figsize=(9, 3))
    sns.heatmap(scores_arr, annot=True, fmt=".1f", cmap="viridis")
    plt.yticks(np.arange(len(optim_names))+0.5, optim_names, rotation=0)
    plt.axis("image")
    plt.tight_layout()
    plt.title(f"{unitname} Proto {score_key}")
    plt.savefig(join(protooutdir, f"{unitname}_score_heatmap.png"))
    plt.show()
    plt.close()
#%%
# crop the montage

# merge the montage

# save the montage



#%% Scratch space for debugging
montage_prefix = [fp.name.split(optim_name)[0] for fp, optim_name in zip(montage_fps, optim_names)]
assert len(set(montage_prefix)) == 1
montage_prefix = montage_prefix[0]
montage_fps = [montage_fp[j][0] for j, _ in enumerate(optim_names)]
montages_all = [plt.imread(fp) for fp in montage_fps]
crops_all = [crop_all_from_montage(montage, imgsize=256, pad=2, autostop=False) for montage in montages_all]
montage_pool = sum([crops[:10] for crops in crops_all], [])
mtg_pool = make_grid_np(montage_pool, nrow=10, pad_value=0, padding=2)
plt.imsave(join(protooutdir, f"{montage_prefix}optim_pool.jpg"), mtg_pool)