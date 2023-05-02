""" to extract the protoimages from the unit directory of Evolution experiments.
Create a montaged image for each optimization method and the score / stats dict for each unit, across methods.
"""
import shutil
import os
import re
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from os.path import join
from easydict import EasyDict as edict
from core.utils.montage_utils import crop_from_montage, crop_all_from_montage
from core.utils.montage_utils import make_grid, make_grid_np, make_grid_T
from collections import defaultdict
import pickle as pkl
#%%
rootdir = r"F:\insilico_exps\GAN_gradEvol_cmp"
rootpath = Path(rootdir)
datalist = glob.glob(join(rootdir, "*", "*.pt"))
figdir = join(rootdir, "protoimgs")
os.makedirs(figdir, exist_ok=True)
#%% Collect gradient based Evolution data
optimnames = ["Adam001Hess", "Adam001", "Adam01Hess_fc6", "Adam01_fc6"]
# datalist = glob.glob(join(rootdir, "*", "imglastgen*_\d\d\d\d\d.jpg"))
unitdirs = list(rootpath.glob("res*"))
df_col = []
for unitdir in tqdm(unitdirs):
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
    print(unit_match, "=", netname, layer, unitid, x, y, RFresize)
    unitdict = edict(netname=netname, layer=layer, unitid=unitid, x=x, y=y, RFresize=RFresize)
    proto_col = defaultdict(list)
    info_col = defaultdict(list)
    score_col = defaultdict(list)
    imgsize = 227 if RFresize else 256
    mtgfiles_pat = re.compile("imglastgen(.*)_(\d\d\d\d\d).jpg$")
    for optimname in optimnames:
        mtgfiles = list(unitdir.glob(f"imglastgen{optimname}*.jpg"))
        if len(mtgfiles) == 0:
            continue
        for mtgfile in mtgfiles:
            mtgfile_match = mtgfiles_pat.findall(mtgfile.name)
            assert len(mtgfile_match) == 1
            mtgfile_match = mtgfile_match[0]
            cur_optimname = mtgfile_match[0]
            if cur_optimname != optimname:
                continue
            RND = int(mtgfile_match[-1])
            optimdict = edict(optimmethod=optimname, RND=RND)
            imgs = crop_all_from_montage(plt.imread(mtgfile), totalnum=None, imgsize=imgsize)
            data = torch.load(unitdir / f"optimdata_{optimname}_{RND:05d}.pt")
            score_traj = data["score_traj"]
            proto_col[optimname].extend(imgs)
            info_col[optimname].extend([(RND, batchi) for batchi in range(len(imgs))])
            score_col[optimname].extend(score_traj[-1, :].tolist())
        mtg = make_grid_np(proto_col[optimname], nrow=10, padding=2, )
        plt.imsave(join(figdir, f"{unitdir.name}_{optimname}.jpg"), mtg)
    pkl.dump(info_col, open(join(figdir, f"{unitdir.name}_proto_info.pkl"), "wb"))
    pkl.dump(score_col, open(join(figdir, f"{unitdir.name}_proto_scores.pkl"), "wb"))
    # savefn = f"imglastgen{optimname}_{RND}.jpg"
    # savepath = unitdir / savefn
    # plt.imsave(savepath, crop)
    # for optim in optimnames:
    #     print(len(proto_col[optim]))
    # raise Exception("stop")
#%%
# for optim in optimnames:
#     print(len(proto_col[optim]))

#%% Evolution experimental data
rootdir = r"F:\insilico_exps\GAN_Evol_cmp"
rootdir = r"/n/scratch3/users/b/biw905/GAN_Evol_cmp"
rootpath = Path(rootdir)
# datalist = glob.glob(join(rootdir, "*", "*.pt"))
figdir = join(rootdir, "protoimgs")
os.makedirs(figdir, exist_ok=True)
#%%
# check path and load data
evoloptimnames = ["CholCMA", "HessCMA", "CholCMA_fc6", "HessCMA500_fc6", ]
# unitdirs = list(rootpath.glob("res*"))
unitdirs = list(rootpath.glob("tf_efficientnet*"))
unitdir_w_missing = []
for unitdir in tqdm(unitdirs[:]): # 340
    for optimname in evoloptimnames:
        # trial_list = list(unitdir.glob(f"lastgen{optimname}_*_score*.jpg"))
        # trial_pat = re.compile(f"lastgen{optimname}_(\d\d\d\d\d)_score([-\d.]*).jpg$")
        trialbest_list = list(unitdir.glob(f"besteachgen{optimname}_*.jpg"))
        if len(trialbest_list) < 10:
            print(unitdir, optimname, f"not enough trials {len(trialbest_list)}")
            unitdir_w_missing.append(unitdir)
            continue
        # elif len(trialbest_list) > 10:
        #     print(unitdir, optimname, f"too many trials {len(trialbest_list)}")
        #     continue
#%%
unitdir_w_missing_uniq = set(unitdir_w_missing)
#%%
# evoloptimnames = ["CholCMA", "HessCMA", "HessCMA500_fc6",]
evoloptimnames = ["CholCMA", "HessCMA", "CholCMA_fc6", "HessCMA500_fc6", ]
# unitdirs = list(rootpath.glob("res*"))
unitdirs = list(rootpath.glob("tf_efficientnet*"))
for unitdir in tqdm(unitdirs[:]): # 340
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
    print(unit_match, "=", netname, layer, unitid, x, y, RFresize)
    unitdict = edict(netname=netname, layer=layer, unitid=unitid, x=x, y=y, RFresize=RFresize)
    proto_col = defaultdict(list)
    info_col = defaultdict(list)
    # score_col = defaultdict(list)
    imgsize = 256  # 227 if RFresize else 256
    for optimname in evoloptimnames:
        # trial_list = list(unitdir.glob(f"lastgen{optimname}_*_score*.jpg"))
        # trial_pat = re.compile(f"lastgen{optimname}_(\d\d\d\d\d)_score([-\d.]*).jpg$")
        trialbest_list = list(unitdir.glob(f"besteachgen{optimname}_*.jpg"))
        trialbest_pat = re.compile(f"besteachgen(.*)_(\d\d\d\d\d).jpg$")
        if len(trialbest_list) == 0:
            continue
        for trailmtgnm in trialbest_list:
            print(trailmtgnm)
            match = trialbest_pat.findall(trailmtgnm.name)
            assert len(match) == 1
            match = match[0]
            if not (match[0] == optimname):
                continue
            RND = int(match[1])
            data = np.load(unitdir / f"scores{optimname}_{RND:05d}.npz")
            scores = data["scores_all"]
            generations = data["generations"]
            score_max = scores.max()
            score_avg = scores[generations == generations.max()].mean()
            score_maxlast = scores[generations == generations.max()].max()
            mtg = plt.imread(trailmtgnm)
            img = crop_from_montage(mtg, -1, imgsize=imgsize)
            proto_col[optimname].append(img)
            info_col[optimname].append({"RND": RND, "score_max": score_max,
                        "score_avg": score_avg, "score_maxlast": score_maxlast})
        mtg = make_grid_np(proto_col[optimname], nrow=10, padding=2, )
        plt.imsave(join(figdir, f"{unitdir.name}_{optimname}.jpg"), mtg)
    if len(info_col) == 0:
        continue
    pkl.dump(info_col, open(join(figdir, f"{unitdir.name}_evolproto_info.pkl"), "wb"))

#%%
list(Path(figdir).glob("*.pkl"))

#%%
evolrootdir = r"E:\Cluster_Backup\GAN_Evol_cmp"
evolrootpath = Path(evolrootdir)
figdir = r"F:\insilico_exps\GAN_Evol_cmp\protoimgs"
os.makedirs(figdir, exist_ok=True)

evoloptimnames = ["CholCMA", "HessCMA", "CholCMA_fc6", "HessCMA500_fc6", ]
unitdirs = list(evolrootpath.glob("resnet50_linf8_*"))
for unitdir in tqdm(unitdirs[:]): # 340
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
    print(unit_match, "=", netname, layer, unitid, x, y, RFresize)
    unitdict = edict(netname=netname, layer=layer, unitid=unitid, x=x, y=y, RFresize=RFresize)
    proto_col = defaultdict(list)
    info_col = defaultdict(list)
    # score_col = defaultdict(list)
    imgsize = 256  # 227 if RFresize else 256
    for optimname in evoloptimnames:
        # trial_list = list(unitdir.glob(f"lastgen{optimname}_*_score*.jpg"))
        # trial_pat = re.compile(f"lastgen{optimname}_(\d\d\d\d\d)_score([-\d.]*).jpg$")
        trialbest_list = list(unitdir.glob(f"besteachgen{optimname}_*.jpg"))
        trialbest_pat = re.compile(f"besteachgen(.*)_(\d\d\d\d\d).jpg$")
        if len(trialbest_list) == 0:
            continue
        for trailmtgnm in trialbest_list:
            print(trailmtgnm)
            match = trialbest_pat.findall(trailmtgnm.name)
            assert len(match) == 1
            match = match[0]
            if not (match[0] == optimname):
                continue
            RND = int(match[1])
            data = np.load(unitdir / f"scores{optimname}_{RND:05d}.npz")
            scores = data["scores_all"]
            generations = data["generations"]
            score_max = scores.max()
            score_avg = scores[generations == generations.max()].mean()
            score_maxlast = scores[generations == generations.max()].max()
            mtg = plt.imread(trailmtgnm)
            img = crop_from_montage(mtg, -1, imgsize=imgsize)
            proto_col[optimname].append(img)
            info_col[optimname].append({"RND": RND, "score_max": score_max,
                        "score_avg": score_avg, "score_maxlast": score_maxlast})
        mtg = make_grid_np(proto_col[optimname], nrow=10, padding=2, )
        plt.imsave(join(figdir, f"{unitdir.name}_{optimname}.jpg"), mtg)
    if len(info_col) == 0:
        continue
    pkl.dump(info_col, open(join(figdir, f"{unitdir.name}_evolproto_info.pkl"), "wb"))
