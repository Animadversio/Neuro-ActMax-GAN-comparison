
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
from core.utils.montage_utils import crop_from_montage, crop_all_from_montage
rootdir = r"F:\insilico_exps\GAN_gradEvol_cmp"
rootpath = Path(rootdir)
datalist = glob.glob(join(rootdir, "*", "*.pt"))
figdir = join(rootdir, "protoimgs")
os.makedirs(figdir, exist_ok=True)
#%%
optimnames = ["Adam001Hess", "Adam001", "Adam01Hess_fc6", "Adam01_fc6"]
#%%
from core.utils.montage_utils import make_grid, make_grid_np, make_grid_T
from collections import defaultdict
import pickle as pkl
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
    mtgfiles_pat = re.compile("imglastgen(.*)_(\d\d\d\d\d).jpg$")
    for optimname in optimnames:
        mtgfiles = list(unitdir.glob(f"imglastgen{optimname}*.jpg"))
        if len(mtgfiles) == 0:
            continue
        for mtgfile in mtgfiles:
            mtgfile_match = mtgfiles_pat.findall(mtgfile.name)
            assert len(mtgfile_match) == 1
            mtgfile_match = mtgfile_match[0]
            RND = int(mtgfile_match[-1])
            optimdict = edict(optimmethod=optimname, RND=RND)
            imgs = crop_all_from_montage(plt.imread(mtgfile), totalnum=None, imgsize=227)
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

#%%
# for optim in optimnames:
#     print(len(proto_col[optim]))
