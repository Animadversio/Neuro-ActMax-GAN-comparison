from pathlib import Path
from os.path import join
from easydict import EasyDict as edict
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from core.utils.montage_utils import crop_all_from_montage
import torch
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
# def sweep_dir(rootdir, unit_pattern, ):
from core.yolo_lib import yolo_process_objconf, yolo_process

#%%
yolomodel = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
# plt.switch_backend('module://backend_interagg')
#%%
yolomodel(imgcrops, size=256)
#%%
def yolo_process_imgs(imglist, batch_size=100, size=256, savename=None, sumdir=sumdir):
    """Process images with yolo model and return results in a list of dataframes"""
    results_dfs = []
    for i in trange(0, len(imglist), batch_size):
        results = yolomodel(imglist[i:i + batch_size], size=size)
        results_dfs.extend(results.pandas().xyxy)
        # yolo_results[i] = results

    yolo_stats = {}
    for i, single_df in tqdm(enumerate(results_dfs)):
        yolo_stats[i] = {"confidence": single_df.confidence.max(),
                         "class": single_df["class"][single_df.confidence.argmax()] if len(single_df) > 0 else None,
                         "n_objs": len(single_df),
                         "imgnum": i}
    yolo_stats_df = pd.DataFrame(yolo_stats).T
    if savename is not None:
        yolo_stats_df.to_csv(sumdir / f"{savename}_yolo_stats.csv")
        pkl.dump(results_dfs, open(sumdir / f"{savename}_dfs.pkl", "wb"))
        print(f"Saved to {sumdir / f'{savename}_dfs.pkl'}")
        print(f"Saved to {sumdir / f'{savename}_yolo_stats.csv'}")
    print("Fraction of images with objects", (yolo_stats_df.n_objs > 0).mean())
    print("confidence", yolo_stats_df.confidence.mean(), "confidence with 0 filled",
          yolo_stats_df.confidence.fillna(0).mean())
    print("n_objs", yolo_stats_df.n_objs.mean(), )
    if len(yolo_stats_df) > 0:
        print("most common class", yolo_stats_df["class"].value_counts().index[0])
    return results_dfs, yolo_stats_df
#%%
from core.yolo_lib import yolo_process_objconf, yolo_process_objconf_arrs
#%%
rootdir = r"F:\insilico_exps\GAN_Evol_cmp"
sumdir = Path(rootdir) / "yolo_objconf_summary"
sumdir.mkdir(exist_ok=True)
finalsumdir = Path(rootdir) / "final_summary"
finalsumdir.mkdir(exist_ok=True)
unit_pattern = "resnet50_linf8_*"
# unit_pattern = r"tf_efficientnet*"
rootpath = Path(rootdir)
unitdirs = list(rootpath.glob(unit_pattern))
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

    savefiles = list(unitdir.glob("scores*.npz"))
    # savefn_pat = re.compile("scores(.*)_(\d\d\d\d\d).npz$")
    imgfiles = list(unitdir.glob("besteachgen*.jpg"))
    imgfn_pat = re.compile("besteachgen(.*)_(\d\d\d\d\d).jpg$")
    for imgfn in imgfiles:
        imgfn_match = imgfn_pat.findall(imgfn.name)
        assert len(imgfn_match) == 1
        imgfn_match = imgfn_match[0]
        optimmethod = imgfn_match[0]
        RND = int(imgfn_match[-1])
        if optimmethod.endswith("_fc6"):
            GANname = "fc6"
        else:
            GANname = "BigGAN"
        # print(optimmethod, RND, GANname)
        optimdict = edict(optimmethod=optimmethod, RND=RND, GANname=GANname)
        mtg = plt.imread(imgfn)
        imgcrops = crop_all_from_montage(mtg, totalnum=100, imgsize=256, pad=2, autostop=False)
        savefn = unitdir / f"scores{optimmethod}_{RND:05d}.npz"
        data = np.load(savefn)
        scores_all = data["scores_all"]
        generations = data["generations"]
        endscores = scores_all[generations == generations.max()].mean()
        mean_scores = [scores_all[generations == i].mean() for i in range(generations.min(), generations.max()+1)]
        max_scores = [scores_all[generations == i].max() for i in range(generations.min(), generations.max()+1)]
        mean_scores = np.array(mean_scores)
        max_scores = np.array(max_scores)
        savename = f"{unitdir.name}_{optimmethod}_{RND:05d}"
        # results_dfs, yolo_stats_df = yolo_process_imgs(imgcrops, size=256, savename=savename, sumdir=unitdir)
        results_dfs, yolo_stats_df = yolo_process_objconf_arrs(yolomodel, imgcrops, size=256,
                                                           savename=savename, sumdir=unitdir)
        for k, v in optimdict.items():
            yolo_stats_df[k] = v
        for k, v in unitdict.items():
            yolo_stats_df[k] = v
        yolo_stats_df["mean_score"] = mean_scores
        yolo_stats_df["max_score"] = max_scores
        # yolo_stats_df.to_csv(sumdir / f"yolo_stats_{savename}.csv")
        yolo_stats_df.to_csv(sumdir / f"yolo_objconf_stats_{savename}.csv")
        df_col.append({**unitdict, **optimdict, "yolo_stats_df": yolo_stats_df, })
    #  raise  Exception
    # df_evol = pd.DataFrame(df_col)
    # change datatype of columns GANname, layer, optimmethod, netname as string
    # df_evol = df_evol.astype({"GANname": str, "layer": str, "optimmethod": str, "netname": str,
    #                           "score": float, "maxscore": float, "maxstep": int, "RFresize": bool})
    # return df_evol
#%%
pkl.dump(df_col, open(finalsumdir / "resnet50_linf8_yolo_objconf_stats_df_all.pkl", "wb"))
# pkl.dump(df_col, open(finalsumdir / "resnet50_linf8_yolo_stats_df_all.pkl", "wb"))
df_all = pd.DataFrame(df_col)
all_yolo_stats = pd.concat(df_all.yolo_stats_df.tolist())
all_yolo_stats.to_csv(finalsumdir / "resnet50_linf8_yolo_objconf_stats_df_all.csv")
#%%
pkl.dump(df_col, open(finalsumdir / "efficientnet_yolo_stats_df_all.pkl", "wb"))
df_all = pd.DataFrame(df_col)
all_yolo_stats = pd.concat(df_all.yolo_stats_df.tolist())
all_yolo_stats.to_csv(finalsumdir / "efficientnet_yolo_stats_df_all.csv")
#%%


#%%
pkl.dump(df_col, open(finalsumdir / "resnet50_linf8_yolo_stats_df_all.pkl", "wb"))
df_all = pd.DataFrame(df_col)
all_yolo_stats = pd.concat(df_all.yolo_stats_df.tolist())
all_yolo_stats.to_csv(finalsumdir / "resnet50_linf8_yolo_stats_df_all.csv")
#%%
plt.switch_backend('module://backend_interagg')
