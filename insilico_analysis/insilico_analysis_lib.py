from pathlib import Path
from os.path import join
from easydict import EasyDict as edict
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
def sweep_dir(rootdir, unit_pattern, save_pattern):
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