"""
    lib to load formatted neural data from mat and pickle files.
"""
import mat73
from scipy.io import loadmat
import pickle
from easydict import EasyDict as edict
import os
import re
import glob
from pathlib import Path
import numpy as np
import os.path
from os.path import join
from tqdm import tqdm
matroot = "E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
#%
def load_neural_data_and_save_pkl():
    """
    Load neural data from a .mat file.
    """
    BFEStats_merge = mat73.loadmat(join(matroot, "Both_BigGAN_FC6_Evol_Stats.mat"))
    pickle.dump(BFEStats_merge, open(join(matroot, "Both_BigGAN_FC6_Evol_Stats.pkl"), "wb"))
    # organize the data as a list
    BFEStats = []
    for Expi in range(len(BFEStats_merge["BFEStats"]["Animal"])):
        S = edict()
        for key in list(BFEStats_merge["BFEStats"]):
            S[key] = BFEStats_merge["BFEStats"][key][Expi]
        BFEStats.append(S)
    pickle.dump(BFEStats, open(join(matroot, "Both_BigGAN_FC6_Evol_Stats_expsep.pkl"), "wb"))
    return BFEStats_merge["BFEStats"], BFEStats


def load_neural_data():
    """
    Load neural data from a .pkl file.
    """
    BFEStats_merge = pickle.load(open(join(matroot, "Both_BigGAN_FC6_Evol_Stats.pkl"), "rb"))
    BFEStats = pickle.load(open(join(matroot, "Both_BigGAN_FC6_Evol_Stats_expsep.pkl"), "rb"))
    return BFEStats_merge, BFEStats


def get_expstr(BFEStats, Expi):
    S = BFEStats[Expi - 1]  # Expi follows matlab convention, starts from 1
    space_names = S['evol']['space_names']
    if isinstance(space_names[0], list):
        space_names = [n[0] for n in space_names]
    elif isinstance(space_names[0], str):
        pass
    expstr = f"Exp {Expi:03d} {S['meta']['ephysFN']} Pref chan{int(S['evol']['pref_chan'][0])} U{int(S['evol']['unit_in_pref_chan'][0])}" \
             f"\nimage size {S['evol']['imgsize']} deg  pos {S['evol']['imgpos'][0]}" \
             f"\nEvol thr0: {space_names[0]}" \
             f"   thr1: {space_names[1]}"
    return expstr


def load_img_resp_pairs(BFEStats, Expi, ExpType, thread=0, stimdrive="S:", output_fmt="vec",
                        rsp_wdw=range(50, 200), bsl_wdw=range(0, 45)):
    """ Extract the image and response pairs from the BFEStats. for one thread / one experiment.
    output_fmt: str, "vec" or "arr",
        "vec" returns a flattened vector of image paths and a vector of responses,
        "arr" returns a nested list of image paths and responses, each element corresponds to one block.
    """
    S = BFEStats[Expi - 1].copy()  # Expi follows matlab convention, starts from 1
    # parse the full path of the images
    stimpath = S["meta"]["stimuli"]
    stimpath = stimpath.replace("N:", stimdrive)
    imglist = S['imageName']
    imgfps_all, refimgfp_dict = _map_evol_imglist_2_imgfps(imglist, stimpath, sfx="bmp")
    if S["evol"] is None:
        return [], [], [], []
    if ExpType == "Evol":
        # % load the resp and stim
        psth_thread = S["evol"]["psth"][thread]
        imgidx_thread = S["evol"]["idx_seq"][thread]
        resp_arr = []
        bsl_arr = []
        imgidx_arr = []
        gen_arr = []
        for blocki in range(len(psth_thread)):
            psth_arr = _format_psth_arr(psth_thread[blocki])  # time x images
            resp_arr.append(psth_arr[rsp_wdw, :].mean(axis=0))
            bsl_arr.append(psth_arr[bsl_wdw, :].mean(axis=0))
            idx_arr = _format_idx_arr(imgidx_thread[blocki])
            imgidx_arr.append(idx_arr)
            gen_arr.append((blocki+1) * np.ones_like(idx_arr))

        resp_vec = np.concatenate(resp_arr, axis=0)
        bsl_vec = np.concatenate(bsl_arr, axis=0)
        imgidx_vec = np.concatenate(imgidx_arr, axis=0)
        gen_vec = np.concatenate(gen_arr, axis=0)
        # % load the image full path
        # note to change the index to python convention
        imgfps_arr = [[imgfps_all[idx - 1] for idx in imgids] for imgids in imgidx_arr]
        imgfps_vec = [imgfps_all[idx - 1] for idx in imgidx_vec]
        if output_fmt == "vec":
            return imgfps_vec, resp_vec, bsl_vec, gen_vec
        elif output_fmt == "arr":
            return imgfps_arr, resp_arr, bsl_arr, gen_arr
        else:
            raise ValueError("output_fmt should be vec or arr")
    elif ExpType == "natref":
        raise NotImplementedError


def extract_evol_activation_array(S, thread, rsp_wdw=range(50, 200), bsl_wdw=range(0, 45)):
    psth_thread = S["evol"]["psth"][thread]
    imgidx_thread = S["evol"]["idx_seq"][thread]
    resp_arr = []
    bsl_arr = []
    imgidx_arr = []
    gen_arr = []
    for blocki in range(len(psth_thread)):
        psth_arr = _format_psth_arr(psth_thread[blocki])  # time x images
        resp_arr.append(psth_arr[rsp_wdw, :].mean(axis=0))
        bsl_arr.append(psth_arr[bsl_wdw, :].mean(axis=0))
        idx_arr = _format_idx_arr(imgidx_thread[blocki])
        imgidx_arr.append(idx_arr)
        gen_arr.append((blocki + 1) * np.ones_like(idx_arr))

    resp_vec = np.concatenate(resp_arr, axis=0)
    bsl_vec = np.concatenate(bsl_arr, axis=0)
    imgidx_vec = np.concatenate(imgidx_arr, axis=0)
    gen_vec = np.concatenate(gen_arr, axis=0)
    return resp_arr, bsl_arr, gen_arr, resp_vec, bsl_vec, gen_vec


def parse_montage(mtg):
    """Parse the montage into different subimages.
    for the proto montage files in the ProtoSummary folder
    """
    from core.utils.montage_utils import crop_from_montage
    mtg = mtg.astype(np.float32) / 255.0
    S = edict()
    S.FC_maxblk = crop_from_montage(mtg, (0, 0), 224, 0)
    S.FC_maxblk_avg = crop_from_montage(mtg, (0, 1), 224, 0)
    S.FC_reevol_G = crop_from_montage(mtg, (0, 2), 224, 0)
    S.FC_reevol_pix = crop_from_montage(mtg, (0, 3), 224, 0)
    S.BG_maxblk = crop_from_montage(mtg, (1, 0), 224, 0)
    S.BG_maxblk_avg = crop_from_montage(mtg, (1, 1), 224, 0)
    S.BG_reevol_G = crop_from_montage(mtg, (1, 2), 224, 0)
    S.BG_reevol_pix = crop_from_montage(mtg, (1, 3), 224, 0)
    S.both_reevol_G = crop_from_montage(mtg, (2, 2), 224, 0)
    S.both_reevol_pix = crop_from_montage(mtg, (2, 3), 224, 0)
    return S


def _format_psth_arr(psth_block_arr):
    if psth_block_arr.ndim == 1 and psth_block_arr.dtype == np.uint64:
        assert np.prod(psth_block_arr) == 0
        psth_block_arr = np.zeros(psth_block_arr, dtype=np.float64) # empty array
    if psth_block_arr.ndim == 1:
        psth_block_arr = psth_block_arr[:, None]
    elif psth_block_arr.ndim == 2:
        pass
    elif psth_block_arr.ndim == 3:
        if psth_block_arr.shape[0] != 1:
            print("Warning: psth_block_arr has 3 dimensions, only the first one is used.")
            raise NotImplementedError
        psth_block_arr = psth_block_arr[0, :, :]
        # raise NotImplementedError
    else:
        raise ValueError("psth_block_arr.ndim should be 1, 2 or 3")
    return psth_block_arr


def _format_idx_arr(idx_arr):
    if idx_arr.ndim == 1 and idx_arr.dtype == np.uint64:
        """ empty array, wrongly loaded as uint64 """
        assert np.prod(idx_arr) == 0
        idx_arr = np.zeros((0, ), dtype=np.float64)  # empty array
    if idx_arr.ndim == 0:
        idx_arr = idx_arr[None]
    if idx_arr.ndim == 1:
        pass
    elif idx_arr.ndim == 2:
        if idx_arr.shape[0] != 1:
            print("Warning: idx_arr has 3 dimensions, only the first one is used.")
            raise NotImplementedError
        idx_arr = idx_arr[0, :]
    else:
        raise ValueError("idx_arr.ndim should be 0, 1, 2 ")
    return idx_arr.astype(int)


def _map_evol_imglist_2_imgfps(imglist, stimpath, sfx="bmp"):
    """map the image list to the full path of the images in the stimpath"""
    refimgfn_set = set([imgfnl[0] for imgfnl in imglist if imgfnl[0].endswith("_nat")])
    refimgfp_dict = {}
    for refimgfn in refimgfn_set:
        # imgfn = '06_n07715103_4286'
        # use regex to find and extract the str before '_thread\d\d\d_nat' in imgfn if there is
        imgfn_strip = re.sub("_thread\d\d\d_nat", "", refimgfn)
        # note the imgfn_strip could contain special charaters like [ ] ( ) which needs to be escaped by `glob.escape`
        refimgfp = [*Path(stimpath).parent.glob(glob.escape(imgfn_strip) + "*")]
        if len(refimgfp) == 0:
            print(f"Warning: {imgfn_strip} does not exist in {Path(stimpath).parent}.")
        elif len(refimgfp) > 1:
            # if multiple matches. AB matches ABC!
            refimgfp = sorted(refimgfp, key=lambda s: len(s.stem)) # shortest to longest
            refimgfp_stems = [s.stem for s in refimgfp]
            if all([refimgfp_stems[0] in stem for stem in refimgfp_stems]):
                pass
            else:
                print(f"Warning: {imgfn_strip} has more than one file.\n {refimgfp}")
        refimgfp_dict[refimgfn] = str(refimgfp[0])

    imgfps_all = []
    for imgfn in imglist:
        if imgfn[0].endswith("_nat"):
            imgfp = refimgfp_dict[imgfn[0]]  # FIXED: use the full path of the ref image
        else:
            imgfp = join(stimpath, imgfn[0] + "." + sfx)
            if not os.path.exists(imgfp):
                print(f"Warning: {imgfp} does not exist.")
        imgfps_all.append(imgfp)
    assert len(imgfps_all) == len(imglist)
    return imgfps_all, refimgfp_dict

#%%
if __name__ == "__main__":
    BFEStats_merge, BFEStats = load_neural_data()
    #%%
    Expi = 12
    imgfps, resp_vec, bsl_vec, gen_vec = load_img_resp_pairs(BFEStats, Expi, "Evol", thread=0, stimdrive="S:", output_fmt="vec")
    #%%
    imgfps_arr, resp_arr, bsl_arr, gen_arr = load_img_resp_pairs(BFEStats, Expi, "Evol", thread=1, stimdrive="S:", output_fmt="arr")
    #%%
