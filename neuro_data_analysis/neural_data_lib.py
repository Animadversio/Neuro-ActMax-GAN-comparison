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
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm
from neuro_data_analysis.neural_data_utils import parse_meta, area_mapping
from scipy.stats import sem, ttest_ind, ttest_1samp, ttest_rel

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


def extract_evol_psth_array(S, thread, ):
    psth_thread = S["evol"]["psth"][thread]
    imgidx_thread = S["evol"]["idx_seq"][thread]
    psth_all = []
    bsl_arr = []
    imgidx_arr = []
    gen_arr = []
    for blocki in range(len(psth_thread)):
        psth_arr = _format_psth_arr(psth_thread[blocki])  # time x images
        psth_all.append(psth_arr[:, :])
        idx_arr = _format_idx_arr(imgidx_thread[blocki])
        imgidx_arr.append(idx_arr)
        gen_arr.append((blocki + 1) * np.ones_like(idx_arr))

    psth_vec = np.concatenate(psth_all, axis=-1)
    imgidx_vec = np.concatenate(imgidx_arr, axis=0)
    gen_vec = np.concatenate(gen_arr, axis=0)
    return psth_all, gen_arr, psth_vec, gen_vec


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


def extract_all_evol_trajectory(BFEStats):
    return extract_all_evol_trajectory_dyna(BFEStats, rsp_wdw=range(50, 200))


def extract_all_evol_trajectory_dyna(BFEStats, rsp_wdw=range(50, 200)):
    """
    Extract the evolution trajectory of all the experiments in the BFEStats list into
    an dictionary of arrays. and a meta data dataframe

    Examples:
        _, BFEStats = load_neural_data()
        rsp_wdw = range(50, 200)
        resp_col, meta_df = extract_all_evol_trajectory_dyna(BFEStats, rsp_wdw=rsp_wdw)
        resp_extrap_arr, extrap_mask_arr, max_len = pad_resp_traj(resp_col)
        Amsk, Bmsk, V1msk, V4msk, ITmsk, \
            length_msk, spc_msk, sucsmsk, \
            bsl_unstable_msk, bsl_stable_msk, validmsk = get_all_masks(meta_df)

    :param BFEStats:
    :return:
        resp_col : OrderedDict of 3d array, each array is a
                    ( generation x thread [mean0, mean1, sem0, sem1, bsl_mean0, bsl_mean1]  x time )
                    key is the Expi
        meta_df : DataFrame of meta data, each row is a experiment. Including t test stats.
    """
    resp_col = OrderedDict()
    meta_col = OrderedDict()
    #%
    for Expi in range(1, len(BFEStats) + 1):
        S = BFEStats[Expi - 1]
        if S["evol"] is None:
            continue
        expstr = get_expstr(BFEStats, Expi)
        print(expstr)
        Animal, expdate = parse_meta(S)
        ephysFN = S["meta"]['ephysFN']
        prefchan = int(S['evol']['pref_chan'][0])
        prefunit = int(S['evol']['unit_in_pref_chan'][0])
        visual_area = area_mapping(prefchan, Animal, expdate)
        spacenames = S['evol']['space_names']
        space1 = spacenames[0] if isinstance(spacenames[0], str) else spacenames[0][0]
        space2 = spacenames[1] if isinstance(spacenames[1], str) else spacenames[1][0]

        # load the evolution trajectory of each pair
        resp_arr0, bsl_arr0, gen_arr0, _, _, _ = extract_evol_activation_array(S, 0, rsp_wdw=rsp_wdw)
        resp_arr1, bsl_arr1, gen_arr1, _, _, _ = extract_evol_activation_array(S, 1, rsp_wdw=rsp_wdw)

        # if the lAST BLOCK has < 10 images, in either thread, then remove it
        if len(resp_arr0[-1]) < 10 or len(resp_arr1[-1]) < 10:
            resp_arr0 = resp_arr0[:-1]
            resp_arr1 = resp_arr1[:-1]
            bsl_arr0 = bsl_arr0[:-1]
            bsl_arr1 = bsl_arr1[:-1]
            gen_arr0 = gen_arr0[:-1]
            gen_arr1 = gen_arr1[:-1]

        resp_m_traj_0 = np.array([resp.mean() for resp in resp_arr0])
        resp_m_traj_1 = np.array([resp.mean() for resp in resp_arr1])
        resp_sem_traj_0 = np.array([sem(resp) for resp in resp_arr0])
        resp_sem_traj_1 = np.array([sem(resp) for resp in resp_arr1])
        bsl_m_traj_0 = np.array([bsl.mean() for bsl in bsl_arr0])
        bsl_m_traj_1 = np.array([bsl.mean() for bsl in bsl_arr1])

        # test the successfulness of the evolution
        # ttest between the last two blocks and the first two blocks
        t_endinit_0, p_endinit_0 = ttest_ind(np.concatenate(resp_arr0[-2:]), np.concatenate(resp_arr0[:2]))
        t_endinit_1, p_endinit_1 = ttest_ind(np.concatenate(resp_arr1[-2:]), np.concatenate(resp_arr1[:2]))
        # ttest between the max two blocks and the first two blocks
        max_id0 = np.argmax(resp_m_traj_0)
        max_id0 = max_id0 if max_id0 < len(resp_arr0) - 2 else len(resp_arr0) - 3
        t_maxinit_0, p_maxinit_0 = ttest_ind(np.concatenate(resp_arr0[max_id0:max_id0+2]), np.concatenate(resp_arr0[:2]))
        max_id1 = np.argmax(resp_m_traj_1)
        max_id1 = max_id1 if max_id1 < len(resp_arr1) - 2 else len(resp_arr1) - 3
        t_maxinit_1, p_maxinit_1 = ttest_ind(np.concatenate(resp_arr1[max_id1:max_id1+2]), np.concatenate(resp_arr1[:2]))

        t_FCBG_end_01, p_FCBG_end_01 = ttest_ind(np.concatenate(resp_arr0[-2:]), np.concatenate(resp_arr1[:2]))
        t_FCBG_max_01, p_FCBG_max_01 = ttest_ind(np.concatenate(resp_arr0[max_id0:max_id0+2]), np.concatenate(resp_arr1[max_id1:max_id1+2]))

        # save the meta data
        meta_dict = edict(Animal=Animal, expdate=expdate, ephysFN=ephysFN, prefchan=prefchan, prefunit=prefunit,
                          visual_area=visual_area, space1=space1, space2=space2, blockN=len(resp_arr0))
        stat_dict = edict(t_endinit_0=t_endinit_0, p_endinit_0=p_endinit_0,
                        t_endinit_1=t_endinit_1, p_endinit_1=p_endinit_1,
                        t_maxinit_0=t_maxinit_0, p_maxinit_0=p_maxinit_0,
                        t_maxinit_1=t_maxinit_1, p_maxinit_1=p_maxinit_1,
                        t_FCBG_end_01=t_FCBG_end_01, p_FCBG_end_01=p_FCBG_end_01,
                        t_FCBG_max_01=t_FCBG_max_01, p_FCBG_max_01=p_FCBG_max_01,)
        meta_dict.update(stat_dict)

        # stack the trajectories together
        resp_bunch = np.stack([resp_m_traj_0, resp_m_traj_1,
                               resp_sem_traj_0, resp_sem_traj_1,
                               bsl_m_traj_0, bsl_m_traj_1, ], axis=1)
        resp_col[Expi] = resp_bunch
        meta_col[Expi] = meta_dict

    meta_df = pd.DataFrame.from_dict(meta_col, orient="index")
    return resp_col, meta_df


def pad_resp_traj(resp_col):
    """
    Pad the response trajectories to the same length by extrapolating the last block with the mean of last two blocks
    And then stack them together into a 3D array

    :return:
         resp_extrap_arr: 3D array of shape (n_exp x n_blocks x 6)
                values order:
                resp_m_traj_0, resp_m_traj_1, resp_sem_traj_0, resp_sem_traj_1, bsl_m_traj_0, bsl_m_traj_1
         extrap_mask_arr: 2d array with shape (n_exp x n_blocks)
         max_len: scalar, the length of the longest trajectory
    """
    # get the length of the longest trajectory
    max_len = max([resp_bunch.shape[0] for resp_bunch in resp_col.values()])
    # extrapolate the last block with the mean of last two blocks
    resp_extrap_col = OrderedDict()  # use OrderedDict instead of list to keep Expi as key
    extrap_mask_col = OrderedDict()
    for Expi, resp_bunch in resp_col.items():
        # resp_bunch: number of blocks x 6
        n_blocks = resp_bunch.shape[0]
        if n_blocks < max_len:
            extrap_vals = resp_bunch[-2:, :].mean(axis=0)
            resp_bunch = np.concatenate([resp_bunch,
                 np.tile(extrap_vals, (max_len - n_blocks, 1))], axis=0)
        resp_extrap_col[Expi] = resp_bunch
        extrap_mask_col[Expi] = np.concatenate([np.ones(n_blocks), np.zeros(max_len - n_blocks)]).astype(bool)

    # concatenate all trajectories
    resp_extrap_arr = np.stack([*resp_extrap_col.values()], axis=0)
    extrap_mask_arr = np.stack([*extrap_mask_col.values()], axis=0)
    # resp_extrap_arr: n_exp x n_blocks x 6,
    #       values order: resp_m_traj_0, resp_m_traj_1, resp_sem_traj_0, resp_sem_traj_1, bsl_m_traj_0, bsl_m_traj_1
    # extrap_mask_arr: n_exp x n_blocks
    return resp_extrap_arr, extrap_mask_arr, max_len


def extract_all_evol_trajectory_psth(BFEStats):
    """ Extract the whole psth across the Evolution blocks
    (in contrast to the scalar values in `extract_all_evol_trajectory_dyna`)

    Examples:
        _, BFEStats = load_neural_data()
        psth_col, meta_df = extract_all_evol_trajectory_psth(BFEStats)
        psth_extrap_arr, extrap_mask_arr, max_len = pad_psth_traj(psth_col)

    :param BFEStats:
    :return:
        psth_col : OrderedDict of 3d array, each array is a
                        ( generation x thread [mean0, mean1, sem0, sem1]  x time )
                   Key is the Expi
        meta_df : DataFrame of meta data, each row is a experiment
    """
    psth_col = OrderedDict()
    meta_col = OrderedDict()
    for Expi in range(1, len(BFEStats) + 1):
        S = BFEStats[Expi - 1]
        if S["evol"] is None:
            continue
        expstr = get_expstr(BFEStats, Expi)
        print(expstr)
        Animal, expdate = parse_meta(S)
        ephysFN = S["meta"]['ephysFN']
        prefchan = int(S['evol']['pref_chan'][0])
        prefunit = int(S['evol']['unit_in_pref_chan'][0])
        visual_area = area_mapping(prefchan, Animal, expdate)
        spacenames = S['evol']['space_names']
        space1 = spacenames[0] if isinstance(spacenames[0], str) else spacenames[0][0]
        space2 = spacenames[1] if isinstance(spacenames[1], str) else spacenames[1][0]
        # load the evolution trajectory of each pair
        psth_col0, gen_arr0, psth_arr0, _ = extract_evol_psth_array(S, 0, )
        psth_col1, gen_arr1, psth_arr1, _ = extract_evol_psth_array(S, 1, )
        # psth_arr0 : time x trial
        # psth_col0 : list with len = generation number, each element is a 2d array of time x trial
        # gen_arr0 : list with len = generation number, each element is a 1d array of trial number
        # if the lAST BLOCK has < 10 images, in either thread, then remove it
        if len(gen_arr0[-1]) < 10 or len(gen_arr1[-1]) < 10:
            psth_col0 = psth_col0[:-1]
            psth_col1 = psth_col1[:-1]
            gen_arr0 = gen_arr0[:-1]
            gen_arr1 = gen_arr1[:-1]
        assert len(gen_arr0) == len(gen_arr1)

        psth_m_traj_0 = np.array([resp.mean(axis=-1) for resp in psth_col0])  # generation x time
        psth_m_traj_1 = np.array([resp.mean(axis=-1) for resp in psth_col1])  # generation x time
        psth_sem_traj_0 = np.array([sem(resp, axis=-1) for resp in psth_col0])  # generation x time
        psth_sem_traj_1 = np.array([sem(resp, axis=-1) for resp in psth_col1])  # generation x time
        # save the meta data
        meta_dict = edict(Animal=Animal, expdate=expdate, ephysFN=ephysFN, prefchan=prefchan, prefunit=prefunit,
                          visual_area=visual_area, space1=space1, space2=space2, blockN=len(gen_arr0))
        # stack the trajectories together
        psth_bunch = np.stack([psth_m_traj_0, psth_m_traj_1,
                               psth_sem_traj_0, psth_sem_traj_1], axis=1)
        # resp_bunch: generation x thread [mean, sem]  x time
        psth_col[Expi] = psth_bunch
        # list of 3d array, each array is a ( generation x thread [mean, sem]  x time )
        meta_col[Expi] = meta_dict

    meta_df = pd.DataFrame.from_dict(meta_col, orient="index")
    return psth_col, meta_df


def pad_psth_traj(psth_col):
    """
    Pad the response trajectories to the same length by extrapolating the last block with the mean of last two blocks
    And then stack them together into a 3D array
    :param psth_col: OrderedDict of 3d array, each array is a
                        ( generation x thread [mean0, mean1, sem0, sem1]  x time )
                     Key is the Expi
    :return:
        psth_extrap_arr: 4d array with shape
                        (n_exp x n_blocks x 4 x n_time,)
                   values order: psth_m_traj_0, psth_m_traj_1, psth_sem_traj_0, psth_sem_traj_1
        extrap_mask_arr: 2d array with shape (n_exp x n_blocks)
        max_len: the length of the longest trajectory

    """
    # get the length of the longest trajectory
    max_len = max([resp_bunch.shape[0] for resp_bunch in psth_col.values()])
    # extrapolate the last block with the mean of last two blocks
    psth_extrap_col = OrderedDict()  # use OrderedDict instead of list to keep Expi as key
    extrap_mask_col = OrderedDict()
    for Expi, psth_bunch in psth_col.items():
        # resp_bunch: number of blocks x 6
        n_blocks = psth_bunch.shape[0]
        if n_blocks < max_len:
            extrap_vals = psth_bunch[-2:, :, :].mean(axis=0)
            psth_bunch = np.concatenate([psth_bunch,
                                         np.tile(extrap_vals, (max_len - n_blocks, 1, 1))], axis=0)
        psth_extrap_col[Expi] = psth_bunch
        extrap_mask_col[Expi] = np.concatenate([np.ones(n_blocks), np.zeros(max_len - n_blocks)]).astype(bool)

    # concatenate all trajectories
    psth_extrap_arr = np.stack([*psth_extrap_col.values()], axis=0)
    extrap_mask_arr = np.stack([*extrap_mask_col.values()], axis=0)
    # psth_extrap_arr: n_exp x n_blocks x 4 x n_time,
    #       values order: psth_m_traj_0, psth_m_traj_1, psth_sem_traj_0, psth_sem_traj_1
    # extrap_mask_arr: n_exp x n_blocks
    return psth_extrap_arr, extrap_mask_arr, max_len
#%%
if __name__ == "__main__":
    BFEStats_merge, BFEStats = load_neural_data()
    #%%
    Expi = 12
    imgfps, resp_vec, bsl_vec, gen_vec = load_img_resp_pairs(BFEStats, Expi, "Evol", thread=0, stimdrive="S:", output_fmt="vec")
    #%%
    imgfps_arr, resp_arr, bsl_arr, gen_arr = load_img_resp_pairs(BFEStats, Expi, "Evol", thread=1, stimdrive="S:", output_fmt="arr")
    #%%
