import datetime
import pandas as pd
from scipy.stats import sem, ttest_ind, ttest_rel, ttest_1samp
def parse_meta(S):
    ephysFN = S["meta"]["ephysFN"]
    expControlFN = S["meta"]["expControlFN"]
    if ephysFN is not None:
        # 'Beto-28072020-006'
        ephysFN_parts = ephysFN.split("-")
        Animal_PL2 = ephysFN_parts[0]
        date_raw = ephysFN_parts[1]
        expdate_PL2 = datetime.datetime.strptime(date_raw, "%d%m%Y")
    if expControlFN is not None:
        # '200728_Beto_generate_BigGAN(1)'
        expctrl_parts = expControlFN.split("_")
        date_raw = expctrl_parts[0]
        Animal_bhv = expctrl_parts[1]
        expdate_bhv = datetime.datetime.strptime(date_raw, "%y%m%d")
    if ephysFN is not None and expControlFN is not None:
        assert Animal_PL2 == Animal_bhv
        assert expdate_PL2 == expdate_bhv
        return Animal_PL2, expdate_PL2.date()
    elif ephysFN is not None:# return the one that is not None
        return Animal_PL2, expdate_PL2.date()
    elif expControlFN is not None:
        return Animal_bhv, expdate_bhv.date()
    else:
        raise ValueError("Both ephysFN and expControlFN are None, cannot parse")


def area_mapping(chan, Animal, expdate):
    if Animal == "Beto" and expdate > datetime.date(2021, 9, 1):
        # beto's new array layout
        if (chan <= 32 and chan >= 17):
            area = "V1"
        if (chan < 17):
            area = "V4"
        if (chan >= 33):
            area = "IT"
    elif Animal in ("Alfa", "Beto"):
        if (chan <= 48 and chan >= 33):
            area = "V1"
        if (chan > 48):
            area = "V4"
        if (chan < 33):
            area = "IT"
    else:
        raise ValueError("Unknown Animal")
    return area


def get_meta_dict(S):
    from easydict import EasyDict as edict
    from neuro_data_analysis.neural_data_lib import extract_evol_activation_array
    if S["evol"] is None:
        return edict()
    # expstr = get_expstr(BFEStats, Expi)
    # print(expstr)
    Animal, expdate = parse_meta(S)
    ephysFN = S["meta"]['ephysFN']
    prefchan = int(S['evol']['pref_chan'][0])
    prefunit = int(S['evol']['unit_in_pref_chan'][0])
    visual_area = area_mapping(prefchan, Animal, expdate)
    spacenames = S['evol']['space_names']
    space1 = spacenames[0] if isinstance(spacenames[0], str) else spacenames[0][0]
    space2 = spacenames[1] if isinstance(spacenames[1], str) else spacenames[1][0]
    resp_arr0, bsl_arr0, gen_arr0, _, _, _ = extract_evol_activation_array(S, 0)
    resp_arr1, bsl_arr1, gen_arr1, _, _, _ = extract_evol_activation_array(S, 1)

    # if the lAST BLOCK has < 10 images, in either thread, then remove it
    if len(resp_arr0[-1]) < 10 or len(resp_arr1[-1]) < 10:
        resp_arr0 = resp_arr0[:-1]
        resp_arr1 = resp_arr1[:-1]
        bsl_arr0 = bsl_arr0[:-1]
        bsl_arr1 = bsl_arr1[:-1]
        gen_arr0 = gen_arr0[:-1]
        gen_arr1 = gen_arr1[:-1]

    meta_dict = edict(Animal=Animal, expdate=expdate, ephysFN=ephysFN, prefchan=prefchan, prefunit=prefunit,
                      visual_area=visual_area, space1=space1, space2=space2, blockN=len(resp_arr0))
    return meta_dict


def get_meta_df(BFEStats):
    from collections import OrderedDict
    from tqdm import tqdm
    meta_col = OrderedDict()
    for i, S in tqdm(enumerate(BFEStats)):
        Expi = i + 1
        meta_dict = get_meta_dict(S)
        if len(meta_dict) == 0:
            continue
        meta_col[Expi] = meta_dict # note Expi starts from 0, but the index starts from 1
    meta_df = pd.DataFrame.from_dict(meta_col, orient="index")
    return meta_df

def get_all_masks(meta_df):
    """
    Get all the masks for different conditions in the analysis
    :param meta_df:
    :return:
    """
    # plot the FC and BG win block number as
    Amsk  = meta_df.Animal == "Alfa"
    Bmsk  = meta_df.Animal == "Beto"
    V1msk = meta_df.visual_area == "V1"
    V4msk = meta_df.visual_area == "V4"
    ITmsk = meta_df.visual_area == "IT"
    length_msk = (meta_df.blockN > 14)
    spc_msk = (meta_df.space1 == "fc6") & meta_df.space2.str.contains("BigGAN")
    sucsmsk = (meta_df.p_maxinit_0 < 0.05) | (meta_df.p_maxinit_1 < 0.05)
    baseline_jump_list = ["Beto-18082020-002",
                          "Beto-07092020-006",
                          "Beto-14092020-002",
                          "Beto-27102020-003",
                          "Alfa-22092020-003",
                          "Alfa-04092020-003"]
    bsl_unstable_msk = meta_df.ephysFN.str.contains("|".join(baseline_jump_list), case=True, regex=True)
    assert bsl_unstable_msk.sum() == len(baseline_jump_list)
    bsl_stable_msk = ~bsl_unstable_msk
    # valid experiments are those with enough blocks, stable baseline and correct fc6-BigGAN pairing
    validmsk = length_msk & bsl_stable_msk & spc_msk
    # print summary of the inclusion criteria
    print("total number of experiments: %d" % len(meta_df))
    print("total number of valid experiments: %d" % validmsk.sum())
    print("total number of valid experiments with suc: %d" % (validmsk & sucsmsk).sum())
    print("Exluded:")
    print("  - short: %d" % (~length_msk).sum())
    print("  - unstable baseline: %d" % bsl_unstable_msk.sum())
    print("  - not fc6-BigGAN: %d" % (~spc_msk).sum())
    return Amsk, Bmsk, V1msk, V4msk, ITmsk, length_msk, spc_msk, sucsmsk, bsl_unstable_msk, bsl_stable_msk, validmsk
