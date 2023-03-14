import datetime
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