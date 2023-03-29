"""Plot the success rate and estimates its error bar for different cortices. """
import scipy
import scipy.special
import matplotlib.pyplot as plt
from neuro_data_analysis.neural_data_lib import *
#%%
def rate_CI(n, k, q):
    """ Confidence interval for success rate
    n: number of trials
    k: number of successes
    q: quantile (0.5 for median)

    Example:
        n = 20
        k = 17
        q = 0.5  # median
        scipy.special.betaincinv(k+1, n+1-k, q)
        # Gives a success rate of 83% (71% - 91%).
    """
    return scipy.special.betaincinv(k+1, n+1-k, q)


rate_CI(10, 1, [0.1, 0.9])
#%%
BFEStats_merge, BFEStats = load_neural_data()
#%%
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

#%%
for i, S in enumerate(BFEStats):
    Animal, expdate = parse_meta(S)
    print(i, Animal, expdate)

#%%
success_col = []
for Expi in range(1, len(BFEStats)+1):
    S = BFEStats[Expi - 1]
    if S["evol"] is None:
        continue
    Animal, expdate = parse_meta(S)
    prefchan = int(S['evol']['pref_chan'][0])
    prefunit = int(S['evol']['unit_in_pref_chan'][0])
    visual_area = area_mapping(prefchan, Animal, expdate)
    #%
    resp_FC, bsl_FC, gen_FC, _, _, _ = extract_evol_activation_array(S, 0)
    resp_BG, bsl_BG, gen_BG, _, _, _ = extract_evol_activation_array(S, 1)
    # np.concatenate
    init_resp_FC = np.concatenate((resp_FC[:2]), axis=0)
    init_resp_BG = np.concatenate((resp_BG[:2]), axis=0)
    end_resp_FC = np.concatenate((resp_FC[-3:-1]), axis=0)
    end_resp_BG = np.concatenate((resp_BG[-3:-1]), axis=0)
    # max_resp_FC = np.concatenate((resp_FC[maxid:maxid+2]), axis=0)
    # max_resp_BG = np.concatenate((resp_BG[maxid:maxid+2]), axis=0)
    endinit_tval_BG, endinit_pval_BG = ttest_ind(end_resp_BG, init_resp_BG)
    endinit_tval_FC, endinit_pval_FC = ttest_ind(end_resp_FC, init_resp_FC)
    # maxinit_tval_BG, maxinit_pval_BG = ttest_ind(max_resp_BG, init_resp_BG)
    # maxinit_tval_FC, maxinit_pval_FC = ttest_ind(max_resp_FC, init_resp_FC)
    Stat = edict(Expi=Expi, Animal=Animal, expdate=expdate, ephysFN=S["meta"]["ephysFN"],
                 prefchan=prefchan, prefunit=prefunit, visual_area=visual_area,
                 endinit_tval_BG=endinit_tval_BG, endinit_pval_BG=endinit_pval_BG,
                 endinit_tval_FC=endinit_tval_FC, endinit_pval_FC=endinit_pval_FC,)
                 # maxinit_tval_BG=maxinit_tval_BG, maxinit_pval_BG=maxinit_pval_BG,
                 # maxinit_tval_FC=maxinit_tval_FC, maxinit_pval_FC=maxinit_pval_FC)

    success_col.append(Stat)
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(success_col)
#%%
df["endinit_success_BG"] = df["endinit_pval_BG"] < 0.01
df["endinit_success_FC"] = df["endinit_pval_FC"] < 0.01
df.groupby("visual_area")[["endinit_success_BG", "endinit_success_FC"]].mean()
sumtable = df.groupby("visual_area")[["endinit_success_BG", "endinit_success_FC"]].mean()
#%%

#%%
sumtable.iloc[[1,2,0]].plot(kind="bar")
plt.ylabel("Proportion of significant Evolution")
plt.title("T-test of end vs init response")
plt.show()
#%%
count_table = df.groupby("visual_area").agg({"endinit_success_FC":["sum", "count"],
                                             "endinit_success_BG":["sum", "count"],}).iloc[[1,2,0]]
#%%
FC_rate_upr = rate_CI(count_table[("endinit_success_FC","count")].to_numpy(),
        count_table[("endinit_success_FC","sum")].to_numpy(), 0.95)
FC_rate_lwr = rate_CI(count_table[("endinit_success_FC","count")].to_numpy(),
        count_table[("endinit_success_FC","sum")].to_numpy(), 0.05)
FC_rate = count_table[("endinit_success_FC","sum")].to_numpy() / \
          count_table[("endinit_success_FC","count")].to_numpy()
FC_success = count_table[("endinit_success_FC","sum")].to_numpy()
FC_total = count_table[("endinit_success_FC","count")].to_numpy()

BG_rate_upr = rate_CI(count_table[("endinit_success_BG","count")].to_numpy(),
            count_table[("endinit_success_BG","sum")].to_numpy(), 0.95)
BG_rate_lwr = rate_CI(count_table[("endinit_success_BG","count")].to_numpy(),
            count_table[("endinit_success_BG","sum")].to_numpy(), 0.05)
BG_rate = count_table[("endinit_success_BG","sum")].to_numpy() / \
            count_table[("endinit_success_BG","count")].to_numpy()
BG_success = count_table[("endinit_success_BG","sum")].to_numpy()
BG_total = count_table[("endinit_success_BG","count")].to_numpy()
#%%
from core.utils.plot_utils import saveallforms
outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\SuccessRate"
plt.figure(figsize=(4.5,4))
plt.plot(FC_rate, label="DeePSim", lw=2)
plt.plot(BG_rate, label="BigGAN", lw=2)
plt.fill_between(np.arange(3), FC_rate_lwr, FC_rate_upr, alpha=0.2)
plt.fill_between(np.arange(3), BG_rate_lwr, BG_rate_upr, alpha=0.2)
plt.xticks(np.arange(3), ["V1", "V4", "IT"])
plt.ylabel("Success rate")
plt.title("Success rate of Evolution per visual area\nT-test of end vs init; mean, 90% CI")
plt.legend()
saveallforms(outdir, "evol_endinit_success_rate_per_area")
plt.show()
#%% Add the number of trials annotation.
plt.figure(figsize=(4.5,4))
plt.plot(FC_rate, label="DeePSim", lw=2)
plt.plot(BG_rate, label="BigGAN", lw=2)
plt.fill_between(np.arange(3), FC_rate_lwr, FC_rate_upr, alpha=0.2)
plt.fill_between(np.arange(3), BG_rate_lwr, BG_rate_upr, alpha=0.2)
# annotate the fraction at each point of the line plot
for i in range(3):
    plt.annotate(f"{FC_success[i]}/{FC_total[i]}", (i+0.02, FC_rate[i]), ha="left", va="bottom", fontsize=12)
    plt.annotate(f"{BG_success[i]}/{BG_total[i]}", (i+0.02, BG_rate[i]), ha="left", va="bottom", fontsize=12)
plt.xticks(np.arange(3), ["V1", "V4", "IT"])
plt.ylabel("Success rate")
plt.title("Success rate of Evolution per visual area\nT-test of end vs init; mean, 90% CI")
plt.legend()
saveallforms(outdir, "evol_endinit_success_rate_per_area_annot")
plt.show()
#%%
import numpy as np
import scipy.stats

def mean_CI_up(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m+h


def mean_CI_low(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h

# df.groupby("visual_area")[["endinit_tval_FC", "endinit_tval_BG"]].mean()
tval_table = df.groupby("visual_area").agg({"endinit_tval_FC":(np.mean, mean_CI_up, mean_CI_low),
                               "endinit_tval_BG":(np.mean, mean_CI_up, mean_CI_low),}).iloc[[1,2,0]]#.sem()
#%% Plot
plt.figure(figsize=(4.5,4))
plt.plot(tval_table[("endinit_tval_FC","mean")], label="DeePSim", lw=2)
plt.plot(tval_table[("endinit_tval_BG","mean")], label="BigGAN", lw=2)
plt.fill_between(np.arange(3), tval_table[("endinit_tval_FC", "mean_CI_low")],
                    tval_table[("endinit_tval_FC", "mean_CI_up")], alpha=0.2)
plt.fill_between(np.arange(3), tval_table[("endinit_tval_BG", "mean_CI_low")],
                    tval_table[("endinit_tval_BG", "mean_CI_up")], alpha=0.2)
plt.xticks(np.arange(3), ["V1", "V4", "IT"])
plt.ylabel("T-value")
plt.title("T-value of Evolution per visual area\nT-test of end vs init; mean, 90% CI")
plt.legend()
saveallforms(outdir, "evol_endinit_tval_per_area")
plt.show()

#%%
plt.figure(figsize=(4.5, 4))
sns.lineplot(data=df, x="visual_area", y="endinit_tval_FC", sort=False, label="DeePSim")  # order=["V1", "V4", "IT", ]
sns.lineplot(data=df, x="visual_area", y="endinit_tval_BG", sort=False, label="BigGAN")  # order=["V1", "V4", "IT", ]
plt.title("T-test of end vs init response")
plt.ylabel("T-value between end and init response")
plt.legend()
saveallforms(outdir, "evol_success_tval_lineplot")
plt.show()
#%% 
