"""
Plot the success rate of Evolution and estimates its error bar for different cortices.
"""
import sys
import numpy as np
import scipy.stats
import datetime
import scipy
import scipy.special
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms
from neuro_data_analysis.neural_data_lib import *
from scipy.stats import sem, ttest_ind, ttest_rel, ttest_1samp


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


def rate_CI_data(data, confidence=0.95):
    return rate_CI(len(data), np.sum(data), confidence)


def mean_CI_up(data, confidence=0.95):
    """Util function to calculate the upper bound of the confidence interval of the data"""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m+h


def mean_CI_low(data, confidence=0.95):
    """Util function to calculate the lower bound of the confidence interval of the data"""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h
# rate_CI(10, 1, [0.1, 0.9])
#%%
from neuro_data_analysis.neural_data_utils import parse_meta, area_mapping, get_all_masks, \
    get_meta_df, get_meta_dict

# meta_df = get_meta_df(BFEStats)
outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\SuccessRate"
dfdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables"
meta_df = pd.read_csv(join(dfdir, "meta_stats_w_optimizer.csv"))
#%%
Amsk, Bmsk, V1msk, V4msk, ITmsk, \
    length_msk, spc_msk, sucsmsk, \
    bsl_unstable_msk, bsl_stable_msk, validmsk = get_all_masks(meta_df)
#%%
# pandas config for printing, long table, not truncate
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
#%%
def _print_stat_row(row):
    print(f"{int(row['count']):d}/{int(row['total']):d} rate={row['rate']:.3f} [{row['CI0.05']:.3f}, {row['CI0.95']:.3f}]")


def print_stats_df(stats_sumdf, prefix="max>init"):
    for area in ["V1", "V4", "IT", "Total"]:
        print(f"[{area}]", )#end=": "
        print("DeePSim", end=": ")
        _print_stat_row(stats_sumdf.loc[area, prefix+" FC"])
        print("BigGAN", end=": ")
        _print_stat_row(stats_sumdf.loc[area, prefix+" BG"])
        print("Both", end=": ")
        _print_stat_row(stats_sumdf.loc[area, prefix+" both"])


def visualize_success_rate(stats_sumdf, prefix="max>init", ax=None, title=None, ylim=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_index = ["V1", "V4", "IT"]
    ax.plot(["V1", "V4", "IT"], stats_sumdf.loc[plot_index, prefix+" FC"].rate, color="b", marker="o", label="DeePSim")
    ax.fill_between(["V1", "V4", "IT"], stats_sumdf.loc[plot_index, prefix+" FC"]["CI0.05"].to_list(),
                                        stats_sumdf.loc[plot_index, prefix+" FC"]["CI0.95"].to_list(),
                                        color="b", alpha=0.2)
    ax.plot(["V1", "V4", "IT"], stats_sumdf.loc[plot_index, prefix+" BG"].rate, color="r", marker="o", label="BigGAN")
    ax.fill_between(["V1", "V4", "IT"], stats_sumdf.loc[plot_index, prefix+" BG"]["CI0.05"].to_list(),
                                        stats_sumdf.loc[plot_index, prefix+" BG"]["CI0.95"].to_list(),
                                        color="r", alpha=0.2)
    # annotate the number of trials
    for i, area in enumerate(["V1", "V4", "IT"]):
        ax.annotate(f"{int(stats_sumdf.loc[area, prefix+' FC']['count']):d}/{int(stats_sumdf.loc[area, prefix+' FC']['total']):d}",
                    (i + 0.02, stats_sumdf.loc[area, prefix+" FC"].rate+0.025))
        ax.annotate(f"{int(stats_sumdf.loc[area, prefix+' BG']['count']):d}/{int(stats_sumdf.loc[area, prefix+' BG']['total']):d}",
                    (i + 0.02, stats_sumdf.loc[area, prefix+" BG"].rate-0.025))
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Visual areas")
    plt.tight_layout()
    return ax


figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\SuccessRate"
tabdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables"
for thresh in [0.05, 0.01, 0.005, 0.001]:
    aggfunclist = ["sum", "mean", lambda x: rate_CI_data(x, 0.05),
                   lambda x: rate_CI_data(x, 0.95), "count"]
    meta_df["max>init FC"] = (meta_df.p_maxinit_0 < thresh) & (meta_df.t_maxinit_0 > 0)
    meta_df["max>init BG"] = (meta_df.p_maxinit_1 < thresh) & (meta_df.t_maxinit_1 > 0)
    meta_df["max>init both"] = meta_df["max>init FC"] & meta_df["max>init BG"]
    meta_df["end>init FC"] = (meta_df.p_endinit_0 < thresh) & (meta_df.t_endinit_0 > 0)
    meta_df["end>init BG"] = (meta_df.p_endinit_1 < thresh) & (meta_df.t_endinit_1 > 0)
    meta_df["end>init both"] = meta_df["end>init FC"] & meta_df["end>init BG"]
    meta_df["end<init FC"] = (meta_df.p_endinit_0 < thresh) & (meta_df.t_endinit_0 < 0)
    meta_df["end<init BG"] = (meta_df.p_endinit_1 < thresh) & (meta_df.t_endinit_1 < 0)
    meta_df["end<init both"] = meta_df["end<init FC"] & meta_df["end<init BG"]
    meta_df["max>end FC"] = (meta_df.p_maxend_0 < thresh) & (meta_df.t_maxend_0 > 0)
    meta_df["max>end BG"] = (meta_df.p_maxend_1 < thresh) & (meta_df.t_maxend_1 > 0)
    meta_df["max>end both"] = meta_df["max>end FC"] & meta_df["max>end BG"]
    meta_df["max>end max>init FC"] = (meta_df.p_maxend_0 < thresh) & (meta_df.t_maxend_0 > 0) & (meta_df.p_maxinit_0 < thresh) & (meta_df.t_maxinit_0 > 0)
    meta_df["max>end max>init BG"] = (meta_df.p_maxend_1 < thresh) & (meta_df.t_maxend_1 > 0) & (meta_df.p_maxinit_1 < thresh) & (meta_df.t_maxinit_1 > 0)
    meta_df["max>end max>init both"] = meta_df["max>end max>init FC"] & meta_df["max>end max>init BG"]
    stats_sumdf = meta_df[validmsk].groupby("visual_area").agg(
               {"max>init FC": aggfunclist,
                "max>init BG": aggfunclist,
                "max>init both": aggfunclist,
                "end>init FC": aggfunclist,
                "end>init BG": aggfunclist,
                "end>init both": aggfunclist,
                "end<init FC": aggfunclist,
                "end<init BG": aggfunclist,
                "end<init both": aggfunclist,
                "max>end FC": aggfunclist,
                "max>end BG": aggfunclist,
                "max>end both": aggfunclist,
                "max>end max>init FC": aggfunclist,
                "max>end max>init BG": aggfunclist,
                "max>end max>init both": aggfunclist,
                }).loc[["V1", "V4", "IT"]]
    total_sumdf = meta_df[validmsk].agg(
        {"max>init FC": aggfunclist,
         "max>init BG": aggfunclist,
         "max>init both": aggfunclist,
         "end>init FC": aggfunclist,
         "end>init BG": aggfunclist,
         "end>init both": aggfunclist,
         "end<init FC": aggfunclist,
         "end<init BG": aggfunclist,
         "end<init both": aggfunclist,
         "max>end FC": aggfunclist,
         "max>end BG": aggfunclist,
         "max>end both": aggfunclist,
        "max>end max>init FC": aggfunclist,
        "max>end max>init BG": aggfunclist,
        "max>end max>init both": aggfunclist,
         })
    total_sumdf_row = total_sumdf.melt(ignore_index=False).T
    total_sumdf_row = total_sumdf_row.drop("variable").rename(index={"value": "Total"})
    # Adjust columns to multi-index
    total_sumdf_row.columns = pd.MultiIndex.from_product(
        [total_sumdf.columns, ['sum', 'mean', '<lambda_0>', '<lambda_1>', 'count']])
    # turn the rownames into multiindex of columns
    stats_sumdf = pd.concat([stats_sumdf, total_sumdf_row])
    stats_sumdf.rename(columns={"sum": "count", "mean": "rate", '<lambda_0>': 'CI0.05', '<lambda_1>': 'CI0.95', "count": "total"}, inplace=True)
    stats_sumdf.loc[:, (slice(None), ["total", "count"])] = \
        stats_sumdf.loc[:, (slice(None), ["total", "count"])].astype(int)
    stats_sumdf.loc[:, (slice(None), ["rate", "CI0.05", "CI0.95"])] = \
        stats_sumdf.loc[:, (slice(None), ["rate", "CI0.05", "CI0.95"])].astype(float)
    # stats_sumdf.rename(columns={("end>init both", "total"): ("", "total")}, inplace=True)
    stats_sumdf.to_csv(join(tabdir, f"Evol_success_rate_{thresh:.3f}.csv"))

    sys.stdout = open(join(tabdir, f"Evol_success_rate_{thresh:.3f}_summary.txt"), "w")
    print(f"t-test thresh = {thresh:.3f}")
    print("success criterion = max>init")
    print_stats_df(stats_sumdf, "max>init")
    print("\nsuccess criterion = end>init")
    print_stats_df(stats_sumdf, "end>init")
    print("\nsuccess criterion = end<init")
    print_stats_df(stats_sumdf, "end<init")
    print("\nsuccess criterion = max>end")
    print_stats_df(stats_sumdf, "max>end")
    print("\nsuccess criterion = max>end max>init")
    print_stats_df(stats_sumdf, "max>end max>init")
    sys.stdout.close()
    visualize_success_rate(stats_sumdf, prefix="max>init", title=f"max > init, thresh={thresh:.3f}")
    saveallforms(figdir, f"succss_rate_max-init_thr{thresh:.3f}_per_area")
    visualize_success_rate(stats_sumdf, prefix="end>init", title=f"end > init, thresh={thresh:.3f}")
    saveallforms(figdir, f"succss_rate_end-init_thr{thresh:.3f}_per_area")
    visualize_success_rate(stats_sumdf, prefix="end<init", title=f"end < init, thresh={thresh:.3f}")
    saveallforms(figdir, f"succss_rate_init-end_thr{thresh:.3f}_per_area")
    visualize_success_rate(stats_sumdf, prefix="max>end", title=f"max > end, thresh={thresh:.3f}")
    saveallforms(figdir, f"succss_rate_max-end_thr{thresh:.3f}_per_area")

sys.stdout = sys.__stdout__
#%% plot contingency table
from scipy.stats import chi2_contingency
def test_independence_contingency(A, B):
    AB = sum(A & B)
    AnB = sum(A & ~B)
    nAB = sum(~A & B)
    nAnB = sum(~A & ~B)
    nA = sum(A)
    nB = sum(B)
    n = len(A)
    table = np.array([[AB, AnB], [nAB, nAnB]])
    chi2, p, dof, ex = chi2_contingency(table)
    print(f"chi2 = {chi2:.2f}, p = {p:.1e}, dof = {dof}")
    # format the table
    print(f"ex =\n {ex[0,0]:.1f} {ex[0,1]:.1f}\n {ex[1,0]:.1f} {ex[1,1]:.1f}")
    print(f"AB = {AB}, A~B = {AnB}, ~AB = {nAB}, ~A~B = {nAnB}")
    print(f"nA = {nA}, nB = {nB}, n = {n}")
    print(f"AB/nA = {AB/nA:.3f}, AB/nB = {AB/nB:.3f}, AB/n = {AB/n:.3f}")
    print(f"AnB/nA = {AnB/nA:.3f}, AnB/nB = {AnB/nB:.3f}, AnB/n = {AnB/n:.3f}")
    print(f"nAB/nA = {nAB/nA:.3f}, nAB/nB = {nAB/nB:.3f}, nAB/n = {nAB/n:.3f}")
    return chi2, p, dof, ex


def plot_contingency_table(A, B, title="", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    AB = sum(A & B)
    AnB = sum(A & ~B)
    nAB = sum(~A & B)
    nAnB = sum(~A & ~B)
    nA = sum(A)
    nB = sum(B)
    n = len(A)
    table = np.array([[AB, AnB], [nAB, nAnB]])
    print(title)
    print("observed table\n", table)
    try:
        chi2, p, dof, ex = chi2_contingency(table)
        test_independence_contingency(A, B)
    except ValueError:
        chi2, p, dof, ex = np.nan, np.nan, np.nan, np.nan
    plt.imshow(table, cmap="Blues_r")
    # annotate
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{table[i, j]}",
                     ha="center", va="center", color="black", fontsize=16)
    # plt.colorbar()
    plt.xticks([0, 1], [f"Success\n{nB}", f"Fail\n{n-nB}"], fontsize=16)
    plt.yticks([0, 1], [f"Success\n{nA}", f"Fail\n{n-nA}"], fontsize=16)
    plt.xlabel("B", fontsize=16)
    plt.ylabel("A", fontsize=16)
    plt.title(f"{title} N={n}\nchi2 = {chi2:.2f}, p = {p:.1e}, dof = {dof}", fontsize=16)
    plt.tight_layout()
    # if title is not None:
    #     plt.title(title)
    return chi2, p, dof, ex


figh, axs = plt.subplots(1, 4, figsize=[18, 5.5])
plot_contingency_table(meta_df[validmsk&V1msk]["max>init FC"],
                       meta_df[validmsk&V1msk]["max>init BG"], title="[V1 sessions]", ax=axs[0])
plt.ylabel("DeePSim", fontsize=16); plt.xlabel("BigGAN", fontsize=16)
plot_contingency_table(meta_df[validmsk&V4msk]["max>init FC"],
                       meta_df[validmsk&V4msk]["max>init BG"], title="[V4 sessions]", ax=axs[1])
plt.ylabel("DeePSim", fontsize=16); plt.xlabel("BigGAN", fontsize=16)
plot_contingency_table(meta_df[validmsk&ITmsk]["max>init FC"],
                      meta_df[validmsk&ITmsk]["max>init BG"], title="[IT sessions]", ax=axs[2])
plt.ylabel("DeePSim", fontsize=16); plt.xlabel("BigGAN", fontsize=16)
plot_contingency_table(meta_df[validmsk]["max>init FC"],
                       meta_df[validmsk]["max>init BG"], title="[all sessions]", ax=axs[3])
plt.ylabel("DeePSim", fontsize=16); plt.xlabel("BigGAN", fontsize=16)
plt.show()
#%%
for thresh in [0.05, 0.01, 0.005, 0.001]:
    sys.stdout = open(join(tabdir, f"contingency_tables_{thresh:.3f}.txt"), "w")
    meta_df["max>init FC"] = (meta_df.p_maxinit_0 < thresh) & (meta_df.t_maxinit_0 > 0)
    meta_df["max>init BG"] = (meta_df.p_maxinit_1 < thresh) & (meta_df.t_maxinit_1 > 0)
    meta_df["end>init FC"] = (meta_df.p_endinit_0 < thresh) & (meta_df.t_endinit_0 > 0)
    meta_df["end>init BG"] = (meta_df.p_endinit_1 < thresh) & (meta_df.t_endinit_1 > 0)
    meta_df["end<init FC"] = (meta_df.p_endinit_0 < thresh) & (meta_df.t_endinit_0 < 0)
    meta_df["end<init BG"] = (meta_df.p_endinit_1 < thresh) & (meta_df.t_endinit_1 < 0)
    meta_df["max>end FC"] = (meta_df.p_maxend_0 < thresh) & (meta_df.t_maxend_0 > 0)
    meta_df["max>end BG"] = (meta_df.p_maxend_1 < thresh) & (meta_df.t_maxend_1 > 0)
    for suc_crit_str, crit_savestr in zip(["max>init", "end>init", "end<init", "max>end"],
                                     ["max-init", "end-init", "init-end", "max-end"],):
        print(f"success criterion: {suc_crit_str}")
        print(f"thresh: {thresh}")

        figh, axs = plt.subplots(1, 4, figsize=[18, 5.5])
        plot_contingency_table(meta_df[validmsk & V1msk][suc_crit_str + " FC"],
                               meta_df[validmsk & V1msk][suc_crit_str + " BG"],
                               title="[V1 sessions]", ax=axs[0])
        plt.ylabel("DeePSim", fontsize=16);
        plt.xlabel("BigGAN", fontsize=16)
        plot_contingency_table(meta_df[validmsk & V4msk][suc_crit_str + " FC"],
                               meta_df[validmsk & V4msk][suc_crit_str + " BG"],
                               title="[V4 sessions]", ax=axs[1])
        plt.ylabel("DeePSim", fontsize=16);
        plt.xlabel("BigGAN", fontsize=16)
        plot_contingency_table(meta_df[validmsk & ITmsk][suc_crit_str + " FC"],
                               meta_df[validmsk & ITmsk][suc_crit_str + " BG"],
                               title="[IT sessions]", ax=axs[2])
        plt.ylabel("DeePSim", fontsize=16);
        plt.xlabel("BigGAN", fontsize=16)
        plot_contingency_table(meta_df[validmsk][suc_crit_str + " FC"],
                               meta_df[validmsk][suc_crit_str + " BG"],
                               title="[all sessions]", ax=axs[3])
        plt.ylabel("DeePSim", fontsize=16);
        plt.xlabel("BigGAN", fontsize=16)
        plt.suptitle(f"{suc_crit_str}  p<{thresh}", fontsize=16)
        saveallforms(figdir, f"Evol_success_contingency_{crit_savestr}_p{thresh:.3f}", figh)
        plt.show()
        print("\n\n\n")
    sys.stdout.close()

sys.stdout = sys.__stdout__
#%%

test_independence_contingency(meta_df["max>init FC"][ITmsk&validmsk], meta_df["max>init BG"][ITmsk&validmsk])

#%%
from statsmodels.stats.proportion import proportions_ztest
proportions_ztest([78, 69], [106, 106])
#%%
proportions_ztest([10, 00], [106, 38])
#%%
thresh = 0.05
for msk, label in zip([validmsk & V1msk, validmsk & V4msk, validmsk & ITmsk],
                      ["V1", "V4", "IT"]):
    print(label, f"N={sum(msk)}")
    print("max > init FC", sum(msk & (meta_df.p_maxinit_0 < thresh) & (meta_df.t_maxinit_0 > 0)))
    print("max > init BG", sum(msk & (meta_df.p_maxinit_1 < thresh) & (meta_df.t_maxinit_1 > 0)))
    print("end > init FC", sum(msk & (meta_df.p_endinit_0 < thresh) & (meta_df.t_endinit_0 > 0)))
    print("end > init BG", sum(msk & (meta_df.p_endinit_1 < thresh) & (meta_df.t_endinit_1 > 0)))
    print("end < init FC", sum(msk & (meta_df.p_endinit_0 < thresh) & (meta_df.t_endinit_0 < 0)))
    print("end < init BG", sum(msk & (meta_df.p_endinit_1 < thresh) & (meta_df.t_endinit_1 < 0)))
#%%
""" Separate the success rate of between different BG spaces """
cls_msk = meta_df.space2 == "BigGAN_class"
full_msk = meta_df.space2 == "BigGAN"
for msk, label in zip([validmsk & V1msk, validmsk & V4msk, validmsk & ITmsk],
                      ["V1", "V4", "IT"]):
    print(label, f"N={sum(msk)}")
    Nclass = sum(msk & cls_msk)
    Nfull = sum(msk & full_msk)
    print(f"max > init BG class %d / {Nclass}" % sum(msk & cls_msk & (meta_df.p_maxinit_1 < 0.01) & (meta_df.t_maxinit_1 > 0)))
    print(f"max > init BG full %d / {Nfull}" % sum(msk & full_msk & (meta_df.p_maxinit_1 < 0.01) & (meta_df.t_maxinit_1 > 0)))
    print(f"end > init BG class %d / {Nclass}" % sum(msk & cls_msk & (meta_df.p_endinit_1 < 0.01) & (meta_df.t_endinit_1 > 0)))
    print(f"end > init BG full %d / {Nfull}" % sum(msk & full_msk & (meta_df.p_endinit_1 < 0.01) & (meta_df.t_endinit_1 > 0)))
    print(f"end < init BG class %d / {Nclass}" % sum(msk & cls_msk & (meta_df.p_endinit_1 < 0.01) & (meta_df.t_endinit_1 < 0)))
    print(f"end < init BG full %d / {Nfull}" % sum(msk & full_msk & (meta_df.p_endinit_1 < 0.01) & (meta_df.t_endinit_1 < 0)))

#%% Use different success criterion and save the result while plotting as curves
sucs_criterion_label, sucs_label = "max > init, P<0.01", "maxinit"
FC_suc_msk = (meta_df.p_maxinit_0 < 0.01) & (meta_df.t_maxinit_0 > 0)
BG_suc_msk = (meta_df.p_maxinit_1 < 0.01) & (meta_df.t_maxinit_1 > 0)
sucs_criterion_label, sucs_label = "end > init, P<0.01", "endinit"
FC_suc_msk = (meta_df.p_endinit_0 < 0.01) & (meta_df.t_endinit_0 > 0)
BG_suc_msk = (meta_df.p_endinit_1 < 0.01) & (meta_df.t_endinit_1 > 0)
sucs_criterion_label, sucs_label = "end < init, P<0.01", "endinitdecrs"
FC_suc_msk = (meta_df.p_endinit_0 < 0.01) & (meta_df.t_endinit_0 < 0)
BG_suc_msk = (meta_df.p_endinit_1 < 0.01) & (meta_df.t_endinit_1 < 0)
# build a dataframe of success rate for plotting
SR_df = []
for msk, label in zip([validmsk & V1msk, validmsk & V4msk, validmsk & ITmsk],
                      ["V1", "V4", "IT"]):
    SR = edict(label=label, FC_suc=sum(msk & FC_suc_msk), BG_suc=sum(msk & BG_suc_msk), total=sum(msk))
    SR["FC_rate"] = SR.FC_suc / SR.total
    SR["BG_rate"] = SR.BG_suc / SR.total
    SR["FC_CI_1"] = rate_CI(SR.total, SR.FC_suc, 0.05)
    SR["FC_CI_2"] = rate_CI(SR.total, SR.FC_suc, 0.95)
    SR["BG_CI_1"] = rate_CI(SR.total, SR.BG_suc, 0.05)
    SR["BG_CI_2"] = rate_CI(SR.total, SR.BG_suc, 0.95)
    print(SR)
    SR_df.append(SR)
SR_df = pd.DataFrame(SR_df)
SR_df = SR_df.set_index("label")
# set dtypes
SR_df = SR_df.astype({"FC_suc": int, "BG_suc": int, "total": int, "FC_rate": float, "BG_rate": float})
SR_df.to_csv(join(outdir, f"SuccessRate_{sucs_label}.csv"))
#%% Plot the success rate
fig, ax = plt.subplots(1, 1, figsize=[3.5, 3.3])
ax.plot(SR_df.index, SR_df.FC_rate, "o-", label="DeePSim", color="b")#"tab:blue"
ax.plot(SR_df.index, SR_df.BG_rate, "o-", label="BigGAN", color="r")#"tab:red"
ax.fill_between(SR_df.index, SR_df.FC_CI_1, SR_df.FC_CI_2, alpha=0.3, color="b")#"tab:blue"
ax.fill_between(SR_df.index, SR_df.BG_CI_1, SR_df.BG_CI_2, alpha=0.3, color="r")#"tab:red"
ax.set_ylim([0, 1.05])
ax.set_ylabel("Success Rate")
ax.set_xlabel("Visual Area")
# annotate the number of trials as fraction
for i, (label, row) in enumerate(SR_df.iterrows()):
    ax.annotate(f"{int(row.FC_suc)}/{int(row.total)}",
                xy=(i + 0.02, min(1.0, row.FC_rate+0.04)),
                ha="center", va="bottom", fontsize=10)
    ax.annotate(f"{int(row.BG_suc)}/{int(row.total)}",
                xy=(i + 0.02, max(0.0, row.BG_rate-0.06)),
                ha="center", va="bottom", fontsize=10)

ax.legend()
fig.suptitle(f"Success Rate of DeePSim and BigGAN\n{sucs_criterion_label} 90% CI")
fig.tight_layout()
saveallforms(outdir, f"evol_{sucs_label}_success_rate_per_area_annot", fig, fmts=["png", "pdf", "svg"])
fig.show()



#%%
"""separated by visual areas and GAN Space"""
sucs_criterion_label, sucs_label = "max > init, P<0.01", "maxinit"
FC_suc_msk = (meta_df.p_maxinit_0 < 0.01) & (meta_df.t_maxinit_0 > 0)
BG_suc_msk = (meta_df.p_maxinit_1 < 0.01) & (meta_df.t_maxinit_1 > 0)
# sucs_criterion_label, sucs_label = "end > init, P<0.01", "endinit"
# FC_suc_msk = (meta_df.p_endinit_0 < 0.01) & (meta_df.t_endinit_0 > 0)
# BG_suc_msk = (meta_df.p_endinit_1 < 0.01) & (meta_df.t_endinit_1 > 0)
# sucs_criterion_label, sucs_label = "end < init, P<0.01", "endinitdecrs"
# FC_suc_msk = (meta_df.p_endinit_0 < 0.01) & (meta_df.t_endinit_0 < 0)
# BG_suc_msk = (meta_df.p_endinit_1 < 0.01) & (meta_df.t_endinit_1 < 0)
# build a dataframe of success rate for plotting
SR_df = []
BG_class_msk = meta_df.space2 == "BigGAN_class"
BG_all_msk = meta_df.space2 == "BigGAN"
for msk, label in zip([validmsk & V1msk & BG_class_msk, validmsk & V1msk & BG_all_msk,
                       validmsk & V4msk & BG_class_msk, validmsk & V4msk & BG_all_msk,
                       validmsk & ITmsk & BG_class_msk, validmsk & ITmsk & BG_all_msk,],
                      ["V1 class", "V1 all", "V4 class", "V4 all", "IT class", "IT all",]):
    SR = edict(label=label, FC_suc=sum(msk & FC_suc_msk), BG_suc=sum(msk & BG_suc_msk), total=sum(msk))
    SR["FC_rate"] = SR.FC_suc / SR.total
    SR["BG_rate"] = SR.BG_suc / SR.total
    SR["FC_CI_1"] = rate_CI(SR.total, SR.FC_suc, 0.05)
    SR["FC_CI_2"] = rate_CI(SR.total, SR.FC_suc, 0.95)
    SR["BG_CI_1"] = rate_CI(SR.total, SR.BG_suc, 0.05)
    SR["BG_CI_2"] = rate_CI(SR.total, SR.BG_suc, 0.95)
    print(SR)
    SR_df.append(SR)
SR_df = pd.DataFrame(SR_df)
SR_df = SR_df.set_index("label")
# set dtypes
SR_df = SR_df.astype({"FC_suc": int, "BG_suc": int, "total": int, "FC_rate": float, "BG_rate": float})
SR_df.to_csv(join(outdir, f"SuccessRate_space_split_{sucs_label}.csv"))

#%%
"""separated by visual areas and GAN Space"""
sucs_criterion_label, sucs_label = "max > init, P<0.01", "maxinit"
FC_suc_msk = (meta_df.p_maxinit_0 < 0.01) & (meta_df.t_maxinit_0 > 0)
BG_suc_msk = (meta_df.p_maxinit_1 < 0.01) & (meta_df.t_maxinit_1 > 0)
# sucs_criterion_label, sucs_label = "end > init, P<0.01", "endinit"
# FC_suc_msk = (meta_df.p_endinit_0 < 0.01) & (meta_df.t_endinit_0 > 0)
# BG_suc_msk = (meta_df.p_endinit_1 < 0.01) & (meta_df.t_endinit_1 > 0)
# sucs_criterion_label, sucs_label = "end < init, P<0.01", "endinitdecrs"
# FC_suc_msk = (meta_df.p_endinit_0 < 0.01) & (meta_df.t_endinit_0 < 0)
# BG_suc_msk = (meta_df.p_endinit_1 < 0.01) & (meta_df.t_endinit_1 < 0)
# build a dataframe of success rate for plotting
SR_df = []
# BG_cmshess_msk = meta_df.optim_names2 != 'CMAES'
# BG_cma_msk = meta_df.optim_names2 == 'CMAES'
BG_cmshess_msk = meta_df.optim_names1 != 'CMAES'
BG_cma_msk = meta_df.optim_names1 == 'CMAES'
for msk, label in zip([validmsk & V1msk & BG_cmshess_msk, validmsk & V1msk & BG_cma_msk,
                       validmsk & V4msk & BG_cmshess_msk, validmsk & V4msk & BG_cma_msk,
                       validmsk & ITmsk & BG_cmshess_msk, validmsk & ITmsk & BG_cma_msk,],
                      ["V1 HessCMA", "V1 CMA", "V4 HessCMA", "V4 CMA", "IT HessCMA", "IT CMA",]):
    SR = edict(label=label, FC_suc=sum(msk & FC_suc_msk), BG_suc=sum(msk & BG_suc_msk), total=sum(msk))

    SR["FC_rate"] = SR.FC_suc / SR.total if SR.total > 0 else np.nan
    SR["BG_rate"] = SR.BG_suc / SR.total if SR.total > 0 else np.nan
    SR["FC_CI_1"] = rate_CI(SR.total, SR.FC_suc, 0.05)
    SR["FC_CI_2"] = rate_CI(SR.total, SR.FC_suc, 0.95)
    SR["BG_CI_1"] = rate_CI(SR.total, SR.BG_suc, 0.05)
    SR["BG_CI_2"] = rate_CI(SR.total, SR.BG_suc, 0.95)
    print(SR)
    SR_df.append(SR)
SR_df = pd.DataFrame(SR_df)
SR_df = SR_df.set_index("label")
# set dtypes
SR_df = SR_df.astype({"FC_suc": int, "BG_suc": int, "total": int, "FC_rate": float, "BG_rate": float})
SR_df
# SR_df.to_csv(join(outdir, f"SuccessRate_optimizer_split_{sucs_label}.csv"))

#%%
fig, ax = plt.subplots(1, 1, figsize=[3.5, 3.3])
ax.plot(SR_df.index, SR_df.FC_rate, "o-", label="DeePSim", color="b")#"tab:blue"
ax.plot(SR_df.index, SR_df.BG_rate, "o-", label="BigGAN", color="r")#"tab:red"
ax.fill_between(SR_df.index, SR_df.FC_CI_1, SR_df.FC_CI_2, alpha=0.3, color="b")#"tab:blue"
ax.fill_between(SR_df.index, SR_df.BG_CI_1, SR_df.BG_CI_2, alpha=0.3, color="r")#"tab:red"
ax.set_ylim([0, 1.05])
ax.set_ylabel("Success Rate")
ax.set_xlabel("Visual Area")
# annotate the number of trials as fraction
for i, (label, row) in enumerate(SR_df.iterrows()):
    ax.annotate(f"{int(row.FC_suc)}/{int(row.total)}",
                xy=(i + 0.02, min(1.0, row.FC_rate+0.04)),
                ha="center", va="bottom", fontsize=10)
    ax.annotate(f"{int(row.BG_suc)}/{int(row.total)}",
                xy=(i + 0.02, max(0.0, row.BG_rate-0.06)),
                ha="center", va="bottom", fontsize=10)

ax.legend()
fig.suptitle(f"Success Rate of DeePSim and BigGAN\n{sucs_criterion_label} 90% CI")
fig.tight_layout()
saveallforms(outdir, f"evol_{sucs_label}_success_rate_per_space_per_area_annot", fig, fmts=["png", "pdf", "svg"])
fig.show()
#%%
# same plot but with bar and errorbar
fig, ax = plt.subplots(1, 1, figsize=[4.0, 3.75])
ax.bar(SR_df.index, SR_df.FC_rate, yerr=[SR_df.FC_rate - SR_df.FC_CI_1, SR_df.FC_CI_2 - SR_df.FC_rate],
         label="DeePSim", color="b", alpha=0.5, error_kw=dict(ecolor='b', capsize=5))
# add points
ax.plot(SR_df.index, SR_df.FC_rate, "o", color="b", alpha=0.7)
ax.bar(SR_df.index, SR_df.BG_rate, yerr=[SR_df.BG_rate - SR_df.BG_CI_1, SR_df.BG_CI_2 - SR_df.BG_rate],
            label="BigGAN", color="r", alpha=0.5, error_kw=dict(ecolor='r', capsize=5))
ax.plot(SR_df.index, SR_df.BG_rate, "o", color="r", alpha=0.7)
# rename the xticks
ax.set_xticklabels(["V1\nclass", "V1\nall", "V4\nclass", "V4\nall", "IT\nclass", "IT\nall"])
ax.set_ylim([0, 1.05])
ax.set_ylabel("Success Rate")
ax.set_xlabel("Visual Area")
ax.set_title(f"Success Rate of DeePSim and BigGAN\n{sucs_criterion_label} 90% CI")
plt.tight_layout()
saveallforms(outdir, f"evol_{sucs_label}_success_rate_per_space_per_area_bar", fig, fmts=["png", "pdf", "svg"])
fig.show()

#%%
# same plot but with bar and errorbar
fig, ax = plt.subplots(1, 1, figsize=[4.0, 3.75])
ax.bar(SR_df.index, SR_df.FC_rate, yerr=[SR_df.FC_rate - SR_df.FC_CI_1, SR_df.FC_CI_2 - SR_df.FC_rate],
         label="DeePSim", color="b", alpha=0.5, error_kw=dict(ecolor='b', capsize=5))
# add points
ax.plot(SR_df.index, SR_df.FC_rate, "o", color="b", alpha=0.7)
ax.bar(SR_df.index, SR_df.BG_rate, yerr=[SR_df.BG_rate - SR_df.BG_CI_1, SR_df.BG_CI_2 - SR_df.BG_rate],
            label="BigGAN", color="r", alpha=0.5, error_kw=dict(ecolor='r', capsize=5))
ax.plot(SR_df.index, SR_df.BG_rate, "o", color="r", alpha=0.7)
# annotate the number of trials as fraction
for i, (label, row) in enumerate(SR_df.iterrows()):
    ax.annotate(f"{int(row.FC_suc)}/{int(row.total)}",
                xy=(i + 0.02, min(1.0, row.FC_rate+0.05)),
                ha="center", va="bottom", fontsize=10)
    ax.annotate(f"{int(row.BG_suc)}/{int(row.total)}",
                xy=(i + 0.02, max(0.0, row.BG_rate-0.08)),
                ha="center", va="bottom", fontsize=10)
# rename the xticks
ax.set_xticklabels(["V1\nclass", "V1\nall", "V4\nclass", "V4\nall", "IT\nclass", "IT\nall"])
ax.set_ylim([0, 1.05])
ax.set_ylabel("Success Rate")
ax.set_xlabel("Visual Area")
ax.set_title(f"Success Rate of DeePSim and BigGAN\n{sucs_criterion_label} 90% CI")
plt.tight_layout()
saveallforms(outdir, f"evol_{sucs_label}_success_rate_per_space_per_area_bar_annot", fig, fmts=["png", "pdf", "svg"])
fig.show()

#%%
# same plot but with bar and errorbar
fig, ax = plt.subplots(1, 1, figsize=[3.0, 3.75])
ax.bar(SR_df.index, SR_df.FC_rate, yerr=[SR_df.FC_rate - SR_df.FC_CI_1, SR_df.FC_CI_2 - SR_df.FC_rate],
         label="DeePSim", color="b", alpha=0.5, error_kw=dict(ecolor='b', capsize=5))
# add points
ax.plot(SR_df.index, SR_df.FC_rate, "o", color="b", alpha=0.7)
ax.bar(SR_df.index, SR_df.BG_rate, yerr=[SR_df.BG_rate - SR_df.BG_CI_1, SR_df.BG_CI_2 - SR_df.BG_rate],
            label="BigGAN", color="r", alpha=0.5, error_kw=dict(ecolor='r', capsize=5))
ax.plot(SR_df.index, SR_df.BG_rate, "o", color="r", alpha=0.7)
# annotate the number of trials as fraction
for i, (label, row) in enumerate(SR_df.iterrows()):
    ax.annotate(f"{int(row.FC_suc)}/{int(row.total)}",
                xy=(i + 0.02, min(1.0, row.FC_rate+0.05)),
                ha="center", va="bottom", fontsize=10)
    ax.annotate(f"{int(row.BG_suc)}/{int(row.total)}",
                xy=(i + 0.02, max(0.0, row.BG_rate-0.08)),
                ha="center", va="bottom", fontsize=10)
# rename the xticks
ax.set_xticklabels(["V1\nclass", "V1\nall", "V4\nclass", "V4\nall", "IT\nclass", "IT\nall"])
ax.set_ylim([0, 1.05])
ax.set_xlim([1.5, 5.5])
ax.set_ylabel("Success Rate")
ax.set_xlabel("Visual Area")
ax.set_title(f"Success Rate of DeePSim and BigGAN\n{sucs_criterion_label} 90% CI")
plt.tight_layout()
saveallforms(outdir, f"evol_{sucs_label}_success_rate_per_space_per_area_bar_annot_noV1", fig, fmts=["png", "pdf", "svg"])
fig.show()

#%%
from statsmodels.stats.proportion import proportions_ztest
# count: number of successes, i.e., A1 and A2
count = np.array([48, 17])
# nobs: number of trials, i.e., B1 and B2
nobs = np.array([83, 23])
# conduct the two-proportion z-test
z, p = proportions_ztest(count, nobs)
# print the z-score and p-value
print(f'z-score: {z:.2f}, p-value: {p:.4f}')
#%% Older code
BFEStats_merge, BFEStats = load_neural_data()
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
    # meta_dict = get_meta_dict(S)
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

df = pd.DataFrame(success_col)
#%%
df["endinit_success_BG"] = df["endinit_pval_BG"] < 0.01
df["endinit_success_FC"] = df["endinit_pval_FC"] < 0.01
df.groupby("visual_area")[["endinit_success_BG", "endinit_success_FC"]].mean()
sumtable = df.groupby("visual_area")[["endinit_success_BG", "endinit_success_FC"]].mean()

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


#%% dev zone
stats_sumdf = meta_df[validmsk].groupby("visual_area").agg(
               {"max>init FC": aggfunclist,
                "max>init BG": aggfunclist,
                "max>init both": aggfunclist,
                "end>init FC": aggfunclist,
                "end>init BG": aggfunclist,
                "end>init both": aggfunclist,
                "end<init FC": aggfunclist,
                "end<init BG": aggfunclist,
                "end<init both": aggfunclist,
                "max>end FC": aggfunclist,
                "max>end BG": aggfunclist,
                "max>end both": aggfunclist,
                }).loc[["V1", "V4", "IT"]]
#%%
total_sumdf = meta_df[validmsk].agg(
               {"max>init FC": aggfunclist,
                "max>init BG": aggfunclist,
                "max>init both": aggfunclist,
                "end>init FC": aggfunclist,
                "end>init BG": aggfunclist,
                "end>init both": aggfunclist,
                "end<init FC": aggfunclist,
                "end<init BG": aggfunclist,
                "end<init both": aggfunclist,
                "max>end FC": aggfunclist,
                "max>end BG": aggfunclist,
                "max>end both": aggfunclist,
                })
total_sumdf_row = total_sumdf.melt(ignore_index=False).T
total_sumdf_row = total_sumdf_row.drop("variable").rename(index={"value": "Total"})
total_sumdf_row.columns = pd.MultiIndex.from_product([total_sumdf.columns, ['sum', 'mean', '<lambda_0>', '<lambda_1>', 'count']])
# set dtype sum count to int
total_sumdf_row.loc[:, (slice(None), ["sum", "count"])] = total_sumdf_row.loc[:, (slice(None), ["sum", "count"])].astype(int)
# turn the rownames into multiindex of columns
# total_sumdf_row.index = pd.MultiIndex.from_product([['Total'], total_sumdf.index])
# Adjust columns to multi-index
stats_sumdf = pd.concat([stats_sumdf, total_sumdf_row])  #.to_csv(join(tabdir, f"Evol_success_rate_{thresh:.3f}.csv"))

