"""
Plot the success rate of Evolution and estimates its error bar for different cortices.
"""
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


# rate_CI(10, 1, [0.1, 0.9])
#%%
from neuro_data_analysis.neural_data_utils import parse_meta, area_mapping, get_all_masks, \
    get_meta_df, get_meta_dict

# meta_df = get_meta_df(BFEStats)
outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\SuccessRate"
dfdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_activation_cmp"
meta_df = pd.read_csv(join(dfdir, "meta_stats.csv"))
#%%
Amsk, Bmsk, V1msk, V4msk, ITmsk, \
    length_msk, spc_msk, sucsmsk, \
    bsl_unstable_msk, bsl_stable_msk, validmsk = get_all_masks(meta_df)

#%%
for msk, label in zip([validmsk & V1msk, validmsk & V4msk, validmsk & ITmsk],
                      ["V1", "V4", "IT"]):
    print(label, f"N={sum(msk)}")
    print("max > init FC", sum(msk & (meta_df.p_maxinit_0 < 0.01) & (meta_df.t_maxinit_0 > 0)))
    print("max > init BG", sum(msk & (meta_df.p_maxinit_1 < 0.01) & (meta_df.t_maxinit_1 > 0)))
    print("end > init FC", sum(msk & (meta_df.p_endinit_0 < 0.01) & (meta_df.t_endinit_0 > 0)))
    print("end > init BG", sum(msk & (meta_df.p_endinit_1 < 0.01) & (meta_df.t_endinit_1 > 0)))
    print("end < init FC", sum(msk & (meta_df.p_endinit_0 < 0.01) & (meta_df.t_endinit_0 < 0)))
    print("end < init BG", sum(msk & (meta_df.p_endinit_1 < 0.01) & (meta_df.t_endinit_1 < 0)))

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
import numpy as np
import scipy.stats


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
