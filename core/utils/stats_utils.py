import numpy as np


def summary_by_block(scores_vec, gens, maxgen=100, sem=True):
    """Summarize a score trajectory and and generation vector into the mean vector, sem, """
    genmin = min(gens)
    genmax = max(gens)
    if maxgen is not None:
        genmax = min(maxgen, genmax)

    score_m = []
    score_s = []
    blockarr = []
    for geni in range(genmin, genmax + 1):
        score_block = scores_vec[gens == geni]
        if len(score_block) == 1:
            continue
        score_m.append(np.mean(score_block))
        if sem:
            score_s.append(np.std(score_block) / np.sqrt(len(score_block)))
        else:
            score_s.append(np.std(score_block))
        blockarr.append(geni)
    score_m = np.array(score_m)
    score_s = np.array(score_s)
    blockarr = np.array(blockarr)
    return score_m, score_s, blockarr


from scipy.stats import pearsonr, ttest_rel, ttest_ind, ttest_1samp
def ttest_rel_df(df, msk, col1, col2):
    return ttest_rel(df[msk][col1], df[msk][col2])


def ttest_ind_df(df, msk1, msk2, col):
    return ttest_ind(df[msk1][col], df[msk2][col])


def ttest_1samp_print(seq, scalar):
    tval, pval = ttest_1samp(seq, scalar)
    print(f"{seq.mean():.3f}+-{seq.std():.3f} ~ {scalar:.3f} tval: {tval:.2f}, pval: {pval:.1e}")
    return tval, pval


def ttest_rel_print(seq1, seq2):
    tval, pval = ttest_rel(seq1, seq2)
    print(f"{seq1.mean():.3f}+-{seq1.std():.3f} ~ {seq2.mean():.3f}+-{seq2.std():.3f} (N={len(seq1)})tval: {tval:.2f}, pval: {pval:.1e}")
    return tval, pval


def ttest_ind_print(seq1, seq2):
    tval, pval = ttest_ind(seq1, seq2, nan_policy="omit")
    print(f"{seq1.mean():.3f}+-{seq1.std():.3f} (N={len(seq1)}) ~ {seq2.mean():.3f}+-{seq2.std():.3f} (N={len(seq2)}) tval: {tval:.2f}, pval: {pval:.1e}")
    return tval, pval


def ttest_rel_print_df(df, msk, col1, col2):
    if msk is None:
        msk = np.ones(len(df), dtype=bool)
    print(f"{col1} ~ {col2} (N={msk.sum()})", end=" ")
    return ttest_rel_print(df[msk][col1], df[msk][col2])


def ttest_ind_print_df(df, msk1, msk2, col):
    print(f"{col} (N={msk1.sum()}) ~ (N={msk2.sum()})", end=" ")
    return ttest_ind_print(df[msk1][col], df[msk2][col])


import matplotlib.pyplot as plt
def paired_strip_plot(df, msk, col1, col2):
    if msk is None:
        msk = np.ones(len(df), dtype=bool)
    vec1 = df[msk][col1]
    vec2 = df[msk][col2]
    xjitter = 0.1 * np.random.randn(len(vec1))
    figh = plt.figure(figsize=[5, 6])
    plt.scatter(xjitter, vec1)
    plt.scatter(xjitter+1, vec2)
    plt.plot(np.arange(2)[:,None]+xjitter[None,:],
             np.stack((vec1, vec2)), color="k", alpha=0.1)
    plt.xticks([0,1], [col1, col2])
    tval, pval = ttest_rel_df(df, msk, col1, col2)
    plt.title(f"tval={tval:.3f}, pval={pval:.1e} N={msk.sum()}")
    plt.show()
    return figh


def trivariate_corr(x, y, z):
    """
    x = np.random.normal(0, 1, size=100)
    y = np.random.normal(0, 1, size=100)
    z = np.random.normal(0, 1, size=100)

    r = trivariate_corr(x, y, z)
    print(r)
    :param x:
    :param y:
    :param z:
    :return:
    """
    xy_corr = np.corrcoef(x, y)[0, 1]
    xz_corr = np.corrcoef(x, z)[0, 1]
    yz_corr = np.corrcoef(y, z)[0, 1]
    r = (xy_corr**2 + xz_corr**2 + yz_corr**2 + 2*xy_corr*xz_corr*yz_corr) / (1 - xy_corr**2 - xz_corr**2- yz_corr**2 + 2*xy_corr*xz_corr*yz_corr)
    return r