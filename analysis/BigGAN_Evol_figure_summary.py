"""BigGAN Evol Figure
Partially inherit from BigGAN_Evol_summary.py but more focused. 
"""
import os
import re
from time import time
from glob import glob
from os.path import join
from easydict import EasyDict
from imageio import imread
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pylab as plt
from scipy.stats import ttest_rel, ttest_ind
from PIL import Image
from core.utils import make_grid_np, saveallforms, crop_from_montage
from stats_utils import summary_by_block
def load_data_from_row(row, imgid=-1):
    npzpath = join(dataroot, row.unitstr, 'scores%s_%05d.npz' % (row.optimizer, row.RND))
    imgtrajpath = glob(join(dataroot, row.unitstr, "besteachgen%s_%05d.jpg" % (row.optimizer, row.RND)))[0]
    data = np.load(npzpath)
    evolmtg = imread(imgtrajpath)
    proto = crop_from_montage(evolmtg, imgid=imgid)
    scorevec = data['scores_all']
    genvec = data["generations"]
    score_m, score_s, blockarr = summary_by_block(scorevec, genvec, sem=False)
    return scorevec, genvec, score_m, score_s, blockarr, proto


def shadedErrorbar(blockarr, score_m, score_s, alpha=0.2, linealpha=1.0, label=None, color=None, linecolor=None):
    L = plt.plot(blockarr, score_m, label=label, c=linecolor, alpha=linealpha)
    if color is None: color = L[0]._color
    plt.fill_between(blockarr, score_m-score_s, score_m+score_s, alpha=alpha, color=color)


pd.set_option('display.width', 200)
pd.set_option("max_colwidth", 60)
pd.set_option('display.max_columns', None)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

#%% Data using alexnet
dataroot = r"E:\Cluster_Backup\BigGAN_Optim_Tune_new"
figdir = r"O:\BigGAN_FC6_insilico"
outdir = r"O:\ThesisProposal\BigGAN"
summarydir = join(figdir, "summary")
os.makedirs(summarydir, exist_ok=True)

# load the table that indexing all experiments
exptab = pd.read_csv(join(summarydir, "optim_raw_score_tab.csv"))
exptab.suffix.fillna("", inplace=True)  # substitute nan as ""
exptab.suffix = exptab.suffix.astype(str)
rfmsk = exptab.layer.str.contains("fc") | exptab.suffix.str.contains("RF")  # Experiments that do RFfitting
fullmsk = exptab.layer.str.contains("fc") | ~exptab.suffix.str.contains("RF")  # Experiments that use full size images
# two masks are overlapping.
#%%
row = exptab.loc[25]
npzpath = join(dataroot, row.unitstr, 'scores%s_%05d.npz'%(row.optimizer, row.RND))
imgtrajpath = glob(join(dataroot, row.unitstr, "besteachgen%s_%05d.jpg"%(row.optimizer, row.RND)))[0]
data = np.load(npzpath)
evolmtg = imread(imgtrajpath)
proto = crop_from_montage(evolmtg)

#%%
for layer in exptab.layer.unique():
    unit_cols = exptab.unitstr[(exptab.layer==layer) & rfmsk].unique()
    # for unitstr in unit_cols:
    unitstr = np.random.choice(unit_cols, 1)[0]
unitstr = "alexnet_conv5_33_RFrsz"
for optim in ['CholCMA', 'HessCMA', 'CholCMA_fc6']:
    msk = (exptab.unitstr==unitstr) & (exptab.optimizer==optim) & rfmsk
    print(layer, optim, unitstr, exptab[msk].shape)
    # row = exptab[msk].sample(1).iloc[0]
#%% Plot the optim trajectory comparison!
# unitstr = "alexnet_conv5_31_RFrsz"
# unitstr = "alexnet_conv1_33_RFrsz"

unitstrs = ["alexnet_conv2_30_RFrsz",
            "alexnet_conv3_32_RFrsz",
            "alexnet_conv4_31_RFrsz",
            #"alexnet_conv5_30_RFrsz",
            "alexnet_conv5_33_RFrsz",
            #"alexnet_fc6_30",
            "alexnet_fc6_32",
            "alexnet_fc7_33",
            "alexnet_fc8_31",]
mtg_col = []
ncol = len(unitstrs)
figh, axs = plt.subplots(1, ncol, figsize=[ncol*2.75, 2.75])
for ci, unitstr in enumerate(unitstrs):
    proto_col = []
    for optim in ['CholCMA', 'HessCMA', 'CholCMA_fc6']:
        msk = (exptab.unitstr == unitstr) & (exptab.optimizer == optim)# & rfmsk
        if "fc6" not in optim: optim+="_BigGAN"
        print(layer, optim, unitstr, exptab[msk].shape)
        # row = exptab[msk].sample(1).iloc[0]
        maxid = exptab[msk].score.argmax()
        row = exptab[msk].loc[maxid]
        _, _, score_m, score_s, blockarr, proto = load_data_from_row(row)
        proto_col.append(proto)
        plt.sca(axs[ci])
        shadedErrorbar(blockarr, score_m, score_s, label=optim)

    plt.title(unitstr)
    if ci == 0: plt.legend()
    mtg = make_grid_np(proto_col, nrow=1)  # np.stack(tuple(proto_col), axis=3)
    mtg_col.append(mtg)
saveallforms([figdir, outdir], "best_trajs_exemplar_alllayer")
plt.show()
mtg_full = make_grid_np(mtg_col, nrow=ncol)
mtg_PIL = Image.fromarray(mtg_full)
mtg_PIL.show()
mtg_PIL.save(join(figdir, "proto_cmp_alllayers.jpg"))
mtg_PIL.save(join(outdir, "proto_cmp_alllayers.jpg"))

#%% New data sourse using ResNet50-robust
sumdir = r"E:\Cluster_Backup\GAN_Evol_cmp\summary"
rootdir = r"E:\Cluster_Backup\GAN_Evol_cmp"
expdirs = os.listdir(rootdir)
expdirs = [*filter(lambda nm: "resnet50_linf" in nm, expdirs)]
# re.findall("resnet50_linf_8_([^_]*)_(\d*)_(\d*)_(\d*)(_RFrsz|)", "resnet50_linf_8_.layer4.Bottleneck2_46_4_4_RFrsz")
# "scoresCholCMA_93259.npz"
exp_col = []
trial_col = []
for expdir in expdirs:
    unit_tup = expdir.split("resnet50_linf_8_")[1].split("_")
    do_resize = ("RFrsz" in unit_tup)
    if do_resize: unit_tup = unit_tup[:-1]
    if len(unit_tup) == 4:
        layer, chan, xid, yid = unit_tup[0], int(unit_tup[1]), int(unit_tup[2]), int(unit_tup[3])
    elif len(unit_tup) == 2:
        layer, chan, xid, yid = unit_tup[0], int(unit_tup[1]), None, None
    else:
        raise ValueError("unit parsing error for %s"%expdir)
    exp_col.append((expdir, layer, chan, xid, yid, do_resize))
    imgtrajpaths = [*map(os.path.basename, glob(join(rootdir, expdir, "traj*.jpg")))]
    for trialnm in imgtrajpaths:
        patt = re.findall("traj(.*)_(\d*)_score([\d.]*).jpg", trialnm)
        if len(patt) == 0:
            raise ValueError(trialnm)
        optimstr, RND, score = patt[0][0], int(patt[0][1]), float(patt[0][2])
        GANstr = "fc6" if "fc6" in optimstr else "BigGAN" 
        trial_col.append((expdir, layer, chan, xid, yid, do_resize, optimstr, GANstr, RND, score))

unit_tab = pd.DataFrame(exp_col, columns=["expdir", "layer", "chan", "xid", "yid", "RFrsz", ])
exptab = pd.DataFrame(trial_col, columns=["expdir", "layer", "chan", "xid", "yid", "RFrsz", "optimstr", "GANstr", "RND", "score", ])
unit_tab.to_csv(join(sumdir, "unit_tab.csv"))
exptab.to_csv(join(sumdir, "trial_tab.csv"))


#%% Fancy pandas way to do trial averaging quick 
exptab_trial_m = exptab.groupby(["expdir", "optimstr"]).mean()
exptab_trial_m = exptab_trial_m.reset_index()
exptab_trial_mW = exptab_trial_m.pivot(index='expdir', columns='optimstr', values='score')
unit_score_tab = unit_tab.copy()
for optim in ["CholCMA", "HessCMA", "HessCMA500_fc6"]:
    unit_score_tab[optim] = np.nan

for ri, row in unit_score_tab.iterrows():
    for optim in ["CholCMA", "HessCMA", "HessCMA500_fc6"]:
        unit_score_tab[optim][ri] = exptab_trial_mW.loc[row.expdir][optim]

maxscore = unit_score_tab[["CholCMA", "HessCMA", "HessCMA500_fc6"]].max(axis=1, skipna=True)
for optimstr in ["CholCMA", "HessCMA", "HessCMA500_fc6"]:
    unit_score_tab[optimstr+"_norm"] = unit_score_tab[optimstr] / maxscore

unit_score_tab.to_csv(join(sumdir, "unit_score_tab.csv"))
#%%
# Melt is the wide to long transform, making each optimizer a row in the table
unit_score_tab_L = unit_score_tab.melt(id_vars=["layer","chan","xid","yid","RFrsz", "expdir"],
                    value_vars=["CholCMA", "HessCMA", "HessCMA500_fc6"], var_name="optimstr", value_name="score")
RFrsz_msk = unit_score_tab_L.RFrsz | (unit_score_tab_L.layer == ".Linearfc")
figh = plt.figure(figsize=[7, 6])
sns.violinplot(x='layer', y='score', hue="optimstr", jitter=0.25,
               hue_order=['CholCMA', 'HessCMA', 'HessCMA500_fc6'], cut=0.1,
               data=unit_score_tab_L[RFrsz_msk], alpha=0.4)
plt.xticks(rotation=20)
figh.savefig(join(sumdir, "resnet_linf8_raw_score_cmp_RFresize.png"))
figh.savefig(join(sumdir, "resnet_linf8_raw_score_cmp_RFresize.pdf"))
plt.show()
#%%
# Melt is the wide to long transform, making each optimizer a row in the table
unit_score_tab_norm_L = unit_score_tab.melt(id_vars=["layer","chan","xid","yid","RFrsz", "expdir"],
                    value_vars=["CholCMA_norm", "HessCMA_norm", "HessCMA500_fc6_norm"],
                    var_name="optimstr", value_name="norm_score")
RFrsz_msk = unit_score_tab_norm_L.RFrsz | (unit_score_tab_norm_L.layer == ".Linearfc")
figh = plt.figure(figsize=[7, 6])
ax = sns.violinplot(x='layer', y='norm_score', hue="optimstr", jitter=0.1, width=0.7, scale="width",
               hue_order=['CholCMA_norm', 'HessCMA_norm', 'HessCMA500_fc6_norm'], cut=0.1,
               data=unit_score_tab_norm_L[RFrsz_msk], alpha=0.2)
plt.xticks(rotation=20)
figh.savefig(join(sumdir, "resnet_linf8_norm_score_cmp_RFresize.png"))
figh.savefig(join(sumdir, "resnet_linf8_norm_score_cmp_RFresize.pdf"))
plt.show()


#%% Visualize the prototypes as showed by different GANs
# from build_montages import make_grid_np
from PIL import Image
from tqdm import tqdm
def plot_scoremap(score_mat, expdir=""):
    plt.figure(figsize=[6,5])
    ax = sns.heatmap(score_mat, annot=True, fmt=".1f",
                     xticklabels=['CholCMA', 'HessCMA', 'HessCMA500_fc6'], )
    plt.ylabel("Trials")
    plt.xlabel("Optimizers")
    plt.axis("image")
    plt.title(expdir+"\nScore map with BigGAN or FC6")
    plt.savefig(join(sumdir, "proto_cmp", "%s_scoremat.jpg" % expdir))
    # plt.show()

for expdir in tqdm(unit_tab.expdir[845:]):
    trial_rows = exptab[exptab.expdir == expdir]
    score_mat = np.zeros((3, 3), dtype=float)
    proto_list = []
    for opti, optim in enumerate(['CholCMA', 'HessCMA', 'HessCMA500_fc6']):
        rows_w_optim = trial_rows[trial_rows.optimstr==optim]
        trN = rows_w_optim.shape[0]
        for itr in range(3):
            if itr < trN:
                row = rows_w_optim.iloc[itr]
                score_mat[itr, opti] = row.score
                # imgnm = "lastgen%s_%05d_score%.1f.jpg"%(row.optimstr, row.RND, row.score)
                # imgfp = join(rootdir, row.expdir, imgnm)
                # assert os.path.exists(imgfp)
                # proto = crop_from_montage(plt.imread(imgfp), 2)
                # proto_list.append(proto)
            else:
                score_mat[itr, opti] = np.nan
                # proto_list.append(np.zeros((256, 256, 3), dtype=np.uint8))

    # mtg = make_grid_np(proto_list, nrow=3, padding=8, rowfirst=False)
    # Image.fromarray(mtg).save(join(sumdir, "proto_cmp", "%s_proto.jpg"%expdir))
    plot_scoremap(score_mat, expdir=expdir)
#%%


#%%
# compare performance across optimizers.
# BigGAN FC6 pair alignment
optimnames = ["CholCMA", "HessCMA", "CholCMA_fc6"]
def BigGANFC6_comparison_plot(norm_scheme="allmax", rffit=True):
    Scol = []
    overallmsk = rfmsk if rffit else fullmsk 
    unitstr_uniq = exptab.loc[overallmsk].unitstr.unique()
    for unitstr in unitstr_uniq:
        unitmsk = (exptab.unitstr == unitstr)
        unit = exptab.unit[unitmsk].iloc[0]
        layer = exptab.layer[unitmsk].iloc[0]
        unitfc6msk = unitmsk & overallmsk & (exptab.optimizer=="CholCMA_fc6")
        unitBGmsk = unitmsk & overallmsk & (exptab.optimizer=="CholCMA")
        unitBGHmsk = unitmsk & overallmsk & (exptab.optimizer=="HessCMA")
        unitoptimmsk = unitmsk & overallmsk & ((exptab.optimizer=="CholCMA")\
                                            |  (exptab.optimizer=="HessCMA")\
                                            |  (exptab.optimizer=="CholCMA_fc6"))
        if any([sum(unitfc6msk)==0, sum(unitBGmsk)==0, sum(unitBGHmsk)==0]): continue
        if norm_scheme is "allmax":
            normalizer = exptab.score[unitoptimmsk].max()
        elif norm_scheme is "fc6max":
            normalizer = exptab.score[unitfc6msk].max()
        elif norm_scheme is "fc6mean":
            normalizer = exptab.score[unitfc6msk].mean()
        else: 
            raise NotImplementedError
         # no fc6 to normalize to
        for optim in optimnames:
            msk = unitmsk & overallmsk & (exptab.optimizer==optim)
            scorevec = exptab.score[msk]
            scorevec_norm = scorevec / normalizer
            newdicts = [{"layer":layer,"unit":unit,"optimizer":optim,"score":score,"score_norm":score_norm}
                        for score, score_norm in zip(scorevec, scorevec_norm)]
            Scol.extend(newdicts)

    BigGANFC6cmptab = pd.DataFrame(Scol)
    deadunitmsk = (BigGANFC6cmptab.score_norm.isna())
    fc6failedmsk = np.isinf(BigGANFC6cmptab.score_norm)
    print("Dead channel trial number:%d"%sum(deadunitmsk))
    print("Failed trial number:%d"%sum(fc6failedmsk))
    figh = plt.figure(figsize=[7, 7])
    ax = sns.violinplot(x='layer', y='score_norm', hue="optimizer", jitter=0.25,
                       hue_order=['CholCMA', 'HessCMA', 'CholCMA_fc6'], cut=0.1,
                       data=BigGANFC6cmptab[~deadunitmsk&~fc6failedmsk], alpha=0.4)
    ax.set_title("Comparison of Optimizer and GAN space over Units of AlexNet %s"%("RFfit" if rffit else "FullImage"))
    ax.set_ylabel("activ normalized by %s"%norm_scheme)
    ax.figure.show()
    ax.figure.savefig(join(summarydir, "BigGANFC6_cmp_%snorm_layer_all%s.jpg"%(norm_scheme, "RFfit" if rffit else
                    "_Full")))
    ax.figure.savefig(join(summarydir, "BigGANFC6_cmp_%snorm_layer_all%s.pdf"%(norm_scheme, "RFfit" if rffit else 
                    "_Full")))
    return BigGANFC6cmptab, figh

# BigGANFC6_comparison_plot(norm_scheme="fc6max", rffit=True)
# BigGANFC6_comparison_plot(norm_scheme="allmax", rffit=True)
BigGANFC6_comparison_plot(norm_scheme="allmax", rffit=False)