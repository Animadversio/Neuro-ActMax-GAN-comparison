import torch
import numpy as np
import os
from os.path import join
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from core.utils.dataset_utils import ImagePathDataset
from torchvision import transforms, utils
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from core.utils.plot_utils import saveallforms
#%%
BFEStats_merge, BFEStats = load_neural_data()
#%%
sumdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_activation_stats"
acttab = pd.read_csv(join(sumdir, "act_increase_summary.csv"))
#%%
# BGerr = np.stack((acttab.BigGAN_endinit_CI_1, acttab.BigGAN_endinit_CI_2),axis=1) \
#        - acttab.BigGAN_endinit_m.to_numpy()[:,None]
# fc6err = np.stack((acttab.fc6_endinit_CI_1, acttab.fc6_endinit_CI_2),axis=1) \
#          - acttab.fc6_endinit_m.to_numpy()[:,None]
BGerr = acttab.BigGAN_endinit_CI_2 - acttab.BigGAN_endinit_m
fc6err = acttab.fc6_endinit_CI_2 - acttab.fc6_endinit_m
plt.figure(figsize=(6, 6))
plt.errorbar(acttab.BigGAN_endinit_m, acttab.fc6_endinit_m,
             xerr=BGerr, yerr=fc6err, alpha=0.1, fmt=".")
plt.axline((0, 0), (1, 1), color="k")
plt.axvline(0, color="k", alpha=0.5)
plt.axhline(0, color="k", alpha=0.5)
plt.title("Comparison of activation increase in BigGAN and fc6 evolution")
plt.xlabel("BigGAN end - init (events/sec)")
plt.ylabel("fc6 end - init (events/sec)")
plt.axis("equal")
saveallforms(sumdir, "BGfc6_endinit_all")
plt.show()
#%%

BGerr = acttab.BigGAN_endinit_CI_2 - acttab.BigGAN_endinit_m
fc6err = acttab.fc6_endinit_CI_2 - acttab.fc6_endinit_m
plt.figure(figsize=(6, 6))
plt.errorbar(acttab.BigGAN_endinit_m, acttab.fc6_endinit_m,
             xerr=BGerr, yerr=fc6err, alpha=0.1, fmt=".", )
plt.axline((0, 0), (1, 1), color="k")
plt.axvline(0, color="k", alpha=0.5)
plt.axhline(0, color="k", alpha=0.5)
plt.title("Comparison of activation increase in BigGAN and fc6 evolution")
plt.xlabel("BigGAN end - init (events/sec)")
plt.ylabel("fc6 end - init (events/sec)")
plt.axis("equal")
saveallforms(sumdir, "BGfc6_endinit_all_unit")
plt.show()
#%%
Alfamsk = acttab.Animal =="Alfa"
Betomsk = acttab.Animal =="Beto"
V4msk = acttab.area =="V4"
ITmsk = acttab.area =="IT"
#%%
plt.figure(figsize=(6, 6))
for label, msk in zip(["V4", "IT"], [V4msk, ITmsk]):
    plt.errorbar(acttab.BigGAN_endinit_m[msk], acttab.fc6_endinit_m[msk],
                 xerr=BGerr[msk], yerr=fc6err[msk], alpha=0.3, fmt=".", label=label)

plt.axline((0, 0), (1, 1), color="k")
plt.axvline(0, color="k", alpha=0.5)
plt.axhline(0, color="k", alpha=0.5)
plt.xlabel("BigGAN end - init (events/sec)")
plt.ylabel("fc6 end - init (events/sec)")
plt.title("Comparison of activation increase in BigGAN and fc6 evolution")
plt.axis("equal")
plt.legend()
saveallforms(sumdir, "BGfc6_endinit_area_sep")
plt.show()
#%%
plt.figure(figsize=(6, 6))
for label, msk in zip(["Alfa", "Beto"], [Alfamsk, Betomsk]):
    plt.errorbar(acttab.BigGAN_endinit_m[msk], acttab.fc6_endinit_m[msk],
                 xerr=BGerr[msk], yerr=fc6err[msk], alpha=0.3, fmt=".", label=label)

plt.axline((0, 0), (1, 1), color="k")
plt.axvline(0, color="k", alpha=0.5)
plt.axhline(0, color="k", alpha=0.5)
plt.axis("equal")
plt.legend()
plt.xlabel("BigGAN end - init (events/sec)")
plt.ylabel("fc6 end - init (events/sec)")
plt.title("Comparison of activation increase in BigGAN and fc6 evolution")
saveallforms(sumdir, "BGfc6_endinit_anim_sep")
plt.show()
#%%
fc6err = acttab.fc6_init_CI_2 - acttab.fc6_init_m
BGerr = acttab.BigGAN_init_CI_2 - acttab.BigGAN_init_m
plt.figure(figsize=(6, 6))
for label, msk in zip(["V4", "IT"], [V4msk, ITmsk]):
    plt.errorbar(acttab.BigGAN_init_m[msk], acttab.fc6_init_m[msk],
                 xerr=BGerr[msk], yerr=fc6err[msk], alpha=0.3, fmt=".", label=label)

plt.axline((0, 0), (1, 1), color="k")
plt.axvline(0, color="k", alpha=0.5)
plt.axhline(0, color="k", alpha=0.5)
plt.axis("equal")
plt.legend()
plt.xlabel("BigGAN init (events/sec)")
plt.ylabel("fc6 init (events/sec)")
plt.title("Comparison of initial activation in BigGAN and fc6 evolution")
saveallforms(sumdir, "BGfc6_init_area_sep")
plt.show()

#%%
fc6err = acttab.fc6_init_CI_2 - acttab.fc6_init_m
BGerr = acttab.BigGAN_init_CI_2 - acttab.BigGAN_init_m
plt.figure(figsize=(6, 6))
for label, msk in zip(["Alfa", "Beto"], [Alfamsk, Betomsk]):
    plt.errorbar(acttab.BigGAN_init_m[msk], acttab.fc6_init_m[msk],
                 xerr=BGerr[msk], yerr=fc6err[msk], alpha=0.3, fmt=".", label=label)

plt.axline((0, 0), (1, 1), color="k")
plt.axvline(0, color="k", alpha=0.5)
plt.axhline(0, color="k", alpha=0.5)
plt.axis("equal")
plt.legend()
plt.xlabel("BigGAN init (events/sec)")
plt.ylabel("fc6 init (events/sec)")
plt.title("Comparison of initial activation in BigGAN and fc6 evolution")
saveallforms(sumdir, "BGfc6_init_anim_sep")
plt.show()
#%%
unittable = acttab.groupby(by=["Animal", "prefchan"]).mean()
#%%
acttab.groupby(by=["Animal", "prefchan", "area"])["ephysFN"].count()
#%%
colororder = plt.rcParams['axes.prop_cycle'].by_key()['color']
unit_filter = acttab.groupby(by=["Animal", "prefchan", "area"])["ephysFN"].count() > 1
rep_units = unit_filter.index[unit_filter]
figh, axs = plt.subplots(6, 7, figsize=(18, 14))
for ui, unit in enumerate(rep_units):
    print(unit)
    msk = (acttab.Animal == unit[0]) & (acttab.prefchan == unit[1]) & (acttab.area == unit[2])
    parttab = acttab[msk]
    clr = colororder[{"IT": 0, "V4": 1, "V1":2}[unit[2]]]
    # print(acttab[msk])
    ax = axs.flatten()[ui]
    # ax.plot(parttab.BigGAN_endinit_m, parttab.fc6_endinit_m, "o", )
    ax.errorbar(parttab.BigGAN_endinit_m, parttab.fc6_endinit_m,
                parttab.BigGAN_init_CI_2 - parttab.BigGAN_init_m,
                parttab.fc6_init_CI_2 - parttab.fc6_init_m, fmt="o", color=clr)
    ax.axline((0, 0), (1, 1), color="k")
    ax.axvline(0, color="k", alpha=0.5)
    ax.axhline(0, color="k", alpha=0.5)
    ax.axis("equal")
    ax.set_title(unit)
figh.supxlabel("BigGAN end - init (events/sec)", fontsize=16)
figh.supylabel("fc6 end - init (events/sec)", fontsize=16)
figh.suptitle("Repetition consistency of activation increase in BigGAN and fc6 evolution", fontsize=18)
plt.tight_layout()
saveallforms(sumdir, "BGfc6_endinit_unit_rep_consistency")
plt.show()
#%%

# get default color order
