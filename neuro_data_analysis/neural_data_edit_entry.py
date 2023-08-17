"""Script to add optimizer info to the neural data .pkl file."""
import pickle
from os.path import join
import numpy as np
import pandas as pd
from neuro_data_analysis.neural_data_lib import load_neural_data, matroot
#%%
tabdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables"
optim_df = pd.read_csv(join(tabdir, "meta_optimizer_info.csv"))
meta_df = pd.read_csv(join(tabdir, "meta_stats.csv"), index_col=0)
meta_act_df = pd.read_csv(join(tabdir, "meta_activation_stats.csv"), index_col=0)
#%%
# meta_df.merge(optim_df, left_index=True, right_on="Expi", how="left")
merge_tmp_df = pd.merge(meta_df, optim_df, left_index=True, right_on="Expi", how="left", suffixes=("", "_y"))
merge_tmp_df.reset_index(drop=True, inplace=True)
merge_tmp_df.set_index("Expi", inplace=True)
merge_tmp_df.to_csv(join(tabdir, "meta_stats_w_optimizer.csv"))
#%%
meta_act_df_merge = meta_act_df.merge(optim_df, left_index=True, right_on="Expi", how="left", suffixes=("", "_y"))
meta_act_df_merge.reset_index(drop=True, inplace=True)
meta_act_df_merge.set_index("Expi", inplace=True)
meta_act_df_merge.to_csv(join(tabdir, "meta_activation_stats_w_optimizer.csv"))
#%%
optim_df.set_index("Expi", inplace=True)
#%%
BFEStats_merge, BFEStats = load_neural_data()
#%%
for Expi in range(1, 190+1):
    if BFEStats[Expi - 1].evol is None:
        print(f"Expi {Expi} is empty")
        continue
    BFEStats[Expi - 1].evol.optim_names = [optim_df.loc[Expi].optim_names1,
                                           optim_df.loc[Expi].optim_names2]
    BFEStats_merge["evol"][Expi - 1]["optim_names"] = [optim_df.loc[Expi].optim_names1,
                                                       optim_df.loc[Expi].optim_names2]
#%%
pickle.dump(BFEStats_merge, open(join(matroot, "Both_BigGAN_FC6_Evol_Stats.pkl"), "wb"))
pickle.dump(BFEStats, open(join(matroot, "Both_BigGAN_FC6_Evol_Stats_expsep.pkl"), "wb"))


# pickle.load(open(join(matroot, "Both_BigGAN_FC6_Evol_Stats.pkl"), "rb"))
# def add_entry2neural_data():
#     """
#     Load neural data from a .pkl file.
#     """
#     BFEStats_merge = pickle.load(open(join(matroot, "Both_BigGAN_FC6_Evol_Stats.pkl"), "rb"))
#     BFEStats = pickle.load(open(join(matroot, "Both_BigGAN_FC6_Evol_Stats_expsep.pkl"), "rb"))
#     return BFEStats_merge, BFEStats