"""
Obsolete script for loading neural data from .mat files into python. Now we use .pkl files instead.
"""

from os.path import join
from scipy.io import loadmat
import mat73
import pickle
from easydict import EasyDict as edict

matroot = "E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
#%
def load_neural_data_and_save_pkl():
    """
    Load neural data from a .mat file.
    """
    data_all = mat73.loadmat(join(matroot, "Both_BigGAN_FC6_Evol_Stats.mat"))
    pickle.dump(data_all, open(join(matroot, "Both_BigGAN_FC6_Evol_Stats.pkl"), "wb"))
    # organize the data as a list
    data_sep = []
    for Expi in range(len(data_all["BFEStats"]["Animal"])):
        S = edict()
        for key in list(data_all["BFEStats"]):
            S[key] = data_all["BFEStats"][key][Expi]
        data_sep.append(S)
    pickle.dump(data_sep, open(join(matroot, "Both_BigGAN_FC6_Evol_Stats_expsep.pkl"), "wb"))
    return data_all["BFEStats"], data_sep


def load_neural_data():
    """
    Load neural data from a .pkl file.
    """
    data_all = pickle.load(open(join(matroot, "Both_BigGAN_FC6_Evol_Stats.pkl"), "rb"))
    data_sep = pickle.load(open(join(matroot, "Both_BigGAN_FC6_Evol_Stats_expsep.pkl"), "rb"))
    return data_all["BFEStats"], data_sep


# BFEStats_merge, BFEStats = load_neural_data()



#%% scratch zone
#%% from scipy.io import loadmat
data_dict = mat73.loadmat(r"E:\OneDrive - Washington University in St. Louis\Evol_BigGAN_FC6_cmp\2020-07-24-Beto-01-Chan17\EvolStat.mat")
#%%
mat73.loadmat(r"E:\OneDrive - Washington University in St. Louis\Evol_BigGAN_FC6_cmp\2020-07-24-Beto-01-Chan17\EvolStat.mat")
#%%
data_all = mat73.loadmat(r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics\Both_BigGAN_FC6_Evol_Stats.mat")
#%%
pickle.dump(data_all, open(r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics\Both_BigGAN_FC6_Evol_Stats.pkl", "wb"))
#%%
data_loaded = pickle.load(open(r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics\Both_BigGAN_FC6_Evol_Stats.pkl", "rb"))
#%%
from easydict import EasyDict as edict
data_sep = []
for Expi in range(len(data_all["BFEStats"]["Animal"])):
    S = edict()
    for key in list(data_all["BFEStats"]):
        S[key] = data_all["BFEStats"][key][Expi]
    data_sep.append(S)
#%%
pickle.dump(data_sep, open(r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics\Both_BigGAN_FC6_Evol_Stats_expsep.pkl", "wb"))
#%%
# import hdf5storage
# import h5py
# mat = hdf5storage.loadmat(r"E:\OneDrive - Washington University in St. Louis\Evol_BigGAN_FC6_cmp\2020-07-24-Beto-01-Chan17\EvolStat.mat")
# file = h5py.File(r"E:\OneDrive - Washington University in St. Louis\Evol_BigGAN_FC6_cmp\2020-07-24-Beto-01-Chan17\EvolStat.mat", 'r')
# print(list(file['EvolStat'].keys()))
# print(file['EvolStat']['Animal'])
# print(list(file['EvolStat']['evol'].keys()))
# print(file[file['EvolStat']['evol']["idx_seq"][0,0]][0])
# #%%