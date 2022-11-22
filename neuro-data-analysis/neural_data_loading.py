from os.path import join
from scipy.io import loadmat
matroot = "E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
def load_neural_data(Animal):
    """
    Load neural data from a .mat file.
    """
    data = loadmat(join(matroot, Animal + "_BigGAN_FC6_Evol_Stats.mat"),
                   matlab_compatible=True)[ 'BFEStats']
    return data

data = load_neural_data("Beto")