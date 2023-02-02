import torch
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from core.utils.dataset_utils import ImagePathDataset
from torchvision import transforms, utils
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data
import matplotlib.pyplot as plt
#%%
_, BFEStats = load_neural_data()
#%%
Expi = 12
thread = 0
imgfps, resp_vec, bsl_vec, gen_vec = load_img_resp_pairs(BFEStats, Expi,
                 "Evol", thread=thread, stimdrive="S:", output_fmt="vec")
#%%
plt.plot(gen_vec, resp_vec, "o", alpha=0.5)
plt.plot(gen_vec, bsl_vec, "o", alpha=0.5)
plt.show()

#%%
#%%
from os.path import join
from CorrFeatTsr_lib import Corr_Feat_Machine, visualize_cctsr, loadimg_preprocess
# from core.utils.layer_hook_utils import featureFetcher_module
from core.utils.CNN_scorers import TorchScorer, load_featnet
figroot = r"E:\Network_Data_Sync\corrFeatTsr_BigGAN\fig_summary"
saveroot = r"E:\Network_Data_Sync\corrFeatTsr_BigGAN"
#%%
model, _ = load_featnet("resnet50_linf8")
recmodule_dict = {"layer1": model.layer1,
                   "layer2": model.layer2,
                   "layer3": model.layer3,
                   "layer4": model.layer4}
#%%
Animal = "Both"
for Expi in tqdm(range(1, 191)):
    for thread in range(2):
        imgfps, resp_vec, bsl_vec, gen_vec = load_img_resp_pairs(BFEStats, Expi,
                         "Evol", thread=thread, stimdrive="S:", output_fmt="vec")
        # create a dataloader for the image response pairs
        evol_ds = ImagePathDataset(imgfps, resp_vec, transform=None, img_dim=(224, 224))
        evol_dl = DataLoader(evol_ds, batch_size=60, shuffle=False, num_workers=0)

        fetcher = Corr_Feat_Machine()
        fetcher.register_module_hooks(recmodule_dict)
        fetcher.init_corr()
        for i, (imgtsr, resps) in tqdm(enumerate(evol_dl)):
            with torch.no_grad():
                model(imgtsr.cuda())
            fetcher.update_corr(resps.float())

        fetcher.calc_corr()
        fetcher.clear_hook()
        savedict = fetcher.make_savedict()

        np.savez(join(saveroot, f"{Animal}_Exp{Expi:02d}_Evol_thr{thread}_res-robust_corrTsr.npz"),
                 **savedict)

        titstr = ""
        figh = visualize_cctsr(fetcher, ["layer1", "layer2", "layer3", "layer4"], None, Expi,
                               "Both", f"BGFC_Evol", titstr, figdir=figroot)

#%%

# fetcher.init_corr()


#%%
error_dict = {}
for Expi in tqdm(range(1, 191)):
    for thread in range(2):
        try:
            imgfps, resp_vec, bsl_vec, gen_vec = load_img_resp_pairs(BFEStats, Expi,
                         "Evol", thread=thread, stimdrive="S:", output_fmt="vec")
        except Exception as e:
            print(f"Exp{Expi} thread{thread}", e)
            error_dict[(Expi, thread)] = e
