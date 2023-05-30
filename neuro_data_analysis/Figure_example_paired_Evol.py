from neuro_data_analysis.neural_data_lib import load_neural_data, extract_all_evol_trajectory, pad_resp_traj
from core.utils.colormap_matlab import parula
#%%
_, BFEStats = load_neural_data()
resp_col, meta_df = extract_all_evol_trajectory(BFEStats, )
resp_extrap_arr, extrap_mask_arr, max_len = pad_resp_traj(resp_col)
#%%
resp_bunch = resp_col[155]
resp_FC_BG = resp_bunch[:,:2]
#%%
LIM = resp_FC_BG.min(), resp_FC_BG.max()
#%%
import numpy as np
import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms
figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ParallelSuccess"
#plot a colorbar with parula colormap, with LIM as the range
fig, ax = plt.subplots(figsize=[1.5, 3])
im = ax.imshow(np.arange(100).reshape(10, 10), cmap=parula, vmin=LIM[0], vmax=LIM[1])
fig.colorbar(im, ax=ax)
ax.remove()
plt.tight_layout()
saveallforms(figdir, "colorbar_parula", fig, ["png", "pdf"])
plt.show()
#%%
