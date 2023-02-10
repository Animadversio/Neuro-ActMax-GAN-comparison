import re

import numpy as np

from core.utils.montage_utils import crop_from_montage, crop_all_from_montage
from pathlib import Path
import matplotlib.pyplot as plt
sumdir = r"F:\insilico_exps\GAN_gradEvol_cmp\protoimgs"
sumpath = Path(sumdir)

mtgfns = list(sumpath.glob("resnet50_linf8_.layer3_1_7_7_RFrsz_*.jpg"))
# r"resnet50_linf8_.layer3_1_7_7_RFrsz_Adam001.jpg"
img_col = {}
for mtgfn in mtgfns:
    optimname = re.findall("resnet50_linf8_.layer3_1_7_7_RFrsz_(.*).jpg", mtgfn.name)
    assert len(optimname) == 1
    optimname = optimname[0]
    mtg = plt.imread(mtgfn)
    imgs = crop_all_from_montage(mtg, None, imgsize=227, pad=2)
    img_col[optimname] = np.stack(imgs)
#%%

plt.imshow(imgs[1])
plt.show()
#%%
import torch
from lpips import LPIPS

Dist = LPIPS(net="alex", )
Dist.cuda().eval().requires_grad_(False)
#%%
imgtsr_col = {k: torch.tensor(v).permute(0, 3, 1, 2).float() / 255 for k, v in img_col.items()}
#%%
device = "cuda"
img1 = torch.tensor(imgs[0]).permute(2, 0, 1).unsqueeze(0).float() /255
img2 = torch.tensor(imgs[1]).permute(2, 0, 1).unsqueeze(0).float() /255
Dist.spatial = True
dist = Dist(img1.to(device), img2.to(device), normalize=True).cpu()
#%%
plt.imshow(dist.squeeze().detach().numpy())
plt.show()
#%%
distmaps = Dist.forward_distmat(imgtsr_col["Adam01_fc6"].to(device),
                                imgtsr_col["Adam001"].to(device), normalize=True, batch_size=40).cpu()
#%%
plt.figure()
plt.imshow(distmaps.mean(dim=(0,1)).squeeze().detach().numpy())
plt.colorbar()
plt.title("Adam FC6 vs Adam BigGAN")
plt.show()
#%%
distmaps_hess = Dist.forward_distmat(imgtsr_col["Adam01Hess_fc6"].to(device),
                                imgtsr_col["Adam001Hess"].to(device), normalize=True, batch_size=40).cpu()
#%%
distmaps_hess_self = Dist.forward_distmat(imgtsr_col["Adam01Hess_fc6"].to(device),
                                None, normalize=True, batch_size=40).cpu()
#%%
plt.figure()
plt.imshow(distmaps_hess.mean(dim=(0,1)).squeeze().detach().numpy())
plt.colorbar()
plt.title("Hessian Adam FC6 vs Hessian Adam BigGAN")
plt.show()
#%%
plt.figure()
plt.imshow(distmaps_hess.std(dim=(0,1)).squeeze().detach().numpy())
plt.colorbar()
plt.title("Hessian Adam FC6 vs Hessian Adam BigGAN")
plt.show()