from pathlib import Path
from os.path import join
import torch
import matplotlib.pyplot as plt
unitdir = r"F:\insilico_exps\GAN_Evol_Dissection\resnet50_.layer3.Bottleneck5_5_7_7"
covdir = join(unitdir, "covtsrs")

#%%
optimnames = ["HessCMA500_fc6", "CholCMA_fc6", "CholCMA", "HessCMA"]
optimname1 = "CholCMA"
covfns = list(Path(covdir).glob(f"covtsrs_{optimname1}*.pt"))
covdict1 = torch.load(covfns[1])

optimname2 = "HessCMA"  # "CholCMA"
covfns = list(Path(covdir).glob(f"covtsrs_{optimname2}*.pt"))
covdict2 = torch.load(covfns[2])
#%%
layernames = list(covdict1["covtsrs"].keys())
#%%
figh, axs = plt.subplots(4, 8, figsize=[21, 12])
for i, layer in enumerate(layernames):
    axs[i, 0].imshow(covdict1["covtsrs"][layer].norm(dim=0)**2)
    axs[i, 0].set_title(f"{layer} cov1")
    axs[i, 1].imshow(covdict2["covtsrs"][layer].norm(dim=0)**2)
    axs[i, 1].set_title(f"{layer} cov2")
    prodcovtsr = covdict1["covtsrs"][layer] * covdict2["covtsrs"][layer]
    dotprodcovtsr = prodcovtsr.sum(dim=0)
    mincovtsr = torch.min(covdict1["covtsrs"][layer], covdict2["covtsrs"][layer])
    absmincovtsr = torch.min(covdict1["covtsrs"][layer].abs(), covdict2["covtsrs"][layer].abs())
    axs[i, 2].imshow(prodcovtsr.norm(dim=0)**2)
    axs[i, 2].set_title(f"{layer} cov1 * cov2")
    axs[i, 3].imshow(torch.clamp(prodcovtsr, 0).norm(dim=0)**2)
    axs[i, 3].set_title(f"{layer} Relu (cov1 * cov2)")
    axs[i, 4].imshow(dotprodcovtsr**2)
    axs[i, 4].set_title(f"{layer} Dot(cov1 , cov2)")
    axs[i, 5].imshow(torch.clamp(mincovtsr, 0).norm(dim=0)**2)
    axs[i, 5].set_title(f"{layer} Relu min(cov1 , cov2)")
    axs[i, 6].imshow(absmincovtsr.norm(dim=0)**2)
    axs[i, 6].set_title(f"{layer} min(|cov1| , |cov2|)")
    cosmap = torch.cosine_similarity(covdict1["covtsrs"][layer], covdict2["covtsrs"][layer], dim=0)
    axs[i, 7].imshow(cosmap)
    axs[i, 7].set_title(f"{layer} Cosine(cov1 , cov2)")
plt.tight_layout()
plt.show()


