from pathlib import Path
from os.path import join
import torch
import matplotlib.pyplot as plt
unitdir = r"F:\insilico_exps\GAN_Evol_Dissection\resnet50_.layer3.Bottleneck5_5_7_7"
unitdir = r"F:\insilico_exps\GAN_Evol_Dissection\tf_efficientnet_b6_.blocks.5_5_4_4"
covdir = join(unitdir, "covtsrs")

#%%
optimnames = ["HessCMA500_fc6", "CholCMA_fc6", "CholCMA", "HessCMA"]
optimname1 = "HessCMA"
covfns = list(Path(covdir).glob(f"covtsrs_{optimname1}*.pt"))
covdict1 = torch.load(covfns[4])
covtsrs1 = covdict1["covtsrs"] if "covtsrs" in covdict1 else covdict1["covtsr"]

optimname2 = "CholCMA"  # "CholCMA"
covfns = list(Path(covdir).glob(f"covtsrs_{optimname2}*.pt"))
covdict2 = torch.load(covfns[1])
covtsrs2 = covdict2["covtsrs"] if "covtsrs" in covdict2 else covdict2["covtsr"]
#%%
layernames = list(covtsrs1.keys())
#%%
figh, axs = plt.subplots(4, 8, figsize=[21, 12])
for i, layer in enumerate(layernames):
    axs[i, 0].imshow(covtsrs1[layer].norm(dim=0)**2)
    axs[i, 0].set_title(f"{layer} cov1")
    axs[i, 1].imshow(covtsrs2[layer].norm(dim=0)**2)
    axs[i, 1].set_title(f"{layer} cov2")
    prodcovtsr = covtsrs1[layer] * covtsrs2[layer]
    dotprodcovtsr = prodcovtsr.sum(dim=0)
    mincovtsr = torch.min(covtsrs1[layer], covtsrs2[layer])
    absmincovtsr = torch.min(covtsrs1[layer].abs(), covtsrs2[layer].abs())
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
    cosmap = torch.cosine_similarity(covtsrs1[layer], covtsrs2[layer], dim=0)
    axs[i, 7].imshow(cosmap)
    axs[i, 7].set_title(f"{layer} Cosine(cov1 , cov2)")
plt.tight_layout()
plt.show()


