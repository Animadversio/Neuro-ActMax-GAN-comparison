
import torch
from pathlib import Path
from os.path import join
from tqdm import tqdm, trange
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms
covroot = r"F:\insilico_exps\GAN_Evol_CovTsr"
savedir = r"F:\insilico_exps\GAN_Evol_CovTsr\consensus"
unitdirs = list(Path(covroot).glob("resnet50*")) + \
           list(Path(covroot).glob("tf_efficientnet_b6*"))
#%%
# change plotting mode to agg
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('module://backend_interagg')
#%%
covdir = unitdirs[-1]
for covdir in tqdm(unitdirs[84:]):
    optimnames = ["HessCMA500_fc6", "CholCMA_fc6", "CholCMA", "HessCMA"]
    unitname = covdir.name
    covfns_col = {optimname: sorted(list(Path(covdir).glob(f"covtsrs_{optimname}_[0-9][0-9][0-9][0-9][0-9].pt")))
                  for optimname in optimnames}
    covfns_col_blk40 = {optimname: sorted(list(Path(covdir).glob(f"covtsrs_{optimname}*_block40.pt")))
                  for optimname in optimnames}

    #%%
    use_blk40 = False
    # optimname1 = "HessCMA"
    # optimname2 = "CholCMA"
    # i1 = 0
    # i2 = 0
    for optimname1, optimname2, i1, i2 in [ \
        ("HessCMA", "CholCMA", 0, 3),
        ("CholCMA_fc6", "CholCMA", 3, 0),
        ("CholCMA_fc6", "HessCMA", 0, 4),
        ("HessCMA", "HessCMA", 1, 2),
        ("CholCMA", "CholCMA", 1, 2),
        ("CholCMA_fc6", "CholCMA_fc6", 1, 2),
        ]:
        assert not ((optimname1 == optimname2) and (i1 == i2)), "Same optimname and i1==i2"
        covfn1 = covfns_col[optimname1][i1] if not use_blk40 else covfns_col_blk40[optimname1][i1]
        RND1 = int(covfn1.name[-8:-3]) if not "block40" in covfn1.name else int(covfn1.name[-16:-11])
        covdict1 = torch.load(covfn1)
        covtsrs1 = covdict1["covtsrs"] if "covtsrs" in covdict1 else covdict1["covtsr"]

        covfn2 = covfns_col[optimname2][i2] if not use_blk40 else covfns_col_blk40[optimname2][i2]
        RND2 = int(covfn2.name[-8:-3]) if not "block40" in covfn1.name else int(covfn2.name[-16:-11])
        covdict2 = torch.load(covfn2)
        covtsrs2 = covdict2["covtsrs"] if "covtsrs" in covdict2 else covdict2["covtsr"]
        assert covfn1 != covfn2, "Same covfn1 and covfn2"
        #%%
        layernames = list(covtsrs1.keys())
        #%%
        figh, axs = plt.subplots(4, 10, figsize=[25, 12])
        for i, layer in enumerate(layernames):
            axs[i, 0].imshow(covtsrs1[layer].norm(dim=0)**2)
            axs[i, 0].set_title(f"{layer} cov1")
            axs[i, 1].imshow(covtsrs2[layer].norm(dim=0)**2)
            axs[i, 1].set_title(f"{layer} cov2")
            axs[i, 2].imshow(torch.clamp(covtsrs1[layer], 0).norm(dim=0)**2)
            axs[i, 2].set_title(f"{layer} cov1 relu")
            axs[i, 3].imshow(torch.clamp(covtsrs2[layer], 0).norm(dim=0)**2)
            axs[i, 3].set_title(f"{layer} cov2 relu")
            prodcovtsr = covtsrs1[layer] * covtsrs2[layer]
            dotprodcovtsr = prodcovtsr.sum(dim=0)
            mincovtsr = torch.min(covtsrs1[layer], covtsrs2[layer])
            absmincovtsr = torch.min(covtsrs1[layer].abs(), covtsrs2[layer].abs())
            cosmap = torch.cosine_similarity(covtsrs1[layer], covtsrs2[layer], dim=0)
            axs[i, 4].imshow(prodcovtsr.norm(dim=0)**2)
            axs[i, 4].set_title(f"{layer} cov1 * cov2")
            axs[i, 5].imshow(torch.clamp(prodcovtsr, 0).norm(dim=0)**2)
            axs[i, 5].set_title(f"{layer} Relu (cov1 * cov2)")
            axs[i, 6].imshow(dotprodcovtsr**2)
            axs[i, 6].set_title(f"{layer} Dot(cov1 , cov2)")
            axs[i, 7].imshow(torch.clamp(mincovtsr, 0).norm(dim=0)**2)
            axs[i, 7].set_title(f"{layer} Relu min(cov1 , cov2)")
            axs[i, 8].imshow(absmincovtsr.norm(dim=0)**2)
            axs[i, 8].set_title(f"{layer} min(|cov1| , |cov2|)")
            axs[i, 9].imshow(cosmap)
            axs[i, 9].set_title(f"{layer} Cosine(cov1 , cov2)")
        plt.suptitle(f"{unitname} {optimname1} {RND1} vs {optimname2} {RND2}", fontsize=18)
        plt.tight_layout()
        saveallforms(savedir, f"{unitname}_{optimname1}-{RND1}_vs_{optimname2}-{RND2}", figh, ["png", "pdf"])
        plt.show()
        #%%
        Sall = edict()
        for layer in layernames:
            cov1 = covtsrs1[layer]
            cov2 = covtsrs2[layer]
            S = edict()
            S.cov1 = cov1.norm(dim=0)**2
            S.cov2 = cov2.norm(dim=0)**2
            S.cov1_relu = torch.clamp(cov1, 0).norm(dim=0)**2
            S.cov2_relu = torch.clamp(cov2, 0).norm(dim=0)**2
            prodcov = cov1 * cov2
            dotcov = prodcov.sum(dim=0)
            S.prodcov = prodcov.norm(dim=0)**2
            S.prodcov_relu = torch.clamp(prodcov, 0).norm(dim=0)**2
            S.dotcov = dotcov**2
            mincov = torch.min(cov1, cov2)
            absmincov = torch.min(cov1.abs(), cov2.abs())
            S.mincov_relu = torch.clamp(mincov, 0).norm(dim=0)**2
            S.absmincov = absmincov.norm(dim=0)**2
            cosmap = torch.cosine_similarity(cov1, cov2, dim=0)
            S.cosmap = cosmap
            Sall[layer] = S
        #%%
        torch.save(Sall, Path(savedir)/f"{unitname}_{optimname1}-{RND1}_vs_{optimname2}-{RND2}_maps.pt")


