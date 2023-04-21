from core.utils.grad_RF_estim import grad_RF_estimate, gradmap2RF_square, \
    GAN_grad_RF_estimate, fit_2dgauss
from core.utils.layer_hook_utils import get_module_names

def get_center_pos_and_rf(model, layer, input_size=(3, 227, 227), device="cuda"):
    if not "fc" in layer:
        module_names, module_types, module_spec = get_module_names(model, input_size=input_size, device=device)
        layer_key = [k for k, v in module_names.items() if v == layer][0]
        feat_outshape = module_spec[layer_key]['outshape']
        assert len(feat_outshape) == 3  # fc layer will fail
        cent_pos = (feat_outshape[1]//2, feat_outshape[2]//2)
    else:
        cent_pos = None
    # rf Mapping,
    if not "fc" in layer:
        print("Computing RF by direct backprop: ")
        gradAmpmap = grad_RF_estimate(model, layer, (slice(None), *cent_pos), input_size=input_size,
                                      device=device, show=False, reps=30, batch=1)
        Xlim, Ylim = gradmap2RF_square(gradAmpmap, absthresh=1E-8, relthresh=0.01, square=True)
        corner = (Xlim[0], Ylim[0])
        imgsize = (Xlim[1] - Xlim[0], Ylim[1] - Ylim[0])
    else:
        print("Computing RF by direct backprop: ")
        gradAmpmap = grad_RF_estimate(model, layer, (slice(None),), input_size=input_size,
                                      device=device, show=False, reps=30, batch=1)
        Xlim, Ylim = gradmap2RF_square(gradAmpmap, absthresh=1E-8, relthresh=0.01, square=True)
        corner = (Xlim[0], Ylim[0])
        imgsize = (Xlim[1] - Xlim[0], Ylim[1] - Ylim[0])
        # imgsize = input_size[-2:]
        # corner = (0, 0)
        # Xlim = (corner[0], corner[0] + imgsize[0])
        # Ylim = (corner[1], corner[1] + imgsize[1])

    return cent_pos, corner, imgsize, Xlim, Ylim, gradAmpmap

# cnnmodel, _ = load_featnet("resnet50_linf8",)
# cent_pos,gradAmpmap = get_center_pos_and_rf(cnnmodel, ".Linearfc", input_size=(3, 227, 227), device="cuda")
#%%
from lpips import LPIPS
from timm import list_models, create_model
from torchvision.models import resnet50, alexnet
from core.utils.CNN_scorers import load_featnet
from core.utils.CNN_scorers import TorchScorer
#%%
from core.utils.GAN_utils import upconvGAN
G = upconvGAN()
G.eval().cuda()
G.requires_grad_(False)
#%%
RFdir = r"F:\insilico_exps\GAN_Evol_cmp\RFmaps"
# netname = "resnet50_linf8"
# cnnmodel, _ = load_featnet("resnet50_linf8",)
for netname in ["resnet50", "resnet50_linf8"]:
    cnnmodel, _ = load_featnet(netname,)
    for layer, unitslice in [(".layer1.Bottleneck1", (slice(None), 28, 28)),
                            (".layer2.Bottleneck3", (slice(None), 14, 14)),
                            (".layer3.Bottleneck5", (slice(None), 7, 7)),
                            (".layer4.Bottleneck2", (slice(None), 4, 4))
                            #(".Linearfc", (slice(None),))
                                ]:
        layer_short = layer[1:].replace(".Bottleneck", "B")
        gradAmpmap = grad_RF_estimate(cnnmodel, layer, unitslice, input_size=(3, 227, 227),
                                              device="cuda", show=True, reps=100, batch=1)
        fit_2dgauss(gradAmpmap, f"{netname}-"+layer_short, outdir=RFdir, plot=True)
        gradAmpmap = GAN_grad_RF_estimate(G, cnnmodel, layer, unitslice, input_size=(3, 227, 227),
                                              device="cuda", show=True, reps=30, batch=1)
        fit_2dgauss(gradAmpmap, f"{netname}-GAN-"+layer_short, outdir=RFdir, plot=True)
        # Xlim, Ylim = gradmap2RF_square(gradAmpmap, relthresh=0.01, square=True)
#%%
RFdir = r"F:\insilico_exps\GAN_Evol_cmp\RFmaps"
for netname in ["tf_efficientnet_b6_ap", "tf_efficientnet_b6"]:
    scorer = TorchScorer(netname, )
    cnnmodel = scorer.model
    for layer, unitslice in [(".blocks.0", (slice(None), 57, 57)),
                            (".blocks.1", (slice(None), 28, 28)),
                            (".blocks.2", (slice(None), 14, 14)),
                            (".blocks.3", (slice(None), 7, 7)),
                            (".blocks.4", (slice(None), 7, 7)),
                            (".blocks.5", (slice(None), 4, 4)),
                            (".blocks.6", (slice(None), 4, 4)),
                            ]:
        layer_short = layer[1:].replace(".Bottleneck", "B")
        gradAmpmap = grad_RF_estimate(cnnmodel, layer, unitslice, input_size=(3, 227, 227),
                                              device="cuda", show=True, reps=100, batch=1)
        fit_2dgauss(gradAmpmap, f"{netname}-"+layer_short, outdir=RFdir, plot=True)
        gradAmpmap = GAN_grad_RF_estimate(G, cnnmodel, layer, unitslice, input_size=(3, 227, 227),
                                              device="cuda", show=True, reps=30, batch=1)
        fit_2dgauss(gradAmpmap, f"{netname}-GAN-"+layer_short, outdir=RFdir, plot=True)
#%% Save in dict
# save as pkl
import pickle
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
fitmaps_dict = {}
for netname in ["resnet50", "resnet50_linf8"]:
    for layer_short in ["layer1B1", "layer2B3", "layer3B5", "layer4B2"]:
        RFdict = np.load(join(RFdir, f"{netname}-{layer_short}_gradAmpMap_GaussianFit.npz"))
        fitmap = RFdict["fitmap"]
        fitmap = fitmap / fitmap.max()
        fitmap_pix = zoom(fitmap, 224 / 227, order=2)
        fitmap_L3 = zoom(fitmap, 14 / 227, order=2)
        fitmap_L4 = zoom(fitmap, 7 / 227, order=2)
        # plot these maps
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(fitmap_pix, cmap="jet")
        axs[1].imshow(fitmap_L3, cmap="jet")
        axs[2].imshow(fitmap_L4, cmap="jet")
        plt.show()
        fitmaps_dict[f"{netname}_{layer_short}"] = (fitmap_pix, fitmap_L3, fitmap_L4)

#%%
for netname in ["tf_efficientnet_b6_ap", "tf_efficientnet_b6"]:
    for layer_short in ["blocks.0", "blocks.1", "blocks.2", "blocks.3",
                        "blocks.4", "blocks.5", "blocks.6"]:
        RFdict = np.load(join(RFdir, f"{netname}-{layer_short}_gradAmpMap_GaussianFit.npz"))
        fitmap = RFdict["fitmap"]
        fitmap = fitmap / fitmap.max()
        fitmap_pix = zoom(fitmap, 224 / 227, order=2)
        fitmap_L3 = zoom(fitmap, 14 / 227, order=2)
        fitmap_L4 = zoom(fitmap, 7 / 227, order=2)
        # plot these maps
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(fitmap_pix, cmap="jet")
        axs[1].imshow(fitmap_L3, cmap="jet")
        axs[2].imshow(fitmap_L4, cmap="jet")
        plt.show()
        fitmaps_dict[f"{netname}_{layer_short}"] = (fitmap_pix, fitmap_L3, fitmap_L4)
#%%

#%%
with open(join(RFdir, "fitmaps_dict.pkl"), "wb") as f:
    pickle.dump(fitmaps_dict, f)

