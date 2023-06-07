import sys
sys.path.append(r"/home/biw905/Github/Neuro-ActMax-GAN-comparison")
import torch
from torchvision.utils import save_image
from tqdm import tqdm, trange
from pathlib import Path
from core.utils.plot_utils import show_imgrid
from core.utils.CNN_scorers import load_featnet, TorchScorer
from core.utils.GAN_utils import BigGAN_wrapper, upconvGAN, loadBigGAN
from core.utils.layer_hook_utils import get_module_names
from core.utils.grad_RF_estim import grad_RF_estimate, gradmap2RF_square
# _, model = load_featnet("resnet50_linf8")
scorer = TorchScorer("resnet50_linf8")
G = upconvGAN("fc6")
G.cuda().requires_grad_(False).eval()
# scores = scorer.score_tsr_wgrad(G.visualize(torch.randn(5, 4096, device="cuda")))
#%%
def _save_imgtsr(imgtsr, savedir, prefix="sample", suffix="png", offset=0):
    """imgtsr: torch image tensor, NCHW"""
    for i, img in enumerate(imgtsr):
        save_image(img, savedir / f"{prefix}{offset+i:04d}.{suffix}")


def grad_evolution(scorer, optim_constructor, z_init, hess_param=False, evc=None, steps=100,
                   ):
    if hess_param:
        assert evc is not None
        evc = evc.cuda()

    z_init = z_init.cuda()
    w = z_init @ evc if hess_param else z_init.detach().clone()
    w.requires_grad_(True)
    # optimizer = optimCls([w], **optimCfg)
    optimizer = optim_constructor([w])
    score_traj = []
    z_traj = []
    img_traj = []
    pbar = trange(steps)
    for i in pbar:
        optimizer.zero_grad()
        z = (w @ evc.t()) if hess_param else w
        img = G.visualize(z)
        score = scorer.score_tsr_wgrad(img)
        score_traj.append(score.detach().cpu())
        z_traj.append(z.detach().cpu())
        loss = - score.sum()
        loss.backward()
        optimizer.step()
        zero_mask = (score == 0)
        if zero_mask.sum() > 0:
            new_z = G.sample_vector(zero_mask.sum())
            w.data[zero_mask] = new_z @ evc if hess_param else new_z
        pbar.set_description("  ".join(["%.2f" % s for s in score.detach()]))
        # print("  ".join(["%.2f" % s for s in score.detach()]))
        img_traj.append(img.detach().cpu().clone())

    idx = torch.argsort(score.detach().cpu(), descending=True)
    score_traj = torch.stack(score_traj)
    z_traj = torch.stack(z_traj)
    img = img.detach()[idx]
    # img_traj = torch.stack(img_traj)[:, idx, :, :, :]
    z_traj = z_traj[:, idx, :]  # sort the sample
    score_traj = score_traj[:, idx]  # sort the sample
    return img, z_traj, score_traj


def get_center_pos_and_rf(model, layer, input_size=(3, 256, 256), device="cuda"):
    module_names, module_types, module_spec = get_module_names(model, input_size=input_size, device=device)
    layer_key = [k for k, v in module_names.items() if v == layer][0]
    feat_outshape = module_spec[layer_key]['outshape']
    if len(feat_outshape) == 3:
        cent_pos = (feat_outshape[1]//2, feat_outshape[2]//2)
    elif len(feat_outshape) == 1:
        cent_pos = ()
    else:
        raise ValueError(f"Unknown layer shape {feat_outshape} for layer {layer}")

    if len(feat_outshape) == 3: # fixit
        print("Computing RF by direct backprop: ")
        gradAmpmap = grad_RF_estimate(model, layer, (slice(None), *cent_pos), input_size=input_size,
                                      device=device, show=False, reps=30, batch=1)
        Xlim, Ylim = gradmap2RF_square(gradAmpmap, absthresh=1E-8, relthresh=0.01, square=True)
        corner = (Xlim[0], Ylim[0])
        imgsize = (Xlim[1] - Xlim[0], Ylim[1] - Ylim[0])
    elif len(feat_outshape) == 1:
        imgsize = input_size[-2:]
        corner = (0, 0)
        Xlim = (corner[0], corner[0] + imgsize[0])
        Ylim = (corner[1], corner[1] + imgsize[1])
    else:
        raise ValueError(f"Unknown layer shape {feat_outshape} for layer {layer}")

    return cent_pos, corner, imgsize, Xlim, Ylim

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--layer", type=str, default=".Linearfc")
parser.add_argument("--dirname", type=str, default="resnet50_linf8_gradevol")
parser.add_argument("--batch_size", type=int, default=25)
parser.add_argument("--img_per_class", type=int, default=50)
parser.add_argument("--class_id_start", type=int, default=0)
parser.add_argument("--class_id_end", type=int, default=1000)
args = parser.parse_args()
print(args)
# example
# python core/GAN_evol_sampling_O2.py --batch_size 25 --class_id_start 0 --class_id_end 200
savedir = rf'/n/scratch3/users/b/biw905/GAN_sample_fid/{args.dirname}'
Path(savedir).mkdir(exist_ok=True, parents=True)

layername = args.layer
class_id_start = args.class_id_start
class_id_end = args.class_id_end
img_per_class = args.img_per_class
batch_size = args.batch_size  # 25
#%%
cent_pos, corner, imgsize, Xlim, Ylim = get_center_pos_and_rf(scorer.model, layername, input_size=(3, 256, 256), device="cuda")
#%%
for class_id in trange(class_id_start, class_id_end):  # 1000
    scorer.select_unit((None, layername, class_id, *cent_pos), allow_grad=True)
    for i in range(0, img_per_class, batch_size):
        z_init = torch.randn(batch_size, 4096, device="cuda")
        optim_constructor = lambda params: torch.optim.Adam(params, lr=0.1)
        img, z_traj, score_traj = grad_evolution(scorer, optim_constructor, z_init)
        _save_imgtsr(img, Path(savedir), prefix=f"class{class_id:03d}_", offset=i)
    scorer.cleanup()
# show_imgrid(img, nrow=5)
