import sys
sys.path.append(r"/home/biw905/Github/Neuro-ActMax-GAN-comparison")
import torch
from torchvision.utils import save_image
from tqdm import tqdm, trange
from pathlib import Path
from core.utils.plot_utils import show_imgrid
from core.utils.CNN_scorers import load_featnet, TorchScorer
from core.utils.GAN_utils import BigGAN_wrapper, upconvGAN, loadBigGAN

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


from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=25)
parser.add_argument("--img_per_class", type=int, default=50)
parser.add_argument("--class_id_start", type=int, default=0)
parser.add_argument("--class_id_end", type=int, default=1000)
args = parser.parse_args()
print(args)
# example
# python core/GAN_evol_sampling_O2.py --batch_size 25 --class_id_start 0 --class_id_end 200
savedir = r'/n/scratch3/users/b/biw905/GAN_sample_fid/resnet50_linf8_gradevol'
Path(savedir).mkdir(exist_ok=True, parents=True)

class_id_start = args.class_id_start
class_id_end = args.class_id_end
img_per_class = args.img_per_class
batch_size = args.batch_size  # 25
for class_id in trange(class_id_start, class_id_end): # 1000
    scorer.select_unit((None, ".Linearfc", class_id), allow_grad=True)
    for i in range(0, img_per_class, batch_size):
        z_init = torch.randn(batch_size, 4096, device="cuda")
        optim_constructor = lambda params: torch.optim.Adam(params, lr=0.1)
        img, z_traj, score_traj = grad_evolution(scorer, optim_constructor, z_init)
        _save_imgtsr(img, Path(savedir), prefix=f"class{class_id:03d}_", offset=i)
    scorer.cleanup()
# show_imgrid(img, nrow=5)
