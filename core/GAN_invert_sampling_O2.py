import torch
from tqdm import trange
from torchvision.utils import save_image
from pathlib import Path
import numpy as np
from pytorch_gan_metrics.utils import ImageDataset
from pytorch_gan_metrics.core  import torch_cov, get_inception_feature, calculate_inception_score, calculate_frechet_distance
from pytorch_pretrained_biggan import BigGAN
from core.utils.GAN_utils import upconvGAN, BigGAN_wrapper, loadBigGAN
from core.utils import saveallforms, showimg, show_imgrid, save_imgrid
from core.utils.GAN_invert_utils import GAN_invert, GAN_invert_with_scheduler
biggan = BigGAN.from_pretrained("biggan-deep-256")
biggan.eval().requires_grad_(False).cuda()
BG = BigGAN_wrapper(biggan)
G = upconvGAN("fc6")
G.cuda().requires_grad_(False).eval()
#%%
# imageset_str = "BigGAN_norm_std07"
# BG_rn_fun = lambda batch_size: \
#     BG.visualize(0.7 * torch.randn(batch_size, 256, device="cuda"))
#%%
# imageset_str = "BigGAN_1000cls_std07"
# BG_cls_fun = lambda batch_size: BG.visualize(BG.sample_vector(batch_size, class_id=None))
#%%
def _save_imgtsr(imgtsr, savedir, prefix="sample", suffix="png", offset=0):
    """imgtsr: torch image tensor, NCHW"""
    for i, img in enumerate(imgtsr):
        save_image(img, savedir / f"{prefix}{offset+i:04d}.{suffix}")


saveroot = r"/n/scratch3/users/b/biw905/GAN_sample_fid"
Path(saveroot).mkdir(exist_ok=True)
#%%
# build a command line interface for this script
#%%
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--imageset_str", type=str, default="BigGAN_1000cls_std07")
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--idx_start", type=int, default=0)
parser.add_argument("--idx_end", type=int, default=1000)
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument("--max_iter", type=int, default=2500)
args = parser.parse_args()
# an example CLI usage
# python core/GAN_invert_sampling_O2.py --imageset_str BigGAN_1000cls_std07 --batch_size 20 --idx_start 0 --idx_end 1000 --lr 2e-3 --max_iter 2500
imageset_str = args.imageset_str
if imageset_str == "BigGAN_1000cls_std07":
    BG_sample_fun = lambda batch_size: BG.visualize(BG.sample_vector(batch_size, class_id=None))
elif imageset_str == "BigGAN_norm_std07":
    BG_sample_fun = lambda batch_size: BG.visualize(0.7 * torch.randn(batch_size, 256, device="cuda"))
else:
    raise ValueError(f"imageset_str {imageset_str} not recognized")


savedir = Path(saveroot) / (imageset_str+"_invert")
savedir.mkdir(exist_ok=True)
lr = args.lr
max_iter = args.max_iter
batch_size = args.batch_size
idx_start, idx_end = args.idx_start, args.idx_end
for i in trange(idx_start, idx_end, batch_size):
    target_img = BG_sample_fun(batch_size)
    z_init = torch.randn(batch_size, 4096, device="cuda")
    z_opt, img_opt, losses = GAN_invert(G, target_img, z_init, lr=lr, max_iter=max_iter, print_progress=False)
    torch.save(losses.cpu(), savedir / f"losses_{i:04d}.pt")
    _save_imgtsr(img_opt, savedir, prefix="FC_invert", suffix="png", offset=i)
    _save_imgtsr(target_img, savedir, prefix="BG", suffix="png", offset=i)

#%%
# target_img = BG_cls_fun(10)
# z_init = torch.randn(10, 4096, device="cuda")
# #%%
# z_opt, img_opt, losses = GAN_invert(G, target_img, z_init, lr=2e-3, max_iter=2500, print_progress=False)
# #%%
# show_imgrid(img_opt, nrows=5)