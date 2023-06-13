import sys
sys.path.append(r"/home/biw905/Github/Neuro-ActMax-GAN-comparison")
import torch
from torchvision.utils import save_image
from core.utils.GAN_utils import BigGAN_wrapper, upconvGAN, loadBigGAN
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
from tqdm import tqdm, trange
from pathlib import Path
from core.utils.plot_utils import show_imgrid
from core.utils.CNN_scorers import load_featnet, TorchScorer
from core.utils.layer_hook_utils import get_module_names
from core.utils.grad_RF_estim import grad_RF_estimate, gradmap2RF_square
# _, model = load_featnet("resnet50_linf8")
# scorer = TorchScorer("resnet50_linf8")
def _save_imgtsr(imgtsr, savedir, prefix="sample", suffix="png", offset=0):
    """imgtsr: torch image tensor, NCHW"""
    for i, img in enumerate(imgtsr):
        save_image(img, savedir / f"{prefix}{offset+i:04d}.{suffix}")
#%%

G = upconvGAN("fc6")
G.cuda().requires_grad_(False).eval()
#%%
savedir = Path("/n/scratch3/users/b/biw905/GAN_sample_fid/DeePSim_4std")
batch_size = 100
for i in trange(0, 50000, batch_size):
    # use torch manual seeds with cuda generator
    z = 4 * torch.randn(batch_size, 4096, device="cuda",
                    generator=torch.cuda.manual_seed(i))
    img = G.visualize(z)
    _save_imgtsr(img, savedir, prefix="sample", offset=i)
#%%
biggan = BigGAN.from_pretrained("biggan-deep-256")
biggan.eval().requires_grad_(False).cuda()
BG = BigGAN_wrapper(biggan)
#%%
savedir = Path("/n/scratch3/users/b/biw905/GAN_sample_fid/BigGAN_std_008")
savedir.mkdir(exist_ok=True)
batch_size = 50
for i in trange(0, 50000, batch_size):
    # use torch manual seeds with cuda generator
    z = 0.08 * torch.randn(batch_size, 256, device="cuda",
                    generator=torch.cuda.manual_seed(i))
    with torch.no_grad():
        img = BG.visualize(z)
    _save_imgtsr(img, savedir, prefix="sample", offset=i)

#%%
savedir = Path("/n/scratch3/users/b/biw905/GAN_sample_fid/BigGAN_trunc07")
savedir.mkdir(exist_ok=True)
batch_size = 50
# noise_std = 0.7
for i in trange(0, 50000, batch_size):
    # use torch manual seeds with cuda generator
    # noise_std * torch.randn(128, batch_size, device="cuda")
    # z = 0.08 * torch.randn(batch_size, 256, device="cuda",
    #                 generator=torch.cuda.manual_seed(i))
    trunc_noise = truncated_noise_sample(batch_size=batch_size, truncation=0.7, seed=i)
    z = torch.cat((torch.from_numpy(trunc_noise).cuda(),
        BG.BigGAN.embeddings.weight[:,
            torch.randint(1000, size=(batch_size,),
                          generator=torch.cuda.manual_seed(i))].T,), dim=1)
    with torch.no_grad():
        img = BG.visualize(z)
    _save_imgtsr(img, savedir, prefix="sample", offset=i)

#%%

