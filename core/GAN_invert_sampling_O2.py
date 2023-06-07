import torch
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
imageset_str = "BigGAN_norm_std07"
BG_rn_fun = lambda batch_size: \
        BG.visualize(0.7 * torch.randn(batch_size, 256, device="cuda"))
#%%
imageset_str = "BigGAN_1000cls_std07"
BG_cls_fun = lambda batch_size: BG.visualize(BG.sample_vector(batch_size, class_id=None))
#%%
target_img = BG_cls_fun(10)
z_init = torch.randn(10, 4096, device="cuda")
#%%
z_opt, img_opt, losses = GAN_invert(G, target_img, z_init, lr=2e-3, max_iter=5000, print_progress=False)
#%%
show_imgrid(img_opt, nrows=5)
