
from core.utils.GAN_utils import loadBigGAN, BigGAN_wrapper, upconvGAN
from neuro_data_analysis.neural_data_lib import get_expstr, load_neural_data, parse_montage
from core.utils.plot_utils import saveallforms, show_imgrid, save_imgrid
import numpy as np
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, LinearLR
from tqdm import trange, tqdm
from pathlib import Path
#%%
BG = loadBigGAN()
BGW = BigGAN_wrapper(BG)
#%%
G = upconvGAN("fc6")
G.eval().cuda()
#%%
z = BGW.sample_vector(5, class_id=359)
#%%
imgs = BGW.visualize(z)
#%%
show_imgrid(imgs)
#%%
target_img = imgs[4].cuda()
# invert the image
z_opt = torch.randn(5, 4096, requires_grad=True, device="cuda")
opt = Adam([z_opt], lr=1e-2, weight_decay=0e-4)
for i in trange(5000):
    img_opt = G.visualize(z_opt)
    loss = ((img_opt - target_img)**2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(i, loss.item())
#%%
show_imgrid([target_img.cpu(), *img_opt.cpu()])

#%%
target_img = imgs[4].cuda()
# invert the image
z_opt = torch.randn(5, 4096, requires_grad=True, device="cuda")
opt = Adam([z_opt], lr=1e-2, weight_decay=0e-4)
scheduler = ExponentialLR(opt, gamma=0.9999)
for i in trange(5000):
    img_opt = G.visualize(z_opt)
    loss = ((img_opt - target_img)**2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    scheduler.step()
    print(i, loss.item(), "lr", scheduler.get_last_lr()[0])
#%%
show_imgrid([target_img.cpu(), *img_opt.cpu()])
#%%
figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\GAN_manif"
Path(figdir).mkdir(exist_ok=True)
torch.save(z_opt, Path(figdir)/"deepsim_z_opt.pth")
torch.save(img_opt, Path(figdir)/"deepsim_img_opt.pth")
save_imgrid([target_img.cpu(), *img_opt.cpu()], Path(figdir)/"deepsim_img_opt.png")
#%%
torch.save(z[4:], Path(figdir)/"BigGAN_z.pth")
torch.save(imgs[4:], Path(figdir)/"BigGAN_img.pth")
save_imgrid(imgs[4:], Path(figdir)/"BigGAN_img.png")
#%%
z_BG = torch.load(Path(figdir)/"BigGAN_z.pth").cpu()
#%%
import numpy as np
def generate_azel_xyz_grid(n_az, n_el, az_lim=(-np.pi, np.pi),
                           el_lim=(-np.pi/2, np.pi/2)):
    """Generate a grid of points in azimuth and elevation, and convert to
    Cartesian coordinates.
    """
    az = np.linspace(*az_lim, n_az)
    el = np.linspace(*el_lim, n_el)
    az, el = np.meshgrid(az, el)
    az = az.flatten()
    el = el.flatten()
    x, y, z = np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)
    return np.stack([x, y, z], axis=1)


def generate_orthogonal_vectors_torch(v1):
    """Generate two random orthogonal vectors to v1.
    """
    # Ensure v1 is a torch tensor
    if not isinstance(v1, torch.Tensor):
        v1 = torch.tensor(v1)
    # Generate two random vectors
    v2 = torch.randn(v1.shape)
    v3 = torch.randn(v1.shape)
    # Make v2 orthogonal to v1
    v2 -= (v2 @ v1) / v1.norm().pow(2) * v1
    # Make v3 orthogonal to both v1 and v2
    v3 -= (v3 @ v1) / v1.norm().pow(2) * v1
    v3 -= (v3 @ v2) / v2.norm().pow(2) * v2
    # Normalize v2 and v3 to have the same length as v1
    norm_v1 = v1.norm()
    v2 = v2 * norm_v1 / v2.norm()
    v3 = v3 * norm_v1 / v3.norm()
    return v2, v3

coords = generate_azel_xyz_grid(7, 7, az_lim=(-np.pi/3, np.pi/3),
                                el_lim=(-np.pi/3, np.pi/3))
vec1 = z_opt[0].detach().cpu()
torch.random.manual_seed(0)
vec2, vec3 = generate_orthogonal_vectors_torch(vec1)
basis = torch.stack([vec1, vec2, vec3], dim=0)
codes = torch.matmul(torch.tensor(coords, dtype=torch.float32), basis)
imgs = G.visualize(codes.cuda())
show_imgrid(imgs.cpu(), nrow=7)
save_imgrid(imgs.cpu(), Path(figdir)/f"deepsim_img_manif_sph_60deg_RND{0}.png", nrow=7)
#%%
vec1 = z_opt[0].detach().cpu()
torch.random.manual_seed(120)
vec2, vec3 = generate_orthogonal_vectors_torch(vec1)
basis = torch.stack([vec1, vec2, vec3], dim=0)
codes = torch.matmul(torch.tensor(coords, dtype=torch.float32), basis)
imgs = G.visualize(codes.cuda())
save_imgrid(imgs.cpu(), Path(figdir)/f"deepsim_img_manif_sph_60deg_RND{120}.png", nrow=7)
#%%
torch.random.manual_seed(0)
coords_BG = generate_azel_xyz_grid(7, 7, az_lim=(-np.pi/12, np.pi/12),
                                el_lim=(-np.pi/12, np.pi/12))
vec2_BG, vec3_BG = generate_orthogonal_vectors_torch(z_BG[0])
basis_BG = torch.stack([z_BG[0], vec2_BG, vec3_BG], dim=0)
codes_BG = torch.matmul(torch.tensor(coords, dtype=torch.float32), basis_BG)
with torch.no_grad():
    imgs_BG = BGW.visualize_batch(codes_BG.cuda(), B=15)
show_imgrid(imgs_BG.cpu(), nrow=7)

#%%
coords_BG = generate_azel_xyz_grid(7, 7, az_lim=(-np.pi/3, np.pi/3),
                                el_lim=(-np.pi/3, np.pi/3))
# manifold in noise space
class_vec = z_BG[0, 128:]
noise_vec = z_BG[0, :128]
torch.random.manual_seed(5)
vec2_BG, vec3_BG = generate_orthogonal_vectors_torch(noise_vec)
basis_BGnois = torch.stack([noise_vec, vec2_BG, vec3_BG], dim=0)
codes_BGnois = torch.matmul(torch.tensor(coords_BG, dtype=torch.float32), basis_BGnois)
codes_all = torch.cat([codes_BGnois, class_vec[None].repeat(codes_BGnois.shape[0], 1)], dim=1)
with torch.no_grad():
    imgs_BG = BGW.visualize_batch(codes_all.cuda(), B=15)
show_imgrid(imgs_BG.cpu(), nrow=7)
save_imgrid(imgs_BG.cpu(), Path(figdir)/f"BigGAN_noise_manif_sph_60deg_RND{5}.png", nrow=7)

#%%
coords_BG = generate_azel_xyz_grid(5, 5, az_lim=(-np.pi/4, np.pi/4),
                                el_lim=(-np.pi/4, np.pi/4))
# manifold in noise space
class_vec = z_BG[0, 128:]
noise_vec = z_BG[0, :128]
torch.random.manual_seed(5)
vec2_BG, vec3_BG = generate_orthogonal_vectors_torch(noise_vec)
basis_BGnois = torch.stack([noise_vec, vec2_BG, vec3_BG], dim=0)
codes_BGnois = torch.matmul(torch.tensor(coords_BG, dtype=torch.float32), basis_BGnois)
codes_all = torch.cat([codes_BGnois, class_vec[None].repeat(codes_BGnois.shape[0], 1)], dim=1)
with torch.no_grad():
    imgs_BG = BGW.visualize_batch(codes_all.cuda(), B=15)
show_imgrid(imgs_BG.cpu(), nrow=5)
save_imgrid(imgs_BG.cpu(), Path(figdir)/f"BigGAN_noise_manif_sph_45deg_RND{5}.png", nrow=5)
#%%
# coords_BG = generate_azel_xyz_grid(5, 5,
#                                 az_lim=(-np.pi/2, np.pi/2),
#                                 el_lim=(-np.pi/2, np.pi/2))
coords_BG = generate_azel_xyz_grid(7, 7,
                                az_lim=(-np.pi/3, np.pi/3),
                                el_lim=(-np.pi/3, np.pi/3))
# manifold in class space
class_vec = z_BG[0, 128:]
noise_vec = z_BG[0, :128]
torch.random.manual_seed(50)
vec2_BGcls, vec3_BGcls = generate_orthogonal_vectors_torch(class_vec)
basis_BGcls = torch.stack([class_vec, vec2_BGcls, vec3_BGcls], dim=0)
codes_BGcls = torch.matmul(torch.tensor(coords_BG, dtype=torch.float32), basis_BGcls)
codes_all = torch.cat([noise_vec[None, :].repeat(codes_BGcls.shape[0], 1), codes_BGcls], dim=1)
with torch.no_grad():
    imgs_BGcls = BGW.visualize_batch(codes_all.cuda(), B=15)
show_imgrid(imgs_BGcls.cpu(), nrow=7)
save_imgrid(imgs_BGcls.cpu(), Path(figdir)/f"BigGAN_class_manif_sph_60deg_RND{50}.png", nrow=7)

#%%
coords_BG = generate_azel_xyz_grid(5, 5,
                                az_lim=(-np.pi/3, np.pi/3),
                                el_lim=(-np.pi/3, np.pi/3))
# manifold in class space
class_vec = z_BG[0, 128:]
noise_vec = z_BG[0, :128]
torch.random.manual_seed(40)
vec2_BGcls, vec3_BGcls = generate_orthogonal_vectors_torch(class_vec)
basis_BGcls = torch.stack([class_vec, vec2_BGcls, vec3_BGcls], dim=0)
codes_BGcls = torch.matmul(torch.tensor(coords_BG, dtype=torch.float32), basis_BGcls)
codes_all = torch.cat([noise_vec[None, :].repeat(codes_BGcls.shape[0], 1), codes_BGcls], dim=1)
with torch.no_grad():
    imgs_BGcls = BGW.visualize_batch(codes_all.cuda(), B=15)
show_imgrid(imgs_BGcls.cpu(), nrow=5)
save_imgrid(imgs_BGcls.cpu(), Path(figdir)/f"BigGAN_class_manif_sph_60deg_RND{40}.png", nrow=5)
