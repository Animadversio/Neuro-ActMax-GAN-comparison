from core.utils.montage_utils import make_grid_np
import matplotlib.pyplot as plt
from os.path import join
imgnetdir = r"E:\Datasets\ImageNet64x64\valid_64x64"
imgcol = []
for i in range(4096):
    img = plt.imread(join(imgnetdir, f"{i+1:05d}.png"))
    imgcol.append(img[:, :, :3])
#%%
mtg = make_grid_np(imgcol, nrow=64, padding=4)
plt.imsave(r"E:\OneDrive - Harvard University"+\
"\PhDDefense_Talk\Figures\ImageMontageSchematic\montage4096.png", mtg)
plt.imsave(r"E:\OneDrive - Harvard University"+\
"\PhDDefense_Talk\Figures\ImageMontageSchematic\montage4096.jpg", mtg)
#%%
mtg = make_grid_np(imgcol[:1024], nrow=32, padding=4)
plt.imsave(r"E:\OneDrive - Harvard University"+\
"\PhDDefense_Talk\Figures\ImageMontageSchematic\montage1024.png", mtg)
plt.imsave(r"E:\OneDrive - Harvard University"+\
"\PhDDefense_Talk\Figures\ImageMontageSchematic\montage1024.jpg", mtg)

#%%
for N in [2, 4, 6, 10, 16, 24, 32, 64]:
    mtg = make_grid_np(imgcol[:N*N], nrow=N, padding=4)
    plt.imsave(rf"E:\OneDrive - Harvard University"+\
    rf"\PhDDefense_Talk\Figures\ImageMontageSchematic\montage{N*N:04d}.png", mtg)
    plt.imsave(rf"E:\OneDrive - Harvard University"+\
    rf"\PhDDefense_Talk\Figures\ImageMontageSchematic\montage{N*N:04d}.jpg", mtg)
