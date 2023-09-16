# Importing required libraries
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import multivariate_normal
import matplotlib
# use vector font
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# Correcting the Gaussian mixture function evaluation
def evaluate_gaussian_mixture(x, y, gaussians):
    z = np.zeros_like(x)
    xy = np.row_stack([x.flatten(), y.flatten()]).T
    for g in gaussians:
        mean, cov, weight = g["mean"], g["cov"], g["weight"]
        z += weight * multivariate_normal.pdf(xy, mean=mean, cov=cov).reshape(z.shape)
        # for i in range(x.shape[0]):
        #     for j in range(x.shape[1]):
        #         z[i, j] += weight * multivariate_normal.pdf([x[i, j], y[i, j]], mean=mean, cov=cov)
    return z

figdir = r"E:\OneDrive - Harvard University\PhDDefense_Talk\Figures\Landscape_objectness"
#%%
# Gaussian components
gaussians = [
    {"mean": [0, 0], "cov": [[6, 0], [0, 6]], "weight": 1.4},
    {"mean": [5, 5], "cov": [[8, 4], [4, 15]], "weight": 1.6},
]
x = np.linspace(-15, 15, 200)
y = np.linspace(-15, 15, 200)
x, y = np.meshgrid(x, y)
# Evaluating the mixture
z = evaluate_gaussian_mixture(x, y, gaussians)
x_obj = np.linspace(-15, 15, 100)
y_obj = np.linspace(-15, 15, 100)
x_obj, y_obj = np.meshgrid(x_obj, y_obj)
objectness = evaluate_gaussian_mixture(x_obj, y_obj, [
    {"mean": [2, 2], "cov": [[20, 0], [0, 20]], "weight": 0.4},
])
# Plotting the 3D contour
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.contour3D(x, y, z, 30, cmap=cm.coolwarm)
ax.contourf(x_obj, y_obj, objectness, 50, zdir='z', offset=np.min(z), cmap=cm.Greens)
ax.set_zlim(0, 0.1)
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.view_init(20, -15)
ax.set_zlabel('firing rate (Hz)')
ax.set_title('3D Contour Line Plot of Gaussian Mixture Landscape')
plt.tight_layout()
plt.savefig(join(figdir, "gaussian_mixture_contour_align.pdf"), format='pdf')
plt.savefig(join(figdir, "gaussian_mixture_contour_align.png"), format='png')
plt.show()
#%%
# Gaussian components
gaussians = [
    {"mean": [-7, -8], "cov": [[5, 4], [4, 8]], "weight": 0.6},
    {"mean": [11, 8], "cov": [[2, 1], [1, 3]], "weight": 0.4},
    {"mean": [-8, 7], "cov": [[5, 1], [1, 2]], "weight": 0.4},
]
x = np.linspace(-15, 15, 200)
y = np.linspace(-15, 15, 200)
x, y = np.meshgrid(x, y)
# Evaluating the mixture
z = evaluate_gaussian_mixture(x, y, gaussians)
x_obj = np.linspace(-15, 15, 100)
y_obj = np.linspace(-15, 15, 100)
x_obj, y_obj = np.meshgrid(x_obj, y_obj)
objectness = evaluate_gaussian_mixture(x_obj, y_obj, [
    {"mean": [2, 2], "cov": [[20, 0], [0, 20]], "weight": 0.4},
])
# Plotting the 3D contour
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.contour3D(x, y, z, 30, cmap=cm.coolwarm)
ax.contourf(x_obj, y_obj, objectness, 50, zdir='z', offset=np.min(z), cmap=cm.Greens)
ax.set_zlim(0, 0.1)
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.view_init(20, -15)
ax.set_zlabel('firing rate (Hz)')
ax.set_title('3D Contour Line Plot of Gaussian Mixture Landscape')
plt.tight_layout()
plt.savefig(join(figdir, "gaussian_mixture_contour_misalign.pdf"), format='pdf')
plt.savefig(join(figdir, "gaussian_mixture_contour_misalign.png"), format='png')
plt.show()



# Returning the paths to the saved files
# png_path, pdf_path