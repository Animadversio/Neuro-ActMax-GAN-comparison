import numpy as np
import matplotlib.pyplot as plt
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid
figdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Schematics_alignment"


def plot_ux_xy_domain_view(ugrid, vgrid, param_func, target_func,
                           skip_every=4, underlay_alpha=0.2, isoline_alpha=0.7):
    # Mapping to x,y using param_func1
    x_uv, y_uv = param_func(ugrid, vgrid)
    # Applying the Gaussian function
    gaussian_values_uv = target_func(x_uv, y_uv)

    # Plotting
    figh = plt.figure(figsize=[16, 8])
    plt.subplot(121)
    # plot the iso-u iso-v param grid as lines
    for i in range(0, len(x_uv), skip_every):
        plt.plot(x_uv[i, :], y_uv[i, :], 'b-', lw=isoline_alpha)
        plt.plot(x_uv[:, i], y_uv[:, i], 'r-', lw=isoline_alpha)
    xlim = plt.xlim()
    ylim = plt.ylim()
    x_gaussian, y_gaussian = np.meshgrid(np.linspace(xlim[0], xlim[1], 200), np.linspace(ylim[0], ylim[1], 200))
    gaussian_values_xy = target_func(x_gaussian, y_gaussian)

    plt.contourf(x_gaussian, y_gaussian, gaussian_values_xy, 50, cmap="viridis", alpha=underlay_alpha)
    plt.xlabel('X', fontsize=18)
    plt.ylabel('Y', fontsize=18)
    plt.axis('image')
    plt.title('Gaussian Function and u, v param grid in x, y domain', fontsize=18)

    plt.subplot(122)
    # plot the target function value on the u,v grid
    plt.contourf(ugrid, vgrid, gaussian_values_uv, 50, cmap="viridis")
    plt.colorbar(label="Gaussian Value")
    plt.xlabel('u', fontsize=18)
    plt.ylabel('v', fontsize=18)
    plt.axis('image')
    plt.title('Gaussian Function seen in u,v Domain', fontsize=18)
    plt.tight_layout()
    plt.show()
    return figh



param_func1 = lambda u, v: (
    u + 0.6 * np.sin(u + v) + \
        0.1 * np.sin(2 * u - 3 * v) + \
        0.25 * np.sin(-3.7 * u + 2.4 * v) + \
        0.35 * np.sin( 0.8 * u - 0.6 * v),
    v + 0.6 * np.cos(u - v) + \
        0.1 * np.cos(3 * v - 2 * u) + \
        0.25 * np.cos(-3.7 * v + 2.4 * u) + \
        0.35 * np.cos( 0.8 * v + 0.6 * u) - 0.4
)
param_func2 = lambda u, v: (
    u + 0.4 * v,
    -0.4 * u + v
)

# Standard Gaussian function in x,y domain
gaussian_func = lambda x, y: np.exp(-(x**2 + y**2) / 2)

# Creating a grid in u,v space
u_values = np.linspace(-5, 5, 400)
v_values = np.linspace(-5, 5, 400)
u, v = np.meshgrid(u_values, v_values)

figh = plot_ux_xy_domain_view(u, v, param_func1, gaussian_func,
                              skip_every=4)
saveallforms(figdir, "param_grid_align1", figh)
figh = plot_ux_xy_domain_view(u, v, param_func1, gaussian_func,
                              skip_every=4, underlay_alpha=0.0)
saveallforms(figdir, "param_grid_align1_grid", figh)

#%%
param_func2 = lambda u, v: (
    u + 0.4 * v,
    -0.4 * u + v
)
figh = plot_ux_xy_domain_view(u, v, param_func2, gaussian_func,
                              skip_every=4)
saveallforms(figdir, "param_grid_align2", figh)
figh = plot_ux_xy_domain_view(u, v, param_func2, gaussian_func,
                              skip_every=4, underlay_alpha=0.0)
saveallforms(figdir, "param_grid_align2_grid", figh)


#%%
param_func3 = lambda u, v: (
    u + 0.2 * v,
    0.2 * u + v
)
figh = plot_ux_xy_domain_view(u, v, param_func3, gaussian_func,
                              skip_every=4)
saveallforms(figdir, "param_grid_align3", figh)
figh = plot_ux_xy_domain_view(u, v, param_func3, gaussian_func,
                              skip_every=4, underlay_alpha=0.0)
saveallforms(figdir, "param_grid_align3_grid", figh)

#%%
# Mapping to x,y using param_func2
x, y = param_func2(u, v)
gaussian_values_uv2 = gaussian_func(x, y)

plt.figure(figsize=[10, 8])
plt.contourf(u, v, gaussian_values_uv2, 50, cmap="viridis")
plt.colorbar(label="Gaussian Value")
plt.xlabel('u')
plt.ylabel('v')
plt.title('Gaussian Function in u,v Domain (Using param_func2)')
plt.show()

#%%
# Creating a grid in x,y space for Gaussian function
x_values = np.linspace(-5, 5, 400)
y_values = np.linspace(-5, 5, 400)
x_gaussian, y_gaussian = np.meshgrid(x_values, y_values)

# Gaussian function values
gaussian_values_xy = gaussian_func(x_gaussian, y_gaussian)

# Plotting Gaussian function
plt.figure(figsize=[10, 8])
plt.contourf(x_gaussian, y_gaussian, gaussian_values_xy, 50, cmap="viridis", alpha=0.6)

# Iso-u curves
for u_const in np.linspace(-5, 5, 50):
    v_curve = np.linspace(-5, 5, 100)
    x_curve, y_curve = param_func1(u_const, v_curve)
    plt.plot(x_curve, y_curve, 'r-', lw=2)

# Iso-v curves
for v_const in np.linspace(-5, 5, 50):
    u_curve = np.linspace(-5, 5, 100)
    x_curve, y_curve = param_func1(u_curve, v_const)
    plt.plot(x_curve, y_curve, 'b-', lw=2)

plt.colorbar(label="Gaussian Value")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gaussian Function with Iso-u (Red) and Iso-v (Blue) Curves (Using param_func1)')
plt.show()




#%%

# Parametrization 2
# x_uv_2_restricted = u_restricted + 0.25 * np.sin(u_restricted) * np.cos(v_restricted) + (u_restricted**2) / 50 - (v_restricted**2) / 50
# y_uv_2_restricted = v_restricted + 0.25 * np.cos(u_restricted) * np.sin(v_restricted) + (u_restricted * v_restricted) / 50

# Create figures for both restricted parametrizations
fig1, ax1 = plt.subplots(figsize=(8, 8))

# Plot Restricted Parametrization 1
for i in range(len(u_restricted)):
    ax1.plot(x_uv_1_restricted[i, :], y_uv_1_restricted[i, :], 'b-', lw=0.5)
    ax1.plot(x_uv_1_restricted[:, i], y_uv_1_restricted[:, i], 'r-', lw=0.5)


ax1.set_title('Restricted Grid (Parametrization 1)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.axis('equal')

plt.show()

# fig2, ax2 = plt.subplots(figsize=(8, 8))

# Plot Restricted Parametrization 2
# for i in range(len(u_restricted)):
#     ax2.plot(x_uv_2_restricted[i, :], y_uv_2_restricted[i, :], 'b-', lw=0.5)
#     ax2.plot(x_uv_2_restricted[:, i], y_uv_2_restricted[:, i], 'r-', lw=0.5)
#
# ax2.set_title('Restricted Grid (Parametrization 2)')
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.axis('equal')
#
# plt.show()

#%%
# Define the restricted u and v ranges to cover the range from -3*pi to 3*pi
u_restricted = np.linspace(-2 * np.pi, 2 * np.pi, 100)
v_restricted = np.linspace(-2 * np.pi, 2 * np.pi, 100)
u_restricted, v_restricted = np.meshgrid(u_restricted, v_restricted)

# Update the functions for the x and y coordinates with the restricted range

# Parametrization 1
x_uv_1_restricted = u_restricted + \
                    0.6 * np.sin(u_restricted + v_restricted) + \
                    0.1 * np.sin(2 * u_restricted - 3 * v_restricted) + \
                    0.25 * np.sin(-3.7 * u_restricted + 2.4 * v_restricted) +\
                    0.35 * np.sin(0.8 * u_restricted + 0.6 * v_restricted)
y_uv_1_restricted = v_restricted + \
                    0.6 * np.cos(u_restricted - v_restricted) + \
                    0.1 * np.cos(2 * v_restricted - 3 * u_restricted) + \
                    0.25 * np.cos(-3.7 * v_restricted + 2.4 * u_restricted) +\
                    0.35 * np.cos(0.8 * v_restricted + 0.6 * u_restricted)