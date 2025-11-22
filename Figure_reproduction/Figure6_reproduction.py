# %%
%load_ext autoreload
%autoreload 2

# %%

import math
import os
import sys
sys.path.append(r"/Users/binxuwang/Github/Neuro-ActMax-GAN-comparison")
import pandas as pd
from os.path import join
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


_parula_data = [[0.2081, 0.1663, 0.5292],
                [0.2116238095, 0.1897809524, 0.5776761905],
                [0.212252381, 0.2137714286, 0.6269714286],
                [0.2081, 0.2386, 0.6770857143],
                [0.1959047619, 0.2644571429, 0.7279],
                [0.1707285714, 0.2919380952, 0.779247619],
                [0.1252714286, 0.3242428571, 0.8302714286],
                [0.0591333333, 0.3598333333, 0.8683333333],
                [0.0116952381, 0.3875095238, 0.8819571429],
                [0.0059571429, 0.4086142857, 0.8828428571],
                [0.0165142857, 0.4266, 0.8786333333],
                [0.032852381, 0.4430428571, 0.8719571429],
                [0.0498142857, 0.4585714286, 0.8640571429],
                [0.0629333333, 0.4736904762, 0.8554380952],
                [0.0722666667, 0.4886666667, 0.8467],
                [0.0779428571, 0.5039857143, 0.8383714286],
                [0.079347619, 0.5200238095, 0.8311809524],
                [0.0749428571, 0.5375428571, 0.8262714286],
                [0.0640571429, 0.5569857143, 0.8239571429],
                [0.0487714286, 0.5772238095, 0.8228285714],
                [0.0343428571, 0.5965809524, 0.819852381],
                [0.0265, 0.6137, 0.8135],
                [0.0238904762, 0.6286619048, 0.8037619048],
                [0.0230904762, 0.6417857143, 0.7912666667],
                [0.0227714286, 0.6534857143, 0.7767571429],
                [0.0266619048, 0.6641952381, 0.7607190476],
                [0.0383714286, 0.6742714286, 0.743552381],
                [0.0589714286, 0.6837571429, 0.7253857143],
                [0.0843, 0.6928333333, 0.7061666667],
                [0.1132952381, 0.7015, 0.6858571429],
                [0.1452714286, 0.7097571429, 0.6646285714],
                [0.1801333333, 0.7176571429, 0.6424333333],
                [0.2178285714, 0.7250428571, 0.6192619048],
                [0.2586428571, 0.7317142857, 0.5954285714],
                [0.3021714286, 0.7376047619, 0.5711857143],
                [0.3481666667, 0.7424333333, 0.5472666667],
                [0.3952571429, 0.7459, 0.5244428571],
                [0.4420095238, 0.7480809524, 0.5033142857],
                [0.4871238095, 0.7490619048, 0.4839761905],
                [0.5300285714, 0.7491142857, 0.4661142857],
                [0.5708571429, 0.7485190476, 0.4493904762],
                [0.609852381, 0.7473142857, 0.4336857143],
                [0.6473, 0.7456, 0.4188],
                [0.6834190476, 0.7434761905, 0.4044333333],
                [0.7184095238, 0.7411333333, 0.3904761905],
                [0.7524857143, 0.7384, 0.3768142857],
                [0.7858428571, 0.7355666667, 0.3632714286],
                [0.8185047619, 0.7327333333, 0.3497904762],
                [0.8506571429, 0.7299, 0.3360285714],
                [0.8824333333, 0.7274333333, 0.3217],
                [0.9139333333, 0.7257857143, 0.3062761905],
                [0.9449571429, 0.7261142857, 0.2886428571],
                [0.9738952381, 0.7313952381, 0.266647619],
                [0.9937714286, 0.7454571429, 0.240347619],
                [0.9990428571, 0.7653142857, 0.2164142857],
                [0.9955333333, 0.7860571429, 0.196652381],
                [0.988, 0.8066, 0.1793666667],
                [0.9788571429, 0.8271428571, 0.1633142857],
                [0.9697, 0.8481380952, 0.147452381],
                [0.9625857143, 0.8705142857, 0.1309],
                [0.9588714286, 0.8949, 0.1132428571],
                [0.9598238095, 0.9218333333, 0.0948380952],
                [0.9661, 0.9514428571, 0.0755333333],
                [0.9763, 0.9831, 0.0538]]

from matplotlib.colors import ListedColormap
parula = ListedColormap(_parula_data, name='parula')
# %%
source_data_root = r"/Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/Manuscript_BigGAN/Submissions/Manuscript_BigGAN - NatNeuro/2025-10-Accepted-In-Principle-Docs/SourceData/"
source_data_dir = join(source_data_root, "Fig6_Source_data")
# os.makedirs(source_data_dir, exist_ok=True)

# %% [markdown]
# ### Figure 6C

# %% [markdown]
# Beto 11.0 Beto-07092020-003 prefchan 5.0 unit 2.0 class 0 axis

# %%
def regression_combined_plot(x_train, y_train, gauss_fit_results, ols_fit_results, gpr_fit_results, anova_results=None, title_str="", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(x_train, y_train, 'o', label='data', alpha=0.5)
    tmp_df = pd.DataFrame({"x": x_train, "y": y_train})
    tmp_df = tmp_df.groupby("x").agg({"y": ["mean", "std"]}).reset_index()
    ax.errorbar(tmp_df["x"], tmp_df["y"]["mean"], yerr=tmp_df["y"]["std"], label='mean y', marker='D', color="C1", markersize=5, capsize=5, alpha=0.5, linestyle="")
    x_eval = np.linspace(x_train.min(), x_train.max(), 100)
    gauss_params = gauss_fit_results["params"]
    ols_params = ols_fit_results["params"]
    gpr_kernel = gpr_fit_results["gpr"].kernel_
    gpr_kernel_str = str(gpr_kernel).replace("length_scale", "len").replace("noise_level", "noise").replace("RBF", "RBF").replace("WhiteKernel", "White")
    if gauss_params is not None:
        ax.plot(x_eval, gaussian_with_baseline(x_eval, *gauss_params), label='Gaussian curvefit', color="k")
    ax.plot(x_eval, ols_params["slope"] * x_eval + ols_params["intercept"], label='OLS fit', color="magenta", linestyle="--")
    ax.plot(gpr_fit_results["x_eval"], gpr_fit_results["y_mean"], label='GPR mean', color="red")
    ax.fill_between(
        gpr_fit_results["x_eval"],
        gpr_fit_results["y_mean"] - gpr_fit_results["y_std"],
        gpr_fit_results["y_mean"] + gpr_fit_results["y_std"],
        alpha=0.2,
        label="GPR std",
        color="red"
    )
    ax.legend()
    title_caption = ""
    if title_str:
        title_caption += f"{title_str}\n"
    if gauss_params is not None:
        title_caption += f"Gauss fit: Ampl.: {gauss_params[0]:.2f} Mean: {gauss_params[1]:.2f} Std: {gauss_params[2]:.2f} Bsl: {gauss_params[3]:.2f} [R2: {gauss_fit_results['explained_variance']:.2f}]\n"
    if ols_params is not None:
        title_caption += f"OLS fit: Slope: {ols_fit_results['params']['slope']:.2f} Intercept: {ols_fit_results['params']['intercept']:.2f} [R2: {ols_fit_results['explained_variance']:.2f}]\n"
    if gpr_fit_results is not None:
        title_caption += f"GPR fit: {gpr_kernel_str} [R2: {gpr_fit_results['explained_variance']:.2f}]"
    if anova_results is not None:
        title_caption += f"\nANOVA: {anova_results['stats_str']}"
    ax.set_title(title_caption)
    fig.tight_layout()
    return fig

# %%
from sklearn.linear_model import LinearRegression

def linear_regression_fitting(x, y, visualize=True):
    # Initialize and fit the linear regression model
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    # Predict using the fitted model
    y_fit = model.predict(x.reshape(-1, 1))
    # Compute the explained variance (R^2 score)
    explained_variance = model.score(x.reshape(-1, 1), y)
    if visualize:
        plot_linear_regression_fit(x, y, model)
    
    return {
        "params": {
            "slope": model.coef_[0],
            "intercept": model.intercept_
        },
        "param_cov": None,  # Covariance matrix not available for simple linear regression
        "explained_variance": explained_variance
    }

def plot_linear_regression_fit(x, y, model):
    plt.plot(x, y, 'o', label='data', alpha=0.5)
    # Generate values for plotting the regression line
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = model.predict(x_fit.reshape(-1, 1))
    plt.plot(x_fit, y_fit, label='OLS fit', color="k")
    plt.legend()

# %%
from scipy.optimize import curve_fit
# Define Gaussian function with baseline
def gaussian_with_baseline(x, amplitude, mean, stddev, baseline):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) + baseline


def gaussian_curve_fitting(x, y, visualize=True, constrained=True):
    df_tmp = pd.DataFrame({"x": x, "y": y})
    df_tmp = df_tmp.groupby("x").agg({"y": ["mean", "std", "sem"]}).reset_index()
    x_uniq = df_tmp["x"]
    y_mean = df_tmp["y"]["mean"]
    y_std = df_tmp["y"]["std"]
    y_sem = df_tmp["y"]["sem"]
    # Initialize parameters
    x_range = np.max(x_uniq) - np.min(x_uniq)
    initial_amplitude = np.max(y_mean) - np.min(y_mean)
    initial_mean = x_uniq[np.argmax(y_mean)] # TODO: fix this, use the denoised / mean y data.
    initial_stddev = x_range / 4  # Arbitrary initial standard deviation
    initial_baseline = np.min(y_mean)

    initial_params = [initial_amplitude, initial_mean, initial_stddev, initial_baseline]
    if constrained:
        lower_bounds = [                 0.0,  np.min(x_uniq),         0.01,             0.0]
        upper_bounds = [1.25 * np.max(y_mean),  np.max(x_uniq),  x_range * 2,  np.max(y_mean)]
        bounds = (lower_bounds, upper_bounds)
    else:
        bounds = None
    try:
        params, param_cov = curve_fit(gaussian_with_baseline, x, y, p0=initial_params, bounds=bounds)
    except RuntimeError as e:
        print("Curve fitting failed:", e)
        params = None
        param_cov = None
    # compute the explained variance
    if params is not None:
        y_fit = gaussian_with_baseline(x, *params)
        explained_variance = 1 - np.var(y - y_fit) / np.var(y)
    else:
        explained_variance = None
    if visualize and params is not None:
        plot_gaussian_curve_fitting(x, y, params, )
    return {"params": params, 
            "param_cov": param_cov, 
            "explained_variance": explained_variance}


def plot_gaussian_curve_fitting(x, y, params,):
    plt.plot(x, y, 'o', label='data', alpha=0.5)
    # plot the mean y value per x value and error bars
    tmp_df = pd.DataFrame({"x": x, "y": y})
    tmp_df = tmp_df.groupby("x").agg({"y": ["mean", "std"]}).reset_index()
    plt.errorbar(tmp_df["x"], tmp_df["y"]["mean"], yerr=tmp_df["y"]["std"], fmt='D', label='mean y', marker='D', markersize=5, capsize=5, alpha=0.5)
    x_fit = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_fit, gaussian_with_baseline(x_fit, *params), label='Gaussian curvefit', color="k")
    plt.legend()



# %%
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
# Suppose x_train is shape (N,1), y_train is shape (N,) or (N,1)
# Even if some x's are identical, the multiple rows with the same x
# will just appear as repeated entries in x_train.
def gaussian_process_regression(x_train, y_train, n_eval_points=100, visualize=True):
    """
    Perform Gaussian process regression on the input data.
    
    Args:
        x_train: Training x values, shape (N,1)
        y_train: Training y values, shape (N,) or (N,1)
        n_eval_points: Number of evaluation points to predict
    
    Returns:
        gpr: Gaussian process regressor
        x_eval: Evaluation x values
        y_mean: Predicted mean values
        y_std: Predicted standard deviation values
    """
    # Set length_scale based on the range of x_train
    length_scale = (x_train.max() - x_train.min()) / 10 # 0.08
    df_tmp = pd.DataFrame({"x": x_train, "y": y_train})
    df_tmp = df_tmp.groupby("x").agg({"y": ['var']}).reset_index()
    noise_var = df_tmp["y"]["var"].mean()
    y_var = y_train.var()
    if np.isnan(noise_var):
        # note this is possible when there is not enough data to estimate the noise variance.
        noise_var = y_var
    noise_level = noise_var * 0.1
    # Set noise_level based on the variance of y_train
    # noise_level = np.var(y_train) * 0.5 # note the noise level is variance.

    kernel = ConstantKernel(y_var, (y_var * 1e-2, y_var*1e2)) * \
        RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2)) \
         + WhiteKernel(noise_level=noise_level,
                       noise_level_bounds=(noise_var * 1e-2, noise_var * 1e1))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=10) # 
    gpr.fit(x_train.reshape(-1, 1), y_train.reshape(-1))
    # compute the explained variance
    y_fit = gpr.predict(x_train.reshape(-1, 1))
    explained_variance = 1 - np.var(y_train - y_fit) / np.var(y_train)
    # Predict at new points
    x_eval = np.linspace(x_train.min(), x_train.max(), n_eval_points)
    y_mean, y_std = gpr.predict(x_eval.reshape(-1, 1), return_std=True)
    y_mean = y_mean.reshape(-1)
    if visualize:
        plot_gaussian_process_regression(x_train, y_train, x_eval, y_mean, y_std)
    return {"gpr": gpr, "explained_variance": explained_variance, "x_eval": x_eval, "y_mean": y_mean, "y_std": y_std, }


def plot_gaussian_process_regression(x_train, y_train, x_eval, y_mean, y_std,):
    plt.plot(x_train, y_train, 'o', label='data', alpha=0.5)
    plt.plot(x_eval, y_mean, label='GPR mean', color="k")
    plt.fill_between(x_eval, y_mean - y_std, y_mean + y_std, alpha=0.2, label="GPR std")
    plt.legend()


# %%

import statsmodels.api as sm
from statsmodels.formula.api import ols
def anova_test_df(df, x_col="lin_dist", y_col="pref_unit_resp"):
    anova_results = {}
    try:
        model = ols(f'{y_col} ~ C({x_col})', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        F_value = anova_table.loc[f'C({x_col})', 'F']
        p_value = anova_table.loc[f'C({x_col})', 'PR(>F)']
        stats_str = f"F-val: {F_value:.2f} | p-val: {p_value:.1e}"
        anova_results["F_value"] = F_value
        anova_results["p_value"] = p_value
        anova_results["stats_str"] = stats_str
        anova_results["anova_table"] = anova_table
        anova_results["error"] = None
    except Exception as e:
        print(f"Error performing ANOVA : {e}")
        anova_results["F_value"] = np.nan
        anova_results["p_value"] = np.nan
        anova_results["stats_str"] = ""
        anova_results["anova_table"] = None
        anova_results["error"] = e
    return anova_results

# %%
sgtr_resp_df = pd.read_csv(join(source_data_dir, "Figure6C_src_B07092020_sgtr_resp_df.csv"))

title_str = "B-07092020-003 | Pref Chan 5B"
sgtr_resp_at_origin = sgtr_resp_df.query(f"lin_dist == 0.0") # if some axis does not have origin stats, add it. 
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
space, eig_id = "class", 0
sgtr_resp_per_axis = sgtr_resp_df.query(f"space_name == @space and eig_id == @eig_id")
if 0.0 not in sgtr_resp_per_axis["lin_dist"].unique():
    print(f"Adding origin trials for {space} {eig_id}")
    sgtr_resp_per_axis = pd.concat([sgtr_resp_per_axis, sgtr_resp_at_origin])
gauss_fit_results = gaussian_curve_fitting(sgtr_resp_per_axis["lin_dist"].values, sgtr_resp_per_axis["pref_unit_resp"].values, visualize=False)
ols_fit_results = linear_regression_fitting(sgtr_resp_per_axis["lin_dist"].values, sgtr_resp_per_axis["pref_unit_resp"].values, visualize=False)
gpr_fit_results = gaussian_process_regression(sgtr_resp_per_axis["lin_dist"].values, sgtr_resp_per_axis["pref_unit_resp"].values, visualize=False)
anova_results = anova_test_df(sgtr_resp_per_axis, x_col="lin_dist", y_col="pref_unit_resp")
regression_combined_plot(sgtr_resp_per_axis["lin_dist"].values, sgtr_resp_per_axis["pref_unit_resp"].values, gauss_fit_results, ols_fit_results, gpr_fit_results, 
                            anova_results=anova_results, title_str=f"{title_str} | {space} {eig_id} axis", ax=ax)
plt.xlabel("Linear Distance")
plt.ylabel("Response (events/s)")
ax.set_title(ax.get_title(), fontsize=10)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Figure 6D

# %% [markdown]
#  Caos-12102024-003 | Pref Channel 68B 

# %%
def plot_heatmap(grouped, space, ax, CLIM, annot=True, fmt=".1f"):
    """Plot heatmap for a given space"""
    space_data = grouped[grouped['space_name'] == space]
    pivot_table = space_data.pivot(index='eig_id', columns='lin_dist', values='pref_unit_resp')
    pivot_table = pivot_table.astype(float)
    if pivot_table.empty:
        return
    plt.sca(ax)
    sns.heatmap(pivot_table, annot=annot, fmt=fmt, cmap=parula, 
                cbar_kws={'label': 'Preferred Unit Response'}, ax=ax, vmin=CLIM[0], vmax=CLIM[1])
    plt.title(f'Heatmap of Preferred Unit Response for Space: {space}')
    plt.xlabel('Linear Distance')
    plt.ylabel('Eigenvalue ID')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.axis('image')


# %%
# from core.utils.colormap_matlab import parula, viridis
# %%
# avgresp_df = pd.read_pickle(join(source_data_dir, "Figure6D_src_prefchan_avgresp_df.pkl"))
avgresp_df = pd.read_csv(join(source_data_dir, "Figure6D_src_prefchan_avgresp_df.csv"))
ephysFN = avgresp_df.iloc[0]["ephysFN"]
prefchan_str = avgresp_df.iloc[0]["prefchan_str"]
CLIM = np.quantile(avgresp_df['pref_unit_resp'], [0.02, 0.98])
figh, ax = plt.subplots(1, 1, figsize=(6.5, 6))
space = "class" # can also be "noise"
fig = plot_heatmap(avgresp_df, space, ax, CLIM)
plt.title(f'{space} space')
cbar = ax.figure.axes[-1]  # Get the last axis, which is the colorbar
cbar.set_ylabel("Response (events/s)")
plt.suptitle(f'Preferred Unit Response for Different Spaces and Eigenvectors \n {ephysFN} | Pref Channel {prefchan_str} ')
plt.show()

# %%

ephysFN = avgresp_df.iloc[0]["ephysFN"]
prefchan_str = avgresp_df.iloc[0]["prefchan_str"]
CLIM = np.quantile(avgresp_df['pref_unit_resp'], [0.02, 0.98])
figh, axs = plt.subplots(1, 2, figsize=(13, 6))
for ax, space in zip(axs, ['class', 'noise']):
    fig = plot_heatmap(avgresp_df, space, ax, CLIM)
    plt.title(f'{space} space')
cbar = ax.figure.axes[-1]  # Get the last axis, which is the colorbar
cbar.set_ylabel("Response (events/s)")
plt.suptitle(f'Preferred Unit Response for Different Spaces and Eigenvectors \n {ephysFN} | Pref Channel {prefchan_str} ')
plt.show()
#%%
avgresp_df = pd.read_csv(join(source_data_dir, "Figure6D_src_prefchan_avgresp_df.csv"))
ephysFN = avgresp_df.iloc[0]["ephysFN"]
prefchan_str = avgresp_df.iloc[0]["prefchan_str"]
CLIM = np.quantile(avgresp_df['pref_unit_resp'], [0.02, 0.98])
figh, ax = plt.subplots(1, 1, figsize=(4.5, 4))
space = "class" # can also be "noise"
fig = plot_heatmap(avgresp_df, space, ax, CLIM, annot=False)
plt.title(f'{space} space')
cbar = ax.figure.axes[-1]  # Get the last axis, which is the colorbar
cbar.set_ylabel("Response (events/s)")
plt.suptitle(f'Preferred Unit Response for Different Spaces and Eigenvectors \n {ephysFN} | Pref Channel {prefchan_str} ')
plt.tight_layout()
plt.show()
# %% [markdown]
# ### Figure 6D alternative
# %%
# from core.utils.colormap_matlab import parula, viridis
# avgresp_df = pd.read_pickle(join(source_data_dir, "Figure6D_src_prefchan_avgresp_df_failevol.pkl"))
avgresp_df = pd.read_csv(join(source_data_dir, "Figure6D_src_prefchan_avgresp_df_failevol.csv"))
ephysFN = avgresp_df.ephysFN.iloc[0]
prefchan_str = avgresp_df.prefchan_str.iloc[0]
CLIM = np.quantile(avgresp_df['pref_unit_resp'], [0.02, 0.98])
figh, axs = plt.subplots(1, 1, figsize=(4.5, 4))
ax = axs
space = "class"
plot_heatmap(avgresp_df.query("eig_id in [0,1,2,3,6,9,13,21,30,60]"), space, ax, CLIM, annot=False)
plt.title(f'{space} space')
cbar = ax.figure.axes[-1]  # Get the last axis, which is the colorbar
cbar.set_ylabel("Response (events/s)")
plt.suptitle(f'Preferred Unit Response for Different Spaces and Eigenvectors \n {ephysFN} | Pref Channel {prefchan_str} ')
plt.tight_layout()
# saveallforms(source_data_dir, f"{ephysFN}_U{prefchan_str}_failed_tuning_maps_noannot_prune")
plt.show()

# %% [markdown]
# ### Figure 6E

# %%

def is_monotonic_np(y_values: np.ndarray, EPS: float = 0.01) -> bool:
    """
    Determines if the given NumPy array y_values is monotonic 
    (entirely non-increasing or non-decreasing).

    Args:
        y_values (np.ndarray): Sequence of numerical y-values.
        EPS (float): Tolerance for considering two values as equal (default is 0.01).

    Returns:
        bool: True if the sequence is monotonic, False otherwise.
    """
    if y_values.size <= 1:
        return True  # Empty or single-element sequences are monotonic

    diffs = np.diff(y_values)

    # Check for non-decreasing: all diffs >= -EPS
    non_decreasing = np.all(diffs >= -EPS)

    # Check for non-increasing: all diffs <= EPS
    non_increasing = np.all(diffs <= EPS)

    return non_decreasing or non_increasing


def is_unimodal_np(y_values: np.ndarray, EPS: float = 0.01) -> bool:
    """
    Determines if the given NumPy array y_values is unimodal 
    (has a single peak).

    Args:
        y_values (np.ndarray): Sequence of numerical y-values.
        EPS (float): Tolerance for considering two values as equal (default is 0.01).

    Returns:
        bool: True if the sequence is unimodal, False otherwise.
    """
    if y_values.size == 0:
        return False  # Empty sequence is not unimodal
    if y_values.size == 1:
        return True  # Single element is trivially unimodal

    # Find the index of the first occurrence of the maximum value
    peak_index = np.argmax(y_values)

    # Split the array into two parts: before and after the peak
    before_peak = y_values[:peak_index + 1]
    after_peak = y_values[peak_index:]

    # Compute differences for both parts
    diffs_before = np.diff(before_peak)
    diffs_after = np.diff(after_peak)

    # Check if the sequence before the peak is non-decreasing
    is_non_decreasing = np.all(diffs_before >= -EPS)

    # Check if the sequence after the peak is non-increasing
    is_non_increasing = np.all(diffs_after <= EPS)

    return is_non_decreasing and is_non_increasing



def is_bellshaped_np(y_values: np.ndarray, EPS: float = 0.01) -> bool:
    """
    Determines if the given NumPy array y_values is bellshaped 
    (has a single peak and is unimodal).

    Args:
        y_values (np.ndarray): Sequence of numerical y-values.
        EPS (float): Tolerance for considering two values as equal (default is 0.01).

    Returns:
        bool: True if the sequence is unimodal, False otherwise.
    """
    if y_values.size == 0:
        return False  # Empty sequence is not unimodal
    if y_values.size == 1:
        return True  # Single element is trivially unimodal

    # Find the index of the first occurrence of the maximum value
    peak_index = np.argmax(y_values)
    if peak_index == 0 or peak_index == y_values.size - 1:
        return False # the peak is at the boundary, not bellshaped
    # Split the array into two parts: before and after the peak
    before_peak = y_values[:peak_index + 1]
    after_peak = y_values[peak_index:]

    # Compute differences for both parts
    diffs_before = np.diff(before_peak)
    diffs_after = np.diff(after_peak)

    # Check if the sequence before the peak is non-decreasing
    is_non_decreasing = np.all(diffs_before >= -EPS)

    # Check if the sequence after the peak is non-increasing
    is_non_increasing = np.all(diffs_after <= EPS)

    return is_non_decreasing and is_non_increasing
# %% [markdown]
# #### Test loading and reproduction
# %%
tuning_fitting_stats_table_sel = pd.read_csv(join(source_data_dir, "Figure6E_src_tuning_shape_fitting_stats_synopsis_selcolumn.csv"))
# Modified code to plot multiple bars side by side
# Filter the data based on the given conditions
filtered_data = tuning_fitting_stats_table_sel.query("anova_p_value < 0.01 and is_common_axis")
# Reshape the data to long format for better plotting
melted_data = filtered_data.melt(
    id_vars='is_BigGAN_evol_success',
    value_vars=['gpr_y_is_bellshaped', 'gpr_y_is_monotonic', 'gpr_y_is_unimodal'],
    var_name='Metric',
    value_name='Value'
)

# Compute counts for annotations and sort to match the barplot order
annotation_data = melted_data.groupby(['is_BigGAN_evol_success', 'Metric']).agg(
    True_Count=('Value', 'sum'),
    Total_Count=('Value', 'count'),
    Ratio=('Value', 'mean'),
).reset_index().sort_values(['is_BigGAN_evol_success', 'Metric'])

plt.figure(figsize=(4.5, 6))
# Create a bar plot with multiple bars side by side
ax = sns.barplot(
    data=melted_data,
    x='is_BigGAN_evol_success',
    y='Value',
    hue='Metric',
    order=["True", "False"],
    hue_order=["gpr_y_is_bellshaped", "gpr_y_is_monotonic", ],
    errorbar=("ci", 95),
)

plt.ylabel("Fraction of tuning axis")
plt.xlabel("BigGAN evolution success")

# Add annotations of num of True / total on top of each bar
for p in ax.patches:
    height = p.get_height()
    # get row from annotation_data with same Ratio 
    row = annotation_data[annotation_data['Ratio'] == height]
    if row.empty: # last 
        continue
    row = row.iloc[0]
    ax.annotate(
        f"{row['Ratio']:.2f}\n{int(row['True_Count'])}/{int(row['Total_Count'])}",
        (p.get_x() + p.get_width() / 2., p.get_height() / 2),
        ha='center', va='center', color="white", fontweight="bold", 
    )
    
plt.suptitle("Tuning curve shape type as a function of BigGAN evolution success\n[signif axis, ANOVA p < 0.01, tuning axis = [-0.4, 0.4]]")
# saveallforms(syndir, "BigGAN_evol_success_tuning_shape_type_barplot_simple")
# Display the plot
plt.show()

# %%



