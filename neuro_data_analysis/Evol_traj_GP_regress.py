import time
from tqdm import trange, tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from neuro_data_analysis.neural_data_lib import extract_evol_activation_array, load_neural_data, get_expstr
from core.utils.plot_utils import saveallforms

#%%
def fit_gpr(gen_vec, resp_vec, noise_std, kernel=None, n_restarts_optimizer=3, normalize_y=False):
    if kernel is None:
        kernel = C(resp_vec.std(), (1e-2, resp_vec.max()+1E-2)) * \
                 RBF(length_scale=2.0, length_scale_bounds=(1, 5)) + \
                 WhiteKernel(noise_std**2, (0.01, resp_vec.std()**2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer,
                                  normalize_y=normalize_y, alpha=1E-2)
    t0 = time.time()
    gp.fit(gen_vec[:, None], resp_vec)
    t1 = time.time()
    print(t1 - t0, "s")
    gen_arr = np.unique(gen_vec)
    gen_arr_finer = np.linspace(1, max(gen_arr), 200)
    traj_finer_mean, traj_finer_std = gp.predict(gen_arr_finer[:, None], return_std=True)
    traj_pred_mean,  traj_pred_std  = gp.predict(gen_arr[:, None], return_std=True)
    return gp, gen_arr_finer, traj_finer_mean, traj_finer_std, \
        gen_arr, traj_pred_mean, traj_pred_std


def visualize_gpr(t_finer, pred_finer, std_finer, color='b', label='Prediction', linestyle='-.', lw=2):
    plt.plot(t_finer, pred_finer, linestyle=linestyle, color=color, label=label, lw=lw)
    plt.fill_between(t_finer, pred_finer - std_finer, pred_finer + std_finer,
                        alpha=0.5, color=color)

savedir = Path(r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_traj_GPregress")
# plt.switch_backend('agg')
# switch back
plt.switch_backend( 'module://backend_interagg')
#%%
_, BFEStats = load_neural_data()
#%%
for Expi in trange(1, 191):
    if BFEStats[Expi - 1]["evol"] is None:
        continue
    titlestr = get_expstr(BFEStats, Expi)
    resp_arr0, bsl_arr0, gen_arr0, resp_vec0, bsl_vec0, gen_vec0 = \
        extract_evol_activation_array(BFEStats[Expi - 1], 0)
    resp_arr1, bsl_arr1, gen_arr1, resp_vec1, bsl_vec1, gen_vec1 = \
        extract_evol_activation_array(BFEStats[Expi - 1], 1)

    gen_arr = np.arange(1, len(resp_arr0) + 1)
    resp_mean0 = np.array([arr.mean() for arr in resp_arr0])
    resp_mean1 = np.array([arr.mean() for arr in resp_arr1])
    resp_std0 = np.array([arr.std() for arr in resp_arr0])
    resp_std1 = np.array([arr.std() for arr in resp_arr1])
    resp_sem0 = np.array([arr.std() / np.sqrt(arr.size) for arr in resp_arr0])
    resp_sem1 = np.array([arr.std() / np.sqrt(arr.size) for arr in resp_arr1])

    if len(resp_arr0[-1]) < 10 or len(resp_arr1[-1]) < 10:
        resp_mean0 = resp_mean0[:-1]
        resp_mean1 = resp_mean1[:-1]
        resp_std0 = resp_std0[:-1]
        resp_std1 = resp_std1[:-1]
        resp_sem0 = resp_sem0[:-1]
        resp_sem1 = resp_sem1[:-1]
        gen_arr = gen_arr[:-1]
        resp_vec0 = resp_vec0[gen_vec0 <= gen_arr[-1]]
        resp_vec1 = resp_vec1[gen_vec1 <= gen_arr[-1]]
        gen_vec0 = gen_vec0[gen_vec0 <= gen_arr[-1]]
        gen_vec1 = gen_vec1[gen_vec1 <= gen_arr[-1]]
    #%%
    gp_0, gen_finer0, traj_finer_mean0, traj_finer_std0, \
        gen_arr0, traj_pred_mean0, traj_pred_std0 = fit_gpr(
        gen_arr, resp_mean0, noise_std=resp_sem0.mean(), normalize_y=True)
    gp_1, gen_finer1, traj_finer_mean1, traj_finer_std1, \
        gen_arr1, traj_pred_mean1, traj_pred_std1 = fit_gpr(
        gen_arr, resp_mean1, noise_std=resp_sem1.mean(), normalize_y=True)
    np.savez(savedir / f"Exp{Expi:03d}_evol_traj_gpr_avg.npz",
            gen_arr=gen_arr, resp_mean0=resp_mean0, resp_mean1=resp_mean1,
            resp_sem0=resp_sem0, resp_sem1=resp_sem1,
            gen_finer0=gen_finer0, traj_finer_mean0=traj_finer_mean0, traj_finer_std0=traj_finer_std0,
            gen_finer1=gen_finer1, traj_finer_mean1=traj_finer_mean1, traj_finer_std1=traj_finer_std1,
            gen_arr0=gen_arr0, traj_pred_mean0=traj_pred_mean0, traj_pred_std0=traj_pred_std0,
            gen_arr1=gen_arr1, traj_pred_mean1=traj_pred_mean1, traj_pred_std1=traj_pred_std1,
            )

    plt.figure(figsize=[5, 5])
    plt.scatter(gen_vec0, resp_vec0, c='b', s=16, label='DeePSim', alpha=0.2)
    plt.scatter(gen_vec1, resp_vec1, c='r', s=16, label='BigGAN', alpha=0.2)
    plt.plot(gen_arr, resp_mean0, 'b-', label='DeePSim mean', lw=0.75)
    plt.plot(gen_arr, resp_mean1, 'r-', label='BigGAN mean', lw=0.75)
    visualize_gpr(gen_finer0, traj_finer_mean0, traj_finer_std0, color="blue", label="DeePSim Pred",
                  linestyle='-.', lw=2)
    visualize_gpr(gen_finer1, traj_finer_mean1, traj_finer_std1, color="red", label="BigGAN Pred",
                  linestyle='-.', lw=2)
    plt.ylabel("Firing rate (events/s)")
    plt.xlabel("Blocks")
    plt.title(titlestr)
    plt.legend()
    saveallforms(str(savedir), f"Exp{Expi:03d}_evol_traj_gpr_avg", )
    plt.show()

    #%%
    gp_0, gen_finer0, traj_finer_mean0, traj_finer_std0, \
        gen_arr0, traj_pred_mean0, traj_pred_std0 = fit_gpr(
        gen_vec0, resp_vec0, noise_std=resp_std0.mean(), normalize_y=True)
    gp_1, gen_finer1, traj_finer_mean1, traj_finer_std1, \
        gen_arr1, traj_pred_mean1, traj_pred_std1 = fit_gpr(
        gen_vec1, resp_vec1, noise_std=resp_std1.mean(), normalize_y=True)
    np.savez(savedir / f"Exp{Expi:03d}_evol_traj_gpr_all.npz",
                gen_vec0=gen_vec0, resp_vec0=resp_vec0, gen_vec1=gen_vec1, resp_vec1=resp_vec1,
                gen_finer0=gen_finer0, traj_finer_mean0=traj_finer_mean0, traj_finer_std0=traj_finer_std0,
                gen_finer1=gen_finer1, traj_finer_mean1=traj_finer_mean1, traj_finer_std1=traj_finer_std1,
                gen_arr0=gen_arr0, traj_pred_mean0=traj_pred_mean0, traj_pred_std0=traj_pred_std0,
                gen_arr1=gen_arr1, traj_pred_mean1=traj_pred_mean1, traj_pred_std1=traj_pred_std1,
                )

    plt.figure(figsize=[5, 5])
    plt.scatter(gen_vec0, resp_vec0, c='b', s=16, label='DeePSim', alpha=0.2)
    plt.scatter(gen_vec1, resp_vec1, c='r', s=16, label='BigGAN', alpha=0.2)
    plt.plot(gen_arr, resp_mean0, 'b-', label='DeePSim mean', lw=0.75)
    plt.plot(gen_arr, resp_mean1, 'r-', label='BigGAN mean', lw=0.75)
    visualize_gpr(gen_finer0, traj_finer_mean0, traj_finer_std0, color="blue", label="DeePSim Pred",
                  linestyle='-.', lw=2)
    visualize_gpr(gen_finer1, traj_finer_mean1, traj_finer_std1, color="red", label="BigGAN Pred",
                  linestyle='-.', lw=2)
    plt.ylabel("Firing rate (events/s)")
    plt.xlabel("Blocks")
    plt.title(titlestr)
    plt.legend()
    saveallforms(str(savedir), f"Exp{Expi:03d}_evol_traj_gpr_all", )
    plt.show()



#%%
gp_tmp, gen_finer_tmp, traj_finer_mean_tmp, traj_finer_std_tmp, \
    gen_arr_tmp, traj_pred_mean_tmp, traj_pred_std_tmp = fit_gpr(np.ones(10), np.random.randn(10), noise_std=0.3)

#%%
# Create a kernel with parameters
kernel = C(1.0, (.5, 20)) * RBF(2.0, (1, 5))
# Create a GaussianProcessRegressor object
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, normalize_y=False,
                              alpha=resp_sem0.mean())
t0 = time.time()
gp.fit(gen_vec0[:, None], resp_vec0)
t1 = time.time()
print(t1 - t0, "s")
gen_finer = np.linspace(1, max(gen_arr), 100)
traj_finer_mean0, traj_finer_std0 = gp.predict(gen_finer[:, None], return_std=True)
traj_pred_mean0,  traj_pred_std0  = gp.predict(gen_arr[:, None], return_std=True)
# gp.fit(gen_arr[:, None], resp_mean0)
# traj_mean0, traj_std0 = gp.predict(np.arange(1, max(gen_vec0) + 1)[:, None], return_std=True)


plt.figure(figsize=[5, 5])
plt.scatter(gen_vec0, resp_vec0, c='r', s=10, label='Observations', alpha=0.3)
plt.plot(gen_arr, resp_mean0, 'b-', label='Observation_mean', lw=0.75)
visualize_gpr(gen_finer, traj_finer_mean0, traj_finer_std0)
visualize_gpr(gen_arr, traj_pred_mean0, traj_pred_std0)
plt.show()
#%%
#%%
gp.fit(gen_vec1[:, None], resp_vec1)
#%%

# Fit the model to the data
gp.fit(X, Y)
t_new = np.array([[6]])  # the input must be a 2D array
predicted_position, std = gp.predict(t_new, return_std=True)
print(f"Predicted position at t = 6: {predicted_position}")
print(f"Standard deviation: {std}")