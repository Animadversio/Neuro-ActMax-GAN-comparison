# %%
# --- Setup ---
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
import pickle
import json

# --- Paths and Experiment Info ---
# Set these paths as needed for your environment
source_data_root = r"/Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/Manuscript_BigGAN/Submissions/Manuscript_BigGAN - NatNeuro/2025-10-Accepted-In-Principle-Docs/SourceData/"
source_data_dir = join(source_data_root, "Fig2_Source_data")
Expi = 155

# --- Figure 2D Functions ---
def _shaded_errorbar(x, y, yerr, label=None, color=None, **kwargs):
    plt.fill_between(x, y - yerr, y + yerr, alpha=0.3, label=None, color=color)
    plt.plot(x, y, color=color, label=label, **kwargs)

def stack_psth_plot(psth_arr, offset=200, titlestr=""):
    """Plot a sequence of PSTHs stacked vertically with offset"""
    blockN = psth_arr.shape[0]
    fig = plt.figure(figsize=[4, 0.5 * blockN + 1])
    for block in range(blockN):
        _shaded_errorbar(np.arange(200), offset * block + psth_arr[block, 0, :], psth_arr[block, 2, :], color="blue")
        _shaded_errorbar(np.arange(200), offset * block + psth_arr[block, 1, :], psth_arr[block, 3, :], color="red")
        plt.axhline(offset * block, color="black", alpha=0.6, linestyle="--")
    plt.axhline(offset * blockN, color="black", alpha=0.3, linestyle=":")
    plt.axhline(offset * (blockN + 1), color="black", alpha=0.3, linestyle=":")
    plt.yticks(np.arange(0, offset * blockN, offset) + offset / 2, 1 + np.arange(0, blockN))
    plt.ylim(0, offset * blockN + np.max(psth_arr[-1, :2, :]))
    plt.xlabel("Time (ms)")
    plt.ylabel(f"Firing Rate each block (events/s)   ({offset} evt/s between dashed lines)")
    if titlestr != "":
        fig.suptitle(titlestr)
    plt.tight_layout()
    plt.show()
    return fig

# ===========================
# ==== Reproducing Figure 2D ====
# ===========================

# --- Option 1: Load from PKL ---
print("=== Figure 2D: Load & plot from PKL ===")
src_pkl_path = join(source_data_dir, f"Figure2D_src_psth_Exp{Expi}.pkl")
with open(src_pkl_path, "rb") as f:
    fig2d_pkl_data = pickle.load(f)
psth_arr = fig2d_pkl_data["psth_arr"]
offset = fig2d_pkl_data["offset"]
expstr = fig2d_pkl_data["expstr"]
fig_2d_pkl = stack_psth_plot(psth_arr, offset=offset, titlestr=expstr)

# --- Option 2: Load from CSV ---
print("=== Figure 2D: Load & plot from CSV ===")
# Load all four slices and stack to (blockN, 4, 200)
csv_slices = []
for slice_label in ["DeePSim_mean", "DeePSim_sem", "BigGAN_mean", "BigGAN_sem"]:
    csv_path = os.path.join(source_data_dir, f"Figure2D_src_psth_Exp{Expi}_{slice_label}.csv")
    csv_slices.append(pd.read_csv(csv_path).values)
psth_arr_csv = np.stack(csv_slices, axis=1)  # (blockN, 4, 200)
# Load metadata
meta_json_path = os.path.join(source_data_dir, f"Figure2D_src_psth_Exp{Expi}_meta.txt")
with open(meta_json_path, "r") as f:
    fig2d_meta_json = json.load(f)
offset_csv = fig2d_meta_json["offset"]
expstr_csv = fig2d_meta_json["expstr"]
fig_2d_csv = stack_psth_plot(psth_arr_csv, offset=offset_csv, titlestr=expstr_csv)


# ===========================
# ==== Reproducing Figure 2B ====
# ===========================

# --- Option 1: Load from PKL ---
print("=== Figure 2B: Load & plot from PKL ===")
pkl_path = join(source_data_dir, f"Figure2B_src_resp_Exp{Expi}.pkl")
with open(pkl_path, "rb") as f:
    fig2b_pkl_data = pickle.load(f)
df_resp0_pkl = fig2b_pkl_data["thread0_resp_df"]
df_resp1_pkl = fig2b_pkl_data["thread1_resp_df"]
df_natref_resp0_pkl = fig2b_pkl_data["thread0_natref_resp_df"]
df_natref_resp1_pkl = fig2b_pkl_data["thread1_natref_resp_df"]
# Use title string (experiment info) from meta json (shared with Fig 2D)
with open(meta_json_path, "r") as f:
    fig2b_meta_json = json.load(f)
title_str = fig2b_meta_json["expstr"]
plt.figure(figsize=(5, 5))
sns.lineplot(data=df_resp0_pkl, x="gen", y="resp", errorbar="se", label="Thread 0 evoked resp (pkl)", color="blue")
sns.lineplot(data=df_resp0_pkl, x="gen", y="bsl", errorbar="se", label="Thread 0 baseline (pkl)", color="blue", linestyle=":")
sns.lineplot(data=df_natref_resp0_pkl, x="gen", y="resp", errorbar="se", label="Thread 0 natref (pkl)", color="blue", linestyle="-.")
sns.lineplot(data=df_resp1_pkl, x="gen", y="resp", errorbar="se", label="Thread 1 evoked resp (pkl)", color="red")
sns.lineplot(data=df_resp1_pkl, x="gen", y="bsl", errorbar="se", label="Thread 1 baseline (pkl)", color="red", linestyle=":")
sns.lineplot(data=df_natref_resp1_pkl, x="gen", y="resp", errorbar="se", label="Thread 1 natref (pkl)", color="red", linestyle="-.")
plt.xlabel("Generations")
plt.ylabel("Firing Rate (events/s)")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.title(f"{title_str}\n(Fig2B loaded from pkl)")
plt.show()

# --- Option 2: Load from CSV ---
print("=== Figure 2B: Load & plot from CSV ===")
# CSV files: DeePSim_thread_evol, BigGAN_thread_evol, DeePSim_thread_natref, BigGAN_thread_natref
csv_names = ["DeePSim_thread_evol", "BigGAN_thread_evol", 
             "DeePSim_thread_natref", "BigGAN_thread_natref"]
csv_files = [join(source_data_dir, f"Figure2B_src_resp_Exp{Expi}_{name}.csv") for name in csv_names]
df_resp0_csv = pd.read_csv(csv_files[0])
df_resp1_csv = pd.read_csv(csv_files[1])
df_natref_resp0_csv = pd.read_csv(csv_files[2])
df_natref_resp1_csv = pd.read_csv(csv_files[3])
plt.figure(figsize=(5, 5))
sns.lineplot(data=df_resp0_csv, x="gen", y="resp", errorbar="se", label="Thread 0 evoked resp (csv)", color="blue")
sns.lineplot(data=df_resp0_csv, x="gen", y="bsl", errorbar="se", label="Thread 0 baseline (csv)", color="blue", linestyle=":")
sns.lineplot(data=df_natref_resp0_csv, x="gen", y="resp", errorbar="se", label="Thread 0 natref (csv)", color="blue", linestyle="-.")
sns.lineplot(data=df_resp1_csv, x="gen", y="resp", errorbar="se", label="Thread 1 evoked resp (csv)", color="red")
sns.lineplot(data=df_resp1_csv, x="gen", y="bsl", errorbar="se", label="Thread 1 baseline (csv)", color="red", linestyle=":")
sns.lineplot(data=df_natref_resp1_csv, x="gen", y="resp", errorbar="se", label="Thread 1 natref (csv)", color="red", linestyle="-.")
plt.xlabel("Generations")
plt.ylabel("Firing Rate (events/s)")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.title(f"{title_str}\n(Fig2B loaded from CSVs)")
plt.show()
