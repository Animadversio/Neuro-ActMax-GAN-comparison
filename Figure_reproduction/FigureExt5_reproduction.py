# %%
# from circuit_toolkit.plot_utils import saveallforms
import matplotlib.pylab as plt
from os.path import join
import seaborn as sns
import pandas as pd


# %%
source_data_dir = r"/Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/Manuscript_BigGAN/Submissions/Manuscript_BigGAN - NatNeuro/2025-10-Accepted-In-Principle-Docs/SourceData/ExtendedFig5_Source_data"

# %% [markdown]
# ### Figure Ext 6 C
# %%
Adam_results_df_pruned = pd.read_csv(join(source_data_dir, "FigureExt5C_Adam_optim_score_as_function_of_depth_and_dim_logscale_quad_case.csv"))
sns.lineplot(data=Adam_results_df, y="gaussian_value", x="stack_size", hue="dimension", style="dimension", palette="Set2", markers=True)
plt.ylabel("optimized score")
plt.xlabel("RealNVP network depth")
plt.title("Adam | Optim score as function of depth and dimension")
# saveallforms(savedir, "Adam_optim_score_as_function_of_depth_and_dim_logscale_quad_case")

# %%
results_df = pd.read_csv(join(source_data_dir, "FigureExt5C_realnvp_layer_dim_cmaes_optim_result_logscale_quad_case.csv"))
sns.lineplot(data=results_df, y="gaussian_value", x="stack_size", hue="dimension", style="dimension", palette="Set2", markers=True)
plt.ylabel("optimized score")
plt.xlabel("RealNVP network depth")
plt.title("CMAES | Optim score as function of depth and dimension")

# %% [markdown]
# ### Figure Ext 6 DE
from scipy.stats import spearmanr
from collections import defaultdict
import numpy as np
def plot_forward_r2_scores_distribution(results_df, savedir=None, plot_type="boxplot", sharey=False, plot_vars=["Test_R2", 'Test_R2_class', 'Test_R2_noise'], GAN_model_name="BigGAN"):
    """
    Plots the distribution of Test R² scores for Non-ReLU and ReLU layers.


    Args:
        results_df (pd.DataFrame): DataFrame containing the R² scores.
        savedir (str): Directory to save the plot.

    Returns:
        None
    """
    corr_dict = defaultdict(lambda: defaultdict(list))
    pval_dict = defaultdict(lambda: defaultdict(list))
    for i in range(100):
        non_relu_df_match_num = results_df.query('not Layer.str.contains("relu")').groupby('Layer').sample(n=96).reset_index(drop=True)
        relu_df_match_num = results_df.query('Layer.str.contains("relu")').groupby('Layer').sample(n=96).reset_index(drop=True)
        for df_split, label in zip([non_relu_df_match_num, relu_df_match_num], 
                                ['Non-ReLU Layers (unit number matched)', 'ReLU Layers (unit number matched)']):
            # print(f"Iteration {i+1}: Spearman correlations for {label}:")
            for metric in plot_vars:
                corr, p_val = spearmanr(df_split['Layer_index'], df_split[metric])
                corr_dict[label][metric].append(corr)
                pval_dict[label][metric].append(p_val)
    
    melted_df = results_df.melt(id_vars=['Layer'], value_vars=plot_vars, var_name='Metric', value_name='Score')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1.5, 1]}, sharey=sharey)
    if plot_type == "boxplot":
        sns.boxplot(x='Layer', y='Score', hue='Metric', data=melted_df.query("not Layer.str.contains('relu')"), ax=ax1)
    elif plot_type == "violinplot":
        sns.violinplot(x='Layer', y='Score', hue='Metric', data=melted_df.query("not Layer.str.contains('relu')"), ax=ax1, cut=0)
    elif plot_type == "stripplot":
        sns.stripplot(x='Layer', y='Score', hue='Metric', data=melted_df.query("not Layer.str.contains('relu')"), ax=ax1, dodge=True, alpha=0.5)
    elif plot_type == "barplot":
        sns.barplot(x='Layer', y='Score', hue='Metric', data=melted_df.query("not Layer.str.contains('relu')"), ax=ax1)
    ax1.axhline(y=0.0, color='black', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_title(f'Test R² Scores Distribution (Non-ReLU Layers)')
    ax1.set_xlabel('Caffenet Layer')

    ax1.set_ylabel('Test R² Score')
    YLIM = ax1.get_ylim()
    ax1.set_ylim(max(-0.3, YLIM[0]), min(1.05, YLIM[1]))
    if plot_type == "boxplot":
        sns.boxplot(x='Layer', y='Score', hue='Metric', data=melted_df.query("Layer.str.contains('relu')"), ax=ax2)
    elif plot_type == "violinplot":
        sns.violinplot(x='Layer', y='Score', hue='Metric', data=melted_df.query("Layer.str.contains('relu')"), ax=ax2, cut=0)
    elif plot_type == "stripplot":
        sns.stripplot(x='Layer', y='Score', hue='Metric', data=melted_df.query("Layer.str.contains('relu')"), ax=ax2, dodge=True, alpha=0.5)
    elif plot_type == "barplot":
        sns.barplot(x='Layer', y='Score', hue='Metric', data=melted_df.query("Layer.str.contains('relu')"), ax=ax2)
    ax2.axhline(y=0.0, color='black', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_title(f'Test R² Scores Distribution (ReLU Layers)')
    ax2.set_xlabel('Caffenet Layer')
    ax2.set_ylabel('Test R² Score')
    # if ylim negative limit is too low, set it to -0.3
    YLIM = ax2.get_ylim()

    ax2.set_ylim(max(-0.3, YLIM[0]), min(1.05, YLIM[1]))
    for ax, label in zip([ax1, ax2], ['Non-ReLU Layers (unit number matched)', 'ReLU Layers (unit number matched)']):
        for mi, metric in enumerate(plot_vars):
            mean_corr = np.mean(corr_dict[label][metric])
            ax.text(0.25, 0.95 - 0.05*mi, f"{metric}: r={mean_corr:.3f}", color=f"C{mi}",
                    transform=ax.transAxes, fontsize=10, verticalalignment='top', )
            
    plt.suptitle(f"{GAN_model_name} latent code predicting Caffenet center units of each layer")
    plt.tight_layout()
    # if savedir is not None:
    #     saveallforms(savedir, "BigGAN_Caffenet_combined_boxplot")
    plt.show()
    return fig, corr_dict, pval_dict

# %%
results_df = pd.read_csv(join(source_data_dir, "FigureExt6D_BigGAN_Caffenet_latent_code_linearity_results.csv"))
fig, corr_dict, pval_dict = plot_forward_r2_scores_distribution(results_df, plot_type="boxplot", sharey=True, GAN_model_name="BigGAN");

# %%
results_df = pd.read_csv(join(source_data_dir, "FigureExt6E_DeePSim_FC6_Caffenet_latent_code_linearity_results.csv"))
fig, corr_dict, pval_dict = plot_forward_r2_scores_distribution(results_df, GAN_model_name="DeePSim", plot_type="boxplot", sharey=True, plot_vars=["Test_R2", 'Test_R2_top_Hess512', 'Test_R2_null_Hess512']);


