import numpy as np

def get_label_dist_in_mask(stats_df_all, mask, 
                           dataset_str="imagenet_valid", stat_str="kNNcoslabel", maxk=50):
    ''' 
    Find the label distribution in a mask of rows in stats_df_all. 
        return labels, counts, cnt_vec, freq_vec
    '''
    knn_coslabels = stats_df_all[mask][[f"{stat_str}_{i}_{dataset_str}" 
                                             for i in range(maxk)]].to_numpy()
    labels, counts = np.unique(knn_coslabels, return_counts=True)
    cnt_vec= np.zeros(1000, dtype=int)
    cnt_vec[labels] = counts
    freq_vec = cnt_vec / cnt_vec.sum()
    return labels, counts, cnt_vec, freq_vec


def get_label_dist_in_multimasks(stats_df_all, masks, label_strs, 
                                 dataset_str="imagenet_valid", stat_str="kNNcoslabel", maxk=50):
    """multi masks version of get_label_dist_in_mask"""
    labels_col = {}
    counts_col = {}
    cnt_vec_col = {}
    freq_vec_col = {}
    for mask, label_str in zip(masks, label_strs):
        labels, counts, cnt_vec, freq_vec = get_label_dist_in_mask(stats_df_all, mask, 
                                dataset_str=dataset_str, stat_str=stat_str, maxk=maxk)
        labels_col[label_str] = labels
        counts_col[label_str] = counts
        cnt_vec_col[label_str] = cnt_vec
        freq_vec_col[label_str] = freq_vec
    return labels_col, counts_col, cnt_vec_col, freq_vec_col


def print_labels_counts(labels, counts, label_dict, topk=30):
    """Print the topk frequent labels and their counts"""
    sorted_indices = np.argsort(-counts)
    sorted_labels = labels[sorted_indices]
    sorted_counts = counts[sorted_indices]
    for label, count in zip(sorted_labels[:topk], sorted_counts[:topk]):
        label_name = label_dict[label]
        print(f"{label_name}: {count}")


import pandas as pd
from scipy.stats import chi2_contingency
def test_print_difference(cnt1, cnt2, label_dict, offset=1, topN=10):
    # global contingency table of the two vectors. 
    chi2_stat, p_value, dof, expected = chi2_contingency(
        np.stack([cnt1, cnt2], axis=0) + offset)
    print(f"Chi2 test p-value: {p_value} chi2={chi2_stat} dof={dof}")
    change_ratio = (cnt2 + 1) / (cnt1 + 1)
    sort_idx = np.argsort(change_ratio)[::-1]
    print(f"Top {topN} freq increased labels")
    for i in range(topN):
        print(f"{i+1}: {label_dict[sort_idx[i]]} {change_ratio[sort_idx[i]]:.1f} | {cnt1[sort_idx[i]]}->{cnt2[sort_idx[i]]}")
    print()
    print(f"Top {topN} freq decreased labels")
    for i in range(topN):
        print(f"{i+1}: {label_dict[sort_idx[-i-1]]} {change_ratio[sort_idx[-i-1]]:.1f} | {cnt1[sort_idx[-i-1]]}->{cnt2[sort_idx[-i-1]]}")
        
    
def test_print_per_class_difference(cnt1, cnt2, label_dict, offset=1, ):
    chi2_all, p_val_all, dof_all, _ = chi2_contingency(
        np.stack([cnt1, cnt2], axis=0) + offset)
    print(f"Chi2 test p-value: {p_val_all:.1e} chi2={chi2_all:.2f} dof={dof_all}")
    n_classes = cnt1.shape[0]
    sum_cnt1 = cnt1.sum()
    sum_cnt2 = cnt2.sum()
    df_col = []
    for label in range(n_classes):
        # build 2x2 contingency table for each class 
        cont_max_class = np.array([[cnt1[label], sum_cnt1 - cnt1[label]],
                                   [cnt2[label], sum_cnt2 - cnt2[label]]])
        try:
            chi2_class, p_class, dof, expected = chi2_contingency(cont_max_class)
        except ValueError:
            # chi2 will throw an error if any of the values in the table are 0.
            # in that case, we can just skip this class as nan. (other handling methods e.g. adding offset?)
            chi2_class, p_class, dof = np.nan, np.nan, np.nan
        df_col.append({
            "label": label,
            "label_name": label_dict[label],
            "p_value": p_class,
            "chi2": chi2_class,
            "dof": dof,
            "cnt1": cnt1[label],
            "cnt2": cnt2[label],
            "ratio": (cnt2[label] / sum_cnt2) / (cnt1[label] / sum_cnt1),
        })
    chi2_df = pd.DataFrame(df_col)
    return chi2_df, chi2_all, p_val_all, dof_all

