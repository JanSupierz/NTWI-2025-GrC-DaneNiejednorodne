import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer
import skfuzzy as fuzz

from Loading import (
    load_folder_data,
    build_dataset_matrix,
)

def mask_random_values(data, mask_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    data_copy = data.copy()
    mask = ~np.isnan(data_copy)
    indices = np.argwhere(mask)
    rng.shuffle(indices)
    n_mask = int(mask_ratio * indices.shape[0])

    selected = indices[:n_mask]
    for i, j in selected:
        data_copy[i, j] = np.nan

    return data_copy

def plot_imputation_comparison(original, filled, max_rows=20):
    original = original[:max_rows, :]
    filled = filled[:max_rows, :]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, matrix, title in zip(axes, [original, filled], ["Before Imputation", "After Imputation"]):
        sns.heatmap(
            matrix,
            ax=ax,
            annot=True,
            fmt=".1f",
            cmap="viridis",
            cbar=False,
            mask=np.isnan(matrix) if title == "Before Imputation" else None
        )
        ax.set_title(title)
        ax.set_xlabel("Features")
        ax.set_ylabel("Samples")

    plt.tight_layout()
    plt.show()

def fill_missing_fcm_knn(data, k=5, n_clusters=3):
    print("Clustering with Fuzzy C-Means...")

    # Use only complete rows for fitting
    valid_mask = ~np.isnan(data).any(axis=1)
    valid_data = data[valid_mask].T

    # Fit FCM
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        valid_data, c=n_clusters, m=2, error=0.005, maxiter=1000
    )

    # Predict fuzzy memberships for all rows (excluding rows with all NaNs)
    incomplete_mask = np.isnan(data).all(axis=1)
    to_predict = data[~incomplete_mask].T

    u_all, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        to_predict, cntr, m=2, error=0.005, maxiter=1000
    )
    hard_clusters_partial = np.argmax(u_all, axis=0)

    # Build full hard cluster assignment (skip rows with all NaNs)
    hard_clusters = np.full(data.shape[0], -1)
    hard_clusters[~incomplete_mask] = hard_clusters_partial

    # Cluster-wise imputation
    filled = np.empty_like(data)
    for cluster_id in range(n_clusters):
        cluster_mask = hard_clusters == cluster_id
        cluster_data = data[cluster_mask]

        if cluster_data.shape[0] == 0:
            continue

        print(f"Imputing cluster {cluster_id} with {cluster_data.shape[0]} rows...")
        imputer = KNNImputer(n_neighbors=k, weights="uniform", metric="nan_euclidean")
        filled_cluster = imputer.fit_transform(cluster_data)
        filled[cluster_mask] = filled_cluster

    return filled

if __name__ == '__main__':
    folder_to_check = 'male'
    base_dir = 'dane'
    nr_rows_to_show = 10

    for subfolder, full_data_list, all_attrs in load_folder_data(base_dir, folder_to_check):
        print(f"Folder: {subfolder}")
        print(f"Combined dataset: {len(full_data_list)} records, {len(all_attrs)} attrs")

        data_mat, attr_idx = build_dataset_matrix(all_attrs, full_data_list, nr_rows_to_show)

        filled_mat = fill_missing_fcm_knn(data_mat, k=5, n_clusters=3)

        plot_imputation_comparison(data_mat, filled_mat, nr_rows_to_show)

        time.sleep(1)

        print("First 10 filled entries:")
        sorted_attrs = sorted(all_attrs, key=int)
        n_print = min(10, filled_mat.shape[0])

        for idx_row in range(n_print):
            entries = [f"'{a}': {filled_mat[idx_row, attr_idx[a]]:.6f}" for a in sorted_attrs]
            print(f"Row {idx_row}: " + "{" + ", ".join(entries) + "}")

        print('\n')
