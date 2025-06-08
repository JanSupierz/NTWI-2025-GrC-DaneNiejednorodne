import time
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.impute import KNNImputer

from Loading import (
    load_folder_data,
    build_dataset_matrix,
)

def mask_random_values(data, mask_ratio=0.2, seed=42):
    """
    Return a copy of the data with a subset of values set to NaN for visualization.
    This allows clearer demonstration of imputation effects.
    """
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
    """Visualize original and filled matrices side-by-side with values."""
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

def fill_missing_knn_with_progress(data, k=5, chunk_size=600):
    n = data.shape[0]

    imputer = KNNImputer(
        n_neighbors=k,
        weights="uniform",
        metric="nan_euclidean"
    )
    imputer.fit(data)

    filled = np.empty_like(data)
    num_chunks = math.ceil(n / chunk_size)

    for chunk_idx in tqdm(range(num_chunks), desc="KNN Imputing", unit="chunk"):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n)
        filled[start:end] = imputer.transform(data[start:end])

    return filled

if __name__ == '__main__':
    folder_to_check = 'male'
    base_dir = 'dane'

    nr_rows_to_show = 10

    for subfolder, full_data_list, all_attrs in load_folder_data(base_dir, folder_to_check):
        print(f"Folder: {subfolder}")
        print(f"Combined dataset: {len(full_data_list)} records, {len(all_attrs)} attrs")

        #Build data matrix
        data_mat, attr_idx = build_dataset_matrix(all_attrs, full_data_list, nr_rows_to_show)

        #Fill missing values using knn_imputer
        filled_mat = fill_missing_knn_with_progress(data_mat, k=5)

        #Plot difference
        plot_imputation_comparison(data_mat, filled_mat, nr_rows_to_show)

        time.sleep(1)

        print("First 10 filled entries:")

        sorted_attrs = sorted(all_attrs, key=int)
        n_print = min(10, filled_mat.shape[0])

        for idx_row in range(n_print):
            entries = [
                f"'{a}': {filled_mat[idx_row, attr_idx[a]]:.6f}"
                for a in sorted_attrs
            ]
            print(f"Row {idx_row}: " + "{" + ", ".join(entries) + "}")

        print('\n')