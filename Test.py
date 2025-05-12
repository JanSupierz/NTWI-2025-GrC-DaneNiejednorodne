import os
import numpy as np
import math

from tqdm import tqdm
from sklearn.impute import KNNImputer

from Loading import (
    load_folder_data,
    build_dataset_matrix,
)

def fill_missing_knn_with_progress(data_mat, k=5, chunk_size=600):
    """
    Fill missing values via k-NN averaging over available dims,
    using scikit-learn’s KNNImputer—but transforming in chunks
    so we can show progress.
    """
    n = data_mat.shape[0]
    imputer = KNNImputer(
        n_neighbors=k,
        weights="uniform",
        metric="nan_euclidean"
    )
    # Fit on the entire dataset (stores training data internally)
    imputer.fit(data_mat)

    # Prepare output array
    filled = np.empty_like(data_mat)

    # Process in chunks so we can tqdm the transform step
    num_chunks = math.ceil(n / chunk_size)
    for chunk_idx in tqdm(
        range(num_chunks),
        desc="KNN Imputing",
        unit="chunk",
    ):
        start = chunk_idx * chunk_size
        end   = min(start + chunk_size, n)
        filled[start:end] = imputer.transform(data_mat[start:end])

    return filled

if __name__ == '__main__':
    folder_to_check = 'male'
    base_dir = 'dane'

    for subfolder, full_data_list, all_attrs in load_folder_data(base_dir, folder_to_check):
        print(f"\n📁 Folder: {subfolder}")
        print(f"📦 Combined dataset: {len(full_data_list)} records, {len(all_attrs)} attrs")

        data_mat, attr_idx = build_dataset_matrix(all_attrs, full_data_list)
        filled_mat = fill_missing_knn_with_progress(data_mat, k=5)

        print("\nFirst 10 filled entries:")
        sorted_attrs = sorted(all_attrs, key=int)
        n_print = min(10, filled_mat.shape[0])
        for idx_row in range(n_print):
            entries = [
                f"'{a}': {filled_mat[idx_row, attr_idx[a]]:.6f}"
                for a in sorted_attrs
            ]
            print(f"Row {idx_row}: " + "{" + ", ".join(entries) + "}")