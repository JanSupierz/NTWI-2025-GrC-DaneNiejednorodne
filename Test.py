import os
import math
import numpy as np
from tqdm import tqdm
from sklearn.impute import KNNImputer

def read_attr_file(attr_path):
    with open(attr_path, 'r') as f:
        return [line.strip() for line in f]

def read_data_file(data_path):
    with open(data_path, 'r') as f:
        return [list(map(float, line.strip().split())) for line in f]

def build_dataset_matrix(all_attrs, full_data_list):
    attrs = sorted(all_attrs, key=int)
    idx = {a: i for i, a in enumerate(attrs)}
    n, m = len(full_data_list), len(attrs)
    data_mat = np.full((n, m), np.nan, dtype=np.float64)
    for i, row in enumerate(full_data_list):
        for a, v in row.items():
            if a in idx:
                try:
                    data_mat[i, idx[a]] = float(v)
                except ValueError:
                    pass
    return data_mat, idx

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
    folder_to_check = 'duze'
    base_dir = f'dane/{folder_to_check}'

    for subfolder in os.listdir(base_dir):
        full_subfolder_path = os.path.join(base_dir, subfolder)
        if not os.path.isdir(full_subfolder_path):
            continue

        full_data_list = []
        all_attrs = set()
        print(f"\n📁 Folder: {subfolder}")

        for i in range(10):
            prefix    = f'{folder_to_check}-{i}'
            attr_path = os.path.join(full_subfolder_path, f'{prefix}.attr')
            data_path = os.path.join(full_subfolder_path, f'{prefix}.data')
            if not os.path.exists(attr_path) or not os.path.exists(data_path):
                continue

            attr_names = read_attr_file(attr_path)
            data_rows  = read_data_file(data_path)
            all_attrs.update(attr_names)
            full_data_list.extend(
                dict(zip(attr_names, row))
                for row in data_rows
            )

        if not all_attrs or not full_data_list:
            continue

        print(f"📦 Combined dataset: "
              f"{len(full_data_list)} records, {len(all_attrs)} attrs")
        data_mat, attr_idx = build_dataset_matrix(all_attrs, full_data_list)

        # This will now print a chunked progress bar to the console:
        filled_mat = fill_missing_knn_with_progress(data_mat, k=5)

        print("\nFirst 20 filled entries:")
        sorted_attrs = sorted(all_attrs, key=int)
        n_print = min(10, filled_mat.shape[0])
        for idx_row in range(n_print):
            entries = [
                f"'{a}': {filled_mat[idx_row, attr_idx[a]]:.6f}"
                for a in sorted_attrs
            ]
            print(f"Row {idx_row}: " + "{" + ", ".join(entries) + "}")