import os
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def read_attr_file(attr_path):
    with open(attr_path, 'r') as f:
        return [line.strip() for line in f]


def read_data_file(data_path):
    with open(data_path, 'r') as f:
        return [list(map(float, line.strip().split())) for line in f]


def build_dataset_matrix(all_attrs, full_data_list):
    """
    Build a numpy matrix of shape (n_records, n_attrs) with np.nan for missing.
    Returns:
      - data_mat: np.ndarray
      - attr_idx: dict mapping attr name to column index
    """
    attrs = sorted(all_attrs)
    idx = {a: i for i, a in enumerate(attrs)}
    n, m = len(full_data_list), len(attrs)
    data_mat = np.full((n, m), np.nan, dtype=float)
    for i, row in enumerate(full_data_list):
        for a, v in row.items():
            if a in idx:
                try:
                    data_mat[i, idx[a]] = float(v)
                except ValueError:
                    pass
    return data_mat, idx


def fill_missing_numpy(data_mat, k):
    """
    Fill missing values (np.nan) by k-NN averaging over available dims.
    """
    n, m = data_mat.shape
    filled = data_mat.copy()

    for i in tqdm(range(n), total=n):
        row = filled[i]
        missing_cols = np.where(np.isnan(row))[0]
        if missing_cols.size == 0:
            continue

        for j in missing_cols:
            # select dims except the target j
            dims = [d for d in range(m) if d != j]
            # mask of valid dims in row
            valid_row = ~np.isnan(row[dims])
            if not valid_row.any():
                continue

            # mask valid rows: must have no nan in those dims
            other = np.isnan(filled[:, dims])
            valid_rows = ~other.any(axis=1)
            valid_rows[i] = False

            if not valid_rows.any():
                continue

            # compute squared distances on dims
            diffs = filled[valid_rows][:, dims] - row[dims]
            sq_dist = np.nansum(diffs**2, axis=1)

            # pick k smallest
            k_eff = min(k, sq_dist.size)
            idx_k = np.argpartition(sq_dist, k_eff - 1)[:k_eff]
            # map back to original row indices
            neighbors = np.where(valid_rows)[0][idx_k]

            # average their j-th values if not nan
            vals = filled[neighbors, j]
            vals = vals[~np.isnan(vals)]
            if vals.size > 0:
                filled[i, j] = vals.mean()

    return filled


def plot_data(before_mat, after_mat, idx, attrs_to_plot):
    """
    Scatter before and after for two given attribute names.
    """
    ax = plt.figure(figsize=(12, 6))

    for pos, (mat, label, color) in enumerate(
        [(before_mat, 'Before Filling', 'red'),
         (after_mat, 'After Filling', 'green')]):
        plt.subplot(1, 2, pos + 1)
        xi, yi = idx[attrs_to_plot[0]], idx[attrs_to_plot[1]]
        x = mat[:, xi]
        y = mat[:, yi]
        mask = ~np.isnan(x) & ~np.isnan(y)
        plt.scatter(x[mask], y[mask], label=label, alpha=0.3)
        plt.title(label)
        plt.xlabel(attrs_to_plot[0])
        plt.ylabel(attrs_to_plot[1])
        plt.legend()

    plt.tight_layout()
    plt.show()


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
            prefix = f'{folder_to_check}-{i}'
            attr_path = os.path.join(full_subfolder_path, f'{prefix}.attr')
            data_path = os.path.join(full_subfolder_path, f'{prefix}.data')

            if not os.path.exists(attr_path) or not os.path.exists(data_path):
                continue

            attr_names = read_attr_file(attr_path)
            data_rows = read_data_file(data_path)

            all_attrs.update(attr_names)
            full_data_list.extend(dict(zip(attr_names, row)) for row in data_rows)

        if not all_attrs or not full_data_list:
            continue

        print(f"📦 Combined dataset: {len(full_data_list)} records, {len(all_attrs)} attrs")

        # Build numpy matrix
        data_mat, attr_idx = build_dataset_matrix(all_attrs, full_data_list)

        # Fill missing
        filled_mat = fill_missing_numpy(data_mat, k=5)

        # Show before/after on first two attrs
        sample_attrs = list(sorted(all_attrs))[:2]
        plot_data(data_mat, filled_mat, attr_idx, sample_attrs)

        # Optionally, convert back to dicts:
        # filled_list = []
        # for i, row in enumerate(filled_mat):
        #     filled_list.append({a: row[attr_idx[a]] for a in all_attrs if not np.isnan(row[attr_idx[a]])})
