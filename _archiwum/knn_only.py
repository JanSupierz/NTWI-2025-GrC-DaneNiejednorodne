import os
import math
import pandas as pd
import data_loading as dl
import numpy as np
import time
import numba
import matplotlib.pyplot as plt
from numba import njit, prange

# ---------------- Configuration ----------------
folder_name = 'duze'
base_path = f'dane/{folder_name}/1'

# ---------------- Load Data ----------------
print(f"Loading data from: {base_path}/{folder_name}")
file_data, all_attrs = dl.load_files(os.path.join(base_path), folder_name)
all_attrs = [str(attr) for attr in all_attrs]
D = len(all_attrs)

print(f"Loaded {len(file_data)} files")
for i, rows in enumerate(file_data):
    print(f"  File {i}: {len(rows)} rows")
print(f"Total attributes: {D}")

# ---------------- Preprocessing ----------------
preprocessed_files = []
for rows in file_data:
    file_rows = [{str(k): v for k, v in row.items()} for row in rows]
    preprocessed_files.append(file_rows)


# ---------------- Prepare X and Mask for Each File ----------------
def prepare_data(rows, all_attrs):
    n = len(rows)
    D = len(all_attrs)
    attr_to_idx = {attr: idx for idx, attr in enumerate(all_attrs)}

    X = np.full((n, D), np.nan)
    present_mask = np.zeros((n, D), dtype=np.bool_)

    for i, row_dict in enumerate(rows):
        for col, val in row_dict.items():
            if col in attr_to_idx:
                try:
                    col_idx = attr_to_idx[col]
                    X[i, col_idx] = float(val)
                    present_mask[i, col_idx] = True
                except (ValueError, TypeError):
                    continue
    return X, present_mask


Xs, masks = [], []
for file_rows in preprocessed_files:
    X, mask = prepare_data(file_rows, all_attrs)
    Xs.append(X)
    masks.append(mask)

# ---------------- Build Row Lookup ----------------
row_lookup = {}
for file_idx, rows in enumerate(preprocessed_files):
    for row_idx, row_dict in enumerate(rows):
        row_lookup[(file_idx, row_idx)] = row_dict


# ---------------- Numba-optimized neighbor computation ----------------
@njit(parallel=True, fastmath=True)
def compute_neighbors(Xs, masks, D, k=5):
    n_files = len(Xs)
    all_top_dists = []
    all_top_indices = []

    for file_i in prange(n_files):
        X_i = Xs[file_i]
        mask_i = masks[file_i]
        n_i = X_i.shape[0]

        file_top_dists = np.full((n_i, k), np.inf)
        file_top_indices = np.full((n_i, k, 2), -1)  # (file_j, row_j)

        for row_i in prange(n_i):
            # Initialize min-heap for top neighbors
            top_dists = np.full(k, np.inf)
            top_indices = np.full((k, 2), -1, dtype=np.int32)

            # Check all files
            for file_j in range(n_files):
                X_j = Xs[file_j]
                mask_j = masks[file_j]
                n_j = X_j.shape[0]

                # Check all rows in this file
                for row_j in range(n_j):
                    if file_i == file_j and row_i == row_j:
                        continue

                    # Find common attributes
                    common_mask = mask_i[row_i] & mask_j[row_j]
                    valid_count = np.sum(common_mask)

                    if valid_count == 0:
                        continue

                    # Compute partial Euclidean distance
                    sq_sum = 0.0
                    for col in range(D):
                        if common_mask[col]:
                            diff = X_i[row_i, col] - X_j[row_j, col]
                            sq_sum += diff * diff

                    partial_dist = math.sqrt(sq_sum)
                    dist = partial_dist * math.sqrt(D / valid_count)

                    # Maintain top k smallest distances
                    if dist < top_dists[k - 1]:
                        # Replace the largest distance
                        top_dists[k - 1] = dist
                        top_indices[k - 1] = [file_j, row_j]

                        # Bubble down to maintain heap order
                        idx = k - 1
                        while idx > 0 and top_dists[idx] < top_dists[idx - 1]:
                            # Swap with previous element
                            top_dists[idx], top_dists[idx - 1] = top_dists[idx - 1], top_dists[idx]
                            top_indices[idx], top_indices[idx - 1] = top_indices[idx - 1], top_indices[idx]
                            idx -= 1

            # Store sorted results for this row
            file_top_dists[row_i] = top_dists
            file_top_indices[row_i] = top_indices

        all_top_dists.append(file_top_dists)
        all_top_indices.append(file_top_indices)

    return all_top_dists, all_top_indices


# ---------------- Compute Nearest Neighbors ----------------
k = 5
print("Computing neighbors with Numba acceleration...")
start = time.time()
all_top_dists, all_top_indices = compute_neighbors(Xs, masks, D, k)
print(f"Neighbor computation took {time.time() - start:.2f} seconds")

# Convert to original format
nearestNeighbors = []
for file_i in range(len(Xs)):
    for row_i in range(len(all_top_dists[file_i])):
        neighbors = []
        for i in range(k):
            dist = all_top_dists[file_i][row_i, i]
            file_j, row_j = all_top_indices[file_i][row_i, i]
            if file_j != -1:  # Valid neighbor
                neighbors.append((dist, file_j, row_j))

        nearestNeighbors.append({
            'source': (file_i, row_i),
            'neighbors': neighbors
        })

# ---------------- Imputation ----------------
start = time.time()
imputed_files = []

for file_idx, rows in enumerate(preprocessed_files):
    print(f"Imputing file {file_idx}...")
    imputed_file_rows = []

    for row_idx, row_dict in enumerate(rows):
        # Find this row in nearestNeighbors
        row_index = next(i for i, n in enumerate(nearestNeighbors)
                         if n['source'] == (file_idx, row_idx))
        neighbor_info = nearestNeighbors[row_index]
        source_row = row_dict.copy()

        missing_keys = set(all_attrs) - set(source_row.keys())

        for key in missing_keys:
            values = []
            for dist, file_j, row_j in neighbor_info['neighbors']:
                neighbor_row = row_lookup.get((file_j, row_j), {})
                if key in neighbor_row:
                    try:
                        val = float(neighbor_row[key])
                        values.append(val)
                    except (ValueError, TypeError):
                        continue

            if values:
                source_row[key] = sum(values) / len(values)

        imputed_file_rows.append(source_row)

    imputed_files.append(imputed_file_rows)
    print(f"  File {file_idx} imputed: {len(imputed_file_rows)} rows")

print(f"Imputation took {time.time() - start:.2f} seconds")

# ---------------- Convert to DataFrames and Display ----------------
file_dfs = []
for file_idx, rows in enumerate(imputed_files):
    df_file = pd.DataFrame(rows)
    for attr in all_attrs:
        if attr not in df_file.columns:
            df_file[attr] = np.nan
    df_file = df_file[all_attrs]
    file_dfs.append(df_file)
    print(f"File {file_idx} DataFrame created: {df_file.shape}")

    # Display first 10 rows
    df_display = df_file.head(10)
    fig, ax = plt.subplots(figsize=(min(15, len(df_display.columns) * 1.2), 4))
    ax.axis('off')

    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        cellLoc='left',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    plt.title(f'Imputed Data - File {file_idx} - Shape: {df_file.shape}', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()