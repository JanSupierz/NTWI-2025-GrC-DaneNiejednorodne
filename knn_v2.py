import os
import data_loading as dl
import numpy as np
import time
import knn_core

# ---------------- Configuration ----------------
folder_name = 'male'
base_path = f'dane/{folder_name}/1'

# ---------------- Load Data ----------------
print(f"Loading data from: {base_path}/{folder_name}")
file_data_list, all_attrs = dl.load_files(os.path.join(base_path), folder_name)
all_attrs = sorted([str(attr) for attr in all_attrs])  # Ensure string consistency

print(f"Loaded {len(file_data_list)} files")
for i, file_rows in enumerate(file_data_list):
    print(f"  File {i}: {len(file_rows)} rows")
print(f"Total attributes: {len(all_attrs)}")

# ---------------- Prepare Data for C++ ----------------

# Map: attribute name → index
attr_to_index = {attr: idx for idx, attr in enumerate(all_attrs)}

# Convert file rows (list of dicts) → numpy arrays with shared attr ordering
np_file_data = []
column_masks = []

for file_rows in file_data_list:
    # Find present attributes in this file
    present_attrs = [attr for attr in all_attrs if attr in file_rows[0]]
    column_masks.append([attr_to_index[attr] for attr in present_attrs])

    # Create numpy array for this file
    rows = []
    for row in file_rows:
        row_values = [row[attr] for attr in present_attrs]
        rows.append(row_values)

    np_array = np.array(rows, dtype=np.float64)
    np_file_data.append(np_array)

# ---------------- Run C++-Accelerated k-NN ----------------

k = 5
all_neighbors_per_file = []

start_time = time.time()

for i in range(len(np_file_data)):
    print(f"Processing file {i} using C++ module...")
    neighbors = knn_core.knn_for_file(
        np_file_data[i],  # current file
        i,                # index of current file
        np_file_data,     # all files
        column_masks,     # mask: which global attributes are present in each file
        k,
        len(all_attrs)    # total global attributes
    )
    all_neighbors_per_file.append(neighbors)

end_time = time.time()

print(f"✅ C++ k-NN completed in {end_time - start_time:.2f} seconds")

imputed_file_data_list = []

for file_idx, file_rows in enumerate(file_data_list):
    imputed_rows = []
    neighbors_per_row = all_neighbors_per_file[file_idx]

    for row_idx, row in enumerate(file_rows):
        row_dict = row.copy()
        neighbors = neighbors_per_row[row_idx]

        missing_attrs = [attr for attr in all_attrs if attr not in row_dict]

        for missing_attr in missing_attrs:
            attr_idx = attr_to_index[missing_attr]
            values = []

            for dist, n_file_idx, n_row_idx in neighbors:
                neighbor_row = file_data_list[n_file_idx][n_row_idx]
                if missing_attr in neighbor_row:
                    values.append(neighbor_row[missing_attr])

            if values:
                imputed_value = np.mean(values)
                row_dict[missing_attr] = imputed_value
            else:
                # Leave missing if no neighbor has the value
                row_dict[missing_attr] = np.nan

        imputed_rows.append(row_dict)

    imputed_file_data_list.append(imputed_rows)

print("\n✅ Imputed data preview for the first file:")

preview_rows = imputed_file_data_list[3][:10]

header = ["Row"] + all_attrs

# Calculate max width per column (including header)
col_widths = []

# For "Row" column, width based on max row index or header length
max_row_idx_len = max(len(str(len(preview_rows)-1)), len("Row"))
col_widths.append(max_row_idx_len)

# For other columns, max length among header and data
for attr in all_attrs:
    max_len = len(attr)
    for row_dict in preview_rows:
        val = row_dict.get(attr, np.nan)
        if isinstance(val, float) and np.isnan(val):
            val_str = "MISSING"
        else:
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        if len(val_str) > max_len:
            max_len = len(val_str)
    col_widths.append(max_len)

# Create a format string with fixed width for each column
fmt_str = ""
for w in col_widths:
    fmt_str += f"{{:<{w + 2}}}"  # +2 for padding space

# Print header
print(fmt_str.format(*header))

# Print rows
for i, row_dict in enumerate(preview_rows):
    row_values = [str(i)]
    for attr in all_attrs:
        val = row_dict.get(attr, np.nan)
        if isinstance(val, float) and np.isnan(val):
            row_values.append("MISSING")
        else:
            if isinstance(val, float):
                row_values.append(f"{val:.4f}")
            else:
                row_values.append(str(val))
    print(fmt_str.format(*row_values))

print(f"\nTotal rows imputed in first file: {len(imputed_file_data_list[0])}")


