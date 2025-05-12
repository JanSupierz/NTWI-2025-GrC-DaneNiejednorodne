import os
import numpy as np


def read_attr_file(attr_path):
    """
    Read an attribute file: one attribute name per line.
    Returns a list of strings.
    """
    with open(attr_path, 'r') as f:
        return [line.strip() for line in f]


def read_data_file(data_path):
    """
    Read a data file: one row of whitespace-separated floats per line.
    Returns a list of lists of floats.
    """
    with open(data_path, 'r') as f:
        return [list(map(float, line.strip().split())) for line in f]


def build_dataset_matrix(all_attrs, full_data_list):
    """
    Given a set of attribute names and a list of row-dicts,
    produce a (n_samples x n_attrs) numpy matrix with NaNs for missing.

    all_attrs: iterable of attribute names (strings, numeric keys)
    full_data_list: list of dict(attr_name->value)

    Returns:
      - data_mat: np.ndarray shape (n_samples, n_attrs)
      - idx: dict mapping attr_name->column index
    """
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
                    # skip non-numeric entries
                    pass

    return data_mat, idx


def load_folder_data(base_dir, folder_to_check, max_parts=10):
    """
    Walk through subfolders of base_dir/folder_to_check,
    read up to max_parts .attr/.data pairs per subfolder,
    and return (full_data_list, all_attrs) for each.

    Yields tuples: (subfolder_name, full_data_list, all_attrs)
    """
    root = os.path.join(base_dir, folder_to_check)
    for subfolder in os.listdir(root):
        full_path = os.path.join(root, subfolder)
        if not os.path.isdir(full_path):
            continue

        full_data_list = []
        all_attrs = set()
        for i in range(max_parts):
            prefix = f"{folder_to_check}-{i}"
            attr_path = os.path.join(full_path, f'{prefix}.attr')
            data_path = os.path.join(full_path, f'{prefix}.data')
            if not os.path.exists(attr_path) or not os.path.exists(data_path):
                continue

            names = read_attr_file(attr_path)
            rows = read_data_file(data_path)
            all_attrs.update(names)
            full_data_list.extend(dict(zip(names, row)) for row in rows)

        if all_attrs and full_data_list:
            yield subfolder, full_data_list, all_attrs
