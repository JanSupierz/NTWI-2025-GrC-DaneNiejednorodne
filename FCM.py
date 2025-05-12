import numpy as np
import skfuzzy as fuzz

from sklearn.impute import KNNImputer

from Loading import (
    load_folder_data,
    build_dataset_matrix,
)

def fill_missing_with_fcm(data_mat, k=5, g=10, fuzziness=2.0, max_iter=1000, error=1e-5):
    """
    1. Temporarily replace NaNs with column means for clustering.
    2. Run Fuzzy C-Means to get a membership matrix (u).
    3. For each sample, find its *highest-membership* cluster.
    4. One-hot encode those hard assignments (or weight by membership if you like).
    5. Augment the data with that cluster info and run KNNImputer.
    """
    # 1) fill NaNs by col means for clustering
    col_means = np.nanmean(data_mat, axis=0)
    temp = np.where(np.isnan(data_mat), col_means, data_mat).T  # skfuzzy expects shape (features, samples)

    # 2) Fuzzy C-Means: returns cluster centers + membership matrix u (shape (g, n_samples))
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        temp,                  # input data
        c=g,                   # number of clusters
        m=fuzziness,           # fuzziness exponent
        error=error,           # stopping criterion
        maxiter=max_iter,
        init=None,
        seed=0
    )
    # 3) Hard assign each sample to the cluster with highest membership
    hard_labels = np.argmax(u, axis=0)

    # 4) One-hot encode hard labels and weight heavily
    n = data_mat.shape[0]
    clusters_ohe = np.zeros((n, g), dtype=float)
    clusters_ohe[np.arange(n), hard_labels] = 1.0e6  # large weight to keep neighbors within same cluster

    # 5) Augment and impute
    augmented = np.hstack([temp.T, clusters_ohe])  # back to shape (n, m+g)
    imputer = KNNImputer(n_neighbors=k, metric="nan_euclidean", weights="uniform")
    filled_aug = imputer.fit_transform(augmented)

    # return only the original m columns
    return filled_aug[:, :data_mat.shape[1]]

if __name__ == "__main__":
    folder_to_check = 'male'
    base_dir = 'dane'

    for subfolder, full_data_list, all_attrs in load_folder_data(base_dir, folder_to_check):
        print(f"\n📁 Folder: {subfolder}")
        print(f"📦 Combined dataset: {len(full_data_list)} records, {len(all_attrs)} attrs")

        data_mat, attr_idx = build_dataset_matrix(all_attrs, full_data_list)
        filled_mat = fill_missing_with_fcm(data_mat, k=5, g=10)

        print("\nFirst 10 filled entries:")
        sorted_attrs = sorted(all_attrs, key=int)
        for row_i in range(min(10, filled_mat.shape[0])):
            entries = [f"'{a}': {filled_mat[row_i, attr_idx[a]]:.6f}" for a in sorted_attrs]
            print(f"Row {row_i}: " + "{" + ", ".join(entries) + "}")
