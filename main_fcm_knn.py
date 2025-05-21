from fcm_knn_imputer import fill_missing_fcm_knn
from data_loading import load_folder_data, build_dataset_matrix
from visualization import plot_imputation_comparison

if __name__ == '__main__':
    folder_to_check = 'duze'
    base_dir = 'dane'
    nr_rows_to_show = 30

    for subfolder, full_data_list, all_attrs in load_folder_data(base_dir, folder_to_check):
        print(f"Folder: {subfolder}")
        print(f"Combined dataset: {len(full_data_list)} records, {len(all_attrs)} attrs")

        data_mat, attr_idx = build_dataset_matrix(all_attrs, full_data_list, nr_rows_to_show)
        filled_mat = fill_missing_fcm_knn(data_mat, k=5, n_clusters=8)

        plot_imputation_comparison(data_mat, filled_mat, nr_rows_to_show)

        print("First 10 filled entries:")
        sorted_attrs = sorted(all_attrs, key=int)
        n_print = min(10, filled_mat.shape[0])
        for idx_row in range(n_print):
            entries = [f"'{a}': {filled_mat[idx_row, attr_idx[a]]:.6f}" for a in sorted_attrs]
            print(f"Row {idx_row}: " + "{" + ", ".join(entries) + "}")
        print('\n')
