def fill_missing_fcm_knn(data, k=5, n_clusters=3):
    import numpy as np
    from sklearn.impute import SimpleImputer, KNNImputer
    import skfuzzy as fuzz
    from visualization import visualize_clusters

    # Cheap mean imputation for clustering only
    simple_imputer = SimpleImputer(strategy='mean')
    data_for_clustering = simple_imputer.fit_transform(data)
    valid_data = data_for_clustering.T

    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        valid_data, c=n_clusters, m=2, error=0.005, maxiter=1000
    )

    u_all, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        valid_data, cntr, m=2, error=0.005, maxiter=1000
    )

    visualize_clusters(data_for_clustering, u_all)

    hard_clusters = np.argmax(u_all, axis=0)

    filled = np.empty_like(data)
    for cluster_id in range(n_clusters):
        cluster_mask = hard_clusters == cluster_id
        cluster_data = data[cluster_mask, :]  # Make sure to keep all features!

        if cluster_data.shape[0] == 0:
            continue

        imputer = KNNImputer(n_neighbors=k, weights="uniform", metric="nan_euclidean")
        filled_cluster = imputer.fit_transform(cluster_data)

        # Check shapes:
        if filled_cluster.shape != cluster_data.shape:
            raise ValueError(f"Shape mismatch in cluster {cluster_id}: filled {filled_cluster.shape} vs original {cluster_data.shape}")

        filled[cluster_mask, :] = filled_cluster  # Assign imputed cluster data back with full features

    return filled
