from sklearn.impute import KNNImputer

def fill_missing_knn(data, k=5):
    imputer = KNNImputer(n_neighbors=k, weights="uniform", metric="nan_euclidean")
    return imputer.fit_transform(data)