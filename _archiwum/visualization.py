import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

def mask_random_values(data, mask_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    data_copy = data.copy()
    mask = ~np.isnan(data_copy)
    indices = np.argwhere(mask)
    rng.shuffle(indices)
    n_mask = int(mask_ratio * indices.shape[0])

    selected = indices[:n_mask]
    for i, j in selected:
        data_copy[i, j] = np.nan

    return data_copy

def plot_imputation_comparison(original, filled, max_rows=20):
    original = original[:max_rows, :]
    filled = filled[:max_rows, :]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, matrix, title in zip(axes, [original, filled], ["Before Imputation", "After Imputation"]):
        sns.heatmap(
            matrix,
            ax=ax,
            annot=True,
            fmt=".1f",
            cmap="viridis",
            cbar=False,
            mask=np.isnan(matrix) if title == "Before Imputation" else None
        )
        ax.set_title(title)
        ax.set_xlabel("Features")
        ax.set_ylabel("Samples")

    plt.tight_layout()
    plt.show()

def visualize_clusters(data, membership_probs):
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data)  # only for PCA

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data_imputed)
    cluster_labels = np.argmax(membership_probs, axis=0)

    plt.figure(figsize=(10, 6))
    for cluster_id in np.unique(cluster_labels):
        points = reduced[cluster_labels == cluster_id]
        plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {cluster_id}", alpha=0.7)

    plt.title("Fuzzy C-Means Clustering (PCA Projection)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
