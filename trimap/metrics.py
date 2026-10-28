"""Evaluation measures used in the TriMap paper."""

import numpy as np


def _center(M):
    return M - M.mean(axis=0)


def min_reconstruction_error(X, Y):
    """E(Y | X) = min_A ||X - A Y||_F^2 (paper Sec. 2), rows are samples here.

    In row convention this is min_A ||X - Y A||_F^2, a least-squares problem;
    the optimum A* = (Y^T Y)^-1 Y^T X also absorbs rotation and scaling of Y.
    """
    X, Y = _center(np.asarray(X, float)), _center(np.asarray(Y, float))
    A, *_ = np.linalg.lstsq(Y, X, rcond=None)
    return float(np.sum((X - Y @ A) ** 2))


def pca_reconstruction_error(X, n_dims):
    """E_PCA: residual variance after keeping the top-n_dims components."""
    Xc = _center(np.asarray(X, float))
    s = np.linalg.svd(Xc, compute_uv=False)
    return float(np.sum(s[n_dims:] ** 2))


def global_score(X, Y, e_pca=None):
    """GS(Y | X) = exp(-(E(Y|X) - E_PCA) / E_PCA) in [0, 1]; equals 1 for PCA."""
    if e_pca is None:
        e_pca = pca_reconstruction_error(X, Y.shape[1])
    e = min_reconstruction_error(X, Y)
    return float(np.exp(-(e - e_pca) / e_pca))


def nn_accuracy(Y, labels, n_samples=None, seed=0):
    """Leave-one-out 1-nearest-neighbour classification accuracy in the embedding.

    For large n, accuracy is estimated on `n_samples` random query points
    (neighbours are still searched among all points).
    """
    from sklearn.neighbors import NearestNeighbors
    labels = np.asarray(labels)
    n = len(Y)
    query = np.arange(n)
    if n_samples is not None and n_samples < n:
        query = np.random.default_rng(seed).choice(n, n_samples, replace=False)
    nn = NearestNeighbors(n_neighbors=2).fit(Y)
    _, idx = nn.kneighbors(Y[query])
    # Column 0 is normally the point itself; fall back if a duplicate came first.
    nbr = np.where(idx[:, 0] == query, idx[:, 1], idx[:, 0])
    return float(np.mean(labels[nbr] == labels[query]))


def pearson_cluster_correlation(X, Y, n_clusters=100, subset=10000,
                                n_repeats=100, seed=0):
    """PCC of Sec. 4.1: correlation between inter-cluster-centre distances.

    For each repeat: take a random subset, run k-means (k=100) in the
    high-dimensional space, compute the cluster centres in both spaces for
    the *same* cluster assignment, and correlate the two condensed distance
    matrices (Mantel statistic). Returns the mean over repeats.
    """
    from scipy.spatial.distance import pdist
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    scores = []
    for r in range(n_repeats):
        sub = rng.choice(len(X), min(subset, len(X)), replace=False)
        km = KMeans(n_clusters=n_clusters, n_init=1, random_state=seed + r)
        lab = km.fit_predict(X[sub])
        present = np.unique(lab)
        cx = np.stack([X[sub][lab == c].mean(0) for c in present])
        cy = np.stack([Y[sub][lab == c].mean(0) for c in present])
        scores.append(np.corrcoef(pdist(cx), pdist(cy))[0, 1])
    return float(np.mean(scores))
