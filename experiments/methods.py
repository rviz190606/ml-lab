"""Wrappers so every DR method is called the same way: f(X, seed) -> Y."""

import numpy as np

from trimap import TriMap
from trimap.core import pca


def run_trimap(X, seed=0, **kw):
    return TriMap(seed=seed, **kw).fit_transform(X)


def run_pca(X, seed=0):
    return pca(np.asarray(X, float), 2)


def _reduce(X, dims=100):
    # Paper: "we reduce the number of dimensions of the dataset to 100 if necessary".
    return pca(np.asarray(X, float), dims) if X.shape[1] > dims else X


def run_tsne(X, seed=0):
    from sklearn.manifold import TSNE
    return TSNE(2, random_state=seed, n_jobs=-1).fit_transform(_reduce(X))


def run_umap(X, seed=0):
    import umap
    # No random_state: fixing it forces UMAP to run single-threaded.
    return umap.UMAP(n_components=2).fit_transform(_reduce(X))


METHODS = {"t-SNE": run_tsne, "UMAP": run_umap, "TriMap": run_trimap, "PCA": run_pca}
