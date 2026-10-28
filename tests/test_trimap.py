import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trimap import TriMap, global_score, tempered_log, trimap_loss_grad  # noqa: E402
from trimap.core import pca  # noqa: E402


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((30, 2))
    T = np.stack([rng.choice(30, 3, replace=False) for _ in range(200)]).astype(np.int64)
    w = rng.random(200)
    _, g, _ = trimap_loss_grad(Y, T, w)
    eps = 1e-6
    num = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        for c in range(2):
            Yp, Ym = Y.copy(), Y.copy()
            Yp[i, c] += eps
            Ym[i, c] -= eps
            num[i, c] = (trimap_loss_grad(Yp, T, w)[0] - trimap_loss_grad(Ym, T, w)[0]) / (2 * eps)
    assert np.allclose(g, num, atol=1e-6)


def test_tempered_log():
    u = np.array([1.0, 2.0, 10.0])
    assert np.allclose(tempered_log(u, 1.0), np.log(u))
    assert np.allclose(tempered_log(u, 0.5), 2 * (np.sqrt(u) - 1))


def test_global_score_pca_is_one_and_invariant():
    X = np.random.default_rng(1).standard_normal((500, 10)) * np.arange(1, 11)
    Y = pca(X, 2)
    assert abs(global_score(X, Y) - 1.0) < 1e-8
    R = np.array([[0.0, -3.0], [2.0, 0.5]])  # any invertible linear map
    assert abs(global_score(X, Y @ R + 7) - 1.0) < 1e-8
    assert global_score(X, np.random.default_rng(2).standard_normal((500, 2))) < 0.5


def test_triplets_and_embedding():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(1000, n_features=20, centers=5, random_state=0)
    tm = TriMap(n_iters=200)
    Y = tm.fit_transform(X)
    n = len(X)
    assert tm.triplets_.shape == (n * (12 * 4 + 3), 3)
    assert np.all(tm.weights_ >= 0)
    i, j, k = tm.triplets_.T
    assert np.all((i != j) & (i != k) & (j != k))
    # Triplets are satisfied in the input space.
    d = lambda a, b: np.sum((X[a] - X[b]) ** 2, 1)  # noqa: E731
    assert np.mean(d(i, j) <= d(i, k)) > 0.9
    # Loss decreases and blobs stay separated.
    assert tm.loss_history_[-1] < tm.loss_history_[0]
    from trimap import nn_accuracy
    assert nn_accuracy(Y, y) > 0.99


def test_parallel_gradient_matches_serial():
    from trimap.core import trimap_loss_grad_parallel
    rng = np.random.default_rng(3)
    Y = rng.standard_normal((500, 2))
    T = rng.integers(0, 500, size=(10007, 3)).astype(np.int64)
    w = rng.random(10007)
    l1, g1, v1 = trimap_loss_grad(Y, T, w)
    l2, g2, v2 = trimap_loss_grad_parallel(Y, T, w)
    assert np.isclose(l1, l2) and np.allclose(g1, g2) and v1 == v2
