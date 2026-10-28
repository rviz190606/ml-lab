"""

A brief overview of the steps: 

1. Preprocess: optionally reduce to 100 dims with PCA, center and rescale.

2. NN search. Here we use PyNNDescent (approximate) for large datasets and sklearn (exact) for small ones.

3. Sample triplets (i, j, k):
     
4. Compute the Weights

5. Initialise with (scaled) PCA and minimise the loss with gradient descent
       
"""

import time

import numba
import numpy as np


# Some helper functions we will need: 

def tempered_log(u, t):
    """log_t(u) = (u^(1-t) - 1) / (1 - t); recovers log(u) as t -> 1."""
    if abs(t - 1.0) < 1e-5:
        return np.log(u)
    return (np.power(u, 1.0 - t) - 1.0) / (1.0 - t)


def pca(X, n_components):
    """Return the top principal-component scores of X (rows are samples)."""
    Xc = X - X.mean(axis=0)
    if Xc.shape[1] > 3 * n_components and Xc.shape[0] > 3 * n_components:
        from sklearn.utils.extmath import randomized_svd
        U, S, _ = randomized_svd(Xc, n_components, random_state=0)
    else:
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        U, S = U[:, :n_components], S[:n_components]
    return U * S


def knn_search(X, k, seed, verbose=False):
    """k nearest neighbours of every point, excluding the point itself.
    Exact brute-force search for small datasets, PyNNDescent for bigger ones (approximate)
    """
    n = X.shape[0]
    k = min(k, n - 1)
    if n <= 10000:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        dist, idx = nn.kneighbors(X)
    else:
        from pynndescent import NNDescent
        index = NNDescent(X, n_neighbors=k + 1, metric="euclidean",
                          random_state=seed, low_memory=True, verbose=verbose)
        idx, dist = index.neighbor_graph
    # Drop self (normally column 0; robust to duplicates / approximate search).
    out_idx = np.empty((n, k), dtype=np.int64)
    out_dist = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        keep = idx[i] != i
        row_i, row_d = idx[i][keep][:k], dist[i][keep][:k]
        out_idx[i], out_dist[i] = row_i, row_d
    return out_idx, out_dist



@numba.njit(cache=True)
def _sq_dist(X, i, j):
    d = 0.0
    for t in range(X.shape[1]):
        diff = X[i, t] - X[j, t]
        d += diff * diff
    return d


@numba.njit(parallel=True, cache=True)
def _sample_knn_triplets(X, nbrs, sig, n_inliers, n_outliers, seed, triplets, weights):
    n = X.shape[0]
    n_cand = nbrs.shape[1]
    per_point = n_inliers * n_outliers
    for i in numba.prange(n):
        np.random.seed(seed + i)
        base = i * per_point
        for a in range(n_inliers):
            j = nbrs[i, a]
            d_ij = _sq_dist(X, i, j) / (sig[i] * sig[j])
            for b in range(n_outliers):
                while True:
                    k = np.random.randint(n)
                    if k == i:
                        continue
                    bad = False
                    for c in range(n_cand):
                        if nbrs[i, c] == k:
                            bad = True
                            break
                    if not bad:
                        break
                d_ik = _sq_dist(X, i, k) / (sig[i] * sig[k])
                p = base + a * n_outliers + b
                triplets[p, 0] = i
                triplets[p, 1] = j
                triplets[p, 2] = k
                weights[p] = d_ik - d_ij


@numba.njit(parallel=True, cache=True)
def _sample_random_triplets(X, sig, n_random, seed, triplets, weights):
    """r random triplets per point; j, k swapped so that j is the nearer one.
    Fills triplets (n*r, 3) and weights in place."""
    n = X.shape[0]
    for i in numba.prange(n):
        np.random.seed(seed + n + i)
        for a in range(n_random):
            j = np.random.randint(n)
            while j == i:
                j = np.random.randint(n)
            k = np.random.randint(n)
            while k == i or k == j:
                k = np.random.randint(n)
            d_ij = _sq_dist(X, i, j) / (sig[i] * sig[j])
            d_ik = _sq_dist(X, i, k) / (sig[i] * sig[k])
            if d_ij > d_ik:
                j, k = k, j
                d_ij, d_ik = d_ik, d_ij
            p = i * n_random + a
            triplets[p, 0] = i
            triplets[p, 1] = j
            triplets[p, 2] = k
            weights[p] = d_ik - d_ij


# The loss and gradients:

@numba.njit(parallel=True, cache=True)
def trimap_loss_grad_parallel(Y, triplets, weights):
    """Multi-threaded trimap_loss_grad: triplets are split into chunks, each
    accumulating into its own gradient buffer (no write races), then summed."""
    n_chunks = numba.get_num_threads()
    n_trip = triplets.shape[0]
    size = (n_trip + n_chunks - 1) // n_chunks
    grads = np.zeros((n_chunks, Y.shape[0], Y.shape[1]))
    losses = np.zeros(n_chunks)
    viols = np.zeros(n_chunks, dtype=np.int64)
    for c in numba.prange(n_chunks):
        lo, hi = c * size, min((c + 1) * size, n_trip)
        if lo < hi:
            losses[c], viols[c] = _accumulate(Y, triplets[lo:hi], weights[lo:hi], grads[c])
    return losses.sum(), grads.sum(axis=0), viols.sum()


@numba.njit(cache=True)
def trimap_loss_grad(Y, triplets, weights):
    grad = np.zeros_like(Y)
    loss, n_viol = _accumulate(Y, triplets, weights, grad)
    return loss, grad, n_viol


@numba.njit(cache=True)
def _accumulate(Y, triplets, weights, grad):
    """
        Computes the loss function and its gradients
    """
    dim = Y.shape[1]
    loss = 0.0
    n_viol = 0
    y_ij = np.empty(dim)
    y_ik = np.empty(dim)
    for t in range(triplets.shape[0]):
        i, j, k = triplets[t, 0], triplets[t, 1], triplets[t, 2]
        d_ij = 1.0
        d_ik = 1.0
        for c in range(dim):
            y_ij[c] = Y[i, c] - Y[j, c]
            y_ik[c] = Y[i, c] - Y[k, c]
            d_ij += y_ij[c] * y_ij[c]
            d_ik += y_ik[c] * y_ik[c]
        a = 1.0 / d_ij
        b = 1.0 / d_ik
        if d_ij > d_ik:
            n_viol += 1
        w = weights[t]
        loss += w * b / (a + b)
        g = 2.0 * w * a * b / ((a + b) * (a + b))
        for c in range(dim):
            gj = g * a * y_ij[c]
            gk = g * b * y_ik[c]
            grad[i, c] += gj - gk
            grad[j, c] -= gj
            grad[k, c] += gk
    return loss, n_viol


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class TriMap:
    """TriMap dimensionality reduction.

    Parameters
    ----------
    n_dims : int            Embedding dimension.
    n_inliers : int         m  - nearest neighbours per point used as j.
    n_outliers : int        m' - outliers k sampled per (i, j) pair.
    n_random : int          r  - random triplets per point.
    weight_temp : float     t  - temperature of the tempered log on weights.
    lr : float              Base learning rate (rescaled by n / |T|).
    n_iters : int           Number of full-batch gradient-descent iterations.
    switch_iter : int       Iteration at which momentum goes 0.5 -> 0.8.
    apply_pca : bool        Reduce inputs to `pca_dims` dims before k-NN.
    init : {"pca","random"} or ndarray  Initial embedding.
    seed : int              Random seed.
    """

    def __init__(self, n_dims=2, n_inliers=12, n_outliers=4, n_random=3,
                 weight_temp=0.5, lr=0.1, n_iters=400, switch_iter=250,
                 apply_pca=True, pca_dims=100, init="pca", seed=0,
                 verbose=False):
        self.n_dims = n_dims
        self.n_inliers = n_inliers
        self.n_outliers = n_outliers
        self.n_random = n_random
        self.weight_temp = weight_temp
        self.lr = lr
        self.n_iters = n_iters
        self.switch_iter = switch_iter
        self.apply_pca = apply_pca
        self.pca_dims = pca_dims
        self.init = init
        self.seed = seed
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"[TriMap] {msg}", flush=True)

    # -- Steps 1-4 -----------------------------------------------------------
    def _preprocess(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.apply_pca and X.shape[1] > self.pca_dims:
            X = pca(X, self.pca_dims)
            self._log(f"PCA -> {self.pca_dims} dims")
        # Same normalisation as the reference implementation.
        X = X - X.min()
        X = X / max(X.max(), 1e-12)
        X = X - X.mean(axis=0)
        return np.ascontiguousarray(X)

    def generate_triplets(self, X):
        n = X.shape[0]
        n_extra = min(self.n_inliers + 50, n - 1)
        t0 = time.time()
        nbrs, dist = knn_search(X, n_extra, self.seed, verbose=self.verbose)
        self._log(f"k-NN search ({n_extra} nbrs): {time.time() - t0:.1f}s")

        # sigma_i = mean distance to the 4th-6th nearest neighbours.
        sig = np.maximum(dist[:, 3:6].mean(axis=1), 1e-10)

        # Re-rank candidates by the density-scaled distance and keep top m.
        scaled = dist ** 2 / (sig[:, None] * sig[nbrs])
        order = np.argsort(scaled, axis=1)
        nbrs = np.take_along_axis(nbrs, order, axis=1)
        nbrs = np.ascontiguousarray(nbrs)

        t0 = time.time()
        # Preallocate one int32 buffer for both triplet kinds: at large m, m'
        # (Fig. 3, |T| = 820n on MNIST) int64 + concatenation exhausts memory.
        n_nn = n * self.n_inliers * self.n_outliers
        triplets = np.empty((n_nn + n * self.n_random, 3), dtype=np.int32)
        weights = np.empty(len(triplets), dtype=np.float64)
        _sample_knn_triplets(X, nbrs, sig, self.n_inliers, self.n_outliers, self.seed,
                             triplets[:n_nn], weights[:n_nn])
        if self.n_random > 0:
            _sample_random_triplets(X, sig, self.n_random, self.seed,
                                    triplets[n_nn:], weights[n_nn:])

        weights[~np.isfinite(weights)] = 0.0
        weights -= weights.min() - 1.0
        weights = tempered_log(weights, self.weight_temp)
        self._log(f"sampled {len(triplets)} triplets: {time.time() - t0:.1f}s")
        return triplets, weights

    def _initial_embedding(self, X):
        n = X.shape[0]
        if isinstance(self.init, np.ndarray):
            return np.array(self.init, dtype=np.float64)
        if self.init == "pca":
            Y = pca(X, self.n_dims)
            return 0.01 * Y / Y[:, 0].std()
        rng = np.random.default_rng(self.seed)
        return 1e-4 * rng.standard_normal((n, self.n_dims))

    # -- Step 5 --------------------------------------------------------------
    def fit_transform(self, X):
        t_start = time.time()
        X = self._preprocess(X)
        n = X.shape[0]
        triplets, weights = self.generate_triplets(X)
        self.triplets_, self.weights_ = triplets, weights

        Y = self._initial_embedding(X)
        vel = np.zeros_like(Y)
        gain = np.ones_like(Y)
        lr = self.lr * n / len(triplets)
        self.loss_history_ = []

        t0 = time.time()
        for it in range(self.n_iters):
            gamma = 0.5 if it < self.switch_iter else 0.8
            # Nesterov-style look-ahead, then delta-bar-delta gain adaptation.
            loss, grad, n_viol = trimap_loss_grad_parallel(Y + gamma * vel, triplets, weights)
            same_sign = np.sign(vel) == np.sign(grad)
            gain = np.where(same_sign, np.maximum(gain * 0.8, 0.01), gain + 0.2)
            vel = gamma * vel - lr * gain * grad
            Y = Y + vel
            self.loss_history_.append(loss)
            if self.verbose and (it + 1) % 100 == 0:
                self._log(f"iter {it + 1:4d}  loss {loss:.3f}  "
                          f"violated {100.0 * n_viol / len(triplets):.2f}%")
        self._log(f"optimisation: {time.time() - t0:.1f}s, "
                  f"total: {time.time() - t_start:.1f}s")
        self.embedding_ = Y
        return Y

    def fit(self, X):
        self.fit_transform(X)
        return self
