"""Reproduces Figure 2: S-curve embedded by t-SNE, UMAP, TriMap and PCA.

Local accuracy is measured as in the paper's caption with a precision/recall
AUC proxy; we report the k-NN preservation (fraction of the 10 nearest
neighbours in 3-D that are kept in 2-D) alongside the global score.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.datasets import s_curve  # noqa: E402
from experiments.methods import METHODS  # noqa: E402
from trimap import global_score  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def knn_preservation(X, Y, k=10):
    from sklearn.neighbors import NearestNeighbors
    ix = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[1][:, 1:]
    iy = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y)[1][:, 1:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ix, iy)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()
    X, t = s_curve(args.n)

    fig = plt.figure(figsize=(20, 4))
    ax = fig.add_subplot(1, 5, 1, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap="Spectral", s=2)
    ax.view_init(10, -70)
    ax.set_title("Original (3-D)")
    for p, (name, fn) in enumerate(METHODS.items(), start=2):
        Y = fn(X)
        gs, kp = global_score(X, Y), knn_preservation(X, Y)
        print(f"{name:7s}  kNN-preservation={kp:.3f}  GS={gs:.3f}")
        ax = fig.add_subplot(1, 5, p)
        ax.scatter(Y[:, 0], Y[:, 1], c=t, cmap="Spectral", s=2)
        ax.set_title(f"{name} (kNN={kp:.2f}, GS={gs:.2f})")
        ax.set_xticks([]), ax.set_yticks([])
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_scurve.png"), dpi=130)
    print("saved results/fig2_scurve.png")


if __name__ == "__main__":
    main()
