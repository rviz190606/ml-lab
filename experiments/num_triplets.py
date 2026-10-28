"""Reproduces Figure 3: effect of the number of triplets on MNIST.

(m, m', r) = c * (2, 1, 1) for c in {1, 2, 5, 10, 20}, i.e. |T| = (2c^2 + c) n.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.datasets import load  # noqa: E402
from trimap import TriMap, global_score, nn_accuracy  # noqa: E402
from trimap.metrics import pca_reconstruction_error  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--cs", nargs="+", type=int, default=[1, 2, 5, 10, 20])
    args = ap.parse_args()
    X, y = load(args.dataset)
    X = X.astype(np.float64)
    e_pca = pca_reconstruction_error(X, 2)

    fig, axes = plt.subplots(1, len(args.cs), figsize=(4 * len(args.cs), 4), squeeze=False)
    for ax, c in zip(axes[0], args.cs):
        m, mp, r = 2 * c, c, c
        Y = TriMap(n_inliers=m, n_outliers=mp, n_random=r).fit_transform(X)
        nn, gs = nn_accuracy(Y, y, n_samples=20000), global_score(X, Y, e_pca)
        print(f"c={c:2d} (m,m',r)=({m},{mp},{r}) |T|={m * mp + r}n  NN={nn:.3f} GS={gs:.3f}",
              flush=True)
        ax.scatter(Y[:, 0], Y[:, 1], c=np.unique(y, return_inverse=True)[1], cmap="tab10",
                   s=0.3, linewidths=0, rasterized=True)
        ax.set_title(f"|T|={m * mp + r}n (NN={nn:.3f}, GS={gs:.3f})", fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "results", "fig3_num_triplets.png"), dpi=120)


if __name__ == "__main__":
    main()
