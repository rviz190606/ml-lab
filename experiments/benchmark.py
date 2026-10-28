"""Reproduces Figure 4, Table 1 (PCC vs GS) and Table 2 (runtime).

Example:
    python experiments/benchmark.py --datasets coil20 usps mnist fashion_mnist
    python experiments/benchmark.py --datasets covertype --methods TriMap UMAP PCA
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.datasets import DATASETS, load  # noqa: E402
from experiments.methods import METHODS  # noqa: E402
from trimap import global_score, nn_accuracy, pearson_cluster_correlation  # noqa: E402
from trimap.metrics import pca_reconstruction_error  # noqa: E402

OUT = os.path.join(ROOT, "results")


def scatter(ax, Y, y, title):
    codes = pd.factorize(y)[0]
    ax.scatter(Y[:, 0], Y[:, 1], c=codes, cmap="tab20" if codes.max() >= 10 else "tab10",
               s=0.5 if len(Y) > 20000 else 2, alpha=0.7, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["coil20", "usps", "mnist", "fashion_mnist"],
                    choices=list(DATASETS))
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--max-n", type=int, default=None, help="random subsample size")
    ap.add_argument("--pcc-repeats", type=int, default=10,
                    help="PCC repeats (paper: 100); 0 disables PCC")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, "embeddings"), exist_ok=True)

    rows = []
    fig, axes = plt.subplots(len(args.datasets), len(args.methods),
                             figsize=(4 * len(args.methods), 4 * len(args.datasets)),
                             squeeze=False)
    for r, ds in enumerate(args.datasets):
        X, y = load(ds)
        if args.max_n and len(X) > args.max_n:
            sub = np.random.default_rng(0).choice(len(X), args.max_n, replace=False)
            X, y = X[sub], y[sub]
        X = X.astype(np.float64)
        e_pca = pca_reconstruction_error(X, 2)
        print(f"\n=== {ds}: n={len(X)}, d={X.shape[1]} ===", flush=True)
        for c, m in enumerate(args.methods):
            t0 = time.time()
            Y = METHODS[m](X)
            secs = time.time() - t0
            nn = nn_accuracy(Y, y, n_samples=20000)
            gs = global_score(X, Y, e_pca)
            pcc = (pearson_cluster_correlation(X, Y, n_repeats=args.pcc_repeats)
                   if args.pcc_repeats > 0 else np.nan)
            print(f"{m:7s} time={secs:8.1f}s  NN={nn:.3f}  GS={gs:.3f}  PCC={pcc:.3f}",
                  flush=True)
            rows.append(dict(dataset=ds, n=len(X), method=m, seconds=round(secs, 1),
                             nn_acc=round(nn, 3), gs=round(gs, 3), pcc=round(pcc, 3)))
            np.save(os.path.join(OUT, "embeddings", f"{ds}_{m}.npy"), Y.astype(np.float32))
            scatter(axes[r, c], Y, y, f"{ds} - {m}\n(NN={nn:.2f}, GS={gs:.2f})")
            # Write incrementally so long runs keep partial results.
            pd.DataFrame(rows).to_csv(os.path.join(OUT, f"benchmark{args.tag}.csv"), index=False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig4_benchmark{args.tag}.png"), dpi=120)
    df = pd.DataFrame(rows)
    print("\n", df.pivot(index="dataset", columns="method", values=["nn_acc", "gs", "pcc", "seconds"]))


if __name__ == "__main__":
    main()
