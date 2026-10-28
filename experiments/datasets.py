"""Dataset loaders (downloaded once and cached under ./data)."""

import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _cached(name, loader):
    path = os.path.join(DATA_DIR, f"{name}.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return d["X"], d["y"]
    X, y = loader()
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    return X, y


def _openml(name, version="active"):
    from sklearn.datasets import fetch_openml
    def load():
        d = fetch_openml(name, version=version, as_frame=False, parser="auto",
                         data_home=DATA_DIR)
        X = np.asarray(d.data, dtype=np.float32)
        y = np.asarray(d.target)
        return X, y
    return load


def s_curve(n=5000, seed=0):
    from sklearn.datasets import make_s_curve
    X, t = make_s_curve(n, random_state=seed)
    return X, t  # t = position along the manifold (continuous "label")


def mnist():
    return _cached("mnist", _openml("mnist_784", 1))


def fashion_mnist():
    return _cached("fashion_mnist", _openml("Fashion-MNIST", 1))


def usps():
    return _cached("usps", _openml("usps", 2))


def coil20():
    url = ("https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/"
           "coil-20/coil-20-proc.zip")

    def load():
        import io
        import re
        import urllib.request
        import zipfile

        from PIL import Image
        path = os.path.join(DATA_DIR, "coil20_raw", "coil-20-proc.zip")
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(url, path)
        X, y = [], []
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                m = re.search(r"obj(\d+)__\d+\.png$", name)
                if m:
                    img = Image.open(io.BytesIO(z.read(name))).convert("L")
                    X.append(np.asarray(img, dtype=np.float32).ravel() / 255.0)
                    y.append(int(m.group(1)))
        return np.stack(X), np.array(y)
    return _cached("coil20", load)


def covertype():
    def load():
        from sklearn.datasets import fetch_covtype
        d = fetch_covtype(data_home=DATA_DIR)
        X = d.data.astype(np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        return X, d.target
    return _cached("covertype", load)


def newsgroups20():
    def load():
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        d = fetch_20newsgroups(subset="all", data_home=DATA_DIR,
                               remove=("headers", "footers", "quotes"))
        tfidf = TfidfVectorizer(max_features=20000, stop_words="english").fit_transform(d.data)
        # Sparse TF-IDF -> 100 dims (the paper also reduces to 100 dims).
        X = TruncatedSVD(100, random_state=0).fit_transform(tfidf).astype(np.float32)
        return X, d.target
    return _cached("newsgroups20", load)


DATASETS = {
    "coil20": coil20,
    "usps": usps,
    "newsgroups20": newsgroups20,
    "mnist": mnist,
    "fashion_mnist": fashion_mnist,
    "covertype": covertype,
}


def load(name):
    X, y = DATASETS[name]()
    return np.asarray(X, dtype=np.float32), np.asarray(y)
