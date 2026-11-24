#!/usr/bin/env python
"""Example 1: fit the paper's default surrogate (ORB embedding + PCA + GP) to a
set of labeled crystal structures and check its predictions on held-out ones.

Runs on a laptop CPU in about a minute. By default it uses data that ships with
the repository: 500 structures of matminer's elastic_tensor_2015 with their
bulk modulus K_VRH, and their ORB embeddings computed once on a GPU. Pass
--recompute to compute the embeddings yourself (this downloads the ORB weights
on first use), or adapt `load_data` to read your own structures and targets.

    python examples/01_fit_surrogate.py
    python examples/01_fit_surrogate.py --n-train 100 --pca 10
    python examples/01_fit_surrogate.py --recompute --device cuda
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "04_revision_round1" / "candidate_table" / "out" / "published"
sys.path.insert(0, str(REPO / "02_closed_loop_bo"))      # src.featurizer, src.ltm
sys.path.insert(0, str(REPO / "01_static_benchmark" / "src"))   # gp_regression


def load_data(recompute=False, device="cpu"):
    """Return (X, y): one embedding row and one target value per structure.

    To use your own data, build a list of ase.Atoms and a matching array of
    targets, and call `embed(atoms_list, device)` to get X.
    """
    table = pd.read_parquet(DATA / "seed_pool_bm.parquet")
    y = table["y"].to_numpy(dtype=float)                  # K_VRH in GPa
    if recompute:
        from src.ltm import atoms_from_json
        X = embed([atoms_from_json(a) for a in table["atoms_json"]], device)
    else:
        X = np.load(DATA / "orb_raw_seed_bm.npy")         # (500, 256), stored embeddings
    return X, y


def embed(atoms_list, device="cpu"):
    """Mean-pooled ORB-v3 node embedding, 256 numbers per structure.

    The paper's other featurizers (MACE, UMA, SOAP) live in
    01_static_benchmark/src/common.py and return arrays of the same kind.
    """
    from src.featurizer import ORBFeaturizer
    feats, failed = ORBFeaturizer(n_components=0, device=device)._featurize_raw(atoms_list)
    if failed.any():
        raise RuntimeError(f"{int(failed.sum())} structures could not be embedded")
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--pca", type=int, default=50, help="embedding dimensions kept (the paper uses 10, 25 or 50)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    from sklearn.preprocessing import StandardScaler
    from gp_regression import train_gp      # the benchmark's GP: BoTorch SingleTaskGP, fitted by marginal likelihood

    X, y = load_data(args.recompute, args.device)
    order = np.random.default_rng(args.seed).permutation(len(y))
    train, test = order[:args.n_train], order[args.n_train:]

    # 1. Reduce the embedding with PCA. The basis is fitted on the training set only.
    pca = PCA(n_components=args.pca).fit(X[train])
    Z_train, Z_test = pca.transform(X[train]), pca.transform(X[test])

    # 2. Standardize inputs and target, then fit the GP (as in gp_regression.py).
    xs, ys = StandardScaler().fit(Z_train), StandardScaler().fit(y[train].reshape(-1, 1))
    Xt = torch.tensor(xs.transform(Z_train), dtype=torch.float64)
    yt = torch.tensor(ys.transform(y[train].reshape(-1, 1)).ravel(), dtype=torch.float64)
    model = train_gp(Xt, yt, device="cpu")
    model.eval()

    # 3. Predict the held-out structures: a mean and a standard deviation for each.
    with torch.no_grad():
        post = model.posterior(torch.tensor(xs.transform(Z_test), dtype=torch.float64), observation_noise=True)
    mu = ys.inverse_transform(post.mean.detach().numpy().reshape(-1, 1)).ravel()
    sd = post.variance.sqrt().detach().numpy().ravel() * float(ys.scale_[0])

    # 4. Score: accuracy, ranking, and whether the predicted intervals hold.
    z = (y[test] - mu) / sd
    print(f"train {len(train)}  test {len(test)}  PCA {args.pca}")
    print(f"R2        {r2_score(y[test], mu):.3f}")
    print(f"RMSE      {np.sqrt(np.mean((y[test] - mu) ** 2)):.1f} GPa")
    print(f"Spearman  {spearmanr(y[test], mu)[0]:.3f}")
    for level in (0.68, 0.95):
        inside = np.mean(np.abs(z) <= norm.ppf((1 + level) / 2))
        print(f"{int(level * 100)}% interval contains {inside:.1%} of the held-out values")


if __name__ == "__main__":
    main()
