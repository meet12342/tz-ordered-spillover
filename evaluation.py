# ---------------------------------------------------------------------------
# evaluation.py  -  folds, metrics and tests.  Eqs. (42)-(51).
# ---------------------------------------------------------------------------
import numpy as np
from scipy import stats


def walk_forward_folds(n, n_folds=4, embargo=21, min_train=0.5):
    """Expanding-window folds; `embargo` rows dropped between train and test."""
    start = int(n * min_train)
    size = (n - start) // n_folds
    folds = []
    for f in range(n_folds):
        te0 = start + f * size
        te1 = n if f == n_folds - 1 else te0 + size
        folds.append((np.arange(0, te0 - embargo), np.arange(te0, te1)))
    return folds


def standardise(train, *others):
    """Eq. (42): z-scores using training moments only."""
    mu, sd = np.nanmean(train, 0), np.nanstd(train, 0)
    return [(a - mu) / sd for a in (train,) + others]


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))          # Eq. (43)


def mae(e):
    return float(np.mean(np.abs(e)))                        # Eq. (43)


def improvement(rmse_model, rmse_bench):
    return 100.0 * (1.0 - rmse_model / rmse_bench)          # Eq. (44)


def newey_west_lrv(d, lags=None):
    """Eq. (46): Bartlett-kernel long-run variance (Newey and West, 1987)."""
    d = np.asarray(d, float) - np.mean(d)
    T = len(d)
    if lags is None:
        lags = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    g0 = d @ d / T
    s = g0
    for k in range(1, lags + 1):
        gk = d[k:] @ d[:-k] / T
        s += 2 * (1 - k / (lags + 1)) * gk
    return s


def diebold_mariano(e1, e2, lags=None):
    """Eq. (45): DM statistic for squared-error loss; two-sided p-value."""
    d = np.square(e1) - np.square(e2)
    T = len(d)
    dm = d.mean() / np.sqrt(newey_west_lrv(d, lags) / T)
    return float(dm), float(2 * (1 - stats.norm.cdf(abs(dm))))


def spearman_permutation(a, b, B=5000, seed=0):
    """Eqs. (48)-(49): Spearman rho with a permutation p-value (Phipson-Smyth)."""
    rng = np.random.default_rng(seed)
    rho = stats.spearmanr(a, b).statistic
    hits = sum(abs(stats.spearmanr(a, rng.permutation(b)).statistic) >= abs(rho)
               for _ in range(B))
    return float(rho), (1 + hits) / (B + 1)


def ablation_decomposition(rmse_by_variant):
    """Eqs. (50)-(51): gains vs the no-graph model and the nested split."""
    r0 = rmse_by_variant["no_graph"]
    G = {k: 100 * (1 - v / r0) for k, v in rmse_by_variant.items()}
    S = G["unconstrained"] - G["shuffled"]
    parts = {
        "capacity": G["shuffled"],
        "admissible": G["strict"] - G["shuffled"],
        "simultaneous_continuous": G["plus_simultaneous"] - G["strict"],
        "look_ahead": G["unconstrained"] - G["plus_simultaneous"],
    }
    binary_share = 100 * (G["unconstrained"] - G["strict"]) / S
    true_share = 100 * parts["look_ahead"] / S
    return G, S, parts, binary_share, true_share, binary_share / true_share
