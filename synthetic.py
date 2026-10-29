# ---------------------------------------------------------------------------
# synthetic.py  -  ground-truth SVAR systems, structure learners and the
#                  recovery metrics.  Eqs. (10), (52)-(57).
# ---------------------------------------------------------------------------
from math import comb
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize


def count_dags(n):
    """Eq. (10): Robinson (1973) recurrence for labelled DAGs on n nodes."""
    a = [1]
    for m in range(1, n + 1):
        a.append(sum((-1) ** (k + 1) * comb(m, k) * 2 ** (k * (m - k)) * a[m - k]
                     for k in range(1, m + 1)))
    return a[n]


def make_system(d=10, T=2500, density=0.25, seed=0, link="linear"):
    """Eq. (52): x_t = f(W0' x_t + W1' x_{t-1}) + e_t, W0 respecting an order."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)                 # order[p] = node at position p
    pos = np.empty(d, int); pos[order] = np.arange(d)
    W0 = np.zeros((d, d))
    for j in range(d):
        for i in range(d):
            if pos[j] < pos[i] and rng.random() < density:
                W0[j, i] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
    W1 = (rng.random((d, d)) < density) * rng.choice([-1, 1], (d, d)) * rng.uniform(0.1, 0.4, (d, d))
    rad = max(abs(np.linalg.eigvals(W1)))
    if rad > 0.8:
        W1 *= 0.8 / rad
    f = (lambda z: z) if link == "linear" else np.tanh
    X = np.zeros((T + 100, d))
    for t in range(1, T + 100):
        for i in order:                         # topological order: no inversion
            X[t, i] = f(X[t] @ W0[:, i] + X[t - 1] @ W1[:, i]) + rng.normal()
    return X[100:], W0, pos


def residualise(X):
    """Strip the lagged part by OLS so that contemporaneous structure remains."""
    Xl, Xc = X[:-1], X[1:]
    B, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xl)), Xl]), Xc, rcond=None)
    return Xc - np.column_stack([np.ones(len(Xl)), Xl]) @ B


def admissible(pos):
    d = len(pos)
    return np.array([[pos[j] < pos[i] for i in range(d)] for j in range(d)], float)


def corrupt(pos, frac, seed=0):
    """Permute a fraction of positions at random (order-corruption study)."""
    rng = np.random.default_rng(seed)
    pos = pos.copy()
    k = int(round(frac * len(pos)))
    if k >= 2:
        idx = rng.choice(len(pos), k, replace=False)
        pos[idx] = pos[rng.permutation(idx)]
    return pos


def thresholded_correlation(R, mask=None):
    S = np.abs(np.corrcoef(R.T)); np.fill_diagonal(S, 0)
    return S if mask is None else S * mask


def notears_linear(R, lam=0.05, mask=None, max_iter=20, h_tol=1e-8, rho_max=1e16):
    """Zheng et al. (2018), Eq. (53). With a mask the graph is
    acyclic by construction, so the acyclicity constraint is switched off."""
    n, d = R.shape
    msk = np.ones((d, d)) - np.eye(d) if mask is None else mask * (1 - np.eye(d))

    def _adj(w):
        return ((w[:d * d] - w[d * d:]).reshape(d, d)) * msk

    def _h(W):
        E = expm(W * W)
        return np.trace(E) - d, E.T * W * 2

    def _func(w, rho, alpha):
        W = _adj(w)
        Rm = R - R @ W
        loss = 0.5 / n * (Rm ** 2).sum()
        g = -1.0 / n * R.T @ Rm
        if mask is None:
            h, gh = _h(W)
            obj = loss + 0.5 * rho * h * h + alpha * h + lam * w.sum()
            g = g + (rho * h + alpha) * gh
        else:
            obj = loss + lam * w.sum()
        g = g * msk
        return obj, np.concatenate([g.ravel() + lam, -g.ravel() + lam])

    w = np.zeros(2 * d * d)
    bnds = [(0, None)] * (2 * d * d)
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        while rho < rho_max:
            sol = minimize(_func, w, args=(rho, alpha), jac=True,
                           method="L-BFGS-B", bounds=bnds)
            h_new = _h(_adj(sol.x))[0] if mask is None else 0.0
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w, h = sol.x, h_new
        alpha += rho * h
        if h <= h_tol or mask is not None:
            break
    return np.abs(_adj(w))


def auroc(scores, truth):
    """Eq. (55): Mann-Whitney form of the area under the ROC curve."""
    from scipy.stats import rankdata
    off = ~np.eye(len(truth), dtype=bool)
    s, y = scores[off], truth[off] != 0
    r = rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def top_k(scores, k):
    off = ~np.eye(len(scores), dtype=bool)
    flat = np.where(off, scores, -np.inf).ravel()
    idx = np.argsort(flat)[::-1][:k]
    return [divmod(int(i), len(scores)) for i in idx]


def f1_and_violation(scores, truth, pos):
    """Eqs. (56)-(57): F1 of the k strongest edges (k = true edge count) and
    the share of those edges pointing against the true order."""
    k = int((truth != 0).sum())
    edges = top_k(scores, k)
    tp = sum(truth[j, i] != 0 for j, i in edges)
    prec, rec = tp / k, tp / k
    f1 = 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)
    viol = sum(pos[j] > pos[i] for j, i in edges) / k
    return float(f1), float(viol)
