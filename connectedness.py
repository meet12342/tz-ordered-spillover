# ---------------------------------------------------------------------------
# connectedness.py  -  generalized FEVD and the clock classification
# ---------------------------------------------------------------------------
import numpy as np


def gfevd(Psi, Sigma, H):
    """Generalized FEVD (Pesaran-Shin). At H = 1 this reduces to the
    squared innovation correlation and is therefore symmetric."""
    K = Sigma.shape[0]
    theta = np.zeros((K, K))
    for i in range(K):
        denom = sum(Psi[h] @ Sigma @ Psi[h].T for h in range(H))[i, i]
        for j in range(K):
            num = sum((Psi[h] @ Sigma)[i, j] ** 2 for h in range(H))
            theta[i, j] = num / (Sigma[j, j] * denom)
    return theta / theta.sum(axis=1, keepdims=True)      # row-normalise


def verify_proposition(Sigma, tol=1e-12):
    """theta_ij(1) == rho_ij^2 for any positive-definite Sigma."""
    K = Sigma.shape[0]
    Psi = [np.eye(K)]                       # Psi_0 = I for any VAR order
    raw = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            raw[i, j] = (Sigma[i, j] ** 2) / (Sigma[j, j] * Sigma[i, i])
    d = np.sqrt(np.diag(Sigma))
    rho = Sigma / np.outer(d, d)
    err = np.abs(raw - rho ** 2).max()
    assert err < tol, f"identity violated by {err:.3e}"
    return err


def classify_by_clock(theta, ranks):
    """Split same-day connectedness mass by the physical order of closes."""
    out = {"forward": 0.0, "backward": 0.0,
           "simultaneous": 0.0, "outside": 0.0}
    K = theta.shape[0]
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            ri, rj = ranks[i], ranks[j]
            if ri is None or rj is None:
                out["outside"] += theta[i, j]
            elif rj < ri:
                out["forward"] += theta[i, j]
            elif rj > ri:
                out["backward"] += theta[i, j]
            else:
                out["simultaneous"] += theta[i, j]
    ordered = out["forward"] + out["backward"]
    out["backward_share"] = out["backward"] / ordered
    return out
