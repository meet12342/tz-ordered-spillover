# ---------------------------------------------------------------------------
# tvp_var.py  -  TVP-VAR(1) by forgetting-factor Kalman filter and
#                Diebold-Yilmaz connectedness.  Eqs. (30)-(38).
# ---------------------------------------------------------------------------
import numpy as np
from .connectedness import gfevd


def tvp_var(Y, kappa1=0.99, kappa2=0.96, prior_var=1.0):
    """Koop and Korobilis (2013) forgetting-factor filter for a TVP-VAR(1).

    y_t = Z_t beta_t + eps_t,   Z_t = I_K (x) [1, y_{t-1}']      (Eq. 32)
    beta_t = beta_{t-1} + eta_t
    Returns A_t (K x K slope matrices) and Sigma_t for t = 1..T-1.
    """
    T, K = Y.shape
    m = K + 1
    beta = np.zeros(K * m)
    P = np.eye(K * m) * prior_var
    Sig = np.cov(Y[: max(30, K + 2)].T)
    A_path, S_path = [], []
    for t in range(1, T):
        x = np.concatenate([[1.0], Y[t - 1]])
        Z = np.kron(np.eye(K), x)                      # K x K*m
        P_pred = P / kappa1                            # Eq. (33)
        e = Y[t] - Z @ beta
        Sig = kappa2 * Sig + (1 - kappa2) * np.outer(e, e)   # Eq. (34)
        Fm = Z @ P_pred @ Z.T + Sig
        Kg = P_pred @ Z.T @ np.linalg.inv(Fm)          # Eq. (35)
        beta = beta + Kg @ e
        P = P_pred - Kg @ Z @ P_pred
        B = beta.reshape(K, m)
        A_path.append(B[:, 1:].copy())
        S_path.append(Sig.copy())
    return np.array(A_path), np.array(S_path)


def ma_coeffs(A, H):
    """Eq. (31): Psi_h = A^h for a VAR(1); Psi_0 = I."""
    K = A.shape[0]
    Psi = [np.eye(K)]
    for _ in range(1, H):
        Psi.append(Psi[-1] @ A)
    return Psi


def connectedness(theta_tilde):
    """Eq. (38): FROM, TO, NET and TCI, in per cent."""
    C = 100 * theta_tilde
    off = C - np.diag(np.diag(C))
    FROM = off.sum(axis=1)
    TO = off.sum(axis=0)
    NET = TO - FROM
    TCI = FROM.mean()
    return dict(C=C, FROM=FROM, TO=TO, NET=NET, TCI=TCI)


def dynamic_connectedness(Y, H=10, **kw):
    A_path, S_path = tvp_var(Y, **kw)
    out = []
    for A, S in zip(A_path, S_path):
        out.append(connectedness(gfevd(ma_coeffs(A, H), S, H)))
    return out
