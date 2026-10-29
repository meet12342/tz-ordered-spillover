# ---------------------------------------------------------------------------
# baselines.py  -  random walk, AR(1) and HAR, Eqs. (26)-(27)
# ---------------------------------------------------------------------------
import numpy as np


def _ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def har_features(y):
    """Daily, weekly (5) and monthly (22) averages of the target, Corsi (2009)."""
    n = len(y)
    d = np.full(n, np.nan); w = np.full(n, np.nan); m = np.full(n, np.nan)
    for t in range(21, n):
        d[t] = y[t]
        w[t] = y[t - 4:t + 1].mean()
        m[t] = y[t - 21:t + 1].mean()
    return np.column_stack([d, w, m])


def fit_predict_har(y_train, y_full, test_idx):
    """Eq. (27). Coefficients from the training span only; one-step forecasts."""
    Ftr = har_features(y_train)
    X, Y = Ftr[21:-1], y_train[22:]
    b = _ols(X, Y)
    F = har_features(y_full)
    return np.array([b[0] + F[t - 1] @ b[1:] for t in test_idx])


def fit_predict_ar1(y_train, y_full, test_idx):
    """Eq. (26), AR(1)."""
    b = _ols(y_train[:-1, None], y_train[1:])
    return np.array([b[0] + b[1] * y_full[t - 1] for t in test_idx])


def predict_rw(y_full, test_idx):
    """Eq. (26), random walk: y_hat_t = y_{t-1}."""
    return np.array([y_full[t - 1] for t in test_idx])
