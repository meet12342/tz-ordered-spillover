# ---------------------------------------------------------------------------
# volatility.py  -  range-based estimators and the forecasting target
# Eqs. (1)-(6) of the manuscript.
# ---------------------------------------------------------------------------
import numpy as np


def overnight_return(O, C):
    """Eq. (1): r_on,t = ln O_t - ln C_{t-1}. First value is NaN."""
    r = np.full_like(np.asarray(O, float), np.nan)
    r[1:] = np.log(O[1:]) - np.log(C[:-1])
    return r


def parkinson(H, L):
    """Eq. (2): Parkinson (1980) range estimator."""
    return (np.log(H / L) ** 2) / (4.0 * np.log(2.0))


def garman_klass(O, H, L, C):
    """Eq. (3): Garman and Klass (1980)."""
    return 0.5 * np.log(H / L) ** 2 - (2 * np.log(2) - 1) * np.log(C / O) ** 2


def rogers_satchell(O, H, L, C):
    """Eq. (4): Rogers and Satchell (1991), drift-robust."""
    return np.log(H / C) * np.log(H / O) + np.log(L / C) * np.log(L / O)


def yang_zhang(O, H, L, C, window=21):
    """Eq. (5): Yang and Zhang (2000) over a rolling window (the REJECTED target)."""
    n = len(C)
    ro = overnight_return(O, C)
    rc = np.log(C / O)
    rs = rogers_satchell(O, H, L, C)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    out = np.full(n, np.nan)
    for t in range(window, n):
        s = slice(t - window + 1, t + 1)
        out[t] = (np.nanvar(ro[s], ddof=1) + k * np.var(rc[s], ddof=1)
                  + (1 - k) * np.mean(rs[s]))
    return out


def target(O, H, L, C, eps=1e-10):
    """Eq. (6): y_t = ln( r_on,t^2 + RS_t ), the daily target actually used."""
    return np.log(overnight_return(O, C) ** 2 + rogers_satchell(O, H, L, C) + eps)


def acf1(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    return float((x[1:] * x[:-1]).sum() / (x * x).sum())
