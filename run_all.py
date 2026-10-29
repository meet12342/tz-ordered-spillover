#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# run_all.py  -  executes every component of the companion code and prints
#                the checks reported in the manuscript.  Runs on CPU in a few
#                minutes.  Market data are NOT downloaded here; parts that need
#                the real panel are demonstrated on synthetic data and say so.
# ---------------------------------------------------------------------------
import time
import numpy as np
import torch

from tzspill.clock_mask import ClockMask
from tzspill.connectedness import gfevd, verify_proposition, classify_by_clock
from tzspill.forecaster import GraphForecaster, train, tau_schedule
from tzspill.leakage_test import LeakageTest, LeakageTestFailure
from tzspill.tvp_var import tvp_var, ma_coeffs, connectedness, dynamic_connectedness
from tzspill.evaluation import (walk_forward_folds, rmse, improvement,
                                diebold_mariano, ablation_decomposition)
from tzspill.volatility import (rogers_satchell, yang_zhang, target, acf1,
                                parkinson, garman_klass)
from tzspill import synthetic as syn

np.set_printoptions(precision=3, suppress=True)
t0 = time.time()


def banner(s):
    print("\n" + "=" * 76 + f"\n{s}\n" + "=" * 76)


# ---------------------------------------------------------------- 1. clock mask
banner("1. Closing ranks and the clock mask  (Eqs. 8-9, Table 1)")
CLOSE_UTC = {"TWSE": "05:30", "NKY": "06:00", "KS11": "06:30", "HSI": "08:00",
             "NIFIT": "10:00", "DAX": "16:30", "SX8P": "16:30", "NDX": "21:00",
             "REMX": "21:00", "SOX": "21:00", "GLD": None, "UST": None}
cm = ClockMask(CLOSE_UTC)
print("derived ranks:", cm.ranks)
print(f"admitted same-day edges: {cm.admitted()} of {cm.possible()}  "
      f"({100 * (1 - cm.admitted() / cm.possible()):.1f}% removed)")
print(f"hypothesis space: 2^{cm.possible()} = {2.0 ** cm.possible():.2e}  ->  "
      f"2^{cm.admitted()} = {2.0 ** cm.admitted():.2e}")
dags = syn.count_dags(12)
print(f"labelled DAGs on 12 nodes (Robinson, Eq. 10): {dags:.3e}")
print(f"clock restriction is {np.log10(dags / 2 ** cm.admitted()):.1f} orders "
      f"of magnitude stronger than acyclicity")
print("mask M (rows = receiver i, cols = transmitter j):")
print(cm.M.astype(int))

# ---------------------------------------------------------------- 2. target
banner("2. Volatility estimators and target autocorrelation  (Eqs. 1-6)")
rng = np.random.default_rng(7)
T = 2500
logv = np.zeros(T)
for t in range(1, T):                      # persistent latent log-variance
    logv[t] = 0.95 * logv[t - 1] + 0.35 * rng.normal()
sig = np.exp(0.5 * (logv - 4.5))
C = 100 * np.exp(np.cumsum(sig * rng.normal(size=T)))
O = C * np.exp(0.4 * sig * rng.normal(size=T)); O[1:] = C[:-1] * np.exp(0.4 * sig[1:] * rng.normal(size=T - 1))
H = np.maximum(O, C) * np.exp(np.abs(0.6 * sig * rng.normal(size=T)))
L = np.minimum(O, C) * np.exp(-np.abs(0.6 * sig * rng.normal(size=T)))
print("synthetic OHLC series (stochastic volatility), T =", T)
print(f"  mean Parkinson      {np.mean(parkinson(H, L)):.3e}")
print(f"  mean Garman-Klass   {np.mean(garman_klass(O, H, L, C)):.3e}")
print(f"  mean Rogers-Satchell {np.mean(rogers_satchell(O, H, L, C)):.3e}")
print(f"  ACF(1) of 21-day Yang-Zhang      : {acf1(yang_zhang(O, H, L, C)):.3f}"
      "   (windowed: near 1, the rejected target)")
print(f"  ACF(1) of ln(r_on^2 + RS) target : {acf1(target(O, H, L, C)):.3f}"
      "   (daily: leaves something to forecast)")

# ---------------------------------------------------------------- 3. proposition
banner("3. Proposition 1: theta_ij(1) = rho_ij^2  (Eqs. 36-40)")
worst = 0.0
for s in range(1000):
    r = np.random.default_rng(s)
    A = r.normal(size=(12, 12))
    Sig = A @ A.T + 12 * np.eye(12)
    worst = max(worst, verify_proposition(Sig))
print(f"1000 random positive-definite 12x12 matrices: max |theta - rho^2| = {worst:.1e}")
Sig = np.cov(np.random.default_rng(1).normal(size=(500, 12)).T)
th = gfevd([np.eye(12)], Sig, 1)
d = np.sqrt(np.diag(Sig)); rho2 = (Sig / np.outer(d, d)) ** 2
R = rho2.sum(1)
i, j = 0, 7
print(f"normalised ratio theta~_ij/theta~_ji = {th[i, j] / th[j, i]:.6f};  "
      f"R_j/R_i = {R[j] / R[i]:.6f}   (Eq. 40)")
ranks = [cm.ranks[c] for c in CLOSE_UTC]
cls = classify_by_clock(th, ranks)
print("clock classification of this H=1 matrix (Eq. 41):",
      {k: round(v, 3) for k, v in cls.items()})

# ---------------------------------------------------------------- 4. TVP-VAR
banner("4. TVP-VAR known-answer tests  (Eqs. 30-38, tests TC-11 to TC-13)")
r = np.random.default_rng(3)
Y = np.zeros((1500, 3))
for t in range(1, 1500):
    e = r.normal(size=3)
    Y[t, 0] = 0.3 * Y[t - 1, 0] + e[0]
    Y[t, 1] = 0.6 * Y[t - 1, 0] + 0.2 * Y[t - 1, 1] + e[1]
    Y[t, 2] = 0.6 * Y[t - 1, 0] + 0.2 * Y[t - 1, 2] + e[2]
res = dynamic_connectedness(Y, H=10)
NET = np.mean([c["NET"] for c in res[100:]], axis=0)
print("three-node system, node 0 drives nodes 1 and 2")
print("  average NET:", NET.round(2), "-> largest net transmitter = node", int(np.argmax(NET)))
rows = np.array([c["C"].sum(1) for c in res])
nets = np.array([c["NET"].sum() for c in res])
print(f"  max |row sum - 100| = {np.abs(rows - 100).max():.1e};  "
      f"max |sum NET| = {np.abs(nets).max():.1e}")
perm = [2, 0, 1]
th0 = gfevd(ma_coeffs(np.diag([.3, .2, .2]), 10), np.cov(Y.T), 10)
P = np.eye(3)[perm]
thp = gfevd(ma_coeffs(P @ np.diag([.3, .2, .2]) @ P.T, 10), P @ np.cov(Y.T) @ P.T, 10)
print(f"  permutation equivariance error = {np.abs(P.T @ thp @ P - th0).max():.1e}")
Yc = np.zeros((2000, 1))
for t in range(1, 2000):
    Yc[t] = 0.5 * Yc[t - 1] + r.normal()
for k1 in (1.00, 0.99):
    A_path, _ = tvp_var(Yc, kappa1=k1, kappa2=0.96)
    print(f"  constant-parameter data, kappa1={k1:.2f}: max |a_t - 0.5| after burn-in = "
          f"{np.abs(A_path[500:, 0, 0] - 0.5).max():.3f}")
Yj = np.zeros((2000, 1))
for t in range(1, 2000):
    Yj[t] = (0.10 if t < 1000 else 0.80) * Yj[t - 1] + r.normal()
A_path, _ = tvp_var(Yj)
print(f"  jump 0.10 -> 0.80 at t=1000: filtered a_t mean before = "
      f"{A_path[300:998, 0, 0].mean():.2f}, after = {A_path[1300:, 0, 0].mean():.2f}")

# ---------------------------------------------------------------- 5. forecaster + gate
banner("5. Graph forecaster and the three-condition leakage gate  (Eqs. 11-25)")
torch.manual_seed(0)
K, Lb = 12, 22
Mt = torch.tensor(cm.M)
ordered = GraphForecaster(K, mask=Mt)
uncon = GraphForecaster(K, mask=None)
n_par = lambda m: sum(p.numel() for p in m.parameters())
print(f"parameters: ordered = {n_par(ordered)}, unconstrained = {n_par(uncon)}  (capacity parity)")
print("tau schedule (Eq. 16):", [round(tau_schedule(e, 300), 3) for e in (0, 100, 200, 299)])
LeakageTest(ordered, cm.M).assert_all()
try:
    ordered.inject_fault = True
    LeakageTest(ordered, cm.M).assert_all()
    print("TC-17 FAILED: injected leak not detected")
except LeakageTestFailure as e:
    print("TC-17 injected leak detected, run halted ->", e)
finally:
    ordered.inject_fault = False

# small synthetic panel with true early-to-late same-day transmission
r = np.random.default_rng(11)
n = 1400
order_rank = np.array([cm.ranks[c] if cm.ranks[c] else 0 for c in CLOSE_UTC])
Ysyn = np.zeros((n, K))
for t in range(1, n):
    e = r.normal(size=K) * 0.6
    for i in np.argsort(order_rank, kind="stable"):
        early = [j for j in range(K) if cm.M[i, j] == 1]
        Ysyn[t, i] = 0.5 * Ysyn[t - 1, i] + 0.15 * Ysyn[t - 1].mean() \
            + (0.35 * Ysyn[t, early].mean() if early else 0) + e[i]
X = np.stack([Ysyn[t - Lb:t].T for t in range(Lb, n)])      # (N, K, L)
S = Ysyn[Lb:n]; Yt = Ysyn[Lb:n]
folds = walk_forward_folds(len(Yt), n_folds=2, embargo=21)
for name, mask in [("ordered", Mt), ("unconstrained", None)]:
    errs = []
    for k, (tr, te) in enumerate(folds):
        m = GraphForecaster(K, mask=mask)
        if mask is not None:
            LeakageTest(m, cm.M).assert_all()
        Xt = torch.tensor(X, dtype=torch.float32)
        St = torch.tensor(S, dtype=torch.float32)
        yt = torch.tensor(Yt, dtype=torch.float32)
        m, ep, _ = train(m, Xt[tr], St[tr], yt[tr], epochs=60, patience=15, seed=k)
        with torch.no_grad():
            p = m(Xt[te], St[te], 0.4, deterministic=True).numpy()
        errs.append(p - Yt[te])
    e = np.concatenate(errs)
    print(f"  {name:13s}: RMSE = {rmse(e):.4f}  (epochs used, last fold: {ep})")
    if name == "ordered":
        e_ord = e
    else:
        e_unc = e
har_like = np.concatenate([Ysyn[Lb - 1:n - 1][te] * 0.5 - Yt[te] for _, te in folds])
print(f"  naive 0.5*y_(t-1) RMSE = {rmse(har_like):.4f}")
dm, p = diebold_mariano(e_ord.ravel(), e_unc.ravel())
print(f"  DM (ordered vs unconstrained): stat = {dm:.2f}, p = {p:.3f}")
print("  note: demonstration on synthetic data; the paper's numbers come from the market panel")

# ---------------------------------------------------------------- 6. ablation arithmetic
banner("6. Nested ablation arithmetic  (Eqs. 50-51, Table 8)")
G, Ssig, parts, b, tshare, factor = ablation_decomposition(
    dict(no_graph=0.8617, shuffled=0.8340, strict=0.8134,
         plus_simultaneous=0.7848, unconstrained=0.7627))
print("gains (pp):", {k: round(v, 2) for k, v in G.items()})
print(f"real signal S = {Ssig:.2f} pp; split:", {k: round(v, 2) for k, v in parts.items()})
print(f"binary contrast share = {b:.1f}%, nested share = {tshare:.1f}%, factor = {factor:.2f}")
print(f"HAR improvement: ordered {improvement(0.8152, 0.8610):.1f}%, "
      f"unconstrained {improvement(0.7593, 0.8610):.1f}%")

# ---------------------------------------------------------------- 7. synthetic structure recovery
banner("7. Synthetic structure recovery with and without the order  (Eqs. 52-57)")
rows = {}
for s in range(5):
    Xs, W0, pos = syn.make_system(d=10, T=2500, density=0.25, seed=s)
    Rr = syn.residualise(Xs)
    Rr = (Rr - Rr.mean(0)) / Rr.std(0)        # standardise: removes variance-sortability
    Mtrue = syn.admissible(pos)
    for name, sc in [
        ("Thresholded correlation", syn.thresholded_correlation(Rr)),
        ("NOTEARS", syn.notears_linear(Rr)),
        ("Correlation + order", syn.thresholded_correlation(Rr, Mtrue)),
        ("NOTEARS + order", syn.notears_linear(Rr, mask=Mtrue)),
        ("NOTEARS + 40% corrupted order",
         syn.notears_linear(Rr, mask=syn.admissible(syn.corrupt(pos, 0.4, s)))),
    ]:
        a = syn.auroc(sc, W0)
        f1, v = syn.f1_and_violation(sc, W0, pos)
        rows.setdefault(name, []).append((a, f1, v))
print(f"{'method':32s} {'AUROC':>14s} {'F1':>7s} {'violation':>10s}")
for name, v in rows.items():
    v = np.array(v)
    print(f"{name:32s} {v[:, 0].mean():.3f} ± {v[:, 0].std():.3f} "
          f"{v[:, 1].mean():7.3f} {v[:, 2].mean():10.3f}")
print("note: fresh synthetic draws from this script; the paper's Table 12 used the project's own seeds")

print(f"\nall sections completed in {time.time() - t0:.0f} s")
