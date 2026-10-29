# ---------------------------------------------------------------------------
# forecaster.py  -  the masked latent-graph forecaster (complete, runnable)
# Implements Eqs. (11)-(24) of the manuscript.
# ---------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphForecaster(nn.Module):
    """GTS-style latent-graph forecaster with two channels.

    lagged channel  : messages from yesterday's encoder states, unrestricted
    same-day channel: messages from today's values, multiplied by the clock
                      mask M (None => unconstrained variant)
    The ordered and unconstrained variants have identical parameter counts.
    """

    def __init__(self, K, hidden=32, emb=16, lookback=22, mask=None):
        super().__init__()
        self.K, self.lookback = K, lookback
        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.float32)
        self.register_buffer("M", mask)                       # None => unconstrained
        # one global logit pair per directed edge, per channel (GTS style)
        self.phi_lag = nn.Parameter(torch.zeros(K, K, 2))
        self.phi_same = nn.Parameter(torch.zeros(K, K, 2))
        self.encoder = nn.GRU(1, hidden, batch_first=True)   # shared weights
        self.same_emb = nn.Linear(1, emb)
        self.W_lag = nn.Linear(hidden, hidden, bias=False)
        self.W_same = nn.Linear(emb, hidden, bias=False)
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.inject_fault = False          # used only by test TC-17

    # ---- Eq. (15): Gumbel-Softmax edge sampling, diagonal removed ----------
    def sample_adjacency(self, phi, tau, deterministic=False):
        if deterministic:
            A = torch.softmax(phi, dim=-1)[..., 1]
        else:
            A = F.gumbel_softmax(phi, tau=tau, hard=False, dim=-1)[..., 1]
        return A * (1.0 - torch.eye(self.K, device=A.device))

    # ---- Eq. (17): the intervention, A_same (.) M ------------------------------
    def sample_same_day(self, tau=0.4, deterministic=False):
        A = self.sample_adjacency(self.phi_same, tau, deterministic)
        return A if self.M is None else A * self.M

    def _decode(self, h, A_lag, A_same, same_day):
        m_lag = torch.einsum("ij,bjh->bih", A_lag, self.W_lag(h))      # Eq. (18)
        e = self.same_emb(same_day.unsqueeze(-1))
        m_same = self.W_same(torch.einsum("ij,bje->bie", A_same, e))    # Eq. (19)
        out = self.decoder(torch.cat([h, m_lag, m_same], -1)).squeeze(-1)  # Eq. (20)
        if self.inject_fault:              # deliberate leak: own value -> own forecast
            out = out + same_day
        return out

    def encode(self, hist):
        B = hist.shape[0]
        h, _ = self.encoder(hist.reshape(B * self.K, self.lookback, 1))   # Eqs. (11)-(14)
        return h[:, -1, :].reshape(B, self.K, -1)

    def forward(self, hist, same_day, tau, deterministic=False):
        h = self.encode(hist)
        A_lag = self.sample_adjacency(self.phi_lag, tau, deterministic)
        A_same = self.sample_same_day(tau, deterministic)
        return self._decode(h, A_lag, A_same, same_day)

    def forward_all_edges_on(self, same_day, hist=None):
        """Every admissible edge forced to 1 (used by leakage condition 3)."""
        B = same_day.shape[0]
        if hist is None:
            hist = torch.zeros(B, self.K, self.lookback)
        h = self.encode(hist)
        off = 1.0 - torch.eye(self.K)
        A_same = off if self.M is None else off * self.M
        return self._decode(h, off, A_same, same_day)

    # ---- Eq. (21): MSE + lambda * L1 on both adjacencies ----------------------
    def loss(self, yhat, y, lam=0.02, tau=0.4):
        mse = F.mse_loss(yhat, y)
        l1 = (self.sample_adjacency(self.phi_lag, tau).abs().mean()
              + self.sample_same_day(tau).abs().mean())
        return mse + lam * l1


def tau_schedule(epoch, n_epochs, tau0=2.0, tau1=0.4):
    """Eq. (16): geometric annealing of the Gumbel-Softmax temperature."""
    return tau0 * (tau1 / tau0) ** (epoch / max(n_epochs - 1, 1))


def train(model, hist, same, y, epochs=300, lr=5e-3, wd=1e-5, batch=128,
          clip=5.0, lam=0.02, patience=30, val_frac=0.1, seed=0, verbose=False):
    """Adam + cosine annealing + clipping + early stopping (Table 5 settings)."""
    torch.manual_seed(seed)
    n = hist.shape[0]
    n_val = max(int(n * val_frac), 1)
    tr, va = slice(0, n - n_val), slice(n - n_val, n)          # held-out tail
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)  # Eq. (22)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)  # Eq. (23)
    best, best_state, bad = math.inf, None, 0
    for ep in range(epochs):
        tau = tau_schedule(ep, epochs)
        model.train()
        idx = torch.randperm(n - n_val)
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            opt.zero_grad()
            l = model.loss(model(hist[tr][b], same[tr][b], tau), y[tr][b], lam, tau)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)   # Eq. (24)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = F.mse_loss(model(hist[va], same[va], tau, deterministic=True), y[va]).item()
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 25 == 0:
            print(f"  epoch {ep:3d}  tau={tau:.3f}  val_mse={v:.4f}")
    model.load_state_dict(best_state)
    return model, ep + 1, best
