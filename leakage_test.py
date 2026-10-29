# ---------------------------------------------------------------------------
# leakage_test.py  -  the three-condition gate; halts the run on failure
# ---------------------------------------------------------------------------
import numpy as np
import torch


class LeakageTestFailure(AssertionError):
    pass


class LeakageTest:
    """Verifies that no path runs from a market's own contemporaneous
    value to its own prediction. Condition 3 is the one that does the
    real work: it tests the assembled graph end to end, not the
    adjacency in isolation."""

    TOL = 1e-12

    def __init__(self, model, M):
        self.model, self.M = model, M

    def check_diagonal(self):
        for deterministic in (True, False):
            A = self.model.sample_same_day(deterministic=deterministic)
            d = torch.diagonal(A).abs().max().item()
            if d > self.TOL:
                raise LeakageTestFailure(
                    f"same-day diagonal {d:.3e} exceeds {self.TOL:.0e} "
                    f"(deterministic={deterministic})")
        return True

    def check_support(self):
        A = self.model.sample_same_day(deterministic=True).detach().numpy()
        outside = np.abs(A[self.M == 0]).sum()
        if outside != 0.0:
            raise LeakageTestFailure(
                f"model places {outside:.3e} mass outside the mask")
        return True

    def noise_r2(self, seed=0, n=2048):
        """An untrained model with every admissible edge forced on must
        score R2 <= 0 against pure noise. A positive score means a path
        exists from a target into its own prediction."""
        g = torch.Generator().manual_seed(seed)
        K = self.M.shape[0]
        x = torch.randn(n, K, generator=g)
        with torch.no_grad():
            yhat = self.model.forward_all_edges_on(x)
        y = x
        ss_res = ((y - yhat) ** 2).sum().item()
        ss_tot = ((y - y.mean(0)) ** 2).sum().item()
        r2 = 1.0 - ss_res / ss_tot
        if r2 > 0.0:
            raise LeakageTestFailure(
                f"untrained model scores R2={r2:.4f} on pure noise")
        return r2

    def assert_all(self):
        self.check_diagonal()
        self.check_support()
        r2 = self.noise_r2()
        print(f"[leakage] all three conditions satisfied (noise R2={r2:.4f})")
        return True
