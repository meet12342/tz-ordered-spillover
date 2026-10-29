# ---------------------------------------------------------------------------
# clock_mask.py  -  derivation of closing ranks and construction of the mask
# ---------------------------------------------------------------------------
import numpy as np


class ClockMask:
    """Builds the binary same-day admissibility mask from published hours.

    The ranks are DERIVED by sorting the configured closing times. No rank
    is ever written as a literal anywhere in this file, which is what makes
    the ordering an observed property rather than a modelling choice.
    """

    def __init__(self, close_times):
        # close_times: {code: "HH:MM" in UTC}  or  {code: None} if continuous
        self.close_times = close_times
        self.codes = list(close_times.keys())
        self.ranks = self._derive_ranks()
        self.M = self._build_mask()

    def _derive_ranks(self):
        timed = {c: t for c, t in self.close_times.items() if t is not None}
        distinct = sorted(set(timed.values()))          # <- the derivation
        pos = {t: i + 1 for i, t in enumerate(distinct)}
        return {c: (pos[t] if t is not None else None)
                for c, t in self.close_times.items()}

    def _build_mask(self):
        K = len(self.codes)
        M = np.zeros((K, K), dtype=np.float32)
        for a, ci in enumerate(self.codes):
            for b, cj in enumerate(self.codes):
                ri, rj = self.ranks[ci], self.ranks[cj]
                if ri is None or rj is None:
                    continue                 # continuous: no same-day edge
                if rj < ri:                  # j closed strictly before i
                    M[a, b] = 1.0
        np.fill_diagonal(M, 0.0)             # never one's own value
        return M

    def admitted(self):
        return int(self.M.sum())

    def possible(self):
        K = len(self.codes)
        return K * (K - 1)
