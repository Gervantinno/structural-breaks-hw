"""Multiple structural breaks at unknown dates: Bai & Perron (1998, 2003).

Pure structural change model (all coefficients change, p = 0):

    y_t = z_t' δ_j + u_t,   t = T_{j-1}+1, ..., T_j,   j = 1, ..., m+1.

Implemented:
  1. SSR matrix of all admissible segments (length ≥ h);
  2. dynamic programming — the global SSR minimum for m = 0..M breaks
     (this is exactly the "triangular matrix" of Figure 1 in the paper);
  3. sup-F(k), UDmax, WDmax tests (H0: no breaks vs k or 1..M breaks);
  4. sequential test sup-F(l+1 | l);
  5. information criteria BIC (Yao, 1988) and LWZ (Liu, Wu & Zidek, 1997);
  6. confidence intervals for the break dates (Bai, 1997) and regime coefficients.

All F statistics are on the "scaled" scale (Wald / k = q·F), as in the
Bai & Perron tables and in EViews. Critical values come from a simulation
(``critical_values.py``) that reproduces the Bai & Perron (2003) tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .critical_values import bp_critical_values
from .ols import as_2d, long_run_cov, newey_west_bandwidth

# Quantiles of the argmax of a two-sided Brownian motion with drift
# (Bai, 1997): used for confidence intervals of the break dates.
ARGMAX_QUANTILES = {0.90: 7.687, 0.95: 11.033, 0.99: 19.0}


# ---------------------------------------------------------------------------
#  1. SSR matrix
# ---------------------------------------------------------------------------
def ssr_matrix(y, X, h: int) -> np.ndarray:
    """S[i, j] = SSR of regressing y on X over segment [i, j) (j - i ≥ h), else +inf."""
    y = np.asarray(y, float)
    X = as_2d(X)
    T, q = X.shape
    S = np.full((T, T + 1), np.inf)
    XX = np.einsum("ti,tj->tij", X, X)
    Xy = X * y[:, None]
    yy = y * y
    for i in range(0, T - h + 1):
        cXX = np.cumsum(XX[i:], axis=0)[h - 1:]
        cXy = np.cumsum(Xy[i:], axis=0)[h - 1:]
        cyy = np.cumsum(yy[i:])[h - 1:]
        if q == 1:
            ssr = cyy - cXy[:, 0] ** 2 / cXX[:, 0, 0]
        else:
            try:
                b = np.linalg.solve(cXX, cXy[..., None])[..., 0]
            except np.linalg.LinAlgError:
                b = np.array([np.linalg.lstsq(a, c, rcond=None)[0] for a, c in zip(cXX, cXy)])
            ssr = cyy - np.einsum("ni,ni->n", cXy, b)
        S[i, i + h:] = np.maximum(ssr, 0.0)
    return S


# ---------------------------------------------------------------------------
#  2. Dynamic programming
# ---------------------------------------------------------------------------
def dynamic_programming(S: np.ndarray, M: int, h: int):
    """Global SSR minimisation for m = 0..M breaks.

    Returns (ssr[m], ends[m]), where ends[m] lists the right (exclusive) boundaries
    of the first m segments, i.e. "the number of observations before each break".
    """
    T = S.shape[0]
    cost = np.full((M + 1, T + 1), np.inf)
    arg = np.zeros((M + 1, T + 1), dtype=int)
    cost[0] = S[0]
    for m in range(1, M + 1):
        for j in range((m + 1) * h, T + 1):
            i = np.arange(m * h, j - h + 1)
            vals = cost[m - 1, i] + S[i, j]
            k = int(np.argmin(vals))
            cost[m, j], arg[m, j] = vals[k], i[k]
    ssr, ends = [], []
    for m in range(M + 1):
        ssr.append(float(cost[m, T]))
        e, j = [], T
        for mm in range(m, 0, -1):
            j = arg[mm, j]
            e.append(int(j))
        ends.append(sorted(e))
    return ssr, ends


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def segment_design(X, ends):
    """Block-diagonal matrix Z̄ = diag(Z_1, ..., Z_{m+1})."""
    X = as_2d(X)
    T, q = X.shape
    bounds = [0] + list(ends) + [T]
    Z = np.zeros((T, q * (len(bounds) - 1)))
    for j in range(len(bounds) - 1):
        Z[bounds[j]:bounds[j + 1], j * q:(j + 1) * q] = X[bounds[j]:bounds[j + 1]]
    return Z


def _segment_fit(y, X, ends, cov="ols", lags=None):
    """Segment-wise estimates δ_j and their covariance (OLS / HAC with heterogeneity)."""
    y = np.asarray(y, float)
    X = as_2d(X)
    T, q = X.shape
    Z = segment_design(X, ends)
    delta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    u = y - Z @ delta
    ZZ_inv = np.linalg.inv(Z.T @ Z)
    if cov == "ols":
        V = ZZ_inv * (u @ u) / (T - Z.shape[1])
    else:
        # HAC: the long-run covariance is estimated within each segment, which
        # allows different variance/autocorrelation across regimes (BP 2003).
        bounds = [0] + list(ends) + [T]
        Omega = np.zeros((Z.shape[1], Z.shape[1]))
        for j in range(len(bounds) - 1):
            a, b = bounds[j], bounds[j + 1]
            sc = X[a:b] * u[a:b, None]
            L = lags if lags is not None else newey_west_bandwidth(b - a)
            Omega[j * q:(j + 1) * q, j * q:(j + 1) * q] = long_run_cov(sc, L) * (b - a)
        V = ZZ_inv @ Omega @ ZZ_inv
    return delta, V, u


def _diff_matrix(k: int, q: int) -> np.ndarray:
    """R such that (Rδ)' = (δ_1' - δ_2', ..., δ_k' - δ_{k+1}')."""
    R = np.zeros((k * q, (k + 1) * q))
    for i in range(k):
        R[i * q:(i + 1) * q, i * q:(i + 1) * q] = np.eye(q)
        R[i * q:(i + 1) * q, (i + 1) * q:(i + 2) * q] = -np.eye(q)
    return R


def wald_f(y, X, ends, cov="ols", lags=None) -> float:
    """Scaled F = (T-(k+1)q)/T · δ'R'(RVR')⁻¹Rδ / k  (Bai-Perron, 2003, eq. (6))."""
    X = as_2d(X)
    T, q = X.shape
    k = len(ends)
    delta, V, _ = _segment_fit(y, X, ends, cov, lags)
    R = _diff_matrix(k, q)
    d = R @ delta
    W = float(d @ np.linalg.solve(R @ V @ R.T, d))
    if cov == "ols":
        return W / k
    return (T - (k + 1) * q) / T * W / k


# ---------------------------------------------------------------------------
#  Result
# ---------------------------------------------------------------------------
@dataclass
class BaiPerronResult:
    T: int
    q: int
    h: int
    trim: float
    M: int
    cov: str
    ssr: list
    ends: list
    supF: dict
    UDmax: float
    WDmax: float
    seq: dict
    bic: np.ndarray
    lwz: np.ndarray
    crit: dict
    index: object = None
    selected: dict = field(default_factory=dict)

    # --- convenience views ---------------------------------------------------
    def break_dates(self, m: int):
        """Break dates = the LAST observation of each regime (as in strucchange)."""
        idx = [e - 1 for e in self.ends[m]]
        if self.index is None:
            return idx
        return [self.index[i] for i in idx]

    def summary(self) -> str:
        c = self.crit
        lines = [f"Bai-Perron: T={self.T}, q={self.q}, h={self.h} (trim={self.trim}), M={self.M}, cov={self.cov}",
                 "",
                 " k |  sup-F(k) | 5% crit. | break date(s) (global SSR minimum)",
                 "---+-----------+----------+-----------------------------------"]
        for k in range(1, self.M + 1):
            star = "*" if self.supF[k] > c["supF"][k] else " "
            lines.append(f"{k:2d} | {self.supF[k]:8.2f}{star} | {c['supF'][k]:8.2f} | {self.break_dates(k)}")
        lines += ["",
                  f"UDmax = {self.UDmax:7.2f}  (5% crit. {c['UDmax']:.2f})  {'*' if self.UDmax > c['UDmax'] else ''}",
                  f"WDmax = {self.WDmax:7.2f}  (5% crit. {c['WDmax']:.2f})  {'*' if self.WDmax > c['WDmax'] else ''}",
                  "", "Sequential test sup-F(l+1|l):"]
        for l, v in self.seq.items():
            star = "*" if v > c["seq"][l] else " "
            lines.append(f"  F({l + 1}|{l}) = {v:8.2f}{star} (5% crit. {c['seq'][l]:.2f})")
        lines += ["", " m |    SSR      |   BIC    |   LWZ"]
        for m in range(self.M + 1):
            lines.append(f"{m:2d} | {self.ssr[m]:11.3f} | {self.bic[m]:8.4f} | {self.lwz[m]:8.4f}")
        lines += ["", "Selected number of breaks: " +
                  ", ".join(f"{k}={v}" for k, v in self.selected.items()),
                  "(* — significant at 5%)"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Main function
# ---------------------------------------------------------------------------
def bai_perron(y, X=None, trim: float = 0.15, M: int = 5, cov: str = "ols",
               lags=None, index=None, alpha: float = 0.05) -> BaiPerronResult:
    """Full Bai-Perron procedure.

    y     : series (T,)
    X     : regressors z_t (default — a constant ⇒ mean-shift model)
    trim  : ε = h/T — minimum fraction of observations in a segment
    M     : maximum number of breaks (capped so that (M+1)·h ≤ T)
    cov   : "ols" (homogeneous errors) or "hac" (heteroskedasticity and autocorrelation,
            different error distributions across regimes)
    """
    y = np.asarray(y, float)
    T = len(y)
    X = np.ones((T, 1)) if X is None else as_2d(X)
    q = X.shape[1]
    h = max(int(np.floor(trim * T)), q + 1)
    M = int(min(M, T // h - 1))
    crit = bp_critical_values(q, trim, M, alpha)
    M = crit["M"]                      # no more than we have critical values for
    S = ssr_matrix(y, X, h)
    ssr, ends = dynamic_programming(S, M, h)

    # sup-F(k): the dates are the global SSR minimisers
    if cov == "ols":
        supF = {k: ((ssr[0] - ssr[k]) / k) / (ssr[k] / (T - (k + 1) * q)) for k in range(1, M + 1)}
    else:
        supF = {k: wald_f(y, X, ends[k], cov, lags) for k in range(1, M + 1)}

    UDmax = max(supF.values())
    w = {k: crit["supF"][1] / crit["supF"][k] for k in range(1, M + 1)}
    WDmax = max(w[k] * supF[k] for k in range(1, M + 1))

    # sequential test sup-F(l+1 | l)
    seq = {}
    for l in range(0, M):
        bounds = [0] + ends[l] + [T]
        sigma2 = ssr[l] / (T - (l + 1) * q)
        best = 0.0
        for j in range(len(bounds) - 1):
            a, b = bounds[j], bounds[j + 1]
            if b - a < 2 * h:
                continue
            taus = np.arange(a + h, b - h + 1)
            split = S[a, taus] + S[taus, b]
            gain = S[a, b] - split.min()
            best = max(best, gain / sigma2)
        seq[l] = best

    # information criteria
    ms = np.arange(M + 1)
    pstar = (ms + 1) * q + ms
    ssr_arr = np.array(ssr)
    bic = np.log(ssr_arr / T) + pstar * np.log(T) / T
    lwz = np.log(ssr_arr / (T - pstar)) + pstar * 0.299 * np.log(T) ** 2.1 / T

    # choice of the number of breaks
    m_seq = 0
    for l in range(M):
        if seq[l] > crit["seq"][l]:
            m_seq = l + 1
        else:
            break
    selected = {"sequential": m_seq, "BIC": int(np.argmin(bic)), "LWZ": int(np.argmin(lwz)),
                "UDmax/WDmax": ("breaks present" if max(UDmax > crit["UDmax"], WDmax > crit["WDmax"])
                                else "no breaks")}
    return BaiPerronResult(T, q, h, trim, M, cov, ssr, ends, supF, UDmax, WDmax, seq,
                           bic, lwz, crit, index, selected)


def regime_estimates(y, X, ends, cov="ols", lags=None):
    """Regime-specific coefficients and their standard errors."""
    X = np.ones((len(y), 1)) if X is None else as_2d(X)
    q = X.shape[1]
    delta, V, u = _segment_fit(y, X, ends, cov, lags)
    se = np.sqrt(np.diag(V))
    return delta.reshape(-1, q), se.reshape(-1, q), u


def break_confidence_intervals(y, X, ends, level: float = 0.95):
    """Confidence intervals for break dates (Bai, 1997), homogeneous errors in adjacent regimes.

    T̂_i ± c_level · σ̂² / (Δ_i' Q_i Δ_i),  Δ_i = δ_{i+1} - δ_i,
    Q_i is the average z z' over the two adjacent regimes. Returns [(lo, hi)] as indices
    of the last observation of the first regime (same convention as break_dates).
    """
    y = np.asarray(y, float)
    T = len(y)
    X = np.ones((T, 1)) if X is None else as_2d(X)
    q = X.shape[1]
    delta, _, u = _segment_fit(y, X, ends)
    delta = delta.reshape(-1, q)
    bounds = [0] + list(ends) + [T]
    c = ARGMAX_QUANTILES[level]
    out = []
    for i, e in enumerate(ends):
        a, b = bounds[i], bounds[i + 2]
        sigma2 = float(u[a:b] @ u[a:b]) / (b - a)
        Q = X[a:b].T @ X[a:b] / (b - a)
        D = delta[i + 1] - delta[i]
        width = int(np.ceil(c * sigma2 / float(D @ Q @ D)))
        out.append((max(e - 1 - width, 0), min(e - 1 + width, T - 1)))
    return out
