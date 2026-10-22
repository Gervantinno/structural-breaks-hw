"""Tests for a SINGLE structural break.

* Chow (1960)                    — break at a KNOWN date (F-test and the predictive version);
* Quandt (1960) / Andrews (1993) — break at an UNKNOWN date: sup-F (QLR);
* Andrews & Ploberger (1994)     — optimal exp-W and ave-W statistics.

Asymptotic p-values of the sup/exp/ave statistics are obtained by simulating
functionals of a Brownian bridge (see ``simulate_andrews_distribution``) — the
same approach Andrews (1993) used to tabulate his critical values.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .ols import as_2d, hac_cov, ols


# ---------------------------------------------------------------------------
#  Helper: SSR of all prefixes and suffixes in O(T k^3)
# ---------------------------------------------------------------------------
def _cum_moments(y, X):
    XX = np.einsum("ti,tj->tij", X, X)
    Xy = X * y[:, None]
    yy = y * y
    pad = lambda a: np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])
    return pad(XX), pad(Xy), pad(yy)


def _ssr_from_moments(XX, Xy, yy):
    """SSR = y'y - (X'y)'(X'X)^{-1}(X'y), vectorised over the first axis."""
    beta = np.linalg.solve(XX, Xy[..., None])[..., 0]
    return yy - np.einsum("...i,...i->...", Xy, beta)


def split_ssr(y, X, candidates):
    """SSR(left part) + SSR(right part) for every candidate b.

    b is the number of observations in the first regime (regime 1 = [0, b), regime 2 = [b, T)).
    """
    y = np.asarray(y, float)
    X = as_2d(X)
    cXX, cXy, cyy = _cum_moments(y, X)
    b = np.asarray(candidates)
    left = _ssr_from_moments(cXX[b], cXy[b], cyy[b])
    right = _ssr_from_moments(cXX[-1] - cXX[b], cXy[-1] - cXy[b], cyy[-1] - cyy[b])
    return left + right


# ---------------------------------------------------------------------------
#  Chow (1960)
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    name: str
    statistic: float
    pvalue: float
    df: tuple = ()
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # compact notebook output
        df = f", df={self.df}" if self.df else ""
        return f"{self.name}: stat={self.statistic:.3f}, p-value={self.pvalue:.4f}{df}"


def chow_test(y, X, n1: int) -> TestResult:
    """Classical Chow test. H0: the coefficients are equal before and after date n1.

    n1 is the number of observations in the first subsample (n1 > k and T - n1 > k).
        F = [(SSR_R - SSR_1 - SSR_2)/k] / [(SSR_1 + SSR_2)/(T - 2k)]  ~  F(k, T-2k)
    """
    y = np.asarray(y, float)
    X = as_2d(X)
    T, k = X.shape
    if n1 <= k or T - n1 <= k:
        raise ValueError("Each subsample needs more than k observations; use chow_predictive instead")
    ssr_r = ols(y, X).ssr
    ssr_u = ols(y[:n1], X[:n1]).ssr + ols(y[n1:], X[n1:]).ssr
    F = ((ssr_r - ssr_u) / k) / (ssr_u / (T - 2 * k))
    return TestResult("Chow F-test", F, float(stats.f.sf(F, k, T - 2 * k)), (k, T - 2 * k))


def chow_predictive(y, X, n1: int) -> TestResult:
    """Chow predictive test (the second case in the paper: n2 < k).

        F = [(SSR_R - SSR_1)/n2] / [SSR_1/(n1 - k)]  ~  F(n2, n1-k)
    """
    y = np.asarray(y, float)
    X = as_2d(X)
    T, k = X.shape
    n2 = T - n1
    ssr_r = ols(y, X).ssr
    ssr_1 = ols(y[:n1], X[:n1]).ssr
    F = ((ssr_r - ssr_1) / n2) / (ssr_1 / (n1 - k))
    return TestResult("Chow predictive test", F, float(stats.f.sf(F, n2, n1 - k)), (n2, n1 - k))


def chow_dummy_wald(y, X, n1: int, lags: int | None = None) -> TestResult:
    """Chow test in dummy-variable form with a HAC covariance (robust to
    heteroskedasticity and autocorrelation): y = X b + D*X d + e, H0: d = 0.
    The reported statistic is Wald / k (F scale)."""
    y = np.asarray(y, float)
    X = as_2d(X)
    T, k = X.shape
    D = (np.arange(T) >= n1).astype(float)[:, None]
    Z = np.hstack([X, X * D])
    res = ols(y, Z)
    V = hac_cov(Z, res.resid, lags)
    d = res.beta[k:]
    W = float(d @ np.linalg.solve(V[k:, k:], d))
    return TestResult("Chow Wald (HAC)", W / k, float(stats.chi2.sf(W, k)), (k,))


# ---------------------------------------------------------------------------
#  Asymptotics of Andrews (1993) / Andrews-Ploberger (1994)
# ---------------------------------------------------------------------------
_SIM_CACHE: dict = {}


def simulate_andrews_distribution(q: int, trim: float = 0.15, nsim: int = 20000,
                                  grid: int = 1000, seed: int = 12345) -> dict:
    """Simulate the limiting distributions of sup-W, exp-W and ave-W.

    Under H0:  W_T(λ) ⇒ ||B_q(λ)||² / (λ(1-λ)),  where B_q is a q-dimensional Brownian bridge.
    Returns a dict of sorted draws for each functional.
    """
    key = (q, round(trim, 4), nsim, grid, seed)
    if key in _SIM_CACHE:
        return _SIM_CACHE[key]
    rng = np.random.default_rng(seed)
    lam = np.arange(1, grid) / grid
    mask = (lam >= trim) & (lam <= 1 - trim)
    lam_m = lam[mask]
    sup, expf, ave = [], [], []
    chunk = 2000
    for start in range(0, nsim, chunk):
        n = min(chunk, nsim - start)
        Fsum = np.zeros((n, mask.sum()))
        for _ in range(q):
            W = np.cumsum(rng.standard_normal((n, grid)), axis=1) / np.sqrt(grid)
            B = W[:, :-1] - lam[None, :] * W[:, -1:]
            Fsum += B[:, mask] ** 2
        F = Fsum / (lam_m * (1 - lam_m))
        sup.append(F.max(axis=1))
        m = F.max(axis=1, keepdims=True)  # numerically stable log-mean-exp
        expf.append((m[:, 0] / 2) + np.log(np.mean(np.exp(F / 2 - m / 2), axis=1)))
        ave.append(F.mean(axis=1))
    out = {"sup": np.sort(np.concatenate(sup)), "exp": np.sort(np.concatenate(expf)),
           "ave": np.sort(np.concatenate(ave))}
    _SIM_CACHE[key] = out
    return out


def _sim_pvalue(sample_sorted: np.ndarray, stat: float) -> float:
    n = len(sample_sorted)
    return float((n - np.searchsorted(sample_sorted, stat, side="left") + 1) / (n + 1))


def andrews_critical_values(q: int, trim: float = 0.15, levels=(0.10, 0.05, 0.01)) -> dict:
    """Critical values on the Wald scale (q·F), as tabulated by Andrews (1993)."""
    sim = simulate_andrews_distribution(q, trim)
    return {kind: {a: float(np.quantile(sim[kind], 1 - a)) for a in levels} for kind in sim}


@dataclass
class SupFResult:
    candidates: np.ndarray       # b = number of observations in the first regime
    F: np.ndarray                # sequence F(b) (Wald/k)
    sup_f: float
    exp_f: float
    ave_f: float
    argmax: int                  # estimated b
    pvalues: dict
    q: int
    trim: float
    cov: str

    @property
    def break_index(self) -> int:
        """0-based index of the LAST observation of the first regime."""
        return int(self.argmax - 1)

    def summary(self, index=None) -> str:
        date = index[self.break_index] if index is not None else self.break_index
        return (f"Quandt-Andrews (trim={self.trim:.2f}, cov={self.cov}, q={self.q}):\n"
                f"  sup-F = {self.sup_f:8.3f}  p = {self.pvalues['sup']:.4f}\n"
                f"  exp-W = {self.exp_f:8.3f}  p = {self.pvalues['exp']:.4f}   (Andrews-Ploberger)\n"
                f"  ave-W = {self.ave_f:8.3f}  p = {self.pvalues['ave']:.4f}\n"
                f"  estimated break date (last observation of regime 1): {date}")


def sup_f_test(y, X, trim: float = 0.15, cov: str = "ols", lags: int | None = None) -> SupFResult:
    """Quandt-Andrews test for one break at an unknown date.

    cov="ols" — classical F statistic (a Chow test at every date);
    cov="hac" — Wald statistic with a HAC (Newey-West) covariance, divided by the number of restrictions.
    Returns sup-F (QLR), exp-W and ave-W with asymptotic p-values.
    """
    y = np.asarray(y, float)
    X = as_2d(X)
    T, k = X.shape
    lo, hi = int(np.floor(trim * T)), int(np.ceil((1 - trim) * T))
    cand = np.arange(max(lo, k + 1), min(hi, T - k - 1) + 1)
    if cov == "ols":
        ssr_r = ols(y, X).ssr
        ssr_u = split_ssr(y, X, cand)
        F = ((ssr_r - ssr_u) / k) / (ssr_u / (T - 2 * k))
    elif cov == "hac":
        F = np.array([chow_dummy_wald(y, X, int(b), lags).statistic for b in cand])
    else:
        raise ValueError("cov must be 'ols' or 'hac'")
    m = F.max()
    # exp/ave are defined (as in Andrews-Ploberger) on the Wald scale: W = k * F
    W = k * F
    Wm = W.max()
    exp_f = float(Wm / 2 + np.log(np.mean(np.exp((W - Wm) / 2))))
    ave_f = float(W.mean())
    sim = simulate_andrews_distribution(k, trim)
    pv = {"sup": _sim_pvalue(sim["sup"], Wm),
          "exp": _sim_pvalue(sim["exp"], exp_f),
          "ave": _sim_pvalue(sim["ave"], ave_f)}
    return SupFResult(cand, F, float(m), exp_f, ave_f, int(cand[F.argmax()]), pv, k, trim, cov)
