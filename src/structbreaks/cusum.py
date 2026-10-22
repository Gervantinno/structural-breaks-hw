"""Cumulative-sum tests.

* CUSUM and CUSUM of squares — Brown, Durbin & Evans (1975), recursive residuals;
* OLS-CUSUM — Ploberger & Krämer (1992), cumulated OLS residuals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ols import as_2d, ols

# Critical constants of the CUSUM boundaries (Brown, Durbin & Evans, 1975)
CUSUM_A = {0.10: 0.850, 0.05: 0.948, 0.01: 1.143}
# Quantiles of sup|Brownian bridge| (Kolmogorov distribution)
KOLMOGOROV = {0.10: 1.224, 0.05: 1.358, 0.01: 1.628}


def recursive_residuals(y, X) -> np.ndarray:
    """Standardised recursive residuals w_t, t = k+1..T.

    w_t = (y_t - x_t' b_{t-1}) / sqrt(1 + x_t' (X_{t-1}'X_{t-1})^{-1} x_t),
    where b_{t-1} is OLS on the first t-1 observations. Uses the Sherman-Morrison
    update, so the whole path costs O(T k^2).
    """
    y = np.asarray(y, float)
    X = as_2d(X)
    T, k = X.shape
    P = np.linalg.inv(X[:k].T @ X[:k])
    b = P @ X[:k].T @ y[:k]
    w = np.empty(T - k)
    for t in range(k, T):
        x = X[t]
        f = 1.0 + x @ P @ x
        e = y[t] - x @ b
        w[t - k] = e / np.sqrt(f)
        Px = P @ x
        P = P - np.outer(Px, Px) / f
        b = b + Px * e / f
    return w


@dataclass
class CusumResult:
    name: str
    path: np.ndarray      # path of the statistic
    lower: np.ndarray
    upper: np.ndarray
    t_index: np.ndarray   # observation positions the path refers to
    alpha: float

    @property
    def reject(self) -> bool:
        return bool(np.any((self.path > self.upper) | (self.path < self.lower)))

    @property
    def first_crossing(self):
        out = np.where((self.path > self.upper) | (self.path < self.lower))[0]
        return int(self.t_index[out[0]]) if len(out) else None

    def __repr__(self) -> str:
        verdict = "H0 (stability) REJECTED" if self.reject else "H0 not rejected"
        fc = "" if self.first_crossing is None else f", first crossing at t={self.first_crossing}"
        return f"{self.name} (alpha={self.alpha}): {verdict}{fc}"


def cusum_test(y, X, alpha: float = 0.05) -> CusumResult:
    """CUSUM: W_r = Σ_{t=k+1}^{r} w_t / σ̂_w; boundaries ±a[√(T-k) + 2(r-k)/√(T-k)]."""
    X = as_2d(X)
    T, k = X.shape
    w = recursive_residuals(y, X)
    W = np.cumsum(w) / np.std(w, ddof=1)
    n = T - k
    r = np.arange(1, n + 1)
    a = CUSUM_A[alpha]
    bound = a * (np.sqrt(n) + 2 * r / np.sqrt(n))
    return CusumResult("CUSUM (Brown-Durbin-Evans)", W, -bound, bound, np.arange(k, T), alpha)


def cusumsq_test(y, X, alpha: float = 0.05) -> CusumResult:
    """CUSUM of squares: S_r = Σ w_t² / Σ w² with boundaries (r-k)/(T-k) ± c0.

    c0 uses the asymptotic approximation S_r - E(S_r) ≈ √(2/n)·B(r/n), where B is
    a Brownian bridge, so c0 = K_α·√(2/n) (K_α is the Kolmogorov quantile).
    """
    X = as_2d(X)
    T, k = X.shape
    w = recursive_residuals(y, X)
    S = np.cumsum(w ** 2) / np.sum(w ** 2)
    n = T - k
    E = np.arange(1, n + 1) / n
    c0 = KOLMOGOROV[alpha] * np.sqrt(2.0 / n)
    return CusumResult("CUSUM of squares", S, E - c0, E + c0, np.arange(k, T), alpha)


def ols_cusum_test(y, X, alpha: float = 0.05) -> CusumResult:
    """OLS-CUSUM (Ploberger & Krämer, 1992): Σ_{t≤r} e_t / (σ̂ √T) ⇒ Brownian bridge."""
    X = as_2d(X)
    T = X.shape[0]
    res = ols(y, X)
    path = np.cumsum(res.resid) / (np.sqrt(res.ssr / T) * np.sqrt(T))
    c = KOLMOGOROV[alpha]
    return CusumResult("OLS-CUSUM (Ploberger-Krämer)", path, -c * np.ones(T), c * np.ones(T),
                       np.arange(T), alpha)
