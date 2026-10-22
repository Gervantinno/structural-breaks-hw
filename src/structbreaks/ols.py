"""Basic tools: OLS and a HAC (Newey-West) covariance matrix.

All functions work with numpy arrays:
    y : (T,)   dependent variable
    X : (T, k) regressor matrix (the user adds the constant)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OLSResult:
    beta: np.ndarray
    resid: np.ndarray
    ssr: float
    nobs: int
    k: int

    @property
    def sigma2(self) -> float:
        """Unbiased estimate of the error variance."""
        return self.ssr / (self.nobs - self.k)


def as_2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def ols(y, X) -> OLSResult:
    """Ordinary least squares via lstsq (robust to poor conditioning)."""
    y = np.asarray(y, dtype=float)
    X = as_2d(X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return OLSResult(beta=beta, resid=resid, ssr=float(resid @ resid),
                     nobs=len(y), k=X.shape[1])


def newey_west_bandwidth(T: int) -> int:
    """Newey-West (1994) rule of thumb: floor(4 * (T/100)^(2/9))."""
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


def long_run_cov(scores: np.ndarray, lags: int | None = None) -> np.ndarray:
    """Long-run covariance of a vector process scores_t (T x k), Bartlett kernel."""
    scores = as_2d(scores)
    T = scores.shape[0]
    if lags is None:
        lags = newey_west_bandwidth(T)
    S = scores.T @ scores / T
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1.0)
        G = scores[j:].T @ scores[:-j] / T
        S += w * (G + G.T)
    return S


def hac_cov(X, resid, lags: int | None = None) -> np.ndarray:
    """HAC (Newey-West) covariance of the OLS coefficients. lags=0 gives White's estimator."""
    X = as_2d(X)
    T = X.shape[0]
    Q_inv = np.linalg.inv(X.T @ X / T)
    S = long_run_cov(X * resid[:, None], lags)
    return Q_inv @ S @ Q_inv / T
