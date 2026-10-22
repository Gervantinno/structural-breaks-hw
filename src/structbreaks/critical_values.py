"""Critical values of the Bai-Perron tests.

The asymptotic distributions of sup-F(k), UDmax, WDmax and sup-F(l+1|l) have no
closed form, so Bai & Perron (1998, 2003) obtained them by simulating Brownian
motions. We do the same: under H0 (no breaks) we generate series
y_t ~ iid N(0,1) of length T_SIM with regressors z_t = (1, ξ_{2t}, ..., ξ_{qt}),
ξ ~ N(0,1), and apply exactly the same dynamic-programming procedure that is
applied to real data.

Results are cached in ``structbreaks/data/critical_values.json``;
recompute with ``python scripts/simulate_critical_values.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

T_SIM = 500
CACHE = Path(__file__).resolve().parent / "data" / "critical_values.json"

# Maximum number of breaks for which Bai & Perron (2003) tabulate critical values
BP_MAX_BREAKS = {0.05: 9, 0.10: 8, 0.15: 5, 0.20: 3, 0.25: 2}

# Published 5% values of Bai & Perron (1998, Table I; 2003), q = 1, ε = 0.15, M = 5 —
# used to check our own simulation.
BP2003_PUBLISHED_5PCT = {
    1: {"supF": [8.58, 7.22, 5.96, 4.99, 3.91], "UDmax": 8.88, "WDmax": 9.91,
        "seq": [8.58, 10.13, 11.14, 11.83, 12.25]},
}


def default_max_breaks(trim: float, T: int = T_SIM) -> int:
    h = int(np.floor(trim * T))
    return int(min(BP_MAX_BREAKS.get(round(trim, 2), 5), T // h - 1))


def simulate(q: int, trim: float, M: int | None = None, nsim: int = 2000, T: int = T_SIM,
             seed: int = 2026) -> dict:
    """Monte Carlo draws of sup-F(k), k = 1..M, under H0."""
    from .bai_perron import dynamic_programming, ssr_matrix  # local import avoids a circular import

    rng = np.random.default_rng(seed)
    h = int(np.floor(trim * T))
    M = default_max_breaks(trim, T) if M is None else int(min(M, T // h - 1))
    supF = np.zeros((nsim, M))
    for r in range(nsim):
        X = np.column_stack([np.ones(T)] + [rng.standard_normal(T) for _ in range(q - 1)])
        y = rng.standard_normal(T)
        ssr, _ = dynamic_programming(ssr_matrix(y, X, h), M, h)
        for k in range(1, M + 1):
            supF[r, k - 1] = ((ssr[0] - ssr[k]) / k) / (ssr[k] / (T - (k + 1) * q))
    return {"supF": supF, "M": M}


def critical_values_from_sim(sim: dict, alpha: float = 0.05) -> dict:
    """sup-F(k), UDmax(M), WDmax(M) for every M ≤ Mmax, and sup-F(l+1|l)."""
    supF = sim["supF"]
    Mmax = sim["M"]
    c_sup = {k: float(np.quantile(supF[:, k - 1], 1 - alpha)) for k in range(1, Mmax + 1)}
    ud, wd = {}, {}
    for M in range(1, Mmax + 1):
        ud[M] = float(np.quantile(supF[:, :M].max(axis=1), 1 - alpha))
        w = np.array([c_sup[1] / c_sup[k] for k in range(1, M + 1)])
        wd[M] = float(np.quantile((supF[:, :M] * w).max(axis=1), 1 - alpha))
    # sup-F(l+1|l): asymptotically the max of (l+1) independent sup-F(1) ⇒ quantile (1-α)^{1/(l+1)}
    seq = {l: float(np.quantile(supF[:, 0], (1 - alpha) ** (1 / (l + 1)))) for l in range(Mmax)}
    return {"supF": c_sup, "UDmax": ud, "WDmax": wd, "seq": seq}


def _key(q, trim, alpha):
    return f"q={q}|trim={trim:.2f}|alpha={alpha:.2f}"


def _to_json(cv: dict, nsim: int) -> dict:
    return {name: ({str(k): round(v, 3) for k, v in cv[name].items()})
            for name in ("supF", "UDmax", "WDmax", "seq")} | {"nsim": nsim, "T": T_SIM}


def bp_critical_values(q: int, trim: float, M: int, alpha: float = 0.05,
                       nsim: int = 1000) -> dict:
    """Critical values from the cache; if missing, simulates (nsim replications) and stores them."""
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    key = _key(q, trim, alpha)
    if key not in cache:
        print(f"[critical_values] {key} not cached: simulating ({nsim} replications)...")
        sim = simulate(q, trim, nsim=nsim)
        cache[key] = _to_json(critical_values_from_sim(sim, alpha), nsim)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    c = cache[key]
    Mmax = len(c["supF"])
    M = min(M, Mmax)
    return {"supF": {int(k): v for k, v in c["supF"].items() if int(k) <= M},
            "UDmax": c["UDmax"][str(M)], "WDmax": c["WDmax"][str(M)],
            "seq": {int(k): v for k, v in c["seq"].items() if int(k) < M}, "M": M}
