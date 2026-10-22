"""Monte Carlo critical values for the Bai-Perron tests.

Run (from the repository root):
    python scripts/simulate_critical_values.py --nsim 6000   (≈15 min on 2 cores)

The result is written to src/structbreaks/data/critical_values.json and then
used by ``bai_perron``. The script also prints a comparison with the published
Bai & Perron (2003) tables for ε = 0.15.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from structbreaks import critical_values as cv  # noqa: E402

TRIMS = [0.05, 0.10, 0.15, 0.20, 0.25]
QS = [1, 2]
ALPHAS = [0.10, 0.05, 0.01]


def job(args):
    q, trim, nsim = args
    sim = cv.simulate(q, trim, nsim=nsim, seed=1000 + 17 * q + int(trim * 100))
    return q, trim, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsim", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    tasks = [(q, t, a.nsim if q == 1 else a.nsim // 2) for q in QS for t in TRIMS]
    with Pool(a.workers) as pool:
        results = pool.map(job, tasks)

    cache = {}
    for q, trim, sim in results:
        for alpha in ALPHAS:
            c = cv.critical_values_from_sim(sim, alpha)
            cache[cv._key(q, trim, alpha)] = cv._to_json(c, int(sim["supF"].shape[0]))
    cv.CACHE.parent.mkdir(parents=True, exist_ok=True)
    cv.CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {cv.CACHE}")

    print("\nComparison with Bai & Perron (2003), ε = 0.15, α = 5%:")
    for q in (1,):
        mine = cache[cv._key(q, 0.15, 0.05)]
        pub = cv.BP2003_PUBLISHED_5PCT[q]
        print(f"q={q}")
        print("  supF(k) simulated :", [mine["supF"][str(k)] for k in range(1, 6)])
        print("  supF(k) BP (2003) :", pub["supF"])
        print(f"  UDmax {mine['UDmax']['5']} vs {pub['UDmax']};  WDmax {mine['WDmax']['5']} vs {pub['WDmax']}")
        print("  seq  simulated    :", [mine["seq"][str(k)] for k in range(5)])
        print("  seq  BP (2003)    :", pub["seq"])


if __name__ == "__main__":
    main()
