"""Correctness checks for the implementation.

Run:  python -m unittest discover -s tests -v      (pytest works too)
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from structbreaks import (bai_perron, chow_test, cusum_test, dynamic_programming,  # noqa: E402
                          ols, recursive_residuals, ssr_matrix, sup_f_test)
from structbreaks.single_break import andrews_critical_values  # noqa: E402


class TestOLSAndChow(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.T = 200
        self.X = np.column_stack([np.ones(self.T), rng.standard_normal(self.T)])
        self.y = self.X @ np.array([1.0, 2.0]) + rng.standard_normal(self.T)

    def test_ols_matches_normal_equations(self):
        b = np.linalg.solve(self.X.T @ self.X, self.X.T @ self.y)
        np.testing.assert_allclose(ols(self.y, self.X).beta, b, rtol=1e-10)

    def test_chow_equals_dummy_regression_f(self):
        """Chow F = F-test on dummy interactions (the equivalence noted in the Discussion section)."""
        n1 = 90
        D = (np.arange(self.T) >= n1).astype(float)[:, None]
        Z = np.hstack([self.X, self.X * D])
        ssr_u, ssr_r = ols(self.y, Z).ssr, ols(self.y, self.X).ssr
        F = ((ssr_r - ssr_u) / 2) / (ssr_u / (self.T - 4))
        self.assertAlmostEqual(chow_test(self.y, self.X, n1).statistic, F, places=8)

    def test_recursive_residuals_brute_force(self):
        w = recursive_residuals(self.y, self.X)
        k = 2
        for t in [k, 10, 50, self.T - 1]:
            Xp, yp = self.X[:t], self.y[:t]
            b = np.linalg.lstsq(Xp, yp, rcond=None)[0]
            f = 1 + self.X[t] @ np.linalg.inv(Xp.T @ Xp) @ self.X[t]
            self.assertAlmostEqual(w[t - k], (self.y[t] - self.X[t] @ b) / np.sqrt(f), places=8)


class TestDynamicProgramming(unittest.TestCase):
    def test_dp_equals_brute_force(self):
        rng = np.random.default_rng(1)
        T, h = 60, 8
        y = np.r_[rng.normal(0, 1, 20), rng.normal(2, 1, 25), rng.normal(-1, 1, 15)]
        X = np.column_stack([np.ones(T), rng.standard_normal(T)])
        S = ssr_matrix(y, X, h)
        ssr, ends = dynamic_programming(S, 2, h)
        best = min((S[0, a] + S[a, b] + S[b, T], (a, b))
                   for a in range(h, T) for b in range(a + h, T - h + 1))
        self.assertAlmostEqual(ssr[2], best[0], places=8)
        self.assertEqual(tuple(ends[2]), best[1])

    def test_ssr_matrix_entries(self):
        rng = np.random.default_rng(2)
        y = rng.standard_normal(40)
        X = np.column_stack([np.ones(40), np.arange(40.0)])
        S = ssr_matrix(y, X, 5)
        self.assertAlmostEqual(S[3, 20], ols(y[3:20], X[3:20]).ssr, places=8)
        self.assertTrue(np.isinf(S[3, 6]))


class TestKnownResults(unittest.TestCase):
    """Benchmark: annual Nile flow 1871-1970 — the classic example (Aswan dam, 1898)."""

    @classmethod
    def setUpClass(cls):
        d = pd.read_csv(ROOT / "data" / "raw" / "nile.csv")
        cls.y, cls.years = d["value"].values, d["time"].values

    def test_nile_supf_and_date(self):
        r = sup_f_test(self.y, np.ones(len(self.y)))
        self.assertAlmostEqual(r.sup_f, 75.93, places=2)     # matches R strucchange::Fstats
        self.assertEqual(self.years[r.break_index], 1898)

    def test_nile_bai_perron_one_break(self):
        res = bai_perron(self.y, trim=0.15, M=5)
        self.assertEqual(res.selected["sequential"], 1)
        self.assertEqual(res.break_dates(1), [27])            # 28th observation = 1898

    def test_cusum_rejects_on_nile(self):
        self.assertTrue(cusum_test(self.y, np.ones(len(self.y))).reject)


class TestCriticalValues(unittest.TestCase):
    def test_andrews_table(self):
        """Andrews (1993), Table 1, p=1, π0=0.15: 5% = 8.68, 1% = 12.16."""
        cv = andrews_critical_values(1, 0.15)["sup"]
        self.assertAlmostEqual(cv[0.05], 8.68, delta=0.35)
        self.assertAlmostEqual(cv[0.01], 12.16, delta=0.8)


class TestSizeUnderNull(unittest.TestCase):
    def test_supf_size_close_to_nominal(self):
        rng = np.random.default_rng(3)
        rej = 0
        n = 300
        for _ in range(n):
            y = rng.standard_normal(150)
            rej += sup_f_test(y, np.ones(150)).pvalues["sup"] < 0.05
        self.assertLess(abs(rej / n - 0.05), 0.04)


if __name__ == "__main__":
    unittest.main(verbosity=2)
