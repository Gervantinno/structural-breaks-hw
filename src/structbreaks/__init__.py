"""structbreaks — structural break tests for time series (from scratch, numpy/scipy).

Implementation of the methods reviewed in
Muthuramu & Uma Maheswari (2019) "Tests for Structural Breaks in Time Series
Analysis: A Review of Recent Development", Shanlax Int. J. of Economics 7(4).
"""
from .ols import ols, hac_cov
from .single_break import (chow_test, chow_predictive, chow_dummy_wald, sup_f_test,
                           andrews_critical_values, split_ssr)
from .cusum import recursive_residuals, cusum_test, cusumsq_test, ols_cusum_test
from .bai_perron import (bai_perron, ssr_matrix, dynamic_programming, regime_estimates,
                         break_confidence_intervals)

__all__ = ["ols", "hac_cov", "chow_test", "chow_predictive", "chow_dummy_wald", "sup_f_test",
           "andrews_critical_values", "split_ssr", "recursive_residuals", "cusum_test",
           "cusumsq_test", "ols_cusum_test", "bai_perron", "ssr_matrix", "dynamic_programming",
           "regime_estimates", "break_confidence_intervals"]
__version__ = "1.0.0"
