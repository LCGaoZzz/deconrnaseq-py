"""Hard accuracy gates on the exact dataset (use_scale=False vs generator truth)
and cross-solver agreement.  Gates from the port specification:
max_abs_error <= 1e-8, RMSE <= 1e-9, sum-to-1 error <= 1e-10, nonneg tol 1e-12.
"""

from __future__ import annotations

import numpy as np
import pytest

import deconrnaseq as drs
from deconrnaseq import deconvolve_core

from conftest import ALL_SOLVERS


@pytest.mark.parametrize("solver", ALL_SOLVERS)
def test_exact_unscaled_vs_truth(signatures, mixtures_exact, truth, solver):
    est = deconvolve_core(
        signatures.to_numpy(np.float64),
        mixtures_exact.to_numpy(np.float64),
        use_scale=False,
        solver=solver,
        backend="numpy",
    )
    t = truth.to_numpy(np.float64)
    assert np.isfinite(est).all()
    assert est.min() >= -1e-12
    assert np.abs(est.sum(axis=1) - 1.0).max() <= 1e-10
    assert np.abs(est - t).max() <= 1e-8
    assert np.sqrt(((est - t) ** 2).mean()) <= 1e-9


def test_solvers_agree(signatures, mixtures_exact):
    A = signatures.to_numpy(np.float64)
    Y = mixtures_exact.to_numpy(np.float64)
    ref = deconvolve_core(A, Y, use_scale=False, solver="enum")
    for solver in ("enum_fast", "interior", "activeset", "activeset_warm"):
        est = deconvolve_core(A, Y, use_scale=False, solver=solver, backend="numpy")
        assert np.abs(est - ref).max() <= 1e-9, solver


@pytest.mark.skipif(not drs.rust_available(), reason="rust backend not installed")
def test_rust_matches_numpy(signatures, mixtures_noisy):
    A = signatures.to_numpy(np.float64)
    Y = mixtures_noisy.to_numpy(np.float64)
    a = deconvolve_core(A, Y, use_scale=False, solver="interior", backend="numpy")
    b = deconvolve_core(A, Y, use_scale=False, solver="interior", backend="rust")
    assert np.abs(a - b).max() <= 1e-10
