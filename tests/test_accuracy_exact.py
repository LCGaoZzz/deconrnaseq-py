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


@pytest.mark.skipif(not drs.rust_available(), reason="rust backend not installed")
def test_rust_enum_is_repeatable_and_preserves_inputs(signatures, mixtures_noisy):
    """Guard the direct Rust boundary kernel used by the interior solver."""
    import deconrnaseq_rust
    from deconrnaseq.solvers import solve_lsei_enum_fast

    signature = np.ascontiguousarray(signatures.to_numpy(np.float64))
    mixtures = np.ascontiguousarray(mixtures_noisy.to_numpy(np.float64))
    gram = np.ascontiguousarray(signature.T @ signature)
    cross = np.ascontiguousarray(signature.T @ mixtures)
    gram_before = gram.copy()
    cross_before = cross.copy()

    expected = solve_lsei_enum_fast(gram, cross)
    actual_first = np.asarray(deconrnaseq_rust.solve_lsei_enum(gram, cross))
    actual_second = np.asarray(deconrnaseq_rust.solve_lsei_enum(gram, cross))

    np.testing.assert_allclose(actual_first, expected, rtol=0.0, atol=1e-10)
    np.testing.assert_array_equal(actual_second, actual_first)
    np.testing.assert_array_equal(gram, gram_before)
    np.testing.assert_array_equal(cross, cross_before)
