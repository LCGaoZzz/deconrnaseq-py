"""Regression against the exported R 1.50.0 outputs.

Gates from the port specification:
  exact data  max_abs_error <= 1e-7   (both use_scale modes)
  noisy data  max_abs_error <= 1e-6   (both use_scale modes)
Skipped (not failed) when r_baseline/*.csv are absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from deconrnaseq import deconvolve_core

CASES = [
    ("exact", False, 1e-7),
    ("exact", True, 1e-7),
    ("noisy", False, 1e-6),
    ("noisy", True, 1e-6),
]

SOLVERS = ["enum", "enum_fast", "interior", "activeset_warm"]


@pytest.mark.parametrize("case,use_scale,gate", CASES)
@pytest.mark.parametrize("solver", SOLVERS)
def test_vs_r(signatures, mixtures_exact, mixtures_noisy, truth, r_reference, case, use_scale, gate, solver):
    if r_reference is None:
        pytest.skip("R baseline outputs not found")
    mix = mixtures_exact if case == "exact" else mixtures_noisy
    ref = r_reference[(case, "scaled" if use_scale else "unscaled")]
    est = deconvolve_core(
        signatures.to_numpy(np.float64),
        mix.to_numpy(np.float64),
        use_scale=use_scale,
        solver=solver,
        backend="numpy",
    )
    ref_np = ref.to_numpy(np.float64)
    # name/order agreement
    assert list(ref.index) == list(mix.columns)
    assert list(ref.columns) == list(signatures.columns)
    assert np.isfinite(est).all()
    assert est.min() >= -1e-12
    assert np.abs(est.sum(axis=1) - 1.0).max() <= 1e-10
    assert np.abs(est - ref_np).max() <= gate


@pytest.mark.parametrize("case,use_scale,gate", CASES)
def test_vs_r_rust_backend(signatures, mixtures_exact, mixtures_noisy, r_reference, case, use_scale, gate):
    import deconrnaseq as drs

    if r_reference is None:
        pytest.skip("R baseline outputs not found")
    if not drs.rust_available():
        pytest.skip("rust backend not installed")
    mix = mixtures_exact if case == "exact" else mixtures_noisy
    ref = r_reference[(case, "scaled" if use_scale else "unscaled")]
    est = deconvolve_core(
        signatures.to_numpy(np.float64),
        mix.to_numpy(np.float64),
        use_scale=use_scale,
        solver="interior",
        backend="rust",
    )
    assert np.abs(est - ref.to_numpy(np.float64)).max() <= gate
