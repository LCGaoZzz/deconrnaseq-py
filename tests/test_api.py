"""API, output-structure, alignment and edge-case tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import deconrnaseq as drs
from deconrnaseq import deconvolve_core


def test_output_structure(signatures, mixtures_exact):
    res = drs.deconrnaseq(mixtures_exact, signatures, use_scale=False, fig=False, checksig=True)
    assert isinstance(res.out_all, pd.DataFrame)
    assert list(res.out_all.index) == list(mixtures_exact.columns)
    assert list(res.out_all.columns) == list(signatures.columns)
    assert res.out_all.shape == (48, 8)
    assert res.out_pca is not None
    assert res.cond is not None          # checksig=True + 5000 genes >= 40
    assert res["out.cond"] is res.cond
    assert "out.rmse" not in res        # known_prop not requested
    with pytest.raises(AttributeError):
        _ = res.out_rmse


def test_row_order_alignment(signatures, mixtures_exact):
    """Mixtures whose rows are shuffled must give the identical result."""
    rng = np.random.default_rng(11)
    perm = rng.permutation(len(mixtures_exact.index))
    shuffled = mixtures_exact.iloc[perm, :]
    a = drs.deconrnaseq(mixtures_exact, signatures, use_scale=False, fig=False).out_all
    b = drs.deconrnaseq(shuffled, signatures, use_scale=False, fig=False).out_all
    pd.testing.assert_frame_equal(a, b)


def test_signature_extra_genes_and_subset(signatures, mixtures_exact):
    """Extra signature-only genes are dropped; result unchanged vs the
    gene-intersection computation."""
    extra = pd.DataFrame(
        0.5,
        index=[f"extra_{i}" for i in range(7)],
        columns=signatures.columns,
    )
    sig2 = pd.concat([signatures, extra])
    a = drs.deconrnaseq(mixtures_exact, signatures, use_scale=False, fig=False).out_all
    b = drs.deconrnaseq(mixtures_exact, sig2, use_scale=False, fig=False).out_all
    pd.testing.assert_frame_equal(a, b)


def test_numpy_input_generated_names(signatures, mixtures_exact):
    A = signatures.to_numpy(np.float64)
    Y = mixtures_exact.to_numpy(np.float64)
    res = drs.deconrnaseq(Y, A, use_scale=False, fig=False)
    assert list(res.out_all.index) == [f"Sample{i + 1}" for i in range(Y.shape[1])]
    assert list(res.out_all.columns) == [f"Type{i + 1}" for i in range(A.shape[1])]


def test_numpy_input_explicit_names(signatures, mixtures_exact):
    A = signatures.to_numpy(np.float64)
    Y = mixtures_exact.to_numpy(np.float64)
    names = [f"S{i}" for i in range(Y.shape[1])]
    ctypes = [f"ct{i}" for i in range(A.shape[1])]
    res = drs.deconrnaseq(
        Y, A, use_scale=False, fig=False, sample_names=names, cell_type_names=ctypes
    )
    assert list(res.out_all.index) == names
    assert list(res.out_all.columns) == ctypes


def test_single_sample_and_fig(signatures, mixtures_exact, truth):
    plt = pytest.importorskip("matplotlib")  # fig=True needs the [plots] extra
    import matplotlib
    matplotlib.use("Agg")
    one = mixtures_exact.iloc[:, [0]]
    res = drs.deconrnaseq(
        one,
        signatures,
        proportions=truth.iloc[[0], :],
        known_prop=True,
        use_scale=False,
        fig=True,
        checksig=False,
    )
    assert res.out_all.shape == (1, 8)
    assert np.isfinite(res["out.rmse"])  # single sample + fig=True: barplot path runs


def test_known_prop_rmse(signatures, mixtures_exact, truth):
    res = drs.deconrnaseq(
        mixtures_exact,
        signatures,
        proportions=truth,
        known_prop=True,
        use_scale=False,
        fig=False,
    )
    assert res["out.rmse"] < 1e-9  # mean per-cell-type RMSE vs truth (exact data)
    assert res.rmse_per_type is not None
    assert len(res.rmse_per_type) == signatures.shape[1]
    assert (res.rmse_per_type.to_numpy() < 1e-9).all()


def test_known_prop_accepts_dataframe_shortcut(signatures, mixtures_exact, truth):
    """Passing the proportions table directly as known_prop is a supported
    convenience (coerced to proportions + known_prop=True)."""
    res = drs.deconrnaseq(
        mixtures_exact, signatures, known_prop=truth, use_scale=False, fig=False
    )
    assert res["out.rmse"] < 1e-9


def test_fig_false_default():
    """fig parameter is optional and defaults to False."""
    res = drs.deconrnaseq(
        np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 1.5]]),
        np.array([[1.0, 0.2], [0.2, 1.0], [0.6, 0.4]]),
        fig=False,
    )
    assert "out.rmse" not in res
    assert res.out_all.shape == (2, 2)
    assert np.isfinite(res.out_all.to_numpy()).all()


def test_two_cell_types():
    rng = np.random.default_rng(13)
    A = rng.random((40, 2)) + 0.05
    p = np.array([0.3, 0.7])
    y = A @ p
    est = deconvolve_core(A, y.reshape(-1, 1), use_scale=False, solver="interior")
    assert np.abs(est[0] - p).max() < 1e-8


def test_dtype_float64(signatures, mixtures_exact):
    est = deconvolve_core(
        signatures.to_numpy(np.float64), mixtures_exact.to_numpy(np.float64),
        use_scale=False, solver="interior",
    )
    assert est.dtype == np.float64


def test_scale_matches_r_semantics(signatures, mixtures_noisy, r_reference):
    """scaled pipeline output must match R's scale() (ddof=1) baseline."""
    if r_reference is None:
        pytest.skip("R baseline outputs not found")
    est = drs.deconrnaseq(mixtures_noisy, signatures, use_scale=True, fig=False).out_all
    ref = r_reference[("noisy", "scaled")]
    assert np.abs(est.to_numpy() - ref.to_numpy()).max() <= 1e-6
