"""Input-validation tests: every illegal input class from the port spec."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import deconrnaseq as drs


def base_data(rng):
    sig = pd.DataFrame(
        rng.random((50, 3)) + 0.1,
        index=[f"g{i}" for i in range(50)],
        columns=["A", "B", "C"],
    )
    mix = pd.DataFrame(
        rng.random((50, 4)) + 0.1,
        index=[f"g{i}" for i in range(50)],
        columns=[f"s{i}" for i in range(4)],
    )
    return sig, mix


def test_none_inputs():
    with pytest.raises(ValueError, match="mixture"):
        drs.deconrnaseq(None, pd.DataFrame(np.ones((5, 2))), fig=False)
    with pytest.raises(ValueError, match="signature"):
        drs.deconrnaseq(pd.DataFrame(np.ones((5, 2))), None, fig=False)


def test_na_and_inf_rejected():
    rng = np.random.default_rng(0)
    sig, mix = base_data(rng)
    mix_bad = mix.copy()
    mix_bad.iloc[3, 1] = np.nan
    with pytest.raises(ValueError, match="NA/Inf"):
        drs.deconrnaseq(mix_bad, sig, fig=False)
    sig_bad = sig.copy()
    sig_bad.iloc[1, 0] = np.inf
    with pytest.raises(ValueError, match="NA/Inf"):
        drs.deconrnaseq(mix, sig_bad, fig=False)


def test_negative_rejected():
    rng = np.random.default_rng(1)
    sig, mix = base_data(rng)
    mix.iloc[0, 0] = -1e-6
    with pytest.raises(ValueError, match="negative"):
        drs.deconrnaseq(mix, sig, fig=False)


def test_duplicate_genes_rejected():
    rng = np.random.default_rng(2)
    sig, mix = base_data(rng)
    sig_dup = sig.copy()
    sig_dup.index = ["g0"] * 2 + [f"g{i}" for i in range(2, 50)]
    with pytest.raises(ValueError, match="duplicate gene"):
        drs.deconrnaseq(mix, sig_dup, fig=False)
    mix_dup = mix.copy()
    mix_dup.index = ["g1", "g1"] + [f"g{i}" for i in range(2, 50)]
    with pytest.raises(ValueError, match="duplicate gene"):
        drs.deconrnaseq(mix_dup, sig, fig=False)


def test_no_common_genes_rejected():
    rng = np.random.default_rng(3)
    sig, mix = base_data(rng)
    mix.index = [f"x{i}" for i in range(50)]
    with pytest.raises(ValueError, match="no common genes"):
        drs.deconrnaseq(mix, sig, fig=False)


def test_genes_lt_cell_types_rejected():
    rng = np.random.default_rng(4)
    sig = pd.DataFrame(rng.random((2, 3)) + 0.1, index=["a", "b"], columns=list("ABC"))
    mix = pd.DataFrame(rng.random((2, 2)) + 0.1, index=["a", "b"], columns=["s1", "s2"])
    with pytest.raises(ValueError, match="less than the number of cell types"):
        drs.deconrnaseq(mix, sig, fig=False)


def test_common_genes_lt_cell_types_rejected():
    rng = np.random.default_rng(5)
    sig = pd.DataFrame(
        rng.random((10, 3)) + 0.1,
        index=[f"g{i}" for i in range(10)],
        columns=list("ABC"),
    )
    mix = pd.DataFrame(
        rng.random((10, 2)) + 0.1,
        index=["g0", "g1"] + [f"x{i}" for i in range(8)],
        columns=["s1", "s2"],
    )
    with pytest.raises(ValueError, match="underdetermined"):
        drs.deconrnaseq(mix, sig, fig=False)


def test_zero_variance_scale_rejected():
    rng = np.random.default_rng(6)
    sig, mix = base_data(rng)
    mix_const = mix.copy()
    mix_const.iloc[:, 0] = 1.0  # constant sample -> sd = 0 under use_scale
    with pytest.raises(ValueError, match="zero-variance"):
        drs.deconrnaseq(mix_const, sig, use_scale=True, fig=False)
    # same input is fine without scaling
    res = drs.deconrnaseq(mix_const, sig, use_scale=False, fig=False)
    assert np.isfinite(res.out_all.to_numpy()).all()


def test_ndim_and_length_mismatch():
    rng = np.random.default_rng(7)
    sig, mix = base_data(rng)
    with pytest.raises(ValueError, match="2-dimensional"):
        drs.deconrnaseq(np.zeros((10, 2, 2)), sig.to_numpy(), fig=False)
    with pytest.raises(ValueError, match="gene ids"):
        drs.deconrnaseq(mix.to_numpy(), sig.to_numpy(), dataset_genes=[f"g{i}" for i in range(49)], fig=False)


def test_duplicate_cell_types_and_samples_rejected():
    rng = np.random.default_rng(8)
    sig, mix = base_data(rng)
    sig_dup = sig.copy()
    sig_dup.columns = ["A", "A", "C"]
    with pytest.raises(ValueError, match="duplicate cell-type"):
        drs.deconrnaseq(mix, sig_dup, fig=False)
    mix_dup = mix.copy()
    mix_dup.columns = ["s0", "s0", "s2", "s3"]
    with pytest.raises(ValueError, match="duplicate sample"):
        drs.deconrnaseq(mix_dup, sig, fig=False)
