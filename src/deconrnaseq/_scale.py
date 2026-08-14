"""R-compatible scaling helpers (float64 throughout).

R's `scale(x, center=TRUE, scale=TRUE)` centres each column by its mean and
divides by the *sample* standard deviation (ddof=1).  This module reproduces
that behaviour exactly (same formulas, same order of operations) so that
`use_scale=True` matches the original DeconRNASeq numerics.
"""

from __future__ import annotations

import numpy as np


def r_scale(x: np.ndarray, *, _name: str = "matrix") -> np.ndarray:
    """Column-wise equivalent of R ``scale(x)`` (center=TRUE, scale=TRUE).

    Parameters
    ----------
    x : (n, m) float64 array
    _name : label used in error messages

    Returns
    -------
    (n, m) float64 array, centred and scaled column-wise.

    Raises
    ------
    ValueError if any column has zero standard deviation (R would produce
    NaN/Inf there and the downstream QP would fail opaquely; we fail loudly).
    """
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean  # single large temporary
    # R sd: sqrt(sum((x-mean)^2)/(n-1)); einsum fuses the square+reduce with
    # no second temporary (identical formula; fp association may differ from
    # np.sum by ~1 ulp, far below every accuracy gate).
    n = x.shape[0]
    if n < 2:
        raise ValueError(f"{_name}: need at least 2 observations to scale (sd undefined)")
    ssq = np.einsum("ij,ij->j", xc, xc, optimize=True)
    sd = np.sqrt(ssq / (n - 1.0))
    if (sd == 0).any():
        raise ValueError(
            f"{_name}: zero-variance column encountered; R scale() would yield NaN. "
            "Remove constant columns/samples or use use_scale=False."
        )
    xc /= sd  # in-place: reuse the temporary as the output
    return xc


def r_scale_vector(y: np.ndarray, *, _name: str = "sample") -> np.ndarray:
    """1-D equivalent of R ``scale(y)`` for a single mixture column."""
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    return r_scale(y, _name=_name)[:, 0]
