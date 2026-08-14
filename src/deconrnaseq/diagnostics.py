"""Signature condition-number diagnostic (R ``checksig`` branch).

R code:
    step <- seq(20, numofg, by = 20)
    sig.cond <- sapply(step, function(x) kappa(scale(x.signature[1:x, ])))

kappa() is the 2-norm condition number; scale() is R's centre + sd(ddof=1).
Diagnostic only — excluded from the core performance path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._scale import r_scale


def signature_condition_curve(signature: np.ndarray, step: int = 20, start: int = 20) -> pd.DataFrame:
    """Condition number of the column-scaled signature vs number of genes."""
    A = np.asarray(signature, dtype=np.float64)
    n_genes = A.shape[0]
    steps = list(range(start, n_genes + 1, step))
    conds = [float(np.linalg.cond(r_scale(A[:n, :]))) for n in steps]
    return pd.DataFrame({"n_genes": steps, "condition_number": conds})
