"""PCA diagnostics equivalent to the pcaMethods call inside DeconRNASeq.

R code:
    x.data.temp <- prep(x.data, scale = "none", center = TRUE)
    x.data.pca  <- pca(x.data.temp, method = "svd", center = FALSE, nPcs = Numofx)
    Var         <- R2cum(x.data.pca)
    numofmix    <- order(Var > 0.99, decreasing = TRUE)[1]

pcaMethods treats matrix rows as observations, so the (genes x samples)
mixture matrix is decomposed directly and each *sample column* is centred.
Diagnostic only — excluded from the core performance path.
"""

from __future__ import annotations

import numpy as np


def mixture_pca(mixtures: np.ndarray, n_pcs: int) -> dict:
    """SVD-PCA of the column-centred mixture matrix.

    Returns a dict with per-PC R2, cumulative R2 (R2cum), singular values,
    and ``numofmix`` replicating the R expression
    ``order(R2cum > 0.99, decreasing = TRUE)[1]`` literally (1-based index of
    the first PC whose cumulative R2 exceeds 0.99; R's stable ordering yields
    1 when no PC crosses the threshold).
    """
    X = np.asarray(mixtures, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    var = s * s
    total = var.sum()
    r2 = var / total if total > 0 else np.zeros_like(var)
    r2cum = np.cumsum(r2)[:n_pcs]
    over = r2cum > 0.99
    # R: order(Var>0.99, decreasing=T)[1] -> first TRUE position, else 1
    numofmix = int(np.argmax(over)) + 1 if over.any() else 1
    return {
        "n_pcs": int(n_pcs),
        "sdev": s[:n_pcs] / np.sqrt(max(X.shape[0] - 1, 1)),
        "R2": r2[:n_pcs],
        "R2cum": r2cum,
        "numofmix": numofmix,
    }
