"""Python port of Bioconductor DeconRNASeq 1.50.0 (R/DeconRNASeq.R).

Original R package: Ting Gong, Joseph D. Szustakowski (GPL-2).
This is a faithful re-implementation; see LICENSE and NOTICE.

Public entry point: :func:`deconrnaseq`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._scale import r_scale
from ._validate import AlignedInputs, validate_and_align
from .solvers import solve_lsei

__all__ = ["deconrnaseq", "DeconResult", "deconvolve_core"]


class DeconResult(dict):
    """Result container mirroring the R list(out.all, out.pca, out.rmse).

    Both R-style keys (``result["out.all"]``) and attribute access
    (``result.out_all``) are supported.
    """

    _ALIASES = {
        "out.all": "out_all",
        "out.pca": "out_pca",
        "out.rmse": "out_rmse",
        "out.rmse.per_type": "rmse_per_type",
        "out.cond": "cond",
    }

    def __getattr__(self, name: str) -> Any:
        for key, attr in self._ALIASES.items():
            if attr == name and key in self:
                return self[key]
        raise AttributeError(name)

    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(self._ALIASES.get(key, key) if key in self._ALIASES and key not in self else key)


def deconvolve_core(
    signature: np.ndarray,
    mixtures: np.ndarray,
    *,
    use_scale: bool = True,
    solver: str = "auto",
    backend: str = "auto",
) -> np.ndarray:
    """The performance-critical core: scaling + Gram/cross + QP solve.

    Parameters
    ----------
    signature : (G, K) float64, already validated & gene-aligned
    mixtures  : (G, S) float64, rows aligned to ``signature``
    use_scale : apply R ``scale()`` to signature and to every sample column
    solver    : solver method passed to :func:`deconrnaseq.solvers.solve_lsei`
    backend   : "auto" (Rust acceleration if the optional wheel is installed,
        else pure NumPy), "numpy" (force pure NumPy) or "rust" (require Rust)

    Returns
    -------
    (S, K) float64 proportions, rows sum to 1, non-negative.
    """
    A = np.asarray(signature, dtype=np.float64)
    Y = np.asarray(mixtures, dtype=np.float64)
    if use_scale:
        A = r_scale(A, _name="signature")
        Y = r_scale(Y, _name="mixture dataset")
    gram = A.T @ A
    cross = A.T @ Y
    return solve_lsei(gram, cross, method=solver, backend=backend).T.copy()


def deconrnaseq(
    datasets,
    signatures,
    proportions=None,
    checksig: bool = False,
    known_prop: bool = False,
    use_scale: bool = True,
    fig: bool = True,
    *,
    solver: str = "auto",
    backend: str = "auto",
    dataset_genes=None,
    signature_genes=None,
    sample_names=None,
    cell_type_names=None,
) -> DeconResult:
    """Deconvolute mixture samples into cell-type proportions.

    Faithful port of R ``DeconRNASeq(datasets, signatures, proportions,
    checksig, known.prop, use.scale, fig)``.

    Parameters
    ----------
    datasets : genes x samples matrix. pandas DataFrame (index = gene ids,
        columns = sample names) or NumPy array (then pass ``dataset_genes``).
    signatures : genes x cell types matrix (DataFrame or NumPy +
        ``signature_genes`` / ``cell_type_names``).
    proportions : optional samples x cell types DataFrame of known fractions
        (required when ``known_prop=True``); rows are matched by sample name,
        exactly like the R code ``x.proportions[colnames(x.data), ]``.
    checksig : compute the stepwise condition-number diagnostic (R ``checksig``).
    known_prop : compute per-cell-type RMSE against ``proportions``.
    use_scale : R ``scale()`` preprocessing of signature and samples
        (centre + sample-sd, ddof=1).
    fig : draw figures (PCA message / condition plot / proportion scatter).
    solver : "auto" (default), "interior", "interior_rust", "enum",
        "enum_fast", "activeset", "activeset_warm" — all solve the identical QP.
    backend : "auto" | "numpy" | "rust" — backend for the boundary
        enumeration in the default interior-first solver.

    Returns
    -------
    DeconResult with
      - ``out.all``  : DataFrame, samples x cell types (R ``out.all``)
      - ``out.pca``  : dict with PCA diagnostics (R ``out.pca`` summary)
      - ``out.rmse`` : mean RMSE when ``known_prop=True`` (R ``out.rmse``)
    """
    # convenience: allow passing the known proportions directly as
    # ``known_prop`` (the R API uses a logical flag + ``proportions``)
    if not isinstance(known_prop, bool):
        if proportions is None:
            proportions = known_prop
        known_prop = True

    aligned: AlignedInputs = validate_and_align(
        datasets,
        signatures,
        dataset_genes=dataset_genes,
        signature_genes=signature_genes,
        sample_names=sample_names,
        cell_type_names=cell_type_names,
    )

    # ---- PCA diagnostic (always computed, as in the R code) ----------------
    from .pca import mixture_pca

    out_pca = mixture_pca(aligned.mixtures, n_pcs=len(aligned.cell_types))
    if fig and out_pca["numofmix"] != len(aligned.cell_types):
        print(
            "\n Attention: the number of pure cell types =",
            len(aligned.cell_types),
            " defined in the signature matrix;\n",
        )
        print(
            "\n PCA results indicate that the number of cell types in the"
            " mixtures =",
            out_pca["numofmix"],
            "\n",
        )

    # ---- optional condition-number diagnostic ------------------------------
    cond_table = None
    if checksig and aligned.signature.shape[0] >= 40:
        from .diagnostics import signature_condition_curve

        cond_table = signature_condition_curve(aligned.signature, step=20)
        if fig:
            from .plots import condplot

            condplot(cond_table["n_genes"].to_numpy(), cond_table["condition_number"].to_numpy())

    # ---- core solve ---------------------------------------------------------
    estimates = deconvolve_core(
        aligned.signature,
        aligned.mixtures,
        use_scale=use_scale,
        solver=solver,
        backend=backend,
    )
    out_all = pd.DataFrame(
        estimates, index=pd.Index(aligned.samples, name=None), columns=aligned.cell_types
    )

    result = DeconResult({"out.all": out_all, "out.pca": out_pca})
    if cond_table is not None:
        result["out.cond"] = cond_table

    # ---- optional RMSE against known proportions ---------------------------
    if known_prop:
        if proportions is None:
            raise ValueError(
                "missing the known proportions; provide `proportions` when known_prop=True"
            )
        prop = proportions if isinstance(proportions, pd.DataFrame) else pd.DataFrame(proportions)
        prop = prop.reindex(index=aligned.samples, columns=aligned.cell_types)
        if prop.isna().any().any():
            raise ValueError(
                "proportions could not be matched to the mixture samples/cell types "
                "(check row and column names)"
            )
        truth = prop.to_numpy(dtype=np.float64)
        rmses = np.sqrt(((estimates - truth) ** 2).mean(axis=0))  # per cell type
        result["out.rmse"] = float(rmses.mean())
        result["out.rmse.per_type"] = pd.Series(rmses, index=aligned.cell_types)
        if fig:
            from .plots import proportion_scatter

            proportion_scatter(truth, estimates, aligned.cell_types, rmses)
    return result
