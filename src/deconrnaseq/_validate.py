"""Input coercion + validation for the Python port of DeconRNASeq.

The R original checks: data.frame type, absence of NA, genes >= cell types.
This port additionally rejects (as required by the port specification):
non-finite values, negative values, duplicate gene ids, no common genes,
and (post-alignment) fewer genes than cell types.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlignedInputs:
    """Validated, gene-aligned numeric inputs (all float64)."""

    signature: np.ndarray  # (G, K) genes x cell types, signature gene order
    mixtures: np.ndarray  # (G, S) genes x samples, rows aligned to signature
    genes: list[str]  # common genes in signature order
    cell_types: list[str]  # signature column names (length K)
    samples: list[str]  # mixture column names (length S)


def _coerce_matrix(obj, genes, col_names, default_col_prefix, what):
    """Return (values float64 (G,N), gene_ids list, column_names list)."""
    if obj is None:
        raise ValueError(f"missing {what}: please provide the {what} matrix.")
    if isinstance(obj, pd.DataFrame):
        values = obj.to_numpy(dtype=np.float64, copy=False)
        gene_ids = [str(g) for g in obj.index.tolist()]
        columns = [str(c) for c in obj.columns.tolist()]
    elif isinstance(obj, pd.Series):
        values = obj.to_numpy(dtype=np.float64, copy=False).reshape(-1, 1)
        gene_ids = [str(g) for g in obj.index.tolist()]
        columns = [col_names[0] if col_names else f"{default_col_prefix}1"]
    else:
        arr = np.asarray(obj)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"{what} must be 2-dimensional, got shape {arr.shape}")
        values = np.asarray(arr, dtype=np.float64)
        n_rows, n_cols = values.shape
        if genes is None:
            gene_ids = [f"g{i + 1}" for i in range(n_rows)]
        else:
            if len(genes) != n_rows:
                raise ValueError(
                    f"{what}: {n_rows} rows but {len(genes)} gene ids supplied"
                )
            gene_ids = [str(g) for g in genes]
        if col_names is None:
            columns = [f"{default_col_prefix}{i + 1}" for i in range(n_cols)]
        else:
            if len(col_names) != n_cols:
                raise ValueError(
                    f"{what}: {n_cols} columns but {len(col_names)} names supplied"
                )
            columns = [str(c) for c in col_names]
    return values, gene_ids, columns


def _check_matrix(name, values, row_ids):
    if values.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(values).all():
        raise ValueError(
            f"{name} contains NA/Inf values; exclude or impute missing values first."
        )
    if (values < 0).any():
        raise ValueError(f"{name} contains negative values; expression data must be >= 0.")
    if len(set(row_ids)) != len(row_ids):
        dup = pd.Index(row_ids)[pd.Index(row_ids).duplicated()].unique().tolist()
        raise ValueError(f"{name} has duplicate gene ids: {dup[:5]} (deduplicate first)")


def validate_and_align(
    datasets,
    signatures,
    *,
    dataset_genes=None,
    signature_genes=None,
    sample_names=None,
    cell_type_names=None,
) -> AlignedInputs:
    """Validate inputs and align rows by gene id, preserving signature order.

    Mirrors R DeconRNASeq 1.50.0:
      common.signature <- rownames(x.signature) %in% rownames(x.data)
      x.signature      <- x.signature[common.signature, ]
      x.subdata        <- x.data[rownames(x.signature), ]
    """
    sig, sig_genes, cell_types = _coerce_matrix(
        signatures, signature_genes, cell_type_names, "Type", "signature"
    )
    dat, dat_genes, samples = _coerce_matrix(
        datasets, dataset_genes, sample_names, "Sample", "mixture dataset"
    )

    _check_matrix("signature", sig, sig_genes)
    _check_matrix("mixture dataset", dat, dat_genes)
    if len(set(cell_types)) != len(cell_types):
        raise ValueError("signature has duplicate cell-type names")
    if len(set(samples)) != len(samples):
        raise ValueError("mixture dataset has duplicate sample names")

    n_genes, n_types = sig.shape
    if n_genes < n_types:
        raise ValueError(
            "The number of genes is less than the number of cell types, "
            "which means less independent equations than unknowns."
        )

    dat_gene_set = set(dat_genes)
    keep = [i for i, g in enumerate(sig_genes) if g in dat_gene_set]
    if len(keep) == 0:
        raise ValueError("no common genes between signature and mixture dataset")
    if len(keep) < n_types:
        raise ValueError(
            f"only {len(keep)} common genes remain after alignment, fewer than "
            f"the {n_types} cell types; the system would be underdetermined."
        )

    genes_common = [sig_genes[i] for i in keep]
    sig_aligned = np.ascontiguousarray(sig[keep, :], dtype=np.float64)

    # reorder mixture rows to the signature gene order (x.subdata in R)
    dat_pos = {g: i for i, g in enumerate(dat_genes)}
    row_idx = np.fromiter((dat_pos[g] for g in genes_common), dtype=np.intp, count=len(keep))
    dat_aligned = np.ascontiguousarray(dat[row_idx, :], dtype=np.float64)

    return AlignedInputs(
        signature=sig_aligned,
        mixtures=dat_aligned,
        genes=genes_common,
        cell_types=cell_types,
        samples=samples,
    )
