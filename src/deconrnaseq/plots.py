"""Plotting equivalents of the R helpers (condplot / multiplot / scatter).

All functions return matplotlib Figure objects and never write files unless
asked to.  Diagnostic only — excluded from the core performance path.
"""

from __future__ import annotations

import math

import numpy as np


def condplot(steps, conds, ax=None):
    """Line chart of condition number vs number of genes (R ``condplot``)."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    ax.plot(np.arange(1, len(steps) + 1), conds, "b-o", lw=1.5)
    ax.set_xticks(np.arange(1, len(steps) + 1, 3))
    ax.set_xticklabels([steps[i] for i in range(0, len(steps), 3)])
    ax.set_xlabel("Number of genes in the signature")
    ax.set_ylabel("Condition number")
    ax.set_title("Condition number of the signature matrix")
    return fig


def proportion_scatter(truth: np.ndarray, estimates: np.ndarray, cell_types, rmses, cols: int = 2):
    """Per-cell-type estimated-vs-actual scatter (R ggplot + multiplot)."""
    import matplotlib.pyplot as plt

    k = len(cell_types)
    rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
    for i in range(k):
        ax = axes[i // cols][i % cols]
        ax.scatter(estimates[:, i], truth[:, i], alpha=0.3)
        lims = [0, 1]
        ax.plot(lims, lims, "r-")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"estimated {cell_types[i]}")
        ax.set_ylabel(f"actual {cell_types[i]}")
        ax.set_title(f"Scatter plot of proportions,\n RMSE = {rmses[i]:.3f}")
    for j in range(k, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    return fig
