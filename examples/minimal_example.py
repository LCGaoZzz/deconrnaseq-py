"""Minimal end-to-end example for deconrnaseq-py.

Run from the turn output directory (the one containing test_data/):

    python deconrnaseq_py/examples/minimal_example.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

import deconrnaseq as drs

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)  # deconrnaseq_py repo root


def _find_dir(name):
    in_repo = os.path.join(REPO, name)
    return in_repo if os.path.isdir(in_repo) else os.path.dirname(REPO)


BASE = os.path.dirname(_find_dir("test_data"))  # dir containing test_data/

signatures = pd.read_csv(os.path.join(BASE, "test_data", "reference_signatures.csv"), index_col=0)
mixtures = pd.read_csv(os.path.join(BASE, "test_data", "mixtures_noisy.csv"), index_col=0)
truth = pd.read_csv(os.path.join(BASE, "test_data", "true_proportions.csv"), index_col=0)

# 1) one call -> proportions (samples x cell types)
res = drs.deconrnaseq(mixtures, signatures, use_scale=True, fig=False)
print("estimated proportions (first 3 samples):")
print(res.out_all.iloc[:3].round(4))
print("row sums:", res.out_all.sum(axis=1).round(12).unique())

# 2) with known ground truth -> RMSE, exactly like R's known.prop
res2 = drs.deconrnaseq(
    mixtures, signatures, proportions=truth, known_prop=True, use_scale=False, fig=False
)
print("\nmean per-cell-type RMSE (noisy, unscaled):", round(res2["out.rmse"], 6))
print(res2.rmse_per_type.round(6))

# 3) solver/backend are explicit and swappable; identical QP either way
print("\nrust backend installed:", drs.rust_available())
fast = drs.deconrnaseq(mixtures, signatures, solver="interior", backend="auto", fig=False)
slow = drs.deconrnaseq(mixtures, signatures, solver="enum", backend="numpy", fig=False)
print("max |interior - enum|:", (fast.out_all - slow.out_all).abs().max().max())
