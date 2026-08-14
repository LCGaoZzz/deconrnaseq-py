"""Post-install smoke test: exercises the INSTALLED deconrnaseq wheel
(no sys.path hacks) against the shipped test data and R baselines.

Usage: <python> smoke_install.py [repo_or_turn_dir]
Exit code 0 = all checks passed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

import deconrnaseq as drs


def _find_dir(root, name):
    cand = os.path.join(root, name)
    return cand if os.path.isdir(cand) else os.path.join(root, "deconrnaseq_py", name)


if len(sys.argv) > 1:
    _ROOT = sys.argv[1]
else:  # default: the repo this file lives in
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(_find_dir(_ROOT, "test_data"))
sig = pd.read_csv(os.path.join(BASE, "test_data", "reference_signatures.csv"), index_col=0)
mix_e = pd.read_csv(os.path.join(BASE, "test_data", "mixtures_exact.csv"), index_col=0)
mix_n = pd.read_csv(os.path.join(BASE, "test_data", "mixtures_noisy.csv"), index_col=0)
truth = pd.read_csv(os.path.join(BASE, "test_data", "true_proportions.csv"), index_col=0)
rb = lambda c, s: pd.read_csv(os.path.join(BASE, "r_baseline", f"mixtures_{c}_r_{s}.csv"), index_col=0)

fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


print("package file:", drs.__file__)
print("rust_available:", drs.rust_available())

# 1) end-to-end API on DataFrames
res = drs.deconrnaseq(mix_n, sig, use_scale=True, fig=False)
check("out_all shape", res.out_all.shape == (48, 8))
check("index = sample names", list(res.out_all.index) == list(mix_n.columns))
check("columns = cell types", list(res.out_all.columns) == list(sig.columns))
check("finite", np.isfinite(res.out_all.to_numpy()).all())
check("rows sum to 1", np.abs(res.out_all.sum(axis=1) - 1).max() <= 1e-10)
check("non-negative", res.out_all.to_numpy().min() >= -1e-12)

# 2) hard accuracy gates vs R baselines (both scales, exact + noisy)
for case, mix, gate in (("exact", mix_e, 1e-7), ("noisy", mix_n, 1e-6)):
    for sc in (False, True):
        est = drs.deconrnaseq(mix, sig, use_scale=sc, fig=False).out_all.to_numpy()
        ref = rb(case, "scaled" if sc else "unscaled").to_numpy()
        check(f"vs R {case} scale={sc}", np.abs(est - ref).max() <= gate)

# 3) truth gates (exact + unscaled)
est = drs.deconrnaseq(mix_e, sig, use_scale=False, fig=False).out_all.to_numpy()
t = truth.to_numpy()
check("truth max_abs", np.abs(est - t).max() <= 1e-8)
check("truth rmse", np.sqrt(((est - t) ** 2).mean()) <= 1e-9)

# 4) known_prop / RMSE path
res = drs.deconrnaseq(mix_e, sig, proportions=truth, known_prop=True, use_scale=False, fig=False)
check("rmse present", res["out.rmse"] < 1e-9)

# 5) backend behaviour
if drs.rust_available():
    a = drs.deconvolve_core(sig.to_numpy(), mix_n.to_numpy(), use_scale=False, backend="numpy")
    b = drs.deconvolve_core(sig.to_numpy(), mix_n.to_numpy(), use_scale=False, backend="rust")
    check("rust==numpy", np.abs(a - b).max() <= 1e-10)
else:
    try:
        drs.deconvolve_core(sig.to_numpy(), mix_n.to_numpy(), use_scale=False, backend="rust")
        check("backend=rust without wheel raises", False)
    except Exception as exc:  # noqa: BLE001
        check("backend=rust without wheel raises", True)
        print("   ->", type(exc).__name__, str(exc)[:120])
    est = drs.deconvolve_core(sig.to_numpy(), mix_n.to_numpy(), use_scale=False, backend="auto")
    check("backend=auto falls back to numpy", np.isfinite(est).all())

# 6) validation rejects bad input
bad = mix_e.copy()
bad.iloc[0, 0] = np.nan
try:
    drs.deconrnaseq(bad, sig, fig=False)
    check("NA rejected", False)
except ValueError:
    check("NA rejected", True)

print()
if fails:
    print("SMOKE FAILED:", fails)
    sys.exit(1)
print("SMOKE OK — rust_available =", drs.rust_available())
