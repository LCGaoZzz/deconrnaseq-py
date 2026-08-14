"""Benchmark driver: runs the full protocol matrix in fresh subprocesses
(interleaved timing inside each) and writes tidy CSVs to results/.

Protocol (fixed):
  * data: the synthetic validation set (correctness) + deterministic
    Dirichlet-generated larger sets (performance only, never correctness)
  * threads: 1 (pinned via env before numpy loads in the worker)
  * warmup 3, then 21 interleaved repetitions per variant
  * timing excludes file I/O, imports, warmup, and the memory pass
  * reports median / p95 / min / max, peak tracemalloc bytes, RSS delta
  * accuracy vs R 1.50.0 reference CSVs and generator truth in the same run;
    variants failing gates are kept in the CSV but flagged gate_pass=0
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent


def _find_dir(name: str) -> Path:
    """Prefer the in-repo copy (publishable layout); fall back to the
    original sibling layout used during development."""
    in_repo = PKG / name
    return in_repo if in_repo.is_dir() else PKG.parent / name


DATA = _find_dir("test_data")
R_DIR = _find_dir("r_baseline")
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

ALL_IMPLS = [
    "baseline_loop",
    "enum_cachedgram_loop",
    "enum_fast_persample_loop",
    "enum_batched",
    "enum_batched_interior",
    "activeset_loop_cold",
    "activeset_loop_warm",
    "rust_enum_batched",
    "rust_enum_interior",
    "final_numpy",
    "final_rust",
    "floor_probe",
]
FAST_IMPLS = [i for i in ALL_IMPLS if i not in ("baseline_loop", "enum_cachedgram_loop", "enum_fast_persample_loop")]

GATES = {("exact", "off"): 1e-7, ("exact", "on"): 1e-7, ("noisy", "off"): 1e-6, ("noisy", "on"): 1e-6}


def run(impls, case, scale, samples, reps=21, large_n=0):
    cmd = [
        sys.executable, str(HERE / "benchmark.py"),
        "--impls", ",".join(impls),
        "--case", case, "--scale", scale, "--samples", str(samples),
        "--reps", str(reps), "--threads", "1",
        "--data-dir", str(DATA), "--r-dir", str(R_DIR), "--pkg-src", str(PKG / "src"),
    ]
    if large_n:
        cmd += ["--large-n", str(large_n)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        raise RuntimeError(f"benchmark failed {case}/{scale}/{samples}:\n{r.stderr[-3000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    rows = []
    # ---- correctness + speed matrix: 4 combos x {1, 8, 48} samples ---------
    for case in ("exact", "noisy"):
        for scale in ("off", "on"):
            for samples in (1, 8, 48):
                impls = ALL_IMPLS if samples == 48 else [
                    "baseline_loop", "enum_batched", "enum_batched_interior",
                    "activeset_loop_warm", "rust_enum_interior", "final_numpy", "final_rust",
                ]
                res = run(impls, case, scale, samples)
                for name, m in res.items():
                    gate = GATES[(case, scale)]
                    m["gate_pass"] = int(
                        m["finite"]
                        and m["min_value"] >= -1e-12
                        and m["max_sum_error"] <= 1e-10
                        and m.get("max_abs_vs_r", 0.0) <= gate
                        and (case != "exact" or scale != "off" or (
                            m["max_abs_vs_truth"] <= 1e-8 and m["rmse_vs_truth"] <= 1e-9))
                        and name != "floor_probe"
                    )
                    m["gate_threshold_vs_r"] = gate
                    rows.append(m)
                print(f"done {case}/{scale}/n={samples}")

    # ---- performance-only larger sets (never correctness) ------------------
    for large_n in (480, 4800):
        res = run(FAST_IMPLS, "exact", "off", "all", reps=11, large_n=large_n)
        for name, m in res.items():
            m["case"] = f"synthetic_{large_n}"
            m["gate_pass"] = 0  # perf-only set: not a correctness row
            m["note"] = "performance-only synthetic set; correctness gates do not apply"
            rows.append(m)
        print(f"done synthetic n={large_n}")

    import csv

    keys = sorted({k for row in rows for k in row})
    out = RESULTS / "benchmark_results.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
