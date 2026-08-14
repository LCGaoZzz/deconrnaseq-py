"""Regenerate the definitive core-solve suite (results/confirmation_suite.csv).

Protocol (matches benchmarks/REPORT.md):
  * 4 configurations: {exact, noisy} x {scale off, on}, all 48 samples
  * implementations: baseline_loop, enum_batched, activeset_loop_warm,
    final_numpy, final_rust, floor_probe (harness-floor control)
  * per measurement: 3 warmup + 21 interleaved timed reps, 1 BLAS thread
  * the whole configuration set is repeated 3 times round-robin
    (suite_rep 0..2) to cancel machine drift
  * tracemalloc runs outside the timed path

Usage:  python benchmarks/run_confirmation.py [--suite-reps 3] [--reps 21]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent


def _find_dir(name: str) -> Path:
    in_repo = PKG / name
    return in_repo if in_repo.is_dir() else PKG.parent / name


IMPLS = [
    "baseline_loop",
    "enum_batched",
    "activeset_loop_warm",
    "final_numpy",
    "final_rust",
    "floor_probe",
]

GATES = {("exact", "off"): 1e-7, ("exact", "on"): 1e-7, ("noisy", "off"): 1e-6, ("noisy", "on"): 1e-6}


def run_one(case, scale, suite_rep, reps):
    cmd = [
        sys.executable, str(HERE / "benchmark.py"),
        "--impls", ",".join(IMPLS),
        "--case", case, "--scale", scale, "--samples", "48",
        "--reps", str(reps), "--threads", "1",
        "--data-dir", str(_find_dir("test_data")),
        "--r-dir", str(_find_dir("r_baseline")),
        "--pkg-src", str(PKG / "src"),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        raise RuntimeError(f"benchmark failed {case}/{scale} rep{suite_rep}:\n{r.stderr[-3000:]}")
    res = json.loads(r.stdout.strip().splitlines()[-1])
    rows = []
    for name, m in res.items():
        m["suite_rep"] = suite_rep
        gate = GATES[(case, scale)]
        m["gate_pass"] = int(
            m["finite"] and m["min_value"] >= -1e-12 and m["max_sum_error"] <= 1e-10
            and m.get("max_abs_vs_r", 0.0) <= gate and name != "floor_probe"
        )
        rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-reps", type=int, default=3)
    ap.add_argument("--reps", type=int, default=21)
    args = ap.parse_args()

    rows = []
    for suite_rep in range(args.suite_reps):  # round-robin across configs
        for case in ("exact", "noisy"):
            for scale in ("off", "on"):
                rows.extend(run_one(case, scale, suite_rep, args.reps))
                print(f"done suite_rep={suite_rep} {case}/{scale}")

    keys = sorted({k for row in rows for k in row})
    out = HERE / "results" / "confirmation_suite.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
