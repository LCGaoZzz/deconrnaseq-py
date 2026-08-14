"""One-click reproduction entry point for reviewers.

Runs the full acceptance chain against THIS checkout:

  1. check (or optionally regenerate) the R 1.50.0 baseline reference CSVs
  2. pytest suite (validation / API / accuracy gates / R regression)
  3. benchmark rerun (quick: definitive implementations, 48 samples;
     --full: complete matrix incl. n=1/8 and synthetic 480/4800)
  4. re-verify every hard accuracy gate on the freshly written CSV
  5. print a pass/fail summary and where each evidence file landed

Usage (from the repository root):

    python reproduce.py                 # quick acceptance (~2-4 min)
    python reproduce.py --full          # the complete benchmark matrix
    python reproduce.py --run-r "G:\\R\\R-4.3.1\\bin\\Rscript.exe"
                                        # regenerate R baseline first
    python reproduce.py --skip-tests    # benchmarks only

Nothing is written outside the repository directory.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "benchmarks" / "results"

R_FILES = [
    "mixtures_exact_r_unscaled.csv",
    "mixtures_exact_r_scaled.csv",
    "mixtures_noisy_r_unscaled.csv",
    "mixtures_noisy_r_scaled.csv",
]


def step(title):
    print(f"\n{'=' * 72}\n== {title}\n{'=' * 72}", flush=True)


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    dt = time.perf_counter() - t0
    tail = (r.stdout or "")[-1500:]
    if tail.strip():
        print(tail)
    if r.returncode != 0:
        print((r.stderr or "")[-3000:])
    print(f"[{dt:.1f}s] exit={r.returncode}")
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="complete benchmark matrix")
    ap.add_argument("--quick-reps", type=int, default=11, help="timed reps in quick mode")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-benchmarks", action="store_true")
    ap.add_argument("--run-r", metavar="RSCRIPT", default=None,
                    help="path to Rscript; regenerate the R baseline before testing")
    args = ap.parse_args()

    failures = []

    # ---- 1. R baseline -----------------------------------------------------
    step("1/5  R 1.50.0 baseline reference")
    if args.run_r:
        ok = run([args.run_r, str(REPO / "r_baseline" / "run_r_baseline_timed.R"),
                  str(REPO / "test_data"), str(REPO / "r_baseline")], cwd=REPO)
        if not ok:
            failures.append("r_baseline_export")
    missing = [f for f in R_FILES if not (REPO / "r_baseline" / f).exists()]
    if missing:
        print("MISSING R baseline files:", missing)
        print("  -> regenerate with: Rscript r_baseline/run_r_baseline_timed.R <repo_root>")
        failures.append("r_baseline_missing")
    else:
        print("R baseline reference CSVs present:", ", ".join(R_FILES))

    # ---- 2. pytest ----------------------------------------------------------
    if not args.skip_tests:
        step("2/5  pytest suite (validation / API / accuracy gates / R regression)")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
        ok = run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
                 cwd=REPO, env=env)
        if not ok:
            failures.append("pytest")
    else:
        step("2/5  pytest suite — SKIPPED")

    # ---- 3. benchmarks ------------------------------------------------------
    if not args.skip_benchmarks:
        if args.full:
            step("3/5  FULL benchmark matrix (run_benchmarks.py)")
            ok = run([sys.executable, str(REPO / "benchmarks" / "run_benchmarks.py")], cwd=REPO)
        else:
            step("3/5  quick benchmark (definitive suite, 1 suite rep)")
            ok = run([sys.executable, str(REPO / "benchmarks" / "run_confirmation.py"),
                      "--suite-reps", "1", "--reps", str(args.quick_reps)], cwd=REPO)
        if not ok:
            failures.append("benchmarks")
    else:
        step("3/5  benchmarks — SKIPPED")

    # ---- 4. gate re-verification on the freshly written CSV -----------------
    step("4/5  hard-gate re-verification")
    import pandas as pd

    suite = RESULTS / "confirmation_suite.csv"
    if suite.exists():
        df = pd.read_csv(suite)
        fresh = df[df.suite_rep == df.suite_rep.max()]
        gated = fresh[fresh.impl != "floor_probe"]
        bad = gated[
            ~((gated.finite) & (gated.min_value >= -1e-12)
              & (gated.max_sum_error <= 1e-10)
              & (gated.apply(lambda r: r.max_abs_vs_r <=
                             (1e-7 if r["case"] == "exact" else 1e-6), axis=1)))
        ]
        if len(bad):
            print("GATE FAILURES:\n", bad[["case", "use_scale", "impl", "max_abs_vs_r"]])
            failures.append("gates")
        else:
            print(f"all hard gates pass on the fresh suite ({len(gated)} rows checked)")
        piv = (fresh.pivot_table(index=["case", "use_scale"], columns="impl",
                                 values="median_ms").round(3))
        print("\nfresh medians (ms):\n", piv.to_string())
    else:
        print("confirmation_suite.csv not found — run the benchmarks first")
        failures.append("gates_no_data")

    # ---- 5. summary ---------------------------------------------------------
    step("5/5  summary")
    print("evidence files:")
    for p in sorted(RESULTS.glob("*")):
        print(f"  {p.relative_to(REPO)}  ({p.stat().st_size:,} bytes)")
    print("\nRESULT:", "FAIL — " + ", ".join(failures) if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
