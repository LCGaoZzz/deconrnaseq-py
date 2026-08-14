"""Benchmark worker: times ALL selected variants, interleaved, in one process.

Interleaving (round-robin over variants within each repetition) cancels
machine-level drift, which between separate processes was observed to exceed
2x.  Timing excludes file reading, imports and warmup.  Memory is measured in
a separate untimed pass so tracemalloc overhead never contaminates timings.
Accuracy versus the R 1.50.0 reference CSVs and the generator truth is
measured in the same run; implementations failing the gates are flagged and
excluded from any speed ranking downstream.

Usage:
    python benchmark.py --impls enum_batched,activeset_loop_warm --case exact \
        --scale off --samples 48 --reps 21 --threads 1 \
        --data-dir <test_data> --r-dir <r_baseline> --pkg-src <src>
Prints one JSON object: {impl: {metrics...}}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--impls", default="all")
    p.add_argument("--case", choices=["exact", "noisy"], required=True)
    p.add_argument("--scale", choices=["on", "off"], required=True)
    p.add_argument("--samples", default="48")  # int or "all"
    p.add_argument("--reps", type=int, default=21)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--r-dir", required=True)
    p.add_argument("--pkg-src", required=True)
    p.add_argument("--large-n", type=int, default=0,
                   help="if >0: deterministic synthetic perf-only set with this many samples")
    p.add_argument("--skip", default="", help="comma-separated impls to skip (e.g. slow baseline for large-n)")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(args.threads)
    sys.path.insert(0, args.pkg_src)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import time
    import tracemalloc

    import numpy as np
    import pandas as pd
    import psutil

    from bench_variants import VARIANTS

    skip = set(args.skip.split(",")) if args.skip else set()
    impl_names = ([n for n in VARIANTS if n not in skip] if args.impls == "all"
                  else [n for n in args.impls.split(",") if n not in skip])

    # ---------------- data (loaded OUTSIDE the timed region) ----------------
    sig = pd.read_csv(os.path.join(args.data_dir, "reference_signatures.csv"), index_col=0)
    truth = pd.read_csv(os.path.join(args.data_dir, "true_proportions.csv"), index_col=0)
    A = sig.to_numpy(np.float64)

    if args.large_n > 0:
        rng = np.random.default_rng(20260814)
        P = rng.dirichlet(np.ones(A.shape[1]), size=args.large_n)
        Y_full = A @ P.T
        r_ref = None
        truth_full = P
    else:
        mix = pd.read_csv(os.path.join(args.data_dir, f"mixtures_{args.case}.csv"), index_col=0)
        Y_full = mix.to_numpy(np.float64)
        suffix = "scaled" if args.scale == "on" else "unscaled"
        r_ref = pd.read_csv(
            os.path.join(args.r_dir, f"mixtures_{args.case}_r_{suffix}.csv"), index_col=0
        ).to_numpy(np.float64)
        truth_full = truth.to_numpy(np.float64)

    n_samples = Y_full.shape[1] if args.samples == "all" else int(args.samples)
    Y = np.ascontiguousarray(Y_full[:, :n_samples])
    use_scale = args.scale == "on"

    # ---------------- accuracy (untimed, once per impl) ---------------------
    results = {}
    fns = {}
    for name in impl_names:
        fn = VARIANTS[name]
        fns[name] = fn
        est = fn(A, Y, use_scale)
        acc = {
            "finite": bool(np.isfinite(est).all()),
            "min_value": float(est.min()),
            "max_sum_error": float(np.abs(est.sum(axis=1) - 1.0).max()),
            "max_abs_vs_truth": float(np.abs(est - truth_full[:n_samples]).max()),
            "rmse_vs_truth": float(np.sqrt(((est - truth_full[:n_samples]) ** 2).mean())),
        }
        if r_ref is not None:
            acc["max_abs_vs_r"] = float(np.abs(est - r_ref[:n_samples]).max())
        results[name] = acc

    # ---------------- warmup ------------------------------------------------
    for _ in range(args.warmup):
        for name in impl_names:
            fns[name](A, Y, use_scale)

    # ---------------- interleaved timing ------------------------------------
    times = {n: [] for n in impl_names}
    for _ in range(args.reps):
        for name in impl_names:
            t0 = time.perf_counter()
            fns[name](A, Y, use_scale)
            t1 = time.perf_counter()
            times[name].append(t1 - t0)

    # ---------------- memory pass (untimed, tracemalloc) --------------------
    proc = psutil.Process()
    for name in impl_names:
        tracemalloc.start()
        tracemalloc.clear_traces()
        rss_before = proc.memory_info().rss
        fns[name](A, Y, use_scale)
        _, peak = tracemalloc.get_traced_memory()
        peak_rss = proc.memory_info().rss
        tracemalloc.stop()
        results[name]["peak_tracemalloc_bytes"] = int(peak)
        results[name]["rss_delta_bytes"] = int(peak_rss - rss_before)

    for name in impl_names:
        t = np.asarray(times[name])
        results[name].update({
            "impl": name,
            "case": args.case,
            "use_scale": use_scale,
            "n_samples": int(n_samples),
            "threads": args.threads,
            "reps": args.reps,
            "median_ms": float(np.median(t) * 1e3),
            "p95_ms": float(np.quantile(t, 0.95) * 1e3),
            "min_ms": float(t.min() * 1e3),
            "max_ms": float(t.max() * 1e3),
        })
    print(json.dumps(results))


if __name__ == "__main__":
    main()
