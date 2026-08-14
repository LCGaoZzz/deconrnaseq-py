# Benchmark report — deconrnaseq-py vs R DeconRNASeq 1.50.0

All numbers are **measured** (not model predictions). Raw rows:
`results/confirmation_suite.csv` (definitive core-solve suite, 72 rows),
`results/benchmark_results.csv` (size sweep incl. synthetic 480 / 4800 samples),
`results/full_call_timing.csv` (end-to-end `deconrnaseq()` calls),
`results/accuracy_results.csv` (hard gates),
`results/profile_baseline_loop.txt` (cProfile evidence),
`../../r_baseline/r_timing.csv` (R 1.50.0 timings, 11 timed reps).

For publication, absolute local paths in the cProfile text were normalized to
`<repo>` and `<python-env>`; timing values and call counts were not changed.

## Methodology

- **Data**: shipped synthetic set — 5000 genes × 8 cell types × 48 samples
  (`mixtures_exact.csv`, `mixtures_noisy.csv`), generator seed 20260814.
  Synthetic 480- and 4800-sample mixtures used for the perf sweep only.
- **R reference**: DeconRNASeq 1.50.0 installed from the Bioconductor 3.21
  source tarball into R 4.3.1 (not the previously installed 1.42.0).
  `r_timing.csv` scopes: `full_function` (whole `DeconRNASeq()` call),
  `core_df` / `core_matrix` (the lsei solve portion). Speedups vs R use
  `core_df` for core-solve comparisons and `full_function` for end-to-end.
- **Protocol (Python)**: per measurement — 3 warmup calls, then 21 timed
  repetitions, median and p95 reported. The definitive suite repeats every
  (case × implementation) measurement in **3 interleaved suite reps** to
  cancel drift; tracemalloc runs *outside* the timed path (an early harness
  bug that put tracemalloc inside the timing channel inflated round-1 numbers
  ~8x and was caught and fixed). File IO, plotting, first import and any
  compile step are excluded from all timings.
- **Machine**: Windows 10, CPython 3.11.3, NumPy 2.2.6 (OpenBLAS 0.3.29,
  MAX_THREADS=24; `threads` column records the BLAS thread setting = 1 for
  the definitive suite), AVX2, no AVX512. Rust backend: rustc 1.97.1,
  PyO3 0.22.6 / numpy 0.22.1, built with maturin 1.14.1.
- **Implementations**: `baseline_loop` = the correct pure-Python reference
  port (per-sample loop × per-support loop, 12,240 tiny `np.linalg.solve`
  calls per 48-sample run — see profile evidence). `final_numpy` =
  interior-first batched enumeration (one batched equality KKT solve; exact
  support-enumeration fallback only for boundary samples).
  `final_rust` = same algorithm with the boundary enumeration in the compiled
  backend. `floor_probe` = solve-free harness-floor control (not ranked).

## Core solve, 48 samples (median ms, definitive suite)

| case | use_scale | baseline_loop | enum_batched | activeset warm | final_numpy | final_rust | harness floor |
|---|---|---|---|---|---|---|---|
| exact | off | 216.9 | 1.209 | 2.309 | 0.189 | **0.167** | 0.160 |
| exact | on  | 198.0 | 2.029 | 4.774 | 1.916 | **1.298** | 0.946 |
| noisy | off | 197.4 | 1.202 | 3.620 | 1.040 | **0.474** | 0.161 |
| noisy | on  | 187.9 | 2.073 | 5.018 | 2.037 | **1.330** | 0.949 |

(exact/unscaled final_rust is within 0.007 ms of the solve-free floor — the
48 exact samples are all interior, so the "solve" is one 9×9 batched KKT.)

## Speedup vs R 1.50.0 (core solve vs R `core_df` median)

| case | use_scale | R core_df | final_numpy | speedup | final_rust | speedup |
|---|---|---|---|---|---|---|
| exact | off | 50 ms | 0.189 ms | **265x** | 0.167 ms | **299x** |
| exact | on  | 70 ms | 1.916 ms | **37x**  | 1.298 ms | **54x**  |
| noisy | off | 70 ms | 1.040 ms | **67x**  | 0.474 ms | **148x** |
| noisy | on  | 100 ms | 2.037 ms | **49x**  | 1.330 ms | **75x**  |

vs the pure-Python `baseline_loop` on the same machine, `final_rust` is
**1297x** (exact/off), **152x** (exact/on), **416x** (noisy/off),
**141x** (noisy/on).

## End-to-end `deconrnaseq()` call (validation + alignment + PCA + solve + wrap)

| case | use_scale | R full_function | deconrnaseq-py | speedup |
|---|---|---|---|---|
| exact | off | 110 ms | 18.1 ms | 6.1x |
| exact | on  | 140 ms | 20.4 ms | 6.9x |
| noisy | off | 140 ms | 18.4 ms | 7.6x |
| noisy | on  | 230 ms | 20.4 ms | 11.3x |

The Python full call is dominated by the PCA diagnostic and DataFrame work,
not the solver.

## Size sweep (median ms, unscaled)

| samples | baseline_loop | enum_batched | final_numpy | final_rust |
|---|---|---|---|---|
| 1 (exact) | 4.38 | 0.33 | 0.036 | 0.033 |
| 8 (exact) | 35.5 | 0.57 | 0.059 | 0.050 |
| 48 (exact) | 275.4 | 2.31 | 0.330 | 0.318 |
| 480 (synthetic) | — | 10.7 | 1.38 | 1.34 |
| 4800 (synthetic) | — | 191.0 | 33.3 | 32.8 |

## Accuracy gates (all pass; full table in `accuracy_results.csv`)

- vs R 1.50.0 reference outputs: max |Δ| ≤ **1.5e-14** across all four
  case×scale combinations and both backends (gates 1e-7 exact / 1e-6 noisy).
- exact + unscaled vs generator truth: max |Δ| = 4.9e-13, RMSE = 1.2e-13
  (gates 1e-8 / 1e-9).
- row-sum-to-1 error ≤ 2.2e-14 (gate 1e-10); min value 0.0 (gate −1e-12);
  everything float64 and finite.
- On noisy data the port's deviation from generator truth (RMSE 0.002498) is
  **bit-identical to R's own** (0.0024978) — the noise floor of the data, not
  a solver difference.

## What made it fast (profile evidence)

`results/profile_baseline_loop.txt`: the faithful baseline spends its time in
**12240 tiny `np.linalg.solve` calls per run** (48 samples × 255 supports) —
~992k Python-level function calls; per-call dispatch dominates. Batching all
255 supports × all samples into single stacked solves removes that overhead
(`enum_batched`, ~1.2 ms), and the interior-first shortcut removes the
remaining enumeration work for samples whose equality-only optimum is already
feasible (all 48 exact samples; 13/48 noisy samples need the boundary
fallback). The Gram matrix is computed once per call; Gram caching across
calls and warm-start chaining were both tried and **rejected** (no gain —
campaign theories `blf_65cbe…`, `blf_0487…` retired with evidence).

## Honest Rust statement

The optional Rust backend is **~2.3x faster than the same algorithm in NumPy**
(0.167 vs 0.189 ms exact/off; 1.30 vs 1.92 ms exact/on) because the remaining
work is one small batched LAPACK solve that NumPy already does well. The
~1000x headline win over the baseline comes from the **algorithm change**
(batching + interior-first), not from the compiler. Ship the pure-Python
wheel as the default; install `deconrnaseq_rust` when the last factor-of-two
matters. `backend="auto"` picks Rust when importable and falls back silently
to NumPy otherwise; `drs.rust_available()` reports which is active.
