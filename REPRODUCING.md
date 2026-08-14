# Reproduction guide for reviewers — deconrnaseq-py 1.50.0

Everything below runs from the repository root and writes **only inside the
repository**. Three things are verified, in order:

1. **Accuracy gates** — the Python port reproduces the R 1.50.0 reference
   outputs to float64 accuracy (and the generator truth where the port spec
   gates it).
2. **Tests** — 49 pytest tests (input validation, API, exact gates,
   R regression for every solver × backend).
3. **Performance claims** — the benchmark tables in `benchmarks/REPORT.md`
   are regenerated from scratch on your machine.

Expected wall time: quick acceptance **~3–5 minutes**; full matrix
**~15–30 minutes** (the faithful pure-Python baseline is intentionally slow).

---

## 0. Prerequisites

- **Python ≥ 3.9** (reference runs used CPython 3.11.3 on Windows 10;
  Linux/macOS work identically — the core is pure NumPy/Pandas)
- **NumPy ≥ 1.22, pandas ≥ 1.4, pytest ≥ 7** for testing
- Optional: **R ≥ 4.1** with DeconRNASeq **1.50.0** installed — only needed if
  you want to *regenerate* the R baseline; the exported reference CSVs already
  ship in `r_baseline/`, so R is **not** required to review
- Optional: the prebuilt Rust wheel (CPython 3.11 / win_amd64) if you want to
  verify the accelerated backend; the pure-Python path is the default and is
  what the accuracy gates are stated against

Install the package itself:

```bash
pip install .                 # from the repository root (pure Python)
# or the built wheel:
pip install dist/deconrnaseq_py-1.50.0-py3-none-any.whl
# optional acceleration (auto-detected; silent fallback to NumPy when absent):
pip install rust/target/wheels/deconrnaseq_rust-1.50.0-cp311-cp311-win_amd64.whl
```

Sanity check:

```bash
python -c "import deconrnaseq as drs; print(drs.__version__, drs.rust_available())"
```

---

## 1. The data and the R baseline

`test_data/` ships the synthetic validation set (5000 genes × 8 cell types ×
48 samples; `manifest.json` records generator seeds and hashes;
`test_data/README.md` documents generation; `validate_dataset.py` is an
independent validator).

`r_baseline/` ships the R 1.50.0 reference outputs the port is gated against:

| file | what |
|---|---|
| `mixtures_exact_r_unscaled.csv` / `..._scaled.csv` | R `DeconRNASeq()` on exact mixtures, both `use.scale` modes |
| `mixtures_noisy_r_unscaled.csv` / `..._scaled.csv` | same on noisy mixtures |
| `r_timing.csv` | R timings, 3 scopes (`full_function`, `core_df`, `core_matrix`), median/p95 over 11 reps |

### (Optional) Regenerating the R baseline

Only if you want to verify the baseline itself rather than trust the shipped
CSVs. Requires R with DeconRNASeq **1.50.0** (Bioconductor 3.21 source
tarball; the original `DESCRIPTION` records GPL-2 and the exact source):

```bash
# Windows example — any Rscript works:
Rscript r_baseline/run_r_baseline_timed.R test_data r_baseline 11
```

The script (a) runs the official `DeconRNASeq()` on both mixture sets × both
scale modes and writes the four reference CSVs, (b) times three scopes and
writes `r_timing.csv`. It prints `sessionInfo()` and BLAS/LAPACK versions so
the environment is recorded. Diff against the shipped CSVs — they should be
identical up to BLAS-dependent last-bit noise (`max |Δ|` far below the
1e-7/1e-6 gates).

---

## 2. One-click acceptance

```bash
python reproduce.py
```

This runs, in order, with a printed PASS/FAIL per step:

1. **R-baseline check** — verifies the four reference CSVs exist
   (`--run-r "path/to/Rscript"` regenerates them first).
2. **pytest** — the full suite in a subprocess (subprocess matters: an
   interactive kernel may hold stale modules; see §5).
3. **Quick benchmark** — the definitive suite, one round: implementations
   `baseline_loop, enum_batched, activeset_loop_warm, final_numpy,
   final_rust, floor_probe` × 4 case/scale configs × 48 samples, 3 warmup +
   11 interleaved timed reps, 1 BLAS thread, tracemalloc outside the timed
   path. Writes a fresh `benchmarks/results/confirmation_suite.csv`.
4. **Gate re-verification** — re-reads the *fresh* CSV and re-checks every
   hard gate (R-regression 1e-7/1e-6, sum-to-1 ≤ 1e-10, non-negativity,
   finiteness) rather than trusting the run's own pass flags.
5. **Summary** — lists every evidence file with sizes and prints
   `RESULT: PASS` or the failed step names.

Flags: `--full` (complete matrix: n=1/8/48 plus synthetic 480/4800
performance-only sets — regenerates `benchmark_results.csv`),
`--skip-tests`, `--skip-benchmarks`, `--quick-reps N`.

Exit code is 0 only if every step passed — usable as CI.

---

## 3. What you should see

**Tests**: `49 passed` (validation rejections, alignment, API structure,
exact-vs-truth gates, R regression for all five solvers × NumPy/Rust
backends).

**Accuracy** (`benchmarks/results/accuracy_results.csv`; regenerated numbers
land in the fresh confirmation suite):

| gate | threshold | measured on the reference machine |
|---|---|---|
| max abs diff vs R, exact (both scales, both backends) | ≤ 1e-7 | ~1.5e-14 |
| max abs diff vs R, noisy (both scales, both backends) | ≤ 1e-6 | ~4.9e-15 |
| exact+unscaled vs truth: max abs / RMSE | ≤ 1e-8 / ≤ 1e-9 | 4.9e-13 / 1.2e-13 |
| row-sum-to-1 error | ≤ 1e-10 | ≤ 2.2e-14 |
| min value | ≥ -1e-12 | 0.0 |

**Performance** (medians on the reference machine; your absolute numbers will
differ with hardware, the *ordering and the R speedups* should not):

| case | scale | R core_df | baseline_loop | final_numpy | final_rust |
|---|---|---|---|---|---|
| exact | off | 50 ms | 216.9 ms | 0.189 ms | 0.167 ms |
| exact | on  | 70 ms | 198.0 ms | 1.916 ms | 1.298 ms |
| noisy | off | 70 ms | 197.4 ms | 1.040 ms | 0.474 ms |
| noisy | on  | 100 ms | 187.9 ms | 2.037 ms | 1.330 ms |

Cross-checks to try if numbers look off:

- `floor_probe` (a solve-free control) should sit at or just below
  `final_rust` on exact/unscaled — if it is far above, the machine is noisy;
  rerun with `--quick-reps 21`.
- `final_numpy` must beat `enum_batched`, which must beat `baseline_loop` by
  two orders of magnitude.
- p95/median should be close (< 1.3×) on a quiet machine.

---

## 4. Verifying an installed wheel in isolation

`tests/smoke_install.py` exercises the **installed** package (no source tree
on the path) against the shipped data and R baselines:

```bash
python -m venv .venv_verify
.venv_verify/Scripts/python -m pip install dist/deconrnaseq_py-1.50.0-py3-none-any.whl numpy pandas
.venv_verify/Scripts/python tests/smoke_install.py
# then install the Rust wheel into the same venv and rerun to verify
# auto-detection and rust==numpy agreement:
.venv_verify/Scripts/python -m pip install rust/target/wheels/deconrnaseq_rust-1.50.0-cp311-cp311-win_amd64.whl
.venv_verify/Scripts/python tests/smoke_install.py
```

Expected: `SMOKE OK — rust_available = False`, then `True` after the Rust
wheel install, with `rust==numpy` passing.

---

## 5. Troubleshooting

- **pytest collects zero tests / import errors from an interactive session** —
  always run pytest as a subprocess (`python -m pytest tests`) from the repo
  root, never in-process from a long-lived kernel that may have imported an
  older `deconrnaseq`.
- **`backend="rust"` raises `ModuleNotFoundError`** — expected when the wheel
  is not installed; use `backend="auto"` (default) for automatic fallback.
- **Slow benchmark** — the pure-Python `baseline_loop` is intentionally the
  faithful slow reference (~0.2 s per run × reps × configs); `--skip-benchmarks`
  if you only need correctness.
- **BLAS threads** — the harness pins 1 thread inside its worker; timing on a
  busy machine widens p95 but does not change medians materially.
- **R regeneration diffs** — different BLAS builds move the last bits;
  anything under 1e-7 is the gate, typical drift is ≤ 1e-12.

## 6. Where every claim lives

| claim in README / REPORT | evidence file | regenerate |
|---|---|---|
| accuracy gates | `benchmarks/results/accuracy_results.csv` + fresh `confirmation_suite.csv` | `python reproduce.py` |
| core-solve speed table | `benchmarks/results/confirmation_suite.csv` | `python benchmarks/run_confirmation.py` |
| full matrix + n-sweep + synthetic | `benchmarks/results/benchmark_results.csv` | `python benchmarks/run_benchmarks.py` |
| end-to-end call speedups | `benchmarks/results/full_call_timing.csv` | see `benchmarks/REPORT.md` methodology |
| why the baseline is slow | `benchmarks/results/profile_baseline_loop.txt` | cProfile on `bench_variants.baseline_loop` |
| R reference numbers | `r_baseline/*.csv` | §1 above |
| test suite green | pytest output | `python -m pytest tests -q` |
