# Repository layout — deconrnaseq-py

Annotated directory tree (as of release 1.50.0). Paths marked `evidence` are
measured artifacts committed with the repo; everything under `results/` is
regenerable with `python reproduce.py --full` (see `REPRODUCING.md`).

```
deconrnaseq_py/
│
├── pyproject.toml                # build config (setuptools, src-layout); extras: [plots], [test], [rust]
├── MANIFEST.in                   # sdist file rules (tests/benchmarks/data ship; build dirs pruned)
├── README.md                     # install, quickstart, API, accuracy gates, performance summary
├── REPRODUCING.md                # reviewer guide: R baseline export -> tests -> one-click benchmark rerun
├── REPOSITORY.md                 # this file
├── reproduce.py                  # one-click acceptance: R-baseline check -> pytest -> benchmark -> gate re-check
├── LICENSE                       # GPL-2.0 verbatim (same licence as upstream R package)
├── NOTICE                        # attribution: original authors Gong & Szustakowski, Bioconductor 3.21 source
├── .gitignore
│
├── src/deconrnaseq/              # THE PACKAGE (src layout)
│   ├── __init__.py               #   public API + version (mirrors upstream 1.50.0)
│   ├── core.py                   #   deconrnaseq() entry, DeconResult, deconvolve_core() perf-critical path
│   ├── solvers.py                #   simplex-constrained LS solvers: interior-first batched enumeration
│   │                             #   (default, K<=12), exact enum, batched enum, active-set (cold/warm);
│   │                             #   rust_available() backend probe; all solve the identical float64 QP
│   ├── _scale.py                 #   exact re-implementation of R scale() (center=mean, sd with ddof=1)
│   ├── _validate.py              #   input validation + gene-id alignment preserving signature order
│   ├── pca.py                    #   mixture PCA diagnostic (R out.pca; optional-module, outside core timing)
│   ├── diagnostics.py            #   checksig condition-number curve (optional module)
│   ├── plots.py                  #   matplotlib figures (optional; only with [plots] extra + fig=True)
│   └── py.typed                  #   PEP 561 marker
│
├── rust/                         # OPTIONAL compiled backend (separate maturin project; NOT required)
│   ├── Cargo.toml                #   pyo3 0.22 / numpy 0.22 / ndarray 0.16 (locked versions)
│   ├── Cargo.lock                #   pinned dependency graph for reproducible builds
│   ├── src/lib.rs                #   one-call interior KKT + exact boundary-enumeration kernel
│   └── target/wheels/            #   (gitignored except the shipped prebuilt wheel for reviewers:
│                                 #    deconrnaseq_rust-1.50.0-cp311-cp311-win_amd64.whl)
│
├── tests/                        # 52 pytest tests; optional extras may be explicitly skipped
│   ├── conftest.py               #   fixtures; resolves test_data/ in-repo first, sibling layout as fallback
│   ├── test_validation.py        #   rejection: NA/inf/negatives/duplicate genes/no common genes/genes<K
│   ├── test_api.py               #   output structure, naming, alignment, known_prop RMSE, edge dims
│   ├── test_accuracy_exact.py    #   exact-vs-truth hard gates (max_abs<=1e-8, RMSE<=1e-9, sum-to-1, nonneg)
│   ├── test_r_regression.py      #   every solver x backend vs the R 1.50.0 reference CSVs (4 case x scale)
│   └── smoke_install.py          #   post-install smoke: run with the venv python against the INSTALLED wheel
│
├── benchmarks/
│   ├── bench_variants.py         #   candidate implementations (baseline_loop ... final_rust, floor_probe)
│   ├── benchmark.py              #   timing worker: one config in a fresh subprocess, JSON to stdout;
│   │                             #   warmup 3 + interleaved timed reps; tracemalloc OUTSIDE the timed path
│   ├── run_benchmarks.py         #   full matrix driver: 4 case x scale x {1,8,48} samples + synthetic
│   │                             #   480/4800 (perf only) -> results/benchmark_results.csv
│   ├── run_confirmation.py       #   definitive suite driver: 3 round-robin suite reps, 48 samples
│   │                             #   -> results/confirmation_suite.csv
│   ├── REPORT.md                 #   methodology + result tables + the honest Rust statement
│   └── results/                  #   evidence (committed; regenerable)
│       ├── confirmation_suite.csv    # 72 rows: definitive core-solve suite (3 suite reps)
│       ├── benchmark_results.csv     # 122 rows: full matrix incl. n=1/8/48 and synthetic 480/4800
│       ├── full_call_timing.csv      # end-to-end deconrnaseq() vs R full_function
│       ├── accuracy_results.csv      # every hard gate, measured values, pass flags
│       ├── rust_scratch_buffer_ab.csv # 5-round old/new Rust A/B; bitwise identity + timing summary
│       ├── rust_interior_kkt_ab.csv  # 5-round complete Rust-KKT A/B; speed + precision evidence
│       └── profile_baseline_loop.txt # cProfile: why the faithful baseline is slow (12,240 tiny solves/run)
│
├── examples/
│   └── minimal_example.py        # 3-block quickstart: deconvolve / RMSE vs truth / solver swap
│
├── test_data/                    # synthetic validation set (5000 genes x 8 cell types x 48 samples,
│   │                             #  generator seed 20260814; README.md inside documents generation)
│   ├── reference_signatures.csv  #   genes x cell types
│   ├── mixtures_exact.csv        #   noise-free mixtures
│   ├── mixtures_noisy.csv        #   noisy mixtures (R regression target)
│   ├── true_proportions.csv      #   generator ground truth (48 x 8)
│   ├── gene_metadata.csv / sample_metadata.csv
│   ├── manifest.json             #   dataset id deconrnaseq_synthetic_v1, seeds, hashes
│   ├── validate_dataset.py       #   independent dataset validator
│   └── run_r_baseline.R          #   minimal R export (superseded by r_baseline/run_r_baseline_timed.R)
│
└── r_baseline/                   # R 1.50.0 reference outputs (the regression target)
    ├── run_r_baseline_timed.R    #   export + 3-scope timing script (see REPRODUCING.md §2)
    ├── mixtures_exact_r_unscaled.csv / mixtures_exact_r_scaled.csv
    ├── mixtures_noisy_r_unscaled.csv / mixtures_noisy_r_scaled.csv
    └── r_timing.csv              #   R timings: full_function / core_df / core_matrix, median/p95, 11 reps

NOT committed (built locally, or throwaway):
├── dist/                         # built wheel + sdist (rebuild: python -m build)
├── .venv_verify/                 # clean-room venv used by the acceptance run
├── build/, **/__pycache__/
└── rust/target/                  # cargo/maturin build tree (keeps only the shipped wheel)
```

## Conventions

- **src layout**: the importable package is only `src/deconrnaseq/`; tests and
  benchmarks import it either installed (wheel) or via `src` on `PYTHONPATH`.
- **Data resolution**: `tests/conftest.py`, `benchmarks/run_*.py`,
  `examples/` and `tests/smoke_install.py` all prefer the in-repo
  `test_data/` + `r_baseline/`; the development-time sibling layout
  (`../test_data`) still works as a fallback.
- **Evidence is committed**: `benchmarks/results/` and `r_baseline/*.csv` are
  part of the release so reviewers can diff their rerun against them.
- **GPL-2**: the port is a derivative of the R package; `LICENSE` is the
  verbatim licence text and `NOTICE` carries attribution — both ship in the
  wheel and sdist.
