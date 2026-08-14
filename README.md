# deconrnaseq-py

[![Python >=3.9](https://img.shields.io/badge/python-%3E%3D3.9-blue.svg)](https://www.python.org/)
[![License: GPL-2.0](https://img.shields.io/badge/license-GPL--2.0-green.svg)](LICENSE)

Fast, R-free Python reimplementation of Bioconductor DeconRNASeq 1.50.0 for
estimating cell-type proportions from bulk RNA-seq mixtures.

The package provides a NumPy/Pandas implementation and an optional Rust
backend. R is not required for normal installation or use. The original
DeconRNASeq authors and GPL-2.0 license are acknowledged in [NOTICE](NOTICE).

## What it computes

For each mixture sample `y` and signature matrix `A`, the package solves:

```text
minimize   ||A x - y||^2
subject to x >= 0 and sum(x) = 1
```

Inputs are gene-by-sample and gene-by-cell-type matrices. Output rows are
mixture samples, output columns are cell types, and every row sums to one.

## Installation

### Install directly from GitHub

```bash
python -m pip install "git+https://github.com/LCGaoZzz/deconrnaseq-py.git"
```

### Clone and install locally

```bash
git clone https://github.com/LCGaoZzz/deconrnaseq-py.git
cd deconrnaseq-py

python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux or macOS
source .venv/bin/activate
```

Then install:

```bash
python -m pip install --upgrade pip
python -m pip install .
```

Optional plotting and test dependencies:

```bash
python -m pip install ".[plots]"
python -m pip install ".[test]"
```

### Optional Rust acceleration

The NumPy backend is complete and is used automatically when the Rust module
is absent. The extension is packaged separately as `deconrnaseq-rust` and
installs the importable module `deconrnaseq_rust`; it can therefore coexist
with the pure-Python `deconrnaseq-py` distribution without replacing its
package metadata.

To compile and install both distributions locally, install Rust and run these
commands from the repository root:

```bash
python -m pip install .
python -m pip install ./rust
```

For a release wheel instead of an in-place installation:

```bash
python -m pip install "maturin>=1.7,<2"
cd rust
python -m maturin build --release
python -m pip install target/wheels/deconrnaseq_rust-*.whl
```

Confirm the installed version and backend:

```bash
python -c "import deconrnaseq as drs; print(drs.__version__, drs.rust_available())"
```

## Quick start

Prepare two CSV files with gene identifiers in the first column:

- `reference_signatures.csv`: genes x cell types
- `mixtures.csv`: genes x mixture samples

Gene order may differ. The package aligns shared genes by identifier while
preserving signature-matrix order.

```python
import pandas as pd
import deconrnaseq as drs

signatures = pd.read_csv("reference_signatures.csv", index_col=0)
mixtures = pd.read_csv("mixtures.csv", index_col=0)

result = drs.deconrnaseq(
    datasets=mixtures,
    signatures=signatures,
    use_scale=True,
    fig=False,
)

proportions = result.out_all
print(proportions.head())
print(proportions.sum(axis=1))  # each row is approximately 1.0
```

The returned `DeconResult` supports both Python-style attributes and R-style
keys:

```python
result.out_all
result["out.all"]
result.out_pca
result["out.pca"]
```

### Compare with known proportions

Known proportions must be a samples-by-cell-types DataFrame with matching row
and column names:

```python
truth = pd.read_csv("true_proportions.csv", index_col=0)

result = drs.deconrnaseq(
    datasets=mixtures,
    signatures=signatures,
    proportions=truth,
    known_prop=True,
    use_scale=False,
    fig=False,
)

print("Mean RMSE:", result.out_rmse)
print("RMSE by cell type:")
print(result.rmse_per_type)
```

### NumPy input

NumPy arrays are supported when labels are supplied separately:

```python
result = drs.deconrnaseq(
    datasets=mixture_array,
    signatures=signature_array,
    dataset_genes=mixture_gene_ids,
    signature_genes=signature_gene_ids,
    sample_names=sample_names,
    cell_type_names=cell_type_names,
    use_scale=True,
    fig=False,
)
```

### Performance-critical core API

If inputs are already validated and gene-aligned, the lower-level API avoids
Pandas alignment and PCA diagnostics:

```python
X = drs.deconvolve_core(
    signature_array,  # shape: genes x cell types
    mixture_array,    # shape: genes x samples
    use_scale=False,
    solver="auto",
    backend="auto",
)
# X shape: samples x cell types
```

## Solver and backend selection

```python
result = drs.deconrnaseq(
    mixtures,
    signatures,
    solver="auto",
    backend="auto",
    fig=False,
)
```

- `solver="auto"`: recommended; selects the exact solver strategy.
- `backend="auto"`: use Rust when installed, otherwise NumPy.
- `backend="numpy"`: force the portable NumPy implementation.
- `backend="rust"`: require the compiled Rust module.
- `drs.rust_available()`: report whether the Rust backend is importable.

All production solvers use float64 and solve the same constrained problem.
Approximate algorithms and float32 are not used by default.

## Benchmark results

All values below are measured on the repository's fixed synthetic benchmark:
5,000 genes, 8 cell types, and 48 samples. The reference implementation is
Bioconductor DeconRNASeq 1.50.0 running under R 4.3.1. Python measurements use
Windows 10, CPython 3.11.3, NumPy 2.2.6/OpenBLAS with one BLAS thread, and a
Rust release build from rustc 1.97.1. File I/O, first import, and compilation
are excluded. Results on other machines will vary.

### Accuracy versus the original R package

The Python/Rust implementation remains numerically equivalent to the original
R output. All estimates use float64.

| dataset | scaling | max absolute difference vs R 1.50.0 | acceptance gate | result |
|---|---:|---:|---:|---:|
| exact | off | `1.50e-14` | `1e-7` | pass |
| exact | on | `5.00e-15` | `1e-7` | pass |
| noisy | off | `4.88e-15` | `1e-6` | pass |
| noisy | on | `5.00e-15` | `1e-6` | pass |

For the exact unscaled data, maximum error against the generator truth is
`4.92e-13` and RMSE is `1.18e-13`. Across the four benchmark configurations,
maximum row-sum error is `2.16e-14` and the minimum estimated proportion is
`0.0`. The complete Rust-KKT change differs from the preceding hybrid
Python/Rust path by at most `4.44e-16`, approximately one float64 rounding
unit at this scale. It is numerically equivalent, although not promised to be
bitwise identical on every platform.

### Core solver speed versus R DeconRNASeq 1.50.0

This comparison uses the original R `core_df` median and the current Rust
`deconvolve_core()` median. It measures aligned/scaled matrix preparation plus
the constrained solve, not DataFrame validation, PCA, plotting, or wrapping.

| dataset | scaling | original R core | current Rust core | speedup vs R |
|---|---:|---:|---:|---:|
| exact | off | 50.0 ms | 0.124 ms | **403.6x** |
| exact | on | 70.0 ms | 1.032 ms | **67.8x** |
| noisy | off | 70.0 ms | 0.373 ms | **187.6x** |
| noisy | on | 100.0 ms | 1.067 ms | **93.8x** |

### End-to-end public API speed versus R

The end-to-end measurement includes validation, gene alignment, PCA,
deconvolution, and result wrapping. It is the more representative number for
a normal `deconrnaseq()` call.

| dataset | scaling | original R full call | deconrnaseq-py full call | speedup vs R |
|---|---:|---:|---:|---:|
| exact | off | 110 ms | 18.1 ms | **6.1x** |
| exact | on | 140 ms | 20.4 ms | **6.9x** |
| noisy | off | 140 ms | 18.4 ms | **7.6x** |
| noisy | on | 230 ms | 20.4 ms | **11.3x** |

### Gain from the complete Rust interior-KKT path

The latest Rust change moves the equality KKT solve, feasibility
classification, boundary collection, exact enumeration, and scatter into one
compiled call. Relative to the preceding Python-KKT + Rust-enumeration path:

| dataset | scaling | boundary samples | solver speedup | complete core speedup |
|---|---:|---:|---:|---:|
| exact | off | 0/48 | **3.656x** | **1.081x** |
| exact | on | 39/48 | **1.106x** | **1.036x** |
| noisy | off | 35/48 | **1.112x** | **1.084x** |
| noisy | on | 47/48 | **1.106x** | **1.054x** |

The focused Rust A/B used five alternating-process rounds, one BLAS thread,
100 warmups, and 21 timing blocks per process. See the
[full benchmark report](benchmarks/REPORT.md) and raw evidence for
[Rust KKT A/B](benchmarks/results/rust_interior_kkt_ab.csv),
[accuracy](benchmarks/results/accuracy_results.csv),
[end-to-end timing](benchmarks/results/full_call_timing.csv), and
[R timing](r_baseline/r_timing.csv).

## Input validation

The public API rejects:

- missing, non-finite, or negative values;
- duplicate gene, sample, or cell-type identifiers;
- inputs without common genes;
- fewer aligned genes than cell types;
- zero-variance columns when `use_scale=True`.

`use_scale=True` reproduces R `scale()` behavior: mean centering followed by
sample-standard-deviation scaling with `ddof=1`.

## Reproduce tests and benchmarks

The repository ships its synthetic validation data, R reference CSVs, tests,
benchmark harness, and raw evidence.

Run the complete acceptance workflow:

```bash
python -m pip install ".[test]"
python reproduce.py
```

Run only the tests:

```bash
python -m pytest tests -q
```

The current dependency-minimal reference run passes 51 tests and skips one
optional plotting test. On the shipped 5,000-gene, 8-cell-type, 48-sample
data, the exact unscaled solution has maximum absolute error about `4.9e-13`
against generator truth. Python/R regression thresholds are `1e-7` for exact
mixtures and `1e-6` for noisy mixtures.

See [REPRODUCING.md](REPRODUCING.md) for reviewer instructions and
[benchmarks/REPORT.md](benchmarks/REPORT.md) for methodology, measured timings,
and hardware details. Performance numbers are evidence for the documented
reference machine, not a guarantee for every system.

## Repository layout

```text
src/deconrnaseq/   Python package
rust/              optional PyO3 backend source
tests/             validation, API, accuracy, and R-regression tests
test_data/         synthetic correctness dataset and manifest
r_baseline/        exported R 1.50.0 reference results
benchmarks/        benchmark harness, report, and raw results
examples/          runnable usage example
reproduce.py       one-command acceptance workflow
```

For the fully annotated tree and commit policy, see [REPOSITORY.md](REPOSITORY.md).

## Citation and attribution

If you use this package, cite the original DeconRNASeq work:

> Gong T, Szustakowski JD. DeconRNASeq: a statistical framework for
> deconvolution of heterogeneous tissue samples based on mRNA-Seq data.
> Bioinformatics. 2013;29(8):1083-1085.

Upstream package: [Bioconductor DeconRNASeq](https://bioconductor.org/packages/DeconRNASeq/).

## License

GPL-2.0-only. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
