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
is absent. To compile the optional Rust backend locally, install Rust and
Maturin, then run:

```bash
python -m pip install maturin
maturin develop --release --manifest-path rust/Cargo.toml
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

The accepted reference run passes 49 tests. On the shipped 5,000-gene,
8-cell-type, 48-sample data, the exact unscaled solution has maximum absolute
error about `4.9e-13` against generator truth. Python/R regression thresholds
are `1e-7` for exact mixtures and `1e-6` for noisy mixtures.

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
