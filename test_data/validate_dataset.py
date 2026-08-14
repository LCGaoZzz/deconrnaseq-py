"""Validate the synthetic DeconRNASeq dataset without SciPy.

The constrained least-squares reference solver enumerates active sets. This is
practical here because the dataset deliberately has only eight cell types.
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def read_matrix(name: str) -> tuple[list[str], list[str], np.ndarray]:
    with (ROOT / name).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        row_ids: list[str] = []
        values: list[list[float]] = []
        for row in reader:
            row_ids.append(row[0])
            values.append([float(value) for value in row[1:]])
    return row_ids, header[1:], np.asarray(values, dtype=np.float64)


def solve_simplex_least_squares(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve min ||A x - y||², x>=0, sum(x)=1 for all columns of y."""

    gram = a.T @ a
    cross = a.T @ y
    cell_count = a.shape[1]
    sample_count = y.shape[1]
    best_objective = np.full(sample_count, np.inf)
    best_x = np.zeros((sample_count, cell_count))

    for size in range(1, cell_count + 1):
        for support_tuple in combinations(range(cell_count), size):
            support = np.asarray(support_tuple)
            gram_support = gram[np.ix_(support, support)]
            kkt = np.block(
                [
                    [gram_support, np.ones((size, 1))],
                    [np.ones((1, size)), np.zeros((1, 1))],
                ]
            )
            rhs = np.vstack([cross[support, :], np.ones((1, sample_count))])
            try:
                solution = np.linalg.solve(kkt, rhs)
            except np.linalg.LinAlgError:
                continue

            supported_x = solution[:-1, :]
            feasible = np.all(supported_x >= -1e-9, axis=0)
            if not np.any(feasible):
                continue

            full_x = np.zeros((cell_count, sample_count))
            full_x[support, :] = supported_x
            full_x[np.abs(full_x) < 1e-12] = 0.0
            objective = (
                0.5 * np.einsum("is,ij,js->s", full_x, gram, full_x)
                - np.einsum("is,is->s", full_x, cross)
            )
            improve = feasible & (objective < best_objective)
            best_objective[improve] = objective[improve]
            best_x[improve, :] = full_x[:, improve].T

    if not np.isfinite(best_objective).all():
        raise RuntimeError("No feasible solution for one or more samples")
    return best_x


def main() -> None:
    genes_s, cell_types, signatures = read_matrix("reference_signatures.csv")
    genes_e, samples_e, exact = read_matrix("mixtures_exact.csv")
    genes_n, samples_n, noisy = read_matrix("mixtures_noisy.csv")
    sample_ids, proportion_types, truth = read_matrix("true_proportions.csv")

    assert signatures.shape == (5000, 8)
    assert exact.shape == noisy.shape == (5000, 48)
    assert truth.shape == (48, 8)
    assert genes_s == genes_e == genes_n
    assert samples_e == samples_n == sample_ids
    assert cell_types == proportion_types

    for name, matrix in {
        "signatures": signatures,
        "exact": exact,
        "noisy": noisy,
        "truth": truth,
    }.items():
        assert np.isfinite(matrix).all(), f"{name} contains non-finite values"
        assert (matrix >= 0).all(), f"{name} contains negative values"

    identity_error = np.max(np.abs(signatures @ truth.T - exact))
    sum_error = np.max(np.abs(truth.sum(axis=1) - 1.0))
    exact_hat = solve_simplex_least_squares(signatures, exact)
    noisy_hat = solve_simplex_least_squares(signatures, noisy)

    r_scaled = (signatures - signatures.mean(axis=0)) / signatures.std(axis=0, ddof=1)
    print(f"shapes: signatures={signatures.shape}, mixtures={exact.shape}, truth={truth.shape}")
    print(f"max proportion-sum error: {sum_error:.3e}")
    print(f"max exact identity error: {identity_error:.3e}")
    print(f"condition number (unscaled): {np.linalg.cond(signatures):.6f}")
    print(f"condition number (R-scaled): {np.linalg.cond(r_scaled):.6f}")
    print(f"exact max_abs proportion error: {np.max(np.abs(exact_hat - truth)):.3e}")
    print(f"exact proportion RMSE: {np.sqrt(np.mean((exact_hat - truth) ** 2)):.3e}")
    print(f"noisy proportion RMSE: {np.sqrt(np.mean((noisy_hat - truth) ** 2)):.6e}")
    print(f"noisy max_abs proportion error: {np.max(np.abs(noisy_hat - truth)):.6e}")

    assert sum_error <= 2e-12
    assert identity_error <= 5e-9
    assert np.max(np.abs(exact_hat - truth)) <= 1e-8
    assert np.sqrt(np.mean((noisy_hat - truth) ** 2)) <= 0.003
    print("PASS")


if __name__ == "__main__":
    main()

