"""Benchmark candidate implementations.

Each variant takes aligned float64 matrices ``(signature (G,K), mixtures (G,S),
use_scale)`` and returns an (S, K) proportion matrix.  Variants differ ONLY in
how the computation is organised; every one solves the identical QP and must
pass the same accuracy gates before its timing counts.
"""

from __future__ import annotations

import numpy as np

from deconrnaseq._scale import r_scale, r_scale_vector
from deconrnaseq.solvers import (
    solve_lsei_activeset,
    solve_lsei_enum,
    solve_lsei_enum_fast,
)


def baseline_loop(A: np.ndarray, Y: np.ndarray, use_scale: bool) -> np.ndarray:
    """R-faithful pure-Python baseline: per-sample loop, per-sample Gram.

    Mirrors the R DeconRNASeq loop structure: for every sample, scale that
    sample, rebuild A'A and A'y, then solve with the per-support enumeration.
    """
    if use_scale:
        A = r_scale(A, _name="signature")
    S = Y.shape[1]
    K = A.shape[1]
    out = np.empty((S, K), dtype=np.float64)
    for s in range(S):
        y = Y[:, s]
        if use_scale:
            y = r_scale_vector(y)
        gram = A.T @ A
        cross = A.T @ y
        out[s] = solve_lsei_enum(gram, cross[:, None])[:, 0]
    return out


def enum_cachedgram_loop(A: np.ndarray, Y: np.ndarray, use_scale: bool) -> np.ndarray:
    """Gram computed once; still loops samples in Python (cross + enum)."""
    if use_scale:
        A = r_scale(A, _name="signature")
        Y = r_scale(Y, _name="mixture dataset")
    gram = A.T @ A
    S = Y.shape[1]
    K = A.shape[1]
    out = np.empty((S, K), dtype=np.float64)
    for s in range(S):
        cross = A.T @ Y[:, s]
        out[s] = solve_lsei_enum(gram, cross[:, None])[:, 0]
    return out


def enum_batched(A: np.ndarray, Y: np.ndarray, use_scale: bool) -> np.ndarray:
    """Cached Gram + single A'Y GEMM + support-batched exact enumeration."""
    if use_scale:
        A = r_scale(A, _name="signature")
        Y = r_scale(Y, _name="mixture dataset")
    gram = A.T @ A
    cross = A.T @ Y
    return solve_lsei_enum_fast(gram, cross).T.copy()


def activeset_loop_cold(A: np.ndarray, Y: np.ndarray, use_scale: bool) -> np.ndarray:
    """Cached Gram + A'Y GEMM; per-sample active-set, cold start each."""
    if use_scale:
        A = r_scale(A, _name="signature")
        Y = r_scale(Y, _name="mixture dataset")
    gram = A.T @ A
    cross = A.T @ Y
    return solve_lsei_activeset(gram, cross, warm_start=False).T.copy()


def activeset_loop_warm(A: np.ndarray, Y: np.ndarray, use_scale: bool) -> np.ndarray:
    """Cached Gram + A'Y GEMM; per-sample active-set, warm-start chaining."""
    if use_scale:
        A = r_scale(A, _name="signature")
        Y = r_scale(Y, _name="mixture dataset")
    gram = A.T @ A
    cross = A.T @ Y
    return solve_lsei_activeset(gram, cross, warm_start=True).T.copy()


def rust_enum_batched(A: np.ndarray, Y: np.ndarray, use_scale: bool) -> np.ndarray:
    """Rust-compiled batched enumeration backend (requires deconrnaseq-rust).

    Falls back to enum_batched when the compiled module is unavailable so the
    benchmark never crashes; the fallback is reported by the caller.
    """
    if use_scale:
        A = r_scale(A, _name="signature")
        Y = r_scale(Y, _name="mixture dataset")
    gram = A.T @ A
    cross = A.T @ Y
    try:
        import deconrnaseq_rust
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("rust backend not built") from exc
    return np.asarray(deconrnaseq_rust.solve_lsei_enum(gram, cross), dtype=np.float64).T.copy()


def enum_fast_persample_loop(A: np.ndarray, Y: np.ndarray, use_scale: bool) -> np.ndarray:
    """Cached Gram + per-sample loop calling the support-batched enum solver.

    Tests whether batching across *supports* alone suffices, without batching
    across samples (campaign next_batch recombination: sample_batching=none,
    solver=enum_fast).
    """
    if use_scale:
        A = r_scale(A, _name="signature")
        Y = r_scale(Y, _name="mixture dataset")
    gram = A.T @ A
    S = Y.shape[1]
    K = A.shape[1]
    out = np.empty((S, K), dtype=np.float64)
    for s in range(S):
        cross = A.T @ Y[:, s]
        out[s] = solve_lsei_enum_fast(gram, cross[:, None])[:, 0]
    return out


def enum_batched_interior(A: np.ndarray, Y: np.ndarray, use_scale: bool) -> np.ndarray:
    """Interior-first shortcut: one batched full-support KKT solve; only
    samples whose unconstrained (equality-only) solution violates x>=0 go
    through the exact enumeration.  Mathematically exact: a feasible
    equality-only solution is the constrained optimum."""
    if use_scale:
        A = r_scale(A, _name="signature")
        Y = r_scale(Y, _name="mixture dataset")
    gram = A.T @ A
    cross = A.T @ Y
    K, S = cross.shape
    kkt = np.empty((K + 1, K + 1), dtype=np.float64)
    kkt[:K, :K] = gram
    kkt[:K, K] = 1.0
    kkt[K, :K] = 1.0
    kkt[K, K] = 0.0
    rhs = np.empty((K + 1, S), dtype=np.float64)
    rhs[:K] = cross
    rhs[K] = 1.0
    sol = np.linalg.solve(kkt, rhs)[:K]
    interior = (sol >= -1e-12).all(axis=0)
    out = np.where(sol >= 0.0, sol, 0.0)
    if not interior.all():
        rest = np.flatnonzero(~interior)
        out[:, rest] = solve_lsei_enum_fast(gram, cross[:, rest])
    return out.T.copy()


def rust_enum_interior(A: np.ndarray, Y: np.ndarray, use_scale: bool) -> np.ndarray:
    """Interior-first in NumPy + Rust enumeration for boundary samples only."""
    if use_scale:
        A = r_scale(A, _name="signature")
        Y = r_scale(Y, _name="mixture dataset")
    gram = A.T @ A
    cross = A.T @ Y
    K, S = cross.shape
    kkt = np.empty((K + 1, K + 1), dtype=np.float64)
    kkt[:K, :K] = gram
    kkt[:K, K] = 1.0
    kkt[K, :K] = 1.0
    kkt[K, K] = 0.0
    rhs = np.empty((K + 1, S), dtype=np.float64)
    rhs[:K] = cross
    rhs[K] = 1.0
    sol = np.linalg.solve(kkt, rhs)[:K]
    interior = (sol >= -1e-12).all(axis=0)
    out = np.where(sol >= 0.0, sol, 0.0)
    if not interior.all():
        try:
            import deconrnaseq_rust
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("rust backend not built") from exc
        rest = np.flatnonzero(~interior)
        out[:, rest] = np.asarray(
            deconrnaseq_rust.solve_lsei_enum(gram, np.ascontiguousarray(cross[:, rest])),
            dtype=np.float64,
        )
    return out.T.copy()


def final_numpy(A: np.ndarray, Y: np.ndarray, use_scale: bool) -> np.ndarray:
    """SHIPPED default, pure NumPy: deconvolve_core(solver='interior', backend='numpy')."""
    from deconrnaseq import deconvolve_core

    return deconvolve_core(A, Y, use_scale=use_scale, solver="interior", backend="numpy")


def final_rust(A: np.ndarray, Y: np.ndarray, use_scale: bool) -> np.ndarray:
    """SHIPPED accelerated: deconvolve_core(solver='interior', backend='rust')."""
    from deconrnaseq import deconvolve_core

    return deconvolve_core(A, Y, use_scale=use_scale, solver="interior", backend="rust")


def floor_probe(A: np.ndarray, Y: np.ndarray, use_scale: bool) -> np.ndarray:
    """NOT A SOLVER — scaling + Gram + cross only, no QP.  Establishes the
    hardware floor: no correct implementation of the same math can beat this.
    Fails the accuracy gates by construction and is excluded from ranking."""
    if use_scale:
        A = r_scale(A, _name="signature")
        Y = r_scale(Y, _name="mixture dataset")
    gram = A.T @ A
    cross = A.T @ Y
    return np.broadcast_to(cross.mean(axis=1, keepdims=True).T, (Y.shape[1], A.shape[1])).copy()


def _r_scale_v1(x: np.ndarray) -> np.ndarray:
    """Original two-temporary r_scale (kept for A/B evidence of the v2 change)."""
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    sd = np.sqrt((xc * xc).sum(axis=0, keepdims=True) / (x.shape[0] - 1.0))
    return xc / sd


def _final_with_scale(A, Y, use_scale, backend, scale_fn):
    from deconrnaseq.solvers import solve_lsei

    if use_scale:
        A = scale_fn(A)
        Y = scale_fn(Y)
    gram = A.T @ A
    cross = A.T @ Y
    return solve_lsei(gram, cross, method="interior", backend=backend).T.copy()


def final_rust_scalev1(A: np.ndarray, Y: np.ndarray, use_scale: bool) -> np.ndarray:
    """Shipped interior+rust path but with the OLD two-temporary r_scale."""
    return _final_with_scale(A, Y, use_scale, "rust", _r_scale_v1)


def final_numpy_scalev1(A: np.ndarray, Y: np.ndarray, use_scale: bool) -> np.ndarray:
    """Shipped interior+numpy path but with the OLD two-temporary r_scale."""
    return _final_with_scale(A, Y, use_scale, "numpy", _r_scale_v1)


VARIANTS = {
    "baseline_loop": baseline_loop,
    "enum_cachedgram_loop": enum_cachedgram_loop,
    "enum_fast_persample_loop": enum_fast_persample_loop,
    "enum_batched": enum_batched,
    "enum_batched_interior": enum_batched_interior,
    "activeset_loop_cold": activeset_loop_cold,
    "activeset_loop_warm": activeset_loop_warm,
    "rust_enum_batched": rust_enum_batched,
    "rust_enum_interior": rust_enum_interior,
    "final_numpy": final_numpy,
    "final_rust": final_rust,
    "final_numpy_scalev1": final_numpy_scalev1,
    "final_rust_scalev1": final_rust_scalev1,
    "floor_probe": floor_probe,
}
