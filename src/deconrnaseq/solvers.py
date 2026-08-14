"""Constrained least-squares solvers used by the DeconRNASeq port.

Mathematical problem (identical to R DeconRNASeq 1.50.0, which calls
``limSolve::lsei`` -> Lawson-Hanson LSEI):

    minimize_x  || A x - y ||^2
    subject to  sum(x) = 1  and  x >= 0

Only the Gram matrix ``G = A.T @ A`` and cross-products ``c = A.T @ y`` are
needed, so all solvers take ``gram`` (K, K) and ``cross`` (K, S) and return
``X`` (K, S) with one column per sample.  Everything is float64.

Three implementations are provided:

* ``solve_lsei_enum``        - exact support enumeration, per-support loop
                               (reference/baseline implementation; mirrors the
                               independent validator validate_dataset.py).
* ``solve_lsei_enum_fast``   - same exact enumeration, but supports are grouped
                               by size and solved as stacked batched linear
                               systems (vectorised across supports *and*
                               samples).
* ``solve_lsei_activeset``   - classical active-set QP for arbitrary K,
                               optional warm-start chaining across samples.

All three solve the same strictly convex QP and therefore agree to float64
accuracy on well-conditioned problems; the exact gates are verified against
the R reference outputs in tests/test_r_regression.py.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import numpy as np

__all__ = [
    "solve_lsei",
    "solve_lsei_enum",
    "solve_lsei_enum_fast",
    "solve_lsei_activeset",
    "solve_lsei_interior",
    "rust_available",
]

#: enumeration is exact and very fast for small K; above this use active-set
_ENUM_MAX_K = 12


@lru_cache(maxsize=64)
def _supports_by_size(k_types: int) -> tuple:
    """All non-empty supports grouped by size: tuple of (m, k) int arrays."""
    return tuple(
        np.asarray(list(combinations(range(k_types), k)), dtype=np.intp)
        for k in range(1, k_types + 1)
    )


def _objective(gram: np.ndarray, cross: np.ndarray, x: np.ndarray) -> np.ndarray:
    """0.5 x'Gx - c'x per column of x (K, S) -> (S,)."""
    return 0.5 * np.einsum("is,ij,js->s", x, gram, x) - np.einsum("is,is->s", x, cross)


def solve_lsei_enum(
    gram: np.ndarray, cross: np.ndarray, *, tol: float = 1e-12
) -> np.ndarray:
    """Exact enumeration over all 2^K - 1 supports (baseline implementation).

    For each candidate support, the equality-constrained KKT system is solved
    for *all* samples at once; the feasible solution with the smallest
    objective wins per sample.  Loops over supports in Python.
    """
    gram = np.ascontiguousarray(gram, dtype=np.float64)
    cross = np.ascontiguousarray(cross, dtype=np.float64)
    K, S = cross.shape
    best_obj = np.full(S, np.inf)
    best_x = np.zeros((K, S), dtype=np.float64)
    arange_s = np.arange(S)

    for supports in _supports_by_size(K):
        for support in supports:
            k = support.size
            kkt = np.empty((k + 1, k + 1), dtype=np.float64)
            kkt[:k, :k] = gram[np.ix_(support, support)]
            kkt[:k, k] = 1.0
            kkt[k, :k] = 1.0
            kkt[k, k] = 0.0
            rhs = np.empty((k + 1, S), dtype=np.float64)
            rhs[:k] = cross[support]
            rhs[k] = 1.0
            try:
                sol = np.linalg.solve(kkt, rhs)
            except np.linalg.LinAlgError:
                continue
            xs = sol[:k]
            feasible = (xs >= -tol).all(axis=0)
            if not feasible.any():
                continue
            xs = np.where(xs >= 0.0, xs, 0.0)
            full_x = np.zeros((K, S), dtype=np.float64)
            full_x[support] = xs
            obj = _objective(gram, cross, full_x)
            upd = feasible & (obj < best_obj)
            if upd.any():
                best_obj[upd] = obj[upd]
                best_x[:, upd] = full_x[:, upd]

    if not np.isfinite(best_obj).all():
        raise RuntimeError("no feasible solution for one or more samples")
    return best_x


def solve_lsei_enum_fast(
    gram: np.ndarray, cross: np.ndarray, *, tol: float = 1e-12
) -> np.ndarray:
    """Exact support enumeration, batched across supports and samples.

    Mathematically identical to :func:`solve_lsei_enum`; supports of equal
    size are stacked into a single batched ``np.linalg.solve`` call.
    """
    gram = np.ascontiguousarray(gram, dtype=np.float64)
    cross = np.ascontiguousarray(cross, dtype=np.float64)
    K, S = cross.shape
    best_obj = np.full(S, np.inf)
    best_x = np.zeros((K, S), dtype=np.float64)
    arange_s = np.arange(S)

    for supports in _supports_by_size(K):
        m, k = supports.shape
        g_s = gram[supports[:, :, None], supports[:, None, :]]  # (m, k, k)
        kkt = np.zeros((m, k + 1, k + 1), dtype=np.float64)
        kkt[:, :k, :k] = g_s
        kkt[:, :k, k] = 1.0
        kkt[:, k, :k] = 1.0
        rhs = np.empty((m, k + 1, S), dtype=np.float64)
        rhs[:, :k, :] = cross[supports]  # (m, k, S)
        rhs[:, k, :] = 1.0
        try:
            sol = np.linalg.solve(kkt, rhs)  # (m, k+1, S)
            xs = sol[:, :k, :]
        except np.linalg.LinAlgError:
            # rare: a numerically singular support in the batch -> per-support
            xs = np.empty((m, k, S), dtype=np.float64)
            for i, support in enumerate(supports):
                try:
                    sol_i = np.linalg.solve(kkt[i], rhs[i])
                except np.linalg.LinAlgError:
                    xs[i] = -np.inf  # infeasible marker
                    continue
                xs[i] = sol_i[:k]
        feasible = (xs >= -tol).all(axis=1)  # (m, S)
        xs = np.where(xs >= 0.0, xs, 0.0)
        obj = 0.5 * np.einsum("mks,mkl,mls->ms", xs, g_s, xs) - np.einsum(
            "mks,mks->ms", xs, cross[supports]
        )
        obj = np.where(feasible, obj, np.inf)
        idx = obj.argmin(axis=0)  # (S,)
        val = obj[idx, arange_s]
        upd = val < best_obj
        if upd.any():
            best_obj[upd] = val[upd]
            chosen = xs[idx, :, arange_s]  # (S, k)
            best_x[:, upd] = 0.0
            cols = arange_s[upd]
            best_x[supports[idx[upd]], cols[:, None]] = chosen[upd]

    if not np.isfinite(best_obj).all():
        raise RuntimeError("no feasible solution for one or more samples")
    return best_x


def _activeset_one(
    gram: np.ndarray,
    c: np.ndarray,
    x0: np.ndarray | None,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """Single-sample active-set QP: min 0.5 x'Gx - c'x s.t. 1'x=1, x>=0."""
    K = c.shape[0]
    if x0 is None:
        x = np.zeros(K)
        x[int(np.argmax(c))] = 1.0
    else:
        x = np.where(x0 > 0.0, x0, 0.0)
        s = x.sum()
        if s <= 0.0:
            x = np.full(K, 1.0 / K)
        else:
            x = x / s
    support = np.flatnonzero(x > 0.0)
    if support.size == 0:
        support = np.array([int(np.argmax(c))])
        x = np.zeros(K)
        x[support[0]] = 1.0

    for _ in range(max_iter):
        k = support.size
        g_p = gram[np.ix_(support, support)]
        kkt = np.empty((k + 1, k + 1))
        kkt[:k, :k] = g_p
        kkt[:k, k] = 1.0
        kkt[k, :k] = 1.0
        kkt[k, k] = 0.0
        rhs = np.empty(k + 1)
        rhs[:k] = c[support]
        rhs[k] = 1.0
        try:
            sol = np.linalg.solve(kkt, rhs)
        except np.linalg.LinAlgError:
            # singular support: drop the smallest-current-value member
            drop = support[np.argmin(x[support])]
            support = support[support != drop]
            continue
        x_p, lam = sol[:k], sol[k]
        if (x_p >= -tol).all():
            x_new = np.zeros(K)
            x_new[support] = np.where(x_p > 0.0, x_p, 0.0)
            # KKT multiplier check for the zero (out-of-support) variables.
            # Stationarity: Gx - c - lambda*1 - mu = 0, mu >= 0.
            # The KKT solve above gives (Gx - c)_j = -lam on the support,
            # i.e. lambda = -lam, hence mu_j = (Gx - c)_j + lam for j off-support.
            resid = gram @ x_new - c
            mask = np.ones(K, dtype=bool)
            mask[support] = False
            if mask.any():
                mu = resid[mask] + lam
                worst_rel = int(np.argmin(mu))
                if mu[worst_rel] >= -tol:
                    return x_new
                support = np.append(support, np.flatnonzero(mask)[worst_rel])
                x = x_new
            else:
                return x_new
        else:
            # line search from current feasible x towards the infeasible
            # subproblem solution, blocking at the first boundary
            x_trial = np.zeros(K)
            x_trial[support] = x_p
            d = x_trial - x
            neg = d < 0.0
            if not neg.any():
                x = x_trial
                continue
            alpha = min(1.0, float(np.min(x[neg] / (-d[neg]))))
            x = x + alpha * d
            x[x < tol] = 0.0
            support = np.flatnonzero(x > 0.0)
            if support.size == 0:
                support = np.array([int(np.argmax(c))])
                x = np.zeros(K)
                x[support[0]] = 1.0
    raise RuntimeError("active-set QP did not converge within max_iter")


def solve_lsei_activeset(
    gram: np.ndarray,
    cross: np.ndarray,
    *,
    tol: float = 1e-12,
    max_iter: int = 1000,
    warm_start: bool = False,
) -> np.ndarray:
    """Active-set QP, looping over samples; optional warm-start chaining."""
    gram = np.ascontiguousarray(gram, dtype=np.float64)
    cross = np.ascontiguousarray(cross, dtype=np.float64)
    K, S = cross.shape
    out = np.empty((K, S), dtype=np.float64)
    x_prev = None
    for s in range(S):
        x = _activeset_one(
            gram, cross[:, s], x_prev if warm_start else None, tol, max_iter
        )
        out[:, s] = x
        if warm_start:
            x_prev = x
    return out


def rust_available() -> bool:
    """True when the optional compiled backend (deconrnaseq-rust) is installed."""
    try:
        import deconrnaseq_rust  # noqa: F401
    except ImportError:
        return False
    return True


def _solve_enum_rust(gram: np.ndarray, cross: np.ndarray) -> np.ndarray:
    import deconrnaseq_rust

    return np.asarray(
        deconrnaseq_rust.solve_lsei_enum(
            np.ascontiguousarray(gram), np.ascontiguousarray(cross)
        ),
        dtype=np.float64,
    )


def solve_lsei_interior(
    gram: np.ndarray,
    cross: np.ndarray,
    *,
    tol: float = 1e-12,
    backend: str = "auto",
) -> np.ndarray:
    """Interior-first exact solver (the default for K <= 12).

    One batched equality-only KKT solve handles every sample whose
    unconstrained solution already satisfies x >= 0 (such a point is exactly
    the constrained optimum).  Only the remaining boundary samples go through
    the exact support enumeration — via the compiled Rust backend when
    ``backend`` is "rust" (or "auto" and the wheel is installed), otherwise
    via the pure-NumPy batched enumeration.
    """
    gram = np.ascontiguousarray(gram, dtype=np.float64)
    cross = np.ascontiguousarray(cross, dtype=np.float64)
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
    interior = (sol >= -tol).all(axis=0)
    out = np.where(sol >= 0.0, sol, 0.0)
    if not interior.all():
        rest = np.flatnonzero(~interior)
        use_rust = backend == "rust" or (backend == "auto" and rust_available())
        if use_rust:
            out[:, rest] = _solve_enum_rust(gram, cross[:, rest])
        else:
            out[:, rest] = solve_lsei_enum_fast(gram, cross[:, rest], tol=tol)
    return out


def solve_lsei(
    gram: np.ndarray,
    cross: np.ndarray,
    *,
    method: str = "auto",
    backend: str = "auto",
) -> np.ndarray:
    """Dispatch to a solver.

    ``auto``: interior-first exact method for K <= 12 (Rust-accelerated
    boundary enumeration when the optional ``deconrnaseq-rust`` wheel is
    installed and ``backend="auto"``), warm-started active-set for larger K.
    All methods solve the identical QP to float64 accuracy.
    """
    K = gram.shape[0]
    if method == "auto":
        method = "interior" if K <= _ENUM_MAX_K else "activeset_warm"
    solvers = {
        "enum": solve_lsei_enum,
        "enum_fast": solve_lsei_enum_fast,
        "interior": solve_lsei_interior,
        "interior_rust": lambda g, c: solve_lsei_interior(g, c, backend="rust"),
        "activeset": solve_lsei_activeset,
        "activeset_warm": lambda g, c: solve_lsei_activeset(g, c, warm_start=True),
    }
    if method not in solvers:
        raise ValueError(f"unknown solver method {method!r}; choose from {sorted(solvers)}")
    if method == "interior":
        return solve_lsei_interior(gram, cross, backend=backend)
    return solvers[method](gram, cross)
