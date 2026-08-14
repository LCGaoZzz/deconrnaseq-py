//! deconrnaseq-rust: optional compiled backend for deconrnaseq-py.
//!
//! Interior-first exact solver for the simplex-constrained least squares
//!     min ||A x - y||^2  s.t.  sum(x) = 1, x >= 0
//! given gram = A'A (K,K) and cross = A'Y (K,S).
//!
//! The equality KKT solve handles interior samples first. Boundary samples
//! use the same support enumeration order (size ascending, lexicographic
//! within a size, matching itertools.combinations), the same (k+1)x(k+1)
//! KKT system, feasibility tolerance 1e-12, tiny-negative clamp, and strict
//! `<` objective comparison so ties resolve to the earliest support.
//!
//! Licence: GPL-2.0-only (derivative of GPL-2 DeconRNASeq; see NOTICE).

// PyO3 0.22's generated wrappers trigger this lint on Rust 1.97.
#![allow(clippy::useless_conversion)]

use numpy::ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

const TOL: f64 = 1e-12;

/// Lexicographic combinations of `size` indices out of 0..k, in the same
/// order as Python's itertools.combinations.
fn combinations(k: usize, size: usize) -> Vec<Vec<usize>> {
    fn rec(start: usize, size: usize, k: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == size {
            out.push(cur.clone());
            return;
        }
        for i in start..k {
            cur.push(i);
            rec(i + 1, size, k, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    rec(0, size, k, &mut Vec::new(), &mut out);
    out
}

/// Solve A (n x n, row-major) * X = B (n x s, row-major) in place via
/// Gaussian elimination with partial pivoting.  Returns false if singular.
/// B is overwritten with the solution.
fn solve_square(a: &mut [f64], n: usize, b: &mut [f64], s: usize) -> bool {
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best == 0.0 {
            return false;
        }
        if piv != col {
            for j in 0..n {
                a.swap(col * n + j, piv * n + j);
            }
            for j in 0..s {
                b.swap(col * s + j, piv * s + j);
            }
        }
        let d = a[col * n + col];
        for r in (col + 1)..n {
            let f = a[r * n + col] / d;
            if f == 0.0 {
                continue;
            }
            for j in col..n {
                a[r * n + j] -= f * a[col * n + j];
            }
            for j in 0..s {
                b[r * s + j] -= f * b[col * s + j];
            }
        }
    }
    for col in (0..n).rev() {
        let d = a[col * n + col];
        for j in 0..s {
            let mut acc = b[col * s + j];
            for r in (col + 1)..n {
                acc -= a[col * n + r] * b[r * s + j];
            }
            b[col * s + j] = acc / d;
        }
    }
    true
}

fn solve_lsei_enum_impl(
    g: ArrayView2<'_, f64>,
    c: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, &'static str> {
    let k_types = g.nrows();
    if g.ncols() != k_types || c.nrows() != k_types {
        return Err("gram must be (K,K) and cross (K,S)");
    }
    if k_types > 62 {
        return Err("enumeration backend supports at most 62 cell types");
    }
    let n_samples = c.ncols();

    let mut best_obj = vec![f64::INFINITY; n_samples];
    let mut best_x = vec![0.0f64; k_types * n_samples]; // (K, S) row-major

    for size in 1..=k_types {
        // KKT and RHS shapes depend only on support size. Reuse both scratch
        // buffers for every support of that size instead of allocating and
        // freeing two Vecs for each of the 2^K - 1 supports. The buffers are
        // fully overwritten below, so arithmetic and tie-breaking order are unchanged.
        let n = size + 1;
        let mut kkt = vec![0.0f64; n * n];
        let mut rhs = vec![0.0f64; n * n_samples];
        for support in combinations(k_types, size) {
            for (i, &si) in support.iter().enumerate() {
                for (j, &sj) in support.iter().enumerate() {
                    kkt[i * n + j] = g[[si, sj]];
                }
                kkt[i * n + size] = 1.0;
                kkt[size * n + i] = 1.0;
            }
            kkt[size * n + size] = 0.0;
            for (i, &si) in support.iter().enumerate() {
                for sm in 0..n_samples {
                    rhs[i * n_samples + sm] = c[[si, sm]];
                }
            }
            for sm in 0..n_samples {
                rhs[size * n_samples + sm] = 1.0;
            }
            if !solve_square(&mut kkt, n, &mut rhs, n_samples) {
                continue; // singular support, same as the LinAlgError skip
            }
            for sm in 0..n_samples {
                let mut feasible = true;
                for i in 0..size {
                    if rhs[i * n_samples + sm] < -TOL {
                        feasible = false;
                        break;
                    }
                }
                if !feasible {
                    continue;
                }
                let mut xs = [0.0f64; 64]; // k_types <= 62 checked above
                for i in 0..size {
                    let v = rhs[i * n_samples + sm];
                    xs[i] = if v > 0.0 { v } else { 0.0 };
                }
                let mut obj = 0.0f64;
                let mut quad = 0.0f64;
                for (i, &si) in support.iter().enumerate() {
                    let mut row = 0.0f64;
                    for (j, &sj) in support.iter().enumerate() {
                        row += g[[si, sj]] * xs[j];
                    }
                    quad += xs[i] * row;
                    obj -= xs[i] * c[[si, sm]];
                }
                obj += 0.5 * quad;
                if obj < best_obj[sm] {
                    best_obj[sm] = obj;
                    for t in 0..k_types {
                        best_x[t * n_samples + sm] = 0.0;
                    }
                    for (i, &si) in support.iter().enumerate() {
                        best_x[si * n_samples + sm] = xs[i];
                    }
                }
            }
        }
    }

    if best_obj.iter().any(|v| !v.is_finite()) {
        return Err("no feasible solution for one or more samples");
    }
    Ok(Array2::from_shape_vec((k_types, n_samples), best_x).expect("shape matches construction"))
}

/// Exact enumeration over all 2^K - 1 supports.
///
/// gram: (K, K) C-order, cross: (K, S) C-order -> returns X (K, S) C-order.
#[pyfunction]
fn solve_lsei_enum<'py>(
    py: Python<'py>,
    gram: PyReadonlyArray2<'py, f64>,
    cross: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr =
        solve_lsei_enum_impl(gram.as_array(), cross.as_array()).map_err(PyRuntimeError::new_err)?;
    Ok(arr.into_pyarray_bound(py))
}

/// Interior-first exact solver. The equality-only KKT solve, feasibility
/// classification, boundary collection, exact enumeration, and scatter all
/// stay in Rust so the Python/Rust boundary is crossed only once.
#[pyfunction(signature = (gram, cross, tol=1e-12))]
fn solve_lsei_interior<'py>(
    py: Python<'py>,
    gram: PyReadonlyArray2<'py, f64>,
    cross: PyReadonlyArray2<'py, f64>,
    tol: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let g = gram.as_array();
    let c = cross.as_array();
    let k_types = g.nrows();
    if g.ncols() != k_types || c.nrows() != k_types {
        return Err(PyRuntimeError::new_err(
            "gram must be (K,K) and cross (K,S)",
        ));
    }
    if k_types > 62 {
        return Err(PyRuntimeError::new_err(
            "enumeration backend supports at most 62 cell types",
        ));
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err(PyRuntimeError::new_err(
            "tol must be finite and non-negative",
        ));
    }

    let n_samples = c.ncols();
    let n = k_types + 1;
    let mut kkt = vec![0.0f64; n * n];
    let mut rhs = vec![0.0f64; n * n_samples];
    for i in 0..k_types {
        for j in 0..k_types {
            kkt[i * n + j] = g[[i, j]];
        }
        kkt[i * n + k_types] = 1.0;
        kkt[k_types * n + i] = 1.0;
        for sm in 0..n_samples {
            rhs[i * n_samples + sm] = c[[i, sm]];
        }
    }
    for sm in 0..n_samples {
        rhs[k_types * n_samples + sm] = 1.0;
    }
    if !solve_square(&mut kkt, n, &mut rhs, n_samples) {
        return Err(PyRuntimeError::new_err(
            "singular equality-constrained KKT system",
        ));
    }

    let mut out = vec![0.0f64; k_types * n_samples];
    let mut boundary = Vec::new();
    for sm in 0..n_samples {
        let interior = (0..k_types).all(|i| rhs[i * n_samples + sm] >= -tol);
        if interior {
            for i in 0..k_types {
                let value = rhs[i * n_samples + sm];
                out[i * n_samples + sm] = if value >= 0.0 { value } else { 0.0 };
            }
        } else {
            boundary.push(sm);
        }
    }

    if !boundary.is_empty() {
        let n_boundary = boundary.len();
        let mut boundary_cross = vec![0.0f64; k_types * n_boundary];
        for i in 0..k_types {
            for (dst, &src) in boundary.iter().enumerate() {
                boundary_cross[i * n_boundary + dst] = c[[i, src]];
            }
        }
        let boundary_cross = Array2::from_shape_vec((k_types, n_boundary), boundary_cross)
            .expect("shape matches construction");
        let boundary_out =
            solve_lsei_enum_impl(g, boundary_cross.view()).map_err(PyRuntimeError::new_err)?;
        for i in 0..k_types {
            for (src, &dst) in boundary.iter().enumerate() {
                out[i * n_samples + dst] = boundary_out[[i, src]];
            }
        }
    }

    let arr =
        Array2::from_shape_vec((k_types, n_samples), out).expect("shape matches construction");
    Ok(arr.into_pyarray_bound(py))
}

#[pymodule]
fn deconrnaseq_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_lsei_enum, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lsei_interior, m)?)?;
    Ok(())
}
