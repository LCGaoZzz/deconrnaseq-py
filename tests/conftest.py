"""Shared fixtures: paths and loaded data for the deconrnaseq-py test suite."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PKG_ROOT, "src"))

def _find_dir(name: str) -> str:
    """Prefer the in-repo copy (publishable layout); fall back to the
    original sibling layout used during development."""
    in_repo = os.path.join(PKG_ROOT, name)
    if os.path.isdir(in_repo):
        return in_repo
    return os.path.normpath(os.path.join(PKG_ROOT, "..", name))


DATA_DIR = _find_dir("test_data")
R_DIR = _find_dir("r_baseline")


@pytest.fixture(scope="session")
def signatures() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "reference_signatures.csv"), index_col=0)


@pytest.fixture(scope="session")
def mixtures_exact() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "mixtures_exact.csv"), index_col=0)


@pytest.fixture(scope="session")
def mixtures_noisy() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "mixtures_noisy.csv"), index_col=0)


@pytest.fixture(scope="session")
def truth() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "true_proportions.csv"), index_col=0)


@pytest.fixture(scope="session")
def r_reference():
    """R 1.50.0 reference outputs; None when the baseline has not been run."""
    out = {}
    ok = True
    for case in ("exact", "noisy"):
        for suffix in ("unscaled", "scaled"):
            path = os.path.join(R_DIR, f"mixtures_{case}_r_{suffix}.csv")
            if not os.path.exists(path):
                ok = False
                break
            out[(case, suffix)] = pd.read_csv(path, index_col=0)
    return out if ok else None


ALL_SOLVERS = ["enum", "enum_fast", "interior", "activeset", "activeset_warm"]
