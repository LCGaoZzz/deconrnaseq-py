"""deconrnaseq-py: fast Python port of Bioconductor DeconRNASeq 1.50.0.

Original R package: Ting Gong, Joseph D. Szustakowski, GPL-2
(Bioconductor 3.21, DeconRNASeq 1.50.0, git commit 20915cd).
This Python re-implementation is distributed under the same GPL-2 licence;
see LICENSE and NOTICE.
"""

from .core import DeconResult, deconrnaseq, deconvolve_core
from .solvers import (
    rust_available,
    solve_lsei,
    solve_lsei_activeset,
    solve_lsei_enum,
    solve_lsei_enum_fast,
    solve_lsei_interior,
)

__version__ = "1.50.0"
__upstream_version__ = "1.50.0"  # Bioconductor DeconRNASeq release mirrored

__all__ = [
    "deconrnaseq",
    "deconvolve_core",
    "DeconResult",
    "solve_lsei",
    "solve_lsei_enum",
    "solve_lsei_enum_fast",
    "solve_lsei_activeset",
    "solve_lsei_interior",
    "rust_available",
    "__version__",
    "__upstream_version__",
]
