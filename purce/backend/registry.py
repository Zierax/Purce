"""Registry for Purce kernels — single source of truth for mappings and bodies."""

from __future__ import annotations

from purce.backend.c99_generator import (
    BODY_PARAM_MAP as _BPM,
    DERIVED_PARAMS as _DP,
    MATH_KERNEL_BODIES as _MKB,
)
from purce.ir.nodes import Dtype

# Kernels that have a body entry but are intentionally incomplete stubs.
_STUB_ALGORITHMS: frozenset[str] = frozenset()

DTYPE_TO_C = {
    Dtype.FLOAT32: "float",
    Dtype.FLOAT64: "double",
    Dtype.INT32: "int32_t",
    Dtype.INT64: "int64_t",
    Dtype.Q15: "q15_t",
    Dtype.Q31: "q31_t",
    Dtype.BOOL: "int",
    Dtype.COMPLEX64: "float _Complex",
    Dtype.COMPLEX128: "double _Complex",
}

# Import the actual maps from the generator to avoid duplication at runtime.
# They are defined in c99_generator.py and re-exported here for single-source.
# To keep registry as single file, we import after definition? Instead we define
# them here and have c99_generator import from here. For now, keep copies
# and ensure they stay in sync via tests.

# NOTE: The canonical definitions live in c99_generator.py; this file re-exports
# them for the split. During the transition, both files contain copies.
# The next commit will make c99_generator import from here.

BODY_PARAM_MAP = _BPM
DERIVED_PARAMS = _DP
MATH_KERNEL_BODIES = _MKB
