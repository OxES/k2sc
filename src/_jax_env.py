"""JAX float64 precision configuration and worker environment utilities."""

from __future__ import annotations

import os
import sys
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax

_SINGLE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_XLA_SINGLE_THREAD_FLAGS = "--xla_cpu_multi_thread_eigen=false"


def configure_single_thread_env() -> None:
    """Configure BLAS/OpenMP/XLA environment variables for single-threaded worker processes.

    This module uses only the Python standard library at import time, so calling
    ``from k2sc import configure_single_thread_env; configure_single_thread_env()``
    at the top of a worker entrypoint sets BLAS/OpenMP thread limits *before*
    ``numpy``, ``scipy``, or ``jax`` initialize their native C/C++ runtime pools.
    Explicit caller environment overrides are preserved via ``os.environ.setdefault``,
    and ``--xla_cpu_multi_thread_eigen=false`` is appended to ``XLA_FLAGS`` idempotently.
    """
    for var in _SINGLE_THREAD_ENV_VARS:
        os.environ.setdefault(var, "1")

    existing_xla = os.environ.get("XLA_FLAGS", "")
    if "--xla_cpu_multi_thread_eigen" not in existing_xla:
        stripped = existing_xla.strip()
        os.environ["XLA_FLAGS"] = (
            f"{stripped} {_XLA_SINGLE_THREAD_FLAGS}"
            if stripped
            else _XLA_SINGLE_THREAD_FLAGS
        )


def ensure_float64() -> None:
    """Enable 64-bit floating-point precision in JAX without eagerly importing JAX."""
    os.environ["JAX_ENABLE_X64"] = "True"
    jax_mod = sys.modules.get("jax")
    if jax_mod is not None and not jax_mod.config.jax_enable_x64:
        jax_mod.config.update("jax_enable_x64", True)


ensure_float64()


def as_f64(x) -> jax.Array:
    """Convert input to a float64 JAX array, enforcing jax_enable_x64=True."""
    import jax
    import jax.numpy as jnp

    ensure_float64()
    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Explicitly requested dtype.*float64",
            category=UserWarning,
        )
        arr = jnp.asarray(x, dtype=jnp.float64)
    if arr.dtype != jnp.float64:
        raise RuntimeError("k2sc requires JAX float64 mode (jax_enable_x64=True).")
    return arr
