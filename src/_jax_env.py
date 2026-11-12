"""JAX float64 precision configuration and array validation utilities."""

from __future__ import annotations

import os
import warnings

import jax
import jax.numpy as jnp

_SINGLE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_XLA_SINGLE_THREAD_FLAGS = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)


def configure_single_thread_env() -> None:
    """Configure BLAS/OpenMP/XLA environment variables for single-threaded worker processes.

    Uses ``os.environ.setdefault`` so explicit caller overrides are respected, and
    appends single-thread Eigen flags to ``XLA_FLAGS`` idempotently when not already set.
    Call this at the top of multi-worker batch entrypoints (ideally before JAX initializes
    its CPU backend) to prevent ``N_workers * N_cores`` thread oversubscription.
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
    """Enable 64-bit floating-point precision in JAX."""
    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)


ensure_float64()


def as_f64(x) -> jax.Array:
    """Convert input to a float64 JAX array, enforcing jax_enable_x64=True."""
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
