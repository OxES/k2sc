"""JAX float64 precision configuration and array validation utilities."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp


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
