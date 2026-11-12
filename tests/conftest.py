"""Shared pytest fixtures for fast, deterministic K2SC unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)


@pytest.fixture(autouse=True)
def _seed_global_numpy() -> None:
    """Keep legacy global-RNG tests (e.g. test_utils.py prior to PR 3) deterministic."""
    np.random.seed(42)


@pytest.fixture(scope="session")
def canonical_gp_dataset() -> dict[str, object]:
    """Canonical synthetic K2-like 3D dataset (N=128, N_tr=64) for single-JIT reuse."""
    rng = np.random.default_rng(20261111)
    n_total = 128
    n_tr = 64

    t_base = np.linspace(2300.0, 2380.0, n_total, dtype=np.float64)
    t = t_base + rng.uniform(-0.003, 0.003, size=n_total)
    t.sort()

    # Realistic 6-hour K2 roll-drift sawtooth plus slow secular drift
    roll_phase = ((t - 2300.0) / 0.25) % 1.0
    x = 0.28 * (roll_phase - 0.5) + 0.04 * np.sin(2.0 * np.pi * (t - 2300.0) / 40.0)
    y_pos = -0.22 * (roll_phase - 0.5) + 0.03 * np.cos(2.0 * np.pi * (t - 2300.0) / 55.0)
    X = np.column_stack([t, x, y_pos]).astype(np.float64)

    splits = [2326.0, 2354.0]
    seg = (t > splits[0]).astype(int) + (t > splits[1]).astype(int)
    pos_sys = np.where(seg == 0, 8e-4 * x - 6e-4 * y_pos, np.where(seg == 1, -7e-4 * x + 5e-4 * y_pos, 6e-4 * (x + y_pos)))
    astro = 1.5e-3 * np.sin(2.0 * np.pi * (t - 2300.0) / 7.3)
    y_flux = astro + pos_sys + rng.normal(0.0, 1.5e-4, size=n_total)

    # Mask out 18 points including cadences immediately straddling splits[0] & splits[1]
    mask = np.ones(n_total, dtype=bool)
    near_split0 = int(np.argmin(np.abs(t - splits[0])))
    near_split1 = int(np.argmin(np.abs(t - splits[1])))
    drop_idx = sorted(
        {
            0,
            5,
            19,
            near_split0,
            max(0, near_split0 - 1),
            near_split1,
            min(n_total - 1, near_split1 + 1),
            70,
            88,
            105,
            120,
        }
    )
    mask[drop_idx] = False

    valid_indices = np.flatnonzero(mask)
    tr_idx = np.sort(rng.choice(valid_indices, size=n_tr, replace=False))

    return {
        "X": X,
        "y": y_flux,
        "mask": mask,
        "X_fit": X[mask],
        "y_fit": y_flux[mask],
        "X_tr": X[tr_idx],
        "y_tr": y_flux[tr_idx],
        "splits": splits,
    }
