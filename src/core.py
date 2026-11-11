"""Core flag constants and default K2 campaign split timestamps for K2SC."""

from __future__ import annotations

M_QUALITY: int = 2**0
M_FLARE: int = 2**1
M_TRANSIT: int = 2**2
M_OUTLIER_U: int = 2**3
M_OUTLIER_D: int = 2**4
M_NOTFINITE: int = 2**5
M_PERIODIC: int = 2**6

DEFAULT_K2_SPLITS: dict[int, list[float]] = {
    0: [1956.0],
    1: [2002.0, 2035.0],
    2: [2085.0, 2118.0],
    3: [2154.0, 2190.0],
    4: [2240.0, 2273.0],
    5: [2344.0],
    6: [2390.0, 2428.0],
    7: [2468.5, 2515.0],
    8: [2579.0, 2598.5],
    9: [2652.0, 2685.0],
    91: [2652.0],
    92: [2685.0],
    10: [2772.0, 2800.0],
    101: [2753.0],
    102: [2778.0],
    11: [2830.0],
    111: [2830.0],
    112: [2870.0],
    12: [2916.0, 2949.0],
    13: [2997.0, 3032.0],
    14: [3086.4, 3123.65],
    15: [3170.0, 3207.5],
    16: [3297.5, 3331.0],
    17: [3367.0, 3400.0],
    18: [3425.0, 3460.0],
    19: [3502.0],
}

default_splits = DEFAULT_K2_SPLITS

__all__ = [
    "DEFAULT_K2_SPLITS",
    "M_FLARE",
    "M_NOTFINITE",
    "M_OUTLIER_D",
    "M_OUTLIER_U",
    "M_PERIODIC",
    "M_QUALITY",
    "M_TRANSIT",
    "default_splits",
]
