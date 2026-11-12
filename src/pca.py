"""Principal-component spacecraft roll coordinate projection and robust normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MAD_TO_SIGMA = 1.4826
_TRIM_SIGMA = 5.0
_TRIM_FLOOR = 1e-8
_SCALE_FLOOR = 1e-12
_MIN_CLEAN_MASK_POINTS = 10
_MIN_PCA_POINTS = 3
_MIN_TRIM_POINTS = 5


@dataclass(frozen=True)
class RollPCATransform:
    """Rigid SO(2) spacecraft-roll rotation and robust 4-sigma_MAD axis scaling.

    Attributes
    ----------
    center_xy : np.ndarray
        Shape ``(2,)`` float64 robust ``(x, y)`` centre ``(mu_x, mu_y)``.
    rotation : np.ndarray
        Shape ``(2, 2)`` float64 proper orthogonal rotation matrix ``R`` in ``SO(2)``
        (``det(R) = +1``, rows are unit principal axes ``r_1, r_2``).
    scales_u : np.ndarray
        Shape ``(2,)`` float64 axis normalization denominators ``(s_1, s_2)``.
    """

    center_xy: np.ndarray
    rotation: np.ndarray
    scales_u: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.center_xy, dtype=np.float64).reshape(2).copy()
        rot = np.asarray(self.rotation, dtype=np.float64).reshape(2, 2).copy()
        scales = np.asarray(self.scales_u, dtype=np.float64).reshape(2).copy()
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError(
                f"scales_u must be finite and strictly positive; got {scales!r}."
            )
        object.__setattr__(self, "center_xy", center)
        object.__setattr__(self, "rotation", rot)
        object.__setattr__(self, "scales_u", scales)

    def transform(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Project 1D ``(x, y)`` arrays onto normalized principal roll coordinates ``(u1, u2)``.

        Preserves ``np.nan`` on any cadence where ``x`` or ``y`` is non-finite.
        """
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        if x_arr.size != y_arr.size:
            raise ValueError(
                f"Length mismatch: x has {x_arr.size} elements, y has {y_arr.size}."
            )

        u1_out = np.full(x_arr.size, np.nan, dtype=np.float64)
        u2_out = np.full(x_arr.size, np.nan, dtype=np.float64)
        fin = np.isfinite(x_arr) & np.isfinite(y_arr)
        if np.any(fin):
            dx = x_arr[fin] - self.center_xy[0]
            dy = y_arr[fin] - self.center_xy[1]
            r = self.rotation
            u1_out[fin] = (r[0, 0] * dx + r[0, 1] * dy) / self.scales_u[0]
            u2_out[fin] = (r[1, 0] * dx + r[1, 1] * dy) / self.scales_u[1]
        return u1_out, u2_out

    def inverse_transform(self, u1, u2) -> tuple[np.ndarray, np.ndarray]:
        """Map normalized roll coordinates ``(u1, u2)`` back to detector ``(x, y)`` space.

        Preserves ``np.nan`` on any cadence where ``u1`` or ``u2`` is non-finite.
        """
        u1_arr = np.asarray(u1, dtype=np.float64).ravel()
        u2_arr = np.asarray(u2, dtype=np.float64).ravel()
        if u1_arr.size != u2_arr.size:
            raise ValueError(
                f"Length mismatch: u1 has {u1_arr.size} elements, u2 has {u2_arr.size}."
            )

        x_out = np.full(u1_arr.size, np.nan, dtype=np.float64)
        y_out = np.full(u1_arr.size, np.nan, dtype=np.float64)
        fin = np.isfinite(u1_arr) & np.isfinite(u2_arr)
        if np.any(fin):
            v1 = u1_arr[fin] * self.scales_u[0]
            v2 = u2_arr[fin] * self.scales_u[1]
            r = self.rotation
            x_out[fin] = r[0, 0] * v1 + r[1, 0] * v2 + self.center_xy[0]
            y_out[fin] = r[0, 1] * v1 + r[1, 1] * v2 + self.center_xy[1]
        return x_out, y_out


def fit_roll_pca(x, y, clean_mask=None) -> RollPCATransform:
    """Fit a robust SO(2) spacecraft-roll PCA rotation and 4-sigma_MAD axis normalization.

    Parameters
    ----------
    x, y : array_like
        1D spatial centroid / ``(POS_CORR1, POS_CORR2)`` coordinate arrays.
    clean_mask : array_like of bool, optional
        Optional clean-cadence mask (e.g. excluding thruster firings and bad quality
        cadences). Falls back to all simultaneously finite ``(x, y)`` cadences if
        fewer than 10 cadences survive ``clean_mask``.

    Returns
    -------
    RollPCATransform
        Fitted rigid rotation and robust axis-scaling transform.
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if x_arr.size != y_arr.size:
        raise ValueError(
            f"Length mismatch: x has {x_arr.size} elements, y has {y_arr.size}."
        )

    fin = np.isfinite(x_arr) & np.isfinite(y_arr)
    if clean_mask is not None:
        cm = np.asarray(clean_mask, dtype=bool).ravel()
        if cm.size != x_arr.size:
            raise ValueError(
                f"clean_mask length ({cm.size}) != coordinate length ({x_arr.size})."
            )
        clean = fin & cm
        if int(clean.sum()) < _MIN_CLEAN_MASK_POINTS:
            clean = fin
    else:
        clean = fin

    if int(clean.sum()) < _MIN_PCA_POINTS:
        fin_x = np.isfinite(x_arr)
        fin_y = np.isfinite(y_arr)
        cx = float(np.median(x_arr[fin_x])) if np.any(fin_x) else 0.0
        cy = float(np.median(y_arr[fin_y])) if np.any(fin_y) else 0.0
        return RollPCATransform(
            center_xy=np.array([cx, cy], dtype=np.float64),
            rotation=np.eye(2, dtype=np.float64),
            scales_u=np.ones(2, dtype=np.float64),
        )

    x_clean = x_arr[clean]
    y_clean = y_arr[clean]
    mu_x = float(np.median(x_clean))
    mu_y = float(np.median(y_clean))
    center_xy = np.array([mu_x, mu_y], dtype=np.float64)

    sigma_x0 = _MAD_TO_SIGMA * float(np.median(np.abs(x_clean - mu_x)))
    sigma_y0 = _MAD_TO_SIGMA * float(np.median(np.abs(y_clean - mu_y)))

    if sigma_x0 > 0.0:
        keep_x = np.abs(x_clean - mu_x) <= _TRIM_SIGMA * max(sigma_x0, _TRIM_FLOOR)
    else:
        keep_x = np.ones(x_clean.size, dtype=bool)

    if sigma_y0 > 0.0:
        keep_y = np.abs(y_clean - mu_y) <= _TRIM_SIGMA * max(sigma_y0, _TRIM_FLOOR)
    else:
        keep_y = np.ones(y_clean.size, dtype=bool)

    trim = keep_x & keep_y
    if int(trim.sum()) < _MIN_TRIM_POINTS:
        x_trim = x_clean
        y_trim = y_clean
    else:
        x_trim = x_clean[trim]
        y_trim = y_clean[trim]

    # Mean-centre the outlier-trimmed subset so its 2x2 SVD diagonalizes the trimmed
    # sample covariance matrix cleanly (giving zero Pearson correlation on the trimmed set).
    c_svd = np.column_stack(
        [x_trim - float(np.mean(x_trim)), y_trim - float(np.mean(y_trim))]
    )
    _u, _s, vt = np.linalg.svd(c_svd, full_matrices=False)
    r1 = np.asarray(vt[0, :], dtype=np.float64).copy()
    r1_norm = float(np.hypot(r1[0], r1[1]))
    if r1_norm > 0.0:
        r1 /= r1_norm
    else:
        r1 = np.array([1.0, 0.0], dtype=np.float64)

    # Fix SVD sign ambiguity deterministically
    if (abs(r1[0]) >= abs(r1[1]) and r1[0] < 0.0) or (
        abs(r1[1]) > abs(r1[0]) and r1[1] < 0.0
    ):
        r1 = -r1

    # Enforce a proper right-handed rotation det(R) = +1 in SO(2)
    r2 = np.array([-r1[1], r1[0]], dtype=np.float64)
    rotation = np.vstack([r1, r2])

    dx_t = x_trim - mu_x
    dy_t = y_trim - mu_y
    u_trim = (
        r1[0] * dx_t + r1[1] * dy_t,
        r2[0] * dx_t + r2[1] * dy_t,
    )

    scales = np.ones(2, dtype=np.float64)
    for k, u_k in enumerate(u_trim):
        med_k = float(np.median(u_k))
        sigma_k = _MAD_TO_SIGMA * float(np.median(np.abs(u_k - med_k)))
        if sigma_k <= _SCALE_FLOOR:
            sigma_k = float(np.std(u_k))
        scales[k] = 4.0 * sigma_k if sigma_k > _SCALE_FLOOR else 1.0

    return RollPCATransform(
        center_xy=center_xy,
        rotation=rotation,
        scales_u=scales,
    )


def pca_normalize_xy(
    x,
    y,
    clean_mask=None,
) -> tuple[np.ndarray, np.ndarray, RollPCATransform]:
    """Fit ``RollPCATransform`` on ``(x, y)`` and return ``(u1, u2, transform)``."""
    tf = fit_roll_pca(x, y, clean_mask=clean_mask)
    u1, u2 = tf.transform(x, y)
    return u1, u2, tf


__all__ = ["RollPCATransform", "fit_roll_pca", "pca_normalize_xy"]
