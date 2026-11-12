"""Principal-component spacecraft roll coordinate projection and robust normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

_MAD_TO_SIGMA: float = float(1.0 / ndtri(0.75))
_IQR_HALF_Z: float = float(ndtri(0.75))
_IDR_HALF_Z: float = float(ndtri(0.90))
_TRIM_SIGMA = 5.0
_TRIM_ITERS = 3
_REL_EPS_GUARD = 64.0
_SO2_ATOL = 1e-10
_MIN_PCA_POINTS = 3
_MIN_TRIM_POINTS = 5


def _orient_and_complete_so2(r1_raw: np.ndarray) -> np.ndarray:
    """Normalize a 2-vector, fix SVD sign deterministically, and complete R in SO(2)."""
    r1 = np.asarray(r1_raw, dtype=np.float64).reshape(2).copy()
    r1_norm = float(np.hypot(r1[0], r1[1]))
    if r1_norm > 0.0:
        r1 /= r1_norm
    else:
        r1 = np.array([1.0, 0.0], dtype=np.float64)

    if (abs(r1[0]) >= abs(r1[1]) and r1[0] < 0.0) or (
        abs(r1[1]) > abs(r1[0]) and r1[1] < 0.0
    ):
        r1 = -r1

    r2 = np.array([-r1[1], r1[0]], dtype=np.float64)
    return np.vstack([r1, r2])


def _robust_trim_sigma(u: np.ndarray, tol: float) -> float:
    """Estimate robust Gaussian-equivalent dispersion with IQR and 10-90% IDR fallbacks."""
    med = float(np.median(u))
    sig = _MAD_TO_SIGMA * float(np.median(np.abs(u - med)))
    if sig > tol:
        return sig
    q25, q75 = np.percentile(u, [25.0, 75.0])
    sig = float(q75 - q25) / (2.0 * _IQR_HALF_Z)
    if sig > tol:
        return sig
    q10, q90 = np.percentile(u, [10.0, 90.0])
    sig = float(q90 - q10) / (2.0 * _IDR_HALF_Z)
    return sig if sig > tol else 0.0


@dataclass(frozen=True)
class RollPCATransform:
    """Rigid SO(2) spacecraft-roll rotation and robust 4-sigma_MAD axis scaling.

    Attributes
    ----------
    center_xy : np.ndarray
        Shape ``(2,)`` read-only float64 robust ``(x, y)`` centre ``(mu_x, mu_y)``.
    rotation : np.ndarray
        Shape ``(2, 2)`` read-only float64 proper orthogonal rotation matrix ``R``
        in ``SO(2)`` (``det(R) = +1``, rows are unit principal axes ``r_1, r_2``).
    scales_u : np.ndarray
        Shape ``(2,)`` read-only float64 axis normalization denominators ``(s_1, s_2)``.
    """

    center_xy: np.ndarray
    rotation: np.ndarray
    scales_u: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.center_xy, dtype=np.float64).reshape(2).copy()
        rot = np.asarray(self.rotation, dtype=np.float64).reshape(2, 2).copy()
        scales = np.asarray(self.scales_u, dtype=np.float64).reshape(2).copy()
        if not np.all(np.isfinite(center)):
            raise ValueError(f"center_xy must be finite; got {center!r}.")
        if not np.all(np.isfinite(rot)):
            raise ValueError(f"rotation must be finite; got {rot!r}.")
        if not (
            np.allclose(rot @ rot.T, np.eye(2), atol=_SO2_ATOL)
            and np.isclose(float(np.linalg.det(rot)), 1.0, atol=_SO2_ATOL)
        ):
            raise ValueError(
                f"rotation must be a proper orthogonal SO(2) matrix (R @ R.T = I, det(R) = +1); got {rot!r}."
            )
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError(
                f"scales_u must be finite and strictly positive; got {scales!r}."
            )
        center.setflags(write=False)
        rot.setflags(write=False)
        scales.setflags(write=False)
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

    Uses spherical PCA (SVD of unit-normalized displacements from the coordinate-wise
    median) to obtain a bounded-leverage initial roll frame immune to both along-roll
    and oblique cross-roll flyers, followed by iterative 5-sigma trimming in the
    rotated ``(u1, u2)`` frame before final covariance SVD and scale estimation.

    Parameters
    ----------
    x, y : array_like
        1D spatial centroid / ``(POS_CORR1, POS_CORR2)`` coordinate arrays.
    clean_mask : array_like of bool, optional
        Optional clean-cadence mask (e.g. excluding thruster firings and bad quality
        cadences). Honoured whenever at least 3 finite clean cadences survive;
        otherwise falls back to an identity rotation without unmasking flagged cadences.

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
    else:
        clean = fin

    if int(clean.sum()) < _MIN_PCA_POINTS:
        ref_x = x_arr[clean] if np.any(clean) else x_arr[np.isfinite(x_arr)]
        ref_y = y_arr[clean] if np.any(clean) else y_arr[np.isfinite(y_arr)]
        cx = float(np.median(ref_x)) if ref_x.size > 0 else 0.0
        cy = float(np.median(ref_y)) if ref_y.size > 0 else 0.0
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

    eps = float(np.finfo(np.float64).eps)
    coord_mag_clean = float(
        max(np.max(np.abs(x_clean)), np.max(np.abs(y_clean)))
    )
    rad_tol = _REL_EPS_GUARD * eps * coord_mag_clean

    dx_all = x_clean - mu_x
    dy_all = y_clean - mu_y
    radii = np.hypot(dx_all, dy_all)
    nonzero = radii > rad_tol
    if int(nonzero.sum()) < 2:
        return RollPCATransform(
            center_xy=center_xy,
            rotation=np.eye(2, dtype=np.float64),
            scales_u=np.ones(2, dtype=np.float64),
        )

    # Bounded-leverage initial frame via spherical PCA (Locantore et al. 1999)
    z_sphere = np.column_stack(
        [dx_all[nonzero] / radii[nonzero], dy_all[nonzero] / radii[nonzero]]
    )
    _uz, _sz, vtz = np.linalg.svd(z_sphere, full_matrices=False)
    rotation = _orient_and_complete_so2(vtz[0, :])

    # Iterative 5-sigma box trimming in the rotated principal-axis frame (u1, u2)
    inliers = np.ones(x_clean.size, dtype=bool)
    for _ in range(_TRIM_ITERS):
        u1_all = rotation[0, 0] * dx_all + rotation[0, 1] * dy_all
        u2_all = rotation[1, 0] * dx_all + rotation[1, 1] * dy_all

        med1 = float(np.median(u1_all[inliers]))
        med2 = float(np.median(u2_all[inliers]))
        sig1 = _robust_trim_sigma(u1_all[inliers], rad_tol)
        sig2 = _robust_trim_sigma(u2_all[inliers], rad_tol)

        thresh1 = max(_TRIM_SIGMA * sig1, rad_tol)
        thresh2 = max(_TRIM_SIGMA * sig2, rad_tol)
        next_inliers = (np.abs(u1_all - med1) <= thresh1) & (
            np.abs(u2_all - med2) <= thresh2
        )
        if int(next_inliers.sum()) < _MIN_TRIM_POINTS:
            break
        inliers = next_inliers
        x_in = x_clean[inliers]
        y_in = y_clean[inliers]
        c_iter = np.column_stack(
            [x_in - float(np.mean(x_in)), y_in - float(np.mean(y_in))]
        )
        _ui, _si, vti = np.linalg.svd(c_iter, full_matrices=False)
        rotation = _orient_and_complete_so2(vti[0, :])

    if int(inliers.sum()) >= _MIN_TRIM_POINTS:
        x_trim = x_clean[inliers]
        y_trim = y_clean[inliers]
    else:
        x_trim = x_clean
        y_trim = y_clean

    # Final covariance SVD on the trimmed principal-axis inliers
    c_svd = np.column_stack(
        [x_trim - float(np.mean(x_trim)), y_trim - float(np.mean(y_trim))]
    )
    _u, _s, vt = np.linalg.svd(c_svd, full_matrices=False)
    rotation = _orient_and_complete_so2(vt[0, :])

    dx_t = x_trim - mu_x
    dy_t = y_trim - mu_y
    u_trim = (
        rotation[0, 0] * dx_t + rotation[0, 1] * dy_t,
        rotation[1, 0] * dx_t + rotation[1, 1] * dy_t,
    )

    coord_mag_trim = float(
        max(np.max(np.abs(x_trim)), np.max(np.abs(y_trim)))
    )
    sigma_mad = np.array(
        [
            _MAD_TO_SIGMA * float(np.median(np.abs(u_k - float(np.median(u_k)))))
            for u_k in u_trim
        ],
        dtype=np.float64,
    )
    scale_tol = _REL_EPS_GUARD * eps * max(coord_mag_trim, float(sigma_mad[0]))

    scales = np.ones(2, dtype=np.float64)
    for k, u_k in enumerate(u_trim):
        sigma_k = float(sigma_mad[k])
        if sigma_k <= scale_tol:
            sigma_k = float(np.std(u_k))
        scales[k] = 4.0 * sigma_k if sigma_k > scale_tol else 1.0

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
