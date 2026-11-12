"""Self-contained pure-NumPy + SciPy canonical GP reference oracle (zero jax/tinygp imports).

Implements the exact v2 mathematical specification from plan_final.md §§2.3–2.4, plus optional
explicitly gated flags reproducing historical george >= 0.3 / k2sc v1.0.2.2 quirks solely for
golden-fixture bijection checks.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla


def piecewise_mask_numpy(t1: np.ndarray, t2: np.ndarray, splits=None) -> np.ndarray:
    """Pointwise threshold segment indicator M_seg(t1, t2) = I(c(t1) == c(t2))."""
    t1_arr = np.asarray(t1, dtype=np.float64).ravel()
    t2_arr = np.asarray(t2, dtype=np.float64).ravel()
    if splits is None:
        return np.ones((t1_arr.size, t2_arr.size), dtype=np.float64)
    s_arr = np.asarray(splits, dtype=np.float64).ravel()
    s_fin = np.sort(s_arr[np.isfinite(s_arr)])
    if s_fin.size == 0:
        return np.ones((t1_arr.size, t2_arr.size), dtype=np.float64)
    seg1 = np.sum(t1_arr[:, None] > s_fin[None, :], axis=1)
    seg2 = np.sum(t2_arr[:, None] > s_fin[None, :], axis=1)
    return (seg1[:, None] == seg2[None, :]).astype(np.float64)


def legacy_splitgp_mask_numpy(t1: np.ndarray, t2: np.ndarray, splits=None) -> np.ndarray:
    """Historical k2sc v1.0.2.2 SplitGP._get_mask (for golden provenance checks only)."""
    t1_arr = np.asarray(t1, dtype=np.float64).ravel()
    t2_arr = np.asarray(t2, dtype=np.float64).ravel()
    if splits is None:
        return np.ones((t1_arr.size, t2_arr.size), dtype=np.float64)
    mask = np.zeros((t1_arr.size, t2_arr.size), dtype=bool)
    o1, o2 = 0, 0
    for split in splits:
        e1 = int(np.argmin(np.abs(t1_arr - split)))
        e2 = int(np.argmin(np.abs(t2_arr - split)))
        if (e1 < t1_arr.size - 1) and (e2 < t2_arr.size - 1):
            mask[o1 : e1 + 1, o2 : e2 + 1] = True
            o1, o2 = e1 + 1, e2 + 1
    mask[o1:, o2:] = True
    return mask.astype(np.float64)


def kernel_matrices_numpy(
    kernel_name: str,
    pv,
    X1,
    X2=None,
    splits=None,
    *,
    legacy_george03_gamma: bool = False,
    legacy_split_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Evaluate exact (K1, K2, sigma_n_sq) in pure float64 NumPy."""
    p = np.asarray(pv, dtype=np.float64)
    A1 = np.asarray(X1, dtype=np.float64)
    A2 = A1 if X2 is None else np.asarray(X2, dtype=np.float64)

    t_ref = float(A1[0, 0]) if A1.shape[0] > 0 else 0.0
    t1 = A1[:, 0] - t_ref
    t2 = A2[:, 0] - t_ref
    dt = t1[:, None] - t2[None, :]
    dx = A1[:, 1][:, None] - A2[:, 1][None, :]
    dy = A1[:, 2][:, None] - A2[:, 2][None, :]

    if splits is None:
        splits_shifted = None
    else:
        s_raw = np.asarray(splits, dtype=np.float64).ravel()
        splits_shifted = np.where(np.isfinite(s_raw), s_raw - t_ref, s_raw)

    if legacy_split_mask:
        m_seg = legacy_splitgp_mask_numpy(t1, t2, splits_shifted)
    else:
        m_seg = piecewise_mask_numpy(t1, t2, splits_shifted)

    key = kernel_name.lower()
    if key in ("basickernel", "basic"):
        K1 = (10.0 ** p[0]) * np.exp(-0.5 * p[1] * (dt**2))
        K2 = m_seg * (10.0 ** p[2]) * np.exp(-0.5 * p[3] * (dx**2) - 0.5 * p[4] * (dy**2))
        sigma_n_sq = float(10.0 ** (2.0 * p[5]))
        return K1, K2, sigma_n_sq

    if key in ("basickernelep", "basic_ep"):
        K1 = (10.0 ** p[0]) * np.exp(-0.5 * p[1] * (dt**2))
        K2 = m_seg * (10.0 ** p[2]) * np.exp(-np.sqrt(p[3]) * np.abs(dx) - np.sqrt(p[4]) * np.abs(dy))
        sigma_n_sq = float(10.0 ** (2.0 * p[5]))
        return K1, K2, sigma_n_sq

    if key in ("quasiperiodickernel", "quasiperiodic"):
        gamma = -np.log(p[1]) if legacy_george03_gamma else (1.0 / p[1])
        sin_arg = np.pi * np.abs(dt) / p[2]
        K1 = (10.0 ** p[0]) * np.exp(-gamma * (np.sin(sin_arg) ** 2) - 0.5 * p[3] * (dt**2))
        K2 = m_seg * (10.0 ** p[4]) * np.exp(-0.5 * p[5] * (dx**2) - 0.5 * p[6] * (dy**2))
        sigma_n_sq = float(10.0 ** (2.0 * p[7]))
        return K1, K2, sigma_n_sq

    if key in ("quasiperiodickernelep", "quasiperiodic_ep"):
        gamma = -np.log(p[1]) if legacy_george03_gamma else (1.0 / p[1])
        sin_arg = np.pi * np.abs(dt) / p[2]
        K1 = (10.0 ** p[0]) * np.exp(-gamma * (np.sin(sin_arg) ** 2) - 0.5 * p[3] * (dt**2))
        K2 = m_seg * (10.0 ** p[4]) * np.exp(-np.sqrt(p[5]) * np.abs(dx) - np.sqrt(p[6]) * np.abs(dy))
        sigma_n_sq = float(10.0 ** (2.0 * p[7]))
        return K1, K2, sigma_n_sq

    if key in ("periodickernel", "periodic"):
        gamma = -np.log(p[1]) if legacy_george03_gamma else (1.0 / p[1])
        sin_arg = np.pi * np.abs(dt) / p[2]
        K1 = (10.0 ** p[0]) * np.exp(-gamma * (np.sin(sin_arg) ** 2)) + (10.0 ** p[3]) * np.exp(
            -0.5 * p[4] * (dt**2)
        )
        K2 = m_seg * (10.0 ** p[5]) * np.exp(-0.5 * p[6] * (dx**2) - 0.5 * p[7] * (dy**2))
        sigma_n_sq = float(10.0 ** (2.0 * p[8]))
        return K1, K2, sigma_n_sq

    raise ValueError(f"Unknown kernel_name: {kernel_name!r}")


def gp_lnlikelihood_numpy(
    kernel_name: str,
    pv,
    X_fit,
    y_fit,
    splits=None,
    *,
    include_const: bool = True,
    legacy_george03_gamma: bool = False,
    legacy_split_mask: bool = False,
) -> float:
    """Pure-NumPy/SciPy Cholesky Gaussian log-likelihood."""
    y = np.asarray(y_fit, dtype=np.float64)
    K1, K2, sigma_n_sq = kernel_matrices_numpy(
        kernel_name,
        pv,
        X_fit,
        None,
        splits=splits,
        legacy_george03_gamma=legacy_george03_gamma,
        legacy_split_mask=legacy_split_mask,
    )
    m = y.shape[0]
    K = K1 + K2 + sigma_n_sq * np.eye(m, dtype=np.float64)
    c, lower = sla.cho_factor(K, lower=True)
    alpha = sla.cho_solve((c, lower), y)
    lnlike = -0.5 * float(np.dot(y, alpha)) - float(np.sum(np.log(np.diag(c))))
    if include_const:
        lnlike -= 0.5 * m * np.log(2.0 * np.pi)
    return lnlike


def gp_predict_numpy(
    kernel_name: str,
    pv,
    X_fit,
    y_fit,
    X_test,
    splits=None,
    *,
    legacy_george03_gamma: bool = False,
    legacy_split_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure-NumPy/SciPy GP prediction returning (mu, err, mu_time, mu_pos)."""
    X_tr = np.asarray(X_fit, dtype=np.float64)
    X_te = np.asarray(X_test, dtype=np.float64)
    y = np.asarray(y_fit, dtype=np.float64)

    K1_tr, K2_tr, sigma_n_sq = kernel_matrices_numpy(
        kernel_name,
        pv,
        X_tr,
        None,
        splits=splits,
        legacy_george03_gamma=legacy_george03_gamma,
        legacy_split_mask=legacy_split_mask,
    )
    m = y.shape[0]
    K = K1_tr + K2_tr + sigma_n_sq * np.eye(m, dtype=np.float64)
    c, lower = sla.cho_factor(K, lower=True)
    alpha = sla.cho_solve((c, lower), y)

    K1_s, K2_s, _ = kernel_matrices_numpy(
        kernel_name,
        pv,
        X_te,
        X_tr,
        splits=splits,
        legacy_george03_gamma=legacy_george03_gamma,
        legacy_split_mask=legacy_split_mask,
    )
    Ks = K1_s + K2_s
    mu = Ks @ alpha

    K1_ss, K2_ss, _ = kernel_matrices_numpy(
        kernel_name,
        pv,
        X_te,
        None,
        splits=splits,
        legacy_george03_gamma=legacy_george03_gamma,
        legacy_split_mask=legacy_split_mask,
    )
    Kss = K1_ss + K2_ss + sigma_n_sq * np.eye(X_te.shape[0], dtype=np.float64)
    b = sla.cho_solve((c, lower), Ks.T)
    cov_diag = np.diag(Kss - Ks @ b)
    err = np.sqrt(np.maximum(0.0, cov_diag))

    mu_time_raw = K1_s @ alpha
    mu_pos_raw = K2_s @ alpha
    m0 = np.nanmedian(mu_pos_raw)
    mu_pos = mu_pos_raw - m0
    mu_time = mu_time_raw + m0
    return mu, err, mu_time, mu_pos


def gp_2x2_cramers_oracle(
    kernel_name: str,
    pv,
    X_fit,
    y_fit,
    splits=None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Closed-form 2x2 determinant & Cramer's-rule oracle (zero Cholesky calls)."""
    X_tr = np.asarray(X_fit, dtype=np.float64)
    y = np.asarray(y_fit, dtype=np.float64)
    assert X_tr.shape[0] == 2 and y.shape == (2,)

    K1, K2, sigma_n_sq = kernel_matrices_numpy(kernel_name, pv, X_tr, None, splits=splits)
    K = K1 + K2 + sigma_n_sq * np.eye(2, dtype=np.float64)
    det_k = float(K[0, 0] * K[1, 1] - K[0, 1] * K[1, 0])
    alpha = np.array(
        [
            (K[1, 1] * y[0] - K[0, 1] * y[1]) / det_k,
            (-K[1, 0] * y[0] + K[0, 0] * y[1]) / det_k,
        ],
        dtype=np.float64,
    )
    lnlike = -0.5 * float(np.dot(y, alpha)) - 0.5 * float(np.log(det_k)) - float(np.log(2.0 * np.pi))
    return K, lnlike, alpha
