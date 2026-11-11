"""Deprecated compatibility shim for SegmentedDetrender.

Use :class:`k2sc.detrender.Detrender` with ``splits=...`` instead.
"""

from __future__ import annotations

import warnings

import numpy as np

from .dtdata import DtData
from .gp import SplitGP
from .kernels import BasicKernel

try:
    from .detrender import Detrender as _BaseDetrender
except (ImportError, AttributeError):  # pragma: no cover - fallback before PR 4
    _BaseDetrender = object

warnings.warn(
    "k2sc.segdetrender is deprecated and will be removed in a future release; "
    "use k2sc.detrender.Detrender with splits=... instead.",
    DeprecationWarning,
    stacklevel=2,
)


class SegmentedDetrender(_BaseDetrender):
    """Deprecated compatibility wrapper around :class:`k2sc.detrender.Detrender`."""

    def __init__(
        self,
        flux,
        inputs,
        splits=None,
        mask=None,
        p0=None,
        kernel=None,
        tr_nrandom: int = 100,
        tr_bspan: int = 50,
        tr_nblocks: int = 5,
    ):
        warnings.warn(
            "SegmentedDetrender is deprecated; use Detrender(..., splits=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        splits_list = [] if splits is None else list(splits)
        self.splits = splits_list
        self.data = DtData(flux, inputs, mask=mask)
        self.kernel = kernel if kernel is not None else BasicKernel(p0=p0)
        self.gp = SplitGP(self.kernel, splits=splits_list)
        self.tr_data = self.data.create_training_set(
            nrandom=tr_nrandom,
            nblocks=tr_nblocks,
            bspan=tr_bspan,
        )
        self.gp.set_inputs(self.tr_data.masked_inputs)

    def _get_mask(self, t1, t2, splits=None) -> np.ndarray:
        """Create a boolean piecewise-segment covariance mask from split timestamps."""
        active_splits = self.splits if splits is None else splits
        t1_arr = np.asarray(t1, dtype=np.float64).ravel()
        t2_arr = np.asarray(t2, dtype=np.float64).ravel()
        if not active_splits:
            return np.ones((t1_arr.size, t2_arr.size), dtype=bool)
        splits_arr = np.sort(np.asarray(active_splits, dtype=np.float64).ravel())
        seg1 = np.sum(t1_arr[:, None] > splits_arr[None, :], axis=1)
        seg2 = np.sum(t2_arr[:, None] > splits_arr[None, :], axis=1)
        return seg1[:, None] == seg2[None, :]

    def compute_cmat(
        self,
        pv,
        X1,
        X2=None,
        separate: bool = False,
        add_wn: bool = False,
        splits=None,
    ):
        if splits is not None:
            self.gp.splits = splits
        return self.gp._covariance_matrix(
            X1,
            X2,
            pv=pv,
            separate=separate,
            add_wn=add_wn,
        )

    def negll(self, pv=None, flux=None, inputs=None, splits=None) -> float:
        if splits is not None:
            self.gp.splits = splits
        y = flux if flux is not None else self.data.masked_normalised_flux
        x = inputs if inputs is not None else self.data.masked_inputs
        return self.gp.negll(pv, y=y, x=x)

    def compute_gp(self, pv=None, inputs=None) -> None:
        x = inputs if inputs is not None else self.data.masked_inputs
        self.gp.compute(x=x, pv=pv)

    def predict(
        self,
        pv,
        flux=None,
        inputs=None,
        inputs_pred=None,
        mean_only: bool = True,
        splits=None,
    ):
        if splits is not None:
            self.gp.splits = splits
        y = flux if flux is not None else self.data.masked_normalised_flux
        x_tr = inputs if inputs is not None else self.data.masked_inputs
        x_pr = inputs_pred if inputs_pred is not None else x_tr
        self.gp.compute(x=x_tr, pv=pv)
        return self.gp.predict(x_pr, mean_only=mean_only, y=y)
