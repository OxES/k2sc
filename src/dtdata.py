"""Encapsulates fluxes, covariates, masks, and training-set sampling for Detrender."""

from __future__ import annotations

import numpy as np

from .utils import medsig


class DtData:
    """Utility class that encapsulates the fluxes and inputs for a detrender."""

    def __init__(self, flux, inputs, mask=None):
        self._flux = np.asarray(flux, dtype=np.float64).copy()
        self._inputs = np.asarray(inputs, dtype=np.float64).copy()

        assert self._flux.ndim == 1, "The flux array for DtData should be 1D [npt]"
        assert self._inputs.ndim == 2, "The input array for DtData should be 2D [npt, >=2]"
        assert self._inputs.shape[1] >= 2, (
            "The input array for DtData should be 2D with shape [npt, >=2]"
        )
        assert self._inputs.shape[0] == self._flux.size, (
            "The input array for DtData must have the same number of rows as flux"
        )

        if mask is None:
            self._mask = np.ones(self._flux.size, dtype=bool)
        else:
            self._mask = np.ones(self._flux.size, dtype=bool)
            self.mask = mask

        self._update_stats()

    def _update_stats(self) -> None:
        mflux = self._flux[self._mask]
        if mflux.size > 0 and np.any(np.isfinite(mflux)):
            fm, fs = medsig(mflux)
            self._fm = float(fm)
            self._fs = float(fs)
        else:
            self._fm = float("nan")
            self._fs = float("nan")

    def create_training_set(
        self,
        nrandom: int = 100,
        nblocks: int = 5,
        bspan: int = 50,
        max_tries: int = 100,
        rng=None,
        seed=None,
    ) -> DtData:
        """Creates a new DtData instance containing a subset of the original data.

        Creates a new DtData instance made of nblocks non-overlapping chunks of bspan
        consecutive datapoints (excluding the masked points) and nrandom datapoints
        drawn randomly from the leftover points. This is done in order to try to include
        both short- and long-time-scale trends to the training set.

        Parameters
        ----------
        nrandom : int, optional
            Number of randomly drawn points.
        nblocks : int, optional
            Number of blocks to draw.
        bspan : int, optional
            Size of a block.
        max_tries : int, optional
            Maximum number of tries to create a consecutive block that doesn't
            overlap with any of the previously created blocks.
        rng : numpy.random.Generator or int, optional
            Random number generator or seed for reproducible sampling.
        seed : int or numpy.random.Generator, optional
            Seed for numpy.random.default_rng when rng is None.

        Returns
        -------
        DtData
            The training dataset as a new DtData instance.
        """
        if isinstance(rng, np.random.Generator):
            gen = rng
        elif rng is not None:
            gen = np.random.default_rng(rng)
        else:
            gen = np.random.default_rng(seed)

        ids = np.where(self.mask)[0]
        nrandom_i = max(0, int(nrandom))
        nblocks_i = max(0, int(nblocks))
        bspan_i = max(0, int(bspan))

        if ids.size == 0:
            trd = DtData(self.unmasked_flux[ids], self.unmasked_inputs[ids])
            trd._fm, trd._fs = self._fm, self._fs
            return trd

        if nblocks_i > 0 and bspan_i > 0 and ids.size > bspan_i:
            bstarts = np.full(nblocks_i, -bspan_i, dtype=int)
            i, j = 0, 0
            max_start = ids.size - bspan_i
            while j < nblocks_i and i <= max_tries:
                idx = int(gen.integers(0, max_start + 1))
                if j == 0 or not np.any(np.abs(idx - bstarts[:j]) < bspan_i):
                    bstarts[j] = idx
                    j += 1
                i += 1

            if j > 0:
                valid_bstarts = np.sort(bstarts[:j])
                block_ids = (
                    valid_bstarts[:, np.newaxis] + np.arange(bspan_i, dtype=int)[np.newaxis, :]
                ).ravel()
            else:
                block_ids = np.empty(0, dtype=int)
        else:
            block_ids = np.empty(0, dtype=int)

        npt_blocks = block_ids.size
        remaining_ids = np.delete(ids, block_ids) if npt_blocks > 0 else ids
        target_random = max(0, nrandom_i + bspan_i * nblocks_i - npt_blocks)
        npt_random = min(remaining_ids.size, target_random)

        idb = ids[block_ids] if npt_blocks > 0 else np.empty(0, dtype=int)
        if npt_random > 0:
            idr = np.sort(gen.permutation(remaining_ids)[:npt_random])
        else:
            idr = np.empty(0, dtype=int)

        tr_ids = np.sort(np.r_[idr, idb].astype(int))
        trd = DtData(self.unmasked_flux[tr_ids], self.unmasked_inputs[tr_ids])
        trd._fm, trd._fs = self._fm, self._fs
        return trd

    @property
    def flux(self) -> np.ndarray:
        return self._flux

    @property
    def nptm(self) -> int:
        return int(np.count_nonzero(self._mask))

    @property
    def nptu(self) -> int:
        return int(self._flux.size)

    @property
    def flux_median(self) -> float:
        return self._fm

    @property
    def flux_std(self) -> float:
        return self._fs

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @mask.setter
    def mask(self, mask) -> None:
        mask_arr = np.asarray(mask)
        assert mask_arr.ndim == 1, "The mask array for DtData should be 1D"
        assert mask_arr.size == self._flux.size, (
            "The mask array for DtData should have the same size as the flux array"
        )
        assert mask_arr.dtype == bool, "The mask array should be boolean"
        self._mask = mask_arr.copy()
        self._update_stats()

    @property
    def masked_time(self) -> np.ndarray:
        return self._inputs[self._mask, 0]

    @property
    def unmasked_time(self) -> np.ndarray:
        return self._inputs[:, 0]

    @property
    def masked_flux(self) -> np.ndarray:
        return self._flux[self._mask]

    @property
    def unmasked_flux(self) -> np.ndarray:
        return self._flux

    @property
    def masked_normalised_flux(self) -> np.ndarray:
        return self._flux[self._mask] / self._fm - 1.0

    @property
    def unmasked_normalised_flux(self) -> np.ndarray:
        return self._flux / self._fm - 1.0

    @property
    def masked_inputs(self) -> np.ndarray:
        return self._inputs[self._mask, :]

    @property
    def unmasked_inputs(self) -> np.ndarray:
        return self._inputs

    @property
    def masked_x(self) -> np.ndarray:
        return self._inputs[self._mask, 1]

    @property
    def unmasked_x(self) -> np.ndarray:
        return self._inputs[:, 1]

    @property
    def masked_y(self) -> np.ndarray:
        if self._inputs.shape[1] >= 3:
            return self._inputs[self._mask, 2]
        return np.zeros(self.nptm, dtype=self._inputs.dtype)

    @property
    def unmasked_y(self) -> np.ndarray:
        if self._inputs.shape[1] >= 3:
            return self._inputs[:, 2]
        return np.zeros(self.nptu, dtype=self._inputs.dtype)

    @property
    def mf(self) -> np.ndarray:
        return self.masked_flux

    @property
    def uf(self) -> np.ndarray:
        return self.unmasked_flux

    @property
    def mi(self) -> np.ndarray:
        return self.masked_inputs

    @property
    def ui(self) -> np.ndarray:
        return self.unmasked_inputs

    @property
    def mt(self) -> np.ndarray:
        return self.masked_time

    @property
    def ut(self) -> np.ndarray:
        return self.unmasked_time

    @property
    def mx(self) -> np.ndarray:
        return self.masked_x

    @property
    def ux(self) -> np.ndarray:
        return self.unmasked_x

    @property
    def my(self) -> np.ndarray:
        return self.masked_y

    @property
    def uy(self) -> np.ndarray:
        return self.unmasked_y

    @property
    def outliers(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (outliers_time, outliers_norm_flux) for excluded points (~mask)."""
        out_mask = ~self._mask
        return self._inputs[out_mask, 0], self._flux[out_mask] / self._fm
