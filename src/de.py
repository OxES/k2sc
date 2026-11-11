"""Deprecated compatibility shim for legacy Storn & Price (1997) Differential Evolution.

.. moduleauthor:: Hannu Parviainen <hpparvi@gmail.com>
"""

from __future__ import annotations

import warnings

import numpy as np


class DiffEvol:
    """Deprecated Differential Evolution optimizer shim (use ``k2sc.optimizers`` instead).

    Uses an isolated ``numpy.random.Generator`` (`np.random.default_rng(seed)`) and
    never mutates global ``np.random`` state.
    """

    def __init__(
        self,
        fun,
        bounds,
        npop: int,
        F: float = 0.5,
        C: float = 0.5,
        seed=0,
        maximize: bool = False,
    ):
        warnings.warn(
            "k2sc.de.DiffEvol is deprecated and will be removed in a future release; "
            "use k2sc.optimizers (or Detrender.optimize) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(seed, np.random.Generator):
            self._rng = seed
        else:
            self._rng = np.random.default_rng(seed)

        self.minfun = fun
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.n_pop = int(npop)
        if self.n_pop < 4:
            raise ValueError("DiffEvol requires population size npop >= 4.")
        self.n_par = int(self.bounds.shape[0])
        self.bl = np.tile(self.bounds[:, 0], (self.n_pop, 1))
        self.bw = np.tile(self.bounds[:, 1] - self.bounds[:, 0], (self.n_pop, 1))
        self.m = -1.0 if maximize else 1.0

        self.seed = seed
        self.F = float(F)
        self.C = float(C)

        self._population = self.bl + self._rng.random((self.n_pop, self.n_par)) * self.bw
        self._fitness = np.zeros(self.n_pop, dtype=np.float64)
        self._minidx: int | None = None

    @property
    def population(self) -> np.ndarray:
        """The parameter vector population."""
        return self._population

    @property
    def minimum_value(self) -> float:
        """The best-fit value of the optimized function."""
        if self._minidx is None:
            raise RuntimeError("DiffEvol has not evaluated any generation yet.")
        return float(self._fitness[self._minidx])

    @property
    def minimum_location(self) -> np.ndarray:
        """The best-fit solution."""
        if self._minidx is None:
            raise RuntimeError("DiffEvol has not evaluated any generation yet.")
        return self._population[self._minidx, :]

    @property
    def minimum_index(self) -> int | None:
        """Index of the best-fit solution."""
        return self._minidx

    def optimize(self, ngen: int):
        """Run the optimizer for ``ngen`` generations."""
        res = None
        for res in self(ngen):
            pass
        return res

    def __call__(self, ngen: int = 1):
        t = np.zeros(3, dtype=int)

        for i in range(self.n_pop):
            self._fitness[i] = self.m * float(self.minfun(self._population[i, :]))

        if ngen <= 0:
            self._minidx = int(np.argmin(self._fitness))
            yield self.population[self._minidx, :], self._fitness[self._minidx]
            return

        for _ in range(int(ngen)):
            for i in range(self.n_pop):
                t[:] = i
                while t[0] == i:
                    t[0] = int(self._rng.integers(0, self.n_pop))
                while t[1] == i or t[1] == t[0]:
                    t[1] = int(self._rng.integers(0, self.n_pop))
                while t[2] == i or t[2] == t[0] or t[2] == t[1]:
                    t[2] = int(self._rng.integers(0, self.n_pop))

                v = self._population[t[0], :] + self.F * (
                    self._population[t[1], :] - self._population[t[2], :]
                )

                crossover = self._rng.random(self.n_par) <= self.C
                u = np.where(crossover, v, self._population[i, :])

                ri = int(self._rng.integers(0, self.n_par))
                u[ri] = v[ri]

                ufit = self.m * float(self.minfun(u))
                if ufit < self._fitness[i]:
                    self._population[i, :] = u.copy()
                    self._fitness[i] = ufit

            self._minidx = int(np.argmin(self._fitness))
            yield self.population[self._minidx, :], float(self._fitness[self._minidx])
