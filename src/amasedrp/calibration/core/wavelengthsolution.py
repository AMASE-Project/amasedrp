#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         wavelengthsolution.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Per-fiber pixel-to-wavelength solution.

A WavelengthSolution stores one polynomial per fiber.  Evaluating it at a row
index gives the wavelength of that row, which is what turns an extracted
spectrum into a calibrated one.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import Legendre
from numpy.typing import NDArray

__all__ = ["POLY_FAMILIES", "WavelengthSolution"]

# Polynomial families a solution may be expressed in.  ``wavelength_calibration``
# converts its fit to the default domain of the family, so a solution is
# evaluated with the family defaults.
POLY_FAMILIES = {
    "legendre": Legendre,
    "polynomial": Polynomial,
}


class WavelengthSolution:
    """Polynomial mapping from row index to wavelength, one per fiber.

    Parameters
    ----------
    coeffs
        Polynomial coefficients, shape ``(n_fibers, degree + 1)``.
    fiber_ids
        Fiber ID of each row of *coeffs*.
    poly_kind
        Polynomial family, ``"legendre"`` or ``"polynomial"``.
    scores
        Fitting score per fiber.  A negative score marks a fiber that inherited
        the solution of its neighbour instead of being fitted on its own
        spectrum.

    Raises
    ------
    ValueError
        If *poly_kind* is unknown, if *coeffs* is not 2-D, or if the shape of
        *fiber_ids* or *scores* does not match *coeffs*.

    Examples
    --------
    >>> solution = WavelengthSolution(
    ...     coeffs=np.array([[5000.0, 0.5]]),
    ...     fiber_ids=np.array([0]),
    ...     scores=np.array([0.01]),
    ... )
    >>> solution.eval(np.array([0, 2]))
    array([[5000., 5001.]])
    """

    def __init__(
        self,
        coeffs: NDArray[np.floating],
        fiber_ids: NDArray[np.integer],
        poly_kind: str = "legendre",
        scores: NDArray[np.floating] | None = None,
    ) -> None:
        if poly_kind not in POLY_FAMILIES:
            raise ValueError(
                f"poly_kind must be one of {sorted(POLY_FAMILIES)}, "
                f"got {poly_kind!r}"
            )

        self.coeffs = np.asarray(coeffs, dtype=float)
        if self.coeffs.ndim != 2:
            raise ValueError(
                f"coeffs must be 2-D, got shape {self.coeffs.shape}"
            )

        n_fibers = self.coeffs.shape[0]
        self.fiber_ids = np.asarray(fiber_ids, dtype=int)
        if self.fiber_ids.shape != (n_fibers,):
            raise ValueError(
                f"fiber_ids shape {self.fiber_ids.shape} must be ({n_fibers},)"
            )

        if scores is None:
            scores = np.full(n_fibers, np.nan)
        self.scores = np.asarray(scores, dtype=float)
        if self.scores.shape != (n_fibers,):
            raise ValueError(
                f"scores shape {self.scores.shape} must be ({n_fibers},)"
            )

        self.poly_kind = poly_kind

    # ------------------------------------------------------------------ #
    #  Evaluation
    # ------------------------------------------------------------------ #

    def eval(self, rows: NDArray[np.integer]) -> NDArray[np.floating]:
        """Evaluate the solution at the given rows.

        Parameters
        ----------
        rows
            1-D array of row (spectral) indices.

        Returns
        -------
        ndarray
            Array of shape ``(n_fibers, len(rows))`` with wavelengths.
        """
        rows = np.asarray(rows, dtype=float)
        family = POLY_FAMILIES[self.poly_kind]
        n_fibers = self.coeffs.shape[0]
        wavelengths = np.empty((n_fibers, len(rows)), dtype=float)

        for i in range(n_fibers):
            wavelengths[i, :] = family(self.coeffs[i])(rows)

        return wavelengths

    def wavelength_at(self, fiber_index: int, wl: float, **kwargs) -> float:
        """Return the row index of a wavelength, by bisection.

        Parameters
        ----------
        fiber_index
            Index into *coeffs*, not a fiber ID.
        wl
            Target wavelength.
        **kwargs
            Forwarded to
            :func:`~amasedrp.calibration.methods.wavelength_calibration.inv_poss_poly`,
            for example ``y_min`` and ``y_max``.

        Returns
        -------
        float
            Row index of *wl*.
        """
        from ..methods.wavelength_calibration import inv_poss_poly

        family = POLY_FAMILIES[self.poly_kind]
        return inv_poss_poly(
            family(self.coeffs[fiber_index]), wl, **kwargs
        )

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def n_fibers(self) -> int:
        """Number of fibers in the solution."""
        return self.coeffs.shape[0]

    @property
    def degree(self) -> int:
        """Polynomial degree."""
        return self.coeffs.shape[1] - 1

    @property
    def valid(self) -> NDArray[np.bool_]:
        """Mask of fibers that were fitted rather than inherited."""
        return self.scores >= 0

    @property
    def calibrated_fraction(self) -> float:
        """Fraction of fibers that were fitted rather than inherited.

        This is the quantity to check before trusting a reduction.  During a
        focus sweep it is allowed to fall far, because the arc lines broaden
        and blend at strong defocus.  A real observation must keep it close to
        one; see ``AGENTS.md``.
        """
        if self.scores.size == 0:
            return 0.0
        return float(np.count_nonzero(self.scores >= 0) / self.scores.size)
