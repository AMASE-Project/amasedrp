#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File:         fibermap.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Per-fiber metadata container using astropy.table.Table.

FiberMap is the structured output of fiber identification.  It holds one
row per fiber with columns describing the fiber's identity, location, and
QA status.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy.table import Table
from numpy.typing import NDArray

__all__ = ["FiberMap"]


class FiberMap(Table):
    """Table of per-fiber metadata.

    Columns
    -------
    FIBERID : int
        Global 0-indexed fiber identifier.
    BLOCKID : int
        Block index to which the fiber belongs.
    BLOCK_LOCAL_ID : int
        0-indexed position of the fiber within its block.
    APPROX_X : float
        Approximate cross-dispersion position (pixel) from peak detection.
    CENTER_ROW : int
        Row at which the fiber was identified.
    VALID : bool
        Whether the fiber passes QA (``True`` by default).

    Examples
    --------
    >>> fm = FiberMap.from_arrays(
    ...     fiber_ids=np.arange(3),
    ...     block_ids=np.array([0, 0, 1]),
    ...     approx_x=np.array([100.0, 108.0, 200.0]),
    ...     center_row=1024,
    ... )
    >>> fm.n_fibers
    3
    """

    # Column names that must be present for a valid FiberMap
    _required_columns = (
        "FIBERID",
        "BLOCKID",
        "BLOCK_LOCAL_ID",
        "APPROX_X",
        "CENTER_ROW",
        "VALID",
    )

    @classmethod
    def from_arrays(
        cls,
        fiber_ids: NDArray[np.integer],
        block_ids: NDArray[np.integer],
        approx_x: NDArray[np.floating],
        center_row: int,
        valid: NDArray[np.bool_] | None = None,
    ) -> "FiberMap":
        """Create a FiberMap from 1-D arrays.

        Parameters
        ----------
        fiber_ids
            Global fiber IDs (must be unique).
        block_ids
            Block ID for each fiber.
        approx_x
            Approximate cross-dispersion position (pixel).
        center_row
            Row at which identification was performed.
        valid
            Optional boolean array.  Defaults to all ``True``.

        Returns
        -------
        FiberMap
            A new table with one row per fiber.
        """
        n_fibers = len(fiber_ids)
        if len(block_ids) != n_fibers or len(approx_x) != n_fibers:
            raise ValueError(
                "fiber_ids, block_ids, and approx_x must have the same length."
            )

        if valid is None:
            valid = np.ones(n_fibers, dtype=bool)

        # Auto-compute BLOCK_LOCAL_ID per block
        block_local_ids = np.empty(n_fibers, dtype=int)
        for bid in np.unique(block_ids):
            mask = block_ids == bid
            block_local_ids[mask] = np.arange(mask.sum())

        data = {
            "FIBERID": np.asarray(fiber_ids, dtype=int),
            "BLOCKID": np.asarray(block_ids, dtype=int),
            "BLOCK_LOCAL_ID": block_local_ids,
            "APPROX_X": np.asarray(approx_x, dtype=float),
            "CENTER_ROW": np.full(n_fibers, int(center_row), dtype=int),
            "VALID": np.asarray(valid, dtype=bool),
        }
        return cls(data)

    # ------------------------------------------------------------------ #
    #  Convenience properties
    # ------------------------------------------------------------------ #

    @property
    def n_fibers(self) -> int:
        """Number of fibers (rows) in the table."""
        return len(self)

    # ------------------------------------------------------------------ #
    #  Mutations
    # ------------------------------------------------------------------ #

    def mark_invalid(self, fiber_ids: list[int] | NDArray[np.integer]) -> None:
        """Mark the specified fibers as invalid.

        Parameters
        ----------
        fiber_ids
            List or array of FIBERID values to mark invalid.
        """
        ids = np.asarray(fiber_ids)
        mask = np.isin(self["FIBERID"], ids)
        self["VALID"][mask] = False

    def get_block(self, block_id: int) -> "FiberMap":
        """Return a sub-table containing only fibers from *block_id*.

        Parameters
        ----------
        block_id
            Block index to extract.

        Returns
        -------
        FiberMap
            A new table with rows where ``BLOCKID == block_id``.
        """
        mask = self["BLOCKID"] == block_id
        if not mask.any():
            raise ValueError(f"Block {block_id} not found in FiberMap.")
        return self[mask]

    # ------------------------------------------------------------------ #
    #  Validation helpers
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        """Sanity-check the table contents.

        Raises
        ------
        ValueError
            If required columns are missing or data is inconsistent.
        """
        missing = [c for c in self._required_columns if c not in self.colnames]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if not np.all(np.diff(self["FIBERID"]) >= 0):
            raise ValueError("FIBERID must be monotonically increasing.")

        if len(np.unique(self["FIBERID"])) != len(self):
            raise ValueError("FIBERID values must be unique.")
