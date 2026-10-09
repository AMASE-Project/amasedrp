#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Where the pipeline writes its products.

One module owns the product tree, so no call site invents a path.  The layout
follows the MaNGA DRP convention, adapted to AMASE-P observation keys::

    <products_dir>/<drpver>/<night>/                 one night
    <products_dir>/<drpver>/<night>/calibration/     nightly masters and state
    <products_dir>/<drpver>/<night>/<obsid>/         one pointing sequence
    <products_dir>/<drpver>/<night>/<obsid>/qa/      its quality assessment

``drpver`` is the version that produced the products, so reprocessing a night
writes beside the previous reduction instead of over it.  Science products are
named after their AMASE data level, their channel and their exposure::

    L1-blue-0001.fits

Calibration products are shared by every observation of a night, because the
fast pipeline may reuse the previous night's calibration.  That is why they
belong in ``calibration/`` next to the ``<obsid>`` directories rather than
inside one of them.

NOTE: that placement is not settled.  MaNGA keeps its masters and calibration
state inside the per-observation directory, because one MaNGA MJD observes one
plate once, so it has no shared calibration to hoist.  AMASE-P may reuse a
previous night's calibration, which argues for the night level.  Both are
reachable through ``product_dir(..., subdir="calibration")``, so choosing one
later does not change this module's API.

Which level holds what is AMASE's definition, not this module's: L1 is
pre-processed, L2 extracted and wavelength-calibrated, L3 flux-calibrated and
sky-subtracted.

Examples
--------
>>> product_dir("/archive/products", "20250705", obsid="p0001", drpver="0.1.0", create=False)
PosixPath('/archive/products/0.1.0/20250705/p0001')
>>> product_name("L2", channel="blue", exposure="0001")
'L2-blue-0001.fits'
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

__all__ = ["LEVELS", "product_dir", "product_name"]

#: The AMASE data levels, in pipeline order.
LEVELS = ("L0", "L1", "L2", "L3")


def _drp_version() -> str:
    """Return the version of the installed amasedrp.

    Returns
    -------
    str
        The version, as recorded by the installation metadata.

    Raises
    ------
    RuntimeError
        If the package is not installed, so the version is unknown.  The
        products would land in a directory named after nothing, so this
        fails loudly instead of guessing.
    """
    try:
        return importlib.metadata.version("amasedrp")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "cannot determine the DRP version because amasedrp is not "
            "installed; pass drpver=... explicitly, or install the package "
            "with `pip install -e .`."
        ) from exc


def _token(value: object, what: str) -> str:
    """Return *value* as a single path component.

    Parameters
    ----------
    value
        The value to check.  It is converted with :class:`str`.
    what
        Name of the argument, used in the error message.

    Returns
    -------
    str
        The component.

    Raises
    ------
    ValueError
        If the value is empty, or is not one path component.  A ``night`` of
        ``"a/b"`` would otherwise silently build a directory two levels deep.
    """
    text = str(value)
    if not text or text in {".", ".."} or Path(text).name != text:
        raise ValueError(f"{what} must be a single path component, got {value!r}.")
    return text


def product_dir(
    products_dir: str | Path,
    night: str,
    *,
    obsid: str | None = None,
    subdir: str | None = None,
    drpver: str | None = None,
    create: bool = True,
) -> Path:
    """Return a directory of the product tree.

    Parameters
    ----------
    products_dir
        Root of the product tree, for example ``data/products``.
    night
        The night the products belong to, for example ``"20250705"``.
    obsid
        Identifier of one observation of that night.  ``None`` stops at the
        night, which is where ``subdir="calibration"`` belongs.
    subdir
        Optional directory inside the result, for example ``"qa"``.
    drpver
        Pipeline version to nest under.  ``None`` uses the version of the
        installed package.
    create
        Create the directory when it does not exist.

    Returns
    -------
    pathlib.Path
        The requested directory.

    Raises
    ------
    ValueError
        If any component is not a single path component, or if the resulting
        path exists and is not a directory.
    RuntimeError
        If *drpver* is ``None`` and the installed version is unknown.
    """
    if drpver is None:
        drpver = _drp_version()

    root = Path(products_dir).expanduser() / _token(drpver, "drpver")
    root = root / _token(night, "night")
    if obsid is not None:
        root = root / _token(obsid, "obsid")
    if subdir is not None:
        root = root / _token(subdir, "subdir")

    if root.exists() and not root.is_dir():
        raise ValueError(f"{root} exists and is not a directory.")
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def product_name(
    level: str,
    *,
    channel: str | None = None,
    exposure: str | None = None,
    ext: str = ".fits",
) -> str:
    """Return the file name of one science product.

    Parameters
    ----------
    level
        Data level, one of :data:`LEVELS`.
    channel
        Spectrograph channel, for example ``"blue"``.  Omitted when the
        product covers every channel.
    exposure
        Exposure identifier, for example ``"0001"``.  Omitted for a product
        that combines several exposures.
    ext
        File extension, including the dot.

    Returns
    -------
    str
        The file name, for example ``"L2-blue-0001.fits"``.

    Raises
    ------
    ValueError
        If *level* is not an AMASE data level, or if *channel* or *exposure*
        is not a single path component.

    Notes
    -----
    Calibration products such as a master bias are not data levels, and this
    function does not name them.  Where they go and what they are called is not
    settled; see the note in the module docstring.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}.")

    parts = [level]
    if channel is not None:
        parts.append(_token(channel, "channel"))
    if exposure is not None:
        parts.append(_token(exposure, "exposure"))
    return "-".join(parts) + ext
