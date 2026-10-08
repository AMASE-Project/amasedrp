#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collimator focus sweep of 2025-07-05: line width against focus position.

Runs the full pipeline on every collimator position of both channels and
measures the width of three isolated ThAr lines per fiber.  The width of an
unresolved line is the line-spread function, so the position with the smallest
width is the best collimator focus.

Products go under ``<observation> /reduced`` unless ``--output-dir`` is given.
Raw frames are never written to.

Usage:

    python experiments/run_collimator_sweep.py
    python experiments/run_collimator_sweep.py --output-dir /tmp/sweep --force
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Ensure the package is importable when run standalone
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amasedrp.calibration import (  # noqa: E402
    apply_wavelength_solution,
    fit_line_spread_function,
    solve_wavelength_solution,
    thar_lines,
)
from amasedrp.observation import (  # noqa: E402
    LAMP_ARC,
    LAMP_FLAT,
    collimator_position,
    load_frame,
    reduced_dir,
    select_frames,
)
from amasedrp.preprocessing.methods.bias import subtract_bias  # noqa: E402
from amasedrp.preprocessing.methods.master import combine_master  # noqa: E402
from amasedrp.reduction.reduction import (  # noqa: E402
    extract_spectra,
    identify_and_trace_fibers,
)

DEFAULT_OBSERVATION = "~/data/Observations/20250705-collimator_sweep"

# Channel name and the subdirectory that holds its bias frames.
CHANNELS = (("blue", "Blue"), ("red", "Red"))

N_BLOCKS_EXPECTED = 19
N_FIBERS_PER_BLOCK_EXPECTED = 29
APERTURE_RADIUS = 3

# The bias frames carry no RDNOISE card, so the read noise stays the value the
# prototype assumed.  The gain comes from EGAIN through Image.gain.
READ_NOISE = 1.0

# A wavelength solution with a larger score than this is not trusted, and its
# measurements are left out of the figures.
SCORE_THRESHOLD = 0.1


class PositionResult(NamedTuple):
    """Line widths measured at one collimator position.

    Attributes
    ----------
    channel
        Spectrograph channel.
    col_foc
        Collimator position, in steps.
    target_wls
        Measured wavelengths, shape ``(n_wls,)``.
    fwhm
        Line width, shape ``(n_wls, n_fibers)``.
    block_id
        Block of each fiber, shape ``(n_fibers,)``.
    approx_x
        Cross-dispersion position of each fiber, shape ``(n_fibers,)``.
    score
        Score of each fiber's wavelength solution, shape ``(n_fibers,)``.
    """

    channel: str
    col_foc: int
    target_wls: np.ndarray
    fwhm: np.ndarray
    block_id: np.ndarray
    approx_x: np.ndarray
    score: np.ndarray


def build_master_bias(
    observation_dir: Path,
    channel: str,
    subdirectory: str,
    root: Path,
    shape: tuple[int, int],
    force: bool = False,
):
    """Return the master bias of a channel, building it when needed.

    Parameters
    ----------
    observation_dir
        Directory of the observation.
    channel
        Spectrograph channel.
    subdirectory
        Directory under ``bias/`` that holds the bias frames.
    root
        Output directory, where the master is cached.
    shape
        Expected frame shape in the pipeline orientation.
    force
        Rebuild the master even when a cached one exists.

    Returns
    -------
    amasedrp.preprocessing.core.image.Image
        The master bias.
    """
    cached = root / f"master_bias_{channel}.fits"
    if cached.exists() and not force:
        print(f"  master bias: reusing {cached.name}")
        return load_frame(cached)

    paths = select_frames(observation_dir / "bias" / subdirectory)
    if not paths:
        raise RuntimeError(f"no bias frames found for channel {channel!r}.")

    print(f"  master bias: combining {len(paths)} frames ...", end="", flush=True)
    started = time.time()
    frames = [load_frame(path, shape=shape) for path in paths]
    master = combine_master(frames)
    master.write_to_fits(str(cached))
    print(f" {time.time() - started:.1f} s -> {cached.name}")
    return master


def process_position(
    observation_dir: Path,
    channel: str,
    col_foc: int,
    master_bias,
    root: Path,
    force: bool = False,
) -> PositionResult | None:
    """Run the pipeline at one collimator position and measure line widths.

    Parameters
    ----------
    observation_dir
        Directory of the observation.
    channel
        Spectrograph channel.
    col_foc
        Collimator position, in steps.
    master_bias
        Master bias of the channel.
    root
        Output directory, where the result is cached.
    force
        Recompute even when a cached result exists.

    Returns
    -------
    PositionResult or None
        The measurements, or ``None`` when the position holds no usable
        arc/flat pair.
    """
    cached = root / f"fwhm_{channel}_colfoc{col_foc:+d}.npz"
    if cached.exists() and not force:
        with np.load(cached) as data:
            return PositionResult(
                channel=channel,
                col_foc=col_foc,
                target_wls=data["target_wls"],
                fwhm=data["fwhm"],
                block_id=data["block_id"],
                approx_x=data["approx_x"],
                score=data["score"],
            )

    sweep = observation_dir / "sweep"
    arcs = select_frames(sweep, channel=channel, lamp=LAMP_ARC, col_foc=col_foc)
    flats = select_frames(sweep, channel=channel, lamp=LAMP_FLAT, col_foc=col_foc)
    if not arcs or not flats:
        warnings.warn(
            f"{channel} at COL-FOC {col_foc:+d}: no arc/flat pair "
            f"({len(arcs)} arc, {len(flats)} flat); position skipped.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    # Orientation is normalised once, against the flat, before anything uses
    # the arrays.  See amasedrp.observation.load_frame.
    flat = subtract_bias(load_frame(flats[0]), master_bias)
    arc = subtract_bias(load_frame(arcs[0], shape=flat.data.shape), master_bias)

    fibermap, tracemask = identify_and_trace_fibers(
        image=flat.data,
        n_blocks_expected=N_BLOCKS_EXPECTED,
        n_fibers_per_block_expected=N_FIBERS_PER_BLOCK_EXPECTED,
        strict=False,
    )
    frame = extract_spectra(
        image=arc.data,
        tracemask=tracemask,
        fibermap=fibermap,
        method="optimal",
        flat_image=flat.data,
        aperture_radius=APERTURE_RADIUS,
        read_noise=READ_NOISE,
    )

    lines = thar_lines(channel)
    solution = solve_wavelength_solution(
        frame, tracemask, known_wls=lines.known_wls, poss_wls=lines.poss_wls,
    )
    calibrated = apply_wavelength_solution(frame, solution)
    lsf = fit_line_spread_function(calibrated, target_wls=lines.lsf_wls)

    result = PositionResult(
        channel=channel,
        col_foc=col_foc,
        target_wls=lsf.target_wls,
        fwhm=lsf.fwhm,
        block_id=np.asarray(fibermap["BLOCKID"], dtype=int),
        approx_x=np.asarray(fibermap["APPROX_X"], dtype=float),
        score=np.asarray(solution.scores, dtype=float),
    )
    np.savez(
        cached,
        target_wls=result.target_wls,
        fwhm=result.fwhm,
        block_id=result.block_id,
        approx_x=result.approx_x,
        score=result.score,
    )
    return result


def run_sweep(
    observation_dir: Path,
    output_dir: Path | None = None,
    force: bool = False,
) -> tuple[Path, list[PositionResult]]:
    """Run every collimator position of both channels.

    Parameters
    ----------
    observation_dir
        Directory of the observation.
    output_dir
        Where to write products.  ``None`` uses ``<observation_dir>/reduced``.
    force
        Recompute even when cached products exist.

    Returns
    -------
    root
        The directory that holds the products.
    results
        One entry per processed position, sorted by channel and position.
    """
    root = reduced_dir(observation_dir, output_dir)
    print(f"products: {root}")

    reference = select_frames(
        observation_dir / "sweep", channel=CHANNELS[0][0], lamp=LAMP_FLAT,
    )
    if not reference:
        raise RuntimeError(
            f"no {CHANNELS[0][0]} flat frame found under "
            f"{observation_dir / 'sweep'}; is this an observation directory?"
        )
    shape = load_frame(reference[0]).data.shape

    results: list[PositionResult] = []
    for channel, subdirectory in CHANNELS:
        print(f"\n=== {channel} ===")
        master = build_master_bias(
            observation_dir, channel, subdirectory, root, shape, force=force,
        )

        flats = select_frames(
            observation_dir / "sweep", channel=channel, lamp=LAMP_FLAT,
        )
        positions = sorted({int(collimator_position(path)) for path in flats})
        print(f"  {len(positions)} collimator positions: {positions}")

        for index, col_foc in enumerate(positions, start=1):
            started = time.time()
            result = process_position(
                observation_dir, channel, col_foc, master, root, force=force,
            )
            if result is None:
                continue
            n_valid = int(np.count_nonzero(np.isfinite(result.fwhm)))
            print(
                f"  [{index:2d}/{len(positions)}] COL-FOC {col_foc:+5d}: "
                f"{n_valid:4d}/{result.fwhm.size} widths "
                f"({time.time() - started:.1f} s)"
            )
            results.append(result)

    return root, results


def plot_summary(results: list[PositionResult], path: Path) -> None:
    """Write the headline figure: resolution against collimator position.

    One panel per channel and measured line, holding the median resolving power
    over fibers, with the 16-84 percentile spread as the error bar.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = sorted({result.channel for result in results})
    # Share the y-axis along a row, so that the three lines of one channel are
    # read on one scale instead of three auto-scaled ones.
    fig, axes = plt.subplots(
        len(channels), 3, figsize=(15.0, 8.0),
        sharex=True, sharey="row", squeeze=False,
    )

    for row, channel in enumerate(channels):
        channel_results = [r for r in results if r.channel == channel]
        for column in range(3):
            ax = axes[row][column]
            for result in channel_results:
                if column >= result.target_wls.size:
                    continue
                good = (
                    np.isfinite(result.fwhm[column])
                    & (result.score >= 0.0)
                    & (result.score < SCORE_THRESHOLD)
                )
                if not good.any():
                    continue
                resolution = result.target_wls[column] / result.fwhm[column][good]
                ax.errorbar(
                    result.col_foc,
                    float(np.median(resolution)),
                    yerr=[[float(np.median(resolution) - np.percentile(resolution, 16))],
                          [float(np.percentile(resolution, 84) - np.median(resolution))]],
                    fmt="o", capsize=3, color="tab:blue",
                )
            if column < len(channel_results[0].target_wls):
                ax.set_title(
                    f"{channel}: "
                    f"{channel_results[0].target_wls[column]:.2f} A"
                )
            ax.set_xlabel("collimator position")
            ax.grid(axis="y")
    axes[0][0].set_ylabel("R = wavelength / FWHM")
    for row in range(len(channels)):
        axes[row][0].set_ylabel("R = wavelength / FWHM")

    fig.suptitle("AMASE-P collimator sweep: resolution against focus")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def match_fibers(
    results: list[PositionResult],
    tolerance: float = 4.0,
) -> dict[str, tuple[int, np.ndarray]]:
    """Match the same physical fiber across collimator positions.

    A fiber cannot be followed by array index: identification finds a slightly
    different number of fibers at every position.  Fibers are matched by block
    and by cross-dispersion position instead.

    Parameters
    ----------
    results
        Measurements of every processed position.
    tolerance
        Largest accepted cross-dispersion shift, in pixels.  Fibers sit about
        8 px apart, so half of that cannot confuse a fiber with its neighbor.

    Returns
    -------
    dict
        Per channel, the index of the reference result and an array of shape
        ``(n_positions, n_matched)`` holding, for each position, the fiber
        index of every matched fiber.
    """
    matched: dict[str, tuple[int, np.ndarray]] = {}
    for channel in sorted({result.channel for result in results}):
        channel_results = [r for r in results if r.channel == channel]
        reference = max(
            range(len(channel_results)),
            key=lambda i: channel_results[i].block_id.size,
        )
        ref = channel_results[reference]

        keep = np.ones(ref.block_id.size, dtype=bool)
        partners_per_position = []
        for result in channel_results:
            partners = np.full(ref.block_id.size, -1, dtype=int)
            for block in np.unique(ref.block_id):
                in_ref = np.where(ref.block_id == block)[0]
                in_position = np.where(result.block_id == block)[0]
                if in_position.size == 0:
                    continue
                distance = np.abs(
                    result.approx_x[in_position][None, :]
                    - ref.approx_x[in_ref][:, None]
                )
                nearest = np.argmin(distance, axis=1)
                close = distance[np.arange(in_ref.size), nearest] <= tolerance
                partners[in_ref[close]] = in_position[nearest[close]]
            keep &= partners >= 0
            partners_per_position.append(partners)

        indices = np.asarray(partners_per_position)[:, keep]
        matched[channel] = (reference, indices)
    return matched


def plot_matched(
    results: list[PositionResult],
    matched: dict[str, tuple[int, np.ndarray]],
    path: Path,
) -> None:
    """Write the resolution figure on a balanced fiber panel.

    Only fibers that are matched at every position *and* yield a trusted
    measurement of the line at every position enter the median.  Without that
    restriction the median is taken over a different fiber set at every
    position, and a defocused position looks better simply because only its
    sharpest fibers survived.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = sorted(matched)
    fig, axes = plt.subplots(
        len(channels), 3, figsize=(15.0, 8.0),
        sharex=True, sharey="row", squeeze=False,
    )

    for row, channel in enumerate(channels):
        channel_results = [r for r in results if r.channel == channel]
        _, indices = matched[channel]
        positions = [r.col_foc for r in channel_results]

        for column in range(3):
            ax = axes[row][column]
            usable = np.ones(indices.shape[1], dtype=bool)
            for position_index, result in enumerate(channel_results):
                if column >= result.target_wls.size:
                    continue
                fibre = indices[position_index]
                usable &= np.isfinite(result.fwhm[column][fibre])
                usable &= result.score[fibre] >= 0.0
                usable &= result.score[fibre] < SCORE_THRESHOLD
            if not usable.any():
                ax.set_title(f"{channel}: line {column + 1} (no balanced panel)")
                continue

            medians, low, high = [], [], []
            for position_index, result in enumerate(channel_results):
                fibre = indices[position_index][usable]
                resolution = result.target_wls[column] / result.fwhm[column][fibre]
                medians.append(float(np.median(resolution)))
                low.append(float(np.percentile(resolution, 16)))
                high.append(float(np.percentile(resolution, 84)))
            medians = np.asarray(medians)
            ax.errorbar(
                positions, medians,
                yerr=[medians - np.asarray(low), np.asarray(high) - medians],
                fmt="o", capsize=3, color="tab:blue",
            )
            best = positions[int(np.argmax(medians))]
            ax.axvline(best, color="tab:red", linestyle=":", linewidth=1.0)
            ax.set_title(
                f"{channel}: {channel_results[0].target_wls[column]:.2f} A "
                f"({usable.sum()} fibers, best {best:+d})"
            )
            ax.set_xlabel("collimator position")
            ax.grid(axis="y")
        axes[row][0].set_ylabel("R = wavelength / FWHM")

    fig.suptitle(
        "AMASE-P collimator sweep: resolution on a balanced fiber panel"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def plot_coverage(results: list[PositionResult], path: Path) -> None:
    """Write the number of fibers with a trusted solution against position.

    The count shows how much of the array contributes to each point of the
    resolution figure.  A focus sweep can lose most fibers at strong defocus;
    a real observation must not.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = sorted({result.channel for result in results})
    fig, axes = plt.subplots(1, len(channels), figsize=(12.0, 4.5), squeeze=False)

    for column, channel in enumerate(channels):
        ax = axes[0][column]
        channel_results = [r for r in results if r.channel == channel]
        positions = [r.col_foc for r in channel_results]
        solved = [int((r.score >= 0).sum()) for r in channel_results]
        trusted = [int(((r.score >= 0) & (r.score < SCORE_THRESHOLD)).sum())
                   for r in channel_results]
        total = [r.score.size for r in channel_results]

        ax.plot(positions, total, "o-", label="fibers identified", color="0.6")
        ax.plot(positions, solved, "o-", label="solution found", color="tab:orange")
        ax.plot(positions, trusted, "o-", label=f"score < {SCORE_THRESHOLD}", color="tab:blue")
        ax.set_title(channel)
        ax.set_xlabel("collimator position")
        ax.set_ylabel("fibers")
        ax.grid(axis="y")
        ax.legend(fontsize=8)

    fig.suptitle("Wavelength solution coverage against collimator position")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def plot_blocks(results: list[PositionResult], directory: Path) -> None:
    """Write one figure per fiber block, to show the field dependence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    channels = sorted({result.channel for result in results})
    n_blocks = max(int(result.block_id.max()) for result in results) + 1

    for block in range(n_blocks):
        fig, axes = plt.subplots(
            len(channels), 3, figsize=(15.0, 8.0),
            sharex=True, sharey="row", squeeze=False,
        )
        plotted = False
        for row, channel in enumerate(channels):
            channel_results = [r for r in results if r.channel == channel]
            for column in range(3):
                ax = axes[row][column]
                for result in channel_results:
                    if column >= result.target_wls.size:
                        continue
                    in_block = result.block_id == block
                    good = (
                        in_block
                        & np.isfinite(result.fwhm[column])
                        & (result.score >= 0.0)
                        & (result.score < SCORE_THRESHOLD)
                    )
                    if not good.any():
                        continue
                    plotted = True
                    resolution = result.target_wls[column] / result.fwhm[column][good]
                    ax.scatter(
                        np.full(resolution.size, result.col_foc), resolution,
                        s=4.0, alpha=0.6, color="tab:blue",
                    )
                if column < len(channel_results[0].target_wls):
                    ax.set_title(
                        f"{channel}: "
                        f"{channel_results[0].target_wls[column]:.2f} A"
                    )
                ax.set_xlabel("collimator position")
                ax.grid(axis="y")
        axes[0][0].set_ylabel("R")
        for row in range(len(channels)):
            axes[row][0].set_ylabel("R")
        fig.suptitle(f"Fiber block {block:02d}")
        fig.tight_layout()
        if plotted:
            # An empty panel would only say the block has no good measurement,
            # which the summary figure already shows.
            path = directory / f"block_{block:02d}.png"
            fig.savefig(path, dpi=150)
            print(f"wrote {path}")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observation-dir", default=DEFAULT_OBSERVATION,
        help="directory of the observation (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="where to write products (default: <observation-dir>/reduced)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="recompute even when cached products exist",
    )
    args = parser.parse_args()

    observation_dir = Path(args.observation_dir).expanduser()
    output_dir = None if args.output_dir is None else Path(args.output_dir)

    started = time.time()
    root, results = run_sweep(observation_dir, output_dir, force=args.force)
    if not results:
        print("no position produced a result; nothing to plot.")
        return 1

    print(f"\nrun finished in {time.time() - started:.1f} s")
    plot_summary(results, root / "resolution_vs_focus.png")
    plot_matched(results, match_fibers(results), root / "resolution_vs_focus_matched.png")
    plot_coverage(results, root / "coverage_vs_focus.png")
    plot_blocks(results, root / "blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
