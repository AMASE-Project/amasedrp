# amasedrp

Data reduction pipeline (DRP) for **AMASE-P**, the prototype of the Affordable
Multiple Aperture Spectroscopy Explorer. AMASE plans to replicate small
multi-fiber spectrograph units built from commercial telephoto lenses and CMOS
detectors, to survey ionized gas at R ≈ 15 000. This pipeline turns raw frames
from the prototype into science-ready spectra.

The pipeline has three stages:

- **Pre-processing** (`preprocessing/`) — bias, dark, pixel flat and
  cosmic-ray removal.
- **Reduction** (`reduction/`) — fiber identification, tracing, and boxcar or
  optimal extraction.
- **Calibration** (`calibration/`) — wavelength, fiber flat, sky and flux
  calibration.

## Install

Install into the Python environment you use:

    pip install -e .

Requires Python ≥ 3.10. `environment.yml` lists the same dependencies if you
want a separate conda environment. The pipeline reads FITS frames; the
repository ships no raw data.

## Learn more

- `docs/ARCHITECTURE.md` — module layout and design decisions.
- `docs/tutorials/` — Jupyter walkthroughs. Install Jupyter yourself; it is not
  a pipeline dependency.
- `docs/CONTRIBUTING.md` — development setup, style and test commands.
