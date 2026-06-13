# Architecture of `amasedrp`

This document describes the high-level structure and organization of the `amasedrp` project.

## Overview

`amasedrp` is the data reduction pipeline (DRP) for the AMASE-P instrument. It is designed to be modular and extensible, with a clear separation of concerns between different stages of the data reduction process. The architecture is organized into several key modules, each responsible for a specific aspect of the DRP.

## Directory Structure

```
amasedrp/
├── .gitignore              # Ignores .fits, .pyc, .ipynb_checkpoints, etc.
├── LICENSE                 # MIT License
├── README.md               # Project overview and quick-start installation guide
├── environment.yml         # Conda environment dependency configuration
├── pyproject.toml          # Python project metadata and installation script definitions
│
├── src/                    # Source Code Directory
│   └── amasedrp/           # Core Source Code
│       ├── __init__.py     # Exposes API; defines __version__
│       ├── core.py         # Top-level control flow: The "Pipeline Runner" e.g., run_drp(), run_quick_drp(), or call functions from reduction/ ...
│       │
│       ├── imageprocessing/ # [IMAGE PROCESSING] e.g., CMOS level
│       │   ├── __init__.py
│       │   ├── image_preprocessing.py # Main function for image-pre-processing (e.g., bias, dark, flat-fielding, cosmic ray, ...)
│       │   ├── classes/
│       │   │   ├── __init__.py
│       │   │   ├── Image.py  # Class for handling CMOS detector image data.
│       │   └── functions/
│       │       ├── __init__.py
│       │       ├── stacking.py  # Functions for median/mean frame stacking
│       │       ├── cleaning.py  # Bad pixel masking and artifact removal
│       │       ├── plots.py
│       │       └── qa.py
│       │
│       ├── extraction/      # [SPECTRAL EXTRACTION] From Images to Spectra
│       │   ├── __init__.py
│       │   ├── classes/
│       │   │   ├── __init__.py
│       │   ├── functions/
│       │   │   ├── __init__.py
│       │   │   ├── plots.py
│       │   │   └── qa.py
│       │   ├── boxcar.py   # Boxcar extraction
│       │   ├── optimal.py      # Optimal Extraction
│       │   └── perfectionism.py   # Spectro-Perfectionism
│       │
│       ├── calibration/    # [CALIBRATION] Wavelength Calibration, Sky Subtraction & Flux Calibration
│       │   ├── __init__.py
│       │   ├── classes/
│       │   │   ├── __init__.py
│       │   ├── functions/
│       │   │   ├── __init__.py
│       │   │   ├── plots.py
│       │   │   └── qa.py
│       │   ├── wave_cal.py   # Wavelength solution using Arc lamps (and Sky lines)
│       │   ├── sky_sub.py      # Sky Background subtraction methods
│       │   └── flux_cal.py         # Flux calibration. i.e., Photometric calibration and sensitivity curves
│       │
│       ├── reduction/  # [CORE REDUCTION] Fast/full reduction, incl. image pre-processing, spectral extraction, data calibration, channel combine, ...
│       │
│       ├── utils/          # [UTILITIES] Helper tools
│       │   ├── __init__.py
│       │   ├── fits_io.py      # Specialized Astropy FITS wrappers and Header handling
│       │   └── astronomy.py    # Coordinate transforms and astronomical constants
│       │
│       └── visualization/  # [QA & PLOTTING]
│           ├── __init__.py
│           ├── plotting.py      # Visulize CMOS detector image, spectra, LSF, ...
│           └── qa_plots.py     # Diagnostic plots for pipeline stage verification
│
├── docs/                   # Documentation and User Guides
│   └── tutorials/          # Jupyter Notebook Tutorials
│       ├── 01_quickstart.ipynb   # e.g., 5-min demo: From Raw data to Final Cube
│       ├── 02_calib_steps.ipynb  # e.g., Generating and checking Master Flats
│       ├── 03_extraction.ipynb   # e.g., Understanding fiber extraction
│       └── 0x_whatever_you_build.ipynb   # e.g., Whatever module or submodule you develop, add a basic tutorial here.
│
├── experiments/            # Sandbox: Algorithm drafts and idea validation
├── tests/                  # Quality Assurance: pytest unit & integration tests
├── data/                   # Local storage: Small FITS slices for testing (e.g., not synced)
└── scripts/                # CLI Scripts: Command-line entry points for users
    └── amase_process_night # Script to process an entire night of observations
```

## Tutorials

See `docs/tutorials/` for Jupyter Notebooks that demonstrate how to use the DRP, from quick-start guides to detailed explanations of specific modules.
