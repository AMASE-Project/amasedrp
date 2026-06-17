# Architecture of `amasedrp`

This document describes the high-level module organization of the `amasedrp` data reduction pipeline (DRP).

## Design Philosophy

The project follows a **Three-Part Module Pattern** designed for clarity:

- **`core/`** — Data structures and containers (e.g., `Image`, `RSS`).
- **`methods/`** — Atomic, single-purpose processing steps (e.g., bias subtraction, cosmic ray removal).
- **Top-level `.py`** — Orchestrator that chains `methods/` into a complete pipeline for end users.

This mirrors the scientist's mental model: **data → step-by-step operations → full workflow**. Modules are organized by **scientific function**, not by code type (classes vs. functions).

## Pipeline Stages

The DRP is organized into three consecutive stages that follow the natural data flow from raw detector frames to science-ready spectra:

- **Stage 1: Image Pre-processing** — `preprocessing/`
  Transform raw CMOS frames into calibrated 2D images (bias/dark/flat/cosmic).

- **Stage 2: Spectral Data Reduction** — `reduction/`
  Transform calibrated 2D images into extracted 1D spectra (fiber tracing, boxcar/optimal extraction, spectro-perfectionism).

- **Stage 3: Spectral Data Calibration** — `calibration/`
  Transform extracted spectra into wavelength-calibrated, sky-subtracted, flux-calibrated products (wavelength calibration, fiber flat-fielding, LSF modeling, sky subtraction, flux calibration, coaddition).

Each stage follows the same **Three-Part Module Pattern** (`core/` → `methods/` → orchestrator).

## Directory Structure

```
src/amasedrp/
│
├── preprocessing/            # Stage 1: Image Pre-processing
│   ├── __init__.py           # Public API: `image_preprocessing`, `preprocessing`
│   ├── core/                 # Data structures
│   │   └── image.py          # Image container: data + FITS header + I/O
│   ├── methods/              # Atomic processing steps
│   │   ├── bias.py           # Bias subtraction
│   │   ├── dark.py           # Dark current subtraction
│   │   ├── flat.py           # Pixel flat-field correction
│   │   └── cosmic.py         # Cosmic ray detection & removal
│   └── image_preprocessing.py# Main entry: orchestrates steps from methods/
│
├── reduction/                # Stage 2: Spectral Data Reduction
│   ├── __init__.py           # Public API: `run_reduction`, `run_quick_reduction`
│   ├── core/                 # Data structures
│   │   ├── rss.py            # Row-Stacked Spectra data model
│   │   └── fiber.py          # Fiber metadata container
│   ├── methods/              # Atomic processing steps
│   │   ├── fiber_tracing.py  # Fiber identification & trace modeling
│   │   └── extraction.py     # Spectral extraction (boxcar / optimal / spectro-perfectionism)
│   └── reduction.py          # Stage 2 main entry: orchestrates spectral extraction
│
├── calibration/              # Stage 3: Master Calibration & Data Calibration
│   ├── __init__.py
│   ├── core/                 # Calibration-specific data models
│   ├── methods/              # Calibration algorithms
│   │   ├── fiberflat.py      # Fiber-to-fiber flat-field correction
│   │   ├── wavelength.py     # Wavelength calibration
│   │   ├── sky.py            # Sky background subtraction
│   │   └── fluxcal.py        # Flux calibration
│   └── calibration.py        # Main entry: master calibration builder
│
├── utils/                    # [UTILITIES] Shared helpers
│   ├── fits_io.py            # FITS read/write wrappers
│   └── logging.py            # Logging configuration
│
└── visualization/            # [QA & PLOTTING]
    ├── plotting.py           # General plots (image, spectra, LSF)
    └── qa_plots.py           # Diagnostic plots for pipeline verification
```

## Module Anatomy

Every functional module (`preprocessing/`, `reduction/`, `calibration/`) follows the same internal layout:

### 1. `core/`

- **Purpose:** Hold data and metadata. Provide I/O and basic properties.
- **Rules:**
  - No business logic (e.g., do not put `detect_cosmic_rays` here).
  - Prefer `dataclass` or simple classes over complex OO hierarchies.
  - Use standard types (`np.ndarray`, `astropy.io.fits.Header`) for interoperability.

**Example:** `preprocessing/core/image.py`
```python
class Image:
    def __init__(self, data, header, filename=None): ...
    def copy(self) -> "Image": ...
    def write_to_fits(self, path: str): ...
    @property
    def exptime(self) -> float | None: ...
```

### 2. `methods/`

- **Purpose:** Implement one scientific step per file.
- **Rules:**
  - Functions are **pure** where possible: input `Image` / `ndarray` → output new object.
  - One file = one step. Keep files small (< 100 lines).
  - No file I/O here. Operate on in-memory objects.

**Example:** `preprocessing/methods/bias.py`
```python
def subtract_bias(image: Image, master_bias: Image) -> Image:
    result = image.copy()
    result.data = result.data.astype(np.float32) - master_bias.data.astype(np.float32)
    result.header.add_history("Bias subtracted")
    return result
```

### 3. Top-level `.py`

- **Purpose:** Provide the **user-facing** entry point.
- **Rules:**
  - Validate inputs, orchestrate `methods/` in the correct order, log progress, write outputs.
  - Keep the public API surface small. Scientists call this one function.

**Example:** `preprocessing/image_preprocessing.py`
```python
def image_calibration(input_image, master_bias, master_dark, master_pixflat, steps=("bias","dark","pixflat")):
    result = input_image.copy()
    if "bias" in steps:
        result = bias.subtract_bias(result, master_bias)
    if "dark" in steps:
        result = dark.subtract_dark(result, master_dark, master_bias)
    if "pixflat" in steps:
        result = flat.apply_pixel_flat(result, master_pixflat, master_bias, master_dark)
    return result
```

---

*See `docs/tutorials/` for Jupyter Notebook guides on using the pipeline.*
