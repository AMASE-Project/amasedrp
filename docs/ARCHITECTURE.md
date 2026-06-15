# Architecture of `amasedrp`

This document describes the high-level module organization of the `amasedrp` data reduction pipeline (DRP).

## Design Philosophy

The project follows a **Three-Part Module Pattern** designed for clarity:

- **`core/`** — Data structures and containers (e.g., `Image`, `RSS`).
- **`methods/`** — Atomic, single-purpose processing steps (e.g., bias subtraction, cosmic ray removal).
- **Top-level `.py`** — Orchestrator that chains `methods/` into a complete pipeline for end users.

This mirrors the scientist's mental model: **data → step-by-step operations → full workflow**. Modules are organized by **scientific function**, not by code type (classes vs. functions).

## Directory Structure

```
src/amasedrp/
│
├── imageprocessing/          # [IMAGE PROCESSING] CMOS detector level
│   ├── __init__.py           # Public API: `image_preprocessing`, `image_calibration`
│   ├── core/                 # Data structures
│   │   └── image.py          # Image container: data + FITS header + I/O
│   ├── methods/              # Atomic processing steps
│   │   ├── bias.py           # Bias subtraction
│   │   ├── dark.py           # Dark current subtraction
│   │   ├── flat.py           # Pixel flat-field correction
│   │   └── cosmic.py         # Cosmic ray detection & removal
│   └── image_preprocessing.py # Main entry: orchestrates steps from methods/
│
├── reduction/                # [CORE REDUCTION] Full / quick pipeline
│   ├── __init__.py           # Public API: `run_reduction`, `run_quick_reduction`
│   ├── core/                 # Data structures
│   │   ├── rss.py            # Row-Stacked Spectra data model
│   │   └── fiber.py          # Fiber metadata container
│   ├── methods/              # Atomic processing steps
│   │   ├── extraction.py     # Spectral extraction (boxcar / optimal)
│   │   ├── wavelength.py     # Wavelength calibration
│   │   ├── sky.py            # Sky background subtraction
│   │   └── fluxcal.py        # Flux calibration
│   └── reduction.py          # Main entry: orchestrates the full DRP
│
├── calibration/              # [CALIBRATION] Master frame generation & QC
│   ├── __init__.py
│   ├── core/                 # Calibration-specific data models
│   ├── methods/              # Arc line fitting, LSF modeling, etc.
│   └── calibration.py        # Main entry: master calibration builder
│
├── extraction/               # [SPECTRAL EXTRACTION] (to be merged into reduction/)
│   ├── __init__.py
│   ├── core/
│   ├── methods/
│   └── extraction.py
│
├── utils/                    # [UTILITIES] Shared helpers
│   ├── fits_io.py            # FITS read/write wrappers
│   └── logging.py            # Logging configuration
│
└── visualization/            # [QA & PLOTTING]
    ├── plotting.py           # General plots (image, spectra, LSF)
    └── qa_plots.py         # Diagnostic plots for pipeline verification
```

## Module Anatomy

Every functional module (`imageprocessing/`, `reduction/`, `calibration/`) follows the same internal layout:

### 1. `core/`

- **Purpose:** Hold data and metadata. Provide I/O and basic properties.
- **Rules:**
  - No business logic (e.g., do not put `detect_cosmic_rays` here).
  - Prefer `dataclass` or simple classes over complex OO hierarchies.
  - Use standard types (`np.ndarray`, `astropy.io.fits.Header`) for interoperability.

**Example:** `imageprocessing/core/image.py`
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

**Example:** `imageprocessing/methods/bias.py`
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

**Example:** `imageprocessing/image_preprocessing.py`
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
