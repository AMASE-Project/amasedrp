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
├── preprocessing/              # Stage 1: Image Pre-processing
│   ├── __init__.py             # Public API: `image_preprocessing`, `preprocessing`
│   ├── core/                   # Data structures
│   │   └── image.py            # Image container: data + FITS header + I/O
│   ├── methods/                # Atomic processing steps
│   │   ├── bias.py             # Bias subtraction
│   │   ├── dark.py             # Dark current subtraction
│   │   ├── flat.py             # Pixel flat-field correction
│   │   └── cosmic.py           # Cosmic ray detection & removal
│   └── image_preprocessing.py  # Main entry: orchestrates steps from methods/
│
├── reduction/                  # Stage 2: Spectral Data Reduction
│   ├── __init__.py             # Public API: FiberMap, TraceMask, identify_and_trace_fibers
│   ├── core/                   # Data structures & builders
│   │   ├── fibermap.py         # FiberMap: per-fiber metadata table (astropy.Table)
│   │   ├── fiberidentifier.py  # FibersIdentifier: block + fiber detection from flat
│   │   ├── tracemask.py        # TraceMask: Legendre polynomial trace model
│   │   ├── fiberframe.py       # FiberFrame: extracted 2D spectra container (flux, ivar, mask, wave)
│   │   └── fiberprofile.py     # FiberProfile: normalized cross-dispersion PSF model per fiber
│   ├── methods/                # Atomic, stateless processing steps
│   │   ├── fiber_tracing.py    # Barycenter tracing & Legendre fitting helpers
│   │   ├── profile_modeling.py # Build FiberProfile from master flat
│   │   └── extraction.py       # Boxcar & optimal extraction algorithms
│   └── reduction.py            # Stage 2 orchestrator: identify → trace → extract
│
├── calibration/                # Stage 3: Master Calibration & Data Calibration
│   ├── __init__.py
│   ├── core/                   # Calibration-specific data models
│   ├── methods/                # Calibration algorithms
│   │   ├── fiberflat.py        # Fiber-to-fiber flat-field correction
│   │   ├── wavelength.py       # Wavelength calibration
│   │   ├── sky.py              # Sky background subtraction
│   │   └── fluxcal.py          # Flux calibration
│   └── calibration.py          # Main entry: master calibration builder
│
├── utils/                      # [UTILITIES] Shared helpers
│   ├── fits_io.py              # FITS read/write wrappers
│   └── logging.py              # Logging configuration
│
└── visualization/              # [QA & PLOTTING]
    ├── plotting.py             # General plots (image, spectra, LSF)
    └── qa_plots.py             # Diagnostic plots for pipeline verification
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

## Stage 2: `reduction/` — Detailed Design

This section details the internal design of the spectral data reduction stage, informed by the existing `amasedrp` implementation and architectural patterns from `MaNGA DRP`, `lvmdrp`, and `desispec`.

### 2.1 Data Flow

The reduction stage transforms a **2-D calibrated image** (from `preprocessing/`) into a **2-D row-stacked spectrum** (`FiberFrame`) ready for calibration. The pipeline follows a strict linear data flow:

```
Master Flat Image
       │
       ▼
┌─────────────────────┐
│ FibersIdentifier    │  ← core/fiberidentifier.py
│  .identify()        │     Detect blocks & peaks at central row
└──────────┬──────────┘
           ▼
      FiberMap          ← core/fibermap.py
      (fiber_id, block_id, approx_x, valid)
           │
           ▼
┌─────────────────────┐
│ TraceMask           │  ← core/tracemask.py
│ .from_fibermap()    │     Barycenter trace + Legendre fit
└──────────┬──────────┘
           ▼
      TraceMask         ← core/tracemask.py
      (coeffs, domain, eval())
           │
           ├──────────────────────────────────────┐
           ▼                                      ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│ build_fiber_profile()   │          │ Science Image           │
│ (methods/profile_)      │          │ (from preprocessing/)   │
│  modeling.py)           │          └──────────┬──────────────┘
└───────────┬─────────────┘                     │
            ▼                                   ▼
      FiberProfile                        TraceMask.eval()
      (profile, x_offsets)                (trace_positions)
            │                                   │
            └───────────────┬───────────────────┘
                            ▼
                   ┌─────────────────┐
                   │ extract_spectra │  ← methods/extraction.py
                   │ (boxcar/optimal)│
                   └────────┬────────┘
                            ▼
                      FiberFrame        ← core/fiberframe.py
                      (flux, ivar, mask, wave, fibermap)
```

**Key principle:** `core/` objects are passed between steps; `methods/` are stateless functions that operate on them. No method writes to disk.

### 2.2 `core/` — Data Structures

#### `FiberMap` (`core/fibermap.py`) *[IMPLEMENTED]*
- **Base:** `astropy.table.Table`
- **Columns:** `FIBERID`, `BLOCKID`, `BLOCK_LOCAL_ID`, `APPROX_X`, `CENTER_ROW`, `VALID`
- **Role:** Single source of truth for "which fibers exist and where they are."
- **QA:** `mark_invalid()`, `get_block()`, `validate()`

#### `TraceMask` (`core/tracemask.py`) *[IMPLEMENTED]*
- **Storage:** Legendre polynomial coefficients per fiber `(n_fibers, deg+1)`
- **Role:** Compact, sub-pixel model of fiber curvature on the CCD.
- **API:** `eval(rows)` → `(n_fibers, n_rows)` trace positions.
- **Builder:** `TraceMask.from_fibermap(fibermap, image, poly_deg=10)` traces barycenters and fits.

#### `FiberFrame` (`core/fiberframe.py`) *[RECOMMENDED — NEW]*
Inspired by `desispec.Frame` and `lvmdrp.RSS`.
- **Storage:** `flux`, `ivar`, `mask` as `(n_fibers, n_wave)`; `wave` as `(n_wave,)` or `(n_fibers, n_wave)`
- **Role:** The canonical intermediate data product passed from `reduction/` → `calibration/`.
- **Metadata:** `fibermap` (FiberMap), `meta` (FITS header dict)
- **I/O:** `to_fits()`, `from_fits()` — critical for checkpointing pipeline steps.

**Why not `lvmdrp.RSS`?** `RSS` in lvmdrp inherits from a massive `FiberRows` + `Header` hierarchy. We prefer **composition** to keep the class lightweight and explicit.

#### `FiberProfile` (`core/fiberprofile.py`) *[RECOMMENDED — NEW]*
Required for **flat-relative optimal extraction** (see MaNGA `extract_row`).
- **Storage:** `profile[n_fibers, n_rows, n_offsets]`, `x_offsets[n_offsets]`
- **Role:** Normalized cross-dispersion PSF measured from a master flat.
- **Builder:** `FiberProfile.from_flat(flat_image, tracemask, fibermap, half_width=5)`
- **Invariant:** `sum(profile, axis=-1) == 1` for every fiber/row.

### 2.3 `methods/` — Algorithms

#### `fiber_tracing.py` *[IMPLEMENTED — may extend]*
- `trace_fibers_barycenter(image, approx_positions, center_row)` → dense trace array
- `fit_traces_polynomial(traces, poly_deg)` → coefficients + domain
- *Future:* `trace_fibers_gaussian()` for cross-correlation centroiding.

#### `extraction.py` *[STUB — needs implementation]*
- `extract_boxcar(image, trace_positions, aperture_radius)` → flux, ivar
  - Simple aperture sum. Use for QA, quick-look, and as fallback.
- `extract_optimal(image, trace_positions, fiber_profile)` → flux, ivar
  - **MaNGA-style row-by-row profile fitting.** Fit Gaussian amplitudes (fixed sigma/center from flat profile) plus a low-order polynomial background per row. Iterate with sigma-clipping rejection.
- `extract_spectra(image, tracemask, fibermap, method="boxcar", ...)` → `FiberFrame`
  - **High-level orchestrator.** Evaluates trace positions, dispatches to boxcar or optimal, packages result into `FiberFrame`.

**Design note:** The optimal extractor should follow MaNGA's iterative rejection:
1. Fit model to row.
2. Compute residuals.
3. Mask the single worst pixel in each contiguous bad group.
4. Re-fit until convergence or `maxiter`.

#### `profile_modeling.py` *[RECOMMENDED — NEW]*
- `build_fiber_profile(flat_image, tracemask, fibermap, half_width)` → `FiberProfile`
  - For each fiber/row, cut out a cross-dispersion slice centered on the trace, normalize.
- `normalize_profile(profile)` → ensure sum-to-one.

### 2.4 Orchestrator (`reduction.py`)

The public API is intentionally minimal:

```python
# Existing
fibermap, tracemask = identify_and_trace_fibers(
    image=flat_image,
    n_blocks_expected=19,
    n_fibers_per_block_expected=29,
)

# Recommended extension
fiber_profile = build_fiber_profile(flat_image, tracemask, fibermap)

fiber_frame = extract_spectra(
    image=science_image,
    tracemask=tracemask,
    fibermap=fibermap,
    method="optimal",
    fiber_profile=fiber_profile,
)
```

### 2.5 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Separate `FiberProfile` from `TraceMask`** | `TraceMask` answers "where is the fiber?" (geometry). `FiberProfile` answers "what is its shape?" (PSF). Decoupling lets us update one without the other. |
| **`FiberFrame` as 2-D array `(fiber, wave)`** | Matches `desispec.Frame` and `lvmdrp.RSS`. Each row is one fiber's 1-D spectrum. Easy to feed into `calibration/` (wavelength, sky, flux cal). |
| **Optimal extraction uses flat-derived profile** | MaNGA and lvmdrp both do this. It avoids assuming a theoretical Gaussian and adapts to the real instrument PSF. |
| **Boxcar as first-class citizen** | Not just a placeholder. Needed for: (1) quick-look QA, (2) identifying bright fibers before optimal extraction (MaNGA `find_whopping`), (3) fallback when optimal fails. |
| **No `Aperture` class for boxcar** | `lvmdrp` has a complex `Aperture` class with sub-pixel integration. For a first implementation, an integer `aperture_radius` is sufficient and far simpler. Upgrade path: replace the integer with a `PixelAperture` object later. |

---

*See `docs/tutorials/` for Jupyter Notebook guides on using the pipeline.*
