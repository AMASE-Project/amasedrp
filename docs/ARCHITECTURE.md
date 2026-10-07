# Architecture of `amasedrp`

This document describes the high-level module organization of the `amasedrp` data reduction pipeline (DRP).

## Status

Derived from the code, not from the CDR document. Update this table when a
module lands.

| Stage | Module | Status |
| --- | --- | --- |
| Pre-processing | Image calibration (bias, dark, pixel flat) | Implemented |
| Pre-processing | Cosmic-ray removal | Implemented |
| Reduction | Fiber identification and tracing | Implemented |
| Reduction | Boxcar extraction | Implemented |
| Reduction | Optimal extraction (FOX) | Stub (raises `NotImplementedError`) |
| Reduction | Spectro-perfectionism | Not started |
| Calibration | Wavelength calibration | Implemented, not wired |
| Calibration | LSF fitting | Implemented, not wired |
| Calibration | Fiber flat-fielding | Planned |
| Calibration | Sky subtraction | Planned |
| Calibration | Flux calibration | Planned |
| Post-processing | Coaddition | Planned |

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

Files move during development. Treat this tree as a sketch; the exports in
each `__init__.py` are the source of truth for the public API.

```
src/amasedrp/
│
├── preprocessing/              # Stage 1: Image Pre-processing
│   ├── __init__.py             # Public API (see file)
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
│   ├── __init__.py             # Public API (see file)
│   ├── core/                   # Data structures & builders
│   │   ├── fibermap.py         # FiberMap: per-fiber metadata table (astropy.Table)
│   │   ├── fiberidentifier.py  # FibersIdentifier: block + fiber detection from flat
│   │   ├── tracemask.py        # TraceMask: Legendre polynomial trace model
│   │   ├── fiberframe.py       # FiberFrame: extracted 2D spectra container (flux, ivar, mask, wave)
│   │   └── fiberprofile.py     # FiberProfile: normalized cross-dispersion PSF model per fiber
│   ├── methods/                # Atomic, stateless processing steps
│   │   ├── fiber_tracing.py    # Barycenter tracing & Legendre fitting helpers
│   │   ├── profile_modeling.py # Build FiberProfile from master flat
│   │   ├── boxcar.py           # Boxcar extraction
│   │   └── optimal.py          # Optimal extraction (stub)
│   └── reduction.py            # Stage 2 orchestrator: identify → trace → extract
│
├── calibration/                # Stage 3: Master Calibration & Data Calibration
│   ├── __init__.py
│   ├── core/                   # Calibration-specific data models
│   ├── methods/                # Calibration algorithms
│   │   ├── fiberflat.py        # Fiber-to-fiber flat-field correction
│   │   ├── wavelength_calibration.py # Wavelength calibration
│   │   ├── lsf_fitting.py      # Line-spread function fitting
│   │   ├── sky.py              # Sky background subtraction
│   │   └── fluxcal.py          # Flux calibration
│   └── calibration.py          # Main entry: master calibration builder
│
├── utils/                      # Shared helpers
│   ├── logging.py              # Logging configuration
│   └── parallel_processing.py  # Parallel execution helpers
│
└── visualization/              # QA & plotting
    └── plotting.py             # General plots (image, spectra, LSF)
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

## Data Products and I/O

### Frames on disk

FITS is the only on-disk format. Three classes own the I/O:

- `Image` (`preprocessing/core/image.py`) reads and writes single 2-D frames.
  `Image.from_fits(filename)` takes the physical unit from the `BUNIT`
  keyword, and assumes ADU when the keyword is absent.
  `Image.write_to_fits(filename)` writes the raw ADU array and sets
  `BUNIT = 'adu'`, so a file on disk never holds electrons.
- `FiberFrame` (`reduction/core/fiberframe.py`) reads and writes extracted
  spectra. `to_fits(path)` writes a primary HDU that holds metadata only
  (`N_FIBERS`, `N_WAVE` and the `meta` dictionary), then one `ImageHDU` per
  array (`WAVE`, `FLUX`, `IVAR`, `MASK`) and a `BinTableHDU` named `FIBERMAP`.
- `FiberProfile` (`reduction/core/fiberprofile.py`) reads and writes the
  cross-dispersion profile. `to_fits(path)` writes a primary HDU with the
  metadata (`N_FIBERS`, `N_ROWS`, `N_OFFSET` and the `meta` dictionary), then
  an `ImageHDU` named `PROFILE`, an `ImageHDU` named `XOFFSETS`, and a
  `BinTableHDU` named `FIBERMAP` when a fiber map is attached.

None of these classes encodes a file name. The caller chooses the path. There
is no file-naming convention yet.

### Provenance headers

`image_preprocessing()` records what it did in the output header:

| Keyword | Meaning |
| --- | --- |
| `CALIBRAT` | `True` once the frame is calibrated |
| `MBIAS` | Path of the master bias used |
| `MDARK` | Path of the master dark used |
| `MPIXFLT` | Path of the master pixel flat used |

Each applied step also adds a `HISTORY` entry.

### The `data/` tree

| Directory | Holds | In git |
| --- | --- | --- |
| `data/raw/` | Incoming frames | Ignored, `.gitkeep` only |
| `data/interim/` | Intermediate products | Ignored, `.gitkeep` only |
| `data/products/` | Science-ready products | Ignored, `.gitkeep` only |
| `data/refdata/` | Reference data, e.g. `linelist_ThAr.txt` | Committed |

The code never assumes these paths. Every path is an argument of the calling
function.

---

## Validation and Failure

The trust boundary is disk I/O and the arguments a caller passes in. Once an
object is in memory, the pipeline trusts it. Validation therefore sits in
`from_fits`, in the orchestrators, and at the entry of the methods that need
it.

### Where validation happens

| Location | Checks | On failure |
| --- | --- | --- |
| `Image.__init__` | `unit` is `"adu"` or `"electron"` | `ValueError`; warns when the data holds negative values |
| `Image.cutout` | image has data, cutout is inside the frame | `ValueError` |
| `_image_calibration` | inputs are `Image` objects with data, shapes agree, required `EXPTIME` values are positive, steps are valid and their dependencies are requested | `ValueError` |
| `FibersIdentifier(strict=True)` | block and fiber counts match the expectation | `ValueError` |
| `FiberMap.validate` | required columns present, `FIBERID` monotonic and unique | `ValueError` |
| `FiberProfile` | `profile` is 3-D, `x_offsets` is 1-D, their shapes agree | `ValueError` |
| `extract_spectra` | `method` is known, `fiber_profile` is given for `"optimal"` | `ValueError` |

Unimplemented entry points raise `NotImplementedError` instead of returning a
wrong result: currently `optimal.extract_optimal`, and therefore
`run_reduction()` with its default `method="optimal"`.

### Bad pixels travel as a bitmask

`boxcar.extract_boxcar` does not raise on bad data. It returns a `uint32` mask
next to the flux:

| Bit | Constant | Meaning |
| --- | --- | --- |
| 1 | `MASK_BAD_TRACE` | trace centre is not finite |
| 2 | `MASK_NO_PIXELS` | aperture covers no usable pixel |
| 4 | `MASK_BAD_VARIANCE` | variance is zero or not finite; `ivar` is set to 0 |

### Known exception to "never fail silently"

Two bare `except:` clauses swallow every error and return a sentinel instead:

- `wavelength_calibration.fitting` returns `score = -1` with zeroed coefficients.
- `lsf_fitting.gaussian_fitting` returns `NaN` for the FWHM and the parameters.

Both hide real errors. They contradict the rule in `AGENTS.md` and are the
first place to fix when the calibration stage is wired up.

---

## Parallelism

Parallelism is an optimisation inside a method, never a requirement of the
result. It sits behind parameters, so the same call can run serially:

- `utils/parallel_processing.py` wraps joblib. `run(function, inputs,
  parallel=True, n_jobs=-1, backend='loky')` maps `function` over `inputs`,
  where `n_jobs=-1` means `min(len(inputs), os.cpu_count())`.
- Its only current user is `wavelength_calibration.find_poss_wavelength_solution`,
  which scores candidate line combinations in parallel. The caller controls it
  through the `parallel` and `n_jobs` arguments.
- `reduction/core/tracemask.py` uses numba (`@jit(nopython=True)`) on
  `_trace_single_fiber` and `_barycenter_at_row`, for the per-row barycenter
  loop.

Rule: keep joblib-style parallel dispatch inside `methods/` and behind a
parameter. `core/` may compile a loop with numba, but it does not dispatch
parallel work. A serial call must give the same numbers as a parallel one.

A caveat on the wrapper: it guesses the call form by trying
`function(*input)` and falling back to `function(input)` on `TypeError`. A
`TypeError` raised inside `function` therefore hides behind the retry.

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
                   │ extract_spectra │  ← reduction.py
                   │ (boxcar/optimal)│
                   └────────┬────────┘
                            ▼
                      FiberFrame        ← core/fiberframe.py
                      (flux, ivar, mask, wave, fibermap)
```

**Key principle:** `core/` objects are passed between steps; `methods/` are stateless functions that operate on them. No method writes to disk.

### 2.2 `core/` — Data Structures

#### `FiberMap` (`core/fibermap.py`)
- **Base:** `astropy.table.Table`
- **Columns:** `FIBERID`, `BLOCKID`, `BLOCK_LOCAL_ID`, `APPROX_X`, `CENTER_ROW`, `VALID`
- **Role:** Single source of truth for "which fibers exist and where they are."
- **QA:** `mark_invalid()`, `get_block()`, `validate()`

#### `TraceMask` (`core/tracemask.py`)
- **Storage:** Legendre polynomial coefficients per fiber `(n_fibers, deg+1)`
- **Role:** Compact, sub-pixel model of fiber curvature on the CCD.
- **API:** `eval(rows)` → `(n_fibers, n_rows)` trace positions.
- **Builder:** `TraceMask.from_fibermap(fibermap, image, poly_deg=10)` traces barycenters and fits.

#### `FiberFrame` (`core/fiberframe.py`)
Inspired by `desispec.Frame` and `lvmdrp.RSS`.
- **Storage:** `flux`, `ivar`, `mask` as `(n_fibers, n_wave)`; `wave` as `(n_wave,)` or `(n_fibers, n_wave)`
- **Role:** The canonical intermediate data product passed from `reduction/` → `calibration/`.
- **Metadata:** `fibermap` (FiberMap), `meta` (FITS header dict)
- **I/O:** `to_fits()`, `from_fits()` — critical for checkpointing pipeline steps.

**Why not `lvmdrp.RSS`?** `RSS` in lvmdrp inherits from a massive `FiberRows` + `Header` hierarchy. We prefer **composition** to keep the class lightweight and explicit.

#### `FiberProfile` (`core/fiberprofile.py`)
Required for **flat-relative optimal extraction** (see MaNGA `extract_row`).
- **Storage:** `profile[n_fibers, n_rows, n_offsets]`, `x_offsets[n_offsets]`
- **Role:** Normalized cross-dispersion PSF measured from a master flat.
- **Builder:** `FiberProfile.from_flat(flat_image, tracemask, fibermap, half_width=5)`
- **Invariant:** `sum(profile, axis=-1) == 1` for every fiber/row.

### 2.3 `methods/` — Algorithms

#### `fiber_tracing.py`
- `trace_fibers_barycenter(image, approx_positions, center_row)` → dense trace array
- `fit_traces_polynomial(traces, poly_deg)` → coefficients + domain
- *Future:* `trace_fibers_gaussian()` for cross-correlation centroiding.

#### `boxcar.py`
- `extract_boxcar(image, trace_positions, aperture_radius, variance=None, mask=None)` → flux, ivar, mask
  - Simple aperture sum. Use for QA, quick-look, and as fallback.

#### `optimal.py` — stub
- `extract_optimal(image, trace_positions, fiber_profile, ...)` raises `NotImplementedError`. The signature is fixed; the body is not written.

#### Orchestration
- `extract_spectra(image, tracemask, fibermap, method="boxcar", ...)` → `FiberFrame`, implemented in `reduction.py` (see 2.4). It evaluates trace positions, dispatches to boxcar or optimal, and packages the result into a `FiberFrame`.
- `run_reduction()` defaults to `method="optimal"`, so it raises until optimal extraction lands. `run_quick_reduction()` uses boxcar and works.

**Design note:** The optimal extractor should follow MaNGA's iterative rejection:
1. Fit model to row.
2. Compute residuals.
3. Mask the single worst pixel in each contiguous bad group.
4. Re-fit until convergence or `maxiter`.

#### `profile_modeling.py`
- `build_fiber_profile(flat_image, tracemask, fibermap, half_width)` → `FiberProfile`
  - For each fiber/row, cut out a cross-dispersion slice centered on the trace, normalize.
- `normalize_profile(profile)` → ensure sum-to-one.

### 2.4 Orchestrator (`reduction.py`)

The public API is intentionally minimal:

```python
# Identification and tracing
fibermap, tracemask = identify_and_trace_fibers(
    image=flat_image,
    n_blocks_expected=19,
    n_fibers_per_block_expected=29,
)

# Profile and extraction
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

## Stage 3: `calibration/` — Detailed Design

This section covers only what exists in the repository. Fiber flat-fielding,
sky subtraction, flux calibration and coaddition are planned and have no
implementation yet; see the Status table above.

### 3.1 Data Flow

The implemented part is a wavelength solution for one extracted 1-D spectrum:

```
Extracted 1-D arc spectrum
       │
       ▼
  detect_lines()                     ← methods/wavelength_calibration.py
  (peak positions in the uncalibrated spectrum)
       │
       ▼
  find_poss_wavelength_solution()
  (enumerate reference-line / peak combinations, fit a Legendre
   polynomial per combination, score by RMSE, keep the best)
       │
       ▼
  refine_poss_solution()
  (least-squares refinement, poss_poly = a * guess + b)
       │
       ▼
  wavelength solution, as a polynomial
```

There is no Stage 3 orchestrator yet, so a caller wires these steps itself and
applies the polynomial to science spectra.

### 3.2 `methods/` — Algorithms

#### `wavelength_calibration.py`
- `detect_lines(spectrum, n_strongest_lines=20, n_all_lines=100)` → positions of the strongest peaks and of all peaks.
- `calculate_fitting_score(poss_poly, known_wls, all_peak_ys)` → RMSE-like score between the reference lines and their nearest peaks.
- `find_poss_wavelength_solution(poss_wls, poss_ys, known_wls, all_peak_ys, ...)` → best polynomial and its score, by enumerating combinations. `full_search=True` lets the degree grow beyond `min_deg`.
- `refine_poss_solution(guess_poss_poly, known_wls, all_peak_ys)` → polynomial refined by least squares.
- `wavelength_calibration(poss_wls, poss_ys, known_wls, all_peak_ys, ...)` → final `(polynomial, score)`, with optional iterative refinement.
- `inv_poss_poly(poss_poly, wl, y_min=0., y_max=9600., atol=1e-5)` → y coordinate of a given wavelength, by bisection.
- Candidate combinations run in parallel through `utils/parallel_processing.run` (joblib).

#### `lsf_fitting.py`
- `lsf_fitting(spectrum, spectrum_wls, target_wl)` → FWHM of the line, through `lsf_gaussian_fitting` with `cutout_wl_half_width=3` and `adjust_target_wl=True`.
- `lsf_gaussian_fitting(spectrum, spectrum_wls, target_wl, ...)` → FWHM, the four Gaussian parameters, the fitting function and the cutout.
- Both call `detect_lines` to centre the cutout and to reject a double peak.

### 3.3 Planned Modules

`fiberflat.py`, `sky.py`, `fluxcal.py` and the `calibration.py` orchestrator are
empty files. `calibration/core/` holds no data models yet. These are next work,
not a design to follow.

---

*See `docs/tutorials/` for Jupyter Notebook guides on using the pipeline.*
