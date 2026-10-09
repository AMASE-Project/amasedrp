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
| Reduction | Optimal extraction (FOX) | Implemented |
| Reduction | Spectro-perfectionism | Not started |
| Calibration | Wavelength calibration | Implemented |
| Calibration | LSF fitting | Implemented, not wired |
| Calibration | Fiber flat-fielding | Planned |
| Calibration | Sky subtraction | Planned |
| Calibration | Flux calibration | Planned |
| Post-processing | Coaddition | Planned |
| Data products | Product tree and naming (`products.py`) | Implemented |

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
│   │   ├── master.py           # Master frame stacking (median combine)
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
│   ├── methods/                # Atomic, stateless processing steps
│   │   ├── fiber_tracing.py    # Barycenter tracing & Legendre fitting helpers
│   │   ├── boxcar.py           # Boxcar extraction
│   │   └── optimal.py          # Flat-relative optimal extraction (FOX)
│   └── reduction.py            # Stage 2 orchestrator: identify → trace → extract
│
├── calibration/                # Stage 3: Master Calibration & Data Calibration
│   ├── __init__.py
│   ├── calibration.py          # Wavelength solution: solve and apply
│   ├── lines.py                # ThAr line selections, per channel
│   ├── core/                   # Calibration-specific data models
│   │   ├── wavelengthsolution.py  # WavelengthSolution: per-fiber polynomial
│   │   └── linespreadfunction.py  # LineSpreadFunction: per-fiber line width
│   ├── methods/                # Calibration algorithms
│   │   ├── fiberflat.py        # Fiber-to-fiber flat-field correction
│   │   ├── wavelength_calibration.py # Wavelength calibration
│   │   ├── lsf_fitting.py      # Line-spread function fitting
│   │   ├── sky.py              # Sky background subtraction
│   │   └── fluxcal.py          # Flux calibration
│
├── observation.py              # Frame selection and loading for an observation
├── simulate.py                 # Synthetic frames, for verification and tutorials
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

FITS is the only on-disk format. Two classes own the I/O:

- `Image` (`preprocessing/core/image.py`) reads and writes single 2-D frames.
  `Image.from_fits(filename)` takes the physical unit from the `BUNIT`
  keyword, and assumes ADU when the keyword is absent.
  `Image.write_to_fits(filename)` writes the raw ADU array and sets
  `BUNIT = 'adu'`, so a file on disk never holds electrons.
- `FiberFrame` (`reduction/core/fiberframe.py`) reads and writes extracted
  spectra. `to_fits(path)` writes a primary HDU that holds metadata only
  (`N_FIBERS`, `N_WAVE` and the `meta` dictionary), then one `ImageHDU` per
  array (`WAVE`, `FLUX`, `IVAR`, `MASK`) and a `BinTableHDU` named `FIBERMAP`.

None of these classes encodes a file name; only `products.py` and the
experiment drivers choose where a product goes.

### The product tree

`products.py` owns the layout, so no call site invents a path:

```text
<products_dir>/<drpver>/<night>/                 one night
<products_dir>/<drpver>/<night>/calibration/     nightly masters and state
<products_dir>/<drpver>/<night>/<obsid>/         one pointing sequence
<products_dir>/<drpver>/<night>/<obsid>/qa/      its quality assessment
```

`drpver` is the version that produced the products, so reprocessing a night
writes beside the previous reduction instead of over it. When it is not given,
`product_dir()` reads the installed package version and raises if it cannot, so
a product never lands in a directory named after nothing.

Science products are named `<level>-<channel>-<exposure>`, for example
`L2-blue-0001.fits`. The levels are AMASE's, not this package's: L1 is
pre-processed, L2 extracted and wavelength-calibrated, L3 flux-calibrated and
sky-subtracted.

Calibration products such as a master bias are not data levels, so
`product_name()` does not name them, and their file names are still chosen per
call site.

*Not settled:* whether they belong in the night's `calibration/` or inside each
`<obsid>/`. MaNGA keeps its masters and calibration state inside the
per-observation directory, because one MaNGA MJD observes one plate once. AMASE-P
may reuse a previous night's calibration, which argues for the night level.
`product_dir(..., subdir="calibration")` reaches either, so the choice does not
change the API.

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
| `FibersIdentifier(strict=True)` | block and fiber counts match the expectation | `ValueError`; with `strict=False` the run continues and the affected fibers get `VALID = False` |
| `FiberMap.validate` | required columns present, `FIBERID` monotonic and unique | `ValueError` |
| `extract_spectra` | `method` is known, `flat_image` is given for `"optimal"`, and `variance` is not given for `"optimal"` | `ValueError` |
| `extract_optimal` | `flat_image` matches `image`, `trace_positions` matches `image`, `gain` is positive, `read_noise` is not negative | `ValueError` |
| `WavelengthSolution` | `poly_kind` is known, `coeffs` is 2-D, `fiber_ids` and `scores` match the coefficient rows | `ValueError` |
| `solve_wavelength_solution` | frame carries a fiber map, fiber counts agree, `poly_kind` is known, line lists are not empty, `min_calibrated_fraction` lies in `[0, 1]` | `ValueError`; `RuntimeError` when the reference fiber cannot be calibrated or the calibrated fraction falls below `min_calibrated_fraction` |
| `apply_wavelength_solution` | solution and frame hold the same number of fibers | `ValueError` |

No entry point raises `NotImplementedError` any more. `calibration/` holds no
orchestrator yet, so its two methods cannot be reached from a public function.

### Bad pixels travel as a bitmask

`boxcar.extract_boxcar` and `optimal.extract_optimal` do not raise on bad
data. They return a `uint32` mask next to the flux. Both methods set the same
bits:

| Bit | Constant | Meaning |
| --- | --- | --- |
| 1 | `MASK_BAD_TRACE` | trace centre is not finite |
| 2 | `MASK_NO_PIXELS` | aperture covers no usable pixel |
| 4 | `MASK_BAD_VARIANCE` | variance is zero or not finite; `ivar` is set to 0. For FOX, also set where the flat holds no positive sample to convert the relative spectrum. |

### Known exception to "never fail silently"

One bare `except:` clause swallows every error and returns a sentinel instead:

- `lsf_fitting.gaussian_fitting` returns `NaN` for the FWHM and the parameters.

It hides real errors and contradicts the rule in `AGENTS.md`. Fix it when the
LSF driver is written.

The matching clause in `wavelength_calibration.fitting` was removed: a
combination that cannot be fitted now scores `nan`, which `np.nanargmin`
skips. Before that fix, a single failed combination won the comparison and the
caller got a dummy solution.

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

Second rule: batch the work. A task must be big enough that pickling its
arguments costs less than running it. One task per candidate line assignment
made joblib pass the reference wavelengths, the peak list and a polynomial
class 843 999 times, and reached only 3.0x on 12 cores. Batches of a few
thousand candidates reach 4.7x. Batch in *contiguous* slices: strided slices
make the concatenated results land out of order.

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
│ Flat image              │          │ Science Image           │
│ (from preprocessing/)   │          │ (from preprocessing/)   │
└───────────┬─────────────┘          └──────────┬──────────────┘
            ▼                                   ▼
   flat weights (FOX only)               TraceMask.eval()
                                         (trace_positions)
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

### 2.3 `methods/` — Algorithms

#### `fiber_tracing.py`
- `trace_fibers_barycenter(image, approx_positions, center_row)` → dense trace array
- `fit_traces_polynomial(traces, poly_deg)` → coefficients + domain
- *Future:* `trace_fibers_gaussian()` for cross-correlation centroiding.

#### `boxcar.py`
- `extract_boxcar(image, trace_positions, aperture_radius, variance=None, mask=None)` → flux, ivar, mask
  - Simple aperture sum. Use for QA, quick-look, and as fallback.

#### `optimal.py`
- `extract_optimal(image, flat_image, trace_positions, aperture_radius=3, gain=1.0, read_noise=1.0, mask=None)` → flux, ivar, mask
  - Flat-relative optimal extraction (Naylor 1998). The flat carries the
    cross-dispersion profile, so weighting the science by the flat gives the
    best signal-to-noise ratio. A smoothed boxcar extraction of the flat then
    removes the lamp spectrum and returns the spectrum to the boxcar scale.
  - The noise model is built from the science counts,
    `var = gain * image + read_noise**2`. Negative pixels are clipped to zero
    because they come from an over-subtracted bias.

#### Orchestration
- `extract_spectra(image, tracemask, fibermap, method="boxcar", flat_image=None, ...)` → `FiberFrame`, implemented in `reduction.py` (see 2.4). It evaluates trace positions, masks the rows that fall outside each fiber's fitted range, dispatches to boxcar or optimal, and packages the result into a `FiberFrame`.
- `run_reduction()` defaults to `method="optimal"`. `run_quick_reduction()` uses boxcar and works.

**Design note:** FOX was chosen over MaNGA-style iterative profile fitting
because it is the algorithm the AMASE prototype used, so its output can be
compared against the 2025-07 collimator-sweep reductions. The prototype design
carried a normalized `FiberProfile` PSF model; FOX does not need one, because
the flat itself is the profile. That model was deleted with the
profile-fitting design.

### 2.4 Orchestrator (`reduction.py`)

The public API is intentionally minimal:

```python
# Identification and tracing
fibermap, tracemask = identify_and_trace_fibers(
    image=flat_image,
    n_blocks_expected=19,
    n_fibers_per_block_expected=29,
)

# Extraction
fiber_frame = extract_spectra(
    image=science_image,
    tracemask=tracemask,
    fibermap=fibermap,
    method="optimal",
    flat_image=flat_image,
)
```

### 2.5 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **`FiberFrame` as 2-D array `(fiber, wave)`** | Matches `desispec.Frame` and `lvmdrp.RSS`. Each row is one fiber's 1-D spectrum. Easy to feed into `calibration/` (wavelength, sky, flux cal). |
| **Optimal extraction is flat-relative (FOX)** | The flat already carries the cross-dispersion profile of every fiber, so no separate PSF model is needed. It is also the algorithm the prototype used, so the output can be compared against the 2025-07 collimator-sweep reductions. |
| **One aperture convention for both methods** | Both boxcar and FOX use a fixed-width aperture, `round(position) ± aperture_radius`. The prototype rounded the edges with `floor`/`ceil`, which makes the aperture width follow the fractional trace position. That is a rounding artefact, and two conventions in one package would make the two extractors incomparable. This costs a systematic ~6% against the prototype. |
| **Each fiber keeps its own fit domain** | A trace is fitted over the rows where it was actually traced. Handing every fiber the full image row range would evaluate the polynomial far outside its fit, which moves the trace by up to 28 pixels on the sweep frames. `extract_spectra` masks the rows outside the fitted range instead of extracting them. |
| **Boxcar as first-class citizen** | Not just a placeholder. Needed for: (1) quick-look QA, (2) identifying bright fibers before optimal extraction (MaNGA `find_whopping`), (3) fallback when optimal fails. |
| **No `Aperture` class for boxcar** | `lvmdrp` has a complex `Aperture` class with sub-pixel integration. For a first implementation, an integer `aperture_radius` is sufficient and far simpler. Upgrade path: replace the integer with a `PixelAperture` object later. |

---

## Stage 3: `calibration/` — Detailed Design

This section covers only what exists in the repository. Fiber flat-fielding,
sky subtraction, flux calibration and coaddition are planned and have no
implementation yet; see the Status table above.

### 3.1 Data Flow

```
Extracted arc spectra                (reduction/, one FiberFrame + TraceMask)
       │
       ▼
  solve_wavelength_solution()        ← calibration.py
  pick the longest-trace fiber, run a full search over line combinations,
  then propagate outward: each fiber starts from its solved neighbour
       │
       ▼
  WavelengthSolution                 ← core/wavelengthsolution.py
  (coeffs, fiber_ids, poly_kind, scores)
       │
       ▼
  apply_wavelength_solution(frame, solution)
       │
       ▼
  FiberFrame with wave (n_fibers, n_wave) and fibermap["WAVCAL_SCORE"]
```

The full search only runs for the reference fiber.  Every other fiber is
refined from its neighbour solution, which is far cheaper and keeps
neighbouring fibers consistent.  A fiber whose trace is shorter than 70% of
the reference trace, or whose refinement fails, inherits the neighbour
solution and is marked ``score = -1``.

`calibration/lines.py` holds the ThAr line selections of both channels, as
`known_wls` (used for scoring), `poss_wls` (used to enumerate candidates) and
`lsf_wls` (isolated enough for a line-spread fit).  The lists are selections
for this spectrograph, not a lamp atlas.

### 3.2 `methods/` — Algorithms

#### `wavelength_calibration.py`
- `detect_lines(spectrum, n_strongest_lines=20, n_all_lines=100)` → positions of the strongest peaks and of all peaks.
- `calculate_fitting_score(poss_poly, known_wls, all_peak_ys)` → RMSE-like score between the reference lines and their nearest peaks, or `nan` when no residual survives the outlier cut.
- `score_combination(ys, wls, known_wls, all_peak_ys, ...)` → score of one candidate pairing, without converting the coefficients. `convert()` costs about half of one candidate, so only the winner pays for it.
- `find_poss_wavelength_solution(poss_wls, poss_ys, known_wls, all_peak_ys, ...)` → best polynomial and its score, by enumerating combinations. `full_search=True` lets the degree grow beyond `min_deg`. A negative score means no combination produced a usable fit.
- `refine_poss_solution(guess_poss_poly, known_wls, all_peak_ys)` → polynomial refined by least squares.
- `wavelength_calibration(poss_wls, poss_ys, known_wls, all_peak_ys, ...)` → final `(polynomial, score)`, with optional iterative refinement.
- `inv_poss_poly(poss_poly, wl, y_min=0., y_max=9600., atol=1e-5)` → y coordinate of a given wavelength, by bisection. The row limits default to the sweep-frame height, so pass `y_max` for another detector.
- Candidates are scored in parallel through `utils/parallel_processing.run` (joblib), in contiguous batches of a few thousand. On the 2025-07 data the reference fiber of one frame enumerates 843 999 candidates, which took 63 s one-task-per-candidate and 14 s batched.

**Failure convention.** A combination that cannot be fitted scores `nan`, not
`-1`: `nan` is skipped by `np.nanargmin`, whereas a numeric sentinel would win
the comparison and hand back a dummy solution. A score of `-1` is reserved for
"no combination worked at all".

**Status of the acceptance rate.** `solve_wavelength_solution` warns once per
fiber it could not calibrate, and marks that fiber with `score = -1`.
`WavelengthSolution.calibrated_fraction` reports the share that succeeded, and
the `min_calibrated_fraction` argument turns a shortfall into a
`RuntimeError`.  A focus sweep leaves that argument unset, because at strong
defocus most fibers legitimately fail: on the 2025-07 sweep the fraction fell
from 523 of 539 fibers at good focus to 119 of 539.  An entry point that serves
real observations must pass a value close to one; see the wavelength paragraph
in `AGENTS.md`.

#### `lsf_fitting.py`
- `lsf_fitting(spectrum, spectrum_wls, target_wl)` → FWHM of the line, through `lsf_gaussian_fitting` with `cutout_wl_half_width=3` and `adjust_target_wl=True`.
- `lsf_gaussian_fitting(spectrum, spectrum_wls, target_wl, ...)` → FWHM, the four Gaussian parameters, the fitting function and the cutout.
- Both call `detect_lines` to centre the cutout and to reject a double peak.

### 3.3 Planned Modules

`fiberflat.py`, `sky.py` and `fluxcal.py` are empty files, and no LSF driver
is written yet.  These are next work, not a design to follow.

---

*See [`docs/tutorials/`](tutorials/README.md) for Jupyter Notebook guides on
using the pipeline. They run on the synthetic frames from `simulate.py`, so no
observation data is needed.*
