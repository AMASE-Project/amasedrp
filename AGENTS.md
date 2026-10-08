# amasedrp — notes for agents

`amasedrp` is the data reduction pipeline (DRP) for AMASE-P, the prototype of
the Affordable Multiple Aperture Spectroscopy Explorer. It turns raw CMOS
frames into science-ready spectra for a survey of ionized gas at R ≈ 15 000.
The user installs it with `pip install -e .` and keeps that path working.

The default fiber configuration is 19 blocks of 29 fibers, which matches the
superseded Nikon-based design. The 2025 CDR baseline is 22 blocks of 25
fibers. Confirm which generation is targeted before changing a default.

## FITS metadata

The AMASE metadata scheme is not finished, so FITS header keyword names and
values are not a settled contract. Expect missing cards, unexpected values,
and the same physical quantity written differently in different channels.

Example from the 2025-07 collimator sweep: `IMAGETYP` reads `'Light Frame'`
for blue arcs and `'Flat Field'` for blue flats, but `'LIGHT'` for both red
arcs and red flats. Selecting red flats by `IMAGETYP` returns the arcs as
well, and reports no error.

So never treat a header keyword as reliable. Validate at the FITS boundary,
and prefer a keyword the instrument actually writes over the conventional one.
`LAMP` separates arcs from flats in both channels of the sweep data, so it is
the better key there.

Keep the mapping from a physical quantity to its keyword in one documented
place. When a header is wrong, repair it in one explicit step and record what
changed, instead of compensating at every call site.

## Priority

`amasedrp` is the pipeline of a science project. Its results must be
reliable, and its code must stay maintainable. A human engineer or
scientist will read specific functions and algorithms, not only run the
pipeline, so readability matters: clear logic and explicit steps beat
clever compression.

Robustness matters too. The pipeline does not owe an industrial-grade
guarantee, because it does not have to handle every possible input. It
must still robustly implement the researcher's explicit need, and it must
not fail silently on the inputs that would corrupt a reduction.
Readability wins over robustness, not over correctness.

A silent wrong reduction is the worst outcome, so validate at the FITS
boundary and raise instead of guessing.

A failed wavelength solution needs care, because the same failure means two
different things. During a focus sweep it is expected: at strong defocus the
arc lines broaden and blend, and the line matching cannot succeed. During a
real observation it must not happen, and it must not pass unnoticed, because
it points at a software or hardware fault. The pipeline therefore reports the
fraction of fibers it calibrated, and an observer-facing entry point must
surface that fraction instead of reducing quietly with a partial solution.

## Rules

- Record constraints and intent here, not the current code layout. File and
  module names drift, and this file will not keep up.
- Keep the Three-Part Module Pattern. `core/` holds data structures and I/O;
  `methods/` holds one pure scientific step per file; the top-level `.py`
  orchestrates. Ask before breaking it.
- `methods/` do no file I/O and no logging. Only the orchestrator reads,
  writes and logs.
- A stage must not import from another stage. `reduction/` and `calibration/`
  communicate through `core/` objects, not through imports.
- Keep the public API small. Each stage exposes a few functions; the
  orchestrator is the user-facing entry point.
- `README.md` stays a short overview. Design decisions and evidence go in
  `docs/`, which is where the user looks for depth.
- Implementation status lives in `docs/ARCHITECTURE.md`, not here.
- The `data/` tree is gitignored. Tutorials must run on synthetic frames or on
  frames the user supplies. Never commit FITS data or absolute paths.
- Ask before adding a dependency. `pyproject.toml` must list every import.

## How to check your work

    pip install -e . --no-build-isolation --no-deps --dry-run
    PYTHONPATH=src python -m pytest tests -q
