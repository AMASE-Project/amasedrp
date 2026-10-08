# Tutorials

Seven Jupyter notebooks, numbered in pipeline order. They use synthetic data
that `amasedrp.simulate` generates, so they run on a machine that has never
seen an observation.

Install the package and Jupyter first, in an environment that has `numpy`,
`astropy`, `matplotlib` and `scipy`; see `docs/CONTRIBUTING.md`. Jupyter is
deliberately not a package dependency, so install it yourself.

| # | Notebook | Subject | Assumes |
| --- | --- | --- | --- |
| 01 | [Quickstart](01_quickstart.ipynb) | One flat frame to one extracted spectrum, in a dozen cells | — |
| 02 | [Image preprocessing](02_image_preprocessing.ipynb) | The `Image` container, bias, dark and pixel flat, cosmic rays, frame selection | — |
| 03 | [Fiber identification and tracing](03_fiber_id_trace_tutorial.ipynb) | Block detection, fiber peaks, the per-fiber fit domain | — |
| 04 | [Spectral extraction](04_spectral_data_reduction.ipynb) | Boxcar against optimal (FOX), and what separates them | 03 |
| 05 | [Wavelength calibration](05_spectral_data_calibration.ipynb) | Arc lines to wavelengths, and the line-spread function | 03, 04 |
| 06 | [Logging](06_logging_system.ipynb) | Where the pipeline writes its warnings | 02 |
| 07 | [End to end](07_end_to_end_pipeline.ipynb) | The whole chain on one simulated observation | all |

**Read 01 first.** It shows the shape of the whole thing in a minute. After
that, 03 and 04 cover the reduction stage in depth, and 05 covers calibration.

## Running a notebook

```
jupyter lab docs/tutorials/03_fiber_id_trace_tutorial.ipynb
```

The notebooks write only into temporary directories, so a run leaves nothing
behind in the repository and never touches observation data.

## Outputs are not committed

The notebooks are stored with their outputs cleared. A cell's expected result is
written into the surrounding text instead of being saved into the file, so the
diff of a change to the pipeline stays readable and the repository stays small.
Run a notebook to see its plots and figures.
