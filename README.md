# ZDP02g: temperature and physical-health analysis

Python sources for the revised pooled analysis of Global Flourishing Study (GFS) Waves 1 and 2 and CMIP6 temperature metrics.

## Environment

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then run from the repository root:

```bash
uv sync --python 3.12 --locked
uv run python src/analyses/run_revision_experiments.py --help
```

The lockfile records the dependency versions. The main analysis can run on CPU or an NVIDIA CUDA device. The optional GPU extra supplies CuPy for auxiliary GPU diagnostics; the XGBoost runner itself uses its `--device` flag. Reproducing the GPU estimates requires a compatible NVIDIA driver/runtime. CPU runs may differ slightly numerically.

## Data access

Request GFS data directly through the [GFS/COS access portal](https://www.cos.io/gfs), following the access conditions for the required survey and geographic variables. Individual-level source and linked survey data are not included in this code release.

Climate inputs are CMIP6 ScenarioMIP monthly near-surface temperature (`tas`, `Amon`) from EC-Earth3-Veg, member `r1i1p1f1`, grid `gr`, scenarios `ssp119`, `ssp126`, `ssp245`, `ssp370`, and `ssp585`. Obtain the corresponding NetCDF inputs through ESGF. See [inputs and execution](docs/reproduction.md) for the expected local files and analysis sequence.

## Revised analysis

The primary run uses fixed model parameters, random record-level 10 outer/5 inner folds, split seed 42, and 100 country-stratified native-grid cluster bootstrap draws. Both model stages are refitted for each draw. Annual mean temperature is perturbed by +1°C and annual temperature standard deviation by +0.1°C.

Active revised code is in `src/analyses/`; fixed settings are in `src/analyses/config/`; data preparation and historical Python exports are in `src/analyses/nbs/`; shared helper modules are in `src/analyses/notebooks/` (the directory name does not require notebooks).

Earlier top-level `nbs/`, `notebooks/`, scripts and workflow files are retained as historical material. Use the commands in this README and reproduction guide for the revision. In particular, `continue_grid_bootstrap_queue.py` is a superseded 1000-draw queue, not the manuscript's final specification. Draft fold-layout checks and pilot scripts are also historical diagnostics.

The code source inventory is recorded in [revision-source-manifest.json](docs/revision-source-manifest.json). Local manuscript files, reviewer correspondence, machine credentials, and individual-level datasets are outside this release.
