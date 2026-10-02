# Inputs and execution

## Required local inputs

All paths are relative to the repository root. Obtain GFS data through https://www.cos.io/gfs with access to the variables required by this study. The supplied data version and category mappings matter; a newer GFS release need not recreate the same sample automatically.

| File | Role |
|---|---|
| `data/raw/gfs_all-countries_wave2-with-midyear_sensitive.csv` | Locally supplied GFS survey file used by preprocessing; not distributed |
| `data/raw/cmip6/tas_Amon_EC-Earth3-Veg_{scenario}_r1i1p1f1_gr_{year}01-{year}12.nc` | Monthly climate fields |
| `data/processed/GlobalFlourishingDataWithLonLit.parquet` | Harmonized survey records from `nbs/c01_gfs_data_preprocess.py` |
| `data/processed/df_out_2023.parquet`, `df_out_2024.parquet` | Legacy exposure tables used to preserve the analytical sample |
| `data/processed/revision_exposure_validation/native_grid_location_year.parquet` | Correct native-grid assignments and exposure values |
| `data/processed/revision_exposure_validation/fixed_sample_exposure_variants.parquet` | Fixed-sample corrected-grid and interview-year exposure variants |

The preprocessing source documents variable harmonization, exclusions and income-category mappings. The original Python-export extraction script `src/analyses/nbs/c02_extract_cmip6_temperature.py` also references 2050, 2070 and 2100 climate inputs; those are historical projections, not revised-model exposures. Its legacy exposure tables establish the retained sample; `validate_exposure_corrections.py` calculates the corrected exposure values used by the revised analysis. The two original preprocessing exports import `python-dotenv`; when running them, use `uv run --with python-dotenv python <script>`.

After generating the three processed survey/legacy-exposure files, generate the corrected inputs:

```bash
uv run python src/analyses/validate_exposure_corrections.py
```

This step needs the original CSV and 2023/2024 climate files for all five scenarios. It preserves the fixed sample. The main runner asserts 273,031 analytical records and matching row order. Do not bypass these checks when using another data release. Participant-level processed outputs remain local.

## Main run

From the repository root, after the five processed inputs are present:

```bash
uv run python src/analyses/run_revision_experiments.py \
  --variants grid --repeat-seeds 42 --bootstrap-reps 100 \
  --device cuda:0 --threads 4 \
  --output data/exp/revision_uncertainty_audit
```

For CPU execution use `--device cpu` and a separate output directory. Do not reuse a GPU manifest directory for CPU fitting. Runtime manifests record input/code hashes, software and parameters and reject incompatible resume attempts. Fold checkpoints support resumption. The full bootstrap is computationally intensive; `--help` only checks CLI availability, not numerical reproduction.

Summarize the complete primary run:

```bash
uv run python src/analyses/summarize_bootstrap_uncertainty.py
```

This writes to `data/results/revision_bootstrap_uncertainty/` and validates the 100 draws, fold completion and source hashes. Resuming a run that already recorded completion events may require inspection of duplicate event records; do not edit manifests to bypass validation.

## Supporting scripts

- `run_revision_benchmarks.py`: predictive benchmarks.
- `run_categorical_comparison.py`: categorical-encoding sensitivity.
- `run_spatial_coarsening_sensitivity.py`: exposure aggregation sensitivity.
- `regenerate_probability_pdp.py` and `plot_probability_pdp_original_style.py`: PDP computation and display.
- `plot_revision_feature_importance.py`, `plot_revision_spatial_contrasts.py`, `plot_revision_spatial_supplement.py`: figures.
- `audit_sample_flow.py`, `summarize_revision_sample.py`, `build_revision_table1.py`: sample descriptions and tables.

Supporting scripts retain the project-relative experiment/output paths of the revision. Inspect each script's arguments and input constants before running it; some consume completed model outputs, boundary files, or local audit metadata. The source release does not include these datasets or outputs. No automatic full pipeline is implied by the list.

## Interpretation

The direct/total/indirect outputs are model-based prediction contrasts. The resampling intervals are conditional on the retained model settings and split seed. Record-level folds can contain different waves from the same respondent across train and test sets. The code does not implement a causal mediation estimator.
