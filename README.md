# Polarization of Post-Disaster Resilience Analysis Using NTL Time-Series Data

This project quantifies **post-disaster recovery polarization** in India using grid-level nighttime lights time-series data.

Core interpretation:

> This is not a framework for claiming that AI precisely predicts exact disaster recovery duration.  
> It is a framework for quantifying recovery polarization and identifying long-term delayed recovery risk areas using early NTL recovery patterns and spatial/socioeconomic/hazard exposure features.

## Final research scope

- Country: India
- Disaster types: tropical cyclones + urban flooding
- Primary data: NASA Black Marble VNP46A2 nighttime lights time series
- Sample unit: `event_id × grid_id`
- Main target: event-relative delayed recovery classification
- Auxiliary target: event-relative recovery delay percentile
- Additional target: no recovery within 12/24 months
- Primary validation: Leave-One-Event-Out validation
- Final model: LightGBM tabular baseline + TCN multi-task sequence fusion model + ensemble

## Phase 1 usage

Create the project skeleton and metadata templates:

```bash
python scripts/00_init_project.py
```

Load the base configuration from Python:

```python
from ntlpol.config import load_config

cfg = load_config("configs/base.yaml")
print(cfg.require("project.name"))
```

## Full execution order

```bash
python scripts/00_init_project.py
python scripts/01_build_event_catalog.py
python scripts/02_make_grid.py
python scripts/03_ingest_ntl.py
python scripts/04_build_grid_event_ntl_sequences.py
python scripts/05_make_recovery_metrics.py
python scripts/06_make_targets.py
python scripts/07_make_hazard_features.py
python scripts/08_make_socioeconomic_features.py
python scripts/09_build_modeling_dataset.py
python scripts/10_make_splits.py
python scripts/11_check_leakage.py
python scripts/20_train_lgbm_baseline.py
python scripts/21_train_rf_logistic_baselines.py
python scripts/30_train_tcn_multitask.py
python scripts/31_train_gru_multitask.py
python scripts/32_train_transformer_multitask.py
python scripts/40_ensemble.py
python scripts/50_evaluate_all.py
python scripts/60_make_outputs.py
```

## Leakage rule

The full pre-12 + event + post-24 NTL sequence is allowed only for recovery measurement and target construction.  
Early prediction models must use only pre-12 + event + post-3 or post-6 input windows.

## Real data extraction additions

The project now includes Earth Engine export scripts for the real-data bridge:

```bash
python scripts/70_gee_export_vnp46a2_daily.py --help
python scripts/71_gee_export_gpm_event_rainfall.py --help
python scripts/72_gee_export_ghsl_static_features.py --help
python scripts/73_build_cyclone_hazard_from_ibtracs.py --help
python scripts/74_gee_export_vnp46a2_monthly_optional.py --help  # optional fallback only
python scripts/80_run_after_raw_downloads.py --help
```

See `REAL_DATA_WORKFLOW.md` for the exact order after uploading the grid to Earth Engine.
"# ntl_resilience_polarization" 
