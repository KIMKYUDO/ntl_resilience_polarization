# Polarization of Post-Disaster Resilience Analysis Using NTL Time-Series Data

This project quantifies **post-disaster recovery polarization** in India using grid-level nighttime lights time-series data.

Core interpretation:

> This is not a framework for claiming that AI precisely predicts exact disaster recovery duration.  
> It is a framework for quantifying recovery polarization and identifying delayed-recovery risk areas using early NTL recovery patterns and spatial/socioeconomic/hazard exposure features.

## Final research scope

- Country: India
- Disaster types: tropical cyclones + urban flooding
- Primary data: NASA Black Marble VNP46A2 nighttime lights time series
- Sample unit: `event_id × grid_id`
- Main target: event-relative delayed recovery classification
- Auxiliary target: event-relative recovery delay percentile
- Additional targets: no recovery within 12/24 months
- Primary validation: Leave-One-Event-Out validation
- Early prediction input: pre-event window + event month + early post-disaster NTL trajectory, especially early 3-month input
- Final single model interpretation: **Transformer multi-task sequence model**
- Additional sequence models: **TCN** and **GRU**
- Baseline model: **LightGBM tabular baseline**
- Combined output: **ensemble prediction**, used as a robustness/combined-risk result rather than a separate trained model checkpoint

## Final model interpretation

The final model family predicts grid-level slow-recovery risk from early post-disaster NTL trajectories.

Input:

- Early NTL time-series features
- Tabular/static/event-level features
- `event_id × grid_id` sample structure

Outputs:

- `pred_delayed_prob`: probability of delayed recovery
- `pred_recovery_percentile`: predicted recovery-delay percentile
- `pred_no_recovery_12m_prob`: predicted no-recovery risk within 12 months
- `pred_no_recovery_24m_prob`: predicted no-recovery risk within 24 months

The strongest single model for delayed-recovery discrimination was the **Transformer**, which achieved the highest delayed-recovery AUROC/AUPRC.  
GRU achieved the highest delayed-recovery F1.  
TCN showed strong performance in recovery-percentile regression and long-term risk ranking.  
The ensemble result is useful as a combined prediction layer, but it is not a separate `.pt` model checkpoint.

## Final performance summary

| model       |   delayed_auroc |   delayed_auprc |   delayed_f1 |   delayed_recall |   delayed_top30_recall |   percentile_mae |   percentile_rmse |   percentile_spearman |   no12_auroc |   no12_auprc |   no24_auroc |   no24_auprc |
|:------------|----------------:|----------------:|-------------:|-----------------:|-----------------------:|-----------------:|------------------:|----------------------:|-------------:|-------------:|-------------:|-------------:|
| Ensemble    |          0.9174 |          0.7262 |       0.7739 |           0.862  |                 0.9241 |           0.1008 |            0.1358 |                0.4805 |       0.7574 |       0.0086 |       0.764  |       0.007  |
| GRU         |          0.8994 |          0.6058 |       0.7816 |           0.8318 |                 0.9042 |           0.0957 |            0.1507 |                0.4028 |       0.9112 |       0.0258 |       0.9331 |       0.0351 |
| LightGBM    |          0.5018 |          0.1829 |       0.2593 |           0.3929 |                 0.3137 |           0.1838 |            0.2251 |                0.0587 |       0.6989 |       0.0042 |       0.6472 |       0.0017 |
| TCN         |          0.9198 |          0.667  |       0.7695 |           0.8374 |                 0.9168 |           0.0899 |            0.138  |                0.4739 |       0.8724 |       0.0511 |       0.8801 |       0.0441 |
| Transformer |          0.9264 |          0.7369 |       0.7752 |           0.9019 |                 0.9324 |           0.1036 |            0.144  |                0.4369 |       0.9215 |       0.0244 |       0.9374 |       0.0168 |


- Highest delayed-recovery AUROC: `Transformer` = `0.9264`
- Highest delayed-recovery AUPRC: `Transformer` = `0.7369`
- Highest delayed-recovery F1: `GRU` = `0.7816`
- Highest delayed-recovery Top-30% recall: `Transformer` = `0.9324`
- Lowest recovery-percentile MAE: `TCN` = `0.0899`
- Highest recovery-percentile Spearman correlation: `Ensemble` = `0.4805`


## Generated model files

Deep learning models are stored as PyTorch checkpoints:

```text
outputs/models/tcn/tcn_<event_id>_early3.pt
outputs/models/gru/gru_<event_id>_early3.pt
outputs/models/transformer/transformer_<event_id>_early3.pt
```

LightGBM baseline models are stored as joblib files:

```text
outputs/models/lgbm/lgbm_<event_id>.joblib
```

Prediction files are stored separately:

```text
outputs/predictions/tcn_early3_predictions.parquet
outputs/predictions/gru_early3_predictions.parquet
outputs/predictions/transformer_early3_predictions.parquet
outputs/predictions/lgbm_predictions.parquet
outputs/predictions/ensemble_predictions.parquet
```

Important distinction:

```text
Model checkpoint files: .pt / .joblib
Prediction result files: .parquet / .csv
Map outputs: .geojson / .gpkg / .png
```

## Final spatial risk-map outputs

The final risk map joins `slow_recovery_risk_map_ready.parquet` with the 5 km India grid geometry.

Input geometry:

```text
data/interim/grid/india_grid_5km.shp
data/interim/grid/india_grid_5km.dbf
data/interim/grid/india_grid_5km.shx
data/interim/grid/india_grid_5km.prj
```

Generated map files:

```text
outputs/maps/slow_recovery_risk_map.geojson
outputs/maps/slow_recovery_risk_map.gpkg
outputs/maps/slow_recovery_risk_map_with_centroid.csv
outputs/maps/slow_recovery_risk_grid_aggregated.geojson
```

Generated figure files:

```text
outputs/figures/map_predicted_slow_recovery_risk.png
outputs/figures/map_top10_slow_recovery_risk.png
```

Interpretation:

- `map_predicted_slow_recovery_risk.png` visualizes grid-level predicted slow-recovery risk aggregated across events.
- `map_top10_slow_recovery_risk.png` highlights the top 10% predicted slow-recovery risk areas.
- The aggregated map uses the maximum predicted delayed-recovery probability across evaluated events for each grid cell. Therefore, it should be interpreted as a spatial high-risk screening layer across events, not as a claim that all regions simultaneously experienced delayed recovery.

Suggested figure captions:

**Figure. Predicted slow-recovery risk map aggregated across disaster events.**  
Each grid cell represents the maximum predicted delayed-recovery probability across the evaluated events.

**Figure. Top 10% predicted slow-recovery risk areas.**  
Highlighted grid cells indicate locations with the highest predicted delayed-recovery probabilities across events.

## Final execution order

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

## Windows post-processing for maps

The Windows local post-processing script used for final map generation was:

```text
make_slow_recovery_risk_map_windows_v4.py
```

It avoids Windows Tcl/Tk matplotlib errors by forcing the non-GUI Agg backend.

The successful final status was:

```text
Risk shape: (890953, 13)
Unique risk grid_id: 127284
Unique grid geometry grid_id: 127284
Matched unique grid_id: 127284
Missing geometry rows: 0
[DONE] Map generation completed successfully.
```

## RunPod-modified source files

The key files modified during RunPod execution were:

```text
src/ntlpol/models/train_dl.py
src/ntlpol/models/losses.py
```

They are included in the final output archive if the archive was created with:

```bash
tar -czf ntl_resilience_final_outputs_<timestamp>.tar.gz \
  outputs \
  logs \
  configs \
  src/ntlpol/models/train_dl.py \
  src/ntlpol/models/losses.py
```

## Leakage rule

The full pre-12 + event + post-24 NTL sequence is allowed only for recovery measurement and target construction.  
Early prediction models must use only pre-12 + event + post-3 or post-6 input windows.

## Real data extraction additions

The project includes Earth Engine export scripts for the real-data bridge:

```bash
python scripts/70_gee_export_vnp46a2_daily.py --help
python scripts/71_gee_export_gpm_event_rainfall.py --help
python scripts/72_gee_export_ghsl_static_features.py --help
python scripts/73_build_cyclone_hazard_from_ibtracs.py --help
python scripts/74_gee_export_vnp46a2_monthly_optional.py --help
python scripts/80_run_after_raw_downloads.py --help
```

See `REAL_DATA_WORKFLOW.md` for the exact order after uploading the grid to Earth Engine.

## Recommended external report-writing inputs

For writing the final report, use:

```text
README_UPDATED.md
performance_summary.csv
tcn_early3_event_metrics.csv
gru_early3_event_metrics.csv
transformer_early3_event_metrics.csv
lgbm_event_metrics.csv
lgbm_feature_importance.csv
map_predicted_slow_recovery_risk.png
map_top10_slow_recovery_risk.png
```

Avoid uploading large model checkpoint files unless the report writer specifically needs to inspect model internals:

```text
*.pt
*.joblib
large prediction parquet files
full logs
```
