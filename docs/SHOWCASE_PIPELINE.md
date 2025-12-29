# Showcase Corridor Pipeline

End-to-end steps to build the showcase corridor artifacts (route TMCs already curated, ~38.7 miles).

## Command (run from repo root)
```bash
python3 TrainingModel/scripts/build_showcase_corridor_pipeline.py \
  --probe-path TrainingModel/data/Govt_Data/ShowCase_Data_Set/ShowCase_Data_Set.csv \
  --route-tmcs TrainingModel/data/Govt_Data/tmcs_showcase_route.csv \
  --tmc-id-path TrainingModel/data/Govt_Data/ShowCase_Data_Set/TMC_Identification.csv \
  --out-parquet TrainingModel/data/Govt_Data/npmrds_showcase_5min.parquet \
  --out-dir TrainingModel/tft/artifacts_showcase \
  --freq-minutes 5 \
  --horizons "5,15,30,60" \
  --epochs 20 \
  --batch-size 256 \
  --resume
```

Notes:
- Uses DuckDB to filter/aggregate directly from the large CSV/TSV; outputs Parquet with snappy compression.
- `--resume` skips steps whose outputs already exist. Remove it to force recompute.
- Default creates a filtered probe parquet (`showcase_probe_filtered.parquet`) alongside the raw file; add `--no-filter` to skip.

## What the pipeline produces
1) `TrainingModel/data/Govt_Data/showcase_probe_filtered.parquet` — filtered raw probe rows for only the route TMCs (optional).
2) `TrainingModel/data/Govt_Data/npmrds_showcase_5min.parquet` — 5-minute aggregated probe data (`ts_utc, tmc, speed, reference_speed, confidence`).
3) `TrainingModel/tft/artifacts_showcase/tmc_metadata_showcase.csv` — length from the curated route list; reference_speed_mph = per-TMC 0.9 quantile of aggregated reference_speed (fallback 65).
4) `TrainingModel/tft/artifacts_showcase/dataset.parquet` and `dataset_meta.json` — TFT-ready data (freq=5min, horizons=5/15/30/60).
5) `TrainingModel/tft/artifacts_showcase/tft_bundle.pt` — trained TFT bundle.
6) `TrainingModel/tft/artifacts_showcase/run_summary.json` — run metadata (tmc count, miles, date range, horizons, epochs).

## How to verify
- Route list: ~65 TMCs, ~38–40 miles. The script will abort if miles are outside 20–80.
- Aggregation summary (logged): row count, unique TMCs, min/max ts_utc, % reference_speed missing/zero.
- Check artifacts exist in `TrainingModel/tft/artifacts_showcase/` and the Parquet outputs in `TrainingModel/data/Govt_Data/`.

## Copying artifacts into nextroute
If the backend/API or predictor needs these artifacts:
- Copy `TrainingModel/tft/artifacts_showcase/tft_bundle.pt` into the API/predictor model location.
- Copy `dataset.parquet` and `dataset_meta.json` alongside the bundle if your predictor caches offline features.
- Copy `tmc_metadata_showcase.csv` as `tmc_metadata.csv` into the API resources so distance/speed metadata match the showcase route.

Keep restricted Gov’t data (raw probes) out of GitHub/submissions; only ship the derived artifacts above.***
