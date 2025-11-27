# DEVLOG

## 2025-11-01

- Instrumented `tft.data_pipeline` with stepwise logging, chunked NWS ingestion, dtype downcasting, and atomic writers. Added CLI knobs `--freq` (documented) and `--nws-chunksize` plus rich diagnostics (Python/pandas/pyarrow versions, RSS).
- Baseline: pre-change reports from on-call (user) indicated large CSV runs stalled for several minutes with high RAM; could not reproduce original runtime after refactor because branch already diverged.
- Smoke test (head slices):  
  `PYTHONUNBUFFERED=1 .venv/bin/python -m tft.data_pipeline --ga511-csv /tmp/ga511_10k.csv --nws-csv /tmp/nws_30k.csv --corridors "SR 1" --output tft/artifacts/test_dataset.csv --meta tft/artifacts/test_meta.json`  
  Duration ≈0.9 s wall clock, peak RSS 0.13 GB (see progress log). Generated 4,346 rows and metadata with atomic writes.
- Environment currently lacks `pyarrow/fastparquet`; parquet writes now fail fast with a clear message. Install one of them before running the full 1.1 M/200 k row build so output can remain parquet.
- Next validation: rerun command on full GA-511/NWS archives once pyarrow is available, targeting <10 min wall time and <12 GB RSS. Capture the streamed logs to confirm <20 s step cadence.
