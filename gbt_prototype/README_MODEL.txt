Commands to Install Requirements:
  python -m venv .venv
  .\.venv\Scripts\activate
  pip install --upgrade pip
  pip install -r requirements.txt


Prototype GBT Model for Congestion Proxy (Georgia AADT Stations)
- Label: 1 if AADT_2024 >= 75th percentile across stations; else 0
- Features: AADT trend (last 6y slope), 3-year mean & std, truck % stats, K/D factors, lat/lon, functional class, station type.
- Model: sklearn GradientBoostingClassifier inside a preprocessing Pipeline (imputers + OneHot).
- Notes:
  * This is station-based (no route mapping). To target specific highways (I-75, SR-141, etc.), map stations to segments and aggregate.
  * Remove AADT_LATEST from features to avoid label leakage (done).
  * Use this as a training prototype; replace label with travel-time or jam-level when available.



Live data snapshots (NWS + 511 GA)
 - This repo includes a tiny background collector that logs live weather (NWS) and GA 511 incidents to CSVs every 5 minutes. The predictor (trip_predictor.py) can blend these “live bumps” into the ETA when you pass --live.btw 
  What it does

  - Calls NWS (no API key) for hourly forecast + active alerts at a chosen lat/lon.
  - Calls 511 GA (API key required) for current incidents/closures.

  Appends rows to:

  - data/live_logs/nws_hourly.csv
  - data/live_logs/nws_alerts.csv
  - data/live_logs/ga511_events.csv

  Auto-prunes CSVs to the last 100 days to keep them small.

  Files

  - Collector: scripts/collect_live.py
  - Optional wrapper used by the scheduler: run_collect_live.bat