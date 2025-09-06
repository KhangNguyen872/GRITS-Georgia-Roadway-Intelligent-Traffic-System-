#!/usr/bin/env python
# scripts/collect_live.py
# Append NWS (weather/alerts) + 511 GA (events) to single CSVs on each run.
# Includes retries/backoff, UTC timestamps, and auto-prunes to last 100 days.

import os, csv, argparse, datetime as dt
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------- config ----------
DATA_DIR = Path("data/live_logs")
DATA_DIR.mkdir(parents=True, exist_ok=True)

NWS_HOURLY_CSV = DATA_DIR / "nws_hourly.csv"
NWS_ALERTS_CSV = DATA_DIR / "nws_alerts.csv"
GA511_EVENTS_CSV = DATA_DIR / "ga511_events.csv"

PRUNE_DAYS = 100  # keep only last 100 days of rows

# Put a real contact per NWS guidance
UA = {"User-Agent": "jc-traffic-student/1.0 (your_email@school.org)"}

# Read 511 key from env (DO NOT hardcode)
GA511_KEY = os.getenv("GA511_KEY", "").strip()

def _session():
    retry = Retry(
        total=5,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    s = requests.Session()
    s.headers.update(UA)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def ts_iso(d=None):
    # timezone-aware UTC ISO8601
    return (d or dt.datetime.now(dt.timezone.utc)).replace(microsecond=0).isoformat()

def append_csv(path: Path, header: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in header})

def prune_csv_by_days(path: Path, date_field: str):
    """Keep only rows with date_field within last PRUNE_DAYS. No-op if file missing/empty."""
    if not path.exists(): return
    try:
        import pandas as pd
        df = pd.read_csv(path)
        if date_field not in df.columns or df.empty:
            return
        df[date_field] = pd.to_datetime(df[date_field], errors="coerce", utc=True)
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=PRUNE_DAYS)
        df = df[df[date_field] >= cutoff]
        # atomic-ish write: temp then replace
        tmp = path.with_suffix(".tmp.csv")
        df.to_csv(tmp, index=False)
        tmp.replace(path)
    except Exception as e:
        print(f"[prune] warning for {path.name}: {e}")

# ---------- NWS ----------
def nws_hourly(lat: float, lon: float):
    """Return (periods, alerts) or ([], {'features': []}) on failure."""
    s = _session()
    try:
        p = s.get(f"https://api.weather.gov/points/{lat},{lon}", timeout=(8, 20))
        p.raise_for_status()
        hourly_url = p.json()["properties"]["forecastHourly"]

        fh = s.get(hourly_url, timeout=(8, 20))
        fh.raise_for_status()
        periods = fh.json()["properties"]["periods"]

        al = s.get(f"https://api.weather.gov/alerts/active?point={lat},{lon}", timeout=(8, 20))
        al.raise_for_status()
        alerts = al.json()
        return periods, alerts
    except Exception as e:
        print(f"[NWS] warning: {e}")
        return [], {"features": []}

# ---------- 511 GA ----------
def ga511_events():
    """Return list of events; empty list if key missing or request fails."""
    if not GA511_KEY:
        print("[511] info: GA511_KEY not set; skipping events.")
        return []
    try:
        s = _session()
        url = f"https://511ga.org/api/v2/get/event?format=json&key={GA511_KEY}"
        r = s.get(url, timeout=(8, 20))
        r.raise_for_status()
        data = r.json()
        events = data.get("events") or data.get("event") or data
        if isinstance(events, dict):
            events = [events]
        if not isinstance(events, list):
            events = []
        return events
    except Exception as e:
        print(f"[511] warning: {e}")
        return []

# ---------- main ----------
def run(lat: float, lon: float):
    now = dt.datetime.now(dt.timezone.utc)
    snap_ts = ts_iso(now)

    # NWS hourly forecast snapshot
    periods, alerts = nws_hourly(lat, lon)
    nws_rows = [{
        "snapshot_utc": snap_ts,
        "startTime": p.get("startTime"),
        "endTime": p.get("endTime"),
        "temperature": p.get("temperature"),
        "temperatureUnit": p.get("temperatureUnit"),
        "windSpeed": p.get("windSpeed"),
        "windDirection": p.get("windDirection"),
        "shortForecast": p.get("shortForecast"),
        "detailedForecast": p.get("detailedForecast"),
    } for p in periods]
    append_csv(
        NWS_HOURLY_CSV,
        ["snapshot_utc","startTime","endTime","temperature","temperatureUnit","windSpeed","windDirection","shortForecast","detailedForecast"],
        nws_rows
    )

    # NWS alerts snapshot
    alerts_rows = []
    feats = (alerts or {}).get("features", [])
    for a in feats:
        prop = a.get("properties", {})
        alerts_rows.append({
            "snapshot_utc": snap_ts,
            "id": a.get("id"),
            "event": prop.get("event"),
            "severity": prop.get("severity"),
            "certainty": prop.get("certainty"),
            "urgency": prop.get("urgency"),
            "effective": prop.get("effective"),
            "onset": prop.get("onset"),
            "ends": prop.get("ends"),
            "headline": prop.get("headline"),
            "areaDesc": prop.get("areaDesc"),
        })
    append_csv(
        NWS_ALERTS_CSV,
        ["snapshot_utc","id","event","severity","certainty","urgency","effective","onset","ends","headline","areaDesc"],
        alerts_rows
    )

    # 511 events snapshot
    events = ga511_events()
    ev_rows = [{
        "snapshot_utc": snap_ts,
        "id": e.get("id") or e.get("eventId"),
        "type": e.get("type"),
        "subtype": e.get("subtype"),
        "headline": e.get("headline") or e.get("title") or e.get("description"),
        "status": e.get("status"),
        "startTime": e.get("startTime"),
        "endTime": e.get("endTime"),
        "lat": e.get("latitude") or e.get("lat"),
        "lon": e.get("longitude") or e.get("lon"),
        "lanesBlocked": e.get("lanesBlocked") or e.get("lanes"),
        "roadName": e.get("roadName") or e.get("route"),
        "direction": e.get("direction"),
    } for e in events]
    append_csv(
        GA511_EVENTS_CSV,
        ["snapshot_utc","id","type","subtype","headline","status","startTime","endTime","lat","lon","lanesBlocked","roadName","direction"],
        ev_rows
    )

    # prune old rows
    prune_csv_by_days(NWS_HOURLY_CSV, "snapshot_utc")
    prune_csv_by_days(NWS_ALERTS_CSV, "snapshot_utc")
    prune_csv_by_days(GA511_EVENTS_CSV, "snapshot_utc")

    print(f"Saved NWS hourly: {len(nws_rows)} | alerts: {len(alerts_rows)} | 511 events: {len(ev_rows)} at {snap_ts}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    args = ap.parse_args()
    run(args.lat, args.lon)
