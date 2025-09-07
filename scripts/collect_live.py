#!/usr/bin/env python
# scripts/collect_live.py
# Append NWS (weather/alerts) + 511 GA (events) to single CSVs on each run.
# Adds a 1-row-per-run heartbeat file: data/live_logs/live_snapshots.csv

import os, csv, argparse, datetime as dt
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------- config ----------
DATA_DIR = Path("data/live_logs")

try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"Error creating directory: {e}")
    exit(1)

NWS_HOURLY_CSV    = DATA_DIR / "nws_hourly.csv"
NWS_ALERTS_CSV    = DATA_DIR / "nws_alerts.csv"
GA511_EVENTS_CSV  = DATA_DIR / "ga511_events.csv"
LIVE_SNAP_CSV     = DATA_DIR / "live_snapshots.csv"   # NEW: 1 row per run

PRUNE_DAYS = 90

UA = {"User-Agent": "jc-traffic-student/1.0 (your_real_email@school.org)"}
GA511_KEY = os.getenv("GA511_KEY", "").strip()

def _session():
    retry = Retry(
        total=5, backoff_factor=1.2,
        status_forcelist=[429,500,502,503,504],
        allowed_methods=["GET"], respect_retry_after_header=True,
    )
    s = requests.Session()
    s.headers.update(UA)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def ts_iso(d=None):
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
    if not path.exists(): return
    try:
        import pandas as pd
        df = pd.read_csv(path)
        if df.empty or date_field not in df.columns: return
        df[date_field] = pd.to_datetime(df[date_field], errors="coerce", utc=True)
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=PRUNE_DAYS)
        df = df[df[date_field] >= cutoff]
        tmp = path.with_suffix(".tmp.csv")
        df.to_csv(tmp, index=False); tmp.replace(path)
    except Exception as e:
        print(f"[prune] warning for {path.name}: {e}")

# ---------- NWS ----------
def nws_hourly(lat: float, lon: float):
    s = _session()
    try:
        p = s.get(f"https://api.weather.gov/points/{lat},{lon}", timeout=(8,20)); p.raise_for_status()
        hourly_url = p.json()["properties"]["forecastHourly"]
        fh = s.get(hourly_url, timeout=(8,20)); fh.raise_for_status()
        periods = fh.json()["properties"]["periods"]
        al = s.get(f"https://api.weather.gov/alerts/active?point={lat},{lon}", timeout=(8,20)); al.raise_for_status()
        alerts = al.json()
        return periods, alerts, None
    except Exception as e:
        return [], {"features":[]}, str(e)

# ---------- 511 GA ----------
def ga511_events():
    if not GA511_KEY:
        return [], "GA511_KEY not set"
    try:
        s = _session()
        url = f"https://511ga.org/api/v2/get/event?format=json&key={GA511_KEY}"
        r = s.get(url, timeout=(8,20)); r.raise_for_status()
        data = r.json()
        events = data.get("events") or data.get("event") or data
        if isinstance(events, dict): events = [events]
        if not isinstance(events, list): events = []
        return events, None
    except Exception as e:
        return [], str(e)

# ---------- main ----------
def run(lat: float, lon: float):
    now = dt.datetime.now(dt.timezone.utc)
    snap_ts = ts_iso(now)

    # NWS
    periods, alerts, nws_err = nws_hourly(lat, lon)
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

    # 511
    events, ev_err = ga511_events()
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

    # --- NEW: write a 1-row snapshot per run (easy to see growth) ---
    # Pick the period that covers "now" (or nearest).
    wx_short, wx_temp, wx_wind, wx_precip = None, None, None, None
    try:
        import pandas as pd
        if periods:
            df = pd.DataFrame(periods)
            df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce", utc=True)
            df["endTime"]   = pd.to_datetime(df["endTime"], errors="coerce", utc=True)
            in_now = df[(df["startTime"] <= now) & (df["endTime"] > now)]
            row = in_now.iloc[0] if not in_now.empty else df.iloc[(df["startTime"] - now).abs().argsort()[:1]].iloc[0]
            wx_short = str(row.get("shortForecast") or "")
            wx_temp  = row.get("temperature")
            wx_wind  = row.get("windSpeed")
            s = wx_short.lower()
            wx_precip = any(k in s for k in ["rain","thunder","storm","showers","snow","hail","drizzle"])
    except Exception:
        pass

    snap_row = {
        "snapshot_utc": snap_ts,
        "nws_ok": nws_err is None,
        "wx_short": wx_short,
        "wx_temp": wx_temp,
        "wx_wind": wx_wind,
        "wx_precip_flag": wx_precip,
        "nws_periods_written": len(nws_rows),
        "alerts_count": len(alerts_rows),
        "ga511_ok": ev_err is None and GA511_KEY != "",
        "ga511_events_count": len(ev_rows),
        "errors": "; ".join([e for e in [nws_err, ev_err] if e]) if (nws_err or ev_err) else "",
    }
    append_csv(
        LIVE_SNAP_CSV,
        ["snapshot_utc","nws_ok","wx_short","wx_temp","wx_wind","wx_precip_flag","nws_periods_written","alerts_count","ga511_ok","ga511_events_count","errors"],
        [snap_row]
    )

    print(f"Data directory: {DATA_DIR.absolute()}")


    # prune
    # prune_csv_by_days(NWS_HOURLY_CSV, "snapshot_utc")
    # prune_csv_by_days(NWS_ALERTS_CSV, "snapshot_utc")
    # prune_csv_by_days(GA511_EVENTS_CSV, "snapshot_utc")
    # prune_csv_by_days(LIVE_SNAP_CSV, "snapshot_utc")

    print(f"Saved: hourly={len(nws_rows)} alerts={len(alerts_rows)} 511={len(ev_rows)} snapshot=1 @ {snap_ts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    args = ap.parse_args()
    run(args.lat, args.lon)
