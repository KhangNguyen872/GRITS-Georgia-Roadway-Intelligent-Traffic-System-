#!/usr/bin/env python
# scripts/collect_live.py
# Append NWS (weather/alerts) + 511 GA (events) to single CSVs on each run.
# Also writes a 1-row heartbeat to data/live_logs/live_snapshots.csv

import os
import csv
import argparse
import datetime as dt
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------- config ----------------
DATA_DIR = Path("data/live_logs")
NWS_HOURLY_CSV   = DATA_DIR / "nws_hourly.csv"
NWS_ALERTS_CSV   = DATA_DIR / "nws_alerts.csv"
GA511_EVENTS_CSV = DATA_DIR / "ga511_events.csv"
LIVE_SNAP_CSV    = DATA_DIR / "live_snapshots.csv"

PRUNE_DAYS = 90  # disabled below (uncomment if/when you want pruning)

# Friendly UA for public APIs (you can override with env GRITS_UA)
UA = {"User-Agent": os.getenv("GRITS_UA", "grits-student/1.0 (khangng872@gmail.com)")}
GA511_KEY = os.getenv("GA511_KEY", "").strip()

# ensure directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- helpers ----------------
def _session():
    retry = Retry(
        total=5, backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    s = requests.Session()
    s.headers.update(UA)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def ts_iso(d=None):
    return (d or dt.datetime.now(dt.timezone.utc)).replace(microsecond=0).isoformat()

def append_csv(path: Path, header: list[str], rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in header})

def prune_csv_by_days(path: Path, date_field: str):
    if not path.exists():
        return
    try:
        import pandas as pd
        df = pd.read_csv(path)
        if df.empty or date_field not in df.columns:
            return
        df[date_field] = pd.to_datetime(df[date_field], errors="coerce", utc=True)
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=PRUNE_DAYS)
        df = df[df[date_field] >= cutoff]
        tmp = path.with_suffix(".tmp.csv")
        df.to_csv(tmp, index=False)
        tmp.replace(path)
    except Exception as e:
        print(f"[prune] warning for {path.name}: {e}")


# ---------------- NWS ----------------
def nws_hourly(lat: float, lon: float):
    s = _session()
    try:
        p = s.get(f"https://api.weather.gov/points/{lat},{lon}", timeout=(8, 20))
        p.raise_for_status()
        hourly_url = p.json()["properties"]["forecastHourly"]

        fh = s.get(hourly_url, timeout=(8, 20))
        fh.raise_for_status()
        periods = fh.json().get("properties", {}).get("periods", []) or []

        al = s.get(f"https://api.weather.gov/alerts/active?point={lat},{lon}", timeout=(8, 20))
        al.raise_for_status()
        alerts = al.json()
        return periods, alerts, None
    except Exception as e:
        return [], {"features": []}, str(e)


# ---------------- 511 GA ----------------
def ga511_events():
    key = os.getenv("GA511_KEY", "").strip()
    if not key:
        return [], "GA511_KEY not set"
    try:
        s = _session()
        url = f"https://511ga.org/api/v2/get/event?format=json&key={key}"
        r = s.get(url, timeout=(8, 20))
        r.raise_for_status()
        data = r.json()

        # Normalize to a flat list[dict]
        events: list[dict] = []

        def _flatten(obj):
            if isinstance(obj, dict):
                # common wrappers in various 511 APIs
                for k in ("events", "event", "data"):
                    if k in obj:
                        _flatten(obj[k])
                        return
                events.append(obj)  # treat as single event dict
            elif isinstance(obj, list):
                for x in obj:
                    _flatten(x)
            # ignore scalars/None

        _flatten(data)
        return events, None

    except requests.HTTPError as e:
        # Fix: r is not defined in this context, use e.response instead
        try:
            response = e.response
            return [], f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception:
            return [], f"HTTP error: {e}"
    except Exception as e:
        return [], repr(e)


# ---------------- main ----------------
def run(lat: float, lon: float):
    now = dt.datetime.now(dt.timezone.utc)
    snap_ts = ts_iso(now)

    # ---- NWS (hourly forecast periods) ----
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
        ["snapshot_utc", "startTime", "endTime", "temperature", "temperatureUnit",
         "windSpeed", "windDirection", "shortForecast", "detailedForecast"],
        nws_rows
    )

    # ---- NWS (alerts) ----
    feats = []
    if isinstance(alerts, dict):
        feats = alerts.get("features", []) or []
    elif isinstance(alerts, list):
        feats = alerts  # some responses may already be a list of features

    alerts_rows = []
    for a in feats:
        if not isinstance(a, dict):
            continue
        prop = a.get("properties", {}) if isinstance(a.get("properties"), dict) else {}
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
        ["snapshot_utc", "id", "event", "severity", "certainty", "urgency",
         "effective", "onset", "ends", "headline", "areaDesc"],
        alerts_rows
    )

    # ---- 511 events ----
    events, ev_err = ga511_events()

    def pick(d: dict, *keys):
        for k in keys:
            v = d.get(k)
            if v not in (None, "", "null"):
                return v
        return None

    def pick_nested(d: dict, *paths):
        # paths are tuples like ("location","latitude") or ("geometry","coordinates",1)
        for path in paths:
            cur = d
            ok = True
            for seg in path:
                if isinstance(cur, dict) and seg in cur:
                    cur = cur[seg]
                elif isinstance(cur, (list, tuple)) and isinstance(seg, int) and 0 <= seg < len(cur):
                    cur = cur[seg]
                else:
                    ok = False
                    break
            if ok and cur not in (None, "", "null"):
                return cur
        return None

    ev_rows = []
    for e in events:
        if not isinstance(e, dict):
            continue

        # coordinates from many shapes
        lat = pick(e, "latitude", "lat", "Latitude", "Y")
        lon = pick(e, "longitude", "lon", "Longitude", "X")
        if lat is None or lon is None:
            # try nested: location{}, geometry{coordinates [lon,lat]}
            lat = pick_nested(e, ("location","latitude"), ("point","lat"), ("geo","lat"))
            lon = pick_nested(e, ("location","longitude"), ("point","lon"), ("geo","lon"))
            if lat is None or lon is None:
                # GeoJSON: geometry.coordinates = [lon, lat]
                glon = pick_nested(e, ("geometry","coordinates",0))
                glat = pick_nested(e, ("geometry","coordinates",1))
                lat = lat if lat is not None else glat
                lon = lon if lon is not None else glon
        try:
            flat = float(lat) if lat is not None else None
            flon = float(lon) if lon is not None else None
        except Exception:
            flat, flon = None, None

        ev_rows.append({
            "snapshot_utc": snap_ts,
            "id": pick(e, "id", "eventId", "incidentId", "Id"),
            "type": pick(e, "type", "eventType", "typeDesc"),
            "subtype": pick(e, "subtype", "eventSubType", "subTypeDesc"),
            "headline": pick(e, "headline", "title", "shortDescription", "description", "fullDesc"),
            "status": pick(e, "status", "eventStatus", "Status"),
            "startTime": pick(e, "startTime", "start_time", "starttime", "StartTime"),
            "endTime": pick(e, "endTime", "end_time", "endtime", "EndTime"),
            "lat": flat,
            "lon": flon,
            "lanesBlocked": pick(e, "lanesBlocked", "lanes", "lanesAffected"),
            "roadName": pick(e, "roadName", "route", "routeName", "road", "street", "roadwayName"),
            "direction": pick(e, "direction", "dir", "Direction"),
        })

    append_csv(
        GA511_EVENTS_CSV,
        ["snapshot_utc","id","type","subtype","headline","status","startTime","endTime",
        "lat","lon","lanesBlocked","roadName","direction"],
        ev_rows
    )


    # ---- 1-line heartbeat snapshot (for quick monitoring) ----
    wx_short, wx_temp, wx_wind, wx_precip = None, None, None, None
    try:
        import pandas as pd
        if periods:
            df = pd.DataFrame(periods)
            df["startTime"] = pd.to_datetime(df.get("startTime"), errors="coerce", utc=True)
            df["endTime"]   = pd.to_datetime(df.get("endTime"), errors="coerce", utc=True)
            in_now = df[(df["startTime"] <= now) & (df["endTime"] > now)]
            row = in_now.iloc[0] if not in_now.empty else df.iloc[(df["startTime"] - now).abs().argsort()[:1]].iloc[0]
            wx_short = str((row.get("shortForecast") or "")).strip()
            wx_temp  = row.get("temperature")
            wx_wind  = row.get("windSpeed")
            s = wx_short.lower()
            wx_precip = any(k in s for k in ["rain", "thunder", "storm", "showers", "snow", "hail", "drizzle"])
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
        "ga511_ok": (ev_err is None),
        "ga511_events_count": len(ev_rows),
        "errors": "; ".join([e for e in [nws_err, ev_err] if e]) if (nws_err or ev_err) else "",
    }
    append_csv(
        LIVE_SNAP_CSV,
        ["snapshot_utc", "nws_ok", "wx_short", "wx_temp", "wx_wind",
         "wx_precip_flag", "nws_periods_written", "alerts_count",
         "ga511_ok", "ga511_events_count", "errors"],
        [snap_row]
    )

    print(f"Data directory: {DATA_DIR.absolute()}")
    # Pruning disabled by default; re-enable when your files get large:
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
