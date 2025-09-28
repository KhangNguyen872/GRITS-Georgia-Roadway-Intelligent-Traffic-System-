#!/usr/bin/env python
# scripts/collect_live.py
# Append NWS (weather/alerts) + 511 GA (events) to single CSVs on each run.
# Adds a 1-row-per-run heartbeat file: data/live_logs/live_snapshots.csv

import os, csv, argparse, datetime as dt
import json
from pathlib import Path
import requests
from datetime import datetime, timezone

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

# ---------- 511 GA ----------
GA511_KEY = os.getenv("GA511_KEY", "").strip()
GRITS_UA = os.getenv("GRITS_UA", "grits-student/1.0")

GA511_URLS = [
    # primary
    "https://511ga.org/api/v2/get/event?format=json&key={key}",
    # some deployments expose a v3-ish or alt base; keep as fallbacks
    "https://511ga.org/api/v3/get/event?format=json&key={key}",
    "https://www.511ga.org/api/v2/get/event?format=json&key={key}",
]

def _session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": GRITS_UA,
        "Accept": "application/json, text/plain, */*",
    })
    return s

def _normalize_events(payload):
    """
    Accepts several shapes and returns a list of event dicts.
    Shapes handled:
      - list[dict]
      - {"events": [...]}, {"event": [...]}
      - {"data": {"events":[...]}}
      - GeoJSON: {"features":[{"properties":{...}, "geometry":{...}}, ...]}
      - single dict that looks like one event
    """
    data = payload
    events = None

    if isinstance(data, list):
        events = data
    elif isinstance(data, dict):
        # common keys
        for k in ("events", "event"):
            if k in data and isinstance(data[k], (list, dict)):
                events = data[k]
                break
        if events is None and "data" in data and isinstance(data["data"], dict):
            for k in ("events", "event"):
                if k in data["data"] and isinstance(data["data"][k], (list, dict)):
                    events = data["data"][k]
                    break
        # GeoJSON-style
        if events is None and "features" in data and isinstance(data["features"], list):
            evs = []
            for feat in data["features"]:
                props = feat.get("properties", {}) if isinstance(feat, dict) else {}
                geom = feat.get("geometry", {}) if isinstance(feat, dict) else {}
                # flatten a little
                if isinstance(geom, dict) and geom.get("type") == "Point":
                    coords = geom.get("coordinates", [])
                    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                        props["lon"] = coords[0]
                        props["lat"] = coords[1]
                evs.append(props or feat)
            events = evs

        # single event dict fallback
        if events is None and any(k in data for k in ("id","eventId","type","headline","description")):
            events = [data]

    if isinstance(events, dict):
        events = [events]
    if not isinstance(events, list):
        events = []

    # final: ensure dicts
    events = [e for e in events if isinstance(e, dict)]
    return events

def _v(e, *names):
    """Look up a value across event, properties, attributes by several names."""
    props = e.get("properties", {}) if isinstance(e, dict) else {}
    attrs = e.get("attributes", {}) if isinstance(e, dict) else {}
    for n in names:
        for scope in (e, props, attrs):
            if isinstance(scope, dict) and n in scope and scope[n] not in (None, ""):
                return scope[n]
    return None

def ga511_events(debug=False):
    if not GA511_KEY:
        return [], "GA511_KEY not set"

    s = _session()
    last_err = None

    for base in GA511_URLS:
        url = base.format(key=GA511_KEY)
        try:
            r = s.get(url, timeout=(8, 25))
            status = r.status_code
            if status >= 400:
                last_err = f"HTTP {status} from {url}"
                if debug:
                    print(f"[ga511] {last_err}")
                continue
            try:
                data = r.json()
            except Exception as je:
                snippet = (r.text or "")[:300].replace("\n", " ")
                last_err = f"JSON error: {je}; body[:300]={snippet!r}"
                if debug:
                    print(f"[ga511] {last_err}")
                continue

            evs = _normalize_events(data)
            if debug and evs:
                sample = evs[0]
                print(f"[ga511] sample keys: {list(sample.keys())[:25]}")
                if isinstance(sample.get("properties"), dict):
                    print(f"[ga511] properties keys: {list(sample['properties'].keys())[:25]}")
                if isinstance(sample.get("attributes"), dict):
                    print(f"[ga511] attributes keys: {list(sample['attributes'].keys())[:25]}")
            if debug:
                print(f"[ga511] URL ok: {url}  parsed events={len(evs)}")
                # If zero, print a hint about data shape
                if len(evs) == 0:
                    t = type(data).__name__
                    keys = list(data.keys())[:10] if isinstance(data, dict) else None
                    print(f"[ga511] zero events; payload type={t} keys={keys}")
            return evs, None

        except requests.RequestException as e:
            last_err = f"Network error: {e} for {url}"
            if debug:
                print(f"[ga511] {last_err}")
            continue
        except Exception as e:
            last_err = f"Unexpected: {e} for {url}"
            if debug:
                print(f"[ga511] {last_err}")
            continue

    return [], last_err or "Unknown GA-511 error"

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

# ---------- main ----------
def run(lat: float, lon: float, ga511_debug: bool = False):
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
    events, ev_err = ga511_events(debug=ga511_debug)
    ev_rows = []
    empty_mapped = 0
    for e in events:
        row = {
            "snapshot_utc": snap_ts,
            "id": _v(e, "id", "ID", "eventId", "RID", "ref"),
            "type": _v(e, "type", "eventType", "EventType", "category"),
            "subtype": _v(e, "subtype", "subType", "Subtype", "detailType"),
            "headline": _v(e, "headline", "title", "description", "Description", "shortDesc", "message"),
            "status": _v(e, "status", "Status", "state"),
            "startTime": _v(e, "startTime", "StartDate", "start_time", "start", "created", "Reported"),
            "endTime": _v(e, "endTime", "PlannedEndDate", "end_time", "end", "updated", "LastUpdated"),
            "lat": _v(e, "latitude", "Latitude", "lat", "y"),
            "lon": _v(e, "longitude", "Longitude", "lon", "x"),
            "lanesBlocked": _v(e, "lanesBlocked", "LanesAffected", "lanes", "lanesAffected"),
            "roadName": _v(e, "roadName", "RoadwayName", "route", "roadway", "street"),
            "direction": _v(e, "direction", "DirectionOfTravel", "dir"),
            "source_has_properties": isinstance(e.get("properties"), dict) if isinstance(e, dict) else False,
            "source_has_attributes": isinstance(e.get("attributes"), dict) if isinstance(e, dict) else False,
            "raw_json": json.dumps(e, ensure_ascii=False),
        }
        mapped_values = [row[k] for k in [
            "id","type","subtype","headline","status","startTime","endTime",
            "lat","lon","lanesBlocked","roadName","direction"
        ]]
        if not any(v not in (None, "") for v in mapped_values):
            empty_mapped += 1
        ev_rows.append(row)

    if ga511_debug and empty_mapped:
        print(f"[ga511] warning: {empty_mapped} events had no mapped fields; see raw_json in CSV")

    append_csv(
        GA511_EVENTS_CSV,
        ["snapshot_utc","id","type","subtype","headline","status","startTime","endTime",
         "lat","lon","lanesBlocked","roadName","direction",
         "source_has_properties","source_has_attributes","raw_json"],
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
    ap.add_argument("--ga511-debug", action="store_true", help="Log GA-511 fetch debug info")
    args = ap.parse_args()
    run(args.lat, args.lon, ga511_debug=args.ga511_debug)
