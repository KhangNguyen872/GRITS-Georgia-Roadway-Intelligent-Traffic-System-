#!/usr/bin/env python
# trip_predictor.py — GBT prototype + optional live bumps + optional route polyline
import argparse, json, math, os, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, joblib
import requests
from typing import List, Tuple, Dict, Any, Literal

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODEL_PATH = HERE / "models" / "latest_model.pkl"
SCHEMA_PATH = HERE / "outputs" / "feature_schema.json"
LIVE_ROOT = DATA / "live_logs"

NWS_HOURLY_CSV = LIVE_ROOT / "nws_hourly.csv"
GA511_EVENTS_CSV = LIVE_ROOT / "ga511_events.csv"

# ---------- helpers ----------
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.7613
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def road_freeflow_mph(roadway: str) -> float:
    s = (roadway or "").upper().strip()
    if s.startswith("I-"): return 70.0
    if s.startswith("SR-") or s.startswith("GA-") or s.startswith("US-"): return 45.0
    return 50.0

def build_station_table():
    aadt_new = pd.read_csv(DATA / "aadt_and_truckpct.csv", low_memory=False)
    aadt_old = pd.read_csv(DATA / "GDOT_Traffic_Counts_(AADT_and_Truck_Percent)_2008_to_2017.csv", low_memory=False)
    annual   = pd.read_csv(DATA / "annualized_statistics.csv", low_memory=False)
    def std(df):
        import re
        df = df.copy()
        df.columns = [re.sub(r'[^A-Z0-9_]+','', c.upper().strip().replace(" ","_")) for c in df.columns]
        return df
    aadt_new, aadt_old, annual = std(aadt_new), std(aadt_old), std(annual)
    aadt_old = aadt_old.rename(columns={c: c.replace("TRUCKPCT_", "TRUCK_")
                                        for c in aadt_old.columns if c.startswith("TRUCKPCT_")})
    m = aadt_old.merge(aadt_new, on=["STATION_ID","FUNCTIONAL_CLASS"], how="outer", suffixes=("_OLD","_NEW"))
    m = m.merge(annual[["STATION_ID","KFACTOR","DFACTOR","STATION_TYPE"]].drop_duplicates(), on="STATION_ID", how="left")
    # lat/lon
    if "LAT" in m.columns: m["LAT"] = pd.to_numeric(m["LAT"], errors="coerce")
    elif "LATITUDE" in m.columns: m["LAT"] = pd.to_numeric(m["LATITUDE"], errors="coerce")
    if "LON" in m.columns: m["LON"] = pd.to_numeric(m["LON"], errors="coerce")
    elif "LONG" in m.columns: m["LON"] = pd.to_numeric(m["LONG"], errors="coerce")
    elif "LONGITUDE" in m.columns: m["LON"] = pd.to_numeric(m["LONGITUDE"], errors="coerce")
    # years
    import re
    aadt_cols = sorted([c for c in m.columns if re.match(r"^AADT_\d{4}$", c)])
    truck_cols = sorted([c for c in m.columns if re.match(r"^TRUCK_\d{4}$", c)])
    for c in aadt_cols + truck_cols + ["KFACTOR","DFACTOR"]:
        if c in m.columns: m[c] = pd.to_numeric(m[c], errors="coerce")
    years = [int(c.split("_")[1]) for c in aadt_cols] if aadt_cols else []
    latest = max(years) if years else None
    if latest is None: raise SystemExit("No AADT year columns found.")
    m["AADT_LATEST"] = m.get(f"AADT_{latest}")

    def slope(row, k=6):
        ys = sorted(years)[-k:]; vals = [row.get(f"AADT_{y}", np.nan) for y in ys]
        xs_vs = [(y,v) for y,v in zip(ys, vals) if pd.notna(v)]
        if len(xs_vs) >= 2:
            xs, vs = zip(*xs_vs); x = np.array(xs); v = np.array(vs)
            return float(np.polyfit(x, v, 1)[0])
        return np.nan
    m["AADT_TREND_SLOPE"] = m.apply(slope, axis=1)
    def roll3(row):
        ys = sorted(years)[-3:]; vals = [row.get(f"AADT_{y}", np.nan) for y in ys]
        vals = [v for v in vals if pd.notna(v)]
        return pd.Series({"AADT_MEAN_3YR": np.mean(vals) if vals else np.nan,
                          "AADT_STD_3YR":  np.std(vals) if vals else np.nan})
    m[["AADT_MEAN_3YR","AADT_STD_3YR"]] = m.apply(roll3, axis=1)
    def mean_or_nan(row, cols):
        xs = [row.get(c, np.nan) for c in cols]; xs = [float(x) for x in xs if pd.notna(x)]
        return float(np.mean(xs)) if xs else np.nan
    m["TRUCK_PCT_MEAN"]   = m.apply(lambda r: mean_or_nan(r, truck_cols), axis=1)
    m["TRUCK_PCT_LATEST"] = m.get(f"TRUCK_{latest}", np.nan)
    return m.dropna(subset=["LAT","LON"]).copy()

def estimate_probability_at_point(clf, schema, stations_df, lat, lon, k=10, max_r=25.0):
    dists = stations_df[["LAT","LON"]].apply(lambda r: haversine_miles(lat, lon, r["LAT"], r["LON"]), axis=1)
    local = stations_df.assign(DIST_MILES=dists).sort_values("DIST_MILES").head(k).copy()
    local = local[local["DIST_MILES"] <= max_r] if not local.empty else local
    FEATURES = schema["numeric_features"] + schema["categorical_features"]
    for c in FEATURES:
        if c not in local.columns: local[c] = np.nan
    X = local[FEATURES]
    probs = clf.predict_proba(X)[:,1]
    w = 1.0 / (local["DIST_MILES"].values + 1e-3); w = w / w.sum()
    return float(np.dot(probs, w)), local.assign(PROB=probs, WEIGHT=w)

def adjust_for_departure_time(p, leave_in):
    depart = dt.datetime.now() + dt.timedelta(minutes=leave_in or 0)
    hour = depart.hour
    bump = 0.10 if 7 <= hour <= 9 else 0.12 if 16 <= hour <= 18 else 0.0
    return float(max(0.0, min(1.0, p + bump)))

def probability_to_speed_multiplier(p):
    return float(1.0 - 0.4 * p)  # p=1 => 60% of freeflow

# ---------- LIVE CSV BUMPS (single CSVs) ----------
def apply_live_bumps(p_base, leave_in_min):
    details = {"live_used": False}
    depart_utc = dt.datetime.utcnow() + dt.timedelta(minutes=leave_in_min or 0)
    bump_total = 0.0

    # NWS hourly
    try:
        if NWS_HOURLY_CSV.exists():
            nws = pd.read_csv(NWS_HOURLY_CSV)
            if len(nws):
                nws["startTime"] = pd.to_datetime(nws["startTime"], errors="coerce", utc=True)
                target = depart_utc.replace(tzinfo=dt.timezone.utc)
                idx = (nws["startTime"] - target).abs().argsort()[:1]
                row = nws.iloc[idx]
                short = row["shortForecast"].astype(str).str.lower().iloc[0] if len(row) else ""
                wind = row["windSpeed"].astype(str).str.extract(r"(\d+)").astype(float).fillna(0).iloc[0,0] if len(row) else 0.0
                nb = 0.0
                if any(k in short for k in ["rain","thunder","snow","storm","showers"]): nb += 0.08
                if wind and wind > 20: nb += 0.03
                bump_total += nb; details["nws_bump"] = round(nb,3); details["nws_desc"] = short
    except Exception:
        pass

    # 511 events (coarse bump)
    try:
        if GA511_EVENTS_CSV.exists():
            ev = pd.read_csv(GA511_EVENTS_CSV)
            if len(ev):
                eb = 0.05
                lanes = pd.to_numeric(ev.get("lanesBlocked", pd.Series([])), errors="coerce").fillna(0)
                if (lanes >= 1).any(): eb += 0.10
                if (lanes >= 2).any(): eb += 0.10
                bump_total += eb; details["ga511_bump"] = round(eb,3); details["ga511_events"] = int(len(ev))
    except Exception:
        pass

    p = max(0.0, min(1.0, p_base + bump_total))
    details["live_used"] = ("nws_bump" in details) or ("ga511_bump" in details)
    details["prob_before_live"] = round(p_base,4)
    details["prob_after_live"]  = round(p,4)
    return p, details

# ---------- Polyline encoding ----------
def encode_polyline(coords: List[Tuple[float, float]]) -> str:
    result = []
    prev_lat = 0
    prev_lng = 0
    for lat, lng in coords:
        lat = int(round(lat * 1e5))
        lng = int(round(lng * 1e5))
        dlat = lat - prev_lat
        dlng = lng - prev_lng
        for v in (dlat, dlng):
            v = ~(v << 1) if v < 0 else (v << 1)
            while v >= 0x20:
                result.append(chr((0x20 | (v & 0x1f)) + 63))
                v >>= 5
            result.append(chr(v + 63))
        prev_lat, prev_lng = lat, lng
    return "".join(result)

# ---------- Routing helpers ----------
def route_via_geoapify(olat, olon, dlat, dlon, mode="drive"):
    key = os.getenv("GEOAPIFY_KEY", "").strip()
    if not key:
        return False, {"error": "GEOAPIFY_KEY not set", "provider": "geoapify"}
    url = ("https://api.geoapify.com/v1/routing"
           f"?waypoints={olat},{olon}|{dlat},{dlon}&mode={mode}&apiKey={key}")
    try:
        r = requests.get(url, timeout=20); r.raise_for_status()
        gj = r.json()
        feat = (gj.get("features") or [None])[0]
        if not feat: return False, {"error": "no features", "provider": "geoapify"}
        geom = feat.get("geometry", {}); props = feat.get("properties", {})
        coords_lonlat = geom.get("coordinates")
        # LineString => [[lon,lat],...]; MultiLineString => [[[lon,lat],...], ...]
        if coords_lonlat and isinstance(coords_lonlat[0][0], (int, float)):
            line = coords_lonlat
        else:
            line = (coords_lonlat or [[ ]])[0]
        coords_latlon = [(c[1], c[0]) for c in line]
        dist_mi = (props.get("distance") or 0)/1609.344
        dur_min = (props.get("time") or 0)/60.0
        return True, {"provider":"geoapify","distance_miles":dist_mi,"duration_minutes":dur_min,
                      "coordinates":coords_latlon,"encoded":encode_polyline(coords_latlon) if coords_latlon else None}
    except Exception as e:
        return False, {"error": repr(e), "provider": "geoapify"}

def route_via_ors(olat, olon, dlat, dlon, profile="driving-car"):
    key = os.getenv("ORS_KEY", os.getenv("OPENROUTESERVICE_KEY","")).strip()
    if not key:
        return False, {"error": "ORS_KEY not set", "provider": "ors"}
    url = f"https://api.openrouteservice.org/v2/directions/{profile}"
    payload = {"coordinates": [[olon, olat], [dlon, dlat]]}
    headers = {"Authorization": key, "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=20); r.raise_for_status()
        gj = r.json()
        feat = (gj.get("features") or [None])[0]
        if not feat: return False, {"error":"no features","provider":"ors"}
        geom = feat.get("geometry", {}); props = feat.get("properties", {}).get("summary", {})
        line = geom.get("coordinates") or []
        coords_latlon = [(c[1], c[0]) for c in line]
        dist_mi = (props.get("distance") or 0)/1609.344
        dur_min = (props.get("duration") or 0)/60.0
        return True, {"provider":"ors","distance_miles":dist_mi,"duration_minutes":dur_min,
                      "coordinates":coords_latlon,"encoded":encode_polyline(coords_latlon) if coords_latlon else None}
    except Exception as e:
        return False, {"error": repr(e), "provider":"ors"}

def route_via_google(olat, olon, dlat, dlon, mode="driving"):
    key = os.getenv("GOOGLE_MAPS_API_KEY","").strip()
    if not key:
        return False, {"error":"GOOGLE_MAPS_API_KEY not set", "provider":"google"}
    url = ("https://maps.googleapis.com/maps/api/directions/json"
           f"?origin={olat},{olon}&destination={dlat},{dlon}&mode={mode}&key={key}")
    try:
        r = requests.get(url, timeout=20); r.raise_for_status()
        js = r.json(); routes = js.get("routes") or []
        if not routes: return False, {"error": js.get("status") or "no routes", "provider":"google"}
        r0 = routes[0]
        dist_m = 0; dur_s = 0
        for leg in r0.get("legs", []):
            dist_m += (leg.get("distance", {}).get("value") or 0)
            dur_s += (leg.get("duration", {}).get("value") or 0)
        enc = r0.get("overview_polyline", {}).get("points")
        return True, {"provider":"google","distance_miles":dist_m/1609.344,"duration_minutes":dur_s/60.0,
                      "coordinates": None, "encoded": enc}
    except Exception as e:
        return False, {"error": repr(e), "provider":"google"}

def get_route_geometry(router: Literal["auto","geoapify","ors","google","none"],
                       olat: float, olon: float, dlat: float, dlon: float):
    if router == "none":
        return False, {"provider":"none","error":"routing disabled"}
    if router == "auto":
        if os.getenv("GEOAPIFY_KEY"): router = "geoapify"
        elif os.getenv("ORS_KEY") or os.getenv("OPENROUTESERVICE_KEY"): router = "ors"
        elif os.getenv("GOOGLE_MAPS_API_KEY"): router = "google"
        else: return False, {"provider":"auto","error":"no routing keys set"}
    if router == "geoapify": return route_via_geoapify(olat, olon, dlat, dlon)
    if router == "ors":      return route_via_ors(olat, olon, dlat, dlon)
    if router == "google":   return route_via_google(olat, olon, dlat, dlon)
    return False, {"error": f"unknown router {router}"}

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roadway", required=True)
    ap.add_argument("--origin-lat", type=float, required=True)
    ap.add_argument("--origin-lon", type=float, required=True)
    ap.add_argument("--dest-lat", type=float, required=True)
    ap.add_argument("--dest-lon", type=float, required=True)
    ap.add_argument("--leave-in", type=int, default=0)
    ap.add_argument("--live", action="store_true", help="blend NWS/511 bumps from data/live_logs/*.csv")
    ap.add_argument("--horizon", type=int, default=15)
    # NEW
    ap.add_argument("--router", default="auto", choices=["auto","geoapify","ors","google","none"])
    ap.add_argument("--polyline-format", default="encoded", choices=["encoded","geojson"])
    args = ap.parse_args()

    clf = joblib.load(MODEL_PATH)
    schema = json.load(open(SCHEMA_PATH, "r"))
    stations = build_station_table()

    # base probability at trip midpoint
    mid_lat = (args.origin_lat + args.dest_lat)/2.0
    mid_lon = (args.origin_lon + args.dest_lon)/2.0
    p_base, _ = estimate_probability_at_point(clf, schema, stations, mid_lat, mid_lon, k=10, max_r=25.0)
    p_time = adjust_for_departure_time(p_base, args.leave_in)

    live_info = {"live_used": False}
    if args.live:
        p_time, live_info = apply_live_bumps(p_time, args.leave_in)

    # distance & ETA (fallback)
    dist_mi_sl = haversine_miles(args.origin_lat, args.origin_lon, args.dest_lat, args.dest_lon)
    path_factor = 1.10 if args.roadway.upper().startswith("I-") else 1.20
    route_mi_est = dist_mi_sl * path_factor
    freeflow = road_freeflow_mph(args.roadway)
    speed_mult = probability_to_speed_multiplier(p_time)
    eff_speed = max(5.0, freeflow * speed_mult)
    eta_hr = route_mi_est / eff_speed
    eta_min = eta_hr * 60.0

    # routing geometry
    route_ok, route_info = get_route_geometry(args.router, args.origin_lat, args.origin_lon,
                                              args.dest_lat, args.dest_lon)
    if route_ok:
        r_dist_mi = float(route_info.get("distance_miles") or route_mi_est)
        eta_hr = r_dist_mi / max(5.0, eff_speed)
        eta_min = eta_hr * 60.0
        if args.polyline_format == "encoded":
            encoded = route_info.get("encoded")
            if not encoded and route_info.get("coordinates"):
                encoded = encode_polyline(route_info["coordinates"])
            route_payload = {
                "provider": route_info.get("provider"),
                "distance_miles": round(r_dist_mi, 3),
                "base_duration_minutes": round(float(route_info.get("duration_minutes") or 0.0), 1),
                "polyline": {"format": "encoded", "encoded": encoded},
            }
        else:
            coords = route_info.get("coordinates") or []
            route_payload = {
                "provider": route_info.get("provider"),
                "distance_miles": round(r_dist_mi, 3),
                "base_duration_minutes": round(float(route_info.get("duration_minutes") or 0.0), 1),
                "geojson": {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in coords]},
            }
    else:
        coords = [(args.origin_lat, args.origin_lon), (args.dest_lat, args.dest_lon)]
        enc = encode_polyline(coords)
        route_payload = {
            "provider": route_info.get("provider", "fallback"),
            "distance_miles": round(route_mi_est, 3),
            "base_duration_minutes": None,
            "polyline": {"format": "encoded", "encoded": enc} if args.polyline_format == "encoded"
                       else {"type": "LineString", "coordinates": [[args.origin_lon, args.origin_lat],
                                                                   [args.dest_lon, args.dest_lat]]}
        }

    out = {
        "ok": True,
        "model": MODEL_PATH.name,
        "horizon_minutes": args.horizon,
        "inputs": {
            "roadway": args.roadway,
            "origin": {"lat": args.origin_lat, "lon": args.origin_lon},
            "destination": {"lat": args.dest_lat, "lon": args.dest_lon},
            "leave_in_minutes": args.leave_in
        },
        "distance": {
            "straight_line_miles": round(dist_mi_sl,3),
            "route_distance_miles_used": round(route_payload.get("distance_miles", route_mi_est), 3),
            "path_factor_if_no_router": path_factor
        },
        "speed": {
            "freeflow_mph": freeflow,
            "probability_after_all_adjustments": round(p_time,4),
            "speed_multiplier": round(speed_mult,3),
            "effective_speed_mph": round(eff_speed,2)
        },
        "eta": {"minutes": round(eta_min,1), "hours": round(eta_hr,3)},
        "live": live_info,
        "route": route_payload,
        "notes": [
            "Prototype uses AADT-based proxy; live bumps add coarse weather/incident impact.",
            "Routing geometry is provided for map display; ETA comes from GBT + live bumps.",
            "When TFT is trained, multi-horizon forecasts will replace the coarse bumps."
        ]
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
