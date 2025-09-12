import os
import json
import tempfile
from typing import Any, Dict

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Core model/feature code lives in this module. Do not modify the model here.
import trip_predictor as tp


app = Flask(__name__)

# CORS configuration: allow dev origin by default; override via CORS_ORIGINS env (comma-separated)
_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
CORS(
    app,
    resources={
        r"/predict": {"origins": _cors_origins or "*"},
        r"/health": {"origins": "*"},
    },
)


# Optional remote sources if local files are not present.
MODEL_URL = os.getenv("MODEL_URL")
# Provide a sensible default for schema if not present locally
SCHEMA_URL = os.getenv(
    "SCHEMA_URL",
    "https://raw.githubusercontent.com/KhangNguyen872/GRITS-Georgia-Roadway-Intelligent-Traffic-System-/main/outputs/feature_schema.json",
)

_clf = None
_schema = None
_stations = None
_last_error = None


def _load_resources() -> bool:
    """Lazy-load the model, schema, and stations table.

    - Prefer local paths defined in trip_predictor (tp.MODEL_PATH / tp.SCHEMA_PATH)
    - Fallback to MODEL_URL / SCHEMA_URL if provided
    """
    global _clf, _schema, _stations, _last_error

    if _clf is not None and _schema is not None and _stations is not None:
        return True

    try:
        # Load model
        if getattr(tp, "MODEL_PATH", None) and os.path.exists(tp.MODEL_PATH):
            _clf = joblib.load(tp.MODEL_PATH)
        elif MODEL_URL:
            import requests

            r = requests.get(MODEL_URL, timeout=60)
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(r.content)
                temp_model_path = tf.name
            _clf = joblib.load(temp_model_path)
        else:
            _last_error = f"Model not found at {tp.MODEL_PATH} and MODEL_URL not set"
            return False

        # Load schema
        if getattr(tp, "SCHEMA_PATH", None) and os.path.exists(tp.SCHEMA_PATH):
            with open(tp.SCHEMA_PATH, "r", encoding="utf-8") as f:
                _schema = json.load(f)
        elif SCHEMA_URL:
            import requests

            r = requests.get(SCHEMA_URL, timeout=60)
            r.raise_for_status()
            _schema = r.json()
        else:
            _last_error = f"Schema not found at {tp.SCHEMA_PATH} and SCHEMA_URL not set"
            _clf = None
            return False

        # Build station table from data/
        _stations = tp.build_station_table()
        return True
    except Exception as e:  # pragma: no cover - defensive
        _last_error = str(e)
        _clf = None
        _schema = None
        _stations = None
        return False


@app.get("/health")
def health():
    ok = _load_resources()
    return jsonify(
        {
            "ok": bool(ok),
            "model_ready": _clf is not None,
            "schema_ready": _schema is not None,
            "stations_ready": _stations is not None,
            "error": None if ok else _last_error,
        }
    )


def _bad_request(msg: str):
    return jsonify({"ok": False, "error": msg}), 400


@app.post("/predict")
def predict():
    if not _load_resources():
        return jsonify({"ok": False, "error": _last_error or "Model not ready"}), 500

    try:
        data: Dict[str, Any] = request.get_json(force=True) or {}
    except Exception:
        return _bad_request("Invalid or missing JSON body")

    try:
        roadway = str(data["roadway"]).strip()
        o = data["origin"]; d = data["destination"]
        o_lat, o_lon = float(o["lat"]), float(o["lon"])
        d_lat, d_lon = float(d["lat"]), float(d["lon"])
    except Exception:
        return _bad_request("Payload must include roadway, origin{lat,lon}, destination{lat,lon}")

    leave_in = int(data.get("leave_in_minutes", data.get("leave_in", 0)) or 0)
    live = bool(data.get("live", True))
    router = str(data.get("router", "auto"))
    polyfmt = str(data.get("polyline_format", "encoded"))

    # Base probability at trip midpoint
    mid_lat = (o_lat + d_lat) / 2.0
    mid_lon = (o_lon + d_lon) / 2.0
    p_base, _ = tp.estimate_probability_at_point(_clf, _schema, _stations, mid_lat, mid_lon, k=10, max_r=25.0)
    p_time = tp.adjust_for_departure_time(p_base, leave_in)

    if live:
        try:
            p_time, live_info = tp.apply_live_bumps(p_time, leave_in)
        except Exception:
            # Be resilient if live bumps encounter issues
            live_info = {"live_used": False, "note": "live bumps unavailable"}
    else:
        live_info = {"live_used": False}

    # Distance and ETA fallback
    dist_mi_sl = tp.haversine_miles(o_lat, o_lon, d_lat, d_lon)
    path_factor = 1.10 if roadway.upper().startswith("I-") else 1.20
    route_mi_est = dist_mi_sl * path_factor
    freeflow = tp.road_freeflow_mph(roadway)
    speed_mult = tp.probability_to_speed_multiplier(p_time)
    eff_speed = max(5.0, freeflow * speed_mult)
    eta_hr = route_mi_est / max(5.0, eff_speed)
    eta_min = eta_hr * 60.0

    # Try to obtain route geometry via provider, if available in tp
    route_payload = None
    r_provider = "fallback"
    try:
        if hasattr(tp, "get_route_geometry"):
            ok, info = tp.get_route_geometry(router, o_lat, o_lon, d_lat, d_lon)
            if ok:
                r_dist_mi = float(info.get("distance_miles") or route_mi_est)
                eta_hr = r_dist_mi / max(5.0, eff_speed)
                eta_min = eta_hr * 60.0
                if polyfmt == "encoded":
                    encoded = info.get("encoded")
                    if not encoded and info.get("coordinates"):
                        encoded = tp.encode_polyline(info["coordinates"])  # type: ignore
                    route_payload = {
                        "provider": info.get("provider"),
                        "distance_miles": round(r_dist_mi, 3),
                        "base_duration_minutes": round(float(info.get("duration_minutes") or 0.0), 1),
                        "polyline": {"format": "encoded", "encoded": encoded},
                    }
                else:
                    coords = info.get("coordinates") or []
                    route_payload = {
                        "provider": info.get("provider"),
                        "distance_miles": round(r_dist_mi, 3),
                        "base_duration_minutes": round(float(info.get("duration_minutes") or 0.0), 1),
                        "geojson": {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in coords]},
                    }
                r_provider = route_payload.get("provider") if isinstance(route_payload, dict) else r_provider
    except Exception:
        # fall back below
        pass

    if route_payload is None:
        enc = tp.encode_polyline([(o_lat, o_lon), (d_lat, d_lon)])
        route_payload = {
            "provider": r_provider,
            "distance_miles": round(route_mi_est, 3),
            "base_duration_minutes": None,
            "polyline": {"format": "encoded", "encoded": enc} if polyfmt == "encoded" else None,
            "geojson": {"type": "LineString", "coordinates": [[o_lon, o_lat], [d_lon, d_lat]]} if polyfmt != "encoded" else None,
        }

    return jsonify(
        {
            "ok": True,
            "eta": {"minutes": round(eta_min, 1), "hours": round(eta_hr, 3)},
            "speed": {
                "freeflow_mph": freeflow,
                "probability_after_all_adjustments": round(p_time, 4),
                "speed_multiplier": round(speed_mult, 3),
                "effective_speed_mph": round(eff_speed, 2),
            },
            "route": route_payload,
            "live": live_info,
        }
    )


if __name__ == "__main__":
    # Local dev server. On Render, use: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
