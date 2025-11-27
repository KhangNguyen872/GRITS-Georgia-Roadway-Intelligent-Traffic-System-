from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency
    pa = None
    pq = None

try:
    import resource
except Exception:  # pragma: no cover - platform specific
    resource = None


DEFAULT_NPMRDS_PATH = Path("data/npmrds_subset.parquet")
DEFAULT_INCIDENTS_CSV = Path("data/incidents.csv")


def _memory_gb() -> Optional[float]:
    """Best-effort resident set size in gigabytes."""
    rss_bytes: Optional[int] = None
    if psutil is not None:
        try:
            rss_bytes = psutil.Process(os.getpid()).memory_info().rss
        except Exception:
            rss_bytes = None
    if rss_bytes is None and resource is not None:
        try:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                rss_bytes = int(rss)
            else:
                rss_bytes = int(rss) * 1024
        except Exception:
            rss_bytes = None
    if rss_bytes is None:
        return None
    return rss_bytes / (1024**3)


def _log_message(message: str, *, extra: Optional[str] = None) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    suffix_parts = []
    if extra:
        suffix_parts.append(extra)
    mem_gb = _memory_gb()
    if mem_gb is not None:
        suffix_parts.append(f"rss={mem_gb:.2f} GB")
    suffix = f" | {' | '.join(suffix_parts)}" if suffix_parts else ""
    print(f"[{timestamp}] {message}{suffix}", flush=True)


@contextlib.contextmanager
def progress_step(name: str):
    """Context manager that logs start/end with duration and optional row counts."""
    start = time.perf_counter()
    _log_message(f"{name} - start")
    payload: dict[str, object] = {}
    try:
        yield payload
    finally:
        duration = time.perf_counter() - start
        extras: list[str] = [f"elapsed={duration:.1f}s"]
        rows = payload.get("rows")
        if rows is not None:
            try:
                extras.append(f"rows={int(rows):,}")
            except Exception:
                extras.append(f"rows={rows}")
        if "extra" in payload and payload["extra"]:
            extras.append(str(payload["extra"]))
        _log_message(f"{name} - done", extra=" | ".join(extras))


def _log_environment() -> None:
    """Emit runtime environment information."""
    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "pandas": getattr(pd, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
        "pyarrow": getattr(pa, "__version__", "missing"),
    }
    _log_message(
        "Environment",
        extra=", ".join(f"{k}={v}" for k, v in versions.items()),
    )


def _safe_collect_garbage() -> None:
    try:
        gc.collect()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Configuration


@dataclass(slots=True)
class PipelineConfig:
    """Configuration for the TFT data pipeline."""

    npmrds_path: Path = DEFAULT_NPMRDS_PATH
    incidents_csv: Optional[Path] = DEFAULT_INCIDENTS_CSV
    freq: str = "5min"
    horizons: Sequence[int] = (5, 15, 30, 60)
    seed: int = 42
    output_path: Optional[Path] = None
    meta_path: Optional[Path] = None
    min_confidence: float = 0.0


GA511_USECOLS = [
    "raw_json",
    "roadName",
    "direction",
    "startTime",
    "endTime",
    "severity",
    "type",
    "lanesBlocked",
    "id",
    "snapshot_utc",
]

GA511_STRING_COLS = [
    "raw_json",
    "roadName",
    "direction",
    "severity",
    "type",
    "lanesBlocked",
    "id",
    "snapshot_utc",
]

NWS_USECOLS = ["startTime", "endTime", "temperature", "windSpeed", "shortForecast"]


def _read_csv_with_optional_pyarrow(
    path: Path,
    *,
    usecols: Sequence[str],
    dtype_map: Optional[dict[str, str]] = None,
) -> tuple[pd.DataFrame, str]:
    """Read CSV using pyarrow engine when available, falling back to pandas."""
    engine_used = "pandas"
    if pa is not None:
        try:
            df = pd.read_csv(path, usecols=usecols, engine="pyarrow", dtype_backend="pyarrow")
            return df, "pyarrow"
        except Exception as exc:
            _log_message(
                f"pyarrow CSV engine unavailable for {path.name}, falling back",
                extra=str(exc),
            )
    read_kwargs = {"usecols": usecols}
    if dtype_map:
        read_kwargs["dtype"] = dtype_map
    df = pd.read_csv(path, **read_kwargs)
    return df, engine_used


# ---------------------------------------------------------------------------
# Utility helpers


def normalize_corridor(name: str | float | None) -> Optional[str]:
    """Normalise corridor identifiers (e.g., ``I-85`` or ``SR 1``)."""
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return None
    s = str(name).strip().upper()
    if not s:
        return None
    s = re.sub(r"\s+", " ", s)
    # Ensure expected prefixes keep their separators.
    match = re.match(r"^(I)[-\s]?(\d+)", s)
    if match:
        return f"I-{match.group(2)}"
    match = re.match(r"^(SR)[-\s]?(\d+)", s)
    if match:
        return f"SR {match.group(2)}"
    match = re.match(r"^(US)[-\s]?(\d+)", s)
    if match:
        return f"US-{match.group(2)}"
    match = re.match(r"^(GA)[-\s]?(\d+)", s)
    if match:
        return f"GA-{match.group(2)}"
    return s


def _parse_time(value: object) -> Optional[pd.Timestamp]:
    """Parse unix seconds or ISO strings to UTC timestamps."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(value)
        return ts.tz_convert(timezone.utc) if ts.tzinfo else ts.tz_localize(timezone.utc)
    if isinstance(value, (int, np.integer)):
        return pd.to_datetime(int(value), unit="s", utc=True)
    if isinstance(value, (float, np.floating)):
        return pd.to_datetime(int(value), unit="s", utc=True)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return pd.to_datetime(int(text), unit="s", utc=True)
    try:
        return pd.to_datetime(text, utc=True)
    except Exception:
        return None


def _extract_direction(raw: object) -> list[str]:
    """Return a list of canonical directions for an event."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return ["undirected"]
    s = str(raw).strip().lower()
    if not s:
        return ["undirected"]
    s = s.replace("/", " ").replace("\\", " ")
    tokens = {tok for tok in re.split(r"[\s,;]+", s) if tok}
    if "both" in tokens or "all" in tokens or "bothdirections" in tokens:
        return ["n", "s"]
    mapping = {
        "n": "n",
        "nb": "n",
        "north": "n",
        "northbound": "n",
        "s": "s",
        "sb": "s",
        "south": "s",
        "southbound": "s",
        "e": "e",
        "eb": "e",
        "east": "e",
        "eastbound": "e",
        "w": "w",
        "wb": "w",
        "west": "w",
        "westbound": "w",
    }
    dirs = {mapping[tok] for tok in tokens if tok in mapping}
    return sorted(dirs) if dirs else ["undirected"]


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    s = str(value).strip().lower()
    if not s:
        return False
    return s in {"true", "1", "yes", "y"}


LANE_PATTERN = re.compile(r"(\d+)")


def parse_lanes_blocked(text: object) -> int:
    """Extract the number of blocked lanes from the GA-511 description."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return 0
    s = str(text).lower()
    if "all lanes" in s:
        return 99
    match = LANE_PATTERN.search(s)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    # Fallback: count lane keywords.
    count = s.count("lane")
    return max(1, count) if "lane" in s else 0


def parse_wind_speed(text: object) -> float:
    """Return mean mph from strings like ``'5 mph'`` or ``'5 to 10 mph'``."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return float("nan")
    s = str(text).lower()
    numbers = [float(m.group()) for m in re.finditer(r"\d+(\.\d+)?", s)]
    if not numbers:
        return float("nan")
    return float(np.mean(numbers))


def precip_bucket(short_forecast: object) -> str:
    """Map NWS short forecast text to precipitation buckets."""
    if short_forecast is None or (isinstance(short_forecast, float) and math.isnan(short_forecast)):
        return "none"
    s = str(short_forecast).strip().lower()
    if not s:
        return "none"
    if any(k in s for k in ["thunder", "t-storm", "tstorm", "storm"]):
        return "tstorm"
    if "snow" in s or "sleet" in s or "flurries" in s:
        return "snow"
    if "fog" in s or "mist" in s or "haze" in s:
        return "fog"
    if "heavy" in s and ("rain" in s or "showers" in s):
        return "heavy_rain"
    if "rain" in s or "showers" in s or "drizzle" in s:
        return "light_rain"
    if any(k in s for k in ["sunny", "clear"]):
        return "none"
    return "other"


# ---------------------------------------------------------------------------
# NPMRDS + incident loaders


def _load_npmrds_subset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"NPMRDS subset not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    required = {"tstamp", "speed", "reference_speed", "confidence", "tmc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"NPMRDS subset missing columns: {','.join(sorted(missing))}")
    frame = df.copy(deep=False)
    frame["ts_utc"] = pd.to_datetime(frame["tstamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["ts_utc"])
    frame["tmc"] = frame["tmc"].astype(str)
    frame["speed"] = pd.to_numeric(frame["speed"], errors="coerce").astype("float32")
    frame["reference_speed"] = (
        pd.to_numeric(frame["reference_speed"], errors="coerce").astype("float32")
    )
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce").astype("float32")
    frame = frame.dropna(subset=["speed", "reference_speed", "confidence"])
    frame = frame.sort_values(["tmc", "ts_utc"]).reset_index(drop=True)
    return frame[["ts_utc", "tmc", "speed", "reference_speed", "confidence"]]


def _load_incident_archive(path: Path, freq: str) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(
            columns=[
                "ts_utc",
                "tmc",
                "incident_active",
                "lanes_blocked_count",
                "full_closure",
                "incident_severity_bucket",
                "incident_type_other",
            ]
        )
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    def pick(col_opts: Sequence[str], default: Optional[object] = None) -> pd.Series:
        for c in col_opts:
            if c in df.columns:
                return df[c]
        return pd.Series([default] * len(df))

    start_series = pick(["start_time", "startTime"])
    end_series = pick(["end_time", "endTime"])
    tmc_series = pick(["tmc", "segment_id"])
    incident_type = pick(["incident_type", "type", "event_type"], default="other")
    lanes_series = pick(["lanes_blocked_count", "lanes_blocked"], default=0)
    full_closure = pick(["full_closure", "is_full_closure"], default=False)
    severity = pick(["incident_severity_bucket", "severity"], default=1)

    rows: list[dict] = []
    for start_raw, end_raw, tmc_raw, typ_raw, lanes_raw, full_raw, sev_raw in zip(
        start_series, end_series, tmc_series, incident_type, lanes_series, full_closure, severity
    ):
        start_ts = _parse_time(start_raw)
        if start_ts is None:
            continue
        end_ts = _parse_time(end_raw) or (start_ts + pd.Timedelta(minutes=30))
        tmc = str(tmc_raw) if _coerce_nullable(tmc_raw) is not None else None
        if tmc is None:
            continue
        typ = str(typ_raw or "other").strip().lower() or "other"
        try:
            lanes_val = int(lanes_raw or 0)
        except Exception:
            lanes_val = 0
        try:
            sev_val = int(sev_raw or 1)
        except Exception:
            sev_val = 1
        is_full = _parse_bool(full_raw)
        rows.append(
            {
                "tmc": tmc,
                "start_time": start_ts,
                "end_time": end_ts,
                "incident_type": typ,
                "lanes_blocked_count": lanes_val,
                "full_closure": int(is_full),
                "incident_severity_bucket": sev_val,
            }
        )

    incidents = pd.DataFrame(rows)
    if incidents.empty:
        return pd.DataFrame(
            columns=[
                "ts_utc",
                "tmc",
                "incident_active",
                "lanes_blocked_count",
                "full_closure",
                "incident_severity_bucket",
                "incident_type_other",
            ]
        )
    incidents["start_time"] = pd.to_datetime(incidents["start_time"], utc=True)
    incidents["end_time"] = pd.to_datetime(incidents["end_time"], utc=True)
    incidents["incident_type"] = incidents["incident_type"].astype("category")
    incidents["tmc"] = incidents["tmc"].astype(str)
    return _expand_incidents_tmc(incidents, freq)


def _coerce_nullable(value: object) -> Optional[object]:
    if value is None:
        return None
    if value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        return None
    return value


def load_ga511_events(path: Path, corridor_filter: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Load and normalise GA-511 event rows with optional corridor filtering."""
    if not path.exists():
        raise FileNotFoundError(f"GA-511 CSV not found: {path}")
    header_cols = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [col for col in GA511_USECOLS if col in header_cols]
    missing_cols = sorted(set(GA511_USECOLS) - set(usecols))
    if missing_cols:
        _log_message("GA-511 missing columns", extra=",".join(missing_cols))
    dtype_map = {col: "string" for col in GA511_STRING_COLS if col in usecols}
    with progress_step("GA-511 read_csv") as info:
        raw, engine = _read_csv_with_optional_pyarrow(path, usecols=usecols, dtype_map=dtype_map)
        info["rows"] = len(raw)
        info["extra"] = f"engine={engine}"
    corridor_set = {normalize_corridor(c) for c in corridor_filter} if corridor_filter else None
    rows: list[dict] = []
    with progress_step("GA-511 normalise events") as info:
        for row in raw.itertuples(index=False, name=None):
            record = dict(zip(GA511_USECOLS, row))
            payload: dict = {}
            raw_json = _coerce_nullable(record.get("raw_json"))
            if isinstance(raw_json, (bytes, bytearray)):
                raw_json = raw_json.decode("utf-8", errors="ignore")
            if isinstance(raw_json, str) and raw_json.strip():
                try:
                    payload = json.loads(raw_json)
                except json.JSONDecodeError:
                    try:
                        payload = json.loads(raw_json.replace("''", '"').replace("'", '"'))
                    except Exception:
                        payload = {}
            corridor = normalize_corridor(
                payload.get("RoadwayName") or _coerce_nullable(record.get("roadName"))
            )
            if corridor is None:
                continue
            if corridor_set is not None and corridor not in corridor_set:
                continue
            directions = _extract_direction(
                payload.get("DirectionOfTravel") or _coerce_nullable(record.get("direction"))
            )
            start_time = (
                _parse_time(_coerce_nullable(record.get("startTime")))
                or _parse_time(payload.get("StartDate"))
                or _parse_time(payload.get("startTime"))
            )
            end_time = (
                _parse_time(_coerce_nullable(record.get("endTime")))
                or _parse_time(payload.get("PlannedEndDate"))
                or _parse_time(payload.get("endTime"))
            )
            if start_time is None:
                continue
            if end_time is None or end_time < start_time:
                end_time = start_time + pd.Timedelta(hours=2)
            severity = payload.get("Severity") or _coerce_nullable(record.get("severity"))
            try:
                severity_bucket = int(severity)
            except (TypeError, ValueError):
                severity_bucket = 1
            event_type = str(
                payload.get("EventType") or _coerce_nullable(record.get("type")) or ""
            ).strip().lower()
            lanes = parse_lanes_blocked(
                payload.get("LanesAffected") or _coerce_nullable(record.get("lanesBlocked"))
            )
            lanes = int(lanes or 0)
            is_full = _parse_bool(payload.get("IsFullClosure")) or (
                "all lanes closed"
                in str(_coerce_nullable(record.get("lanesBlocked")) or "").lower()
            )
            snapshot = _parse_time(_coerce_nullable(record.get("snapshot_utc")))
            snapshot_str = snapshot.isoformat() if snapshot is not None else None
            event_id = (
                _coerce_nullable(record.get("id")) or payload.get("ID") or payload.get("id")
            )

            for direction in directions:
                rows.append(
                    {
                        "event_id": event_id,
                        "corridor_id": corridor,
                        "direction": direction,
                        "start_time": start_time,
                        "end_time": end_time,
                        "event_type": event_type,
                        "lanes_blocked_count": lanes,
                        "full_closure": bool(is_full),
                        "incident_severity_bucket": int(severity_bucket),
                        "snapshot_utc": snapshot_str,
                    }
                )
        events = pd.DataFrame(rows)
        info["rows"] = len(events)
        if corridor_set is not None:
            info["extra"] = f"corridors={len(corridor_set)}"
    del raw
    _safe_collect_garbage()
    if events.empty:
        events = pd.DataFrame(
            columns=[
                "event_id",
                "corridor_id",
                "direction",
                "start_time",
                "end_time",
                "event_type",
                "lanes_blocked_count",
                "full_closure",
                "incident_severity_bucket",
                "snapshot_utc",
            ]
        )
        return events
    events["start_time"] = pd.to_datetime(events["start_time"], utc=True)
    events["end_time"] = pd.to_datetime(events["end_time"], utc=True)
    events = events.sort_values(["corridor_id", "direction", "start_time"]).reset_index(drop=True)
    events["corridor_id"] = events["corridor_id"].astype("category")
    events["direction"] = events["direction"].astype("category")
    events["event_type"] = events["event_type"].astype("category")
    events["full_closure"] = events["full_closure"].astype(bool)
    events["lanes_blocked_count"] = events["lanes_blocked_count"].astype("int16")
    events["incident_severity_bucket"] = events["incident_severity_bucket"].astype("int8")
    return events


def _process_nws_dataframe(
    df: pd.DataFrame, time_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]]
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["startTime", "endTime", "temp_f", "wind_mph", "precip_bucket"])
    frame = df.copy(deep=False)
    frame["startTime"] = pd.to_datetime(frame["startTime"], utc=True, errors="coerce")
    frame["endTime"] = pd.to_datetime(frame["endTime"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["startTime"]).sort_values("startTime")
    if time_window is not None:
        lower = time_window[0] - pd.Timedelta(hours=24)
        upper = time_window[1] + pd.Timedelta(hours=24)
        frame = frame[
            (frame["endTime"].fillna(frame["startTime"]) >= lower)
            & (frame["startTime"] <= upper)
        ]
    if frame.empty:
        return pd.DataFrame(columns=["startTime", "endTime", "temp_f", "wind_mph", "precip_bucket"])
    frame["temp_f"] = pd.to_numeric(frame.get("temperature"), errors="coerce").astype("float32")
    frame["wind_mph"] = frame.get("windSpeed", "").map(parse_wind_speed).astype("float32")
    frame["precip_bucket"] = frame.get("shortForecast", "").map(precip_bucket).astype("category")
    result = frame[["startTime", "endTime", "temp_f", "wind_mph", "precip_bucket"]].reset_index(
        drop=True
    )
    return result


def _load_nws_hourly_chunked(
    path: Path,
    time_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
    chunksize: int,
) -> pd.DataFrame:
    chunksize = max(chunksize, 1)
    read_kwargs = {
        "filepath_or_buffer": path,
        "usecols": NWS_USECOLS,
        "chunksize": chunksize,
        "dtype": {"temperature": "float32", "windSpeed": "string", "shortForecast": "string"},
    }

    if pa is None or pq is None:
        frames: list[pd.DataFrame] = []
        total_rows = 0
        with progress_step("NWS chunked read (fallback)") as info:
            reader = pd.read_csv(**read_kwargs)
            for idx, chunk in enumerate(reader):
                processed = _process_nws_dataframe(chunk, time_window)
                rows = len(processed)
                total_rows += rows
                if rows:
                    frames.append(processed)
                    _log_message("NWS chunk processed", extra=f"chunk={idx} rows={rows:,}")
                del chunk
                _safe_collect_garbage()
            info["rows"] = total_rows
        if not frames:
            return pd.DataFrame(columns=["startTime", "endTime", "temp_f", "wind_mph", "precip_bucket"])
        df = pd.concat(frames, ignore_index=True)
        df["precip_bucket"] = df["precip_bucket"].astype("category")
        return df

    tmp_dir = Path(tempfile.mkdtemp(prefix="nws_chunks_"))
    tmp_path = tmp_dir / "nws_hourly_compacted.parquet"
    total_rows = 0
    writer: Optional[pq.ParquetWriter] = None
    try:
        with progress_step("NWS chunked read") as info:
            reader = pd.read_csv(**read_kwargs)
            for idx, chunk in enumerate(reader):
                processed = _process_nws_dataframe(chunk, time_window)
                rows = len(processed)
                total_rows += rows
                if rows == 0:
                    continue
                table = pa.Table.from_pandas(processed, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp_path, table.schema)
                writer.write_table(table)
                _log_message("NWS chunk processed", extra=f"chunk={idx} rows={rows:,}")
                del chunk, processed, table
                _safe_collect_garbage()
            info["rows"] = total_rows
    finally:
        if writer is not None:
            writer.close()
    if not tmp_path.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return pd.DataFrame(columns=["startTime", "endTime", "temp_f", "wind_mph", "precip_bucket"])
    with progress_step("NWS reload compacted parquet") as info:
        df = pd.read_parquet(tmp_path)
        if not df.empty:
            df["temp_f"] = df["temp_f"].astype("float32")
            df["wind_mph"] = df["wind_mph"].astype("float32")
            df["precip_bucket"] = df["precip_bucket"].astype("category")
        info["rows"] = len(df)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return df


def load_nws_hourly(
    path: Path,
    time_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
    chunksize: int,
) -> pd.DataFrame:
    """Load NWS hourly forecasts and map to simpler features."""
    if not path.exists():
        raise FileNotFoundError(f"NWS hourly CSV not found: {path}")
    if pa is not None:
        try:
            with progress_step("NWS read_csv (pyarrow)") as info:
                df = pd.read_csv(
                    path,
                    usecols=NWS_USECOLS,
                    engine="pyarrow",
                    dtype_backend="pyarrow",
                )
                info["rows"] = len(df)
            with progress_step("NWS transform (pyarrow)") as info:
                processed = _process_nws_dataframe(df, time_window)
                info["rows"] = len(processed)
            del df
            _safe_collect_garbage()
            return processed
        except Exception as exc:
            _log_message(
                "NWS pyarrow read failed; falling back to chunked mode",
                extra=str(exc),
            )
    return _load_nws_hourly_chunked(path, time_window, chunksize)


# ---------------------------------------------------------------------------
# Feature engineering


def _expand_events(events: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Expand GA-511 events to the requested time grid."""
    if events.empty:
        return pd.DataFrame(
            columns=[
                "ts_utc",
                "corridor_id",
                "direction",
                "incident_active",
                "full_closure",
                "lanes_blocked_count",
                "incident_severity_bucket",
                "event_type_roadwork",
                "event_type_closures",
            ]
        )
    expanded_rows: list[dict] = []
    freq_offset = pd.tseries.frequencies.to_offset(freq)
    for evt in events.itertuples(index=False):
        if pd.isna(evt.start_time) or pd.isna(evt.end_time):
            continue
        start = evt.start_time.floor(freq)
        # Event is active on [start, end)
        end = (evt.end_time - pd.Timedelta(seconds=1)).ceil(freq)
        if end < start:
            end = start + freq_offset
        timeline = pd.date_range(start, end, freq=freq, inclusive="left")
        if len(timeline) == 0:
            timeline = pd.DatetimeIndex([start])
        for ts in timeline:
            expanded_rows.append(
                {
                    "ts_utc": ts,
                    "corridor_id": evt.corridor_id,
                    "direction": evt.direction,
                    "incident_active": 1,
                    "full_closure": int(evt.full_closure),
                    "lanes_blocked_count": evt.lanes_blocked_count,
                    "incident_severity_bucket": evt.incident_severity_bucket,
                    "event_type_roadwork": 1 if evt.event_type == "roadwork" else 0,
                    "event_type_closures": 1 if evt.event_type == "closures" else 0,
                }
            )
    exp = pd.DataFrame(expanded_rows)
    if exp.empty:
        return exp
    grouped = (
        exp.groupby(["ts_utc", "corridor_id", "direction"], as_index=False)
        .agg(
            {
                "incident_active": "max",
                "full_closure": "max",
                "lanes_blocked_count": "max",
                "incident_severity_bucket": "max",
                "event_type_roadwork": "max",
                "event_type_closures": "max",
            }
        )
        .sort_values(["corridor_id", "direction", "ts_utc"])
    )
    return grouped


def _build_weather_matrix(weather: pd.DataFrame, timestamps: pd.DatetimeIndex, freq: str) -> pd.DataFrame:
    """Reindex weather data to the 5-minute grid."""
    if weather.empty:
        return pd.DataFrame(
            {
                "ts_utc": timestamps,
                "temp_f": np.full(len(timestamps), np.nan, dtype=np.float32),
                "wind_mph": np.full(len(timestamps), np.nan, dtype=np.float32),
                "precip_bucket": pd.Categorical(["none"] * len(timestamps)),
            }
        )
    df = weather.copy()
    df = df.set_index("startTime").sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    # Forward fill across the desired timeline.
    weather_resampled = df.reindex(timestamps, method="ffill")
    weather_resampled = weather_resampled.reset_index().rename(columns={"index": "ts_utc"})
    weather_resampled["ts_utc"] = pd.to_datetime(weather_resampled["ts_utc"], utc=True)
    weather_resampled["precip_bucket"] = weather_resampled["precip_bucket"].fillna("none")
    weather_resampled["temp_f"] = weather_resampled["temp_f"].astype("float32")
    weather_resampled["wind_mph"] = weather_resampled["wind_mph"].astype("float32")
    weather_resampled["precip_bucket"] = weather_resampled["precip_bucket"].astype("category")
    return weather_resampled[["ts_utc", "temp_f", "wind_mph", "precip_bucket"]]


def _expand_incidents_tmc(incidents: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Expand incidents keyed by TMC onto the time grid."""
    if incidents.empty:
        return pd.DataFrame(
            columns=[
                "ts_utc",
                "tmc",
                "incident_active",
                "lanes_blocked_count",
                "full_closure",
                "incident_severity_bucket",
                "incident_type",
            ]
        )
    expanded_rows: list[dict] = []
    freq_offset = pd.tseries.frequencies.to_offset(freq)
    for evt in incidents.itertuples(index=False):
        if pd.isna(evt.start_time) or pd.isna(evt.end_time):
            continue
        start = pd.Timestamp(evt.start_time).floor(freq)
        end = (pd.Timestamp(evt.end_time) - pd.Timedelta(seconds=1)).ceil(freq)
        if end < start:
            end = start + freq_offset
        timeline = pd.date_range(start, end, freq=freq, inclusive="left")
        if len(timeline) == 0:
            timeline = pd.DatetimeIndex([start])
        for ts in timeline:
            expanded_rows.append(
                {
                    "ts_utc": ts,
                    "tmc": evt.tmc,
                    "incident_active": 1,
                    "lanes_blocked_count": int(getattr(evt, "lanes_blocked_count", 0) or 0),
                    "full_closure": int(getattr(evt, "full_closure", 0) or 0),
                    "incident_severity_bucket": int(
                        getattr(evt, "incident_severity_bucket", 1) or 1
                    ),
                    "incident_type": getattr(evt, "incident_type", "other"),
                }
            )
    exp = pd.DataFrame(expanded_rows)
    if exp.empty:
        return exp

    def _mode(series: pd.Series) -> str:
        if series.empty:
            return "other"
        return str(series.value_counts().idxmax())

    grouped = (
        exp.groupby(["ts_utc", "tmc"], as_index=False)
        .agg(
            {
                "incident_active": "max",
                "lanes_blocked_count": "max",
                "full_closure": "max",
                "incident_severity_bucket": "max",
                "incident_type": _mode,
            }
        )
        .sort_values(["tmc", "ts_utc"])
    )
    grouped["lanes_blocked_count"] = grouped["lanes_blocked_count"].astype("int16")
    grouped["full_closure"] = grouped["full_closure"].astype("int8")
    grouped["incident_active"] = grouped["incident_active"].astype("int8")
    grouped["incident_severity_bucket"] = grouped["incident_severity_bucket"].astype("int8")
    grouped["incident_type"] = grouped["incident_type"].astype("category")
    return grouped


def corridor_freeflow_mph(corridor_id: str) -> float:
    """Return base free-flow speeds by corridor classification."""
    cid = corridor_id.upper()
    if cid.startswith("I-"):
        return 70.0
    if cid.startswith("SR"):
        return 55.0
    return 45.0


WEATHER_MULTIPLIERS = {
    "none": 1.00,
    "light_rain": 0.95,
    "heavy_rain": 0.85,
    "tstorm": 0.85,
    "snow": 0.80,
    "fog": 0.90,
    "other": 0.98,
}

SEVERITY_MULTIPLIERS = {1: 0.98, 2: 0.92, 3: 0.85}


def _compute_weather_multiplier(frame: pd.DataFrame) -> np.ndarray:
    base = frame["precip_bucket"].map(WEATHER_MULTIPLIERS).fillna(0.98).astype("float32")
    base_arr = base.to_numpy(dtype=np.float32, copy=False)
    wind_arr = frame["wind_mph"].to_numpy(dtype=np.float32, copy=False)
    high_wind = wind_arr > 25
    high_wind &= ~np.isnan(wind_arr)
    base_arr = np.where(high_wind, base_arr * 0.95, base_arr)
    np.clip(base_arr, 0.05, None, out=base_arr)
    return base_arr


def _compute_incident_multiplier(frame: pd.DataFrame) -> np.ndarray:
    n = len(frame)
    mult = np.ones(n, dtype=np.float32)
    if n == 0:
        return mult
    incident_active = frame["incident_active"].to_numpy(dtype=np.int8, copy=False) > 0
    if not incident_active.any():
        return mult
    full_closure = frame["full_closure"].to_numpy(dtype=np.int8, copy=False) > 0
    lanes = frame["lanes_blocked_count"].to_numpy(dtype=np.int16, copy=False)
    closures = frame["event_type_closures"].to_numpy(dtype=np.int8, copy=False) > 0
    severity = frame["incident_severity_bucket"].to_numpy(dtype=np.int16, copy=False)

    full_mask = incident_active & full_closure
    mult[full_mask] = 0.15

    non_full_mask = incident_active & ~full_mask
    if non_full_mask.any():
        idx = np.where(non_full_mask)[0]
        lane_vals = lanes[idx]
        lane_mult = np.ones(len(idx), dtype=np.float32)
        lane_mult = np.where(lane_vals >= 3, lane_mult * 0.70, lane_mult)
        lane_mult = np.where(lane_vals == 2, lane_mult * 0.80, lane_mult)
        lane_mult = np.where(lane_vals == 1, lane_mult * 0.90, lane_mult)
        lane_mult = np.where(closures[idx], lane_mult * 0.75, lane_mult)
        mult[idx] *= lane_mult

    severity_mult = np.ones(n, dtype=np.float32)
    for bucket, value in SEVERITY_MULTIPLIERS.items():
        severity_mult[severity == bucket] = value
    mult[incident_active] *= severity_mult[incident_active]
    np.clip(mult, 0.05, None, out=mult)
    return mult


def _diurnal_multiplier(hour: int) -> float:
    am_peak = math.exp(-((hour - 8) ** 2) / 9.0)
    pm_peak = math.exp(-((hour - 17) ** 2) / 9.0)
    return float(max(0.2, 1.0 - 0.18 * am_peak - 0.22 * pm_peak))


def build_dataset(cfg: PipelineConfig) -> tuple[pd.DataFrame, dict]:
    """Build TFT-ready dataset from NPMRDS subset + incidents."""
    np.random.default_rng(cfg.seed)

    with progress_step("Load NPMRDS subset") as info:
        speeds = _load_npmrds_subset(cfg.npmrds_path)
        speeds = speeds[speeds["confidence"] >= float(cfg.min_confidence)].reset_index(drop=True)
        info["rows"] = len(speeds)

    if speeds.empty:
        raise ValueError("NPMRDS subset is empty after filtering.")

    offset = pd.tseries.frequencies.to_offset(cfg.freq)
    freq_minutes = int(pd.Timedelta(offset.nanos, unit="ns") / pd.Timedelta(minutes=1))
    if freq_minutes <= 0:
        raise ValueError(f"Unsupported frequency: {cfg.freq}")

    speeds["speed_ratio"] = np.where(
        speeds["reference_speed"] > 0,
        speeds["speed"] / speeds["reference_speed"],
        np.nan,
    ).astype("float32")
    speeds = speeds.dropna(subset=["speed_ratio"])

    with progress_step("Timeline & grid") as info:
        ts_start = speeds["ts_utc"].min().floor(cfg.freq)
        ts_end = speeds["ts_utc"].max().ceil(cfg.freq)
        timestamps = pd.date_range(ts_start, ts_end, freq=cfg.freq)
        tmcs = pd.DataFrame({"tmc": sorted(speeds["tmc"].unique())})
        base = pd.DataFrame({"ts_utc": timestamps, "key": 1})
        tmcs["key"] = 1
        grid = (
            base.merge(tmcs, on="key", how="outer")
            .drop(columns="key")
            .sort_values(["tmc", "ts_utc"])
            .reset_index(drop=True)
        )
        info["rows"] = len(grid)
        info["extra"] = f"{len(tmcs)} tmcs, {ts_start} → {ts_end}"

    with progress_step("Merge speeds") as info:
        grid = grid.merge(speeds, on=["tmc", "ts_utc"], how="left")
        info["rows"] = len(grid)

    with progress_step("Attach incidents") as info:
        incidents = _load_incident_archive(cfg.incidents_csv, cfg.freq) if cfg.incidents_csv else pd.DataFrame()
        incident_cols = [
            "incident_active",
            "lanes_blocked_count",
            "full_closure",
            "incident_severity_bucket",
            "incident_type",
        ]
        if incidents.empty:
            for col in incident_cols:
                grid[col] = 0 if col != "incident_type" else "none"
        else:
            grid = grid.merge(incidents, on=["tmc", "ts_utc"], how="left")
            for col in incident_cols:
                if col == "incident_type":
                    grid[col] = grid[col].fillna("none")
                else:
                    grid[col] = grid[col].fillna(0)
        grid["incident_type"] = grid["incident_type"].astype("category")
        grid["incident_active"] = grid["incident_active"].astype("int8")
        grid["lanes_blocked_count"] = grid["lanes_blocked_count"].astype("int16")
        grid["full_closure"] = grid["full_closure"].astype("int8")
        grid["incident_severity_bucket"] = grid["incident_severity_bucket"].astype("int8")
        info["rows"] = len(grid)

    with progress_step("Engineer temporal features") as info:
        grid["hour"] = grid["ts_utc"].dt.hour.astype("int8")
        grid["dow"] = grid["ts_utc"].dt.dayofweek.astype("int8")
        grid["is_weekend"] = grid["dow"].isin([5, 6]).astype("int8")
        grid["hour_sin"] = np.sin(2 * np.pi * grid["hour"] / 24.0).astype("float32")
        grid["hour_cos"] = np.cos(2 * np.pi * grid["hour"] / 24.0).astype("float32")
        info["rows"] = len(grid)

    horizons = sorted(set(int(h) for h in cfg.horizons))
    grid = grid.sort_values(["tmc", "ts_utc"]).reset_index(drop=True)

    with progress_step("Create forecast targets") as info:
        grouped = grid.groupby("tmc", observed=False)["speed_ratio"]
        for horizon in horizons:
            if horizon % freq_minutes != 0:
                raise ValueError(f"Horizon {horizon} is not aligned with {cfg.freq} frequency.")
            steps = int(horizon // freq_minutes)
            target_col = f"target_speed_ratio(+{horizon})"
            grid[target_col] = grouped.shift(-steps)
        info["rows"] = len(grid)

    target_columns = [f"target_speed_ratio(+{h})" for h in horizons]
    with progress_step("Prune rows with empty targets") as info:
        valid_targets = grid.dropna(subset=target_columns + ["speed_ratio"])
        info["rows"] = len(valid_targets)

    with progress_step("Assign time indices") as info:
        time_lookup = (
            valid_targets[["ts_utc"]].drop_duplicates().sort_values("ts_utc").reset_index(drop=True)
        )
        time_lookup["time_idx"] = np.arange(len(time_lookup), dtype=np.int32)
        data = valid_targets.merge(time_lookup, on="ts_utc", how="left")
        info["rows"] = len(data)

    ts_unique = data["ts_utc"].sort_values().unique()
    cutoff_index = max(0, int(len(ts_unique) * 0.8))
    cutoff_time = ts_unique[cutoff_index] if len(ts_unique) > 0 else None
    if cutoff_time is None:
        raise ValueError("Unable to determine validation cutoff.")
    split = np.where(data["ts_utc"] >= cutoff_time, "val", "train")
    data["split"] = pd.Categorical(split, categories=["train", "val"])

    config_jsonable = {
        key: (str(value) if isinstance(value, Path) else value) for key, value in asdict(cfg).items()
    }
    metadata = {
        "config": config_jsonable,
        "horizons": horizons,
        "cutoff_time": cutoff_time.isoformat(),
        "num_rows": int(len(data)),
        "num_tmcs": int(data["tmc"].nunique()),
        "freq_minutes": freq_minutes,
    }

    _safe_collect_garbage()

    if cfg.output_path:
        output_path = Path(cfg.output_path)
        with progress_step(f"Write dataset to {output_path}") as info:
            suffix = output_path.suffix.lower()
            if suffix in {".parquet", ".pq"}:
                if pa is None and pq is None:
                    raise RuntimeError(
                        "pyarrow or fastparquet is required to write parquet outputs. "
                        "Install one of them or use a CSV output path."
                    )
                _atomic_write_parquet(data, output_path)
            else:
                _atomic_write(output_path, lambda tmp: data.to_csv(tmp, index=False))
            info["rows"] = len(data)

    if cfg.meta_path:
        meta_path = Path(cfg.meta_path)
        with progress_step(f"Write metadata to {meta_path}") as info:
            _atomic_write_text(json.dumps(metadata, indent=2), meta_path)
            info["rows"] = len(data)

    return data, metadata


# ---------------------------------------------------------------------------
# CLI


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    def _writer(tmp: Path) -> None:
        df.to_parquet(tmp, index=False)

    _atomic_write(path, _writer)


def _atomic_write_text(payload: str, path: Path) -> None:
    def _writer(tmp: Path) -> None:
        tmp.write_text(payload)

    _atomic_write(path, _writer)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TFT training dataset.")
    parser.add_argument("--npmrds-path", type=Path, default=PipelineConfig.npmrds_path)
    parser.add_argument("--incidents-csv", type=Path, default=PipelineConfig.incidents_csv)
    parser.add_argument(
        "--horizons",
        type=str,
        default="5,15,30,60",
        help="Comma-separated list of forecast horizons in minutes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freq",
        type=str,
        default="5min",
        help="Time-step frequency for the grid (e.g., '5min', '15min').",
    )
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum probe confidence filter.")
    parser.add_argument("--output", type=Path, default=Path("tft/artifacts/dataset.parquet"))
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("tft/artifacts/dataset_meta.json"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    _log_environment()
    try:
        args = parse_args(argv)
        horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
        cfg = PipelineConfig(
            npmrds_path=args.npmrds_path,
            incidents_csv=args.incidents_csv,
            freq=args.freq,
            horizons=horizons,
            seed=args.seed,
            output_path=args.output,
            meta_path=args.meta,
            min_confidence=args.min_confidence,
        )
        _log_message(
            "Pipeline configuration",
            extra=f"freq={cfg.freq}, tmcs=from subset, min_conf={cfg.min_confidence}",
        )
        data, metadata = build_dataset(cfg)
        if cfg.output_path:
            _log_message(
                "Dataset materialised",
                extra=f"path={cfg.output_path} rows={len(data):,}",
            )
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    else:
        print(json.dumps(metadata, indent=2), flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
