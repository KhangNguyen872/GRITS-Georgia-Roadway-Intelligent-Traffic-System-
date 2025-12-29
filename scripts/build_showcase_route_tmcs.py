from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # shapely is optional; used when available for fast spatial indexing
    from shapely.geometry import LineString, Point  # type: ignore
    from shapely.strtree import STRtree  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LineString = None
    Point = None
    STRtree = None

EARTH_RADIUS_M = 6_371_000.0
SINGLETON_MILE_THRESHOLD = 0.1  # drop singletons shorter than this unless repeated


@dataclass
class TmcSegment:
    tmc: str
    road: str
    direction: str
    miles: float
    road_order: Optional[float]
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def detect_delimiter(path: Path) -> str:
    sample = ""
    with path.open("r", errors="ignore") as f:
        sample = f.read(2048)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        logging.debug("Detected delimiter %r via csv.Sniffer", dialect.delimiter)
        return dialect.delimiter
    except Exception:
        first_line = sample.splitlines()[0] if sample else ""
        tab_count = first_line.count("\t")
        comma_count = first_line.count(",")
        delim = "\t" if tab_count > comma_count else ","
        logging.debug("Delimiter sniff failed; using heuristic delimiter %r", delim)
        return delim


def _coerce_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_tmc_identification(path: Path) -> tuple[list[TmcSegment], int]:
    if not path.exists():
        raise FileNotFoundError(f"TMC identification file not found: {path}")
    delimiter = detect_delimiter(path)
    df = pd.read_csv(path, delimiter=delimiter)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    lower_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=lower_map)

    required_cols = [
        "tmc",
        "start_latitude",
        "start_longitude",
        "end_latitude",
        "end_longitude",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in TMC identification: {missing}")

    df["miles"] = _coerce_float(df.get("miles", np.nan))
    df["road_order"] = _coerce_float(df.get("road_order", np.nan))
    for col in ["start_latitude", "start_longitude", "end_latitude", "end_longitude"]:
        df[col] = _coerce_float(df[col])

    df = df.dropna(
        subset=["tmc", "start_latitude", "start_longitude", "end_latitude", "end_longitude"]
    )
    df["tmc"] = df["tmc"].astype(str)
    df["road"] = df.get("road", "").fillna("").astype(str)
    df["direction"] = df.get("direction", "").fillna("").astype(str)

    segments: list[TmcSegment] = []
    for row in df.itertuples(index=False):
        segments.append(
            TmcSegment(
                tmc=str(getattr(row, "tmc")),
                road=str(getattr(row, "road", "")),
                direction=str(getattr(row, "direction", "")),
                miles=float(getattr(row, "miles", np.nan))
                if not math.isnan(getattr(row, "miles", np.nan))
                else 0.0,
                road_order=float(getattr(row, "road_order", np.nan))
                if not math.isnan(getattr(row, "road_order", np.nan))
                else None,
                start_lat=float(getattr(row, "start_latitude")),
                start_lon=float(getattr(row, "start_longitude")),
                end_lat=float(getattr(row, "end_latitude")),
                end_lon=float(getattr(row, "end_longitude")),
            )
        )

    total_tmcs = df["tmc"].nunique()
    return segments, total_tmcs


def decode_polyline(polyline_str: str, precision: int = 5) -> list[tuple[float, float]]:
    """Decode an encoded polyline string into (lat, lon) tuples."""
    coordinates: list[tuple[float, float]] = []
    index = 0
    lat = 0
    lon = 0
    factor = 10**precision
    length = len(polyline_str)

    while index < length:
        for coord_idx in (0, 1):
            result = 0
            shift = 0
            while True:
                if index >= length:
                    break
                b = ord(polyline_str[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if coord_idx == 0:
                lat += delta
            else:
                lon += delta
        coordinates.append((lat / factor, lon / factor))

    return coordinates


def _extract_linestring_coords(geojson_obj: dict) -> list[tuple[float, float]]:
    def _coords_from_geom(geom: dict) -> Optional[list[tuple[float, float]]]:
        gtype = geom.get("type", "").lower()
        coords = geom.get("coordinates")
        if gtype == "linestring":
            return [(c[1], c[0]) for c in coords]
        if gtype == "multilinestring" and coords:
            flat = coords[0]
            return [(c[1], c[0]) for c in flat]
        return None

    if "type" not in geojson_obj:
        raise ValueError("Invalid GeoJSON: missing type field.")

    gtype = geojson_obj["type"].lower()
    if gtype == "featurecollection":
        for feat in geojson_obj.get("features", []):
            geom = feat.get("geometry")
            if geom:
                coords = _coords_from_geom(geom)
                if coords:
                    return coords
        raise ValueError("No LineString geometry found in FeatureCollection.")
    if gtype == "feature":
        geom = geojson_obj.get("geometry", {})
        coords = _coords_from_geom(geom)
        if coords:
            return coords
        raise ValueError("Feature does not contain a LineString geometry.")
    if gtype in {"linestring", "multilinestring"}:
        coords = _coords_from_geom(geojson_obj)
        if coords:
            return coords
    raise ValueError(f"Unsupported GeoJSON type for route: {geojson_obj.get('type')}")


def load_route_points(path: Path) -> list[tuple[float, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Route path not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = path.read_text().strip()
        if not text:
            raise ValueError("Route polyline text file is empty.")
        return decode_polyline(text)

    if suffix in {".geojson", ".json"}:
        data = json.loads(path.read_text())
        if suffix == ".json" and isinstance(data, list):
            return [(float(pt[0]), float(pt[1])) for pt in data]
        if suffix == ".json" and isinstance(data, dict) and "points" in data:
            return [(float(pt[0]), float(pt[1])) for pt in data["points"]]
        return _extract_linestring_coords(data)

    # Fallback: try to sniff by content
    text = path.read_text().strip()
    if text.startswith("{"):
        data = json.loads(text)
        return _extract_linestring_coords(data)
    return decode_polyline(text)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_M * c


def densify_route(points: list[tuple[float, float]], sample_m: float) -> list[tuple[float, float]]:
    if len(points) < 2:
        raise ValueError("Route must contain at least two points.")
    cumulative = [0.0]
    for i in range(1, len(points)):
        dist = haversine_m(points[i - 1][0], points[i - 1][1], points[i][0], points[i][1])
        cumulative.append(cumulative[-1] + dist)

    total_length = cumulative[-1]
    if total_length == 0:
        return points

    targets = [0.0]
    d = sample_m
    while d < total_length:
        targets.append(d)
        d += sample_m
    targets.append(total_length)

    sampled: list[tuple[float, float]] = []
    seg_idx = 0
    for t in targets:
        while seg_idx < len(points) - 2 and t > cumulative[seg_idx + 1]:
            seg_idx += 1
        seg_len = cumulative[seg_idx + 1] - cumulative[seg_idx]
        if seg_len == 0:
            sampled.append(points[seg_idx])
            continue
        ratio = (t - cumulative[seg_idx]) / seg_len
        lat = points[seg_idx][0] + ratio * (points[seg_idx + 1][0] - points[seg_idx][0])
        lon = points[seg_idx][1] + ratio * (points[seg_idx + 1][1] - points[seg_idx][1])
        sampled.append((lat, lon))
    return sampled


def point_to_segments_distance_m(
    lat: float, lon: float, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    lat0 = math.radians(lat)
    lon0 = math.radians(lon)
    cos_lat = max(math.cos(lat0), 1e-8)

    lat1 = np.radians(starts[:, 0])
    lon1 = np.radians(starts[:, 1])
    lat2 = np.radians(ends[:, 0])
    lon2 = np.radians(ends[:, 1])

    x1 = (lon1 - lon0) * cos_lat * EARTH_RADIUS_M
    y1 = (lat1 - lat0) * EARTH_RADIUS_M
    x2 = (lon2 - lon0) * cos_lat * EARTH_RADIUS_M
    y2 = (lat2 - lat0) * EARTH_RADIUS_M

    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    denom = np.where(denom == 0.0, 1e-12, denom)
    t = np.clip((dx * (-x1) + dy * (-y1)) / denom, 0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.hypot(proj_x, proj_y)


def point_to_segment_distance_single(lat: float, lon: float, seg: TmcSegment) -> float:
    starts = np.array([[seg.start_lat, seg.start_lon]])
    ends = np.array([[seg.end_lat, seg.end_lon]])
    return float(point_to_segments_distance_m(lat, lon, starts, ends)[0])


class NearestTmcLocator:
    def __init__(self, segments: list[TmcSegment], threshold_m: float) -> None:
        self.segments = segments
        self.threshold_m = threshold_m
        self.use_shapely = bool(LineString and Point and STRtree)
        self._tree: Optional[STRtree] = None
        self._geom_id_to_idx: dict[int, int] = {}
        self._geom_wkb_to_idx: dict[bytes, int] = {}
        self._geoms: list = []

        # Always keep numpy arrays for fallback even when shapely is available.
        self.starts = np.array([[s.start_lat, s.start_lon] for s in segments])
        self.ends = np.array([[s.end_lat, s.end_lon] for s in segments])
        self.min_lat = np.minimum(self.starts[:, 0], self.ends[:, 0])
        self.max_lat = np.maximum(self.starts[:, 0], self.ends[:, 0])
        self.min_lon = np.minimum(self.starts[:, 1], self.ends[:, 1])
        self.max_lon = np.maximum(self.starts[:, 1], self.ends[:, 1])

        if self.use_shapely:
            try:
                self._geoms = [
                    LineString([(seg.start_lon, seg.start_lat), (seg.end_lon, seg.end_lat)])
                    for seg in segments
                ]
                self._tree = STRtree(self._geoms)
                self._geom_id_to_idx = {id(geom): idx for idx, geom in enumerate(self._geoms)}
                try:
                    self._geom_wkb_to_idx = {geom.wkb: idx for idx, geom in enumerate(self._geoms)}
                except Exception:
                    self._geom_wkb_to_idx = {}
                logging.info("Using shapely STRtree for nearest lookup.")
            except Exception as exc:  # pragma: no cover - defensive
                logging.warning("Shapely available but STRtree construction failed: %s", exc)
                self.use_shapely = False

        if not self.use_shapely:
            logging.info("Shapely not available; using numpy fallback for nearest lookup.")

    def nearest(self, lat: float, lon: float) -> tuple[Optional[int], float]:
        if not self.segments:
            return None, math.inf
        if self.use_shapely and self._tree is not None:
            pt = Point(lon, lat)
            geom = self._tree.nearest(pt)
            if geom is None:
                return None, math.inf
            idx = self._geom_id_to_idx.get(id(geom))
            if idx is None:
                if self._geom_wkb_to_idx:
                    wkb_val = getattr(geom, "wkb", b"")
                    if isinstance(wkb_val, memoryview):
                        wkb_val = wkb_val.tobytes()
                    idx = self._geom_wkb_to_idx.get(wkb_val)
                if idx is None and self._geoms:
                    try:
                        idx = self._geoms.index(geom)
                    except ValueError:
                        idx = None
            if idx is None:
                return self._nearest_numpy(lat, lon)
            dist = point_to_segment_distance_single(lat, lon, self.segments[idx])
            return idx, dist
        return self._nearest_numpy(lat, lon)

    def _nearest_numpy(self, lat: float, lon: float) -> tuple[Optional[int], float]:
        lat_rad = math.radians(lat)
        lat_buf = self.threshold_m / 111_000.0
        lon_buf = self.threshold_m / max(math.cos(lat_rad) * 111_000.0, 1e-6)

        mask = (
            (lat >= self.min_lat - lat_buf)
            & (lat <= self.max_lat + lat_buf)
            & (lon >= self.min_lon - lon_buf)
            & (lon <= self.max_lon + lon_buf)
        )
        if not mask.any():
            return None, math.inf
        candidates = np.where(mask)[0]
        dists = point_to_segments_distance_m(lat, lon, self.starts[candidates], self.ends[candidates])
        min_idx = int(candidates[int(dists.argmin())])
        min_dist = float(dists.min())
        return min_idx, min_dist


def parse_expected_set(raw: Optional[str]) -> Optional[set[str]]:
    if not raw:
        return None
    values = {part.strip().upper() for part in raw.split(",") if part.strip()}
    return values or None


def normalize_road(road: str) -> str:
    return road.strip().upper()


def compress_runs(sequence: Iterable[Optional[str]]) -> list[tuple[str, int]]:
    runs: list[tuple[str, int]] = []
    last = None
    count = 0
    for tmc in sequence:
        if tmc is None:
            continue
        if tmc != last and last is not None:
            runs.append((last, count))
            count = 0
        last = tmc
        count += 1
    if last is not None and count > 0:
        runs.append((last, count))
    return runs


def denoise_tmcs(
    tmc_sequence: list[Optional[str]],
    tmc_lookup: dict[str, TmcSegment],
) -> list[str]:
    counts = Counter([t for t in tmc_sequence if t])
    runs = compress_runs(tmc_sequence)

    filtered: list[str] = []
    for tmc, run_len in runs:
        seg = tmc_lookup[tmc]
        if run_len == 1 and counts[tmc] == 1 and seg.miles < SINGLETON_MILE_THRESHOLD:
            logging.debug("Dropping singleton TMC %s (run_len=1, miles=%.3f)", tmc, seg.miles)
            continue
        filtered.append(tmc)

    ordered_unique: list[str] = []
    seen: set[str] = set()
    for tmc in filtered:
        if tmc not in seen:
            ordered_unique.append(tmc)
            seen.add(tmc)
    return ordered_unique


def match_route_to_tmcs(
    sampled_points: list[tuple[float, float]],
    segments: list[TmcSegment],
    threshold_m: float,
    expected_dirs: Optional[set[str]],
    expected_roads: Optional[set[str]],
) -> tuple[list[str], dict[str, TmcSegment], int, int, int]:
    locator = NearestTmcLocator(segments, threshold_m)
    tmc_lookup = {seg.tmc: seg for seg in segments}

    matched_sequence: list[Optional[str]] = []
    dropped_dir = 0
    dropped_road = 0
    unmatched = 0

    for lat, lon in sampled_points:
        idx, dist = locator.nearest(lat, lon)
        if idx is None or dist > threshold_m:
            matched_sequence.append(None)
            unmatched += 1
            continue

        seg = segments[idx]
        dir_val = seg.direction.upper()
        road_val = normalize_road(seg.road)

        if expected_dirs and dir_val not in expected_dirs:
            dropped_dir += 1
            matched_sequence.append(None)
            continue
        if expected_roads and road_val not in expected_roads:
            dropped_road += 1
            matched_sequence.append(None)
            continue

        matched_sequence.append(seg.tmc)

    ordered_tmcs = denoise_tmcs(matched_sequence, tmc_lookup)
    return ordered_tmcs, tmc_lookup, dropped_dir, dropped_road, unmatched


def _data_roots() -> list[Path]:
    roots: list[Path] = []
    training_root = Path(__file__).resolve().parents[1]
    for cand in (Path("data"), training_root / "data"):
        if cand.exists() and cand not in roots:
            roots.append(cand)
    return roots


def auto_detect_tmc_file() -> Optional[Path]:
    for root in _data_roots():
        candidates = sorted(root.rglob("TMC_Identification*"))
        if candidates:
            return candidates[0]
    return None


def auto_detect_route_path() -> Optional[Path]:
    static_candidates = [
        Path("nextroute/data/showcase_route_polyline.geojson"),
    ]
    for root in _data_roots():
        static_candidates.extend(
            [
                root / "showcase_route_polyline.geojson",
                root / "showcase_route_polyline.txt",
                root / "showcase_route_points.json",
            ]
        )

    for cand in static_candidates:
        if cand.exists():
            return cand
    return None


def summarize_results(
    tmc_output: list[str],
    tmc_lookup: dict[str, TmcSegment],
    total_tmcs_in_id: int,
    route_points: int,
    sampled_points: int,
) -> None:
    total_miles = sum(tmc_lookup[tmc].miles for tmc in tmc_output)
    road_counts = Counter([normalize_road(tmc_lookup[tmc].road) for tmc in tmc_output])
    logging.info("Total TMCs in identification file: %d", total_tmcs_in_id)
    logging.info("Route points: %d raw, %d sampled", route_points, sampled_points)
    logging.info("Matched unique TMCs: %d", len(tmc_output))
    logging.info("Sum of miles across matched TMCs: %.2f", total_miles)

    if road_counts:
        top_roads = ", ".join([f"{r}:{c}" for r, c in road_counts.most_common(10)])
        logging.info("Top roads by matched TMC count (up to 10): %s", top_roads)

    if tmc_output:
        first_tmcs = ", ".join(tmc_output[:10])
        last_tmcs = ", ".join(tmc_output[-10:])
        logging.info("First TMC IDs: %s", first_tmcs)
        logging.info("Last TMC IDs: %s", last_tmcs)

    if len(tmc_output) < 10:
        logging.warning("Matched TMC count is below 10; check route polyline or threshold.")
    if total_miles < 15 or total_miles > 80:
        logging.warning("Total miles (%.2f) is outside expected showcase range (15-80).", total_miles)


def write_output_csv(out_path: Path, tmcs: list[str], tmc_lookup: dict[str, TmcSegment]) -> None:
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for tmc in tmcs:
        seg = tmc_lookup[tmc]
        rows.append(
            {
                "tmc": seg.tmc,
                "road": seg.road,
                "direction": seg.direction,
                "miles": seg.miles,
                "road_order": seg.road_order if seg.road_order is not None else "",
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logging.info("Wrote %d TMCs to %s", len(rows), out_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ordered TMC list for the showcase route.")
    parser.add_argument("--tmc-id-path", type=Path, default=None, help="Path to TMC_Identification CSV/TSV.")
    parser.add_argument("--route-path", type=Path, default=None, help="Route polyline path (geojson/txt/json).")
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("data/Govt_Data/tmcs_showcase_route.csv"),
        help="Output CSV path for matched TMC list.",
    )
    parser.add_argument("--threshold-m", type=float, default=150.0, help="Max distance (meters) to snap a point.")
    parser.add_argument("--sample-m", type=float, default=50.0, help="Spacing (meters) when sampling the route.")
    parser.add_argument(
        "--expected-directions",
        type=str,
        default=None,
        help='Optional comma list of allowed directions (e.g. "S,E").',
    )
    parser.add_argument(
        "--expected-roads",
        type=str,
        default=None,
        help='Optional comma list of allowed road names (e.g. "GA-141,I-285").',
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)

    tmc_path = args.tmc_id_path or auto_detect_tmc_file()
    if tmc_path is None:
        raise FileNotFoundError("Provide --tmc-id-path; auto-detection failed.")

    route_path = args.route_path or auto_detect_route_path()
    if route_path is None:
        raise FileNotFoundError("Provide --route-path; no default route polyline found.")

    expected_dirs = parse_expected_set(args.expected_directions)
    expected_roads = parse_expected_set(args.expected_roads)
    if expected_dirs:
        logging.info("Enforcing directions: %s", ",".join(sorted(expected_dirs)))
    if expected_roads:
        logging.info("Enforcing roads: %s", ",".join(sorted(expected_roads)))
    if not LineString or not Point or not STRtree:
        logging.info(
            "Install shapely for faster nearest lookup (pip install shapely); using numpy fallback otherwise."
        )

    segments, total_tmcs = load_tmc_identification(tmc_path)
    route_points = load_route_points(route_path)
    sampled_route = densify_route(route_points, args.sample_m)

    matched_tmcs, tmc_lookup, dropped_dir, dropped_road, unmatched = match_route_to_tmcs(
        sampled_route, segments, args.threshold_m, expected_dirs, expected_roads
    )

    logging.info("Dropped %d points due to direction filter and %d due to road filter.", dropped_dir, dropped_road)
    logging.info("Sampled points without any match: %d", unmatched)

    write_output_csv(args.out_path, matched_tmcs, tmc_lookup)
    summarize_results(
        matched_tmcs, tmc_lookup, total_tmcs_in_id=total_tmcs, route_points=len(route_points), sampled_points=len(sampled_route)
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    try:
        main()
    except Exception as exc:
        logging.error("Failed to build showcase route TMC list: %s", exc)
        sys.exit(1)
