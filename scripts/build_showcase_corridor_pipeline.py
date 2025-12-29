from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

try:
    import duckdb  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency guard
    duckdb = None
    _duckdb_import_error = exc
else:
    _duckdb_import_error = None


@dataclass
class RouteSummary:
    tmcs: list[str]
    total_miles: float
    road_counts: Counter
    direction_counts: Counter


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_route_tmcs(path: Path) -> RouteSummary:
    tmcs: list[str] = []
    road_counts: Counter = Counter()
    direction_counts: Counter = Counter()
    total_miles = 0.0
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            tmc = row.get("tmc", "").strip()
            if not tmc:
                continue
            tmcs.append(tmc)
            miles = row.get("miles")
            try:
                total_miles += float(miles) if miles not in (None, "") else 0.0
            except Exception:
                pass
            road = row.get("road", "").strip()
            direction = row.get("direction", "").strip()
            if road:
                road_counts[road] += 1
            if direction:
                direction_counts[direction] += 1
    return RouteSummary(tmcs=tmcs, total_miles=total_miles, road_counts=road_counts, direction_counts=direction_counts)


def _values_clause(values: Iterable[str]) -> str:
    escaped = [v.replace("'", "''") for v in values]
    return ", ".join(f"('{v}')" for v in escaped)


def ensure_duckdb() -> None:
    if duckdb is None:
        raise RuntimeError(f"duckdb is required for this pipeline but failed to import: {_duckdb_import_error}")


def filter_probe(probe_path: Path, tmcs: list[str], output: Path) -> None:
    ensure_duckdb()
    output.parent.mkdir(parents=True, exist_ok=True)
    values_clause = _values_clause(tmcs)
    sql = f"""
    COPY (
      SELECT *
      FROM read_csv_auto('{probe_path}', SAMPLE_SIZE=-1, IGNORE_ERRORS=TRUE)
      WHERE tmc_code IN (SELECT tmc FROM (VALUES {values_clause}) AS v(tmc))
    ) TO '{output}' (FORMAT 'parquet', COMPRESSION 'snappy');
    """
    conn = duckdb.connect(database=":memory:")
    conn.execute(sql)
    conn.close()
    logging.info("Filtered probe written to %s", output)


def aggregate_probe(source_path: Path, tmcs: list[str], output: Path, freq_minutes: int) -> dict[str, str]:
    ensure_duckdb()
    output.parent.mkdir(parents=True, exist_ok=True)
    freq_ms = freq_minutes * 60 * 1000
    values_clause = _values_clause(tmcs)

    if source_path.suffix.lower() in {".parquet", ".pq"}:
        source = f"SELECT * FROM read_parquet('{source_path}')"
    else:
        source = f"SELECT * FROM read_csv_auto('{source_path}', SAMPLE_SIZE=-1, IGNORE_ERRORS=TRUE)"

    sql = f"""
    COPY (
      WITH route_tmcs AS (
        SELECT tmc FROM (VALUES {values_clause}) AS v(tmc)
      ),
      base AS (
        SELECT
          CAST(measurement_tstamp AS TIMESTAMP) AS ts,
          tmc_code AS tmc,
          speed,
          reference_speed,
          confidence
        FROM ({source})
        WHERE tmc_code IN (SELECT tmc FROM route_tmcs)
      ),
      binned AS (
        SELECT
          to_timestamp((floor(epoch_ms(ts) / {freq_ms}) * {freq_ms}) / 1000.0) AS ts_utc,
          tmc,
          speed,
          reference_speed,
          confidence
        FROM base
        WHERE ts IS NOT NULL
      ),
      agg AS (
        SELECT
          ts_utc,
          tmc,
          COALESCE(
            SUM(speed * confidence) FILTER (WHERE confidence > 0) / NULLIF(SUM(confidence) FILTER (WHERE confidence > 0), 0),
            AVG(speed)
          ) AS speed,
          AVG(reference_speed) FILTER (WHERE reference_speed > 0) AS reference_speed,
          AVG(confidence) AS confidence
        FROM binned
        GROUP BY ts_utc, tmc
        ORDER BY ts_utc, tmc
      )
      SELECT ts_utc AS tstamp, * FROM agg
    ) TO '{output}' (FORMAT 'parquet', COMPRESSION 'snappy');
    """
    conn = duckdb.connect(database=":memory:")
    conn.execute(sql)
    # Summaries
    stats = conn.execute(
        f"""
        SELECT
          COUNT(*) AS rows,
          COUNT(DISTINCT tmc) AS tmcs,
          MIN(ts_utc)::VARCHAR AS min_ts,
          MAX(ts_utc)::VARCHAR AS max_ts,
          100.0 * SUM(CASE WHEN reference_speed IS NULL OR reference_speed <= 0 THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS pct_ref_missing
        FROM read_parquet('{output}')
        """
    ).fetchone()
    conn.close()
    summary = {
        "rows": str(stats[0]),
        "tmcs": str(stats[1]),
        "min_ts": stats[2],
        "max_ts": stats[3],
        "pct_ref_missing": f"{stats[4]:.2f}",
    }
    logging.info(
        "Aggregated rows=%s tmcs=%s ts range=[%s, %s] ref_missing=~%s%%",
        summary["rows"],
        summary["tmcs"],
        summary["min_ts"],
        summary["max_ts"],
        summary["pct_ref_missing"],
    )
    return summary


def build_tmc_metadata(route_path: Path, aggregated_path: Path, output: Path) -> None:
    ensure_duckdb()
    # Load route miles
    miles_map: dict[str, float] = {}
    with route_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            tmc = row.get("tmc", "").strip()
            if not tmc:
                continue
            miles = row.get("miles")
            try:
                miles_map[tmc] = float(miles) if miles not in (None, "") else 0.0
            except Exception:
                miles_map[tmc] = 0.0

    conn = duckdb.connect(database=":memory:")
    quant = conn.execute(
        f"""
        SELECT
          tmc,
          quantile_cont(reference_speed, 0.9) AS ref_q90
        FROM read_parquet('{aggregated_path}')
        WHERE reference_speed > 0
        GROUP BY tmc
        """
    ).fetchall()
    conn.close()
    ref_map = {tmc: (float(ref_q90) if ref_q90 is not None else 65.0) for tmc, ref_q90 in quant}

    rows = []
    for tmc, miles in miles_map.items():
        ref = ref_map.get(tmc, 65.0)
        rows.append({"tmc": tmc, "length_miles": miles, "reference_speed_mph": ref})

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tmc", "length_miles", "reference_speed_mph"])
        writer.writeheader()
        writer.writerows(rows)

    logging.info("Wrote %d rows to %s (tmc metadata)", len(rows), output)
    logging.info("Sample (first 5): %s", rows[:5])


def run_subprocess(cmd: list[str], cwd: Optional[Path] = None) -> None:
    env = os.environ.copy()
    extra_path = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = (
        extra_path if not env.get("PYTHONPATH") else f"{extra_path}{os.pathsep}{env['PYTHONPATH']}"
    )
    logging.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def build_dataset(
    npmrds_path: Path,
    out_dir: Path,
    freq_minutes: int,
    horizons: list[int],
    resume: bool,
    incidents_csv: Optional[Path],
) -> None:
    dataset_path = out_dir / "dataset.parquet"
    meta_path = out_dir / "dataset_meta.json"
    if resume and dataset_path.exists() and meta_path.exists():
        logging.info("Dataset already exists; skipping (resume).")
        return
    incidents_arg = incidents_csv if incidents_csv is not None else Path("__NO_INCIDENTS__")
    cmd = [
        sys.executable,
        "-m",
        "tft.data_pipeline",
        "--npmrds-path",
        str(npmrds_path),
        "--output",
        str(dataset_path),
        "--meta",
        str(meta_path),
        "--freq",
        f"{freq_minutes}min",
        "--horizons",
        ",".join(map(str, horizons)),
        "--incidents-csv",
        str(incidents_arg),
    ]
    run_subprocess(cmd)


def train_tft(dataset_path: Path, meta_path: Path, out_dir: Path, epochs: int, batch_size: int, resume: bool) -> Optional[float]:
    bundle_path = out_dir / "tft_bundle.pt"
    if resume and bundle_path.exists():
        logging.info("tft_bundle.pt already exists; skipping training (resume).")
        return None
    cmd = [
        sys.executable,
        "-m",
        "tft.train",
        "--dataset",
        str(dataset_path),
        "--meta",
        str(meta_path),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--output-dir",
        str(out_dir),
    ]
    run_subprocess(cmd)
    return None


def compute_date_range(parquet_path: Path) -> tuple[Optional[str], Optional[str]]:
    ensure_duckdb()
    conn = duckdb.connect(database=":memory:")
    row = conn.execute(
        f"SELECT MIN(ts_utc)::VARCHAR, MAX(ts_utc)::VARCHAR FROM read_parquet('{parquet_path}')"
    ).fetchone()
    conn.close()
    return row[0], row[1]


def write_run_summary(
    out_dir: Path,
    route_summary: RouteSummary,
    date_range: tuple[Optional[str], Optional[str]],
    horizons: list[int],
    freq_minutes: int,
    epochs: int,
    final_val_mae: Optional[float],
) -> None:
    summary = {
        "route_tmcs_count": len(route_summary.tmcs),
        "total_route_miles": route_summary.total_miles,
        "date_range": date_range,
        "horizons": horizons,
        "freq_minutes": freq_minutes,
        "epochs_run": epochs,
        "final_val_mae": final_val_mae,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    logging.info("Run summary written to %s", path)


def parse_horizons(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Showcase corridor end-to-end pipeline.")
    parser.add_argument("--probe-path", type=Path, required=True, help="Raw probe CSV/TSV path.")
    parser.add_argument("--route-tmcs", type=Path, required=True, help="Route TMC list CSV.")
    parser.add_argument("--tmc-id-path", type=Path, required=True, help="TMC identification CSV/TSV (unused beyond validation).")
    parser.add_argument("--out-parquet", type=Path, required=True, help="Output aggregated parquet path (5-min).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for TFT artifacts.")
    parser.add_argument("--freq-minutes", type=int, default=5, help="Aggregation frequency in minutes.")
    parser.add_argument("--horizons", type=str, default="5,15,30,60", help="Comma-separated prediction horizons.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs for TFT.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--incidents-csv", type=Path, default=None, help="Optional incidents CSV; if omitted, incidents are skipped.")
    parser.add_argument("--resume", action="store_true", help="Skip steps whose outputs already exist.")
    parser.add_argument("--no-filter", action="store_true", help="Skip writing filtered parquet (aggregate directly from raw).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)

    route_summary = read_route_tmcs(args.route_tmcs)
    logging.info(
        "Route TMCs: %d | total miles: %.2f | top roads: %s | top directions: %s",
        len(route_summary.tmcs),
        route_summary.total_miles,
        route_summary.road_counts.most_common(5),
        route_summary.direction_counts.most_common(5),
    )
    if route_summary.total_miles < 20 or route_summary.total_miles > 80:
        raise ValueError("Route miles outside expected range (20-80); aborting.")

    if _duckdb_import_error:
        ensure_duckdb()

    filtered_path = Path(args.probe_path.parent / "showcase_probe_filtered.parquet")
    if not args.no_filter:
        if args.resume and filtered_path.exists():
            logging.info("Filtered probe already exists; skipping (resume).")
        else:
            filter_probe(args.probe_path, route_summary.tmcs, filtered_path)
    else:
        logging.info("Skipping filtered parquet creation (--no-filter).")

    source_for_agg = filtered_path if (not args.no_filter and filtered_path.exists()) else args.probe_path
    if args.resume and args.out_parquet.exists():
        logging.info("Aggregated parquet already exists; skipping (resume).")
    else:
        aggregate_probe(source_for_agg, route_summary.tmcs, args.out_parquet, args.freq_minutes)

    meta_csv = args.out_dir / "tmc_metadata_showcase.csv"
    if args.resume and meta_csv.exists():
        logging.info("TMC metadata already exists; skipping (resume).")
    else:
        build_tmc_metadata(args.route_tmcs, args.out_parquet, meta_csv)

    horizons = parse_horizons(args.horizons)
    build_dataset(args.out_parquet, args.out_dir, args.freq_minutes, horizons, args.resume, args.incidents_csv)

    dataset_path = args.out_dir / "dataset.parquet"
    meta_path = args.out_dir / "dataset_meta.json"
    final_mae = train_tft(dataset_path, meta_path, args.out_dir, args.epochs, args.batch_size, args.resume)

    date_range = compute_date_range(args.out_parquet)
    write_run_summary(
        out_dir=args.out_dir,
        route_summary=route_summary,
        date_range=date_range,
        horizons=horizons,
        freq_minutes=args.freq_minutes,
        epochs=args.epochs,
        final_val_mae=final_mae,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    try:
        main()
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        sys.exit(1)
