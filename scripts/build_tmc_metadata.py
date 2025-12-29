from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def find_tmc_meta_file(search_root: Path) -> Optional[Path]:
    """Search under ``search_root`` for a CSV containing the marker string."""
    marker = "Manually corrected on 2/11"
    if not search_root.exists():
        return None
    for csv_path in search_root.rglob("*.csv"):
        try:
            with csv_path.open("r", errors="ignore") as f:
                head = f.read(2048)
            if marker in head:
                return csv_path
        except Exception:
            continue
    return None


def detect_tmc_column(df: pd.DataFrame) -> str:
    """Pick the column whose values end with 'INC' most frequently."""
    candidate_cols = df.select_dtypes(include=["object"]).columns
    best_col: Optional[str] = None
    best_score = -1.0
    for col in candidate_cols:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue
        score = (series.str.endswith("INC")).mean()
        if score > best_score:
            best_score = score
            best_col = col
    if best_col is None or best_score <= 0.1:
        raise ValueError("Unable to detect TMC column (values ending with 'INC' not found).")
    return best_col


def detect_length_column(df: pd.DataFrame) -> str:
    """Pick a length column: prefer name containing 'length', else last numeric column."""
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if not numeric_cols:
        raise ValueError("No numeric columns available to detect length.")
    for col in numeric_cols:
        if "length" in col.lower() or "len" in col.lower():
            return col
    return numeric_cols[-1]


def load_tmc_lengths(path: Path) -> pd.DataFrame:
    """Load the raw TMC metadata CSV and return tmc,length_miles."""
    raw = pd.read_csv(path)
    tmc_col = detect_tmc_column(raw)
    length_col = detect_length_column(raw)

    logging.info("Detected TMC column: %s", tmc_col)
    logging.info("Detected length column: %s", length_col)

    lengths = (
        raw[[tmc_col, length_col]]
        .rename(columns={tmc_col: "tmc", length_col: "length_miles"})
        .dropna(subset=["tmc", "length_miles"])
    )
    lengths["tmc"] = lengths["tmc"].astype(str)
    lengths["length_miles"] = pd.to_numeric(lengths["length_miles"], errors="coerce")
    lengths = lengths.dropna(subset=["length_miles"])
    lengths = lengths.drop_duplicates(subset=["tmc"])

    return lengths


def load_reference_speeds(path: Path, min_count: int = 20, q: float = 0.9) -> pd.DataFrame:
    """Compute per-TMC reference_speed_mph from NPMRDS parquet/CSV."""
    if not path.exists():
        raise FileNotFoundError(f"NPMRDS path not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    tmc_col = None
    for cand in ("tmc", "tmc_code", "TMC"):
        if cand in df.columns:
            tmc_col = cand
            break
    if tmc_col is None:
        raise ValueError("Unable to find TMC column in NPMRDS data (expected 'tmc' or 'tmc_code').")

    ref_col = None
    for cand in ("reference_speed", "reference_speed_mph"):
        if cand in df.columns:
            ref_col = cand
            break
    if ref_col is None:
        raise ValueError("Unable to find reference speed column in NPMRDS data.")

    frame = df[[tmc_col, ref_col]].dropna()
    frame = frame[frame[ref_col] > 0]
    frame = frame.rename(columns={tmc_col: "tmc", ref_col: "reference_speed"})
    frame["tmc"] = frame["tmc"].astype(str)
    frame["reference_speed"] = pd.to_numeric(frame["reference_speed"], errors="coerce")
    frame = frame.dropna(subset=["reference_speed"])

    def agg(group: pd.Series) -> float:
        series = group.dropna()
        if len(series) == 0:
            return np.nan
        if len(series) >= min_count:
            return float(series.quantile(q))
        return float(series.mean())

    ref = (
        frame.groupby("tmc")["reference_speed"]
        .apply(agg)
        .reset_index()
        .rename(columns={"reference_speed": "reference_speed_mph"})
    )
    return ref


def default_reference_speed(tmc: str) -> float:
    # simple heuristic; can be replaced when better metadata is available
    if tmc.startswith("1"):
        return 65.0
    return 55.0


def sanity_checks(df: pd.DataFrame) -> None:
    if df["length_miles"].le(0).any():
        logging.warning("Some TMCs have non-positive length_miles.")
    if (df["length_miles"] > 20).any():
        logging.warning("Some TMCs have length_miles > 20 (check source).")
    if ((df["reference_speed_mph"] < 10) | (df["reference_speed_mph"] > 90)).any():
        logging.warning("Some reference_speed_mph values are outside 10-90 mph.")
    avg_len = df["length_miles"].mean()
    if avg_len < 0.05 or avg_len > 10:
        logging.warning("Average length_miles seems unusual: %.3f", avg_len)


def build_metadata(tmc_meta_path: Path, npmrds_path: Path, output: Path, verbose: bool) -> None:
    configure_logging(verbose)
    logging.info("Using TMC meta: %s", tmc_meta_path)
    logging.info("Using NPMRDS data: %s", npmrds_path)

    lengths = load_tmc_lengths(tmc_meta_path)
    logging.info("Loaded %d unique TMC lengths", len(lengths))

    ref = load_reference_speeds(npmrds_path)
    logging.info("Computed reference speeds for %d TMCs", len(ref))

    meta = lengths.merge(ref, on="tmc", how="left")
    missing_ref = meta["reference_speed_mph"].isna().sum()
    if missing_ref > 0:
        logging.warning("Missing reference_speed_mph for %d TMCs; filling with defaults", missing_ref)
        meta["reference_speed_mph"] = meta["reference_speed_mph"].fillna(
            meta["tmc"].map(default_reference_speed)
        )

    sanity_checks(meta)

    output.parent.mkdir(parents=True, exist_ok=True)
    meta[["tmc", "length_miles", "reference_speed_mph"]].to_csv(output, index=False)

    logging.info("Wrote %d rows to %s", len(meta), output)
    logging.info("Sample:\n%s", meta.head().to_string(index=False))


def auto_detect_tmc_meta() -> Optional[Path]:
    return find_tmc_meta_file(Path("data"))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TMC metadata (length + reference speed).")
    parser.add_argument("--tmc-meta-csv", type=Path, default=None, help="Path to raw TMC metadata CSV.")
    parser.add_argument("--npmrds-path", type=Path, required=True, help="Path to NPMRDS parquet/CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tft/artifacts/tmc_metadata.csv"),
        help="Output CSV path for cleaned metadata.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    tmc_meta = args.tmc_meta_csv or auto_detect_tmc_meta()
    if tmc_meta is None:
        raise FileNotFoundError(
            "TMC metadata CSV not provided and auto-detection failed (looked under data/ for marker)."
        )
    build_metadata(tmc_meta_path=tmc_meta, npmrds_path=args.npmrds_path, output=args.output, verbose=args.verbose)


if __name__ == "__main__":
    main()
