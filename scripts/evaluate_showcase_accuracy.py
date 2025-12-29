from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow.parquet as pq
import pandas as pd


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_val_report(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"val_report not found: {path}")
    data = json.loads(path.read_text())
    mae = data.get("mae") or {}
    return {str(k): float(v) for k, v in mae.items()}


def load_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    # Read only needed columns via pyarrow if available.
    schema = pq.read_schema(dataset_path)
    colnames = set(schema.names)
    needed = []
    use_speed_ratio = False
    if "speed_ratio" in colnames:
        needed.append("speed_ratio")
        use_speed_ratio = True
    if "speed" in colnames:
        needed.append("speed")
    if "reference_speed" in colnames:
        needed.append("reference_speed")
    table = pq.read_table(dataset_path, columns=needed)
    df = table.to_pandas()

    if "reference_speed" not in df.columns:
        raise ValueError("Dataset missing required column: reference_speed")

    ref = df["reference_speed"].to_numpy(dtype=np.float32, copy=False)
    if use_speed_ratio:
        speed_mph = (df["speed_ratio"].to_numpy(dtype=np.float32, copy=False) * ref)
    elif "speed" in df.columns:
        speed_mph = df["speed"].to_numpy(dtype=np.float32, copy=False)
    else:
        raise ValueError("Dataset missing required columns; need speed_ratio or speed alongside reference_speed.")

    return speed_mph, ref


def compute_accuracy(speed_mph: np.ndarray, ref: np.ndarray, mae_ratio: float) -> tuple[float, float, float]:
    if len(speed_mph) == 0:
        raise ValueError("Empty dataset; cannot compute accuracy.")
    mean_ref = float(np.nanmean(ref))
    speed_clean = speed_mph[~np.isnan(speed_mph)]
    if len(speed_clean) == 0:
        raise ValueError("All speeds are NaN; cannot compute accuracy.")
    mad = float(np.mean(np.abs(speed_clean - np.mean(speed_clean))))
    if mad == 0.0:
        raise ValueError("MAD is zero; cannot compute accuracy.")
    error_mph = mae_ratio * mean_ref
    accuracy = 1 - (error_mph / mad)
    return mad, error_mph, accuracy


def parse_horizons(raw: str) -> list[int]:
    return [int(h.strip()) for h in raw.split(",") if h.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate showcase TFT accuracy using val_report.json.")
    parser.add_argument(
        "--val-report",
        type=Path,
        default=Path("TrainingModel/tft/artifacts_showcase/val_report.json"),
        help="Path to val_report.json from training.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("TrainingModel/tft/artifacts_showcase/dataset.parquet"),
        help="Path to TFT dataset parquet used for training.",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="5,15,30,60",
        help="Comma-separated horizons to evaluate (minutes).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)

    mae_map = load_val_report(args.val_report)
    speed_mph, ref_arr = load_dataset(args.dataset)
    horizons = parse_horizons(args.horizons)

    logging.info("Loaded val_report with horizons: %s", sorted(mae_map.keys()))
    logging.info("Dataset rows: %d", len(speed_mph))

    for h in horizons:
        key = str(h)
        if key not in mae_map:
            logging.warning("Horizon %s not found in val_report; skipping.", key)
            continue
        mad, err, acc = compute_accuracy(speed_mph, ref_arr, mae_map[key])
        logging.info(
            "Horizon %s min -> MAE_ratio=%.4f | MAD=%.2f mph | error=%.2f mph | accuracy=%.2f%%",
            key,
            mae_map[key],
            mad,
            err,
            acc * 100.0,
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
