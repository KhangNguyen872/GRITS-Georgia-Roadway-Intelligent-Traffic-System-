import json
import numpy as np
import pandas as pd
from pathlib import Path

VAL_REPORT = Path("tft/artifacts/val_report.json")
DATASET_PATH = Path("tft/artifacts/dataset_train_sample.parquet")
TARGET_HORIZON = 15  # minutes

if not VAL_REPORT.exists():
    raise FileNotFoundError(f"Validation report not found: {VAL_REPORT}")
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset sample not found: {DATASET_PATH}")

# Load validation MAE (ratio space) for the chosen horizon.
with VAL_REPORT.open("r") as f:
    report = json.load(f)

mae_map = report.get("mae") or {}
if str(TARGET_HORIZON) not in mae_map:
    raise KeyError(f"Horizon {TARGET_HORIZON} not found in val_report 'mae' keys: {list(mae_map.keys())}")
mae_ratio = float(mae_map[str(TARGET_HORIZON)])

# Load dataset sample (expects speed_ratio and reference_speed columns).
df = pd.read_parquet(DATASET_PATH)
if "speed_ratio" not in df.columns or "reference_speed" not in df.columns:
    raise KeyError("Dataset is missing required columns: speed_ratio, reference_speed")

# Convert ratio to mph.
df["speed_mph"] = df["speed_ratio"] * df["reference_speed"]

# Natural variation (mean absolute deviation from mean).
mad = float(np.mean(np.abs(df["speed_mph"] - df["speed_mph"].mean())))
if mad == 0:
    raise ValueError("MAD is zero; cannot compute accuracy.")

# Model error in mph (assumes ratio MAE is relative to reference_speed).
error_mph = mae_ratio * float(df["reference_speed"].mean())

accuracy = 1 - (error_mph / mad)

print(f"Horizon: {TARGET_HORIZON} minutes")
print(f"MAE (ratio): {mae_ratio:.4f}")
print(f"MAD (mph): {mad:.2f}")
print(f"Estimated model error (mph): {error_mph:.2f}")
print(f"Model accuracy (1 - err/mad): {accuracy * 100:.2f}%")
