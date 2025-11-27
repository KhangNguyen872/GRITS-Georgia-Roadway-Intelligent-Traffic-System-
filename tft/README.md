# Temporal Fusion Transformer Backend

This package hosts the additive TFT stack that will sit alongside the existing GBT
prototype. The GBT path and `/predict` contract stay untouched; toggling between
backends is controlled via the `PREDICTOR_BACKEND` environment variable (`GBT`
remains the default). Set `PREDICTOR_BACKEND=TFT` and point `TFT_BUNDLE_PATH`
to a trained bundle to enable the deep-learning forecaster.

## Contents

- `data_pipeline.py` – builds the 5‑minute corridor feature grid from GA‑511 and
  NWS snapshots, synthesises speed targets for +5/+15/+30/+60 minute horizons,
  and produces train/validation splits.
- `model.py` – minimal, TensorRT-friendly TFT implementation plus the
  `pytorch-lightning` wrapper and bundle dataclass used for export.
- `train.py` – orchestrates the pipeline → dataset windowing → Lightning Trainer
  loop. Backs up the current GBT model before fitting and exports:
  - `tft/artifacts/tft_bundle.pt`
  - `tft/artifacts/val_report.json`
  - `tft/artifacts/val_plot.png`
- `predictor.py` – runtime helper that loads the bundle, rebuilds the necessary
  context from the live CSVs, and emits multi-horizon speed forecasts with a
  confidence score derived from validation MAE.
- `artifacts/` – default output directory for bundles, reports, and plots.

## Dependencies

Python 3.10+ with the following packages (install via pip in a virtual
environment):

```
torch>=2.2         # CPU wheels available from https://download.pytorch.org
pytorch-lightning>=2.1
numpy>=1.26
pandas>=2.2
matplotlib>=3.9    # optional for plots
```

The training script does **not** install these automatically; install them ahead
of time (air-gap users can preload the wheels).

## Usage

### 1. Generate the dataset (optional)

```
python -m tft.data_pipeline \
  --ga511-csv data/live_logs/ga511_events.csv \
  --nws-csv data/live_logs/nws_hourly.csv \
  --corridors "I-85,SR 1,SR 38" \
  --output tft/artifacts/dataset.parquet
```

This writes a tall table with the engineered features plus synthetic targets and
metadata. The training script calls the same code path internally, so running
this step beforehand is optional.

### 2. Train the TFT

```
python -m tft.train \
  --epochs 20 \
  --batch-size 128 \
  --lr 3e-4 \
  --horizons "5,15,30,60" \
  --corridors "I-85,SR 1,SR 38" \
  --seed 42 \
  --output-dir tft/artifacts
```

What happens:

1. Copies `gbt_prototype/models/latest_model.pkl` to
   `backups/models/latest_model_<UTC>.pkl`.
2. Builds the 5‑minute grid, scales/encodes the features, and windows sequences
   with 24 history steps (2 hours) for each training sample.
3. Fits the TFT (Lightning + custom backbone) with early stopping on 15‑minute
   MAE and saves the best checkpoint.
4. Exports `tft_bundle.pt`, the validation report, and the MAE plot.

If PyTorch or Lightning are missing the script will abort early—install the
dependencies first.

### 3. Toggle inference

```bash
export PREDICTOR_BACKEND=TFT
export TFT_BUNDLE_PATH=tft/artifacts/tft_bundle.pt
```

Then inside your service you can instantiate the predictor:

```python
from tft.predictor import load_predictor_if_enabled

predictor = load_predictor_if_enabled()
if predictor:
    preds = predictor.predict("I-85", datetime.utcnow())
```

Each result item contains the horizon (minutes), predicted speed (mph),
confidence (0–1), direction, and the timestamp the context window was anchored
to.

### 4. Confidence calibration

Confidence scores are derived from validation MAE per horizon:

```
confidence = max(0.05, min(0.99, 1 - mae / freeflow_mph))
```

Lower residuals push confidence towards 1.0; larger residuals lower it. Update
`TFTBundle.residual_mae` if you re-train or alter the loss function.

## Notes & future work

- The synthetic target generator embeds weather and incident effects so the TFT
  learns multi-modal responses before we swap in real probe speeds.
- Extend `tft/predictor.py` with cached contexts or a streaming feature store
  when live ingestion moves off CSV.
- The exported bundle is TorchScript/ONNX friendly—follow the TRT notes under
  `tft_trt/` once we stabilise training on real data.
