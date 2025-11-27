from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "pytorch_lightning is required to train the TFT model. "
        "Install it via `pip install pytorch-lightning`."
    ) from exc

from .data_pipeline import PipelineConfig, build_dataset
from .model import FeatureConfig, TFTBackbone, TFTBundle, TFTLightningModule


# ---------------------------------------------------------------------------
# Dataset utilities


OBS_CONT_COLS = [
    "speed_ratio",
    "speed",
    "reference_speed",
    "confidence",
    "incident_active",
    "lanes_blocked_count",
    "full_closure",
    "incident_severity_bucket",
    "hour_sin",
    "hour_cos",
    "is_weekend",
]

FUT_CONT_COLS = [
    "hour_sin",
    "hour_cos",
    "is_weekend",
]

STATIC_CONT_COLS: list[str] = []

HIST_CAT_COLS = ["incident_type"]
FUT_CAT_COLS = ["dow"]
STATIC_CAT_COLS = ["tmc"]


@dataclass
class DatasetConfig:
    encoder_length: int
    horizons: Sequence[int]
    horizon_steps: Sequence[int]
    scalers: dict[str, dict[str, float]]
    category_maps: dict[str, dict[str, int]]


class TFTDataset(Dataset):
    def __init__(
        self,
        grouped_frames: dict[str, pd.DataFrame],
        sample_index: list[tuple[str, int]],
        cfg: DatasetConfig,
    ) -> None:
        self.grouped_frames = grouped_frames
        self.sample_index = sample_index
        self.cfg = cfg
        self.obs_cont_norm = [f"{col}_norm" for col in OBS_CONT_COLS]
        self.fut_cont_norm = [f"{col}_norm" for col in FUT_CONT_COLS]
        self.static_cont_norm = [f"{col}_norm" for col in STATIC_CONT_COLS]

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tmc, pos = self.sample_index[idx]
        frame = self.grouped_frames[tmc]
        enc_start = pos - self.cfg.encoder_length
        enc_slice = frame.iloc[enc_start:pos]
        base_row = frame.iloc[pos]

        hist_cont = enc_slice[self.obs_cont_norm].to_numpy(dtype=np.float32)
        hist_cat = None
        if HIST_CAT_COLS:
            hist_cat_arrays = []
            for col in HIST_CAT_COLS:
                mapping = self.cfg.category_maps[col]
                hist_cat_arrays.append(
                    enc_slice[col].map(lambda v: mapping.get(str(v), 0)).to_numpy(dtype=np.int64)
                )
            hist_cat = np.stack(hist_cat_arrays, axis=1)  # (T, V)

        fut_cont_rows = []
        fut_cat_rows = []
        for step, horizon in zip(self.cfg.horizon_steps, self.cfg.horizons):
            future_row = frame.iloc[pos + step]
            fut_cont_rows.append(future_row[self.fut_cont_norm].to_numpy(dtype=np.float32))
            if FUT_CAT_COLS:
                fut_cat_values = []
                for col in FUT_CAT_COLS:
                    mapping = self.cfg.category_maps[col]
                    fut_cat_values.append(mapping.get(str(future_row[col]), 0))
                fut_cat_rows.append(fut_cat_values)
        fut_cont = np.stack(fut_cont_rows, axis=0)
        fut_cat = None
        if FUT_CAT_COLS:
            fut_cat = np.array(fut_cat_rows, dtype=np.int64)

        static_cont = None
        if STATIC_CONT_COLS:
            static_cont = base_row[self.static_cont_norm].to_numpy(dtype=np.float32)
        static_cat = None
        if STATIC_CAT_COLS:
            static_cat = []
            for col in STATIC_CAT_COLS:
                mapping = self.cfg.category_maps[col]
                static_cat.append(mapping.get(str(base_row[col]), 0))
            static_cat = np.array(static_cat, dtype=np.int64)

        targets = []
        for horizon in self.cfg.horizons:
            targets.append(base_row[f"target_speed_ratio(+{horizon})"])
        target_arr = np.array(targets, dtype=np.float32)

        batch = {
            "x_hist_cont": torch.from_numpy(hist_cont),
            "x_fut_cont": torch.from_numpy(fut_cont),
            "y": torch.from_numpy(target_arr),
        }
        if hist_cat is not None:
            batch["x_hist_cat"] = torch.from_numpy(hist_cat.astype(np.int64))
        if fut_cat is not None:
            batch["x_fut_cat"] = torch.from_numpy(fut_cat.astype(np.int64))
        if static_cont is not None:
            batch["x_static_cont"] = torch.from_numpy(static_cont)
        if static_cat is not None:
            batch["x_static_cat"] = torch.from_numpy(static_cat.astype(np.int64))
        return batch


# ---------------------------------------------------------------------------
# Helpers


def backup_gbt_model() -> Optional[Path]:
    src = Path("gbt_prototype/models/latest_model.pkl")
    if not src.exists():
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    dst_dir = Path("backups/models")
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"latest_model_{ts}.pkl"
    shutil.copy2(src, dst)
    return dst


def build_category_maps(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for col in set(HIST_CAT_COLS + FUT_CAT_COLS + STATIC_CAT_COLS):
        values = sorted(df[col].astype(str).unique())
        maps[col] = {str(v): i for i, v in enumerate(values)}
    return maps


def compute_scalers(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    scalers: dict[str, dict[str, float]] = {}
    for col in set(OBS_CONT_COLS + FUT_CONT_COLS + STATIC_CONT_COLS):
        series = df[col].astype(float)
        mean = float(series.mean())
        std = float(series.std() or 1.0)
        scalers[col] = {"mean": mean, "std": std}
    return scalers


def apply_scalers(df: pd.DataFrame, scalers: dict[str, dict[str, float]]) -> pd.DataFrame:
    result = df.copy()
    for col, params in scalers.items():
        norm_col = f"{col}_norm"
        result[norm_col] = (result[col] - params["mean"]) / params["std"]
    return result


def group_sequences(df: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    grouped: dict[str, pd.DataFrame] = {}
    for tmc, group in df.groupby("tmc"):
        grouped[str(tmc)] = group.sort_values("time_idx").reset_index(drop=True)
    return grouped


def build_sample_index(
    grouped: dict[str, pd.DataFrame],
    encoder_length: int,
    horizon_steps: Sequence[int],
    target_split: str,
) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    max_step = max(horizon_steps)
    for tmc, group in grouped.items():
        for pos in range(encoder_length, len(group) - max_step):
            if group.iloc[pos]["split"] != target_split:
                continue
            samples.append((tmc, pos))
    return samples


# ---------------------------------------------------------------------------
# Training routine


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Temporal Fusion Transformer backend.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--encoder-length", type=int, default=24, help="Number of history steps (5-min) for encoder.")
    parser.add_argument("--horizons", type=str, default="5,15,30,60")
    parser.add_argument("--freq", type=str, default="5min")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("tft/artifacts"))
    parser.add_argument("--npmrds-path", type=Path, default=PipelineConfig.npmrds_path)
    parser.add_argument("--incidents-csv", type=Path, default=PipelineConfig.incidents_csv)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--dataset", type=Path, default=None, help="Optional prebuilt dataset parquet/csv.")
    parser.add_argument("--meta", type=Path, default=None, help="Optional metadata JSON for prebuilt dataset.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    dataset_df: pd.DataFrame
    metadata: dict

    if args.dataset:
        if args.meta is None:
            raise ValueError("When providing --dataset, you must also provide --meta for horizons/freq.")
        suffix = args.dataset.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            dataset_df = pd.read_parquet(args.dataset)
        else:
            dataset_df = pd.read_csv(args.dataset)
        with args.meta.open() as f:
            metadata = json.load(f)
        if "horizons" not in metadata:
            raise ValueError("Metadata file is missing 'horizons'; cannot infer targets.")
        horizons = [int(h) for h in metadata["horizons"]]
        pl.seed_everything(args.seed, workers=True)
        backup_path = backup_gbt_model()
    else:
        horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
        pl.seed_everything(args.seed, workers=True)
        backup_path = backup_gbt_model()
        cfg = PipelineConfig(
            npmrds_path=args.npmrds_path,
            incidents_csv=args.incidents_csv,
            freq=args.freq,
            horizons=horizons,
            seed=args.seed,
            output_path=None,
            meta_path=None,
            min_confidence=args.min_confidence,
        )
        dataset_df, metadata = build_dataset(cfg)

    train_df = dataset_df[dataset_df["split"] == "train"]
    scalers = compute_scalers(train_df)
    scaled_df = apply_scalers(dataset_df, scalers)
    for col in set(HIST_CAT_COLS + FUT_CAT_COLS + STATIC_CAT_COLS):
        scaled_df[col] = scaled_df[col].astype(str)
    category_maps = build_category_maps(scaled_df)
    grouped = group_sequences(scaled_df)
    freq_minutes = int(metadata.get("freq_minutes", 5))
    if freq_minutes <= 0:
        freq_minutes = 5
    horizon_steps = [h // freq_minutes for h in horizons]

    encoder_length = args.encoder_length
    train_index = build_sample_index(grouped, encoder_length, horizon_steps, target_split="train")
    val_index = build_sample_index(grouped, encoder_length, horizon_steps, target_split="val")
    if not train_index:
        raise RuntimeError("No training samples available after windowing. Reduce encoder length or verify data.")
    if not val_index:
        raise RuntimeError("No validation samples available; increase history window or dataset size.")

    dataset_cfg = DatasetConfig(
        encoder_length=encoder_length,
        horizons=horizons,
        horizon_steps=horizon_steps,
        scalers=scalers,
        category_maps=category_maps,
    )

    train_dataset = TFTDataset(grouped, train_index, dataset_cfg)
    val_dataset = TFTDataset(grouped, val_index, dataset_cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    feature_config = FeatureConfig(
        hist_cont=len(OBS_CONT_COLS),
        fut_cont=len(FUT_CONT_COLS),
        static_cont=len(STATIC_CONT_COLS),
        hist_cat=len(HIST_CAT_COLS),
        fut_cat=len(FUT_CAT_COLS),
        static_cat=len(STATIC_CAT_COLS),
    )
    cat_cardinalities = (
        tuple(len(category_maps[col]) for col in HIST_CAT_COLS),
        tuple(len(category_maps[col]) for col in FUT_CAT_COLS),
        tuple(len(category_maps[col]) for col in STATIC_CAT_COLS),
    )
    model_kwargs = {
        "feature_config": feature_config,
        "cat_cardinalities": cat_cardinalities,
        "horizon_count": len(horizons),
        "d_model": args.d_model,
        "dropout": args.dropout,
        "n_heads": args.n_heads,
    }
    model = TFTBackbone(**model_kwargs)
    lightning_module = TFTLightningModule(
        model=model,
        horizons=horizons,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    monitor_horizon = 15 if 15 in horizons else horizons[len(horizons) // 2]
    monitor_metric = f"val_mae_h{monitor_horizon}"

    callbacks = [
        EarlyStopping(monitor=monitor_metric, patience=3, mode="min"),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="tft-{epoch:02d}-{val_mae:.4f}",
            save_top_k=1,
            monitor=monitor_metric,
            mode="min",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        log_every_n_steps=50,
        enable_model_summary=False,
        gradient_clip_val=1.0,
    )
    trainer.fit(lightning_module, train_loader, val_loader)

    ckpt_callback: ModelCheckpoint = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
    if ckpt_callback.best_model_path:
        best_model_instance = TFTBackbone(**model_kwargs)
        lightning_module = TFTLightningModule.load_from_checkpoint(
            ckpt_callback.best_model_path,
            model=best_model_instance,
            horizons=horizons,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
        )

    device = lightning_module.device
    lightning_module.eval()
    mae_per_horizon: list[float] = [0.0 for _ in horizons]
    total_batches = 0
    all_errors = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = lightning_module(batch)
            errors = torch.abs(preds - batch["y"])
            all_errors.append(errors.cpu())
            total_batches += 1
    if all_errors:
        concatenated = torch.cat(all_errors, dim=0)
        mae_tensor = concatenated.mean(dim=0)
        mae_per_horizon = [float(mae_tensor[i].item()) for i in range(len(horizons))]

    report = {
        "horizons": horizons,
        "mae": {str(h): mae for h, mae in zip(horizons, mae_per_horizon)},
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "backup_model_path": str(backup_path) if backup_path else None,
        "metadata": metadata,
    }
    report_path = output_dir / "val_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(horizons, mae_per_horizon, marker="o")
        ax.set_xlabel("Horizon (minutes)")
        ax.set_ylabel("Validation MAE (mph)")
        ax.set_title("TFT Validation MAE by Horizon")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        plt_path = output_dir / "val_plot.png"
        fig.savefig(plt_path, dpi=150)
        plt.close(fig)
    except Exception:
        plt_path = None

    bundle = TFTBundle(
        state_dict={k: v.cpu() for k, v in lightning_module.model.state_dict().items()},
        model_kwargs={
            "feature_config": asdict(feature_config),
            "cat_cardinalities": [list(card) for card in cat_cardinalities],
            "horizon_count": len(horizons),
            "d_model": args.d_model,
            "dropout": args.dropout,
            "n_heads": args.n_heads,
        },
        horizons=horizons,
        encoder_length=encoder_length,
        scalers=scalers,
        category_maps=category_maps,
        residual_mae={str(h): mae for h, mae in zip(horizons, mae_per_horizon)},
        feature_lists={
            "obs_cont": OBS_CONT_COLS,
            "fut_cont": FUT_CONT_COLS,
            "static_cont": STATIC_CONT_COLS,
            "hist_cat": HIST_CAT_COLS,
            "fut_cat": FUT_CAT_COLS,
            "static_cat": STATIC_CAT_COLS,
        },
        metadata={
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "val_report_path": str(report_path),
            "val_plot_path": str(plt_path) if plt_path else None,
            "freq_minutes": freq_minutes,
            "dataset_meta": metadata,
        },
    )

    bundle_path = output_dir / "tft_bundle.pt"
    torch.save(bundle, bundle_path)
    print(f"Training complete. Bundle saved to {bundle_path}")
    print(f"Validation report: {report_path}")
    if plt_path:
        print(f"Validation plot: {plt_path}")


if __name__ == "__main__":
    main()
