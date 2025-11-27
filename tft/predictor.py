from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import tempfile

from .model import FeatureConfig, TFTBackbone, TFTBundle


def _load_bundle(bundle_path: Path) -> TFTBundle:
    if not bundle_path.exists():
        raise FileNotFoundError(f"TFT bundle not found: {bundle_path}")
    try:
        torch.serialization.add_safe_globals([TFTBundle])  # PyTorch >=2.6
    except AttributeError:
        pass  # Older PyTorch without add_safe_globals
    try:
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older PyTorch without weights_only kwarg
        bundle = torch.load(bundle_path, map_location="cpu")
    except pickle.UnpicklingError:
        # Fallback to trusted load if weights_only is unsupported for this bundle.
        try:
            bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        except TypeError:
            bundle = torch.load(bundle_path, map_location="cpu")
    if not isinstance(bundle, TFTBundle):
        raise TypeError(f"Unexpected bundle object type: {type(bundle)}")
    return bundle


def _reconstruct_model(bundle: TFTBundle) -> TFTBackbone:
    kwargs = bundle.model_kwargs.copy()
    feature_config = FeatureConfig(**kwargs.pop("feature_config"))
    cat_cardinalities = kwargs.pop("cat_cardinalities")
    cat_cardinalities = tuple(tuple(card) for card in cat_cardinalities)
    model = TFTBackbone(feature_config=feature_config, cat_cardinalities=cat_cardinalities, **kwargs)
    model.load_state_dict(bundle.state_dict)
    model.eval()
    return model


def _apply_scalers(df: pd.DataFrame, scalers: dict[str, dict[str, float]]) -> pd.DataFrame:
    result = df.copy()
    for col, params in scalers.items():
        norm_col = f"{col}_norm"
        result[norm_col] = (result[col] - params["mean"]) / params["std"]
    return result


@dataclass
class PredictorConfig:
    bundle_path: str
    ga511_csv: Optional[str] = None
    nws_csv: Optional[str] = None
    live_window_hours: int = 72
    verbose: bool = False

    def __post_init__(self) -> None:
        default_bundle = os.getenv("TFT_BUNDLE_PATH", "tft/artifacts/tft_bundle.pt")
        if self.bundle_path is None or str(self.bundle_path).strip() == "":
            self.bundle_path = default_bundle
        self.bundle_path = str(self.bundle_path)
        self.ga511_csv = str(self.ga511_csv) if self.ga511_csv else None
        self.nws_csv = str(self.nws_csv) if self.nws_csv else None


class TFTPredictor:
    def __init__(self, cfg: PredictorConfig | None = None, **kwargs) -> None:
        if cfg is not None and kwargs:
            raise TypeError("Pass either cfg or keyword overrides, not both.")
        if cfg is None:
            params = dict(kwargs)
            bundle_path = params.pop("bundle_path", None)
            resolved_bundle = bundle_path or os.getenv(
                "TFT_BUNDLE_PATH", "tft/artifacts/tft_bundle.pt"
            )
            cfg = PredictorConfig(bundle_path=resolved_bundle, **params)

        self.cfg = cfg
        self.bundle_path = Path(self.cfg.bundle_path)
        self.bundle = _load_bundle(self.bundle_path)
        self.model = _reconstruct_model(self.bundle)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.horizons = list(self.bundle.horizons)
        self._horizon_index = {h: idx for idx, h in enumerate(self.horizons)}
        self._freq_minutes = int(self.bundle.metadata.get("freq_minutes", 5))
        self.horizon_steps = [int(h // self._freq_minutes) for h in self.horizons]
        self.scalers = self.bundle.scalers
        self.feature_lists = self.bundle.feature_lists
        self.category_maps = {
            key: {str(k): int(v) for k, v in mapping.items()}
            for key, mapping in self.bundle.category_maps.items()
        }
        self._offline_cache: Optional[Dict[str, pd.DataFrame]] = None

    # ------------------------------------------------------------------ helpers

    def _log(self, message: str) -> None:
        if self.cfg.verbose:
            print(message)

    def _ensure_offline_cache(self) -> None:
        if self._offline_cache is not None:
            return

        bundle_dir = self.bundle_path.parent
        candidates = [
            bundle_dir / "dataset.parquet",
            bundle_dir / "dataset.csv",
        ]
        dataset_path: Optional[Path] = None
        for candidate in candidates:
            if candidate.exists():
                dataset_path = candidate
                break
        if dataset_path is None:
            raise FileNotFoundError(
                "Offline dataset not found next to bundle. Provide live CSVs or generate dataset.parquet."
            )

        if dataset_path.suffix == ".parquet":
            df = pd.read_parquet(dataset_path)
        else:
            df = pd.read_csv(dataset_path)

        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts_utc"])
        if "tmc" not in df.columns:
            raise ValueError("Offline dataset is missing required 'tmc' column.")
        df["tmc"] = df["tmc"].astype(str)
        df = _apply_scalers(df, self.scalers)
        categorical_cols = (
            self.feature_lists["hist_cat"]
            + self.feature_lists["fut_cat"]
            + self.feature_lists["static_cat"]
        )
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        grouped: Dict[str, pd.DataFrame] = {}
        for tmc, group in df.groupby("tmc"):
            grouped[str(tmc)] = group.sort_values("time_idx").reset_index(drop=True)
        self._offline_cache = grouped

    def _offline_frame(self, tmc: str) -> pd.DataFrame:
        self._ensure_offline_cache()
        assert self._offline_cache is not None
        if tmc not in self._offline_cache:
            raise ValueError(f"No cached offline data for tmc {tmc}")
        return self._offline_cache[tmc]

    def _build_live_frame(
        self,
        tmc: str,
        ts_utc: pd.Timestamp,
    ) -> pd.DataFrame:
        raise NotImplementedError("Live mode is not supported for the NPMRDS pipeline.")

    def _window_csv(
        self,
        source: Path,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        *,
        kind: str,
    ) -> tuple[Path, Optional[Path]]:
        df = pd.read_csv(source)
        if df.empty:
            return source, None

        start = pd.to_datetime(df.get("startTime"), utc=True, errors="coerce")
        end = pd.to_datetime(df.get("endTime"), utc=True, errors="coerce")

        if start.isna().all() and (end.isna().all() if end is not None else True):
            return source, None

        overlap = (start >= window_start) & (start <= window_end)
        if end is not None:
            overlap |= (start <= window_end) & (end >= window_start)
        overlap |= start.isna() & (end.isna() if end is not None else True)

        filtered = df[overlap]
        if filtered.empty or len(filtered) == len(df):
            return source, None

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        filtered.to_csv(tmp.name, index=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        return tmp_path, tmp_path

    def _select_position(self, frame: pd.DataFrame, ts_utc: pd.Timestamp) -> int:
        target_time = ts_utc.floor(f"{self._freq_minutes}min")
        candidates = frame[frame["ts_utc"] <= target_time]
        if candidates.empty:
            candidates = frame
        position = int(candidates.index[-1])
        max_step = max(self.horizon_steps)
        position = min(position, len(frame) - max_step - 1)
        position = max(position, self.bundle.encoder_length)
        return position

    def _build_sample(
        self,
        frame: pd.DataFrame,
        position: int,
    ) -> dict[str, torch.Tensor]:
        enc_len = self.bundle.encoder_length
        if position < enc_len:
            raise ValueError("Insufficient history to build TFT context window.")
        max_step = max(self.horizon_steps)
        if position + max_step >= len(frame):
            raise ValueError("Insufficient future context for requested horizons.")

        enc_slice = frame.iloc[position - enc_len : position]
        base_row = frame.iloc[position]

        obs_cont_cols = [f"{col}_norm" for col in self.feature_lists["obs_cont"]]
        fut_cont_cols = [f"{col}_norm" for col in self.feature_lists["fut_cont"]]
        static_cont_cols = [f"{col}_norm" for col in self.feature_lists["static_cont"]]

        hist_cont = enc_slice[obs_cont_cols].to_numpy(dtype=np.float32)
        hist_cat = None
        if self.feature_lists["hist_cat"]:
            hist_cat = []
            for col in self.feature_lists["hist_cat"]:
                mapping = self.category_maps[col]
                hist_cat.append(
                    enc_slice[col].map(lambda v: mapping.get(str(v), 0)).to_numpy(dtype=np.int64)
                )
            hist_cat = np.stack(hist_cat, axis=1)

        fut_cont_rows = []
        fut_cat_rows = []
        for step in self.horizon_steps:
            future_row = frame.iloc[position + step]
            fut_cont_rows.append(future_row[fut_cont_cols].to_numpy(dtype=np.float32))
            if self.feature_lists["fut_cat"]:
                fut_cat_row = []
                for col in self.feature_lists["fut_cat"]:
                    mapping = self.category_maps[col]
                    fut_cat_row.append(mapping.get(str(future_row[col]), 0))
                fut_cat_rows.append(fut_cat_row)
        fut_cont = np.stack(fut_cont_rows, axis=0)
        fut_cat = np.array(fut_cat_rows, dtype=np.int64) if fut_cat_rows else None

        static_cont = None
        if static_cont_cols:
            static_cont = base_row[static_cont_cols].to_numpy(dtype=np.float32)
        static_cat = None
        if self.feature_lists["static_cat"]:
            static_cat = []
            for col in self.feature_lists["static_cat"]:
                mapping = self.category_maps[col]
                static_cat.append(mapping.get(str(base_row[col]), 0))
            static_cat = np.array(static_cat, dtype=np.int64)

        sample = {
            "x_hist_cont": torch.from_numpy(hist_cont).unsqueeze(0).to(self.device),
            "x_fut_cont": torch.from_numpy(fut_cont).unsqueeze(0).to(self.device),
        }
        if hist_cat is not None:
            sample["x_hist_cat"] = torch.from_numpy(hist_cat).unsqueeze(0).to(self.device)
        if fut_cat is not None:
            sample["x_fut_cat"] = torch.from_numpy(fut_cat).unsqueeze(0).to(self.device)
        if static_cont is not None:
            sample["x_static_cont"] = torch.from_numpy(static_cont).unsqueeze(0).to(self.device)
        if static_cat is not None:
            sample["x_static_cat"] = torch.from_numpy(static_cat).unsqueeze(0).to(self.device)
        return sample

    def predict(
        self,
        tmc: str,
        ts: datetime,
        horizons: Optional[Sequence[int]] = None,
        live: bool = False,
    ) -> Dict[int, float]:
        requested_horizons = list(horizons) if horizons else self.horizons
        invalid = [h for h in requested_horizons if h not in self._horizon_index]
        if invalid:
            raise ValueError(
                f"Requested horizons {invalid} not in trained horizons {self.horizons}"
            )

        ts_timestamp = pd.Timestamp(ts)
        if ts_timestamp.tzinfo is None:
            ts_utc = ts_timestamp.tz_localize("UTC")
        else:
            ts_utc = ts_timestamp.tz_convert("UTC")
        mode = "live" if live else "offline"

        try:
            if not live:
                self._log(f"INFER[offline] bundle={self.bundle_path}")
                frame = self._offline_frame(tmc)
            else:
                frame = self._build_live_frame(tmc, ts_utc)
            position = self._select_position(frame, ts_utc)
            sample = self._build_sample(frame, position)
            preds = self._forward(sample)
            return {int(h): float(preds[self._horizon_index[h]]) for h in requested_horizons}
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError(
                f"TFTPredictor.predict failed ({mode}, tmc={tmc}, ts={ts_utc})"
            ) from exc

    # ------------------------------------------------------------------ runtime utils

    def _forward(self, sample: dict[str, torch.Tensor]) -> np.ndarray:
        with torch.no_grad():
            output = self.model(
                sample["x_hist_cont"],
                sample["x_fut_cont"],
                sample.get("x_static_cont"),
                sample.get("x_hist_cat"),
                sample.get("x_fut_cat"),
                sample.get("x_static_cat"),
            )
        return output.squeeze(0).cpu().numpy()

    def forward_stub(self) -> np.ndarray:
        hist_len = self.bundle.encoder_length
        fut_len = len(self.horizons)

        def zeros(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.zeros(shape, dtype=torch.float32, device=self.device)

        sample: dict[str, torch.Tensor] = {
            "x_hist_cont": zeros((1, hist_len, max(1, len(self.feature_lists["obs_cont"])))),
            "x_fut_cont": zeros((1, fut_len, max(1, len(self.feature_lists["fut_cont"])))),
        }
        if self.feature_lists["hist_cat"]:
            sample["x_hist_cat"] = torch.zeros(
                (1, hist_len, len(self.feature_lists["hist_cat"])), dtype=torch.long, device=self.device
            )
        if self.feature_lists["fut_cat"]:
            sample["x_fut_cat"] = torch.zeros(
                (1, fut_len, len(self.feature_lists["fut_cat"])), dtype=torch.long, device=self.device
            )
        if self.feature_lists["static_cont"]:
            sample["x_static_cont"] = zeros((1, len(self.feature_lists["static_cont"])))
        if self.feature_lists["static_cat"]:
            sample["x_static_cat"] = torch.zeros(
                (1, len(self.feature_lists["static_cat"])), dtype=torch.long, device=self.device
            )
        return self._forward(sample)


def resolve_backend(default: str = "GBT") -> str:
    return os.getenv("PREDICTOR_BACKEND", default).strip().upper() or default.upper()


def load_predictor_if_enabled(cfg: PredictorConfig | None = None) -> Optional[TFTPredictor]:
    if resolve_backend() != "TFT":
        return None
    return TFTPredictor(cfg)
