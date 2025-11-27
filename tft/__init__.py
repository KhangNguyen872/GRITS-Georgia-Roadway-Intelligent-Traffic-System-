"""
Temporal Fusion Transformer stack for multi-horizon congestion forecasting.

This package contains the data pipeline, model definition, training utilities,
and inference helpers for the TFT backend. Existing GBT code paths remain
untouched; clients can opt into TFT by setting the ``PREDICTOR_BACKEND`` environment
variable to ``"TFT"`` and pointing to a trained bundle via ``TFT_BUNDLE_PATH``.
"""

from __future__ import annotations

__all__ = [
    "data_pipeline",
    "model",
    "train",
    "predictor",
]
