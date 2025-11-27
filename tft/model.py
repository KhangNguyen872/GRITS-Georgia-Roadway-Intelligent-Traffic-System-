from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "pytorch_lightning is required for the TFT module. "
        "Install it via `pip install pytorch-lightning`."
    ) from exc


# ---------------------------------------------------------------------------
# Core building blocks


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GatedResidualNetwork(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(d_out, d_out)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(d_out)
        self.skip = d_in == d_out
        self.skip_proj = nn.Linear(d_in, d_out) if not self.skip else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip else self.skip_proj(x)
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        y = self.dropout(y)
        gated = self.sigmoid(self.gate(y))
        y = gated * y + (1 - gated) * residual
        return self.norm(y)


class VariableSelection(nn.Module):
    """
    Lean variable selection: per-time-step soft attention across variables.
    """

    def __init__(self, d_var: int, d_model: int, n_vars: int) -> None:
        super().__init__()
        self.value = nn.Linear(d_var, d_model)
        self.score = nn.Linear(d_var, 1)
        self.softmax = nn.Softmax(dim=2)
        self.n_vars = n_vars

    def forward(self, x_vars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_vars: (B, T, V, D_var)
        scores = self.score(x_vars)  # (B, T, V, 1)
        alpha = self.softmax(scores.squeeze(-1))  # (B, T, V)
        values = self.value(x_vars)  # (B, T, V, D_model)
        fused = torch.sum(alpha.unsqueeze(-1) * values, dim=2)
        return fused, alpha


class StaticEnrichment(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.grn = GatedResidualNetwork(d_model * 2, d_model, d_model)

    def forward(self, temporal: torch.Tensor, static_context: torch.Tensor) -> torch.Tensor:
        # temporal: (B, T, D), static_context: (B, D)
        b, t, d = temporal.shape
        expanded_static = static_context.unsqueeze(1).expand(b, t, d)
        return self.grn(torch.cat([temporal, expanded_static], dim=-1))


class TemporalFusionDecoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.enc_lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.dec_lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.post_attn_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_inp: torch.Tensor, dec_inp: torch.Tensor) -> torch.Tensor:
        enc_out, (h, c) = self.enc_lstm(enc_inp)
        dec_out, _ = self.dec_lstm(dec_inp, (h, c))
        attn_out, _ = self.mha(dec_out, dec_out, dec_out)
        fused = self.post_attn_grn(self.norm(attn_out))
        return fused


# ---------------------------------------------------------------------------
# TFT Backbone


@dataclass
class FeatureConfig:
    hist_cont: int
    fut_cont: int
    static_cont: int
    hist_cat: int
    fut_cat: int
    static_cat: int


class TFTBackbone(nn.Module):
    """
    Lightweight Temporal Fusion Transformer backbone tailored for our feature layout.
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        cat_cardinalities: Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]],
        horizon_count: int,
        d_model: int = 64,
        dropout: float = 0.1,
        n_heads: int = 4,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.feature_config = feature_config
        self.horizon_count = horizon_count
        self.out_dim = out_dim
        self.d_model = d_model

        Vh_c, Vk_c, Vs_c = feature_config.hist_cont, feature_config.fut_cont, feature_config.static_cont
        Vh_cat, Vk_cat, Vs_cat = feature_config.hist_cat, feature_config.fut_cat, feature_config.static_cat

        def make_emb(cardinalities: Sequence[int]) -> nn.ModuleList:
            return nn.ModuleList([nn.Embedding(c, d_model) for c in cardinalities])

        self.hist_cat_emb = make_emb(cat_cardinalities[0]) if Vh_cat > 0 else nn.ModuleList([])
        self.fut_cat_emb = make_emb(cat_cardinalities[1]) if Vk_cat > 0 else nn.ModuleList([])
        self.stat_cat_emb = make_emb(cat_cardinalities[2]) if Vs_cat > 0 else nn.ModuleList([])

        self.hist_cont_proj = nn.Linear(max(Vh_c, 1), d_model)
        self.fut_cont_proj = nn.Linear(max(Vk_c, 1), d_model)
        self.stat_cont_proj = nn.Linear(max(Vs_c, 1), d_model) if Vs_c > 0 else None

        self.hist_vs = VariableSelection(d_model, d_model, max(1, Vh_c + Vh_cat))
        self.fut_vs = VariableSelection(d_model, d_model, max(1, Vk_c + Vk_cat))
        self.stat_vs = VariableSelection(d_model, d_model, max(1, Vs_c + Vs_cat)) if (Vs_c + Vs_cat) > 0 else None

        self.static_grn = StaticEnrichment(d_model)
        self.decoder = TemporalFusionDecoder(d_model, n_heads=n_heads, dropout=dropout)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def _embed_sequence(
        self,
        cont: torch.Tensor,
        cat: Optional[torch.Tensor],
        cont_proj: nn.Linear,
        cat_embeddings: nn.ModuleList,
    ) -> torch.Tensor:
        # cont: (B, T, Vc), cat: (B, T, Vcat)
        cont = cont_proj(cont)
        if cat is None or len(cat_embeddings) == 0:
            return cont.unsqueeze(2)
        embedded_vars = []
        for idx, emb in enumerate(cat_embeddings):
            embedded_vars.append(emb(cat[..., idx]))
        cat_stack = torch.stack(embedded_vars, dim=2)
        return torch.cat([cont.unsqueeze(2), cat_stack], dim=2)

    def _embed_static(
        self,
        cont: Optional[torch.Tensor],
        cat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        components = []
        if self.stat_cont_proj is not None and cont is not None:
            components.append(self.stat_cont_proj(cont))
        if cat is not None and len(self.stat_cat_emb) > 0:
            embedded = [emb(cat[..., idx]) for idx, emb in enumerate(self.stat_cat_emb)]
            components.extend(embedded)
        if not components:
            raise ValueError("Static embedding requires at least one feature.")
        stacked = torch.stack(components, dim=1)  # (B, V, D)
        fused, _ = self.stat_vs(stacked.unsqueeze(1)) if self.stat_vs else (stacked.sum(dim=1, keepdim=True), None)
        return fused.squeeze(1)

    def forward(
        self,
        x_hist_cont: torch.Tensor,
        x_fut_known_cont: torch.Tensor,
        x_static_cont: Optional[torch.Tensor],
        x_hist_cat: Optional[torch.Tensor] = None,
        x_fut_known_cat: Optional[torch.Tensor] = None,
        x_static_cat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Te, _ = x_hist_cont.shape
        _, Tf, _ = x_fut_known_cont.shape

        hist_vars = self._embed_sequence(x_hist_cont, x_hist_cat, self.hist_cont_proj, self.hist_cat_emb)
        fut_vars = self._embed_sequence(x_fut_known_cont, x_fut_known_cat, self.fut_cont_proj, self.fut_cat_emb)

        hist_fused, _ = self.hist_vs(hist_vars)
        fut_fused, _ = self.fut_vs(fut_vars)

        static_ctx = self._embed_static(x_static_cont, x_static_cat)
        enriched_hist = self.static_grn(hist_fused, static_ctx)
        enriched_fut = self.static_grn(fut_fused, static_ctx)

        decoder_out = self.decoder(enriched_hist, enriched_fut)
        y = self.output_head(decoder_out)  # (B, Tf, out_dim)
        return y.squeeze(-1)


# ---------------------------------------------------------------------------
# Lightning module wrapper


class TFTLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: TFTBackbone,
        horizons: Sequence[int],
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.horizons = list(horizons)
        self.save_hyperparameters(ignore=["model"])
        self.val_abs_errors: list[torch.Tensor] = []

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(
            batch["x_hist_cont"],
            batch["x_fut_cont"],
            batch.get("x_static_cont"),
            batch.get("x_hist_cat"),
            batch.get("x_fut_cat"),
            batch.get("x_static_cat"),
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        preds = self(batch)
        target = batch["y"]
        loss = F.l1_loss(preds, target)
        self.log("train_mae", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        preds = self(batch)
        target = batch["y"]
        abs_err = torch.abs(preds - target)
        self.val_abs_errors.append(abs_err.detach())
        return abs_err

    def on_validation_epoch_end(self) -> None:
        if not self.val_abs_errors:
            return
        errors = torch.cat(self.val_abs_errors, dim=0)
        mae_per_h = errors.mean(dim=0)
        for horizon, mae in zip(self.horizons, mae_per_h):
            self.log(f"val_mae_h{horizon}", mae, prog_bar=(horizon == 15), on_epoch=True, sync_dist=False)
        self.val_abs_errors.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


# ---------------------------------------------------------------------------
# Bundle utilities


@dataclass
class TFTBundle:
    """Artifacts persisted after training for inference."""

    state_dict: dict
    model_kwargs: dict
    horizons: Sequence[int]
    encoder_length: int
    scalers: dict
    category_maps: dict
    residual_mae: dict
    feature_lists: dict
    metadata: dict

    def to_device(self, device: Optional[torch.device] = None) -> None:
        # Ensure state dict tensors are moved appropriately when loading manually.
        dev = device or _default_device()
        for key, value in list(self.state_dict.items()):
            if torch.is_tensor(value):
                self.state_dict[key] = value.to(dev)

