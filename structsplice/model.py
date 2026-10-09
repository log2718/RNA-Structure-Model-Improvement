"""A single configurable bidirectional-LSTM PSI regressor.

One class serves all three feature configurations; only ``input_dim`` changes
(4, 7 or 12). Everything after the input projection, including parameter
shapes, initialisation order and dropout placement, is identical across
configurations, so a difference in test score can be attributed to the input
features rather than to model capacity.

The one unavoidable asymmetry is the input projection itself: it holds
``input_dim * 32`` weights, so configuration C carries 256 more parameters than
A out of roughly 66k. That is a 0.4% difference in a layer that only mixes
channels, and it is reported by :meth:`BiLSTMPSIRegressor.parameter_count`.
Feature scales are already comparable without normalisation: one-hot channels
are exactly 0 or 1 and RNAplfold accessibility channels are probabilities in
``[0, 1]``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import input_dim as config_input_dim

POOLING_MODES = ("mean", "max", "last")


class BiLSTMPSIRegressor(nn.Module):
    """Bidirectional LSTM that predicts one PSI value per sequence.

    Architecture:

    1. Linear input projection ``input_dim -> proj_dim`` with ReLU and dropout.
    2. One bidirectional LSTM layer, ``hidden_size`` units per direction.
    3. Masked sequence-level pooling over the ``2 * hidden_size`` outputs.
    4. Head ``2 * hidden_size -> head_dim -> 1``, i.e. 128 -> 64 -> 1 at the
       defaults, with ReLU and dropout between.

    ``forward`` returns a **logit** per sequence. PSI is
    ``sigmoid(logit)``, which lies strictly in ``(0, 1)``; use
    :meth:`predict_psi`. Keeping the logit as the primary output is what lets
    the loss use the numerically stable ``BCEWithLogitsLoss``.
    """

    def __init__(
        self,
        input_dim: int,
        proj_dim: int = 32,
        hidden_size: int = 64,
        head_dim: int = 64,
        dropout: float = 0.2,
        pooling: str = "mean",
        num_layers: int = 1,
    ) -> None:
        """Build the model.

        Args:
            input_dim: Channels per nucleotide: 4 (A), 7 (B) or 12 (C).
            proj_dim: Width of the input projection.
            hidden_size: LSTM units per direction.
            head_dim: Hidden width of the prediction head.
            dropout: Dropout probability, applied after the projection, after
                pooling and inside the head.
            pooling: ``"mean"``, ``"max"`` or ``"last"``. All three respect the
                padding mask.
            num_layers: Number of stacked LSTM layers.

        Raises:
            ValueError: On a non-positive dimension, a dropout outside
                ``[0, 1)``, or an unknown pooling mode.
        """
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("proj_dim", proj_dim),
            ("hidden_size", hidden_size),
            ("head_dim", head_dim),
            ("num_layers", num_layers),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must lie in [0, 1); got {dropout}")
        if pooling not in POOLING_MODES:
            raise ValueError(
                f"unknown pooling {pooling!r}; expected one of "
                f"{list(POOLING_MODES)}"
            )

        self.hparams = {
            "input_dim": input_dim,
            "proj_dim": proj_dim,
            "hidden_size": hidden_size,
            "head_dim": head_dim,
            "dropout": dropout,
            "pooling": pooling,
            "num_layers": num_layers,
        }
        self.pooling = pooling

        self.input_proj = nn.Linear(input_dim, proj_dim)
        self.input_act = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        pooled_dim = 2 * hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    @classmethod
    def from_config(cls, config: str, **kwargs) -> "BiLSTMPSIRegressor":
        """Build the model for a named feature configuration.

        Args:
            config: ``"A"``, ``"B"`` or ``"C"``.
            **kwargs: Overrides forwarded to ``__init__``.

        Returns:
            A model whose ``input_dim`` matches the configuration.
        """
        return cls(input_dim=config_input_dim(config), **kwargs)

    def parameter_count(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map a padded batch of feature sequences to one logit each.

        Args:
            features: ``(batch, max_len, input_dim)`` float tensor.
            mask: ``(batch, max_len)`` bool tensor, ``True`` at real positions.
                When ``None`` every position is treated as real.
            lengths: ``(batch,)`` true lengths. Derived from ``mask`` when
                omitted. Supplying them enables ``pack_padded_sequence``, which
                stops the LSTM from consuming padding at all.

        Returns:
            ``(batch,)`` float tensor of logits.

        Raises:
            ValueError: If ``features`` is not 3-D, its channel count differs
                from ``input_dim``, or ``mask`` has a row with no real
                positions.
        """
        if features.dim() != 3:
            raise ValueError(
                f"features must be (batch, length, channels); got shape "
                f"{tuple(features.shape)}"
            )
        expected = self.hparams["input_dim"]
        if features.shape[-1] != expected:
            raise ValueError(
                f"model was built for input_dim={expected} but received "
                f"{features.shape[-1]} channels"
            )

        batch, max_len, _ = features.shape
        if mask is None:
            mask = torch.ones(
                (batch, max_len), dtype=torch.bool, device=features.device
            )
        if lengths is None:
            lengths = mask.sum(dim=1)
        if int(lengths.min()) <= 0:
            raise ValueError("every sequence in a batch must have length >= 1")

        # Zero out padding before the projection so the ReLU bias cannot leak
        # a non-zero activation into padded positions.
        projected = self.input_dropout(
            self.input_act(self.input_proj(features))
        ) * mask.unsqueeze(-1)

        packed = pack_padded_sequence(
            projected,
            lengths.detach().to("cpu", dtype=torch.int64),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=max_len
        )

        pooled = self._pool(outputs, mask, lengths)
        return self.head(pooled).squeeze(-1)

    def _pool(
        self,
        outputs: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce per-position LSTM outputs to one vector per sequence.

        Args:
            outputs: ``(batch, max_len, 2 * hidden_size)``.
            mask: ``(batch, max_len)`` bool, ``True`` at real positions.
            lengths: ``(batch,)`` true lengths.

        Returns:
            ``(batch, 2 * hidden_size)``.
        """
        mask_f = mask.unsqueeze(-1).to(outputs.dtype)

        if self.pooling == "mean":
            summed = (outputs * mask_f).sum(dim=1)
            denom = lengths.to(outputs.dtype).clamp(min=1).unsqueeze(-1)
            return summed / denom

        if self.pooling == "max":
            very_negative = torch.finfo(outputs.dtype).min
            masked = outputs.masked_fill(~mask.unsqueeze(-1), very_negative)
            return masked.max(dim=1).values

        # "last": concatenate the forward output at the final real position
        # with the backward output at position 0.
        hidden = outputs.shape[-1] // 2
        last_idx = (lengths - 1).clamp(min=0)
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, outputs.shape[-1])
        last_step = outputs.gather(1, gather_idx).squeeze(1)
        return torch.cat([last_step[:, :hidden], outputs[:, 0, hidden:]], dim=-1)

    @torch.no_grad()
    def predict_psi(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return PSI predictions in ``(0, 1)``.

        Args:
            features: ``(batch, max_len, input_dim)``.
            mask: Optional padding mask.
            lengths: Optional true lengths.

        Returns:
            ``(batch,)`` float tensor of sigmoid-transformed logits.
        """
        return torch.sigmoid(self.forward(features, mask, lengths))
