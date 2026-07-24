"""Pluggable text encoders with one slot-producing interface."""

from typing import get_args

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.text_vqvae_config import EncoderType, TextVQVAEConfig
from models.text_layers import (
    RotaryResidualBlock,
    VQGANAttentionBlock,
    vqgans_compression_factor,
)


class TextEncoder(nn.Module):
    """Convert token embeddings into fixed-size latent slots and a validity mask."""

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class PoolingTextEncoder(TextEncoder):
    """Shared normalization and PAD-aware pooling for full-resolution encoders."""

    def __init__(self, config: TextVQVAEConfig):
        super().__init__()
        self.latent_slots = config.latent_slots
        self.slot_pad_ratio_threshold = config.slot_pad_ratio_threshold
        self.norm = nn.LayerNorm(config.d_model)

    def encode_tokens(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_mask = _validate_attention_mask(hidden, attention_mask)
        encoded = self.encode_tokens(hidden, padding_mask=~attention_mask)
        encoded = self.norm(encoded)
        return pad_aware_adaptive_pool1d(
            encoded,
            attention_mask,
            self.latent_slots,
            slot_pad_ratio_threshold=self.slot_pad_ratio_threshold,
        )


class AbsoluteTextEncoder(PoolingTextEncoder):
    """Transformer encoder with learned absolute position embeddings."""

    def __init__(self, config: TextVQVAEConfig):
        super().__init__(config)
        self.max_seq_len = config.max_seq_len
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * config.ffn_mult,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            enable_nested_tensor=False,
        )

    def encode_tokens(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = hidden.shape[1]
        if not 0 < seq_len <= self.max_seq_len:
            raise ValueError(
                f"encoder sequence length must be in [1, {self.max_seq_len}], got {seq_len}."
            )
        positions = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
        hidden = hidden + self.position_embedding(positions)
        return self.transformer(hidden, src_key_padding_mask=padding_mask)


class RotaryTextEncoder(PoolingTextEncoder):
    """Bidirectional Transformer encoder using RoPE."""

    def __init__(self, config: TextVQVAEConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            RotaryResidualBlock(config) for _ in range(config.encoder_layers)
        )

    def encode_tokens(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden, padding_mask=padding_mask)
        return hidden


class VQGANTextEncoder(TextEncoder):
    """Strided convolution followed by bottleneck attention."""

    def __init__(self, config: TextVQVAEConfig):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.latent_slots = config.latent_slots
        self.slot_pad_ratio_threshold = config.slot_pad_ratio_threshold
        self.compression_factor = vqgans_compression_factor(config)
        self.strided_conv = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=self.compression_factor,
            stride=self.compression_factor,
        )
        self.attention_blocks = nn.ModuleList(
            VQGANAttentionBlock(config) for _ in range(2)
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = hidden.shape
        if not 0 < seq_len <= self.max_seq_len:
            raise ValueError(
                f"encoder sequence length must be in [1, {self.max_seq_len}], got {seq_len}."
            )
        attention_mask = _validate_attention_mask(hidden, attention_mask)
        if seq_len < self.max_seq_len:
            pad_length = self.max_seq_len - seq_len
            hidden = F.pad(hidden, (0, 0, 0, pad_length))
            attention_mask = F.pad(attention_mask, (0, pad_length), value=False)

        hidden = self._preprocess_full_resolution(
            hidden,
            padding_mask=~attention_mask,
        )
        hidden = torch.where(
            attention_mask.unsqueeze(-1),
            hidden,
            torch.zeros((), device=hidden.device, dtype=hidden.dtype),
        )
        hidden = self.strided_conv(hidden.transpose(1, 2)).transpose(1, 2)
        valid_fraction = F.avg_pool1d(
            attention_mask.to(hidden.dtype).unsqueeze(1),
            kernel_size=self.compression_factor,
            stride=self.compression_factor,
        ).squeeze(1)
        latent_mask = (1.0 - valid_fraction) <= self.slot_pad_ratio_threshold
        hidden = torch.where(latent_mask.unsqueeze(-1), hidden, torch.zeros_like(hidden))

        for block in self.attention_blocks:
            hidden = block(hidden, padding_mask=~latent_mask)
        hidden = self.norm(hidden)
        hidden = torch.where(latent_mask.unsqueeze(-1), hidden, torch.zeros_like(hidden))
        if hidden.shape != (batch_size, self.latent_slots, d_model):
            raise RuntimeError(
                "vqgans encoder produced an unexpected shape: "
                f"{hidden.shape}, expected {(batch_size, self.latent_slots, d_model)}."
            )
        return hidden, latent_mask

    def _preprocess_full_resolution(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return hidden


class VQGANPreAttentionTextEncoder(VQGANTextEncoder):
    """Add full-resolution attention before the VQGANS strided convolution."""

    def __init__(self, config: TextVQVAEConfig):
        super().__init__(config)
        self.pre_attention = VQGANAttentionBlock(config)

    def _preprocess_full_resolution(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.pre_attention(hidden, padding_mask=padding_mask)


ENCODER_REGISTRY: dict[str, type[TextEncoder]] = {
    "absolute": AbsoluteTextEncoder,
    "rope": RotaryTextEncoder,
    "vqgans": VQGANTextEncoder,
    "vqganpa": VQGANPreAttentionTextEncoder,
}
ENCODER_TYPES = get_args(EncoderType)


def build_text_encoder(config: TextVQVAEConfig) -> TextEncoder:
    try:
        encoder_class = ENCODER_REGISTRY[config.encoder_type]
    except KeyError as exc:
        choices = ", ".join(ENCODER_TYPES)
        raise ValueError(
            f"Unknown encoder_type {config.encoder_type!r}; expected one of: {choices}."
        ) from exc
    return encoder_class(config)


def _validate_attention_mask(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if attention_mask.shape != hidden.shape[:2]:
        raise ValueError(
            f"attention_mask must have shape {hidden.shape[:2]}, got {attention_mask.shape}."
        )
    return attention_mask.to(device=hidden.device, dtype=torch.bool)


def pad_aware_adaptive_pool1d(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    output_size: int,
    slot_pad_ratio_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average valid tokens in absolute-position bins and mask PAD-heavy bins."""
    if hidden.ndim != 3:
        raise ValueError(
            f"hidden must have shape (batch, sequence, channels), got {hidden.shape}."
        )
    if attention_mask.shape != hidden.shape[:2]:
        raise ValueError(
            f"attention_mask must have shape {hidden.shape[:2]}, got {attention_mask.shape}."
        )
    if output_size < 1:
        raise ValueError(f"output_size must be positive, got {output_size}.")
    if not 0.0 <= slot_pad_ratio_threshold < 1.0:
        raise ValueError(
            "slot_pad_ratio_threshold must be in [0, 1), got "
            f"{slot_pad_ratio_threshold}."
        )

    valid_tokens = attention_mask.to(device=hidden.device, dtype=torch.bool)
    masked_hidden = torch.where(
        valid_tokens.unsqueeze(-1),
        hidden,
        torch.zeros((), device=hidden.device, dtype=hidden.dtype),
    )
    pooled_with_pad_denominator = F.adaptive_avg_pool1d(
        masked_hidden.transpose(1, 2),
        output_size,
    ).transpose(1, 2)
    valid_fraction = F.adaptive_avg_pool1d(
        valid_tokens.to(hidden.dtype).unsqueeze(1),
        output_size,
    ).squeeze(1)
    latent_mask = (1.0 - valid_fraction) <= slot_pad_ratio_threshold
    pooled = pooled_with_pad_denominator / valid_fraction.clamp_min(1e-12).unsqueeze(-1)
    pooled = torch.where(latent_mask.unsqueeze(-1), pooled, torch.zeros_like(pooled))
    return pooled, latent_mask
