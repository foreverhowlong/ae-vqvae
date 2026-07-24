"""Pluggable parallel text decoders."""

from typing import get_args

import torch
import torch.nn as nn

from common.text_vqvae_config import DecoderType, TextVQVAEConfig
from models.text_layers import (
    RotaryResidualBlock,
    VQGANAttentionBlock,
    vqgans_compression_factor,
)


class TextDecoder(nn.Module):
    """Convert quantized latent slots into a full-resolution hidden sequence."""

    def forward(self, memory: torch.Tensor, seq_len: int) -> torch.Tensor:
        raise NotImplementedError


class CrossAttentionTextDecoder(TextDecoder):
    """Learned position queries cross-attending to quantized slots."""

    def __init__(self, config: TextVQVAEConfig):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * config.ffn_mult,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, memory: torch.Tensor, seq_len: int) -> torch.Tensor:
        _validate_decode_length(seq_len, self.max_seq_len)
        batch_size = memory.shape[0]
        positions = torch.arange(seq_len, device=memory.device).unsqueeze(0)
        queries = self.position_embedding(positions).expand(batch_size, -1, -1)
        return self.norm(self.transformer(tgt=queries, memory=memory))


class SubPixelSequenceUpsampler(nn.Module):
    """Project channels and rearrange them into additional sequence positions."""

    def __init__(self, d_model: int, upscale_factor: int):
        super().__init__()
        if upscale_factor < 1:
            raise ValueError(f"upscale_factor must be positive, got {upscale_factor}.")
        self.d_model = d_model
        self.upscale_factor = upscale_factor
        self.projection = nn.Linear(d_model, d_model * upscale_factor)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, latent_slots, _ = hidden.shape
        expanded = self.projection(hidden)
        return expanded.reshape(
            batch_size,
            latent_slots * self.upscale_factor,
            self.d_model,
        )


class MemoryTrunkTextDecoder(TextDecoder):
    """Refine quantized slots directly, then sub-pixel upsample them."""

    def __init__(self, config: TextVQVAEConfig):
        super().__init__()
        if config.latent_slots < 1:
            raise ValueError(f"latent_slots must be positive, got {config.latent_slots}.")
        if config.max_seq_len % config.latent_slots != 0:
            raise ValueError(
                "memory_trunk decoder requires max_seq_len to be an integer multiple of "
                f"latent_slots, got {config.max_seq_len} and {config.latent_slots}."
            )
        self.latent_slots = config.latent_slots
        self.max_seq_len = config.max_seq_len
        self.latent_blocks = nn.ModuleList(
            RotaryResidualBlock(config)
            for _ in range(config.memory_decoder_latent_layers)
        )
        self.upsampler = SubPixelSequenceUpsampler(
            config.d_model,
            config.max_seq_len // config.latent_slots,
        )
        self.output_blocks = nn.ModuleList(
            RotaryResidualBlock(config)
            for _ in range(config.memory_decoder_output_layers)
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, memory: torch.Tensor, seq_len: int) -> torch.Tensor:
        _validate_decode_length(seq_len, self.max_seq_len)
        if memory.shape[1] != self.latent_slots:
            raise ValueError(
                f"Expected {self.latent_slots} latent slots, got {memory.shape[1]}."
            )
        hidden = memory
        for block in self.latent_blocks:
            hidden = block(hidden)
        hidden = self.upsampler(hidden)
        for block in self.output_blocks:
            hidden = block(hidden)
        return self.norm(hidden[:, :seq_len])


class VQGANTextDecoder(TextDecoder):
    """Bottleneck attention followed by symmetric transposed convolution."""

    def __init__(self, config: TextVQVAEConfig):
        super().__init__()
        self.latent_slots = config.latent_slots
        self.max_seq_len = config.max_seq_len
        self.compression_factor = vqgans_compression_factor(config)
        self.attention_blocks = nn.ModuleList(
            VQGANAttentionBlock(config) for _ in range(2)
        )
        self.transposed_conv = nn.ConvTranspose1d(
            config.d_model,
            config.d_model,
            kernel_size=self.compression_factor,
            stride=self.compression_factor,
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, memory: torch.Tensor, seq_len: int) -> torch.Tensor:
        _validate_decode_length(seq_len, self.max_seq_len)
        if memory.shape[1] != self.latent_slots:
            raise ValueError(
                f"Expected {self.latent_slots} latent slots, got {memory.shape[1]}."
            )
        hidden = memory
        for block in self.attention_blocks:
            hidden = block(hidden)
        hidden = self.transposed_conv(hidden.transpose(1, 2)).transpose(1, 2)
        return self.norm(hidden[:, :seq_len])


DECODER_REGISTRY: dict[str, type[TextDecoder]] = {
    "cross_attention": CrossAttentionTextDecoder,
    "memory_trunk": MemoryTrunkTextDecoder,
    "vqgans": VQGANTextDecoder,
}
DECODER_TYPES = get_args(DecoderType)


def build_text_decoder(config: TextVQVAEConfig) -> TextDecoder:
    try:
        decoder_class = DECODER_REGISTRY[config.decoder_type]
    except KeyError as exc:
        choices = ", ".join(DECODER_TYPES)
        raise ValueError(
            f"Unknown decoder_type {config.decoder_type!r}; expected one of: {choices}."
        ) from exc
    return decoder_class(config)


def _validate_decode_length(seq_len: int, max_seq_len: int) -> None:
    if not 0 < seq_len <= max_seq_len:
        raise ValueError(f"seq_len must be in [1, {max_seq_len}], got {seq_len}.")
