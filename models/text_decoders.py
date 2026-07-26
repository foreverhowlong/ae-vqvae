"""Pluggable parallel text decoders."""

from typing import get_args

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.text_vqvae_config import DecoderType, TextVQVAEConfig
from models.text_layers import (
    RotaryResidualBlock,
    TextAttnBlock,
    TextResBlock,
    VQGANAttentionBlock,
    vqganr_num_levels,
    vqgans_compression_factor,
    zero_padded_positions,
)


class TextDecoder(nn.Module):
    """Convert quantized latent slots into a full-resolution hidden sequence."""

    accepts_latent_vectors = False
    input_dim: int

    def forward(
        self,
        memory: torch.Tensor,
        seq_len: int,
        *,
        latent_mask: torch.Tensor | None = None,
        output_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class CrossAttentionTextDecoder(TextDecoder):
    """Learned position queries cross-attending to quantized slots."""

    def __init__(self, config: TextVQVAEConfig):
        super().__init__()
        self.input_dim = config.d_model
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

    def forward(
        self,
        memory: torch.Tensor,
        seq_len: int,
        *,
        latent_mask: torch.Tensor | None = None,
        output_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_decode_length(seq_len, self.max_seq_len)
        batch_size = memory.shape[0]
        latent_mask = _resolve_valid_mask(memory, latent_mask, "latent_mask")
        output_mask = _resolve_output_mask(
            batch_size,
            seq_len,
            memory.device,
            output_mask,
        )
        memory = zero_padded_positions(memory, ~latent_mask)
        safe_latent_mask = _ensure_nonempty_attention_mask(latent_mask)
        safe_output_mask = (
            None
            if output_mask is None
            else _ensure_nonempty_attention_mask(output_mask)
        )
        positions = torch.arange(seq_len, device=memory.device).unsqueeze(0)
        queries = self.position_embedding(positions).expand(batch_size, -1, -1)
        hidden = self.transformer(
            tgt=queries,
            memory=memory,
            tgt_key_padding_mask=(
                None if safe_output_mask is None else ~safe_output_mask
            ),
            memory_key_padding_mask=~safe_latent_mask,
        )
        hidden = self.norm(hidden)
        return zero_padded_positions(
            hidden,
            None if output_mask is None else ~output_mask,
        )


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
        self.input_dim = config.d_model
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

    def forward(
        self,
        memory: torch.Tensor,
        seq_len: int,
        *,
        latent_mask: torch.Tensor | None = None,
        output_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_decode_length(seq_len, self.max_seq_len)
        if memory.shape[1] != self.latent_slots:
            raise ValueError(
                f"Expected {self.latent_slots} latent slots, got {memory.shape[1]}."
            )
        latent_mask = _resolve_valid_mask(memory, latent_mask, "latent_mask")
        hidden = zero_padded_positions(memory, ~latent_mask)
        for block in self.latent_blocks:
            hidden = block(hidden, padding_mask=~latent_mask)
            hidden = zero_padded_positions(hidden, ~latent_mask)
        hidden = self.upsampler(hidden)
        valid_mask = latent_mask.repeat_interleave(
            self.upsampler.upscale_factor,
            dim=1,
        )
        valid_mask = _merge_output_mask(
            valid_mask,
            seq_len,
            output_mask,
        )
        hidden = zero_padded_positions(hidden, ~valid_mask)
        for block in self.output_blocks:
            hidden = block(hidden, padding_mask=~valid_mask)
            hidden = zero_padded_positions(hidden, ~valid_mask)
        hidden = self.norm(hidden)
        hidden = zero_padded_positions(hidden, ~valid_mask)
        return hidden[:, :seq_len]


class VQGANTextDecoder(TextDecoder):
    """Bottleneck attention followed by symmetric transposed convolution."""

    def __init__(self, config: TextVQVAEConfig):
        super().__init__()
        self.input_dim = config.d_model
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

    def forward(
        self,
        memory: torch.Tensor,
        seq_len: int,
        *,
        latent_mask: torch.Tensor | None = None,
        output_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_decode_length(seq_len, self.max_seq_len)
        if memory.shape[1] != self.latent_slots:
            raise ValueError(
                f"Expected {self.latent_slots} latent slots, got {memory.shape[1]}."
            )
        latent_mask = _resolve_valid_mask(memory, latent_mask, "latent_mask")
        hidden = zero_padded_positions(memory, ~latent_mask)
        for block in self.attention_blocks:
            hidden = block(hidden, padding_mask=~latent_mask)
            hidden = zero_padded_positions(hidden, ~latent_mask)
        hidden = self.transposed_conv(hidden.transpose(1, 2)).transpose(1, 2)
        valid_mask = latent_mask.repeat_interleave(
            self.compression_factor,
            dim=1,
        )
        valid_mask = _merge_output_mask(valid_mask, seq_len, output_mask)
        hidden = zero_padded_positions(hidden, ~valid_mask)
        hidden = self._postprocess_full_resolution(
            hidden[:, :seq_len],
            padding_mask=~valid_mask[:, :seq_len],
        )
        hidden = self.norm(hidden)
        return zero_padded_positions(hidden, ~valid_mask[:, :seq_len])

    def _postprocess_full_resolution(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return hidden


class VQGANPreAttentionTextDecoder(VQGANTextDecoder):
    """Mirror VQGANPA encoding with attention after transposed convolution."""

    def __init__(self, config: TextVQVAEConfig):
        super().__init__(config)
        self.post_attention = VQGANAttentionBlock(config)

    def _postprocess_full_resolution(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.post_attention(hidden, padding_mask=padding_mask)


class VQGANRUpsampleLevel(nn.Module):
    """One compressed-resolution refinement stage followed by 2x upsampling."""

    def __init__(self, config: TextVQVAEConfig):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            TextResBlock(config)
            for _ in range(config.vqganr_num_res_blocks + 1)
        )
        self.attention = TextAttnBlock(config)
        self.upsample = nn.ConvTranspose1d(
            config.d_model,
            config.d_model,
            kernel_size=2,
            stride=2,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = ~valid_mask
        for block in self.res_blocks:
            hidden = block(hidden, padding_mask=padding_mask)
        hidden = self.attention(hidden, padding_mask=padding_mask)
        hidden = self.upsample(hidden.transpose(1, 2)).transpose(1, 2)
        valid_mask = valid_mask.repeat_interleave(2, dim=1)
        hidden = zero_padded_positions(hidden, ~valid_mask)
        return hidden, valid_mask


class VQGANRTextDecoder(TextDecoder):
    """Hierarchical decoder mirroring VQGANR with extra residual capacity."""

    accepts_latent_vectors = True

    def __init__(self, config: TextVQVAEConfig):
        super().__init__()
        if config.vqganr_num_res_blocks < 1:
            raise ValueError(
                "vqganr_num_res_blocks must be positive, got "
                f"{config.vqganr_num_res_blocks}."
            )
        self.input_dim = config.resolved_latent_dim
        self.latent_slots = config.latent_slots
        self.max_seq_len = config.max_seq_len
        self.num_levels = vqganr_num_levels(config)
        self.conv_in = nn.Conv1d(
            config.resolved_latent_dim,
            config.d_model,
            kernel_size=3,
            padding=1,
        )
        self.mid_res1 = TextResBlock(config)
        self.mid_attention = TextAttnBlock(config)
        self.mid_res2 = TextResBlock(config)
        self.levels = nn.ModuleList(
            VQGANRUpsampleLevel(config) for _ in range(self.num_levels)
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        memory: torch.Tensor,
        seq_len: int,
        *,
        latent_mask: torch.Tensor | None = None,
        output_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_decode_length(seq_len, self.max_seq_len)
        if memory.shape[1] != self.latent_slots:
            raise ValueError(
                f"Expected {self.latent_slots} latent slots, got {memory.shape[1]}."
            )
        latent_mask = _resolve_valid_mask(memory, latent_mask, "latent_mask")
        hidden = zero_padded_positions(memory, ~latent_mask)
        hidden = self.conv_in(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = zero_padded_positions(hidden, ~latent_mask)
        hidden = self.mid_res1(hidden, padding_mask=~latent_mask)
        hidden = self.mid_attention(hidden, padding_mask=~latent_mask)
        hidden = self.mid_res2(hidden, padding_mask=~latent_mask)

        valid_mask = latent_mask
        for level in self.levels:
            hidden, valid_mask = level(hidden, valid_mask)
        if hidden.shape[1] != self.max_seq_len:
            raise RuntimeError(
                "vqganr decoder produced an unexpected sequence length: "
                f"{hidden.shape[1]}, expected {self.max_seq_len}."
            )
        valid_mask = _merge_output_mask(valid_mask, seq_len, output_mask)
        hidden = F.silu(self.norm(hidden))
        hidden = zero_padded_positions(hidden, ~valid_mask)
        return hidden[:, :seq_len]


DECODER_REGISTRY: dict[str, type[TextDecoder]] = {
    "cross_attention": CrossAttentionTextDecoder,
    "memory_trunk": MemoryTrunkTextDecoder,
    "vqgans": VQGANTextDecoder,
    "vqganpa": VQGANPreAttentionTextDecoder,
    "vqganr": VQGANRTextDecoder,
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


def _resolve_valid_mask(
    hidden: torch.Tensor,
    valid_mask: torch.Tensor | None,
    name: str,
) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones(
            hidden.shape[:2],
            dtype=torch.bool,
            device=hidden.device,
        )
    if valid_mask.shape != hidden.shape[:2]:
        raise ValueError(
            f"{name} must have shape {hidden.shape[:2]}, got {valid_mask.shape}."
        )
    return valid_mask.to(device=hidden.device, dtype=torch.bool)


def _resolve_output_mask(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    output_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if output_mask is None:
        return None
    expected_shape = (batch_size, seq_len)
    if output_mask.shape != expected_shape:
        raise ValueError(
            f"output_mask must have shape {expected_shape}, got {output_mask.shape}."
        )
    return output_mask.to(device=device, dtype=torch.bool)


def _merge_output_mask(
    derived_mask: torch.Tensor,
    seq_len: int,
    output_mask: torch.Tensor | None,
) -> torch.Tensor:
    output_mask = _resolve_output_mask(
        derived_mask.shape[0],
        seq_len,
        derived_mask.device,
        output_mask,
    )
    if output_mask is None:
        return derived_mask
    exact_mask = torch.zeros_like(derived_mask)
    exact_mask[:, :seq_len] = output_mask
    return derived_mask & exact_mask


def _ensure_nonempty_attention_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    """Keep PyTorch MHA finite for rows whose real mask contains no valid keys."""
    empty_rows = ~valid_mask.any(dim=1)
    if not empty_rows.any():
        return valid_mask
    safe_mask = valid_mask.clone()
    safe_mask[empty_rows, 0] = True
    return safe_mask
