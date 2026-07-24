"""Shared attention and residual building blocks for text encoders and decoders."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.text_vqvae_config import TextVQVAEConfig


class RotarySelfAttention(nn.Module):
    """Bidirectional self-attention with rotary Q/K position encoding."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        rope_base: float = 10_000.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires an even attention head dimension, got {self.head_dim}."
            )

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.attention_dropout = dropout
        self.output_dropout = nn.Dropout(dropout)
        inv_freq = 1.0 / (
            rope_base
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, tensor: torch.Tensor) -> torch.Tensor:
        seq_len = tensor.shape[2]
        positions = torch.arange(seq_len, device=tensor.device, dtype=torch.float32)
        angles = torch.outer(positions, self.rope_inv_freq.float())
        cos = angles.cos().to(dtype=tensor.dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=tensor.dtype)[None, None, :, :]

        pairs = tensor.reshape(*tensor.shape[:-1], self.head_dim // 2, 2)
        even, odd = pairs.unbind(dim=-1)
        rotated = torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos),
            dim=-1,
        )
        return rotated.flatten(-2)

    def forward(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = hidden.shape
        qkv = self.qkv(hidden).reshape(
            batch_size,
            seq_len,
            3,
            self.n_heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        query = self._apply_rope(query)
        key = self._apply_rope(key)
        attention_mask = _attention_mask(hidden, padding_mask)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch_size,
            seq_len,
            d_model,
        )
        return self.output_dropout(self.output(attended))


class PlainSelfAttention(nn.Module):
    """Bidirectional self-attention without an additional position encoding."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.attention_dropout = dropout
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = hidden.shape
        qkv = self.qkv(hidden).reshape(
            batch_size,
            seq_len,
            3,
            self.n_heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        attention_mask = _attention_mask(hidden, padding_mask)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch_size,
            seq_len,
            d_model,
        )
        return self.output_dropout(self.output(attended))


def _attention_mask(
    hidden: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if padding_mask is None:
        return None
    if padding_mask.shape != hidden.shape[:2]:
        raise ValueError(
            f"padding_mask must have shape {hidden.shape[:2]}, got {padding_mask.shape}."
        )
    valid_keys = ~padding_mask.to(device=hidden.device, dtype=torch.bool)
    return valid_keys[:, None, None, :]


class RotaryResidualBlock(nn.Module):
    """Pre-norm Transformer block with rotary self-attention."""

    def __init__(self, config: TextVQVAEConfig):
        super().__init__()
        ffn_dim = config.d_model * config.ffn_mult
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = RotarySelfAttention(
            config.d_model,
            config.n_heads,
            config.dropout,
        )
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ffn_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = hidden + self.attention(
            self.attention_norm(hidden),
            padding_mask=padding_mask,
        )
        return hidden + self.ffn(self.ffn_norm(hidden))


class VQGANAttentionBlock(nn.Module):
    """Taming-style residual attention block at the compressed resolution."""

    def __init__(self, config: TextVQVAEConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.attention = PlainSelfAttention(
            config.d_model,
            config.n_heads,
            config.dropout,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return hidden + self.attention(
            self.norm(hidden),
            padding_mask=padding_mask,
        )


def vqgans_compression_factor(config: TextVQVAEConfig) -> int:
    if config.latent_slots < 1:
        raise ValueError(f"latent_slots must be positive, got {config.latent_slots}.")
    if config.max_seq_len % config.latent_slots != 0:
        raise ValueError(
            "vqgans requires max_seq_len to be an integer multiple of latent_slots, "
            f"got {config.max_seq_len} and {config.latent_slots}."
        )
    return config.max_seq_len // config.latent_slots
