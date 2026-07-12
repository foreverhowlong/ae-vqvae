"""Non-autoregressive VQ-VAE for byte-level text compression."""

from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TextVQVAEConfig:
    vocab_size: int = 258
    max_seq_len: int = 256
    latent_slots: int = 128
    d_model: int = 448
    n_heads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 6
    decoder_type: Literal["cross_attention", "memory_trunk"] = "cross_attention"
    memory_decoder_latent_layers: int = 4
    memory_decoder_output_layers: int = 2
    ffn_mult: int = 4
    dropout: float = 0.1
    codebook_size: int = 3072
    commitment_beta: float = 0.25
    pad_token_id: int = 257

    def to_dict(self):
        return asdict(self)


@dataclass
class CollapseControlConfig:
    """Engineering controls commonly used to reduce codebook collapse."""

    enabled: bool = False
    use_ema_codebook: bool = False
    ema_decay: float = 0.99
    ema_eps: float = 1e-5
    entropy_weight: float = 0.0
    entropy_temperature: float = 1.0
    diversity_weight: float = 0.0
    code_dropout: float = 0.0
    stochastic_code_sampling: bool = False
    sampling_temperature: float = 0.5
    sampling_topk: int = 8
    dead_code_reset_every: int = 0
    dead_code_reset_usage_threshold: float = 1.0
    normalize_latents: bool = False
    commitment_beta_start: float | None = None
    commitment_beta_warmup_steps: int = 0

    def to_dict(self):
        return asdict(self)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, d_model: int, collapse_config: CollapseControlConfig | None = None):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.collapse_config = collapse_config or CollapseControlConfig()
        self.codebook = nn.Embedding(codebook_size, d_model)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=d_model**-0.5)

        if self.collapse_config.use_ema_codebook:
            self.codebook.weight.requires_grad_(False)
            self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
            self.register_buffer("ema_embed_sum", self.codebook.weight.detach().clone())

    def forward(self, z_e: torch.Tensor, valid_mask: torch.Tensor | None = None):
        flat = z_e.reshape(-1, self.d_model)
        if valid_mask is None:
            valid_mask = torch.ones(z_e.shape[:2], dtype=torch.bool, device=z_e.device)
        elif valid_mask.shape != z_e.shape[:2]:
            raise ValueError(
                f"valid_mask must have shape {z_e.shape[:2]}, got {valid_mask.shape}."
            )
        valid_mask = valid_mask.to(device=z_e.device, dtype=torch.bool)
        flat_valid_mask = valid_mask.reshape(-1)
        weights = self.codebook.weight

        distance_flat = flat
        distance_weights = weights
        if self.collapse_config.normalize_latents:
            distance_flat = F.normalize(distance_flat, dim=-1)
            distance_weights = F.normalize(distance_weights, dim=-1)

        distances = (
            distance_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * distance_flat @ distance_weights.t()
            + distance_weights.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = self._select_codes(distances)
        z_q_raw = self.codebook(indices).view_as(z_e)
        z_q_raw = torch.where(valid_mask.unsqueeze(-1), z_q_raw, z_e.detach())

        if (
            self.training
            and self.collapse_config.use_ema_codebook
            and flat_valid_mask.any()
        ):
            self._ema_update(
                flat[flat_valid_mask].detach(),
                indices[flat_valid_mask].detach(),
            )

        z_q_st = z_e + (z_q_raw - z_e).detach()
        indices = indices.view(z_e.shape[0], z_e.shape[1])
        indices = indices.masked_fill(~valid_mask, -1)
        return {
            "z_q_raw": z_q_raw,
            "z_q_st": z_q_st,
            "indices": indices,
            "distances": distances.view(z_e.shape[0], z_e.shape[1], self.codebook_size),
        }

    def _select_codes(self, distances: torch.Tensor):
        if self.training and self.collapse_config.stochastic_code_sampling:
            k = min(self.collapse_config.sampling_topk, self.codebook_size)
            top_distances, top_indices = torch.topk(distances, k=k, dim=-1, largest=False)
            temperature = max(self.collapse_config.sampling_temperature, 1e-6)
            logits = -top_distances / temperature
            gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-12)).clamp_min(1e-12))
            sampled_pos = (logits + gumbel).argmax(dim=-1)
            indices = top_indices.gather(1, sampled_pos.unsqueeze(1)).squeeze(1)
        else:
            indices = distances.argmin(dim=-1)

        if self.training and self.collapse_config.code_dropout > 0:
            drop_mask = torch.rand(indices.shape, device=indices.device) < self.collapse_config.code_dropout
            random_indices = torch.randint(0, self.codebook_size, indices.shape, device=indices.device)
            indices = torch.where(drop_mask, random_indices, indices)

        return indices

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, indices: torch.Tensor):
        one_hot = F.one_hot(indices, num_classes=self.codebook_size).type_as(flat)
        cluster_size = one_hot.sum(dim=0)
        embed_sum = one_hot.t() @ flat

        decay = self.collapse_config.ema_decay
        self.ema_cluster_size.mul_(decay).add_(cluster_size, alpha=1 - decay)
        self.ema_embed_sum.mul_(decay).add_(embed_sum, alpha=1 - decay)

        n = self.ema_cluster_size.sum()
        smoothed_cluster_size = (
            (self.ema_cluster_size + self.collapse_config.ema_eps)
            / (n + self.codebook_size * self.collapse_config.ema_eps)
            * n.clamp_min(1.0)
        )
        normalized_embed = self.ema_embed_sum / smoothed_cluster_size.unsqueeze(1).clamp_min(1e-12)
        if self.collapse_config.normalize_latents:
            normalized_embed = F.normalize(normalized_embed, dim=-1)
        self.codebook.weight.data.copy_(normalized_embed)

    @torch.no_grad()
    def reset_dead_codes(
        self,
        z_e: torch.Tensor,
        usage_threshold: float = 1.0,
        valid_mask: torch.Tensor | None = None,
    ):
        flat = z_e.reshape(-1, self.d_model).detach()
        if valid_mask is not None:
            flat = flat[valid_mask.reshape(-1).to(device=flat.device, dtype=torch.bool)]
        if flat.numel() == 0:
            return 0

        if hasattr(self, "ema_cluster_size"):
            dead_mask = self.ema_cluster_size <= usage_threshold
        else:
            return 0

        dead_indices = dead_mask.nonzero(as_tuple=False).flatten()
        if dead_indices.numel() == 0:
            return 0

        sample_indices = torch.randint(0, flat.shape[0], (dead_indices.numel(),), device=flat.device)
        replacements = flat[sample_indices]
        if self.collapse_config.normalize_latents:
            replacements = F.normalize(replacements, dim=-1)

        self.codebook.weight.data[dead_indices] = replacements
        self.ema_embed_sum.data[dead_indices] = replacements
        self.ema_cluster_size.data[dead_indices] = usage_threshold + 1.0
        return int(dead_indices.numel())


class TextDecoder(nn.Module):
    """Common interface for parallel text decoders."""

    def forward(self, memory: torch.Tensor, seq_len: int) -> torch.Tensor:
        raise NotImplementedError


class CrossAttentionTextDecoder(TextDecoder):
    """Original position-query decoder, kept as the compatibility path."""

    def __init__(self, config: TextVQVAEConfig):
        super().__init__()
        ffn_dim = config.d_model * config.ffn_mult
        self.max_seq_len = config.max_seq_len
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, memory: torch.Tensor, seq_len: int) -> torch.Tensor:
        _validate_decode_length(seq_len, self.max_seq_len)
        batch_size = memory.shape[0]
        positions = torch.arange(seq_len, device=memory.device).unsqueeze(0).expand(batch_size, -1)
        queries = self.position_embedding(positions)
        return self.norm(self.transformer(tgt=queries, memory=memory))


class RotarySelfAttention(nn.Module):
    """Bidirectional self-attention with rotary Q/K position encoding."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, rope_base: float = 10_000.0):
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
            rope_base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
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
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(-2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = hidden.shape
        qkv = self.qkv(hidden).reshape(
            batch_size, seq_len, 3, self.n_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        query = self._apply_rope(query)
        key = self._apply_rope(key)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(batch_size, seq_len, d_model)
        return self.output_dropout(self.output(attended))


class RotaryResidualBlock(nn.Module):
    """Pre-norm Transformer block used by both memory-trunk stages."""

    def __init__(self, config: TextVQVAEConfig):
        super().__init__()
        ffn_dim = config.d_model * config.ffn_mult
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = RotarySelfAttention(config.d_model, config.n_heads, config.dropout)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ffn_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.ffn(self.ffn_norm(hidden))


class SubPixelSequenceUpsampler(nn.Module):
    """Increase sequence length by projecting channels and rearranging them into slots."""

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
            batch_size, latent_slots * self.upscale_factor, self.d_model
        )


class MemoryTrunkTextDecoder(TextDecoder):
    """Use quantized memory itself as the decoder residual stream."""

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
            RotaryResidualBlock(config) for _ in range(config.memory_decoder_latent_layers)
        )
        self.upsampler = SubPixelSequenceUpsampler(
            config.d_model, config.max_seq_len // config.latent_slots
        )
        self.output_blocks = nn.ModuleList(
            RotaryResidualBlock(config) for _ in range(config.memory_decoder_output_layers)
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


def _validate_decode_length(seq_len: int, max_seq_len: int) -> None:
    if not 0 < seq_len <= max_seq_len:
        raise ValueError(f"seq_len must be in [1, {max_seq_len}], got {seq_len}.")


def pad_aware_adaptive_pool1d(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    output_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average valid tokens in adaptive-pooling bins and zero fully padded bins."""
    if hidden.ndim != 3:
        raise ValueError(f"hidden must have shape (batch, sequence, channels), got {hidden.shape}.")
    if attention_mask.shape != hidden.shape[:2]:
        raise ValueError(
            f"attention_mask must have shape {hidden.shape[:2]}, got {attention_mask.shape}."
        )
    if output_size < 1:
        raise ValueError(f"output_size must be positive, got {output_size}.")

    valid_tokens = attention_mask.to(device=hidden.device, dtype=torch.bool)
    masked_hidden = torch.where(
        valid_tokens.unsqueeze(-1),
        hidden,
        torch.zeros((), device=hidden.device, dtype=hidden.dtype),
    )
    pooled_with_pad_denominator = F.adaptive_avg_pool1d(
        masked_hidden.transpose(1, 2), output_size
    ).transpose(1, 2)
    valid_fraction = F.adaptive_avg_pool1d(
        valid_tokens.to(hidden.dtype).unsqueeze(1), output_size
    ).squeeze(1)
    latent_mask = valid_fraction > 0
    pooled = pooled_with_pad_denominator / valid_fraction.clamp_min(1e-12).unsqueeze(-1)
    pooled = torch.where(latent_mask.unsqueeze(-1), pooled, torch.zeros_like(pooled))
    return pooled, latent_mask


class TextVQVAE(nn.Module):
    """Compresses a byte sequence into discrete latent slots and decodes in parallel."""

    def __init__(self, config: TextVQVAEConfig, collapse_config: CollapseControlConfig | None = None):
        super().__init__()
        self.config = config
        self.collapse_config = collapse_config or CollapseControlConfig()
        ffn_dim = config.d_model * config.ffn_mult

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder_pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            enable_nested_tensor=False,
        )
        self.encoder_norm = nn.LayerNorm(config.d_model)

        self.latent_proj = nn.Linear(config.d_model, config.d_model)
        self.quantizer = VectorQuantizer(config.codebook_size, config.d_model, self.collapse_config)

        decoder_types: dict[str, type[TextDecoder]] = {
            "cross_attention": CrossAttentionTextDecoder,
            "memory_trunk": MemoryTrunkTextDecoder,
        }
        try:
            decoder_class = decoder_types[config.decoder_type]
        except KeyError as exc:
            choices = ", ".join(sorted(decoder_types))
            raise ValueError(
                f"Unknown decoder_type {config.decoder_type!r}; expected one of: {choices}."
            ) from exc
        self.decoder_impl = decoder_class(config)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_mask: bool = False,
    ):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.token_embedding(input_ids) + self.encoder_pos_embedding(positions)

        if attention_mask is None:
            attention_mask = input_ids != self.config.pad_token_id
        else:
            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        padding_mask = ~attention_mask

        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        hidden = self.encoder_norm(hidden)

        pooled, latent_mask = pad_aware_adaptive_pool1d(
            hidden,
            attention_mask,
            self.config.latent_slots,
        )
        latents = self.latent_proj(pooled)
        # A fully padded segment is represented by a fixed zero vector, including
        # after the projection bias, so it cannot become a trainable PAD prototype.
        latents = torch.where(latent_mask.unsqueeze(-1), latents, torch.zeros_like(latents))
        if return_mask:
            return latents, latent_mask
        return latents

    def decode(self, z_q_st: torch.Tensor, seq_len: int):
        return self.output_head(self.decoder_impl(z_q_st, seq_len))

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load current checkpoints and migrate checkpoints from the original decoder layout."""
        if isinstance(self.decoder_impl, CrossAttentionTextDecoder) and any(
            key.startswith("decoder.") for key in state_dict
        ):
            metadata = getattr(state_dict, "_metadata", None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata
            legacy_prefixes = {
                "decoder_pos_embedding.": "decoder_impl.position_embedding.",
                "decoder.": "decoder_impl.transformer.",
                "decoder_norm.": "decoder_impl.norm.",
            }
            for key in list(state_dict):
                for old_prefix, new_prefix in legacy_prefixes.items():
                    if key.startswith(old_prefix):
                        state_dict[new_prefix + key.removeprefix(old_prefix)] = state_dict.pop(key)
                        break
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        z_e, latent_mask = self.encode(
            input_ids,
            attention_mask=attention_mask,
            return_mask=True,
        )
        quantized = self.quantizer(z_e, valid_mask=latent_mask)
        z_q_raw = quantized["z_q_raw"]
        z_q_st = quantized["z_q_st"]
        indices = quantized["indices"]
        logits = self.decode(z_q_st, seq_len=input_ids.shape[1])
        return {
            "logits": logits,
            "z_e": z_e,
            "z_q_raw": z_q_raw,
            "z_q_st": z_q_st,
            "indices": indices,
            "distances": quantized["distances"],
            "latent_mask": latent_mask,
        }


def assignment_entropy_loss(outputs, temperature: float, codebook_size: int):
    distances = outputs["distances"]
    probs = torch.softmax(-distances / max(temperature, 1e-6), dim=-1)
    latent_mask = outputs.get("latent_mask")
    if latent_mask is not None:
        probs = probs[latent_mask]
    else:
        probs = probs.reshape(-1, probs.shape[-1])
    if probs.numel() == 0:
        zero = distances.sum() * 0.0
        return zero, zero.detach()
    avg_probs = probs.mean(dim=0)
    entropy = -(avg_probs * torch.log(avg_probs + 1e-12)).sum()
    normalized_entropy = entropy / torch.log(torch.tensor(float(codebook_size), device=distances.device))
    return -normalized_entropy, normalized_entropy.detach()


def codebook_diversity_loss(codebook_weight: torch.Tensor):
    normalized = F.normalize(codebook_weight, dim=-1)
    gram = normalized @ normalized.t()
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=torch.bool)
    off_diag = gram.masked_select(~eye)
    return off_diag.pow(2).mean()


def _masked_vector_mse(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    squared_error = (inputs - targets).pow(2)
    if valid_mask is None:
        return squared_error.mean()
    valid_mask = valid_mask.to(device=inputs.device, dtype=torch.bool)
    if not valid_mask.any():
        return inputs.sum() * 0.0
    return squared_error[valid_mask].mean()


def text_vqvae_losses(
    outputs,
    targets,
    pad_token_id: int,
    beta: float,
    collapse_config: CollapseControlConfig | None = None,
    codebook_weight: torch.Tensor | None = None,
):
    collapse_config = collapse_config or CollapseControlConfig()
    logits = outputs["logits"]
    valid_tokens = targets != pad_token_id
    if valid_tokens.any():
        recon_loss = F.cross_entropy(logits[valid_tokens], targets[valid_tokens])
    else:
        recon_loss = logits.sum() * 0.0
    z_e = outputs["z_e"]
    z_q_raw = outputs["z_q_raw"]
    latent_mask = outputs.get("latent_mask")
    if collapse_config.use_ema_codebook:
        codebook_loss = torch.zeros((), device=z_e.device)
    else:
        codebook_loss = _masked_vector_mse(z_q_raw, z_e.detach(), latent_mask)
    commitment_loss = _masked_vector_mse(z_e, z_q_raw.detach(), latent_mask)
    entropy_loss = torch.zeros((), device=z_e.device)
    normalized_entropy = torch.zeros((), device=z_e.device)
    if collapse_config.entropy_weight > 0:
        entropy_loss, normalized_entropy = assignment_entropy_loss(
            outputs,
            temperature=collapse_config.entropy_temperature,
            codebook_size=outputs["distances"].shape[-1],
        )

    diversity_loss = torch.zeros((), device=z_e.device)
    if collapse_config.diversity_weight > 0 and codebook_weight is not None:
        diversity_loss = codebook_diversity_loss(codebook_weight)

    total = (
        recon_loss
        + codebook_loss
        + beta * commitment_loss
        + collapse_config.entropy_weight * entropy_loss
        + collapse_config.diversity_weight * diversity_loss
    )
    return {
        "total": total,
        "recon": recon_loss,
        "codebook": codebook_loss,
        "commitment": commitment_loss,
        "entropy": entropy_loss,
        "normalized_assignment_entropy": normalized_entropy,
        "diversity": diversity_loss,
    }


@torch.no_grad()
def codebook_stats(
    indices: torch.Tensor,
    codebook_size: int,
    valid_mask: torch.Tensor | None = None,
):
    if valid_mask is None:
        flat = indices.reshape(-1)
    else:
        flat = indices[valid_mask.to(device=indices.device, dtype=torch.bool)]
    flat = flat[flat >= 0].detach().cpu()
    counts = torch.bincount(flat, minlength=codebook_size).float()
    if flat.numel() == 0:
        return {
            "used_codes": 0,
            "dead_codes": codebook_size,
            "utilization": 0.0,
            "codebook_perplexity": 0.0,
            "counts": counts,
        }
    probs = counts / (counts.sum() + 1e-12)
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    perplexity = torch.exp(entropy).item()
    used = int((counts > 0).sum().item())
    return {
        "used_codes": used,
        "dead_codes": int(codebook_size - used),
        "utilization": used / codebook_size,
        "codebook_perplexity": perplexity,
        "counts": counts,
    }


def count_parameters(model: nn.Module):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
