"""Segmental BPE VQ-VAE with learned boundaries and an AR decoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.segmental_vqvae_config import SegmentalVQVAEConfig
from common.text_vqvae_config import CollapseControlConfig
from models.text_layers import RotaryResidualBlock, zero_padded_positions
from models.text_vqvae import VectorQuantizer


class BoundaryGater(nn.Module):
    """Predict a boundary after each token from three contextual neighbours."""

    def __init__(self, config: SegmentalVQVAEConfig):
        super().__init__()
        input_dim = 3 * config.d_model
        hidden_dim = config.d_model
        inner_dim = max(config.d_model // 2, 1)
        self.input_norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, 1),
        )

    def forward(self, hidden: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        previous = F.pad(hidden[:, :-1], (0, 0, 1, 0))
        following = F.pad(hidden[:, 1:], (0, 0, 0, 1))
        neighbourhood = torch.cat((previous, hidden, following), dim=-1)
        logits = self.mlp(self.input_norm(neighbourhood)).squeeze(-1)
        return torch.where(valid_mask, logits, torch.zeros_like(logits))


class SegmentPooler(nn.Module):
    """Hard contiguous segmentation with a soft segment-ordinal STE."""

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        hidden: torch.Tensor,
        gate_logits: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        sample_gates: bool,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len, _ = hidden.shape
        valid_mask = valid_mask.to(device=hidden.device, dtype=torch.bool)
        next_valid = F.pad(valid_mask[:, 1:], (0, 1), value=False)
        candidate_mask = valid_mask & next_valid
        final_mask = valid_mask & ~next_valid
        probabilities = torch.sigmoid(gate_logits)

        if sample_gates:
            sampled = torch.bernoulli(probabilities).to(dtype=torch.bool)
        else:
            sampled = probabilities > self.threshold
        hard_boundaries = (sampled & candidate_mask) | final_mask

        # A token belongs to the number of boundaries that occurred before it.
        segment_ids = hard_boundaries.long().cumsum(dim=1) - hard_boundaries.long()
        hard_assignment = F.one_hot(
            segment_ids.clamp(min=0, max=seq_len - 1),
            num_classes=seq_len,
        ).to(dtype=hidden.dtype)
        hard_assignment = hard_assignment * valid_mask.unsqueeze(-1)

        # q[i, k] is the probability that exactly k Bernoulli boundaries have
        # occurred before token i. This is a partition over segment ordinals,
        # not a collection of growing prefixes.
        state = hidden.new_zeros((batch_size, seq_len))
        state[:, 0] = 1.0
        soft_rows = []
        for position in range(seq_len):
            soft_rows.append(state)
            boundary_probability = (
                probabilities[:, position] * candidate_mask[:, position]
            ).unsqueeze(1)
            shifted = F.pad(state[:, :-1], (1, 0))
            state = (
                state * (1.0 - boundary_probability)
                + shifted * boundary_probability
            )
        soft_assignment = torch.stack(soft_rows, dim=1)
        soft_assignment = soft_assignment * valid_mask.unsqueeze(-1)
        assignment = soft_assignment + (hard_assignment - soft_assignment).detach()

        counts = assignment.sum(dim=1)
        sums = torch.einsum("btk,btd->bkd", assignment, hidden)
        pooled = sums / counts.clamp_min(1e-6).unsqueeze(-1)

        chunk_counts = hard_boundaries.sum(dim=1)
        chunk_mask = (
            torch.arange(seq_len, device=hidden.device).unsqueeze(0)
            < chunk_counts.unsqueeze(1)
        )
        pooled = torch.where(
            chunk_mask.unsqueeze(-1),
            pooled,
            torch.zeros_like(pooled),
        )

        lengths = valid_mask.sum(dim=1)
        soft_chunk_counts = 1.0 + (
            probabilities * candidate_mask.to(probabilities.dtype)
        ).sum(dim=1)
        soft_ratio = lengths.to(probabilities.dtype) / soft_chunk_counts.clamp_min(1.0)
        hard_ratio = lengths.to(probabilities.dtype) / chunk_counts.clamp_min(1)
        candidate_logits = gate_logits[candidate_mask]
        if candidate_logits.numel():
            logit_l2 = candidate_logits.square().mean()
        else:
            logit_l2 = gate_logits.sum() * 0.0

        return {
            "pooled": pooled,
            "chunk_mask": chunk_mask,
            "chunk_counts": chunk_counts,
            "hard_boundaries": hard_boundaries,
            "segment_ids": segment_ids,
            "gate_logits": gate_logits,
            "gate_probabilities": probabilities * valid_mask,
            "soft_chunk_counts": soft_chunk_counts,
            "soft_tokens_per_chunk": soft_ratio,
            "hard_tokens_per_chunk": hard_ratio,
            "gate_logit_l2": logit_l2,
        }


class CausalRotarySelfAttention(nn.Module):
    """RoPE self-attention with full-sequence and cached incremental paths."""

    def __init__(self, config: SegmentalVQVAEConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.attention_dropout = config.dropout
        self.output_dropout = nn.Dropout(config.dropout)
        inv_freq = 1.0 / (
            10_000.0
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                / self.head_dim
            )
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def _apply_rope(
        self,
        tensor: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        angles = torch.outer(positions.float(), self.rope_inv_freq.float())
        cos = angles.cos().to(dtype=tensor.dtype)[None, None]
        sin = angles.sin().to(dtype=tensor.dtype)[None, None]
        pairs = tensor.reshape(*tensor.shape[:-1], self.head_dim // 2, 2)
        even, odd = pairs.unbind(dim=-1)
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos),
            dim=-1,
        ).flatten(-2)

    def _project(self, hidden: torch.Tensor):
        batch_size, seq_len, d_model = hidden.shape
        qkv = self.qkv(hidden).reshape(
            batch_size,
            seq_len,
            3,
            self.n_heads,
            self.head_dim,
        )
        return qkv.permute(2, 0, 3, 1, 4).unbind(dim=0), d_model

    def forward(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        (query, key, value), d_model = self._project(hidden)
        seq_len = hidden.shape[1]
        positions = torch.arange(seq_len, device=hidden.device)
        query = self._apply_rope(query, positions)
        key = self._apply_rope(key, positions)
        allowed = torch.ones(
            (seq_len, seq_len),
            device=hidden.device,
            dtype=torch.bool,
        ).tril()
        allowed = allowed[None, None]
        if padding_mask is not None:
            valid_keys = ~padding_mask.to(device=hidden.device, dtype=torch.bool)
            allowed = allowed & valid_keys[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(
            hidden.shape[0], seq_len, d_model
        )
        return self.output_dropout(self.output(attended))

    def step(
        self,
        hidden: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        (query, key, value), d_model = self._project(hidden)
        offset = 0 if cache is None else cache[0].shape[2]
        position = torch.tensor([offset], device=hidden.device)
        query = self._apply_rope(query, position)
        key = self._apply_rope(key, position)
        if cache is not None:
            key = torch.cat((cache[0], key), dim=2)
            value = torch.cat((cache[1], value), dim=2)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).contiguous().reshape(
            hidden.shape[0], 1, d_model
        )
        return self.output_dropout(self.output(attended)), (key, value)


class SegmentalDecoderBlock(nn.Module):
    def __init__(self, config: SegmentalVQVAEConfig):
        super().__init__()
        self.self_norm = nn.LayerNorm(config.d_model)
        self.self_attention = CausalRotarySelfAttention(config)
        self.cross_norm = nn.LayerNorm(config.d_model)
        self.cross_attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * config.ffn_mult),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * config.ffn_mult, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        *,
        target_padding_mask: torch.Tensor | None,
        memory_mask: torch.Tensor,
        disable_cross_attention: bool,
    ) -> torch.Tensor:
        hidden = hidden + self.self_attention(
            self.self_norm(hidden),
            target_padding_mask,
        )
        hidden = zero_padded_positions(hidden, target_padding_mask)
        if not disable_cross_attention:
            query = self.cross_norm(hidden)
            attended, _ = self.cross_attention(
                query,
                memory,
                memory,
                key_padding_mask=~memory_mask,
                need_weights=False,
            )
            hidden = hidden + self.cross_dropout(attended)
            hidden = zero_padded_positions(hidden, target_padding_mask)
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return zero_padded_positions(hidden, target_padding_mask)

    def step(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attended, cache = self.self_attention.step(self.self_norm(hidden), cache)
        hidden = hidden + attended
        query = self.cross_norm(hidden)
        attended, _ = self.cross_attention(
            query,
            memory,
            memory,
            key_padding_mask=~memory_mask,
            need_weights=False,
        )
        hidden = hidden + self.cross_dropout(attended)
        return hidden + self.ffn(self.ffn_norm(hidden)), cache


class SegmentalAutoregressiveDecoder(nn.Module):
    def __init__(self, config: SegmentalVQVAEConfig):
        super().__init__()
        self.config = config
        self.memory_projection = nn.Linear(config.latent_dim, config.d_model)
        self.chunk_position_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model,
        )
        self.memory_norm = nn.LayerNorm(config.d_model)
        self.layers = nn.ModuleList(
            SegmentalDecoderBlock(config) for _ in range(config.decoder_layers)
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def prepare_memory(
        self,
        latents: torch.Tensor,
        latent_mask: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.arange(latents.shape[1], device=latents.device)
        memory = self.memory_projection(latents)
        memory = memory + self.chunk_position_embedding(positions)[None]
        memory = self.memory_norm(memory)
        return torch.where(
            latent_mask.unsqueeze(-1),
            memory,
            torch.zeros_like(memory),
        )

    def forward(
        self,
        token_embeddings: torch.Tensor,
        memory: torch.Tensor,
        *,
        target_padding_mask: torch.Tensor | None,
        memory_mask: torch.Tensor,
        disable_cross_attention: bool = False,
    ) -> torch.Tensor:
        hidden = token_embeddings
        for layer in self.layers:
            hidden = layer(
                hidden,
                memory,
                target_padding_mask=target_padding_mask,
                memory_mask=memory_mask,
                disable_cross_attention=disable_cross_attention,
            )
        return self.output_norm(hidden)

    def step(
        self,
        token_embedding: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor] | None],
    ) -> tuple[
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        hidden = token_embedding
        next_caches = []
        for layer, cache in zip(self.layers, caches, strict=True):
            hidden, cache = layer.step(hidden, memory, memory_mask, cache)
            next_caches.append(cache)
        return self.output_norm(hidden), next_caches


class MonotonicDecoderBlock(nn.Module):
    """Causal decoder block conditioned only on the currently consumed latent."""

    def __init__(self, config: SegmentalVQVAEConfig):
        super().__init__()
        self.self_norm = nn.LayerNorm(config.d_model)
        self.self_attention = CausalRotarySelfAttention(config)
        self.condition_norm = nn.LayerNorm(config.d_model)
        self.condition_gate = nn.Linear(2 * config.d_model, config.d_model)
        self.condition_value = nn.Linear(config.d_model, config.d_model)
        self.condition_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * config.ffn_mult),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * config.ffn_mult, config.d_model),
            nn.Dropout(config.dropout),
        )

    def _condition(
        self,
        hidden: torch.Tensor,
        current_memory: torch.Tensor,
    ) -> torch.Tensor:
        query = self.condition_norm(hidden)
        gate = torch.sigmoid(
            self.condition_gate(torch.cat((query, current_memory), dim=-1))
        )
        update = gate * self.condition_value(current_memory)
        return hidden + self.condition_dropout(update)

    def forward(
        self,
        hidden: torch.Tensor,
        current_memory: torch.Tensor,
        *,
        target_padding_mask: torch.Tensor | None,
        disable_latent_conditioning: bool,
    ) -> torch.Tensor:
        hidden = hidden + self.self_attention(
            self.self_norm(hidden),
            target_padding_mask,
        )
        hidden = zero_padded_positions(hidden, target_padding_mask)
        if not disable_latent_conditioning:
            hidden = self._condition(hidden, current_memory)
            hidden = zero_padded_positions(hidden, target_padding_mask)
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return zero_padded_positions(hidden, target_padding_mask)

    def step(
        self,
        hidden: torch.Tensor,
        current_memory: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attended, cache = self.self_attention.step(self.self_norm(hidden), cache)
        hidden = self._condition(hidden + attended, current_memory)
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden, cache


class MonotonicAutoregressiveDecoder(nn.Module):
    """AR decoder that consumes exactly one latent at each token position."""

    def __init__(self, config: SegmentalVQVAEConfig):
        super().__init__()
        self.config = config
        self.memory_projection = nn.Linear(config.latent_dim, config.d_model)
        self.chunk_position_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model,
        )
        self.local_position_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model,
        )
        self.memory_norm = nn.LayerNorm(config.d_model)
        self.layers = nn.ModuleList(
            MonotonicDecoderBlock(config) for _ in range(config.decoder_layers)
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def prepare_memory(
        self,
        latents: torch.Tensor,
        latent_mask: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.arange(latents.shape[1], device=latents.device)
        memory = self.memory_projection(latents)
        memory = memory + self.chunk_position_embedding(positions)[None]
        memory = self.memory_norm(memory)
        return torch.where(
            latent_mask.unsqueeze(-1),
            memory,
            torch.zeros_like(memory),
        )

    def local_positions(self, positions: torch.Tensor) -> torch.Tensor:
        return self.local_position_embedding(
            positions.clamp(min=0, max=self.config.max_seq_len - 1)
        )

    def forward(
        self,
        token_embeddings: torch.Tensor,
        current_memory: torch.Tensor,
        local_positions: torch.Tensor,
        *,
        target_padding_mask: torch.Tensor | None,
        disable_latent_conditioning: bool = False,
    ) -> torch.Tensor:
        hidden = token_embeddings
        if not disable_latent_conditioning:
            hidden = hidden + self.local_positions(local_positions)
        for layer in self.layers:
            hidden = layer(
                hidden,
                current_memory,
                target_padding_mask=target_padding_mask,
                disable_latent_conditioning=disable_latent_conditioning,
            )
        return self.output_norm(hidden)

    def step(
        self,
        token_embedding: torch.Tensor,
        current_memory: torch.Tensor,
        local_positions: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor] | None],
    ) -> tuple[
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        hidden = token_embedding + self.local_positions(local_positions).unsqueeze(1)
        next_caches = []
        for layer, cache in zip(self.layers, caches, strict=True):
            hidden, cache = layer.step(hidden, current_memory, cache)
            next_caches.append(cache)
        return self.output_norm(hidden), next_caches


class DecoderBoundaryHead(nn.Module):
    """Predict whether the emitted token finishes the currently consumed chunk."""

    def __init__(self, config: SegmentalVQVAEConfig):
        super().__init__()
        input_dim = 4 * config.d_model
        hidden_dim = max(config.d_model // 2, 1)
        self.input_norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        emitted_token_embedding: torch.Tensor,
        current_memory: torch.Tensor,
        local_position_embedding: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (
                hidden,
                emitted_token_embedding,
                current_memory,
                local_position_embedding,
            ),
            dim=-1,
        )
        return self.mlp(self.input_norm(features)).squeeze(-1)


class SegmentalVQVAE(nn.Module):
    """Learn a variable-rate sequence of VQ codes from contiguous BPE chunks."""

    def __init__(self, config: SegmentalVQVAEConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.encoder_layers = nn.ModuleList(
            RotaryResidualBlock(config) for _ in range(config.encoder_layers)
        )
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.gater = BoundaryGater(config)
        self.segment_pooler = SegmentPooler(config.gate_threshold)
        self.latent_projection = nn.Linear(config.d_model, config.latent_dim)
        self.collapse_config = CollapseControlConfig(
            use_ema_codebook=True,
            ema_decay=config.ema_decay,
            ema_eps=config.ema_eps,
        )
        self.quantizer = VectorQuantizer(
            config.codebook_size,
            config.latent_dim,
            self.collapse_config,
        )
        if config.latent_routing == "monotonic_pointer":
            self.decoder = MonotonicAutoregressiveDecoder(config)
            self.boundary_head: DecoderBoundaryHead | None = DecoderBoundaryHead(
                config
            )
        else:
            self.decoder = SegmentalAutoregressiveDecoder(config)
            self.boundary_head = None
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

    def _encode_detailed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        sample_gates: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                "input_ids must have shape [batch, length] with length no greater "
                f"than {self.config.max_seq_len}."
            )
        if attention_mask is None:
            valid_mask = input_ids != self.config.pad_token_id
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids.")
            valid_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        if not valid_mask.any(dim=1).all():
            raise ValueError("Every sequence must contain at least one valid BPE token.")

        hidden = self.token_embedding(input_ids)
        for layer in self.encoder_layers:
            hidden = layer(hidden, padding_mask=~valid_mask)
            hidden = zero_padded_positions(hidden, ~valid_mask)
        hidden = self.encoder_norm(hidden)
        hidden = zero_padded_positions(hidden, ~valid_mask)
        gate_logits = self.gater(hidden, valid_mask)
        segmented = self.segment_pooler(
            hidden,
            gate_logits,
            valid_mask,
            sample_gates=self.training if sample_gates is None else sample_gates,
        )
        latents = self.latent_projection(segmented["pooled"])
        latents = torch.where(
            segmented["chunk_mask"].unsqueeze(-1),
            latents,
            torch.zeros_like(latents),
        )
        return {
            **segmented,
            "z_e": latents,
            "lengths": valid_mask.sum(dim=1),
            "attention_mask": valid_mask,
        }

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_mask: bool = False,
    ):
        encoded = self._encode_detailed(
            input_ids,
            attention_mask,
            sample_gates=False if not self.training else None,
        )
        if return_mask:
            return encoded["z_e"], encoded["chunk_mask"]
        return encoded["z_e"]

    @staticmethod
    def _teacher_local_positions(
        segment_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.arange(
            segment_ids.shape[1],
            device=segment_ids.device,
        ).unsqueeze(0).expand_as(segment_ids)
        starts = torch.zeros_like(attention_mask, dtype=torch.bool)
        starts[:, 0] = attention_mask[:, 0]
        starts[:, 1:] = (
            attention_mask[:, 1:]
            & (segment_ids[:, 1:] != segment_ids[:, :-1])
        )
        segment_starts = torch.where(
            starts,
            positions,
            torch.zeros_like(positions),
        ).cummax(dim=1).values
        local_positions = positions - segment_starts
        return torch.where(
            attention_mask,
            local_positions,
            torch.zeros_like(local_positions),
        )

    @staticmethod
    def _gather_current_memory(
        memory: torch.Tensor,
        pointers: torch.Tensor,
    ) -> torch.Tensor:
        if pointers.ndim == 1:
            indices = pointers[:, None, None].expand(-1, 1, memory.shape[-1])
        elif pointers.ndim == 2:
            indices = pointers.unsqueeze(-1).expand(-1, -1, memory.shape[-1])
        else:
            raise ValueError("pointers must have shape [batch] or [batch, length].")
        return memory.gather(1, indices)

    def _decode_teacher_forced_detailed(
        self,
        latents: torch.Tensor,
        latent_mask: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        segment_ids: torch.Tensor | None,
        disable_cross_attention: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        decoder_inputs = torch.full_like(targets, self.config.pad_token_id)
        decoder_inputs[:, 0] = self.config.bos_token_id
        decoder_inputs[:, 1:] = targets[:, :-1]
        memory = self.decoder.prepare_memory(latents, latent_mask)
        if self.config.latent_routing == "global_cross_attention":
            assert isinstance(self.decoder, SegmentalAutoregressiveDecoder)
            hidden = self.decoder(
                self.token_embedding(decoder_inputs),
                memory,
                target_padding_mask=~attention_mask,
                memory_mask=latent_mask,
                disable_cross_attention=disable_cross_attention,
            )
            return {
                "logits": self.output_head(hidden),
                "decoder_boundary_logits": None,
                "teacher_local_positions": None,
            }

        if segment_ids is None:
            raise ValueError("monotonic_pointer decoding requires segment_ids.")
        assert isinstance(self.decoder, MonotonicAutoregressiveDecoder)
        assert self.boundary_head is not None
        local_positions = self._teacher_local_positions(
            segment_ids,
            attention_mask,
        )
        current_memory = self._gather_current_memory(memory, segment_ids)
        hidden = self.decoder(
            self.token_embedding(decoder_inputs),
            current_memory,
            local_positions,
            target_padding_mask=~attention_mask,
            disable_latent_conditioning=disable_cross_attention,
        )
        boundary_logits = self.boundary_head(
            hidden,
            self.token_embedding(targets),
            current_memory,
            self.decoder.local_positions(local_positions),
        )
        boundary_logits = torch.where(
            attention_mask,
            boundary_logits,
            torch.zeros_like(boundary_logits),
        )
        return {
            "logits": self.output_head(hidden),
            "decoder_boundary_logits": boundary_logits,
            "teacher_local_positions": local_positions,
        }

    def decode_teacher_forced(
        self,
        latents: torch.Tensor,
        latent_mask: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        segment_ids: torch.Tensor | None = None,
        disable_cross_attention: bool = False,
    ) -> torch.Tensor:
        detailed = self._decode_teacher_forced_detailed(
            latents,
            latent_mask,
            targets,
            attention_mask,
            segment_ids=segment_ids,
            disable_cross_attention=disable_cross_attention,
        )
        logits = detailed["logits"]
        assert isinstance(logits, torch.Tensor)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        use_quantizer: bool = True,
        sample_gates: bool | None = None,
    ) -> dict[str, torch.Tensor | bool | None]:
        encoded = self._encode_detailed(
            input_ids,
            attention_mask,
            sample_gates=sample_gates,
        )
        z_e = encoded["z_e"]
        latent_mask = encoded["chunk_mask"]
        if use_quantizer:
            quantized = self.quantizer(z_e, valid_mask=latent_mask)
            z_q_raw = quantized["z_q_raw"]
            z_latent = quantized["z_q_st"]
            indices = quantized["indices"]
        else:
            z_q_raw = z_e
            z_latent = z_e
            indices = torch.full(
                z_e.shape[:2],
                -1,
                device=z_e.device,
                dtype=torch.long,
            )
        decoded = self._decode_teacher_forced_detailed(
            z_latent,
            latent_mask,
            input_ids,
            encoded["attention_mask"],
            segment_ids=encoded["segment_ids"],
        )
        return {
            **encoded,
            **decoded,
            "z_q_raw": z_q_raw,
            "z_q_st": z_latent,
            "z_latent": z_latent,
            "indices": indices,
            "latent_mask": latent_mask,
            "quantizer_active": use_quantizer,
        }

    def _monotonic_pointer_decode(
        self,
        latents: torch.Tensor,
        latent_mask: torch.Tensor,
        *,
        max_length: int,
        teacher_targets: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        assert isinstance(self.decoder, MonotonicAutoregressiveDecoder)
        assert self.boundary_head is not None
        memory = self.decoder.prepare_memory(latents, latent_mask)
        batch_size = latents.shape[0]
        last_pointers = latent_mask.sum(dim=1).long() - 1
        pointers = torch.zeros(batch_size, device=latents.device, dtype=torch.long)
        local_positions = torch.zeros_like(pointers)
        previous = torch.full(
            (batch_size,),
            self.config.bos_token_id,
            device=latents.device,
            dtype=torch.long,
        )
        caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None for _ in self.decoder.layers
        ]
        logits = []
        generated = []
        boundary_logits = []
        boundary_predictions = []
        pointer_trace = []
        local_position_trace = []
        for position in range(max_length):
            if teacher_targets is not None:
                previous = (
                    torch.full_like(previous, self.config.bos_token_id)
                    if position == 0
                    else teacher_targets[:, position - 1]
                )
            current_memory = self._gather_current_memory(memory, pointers)
            hidden, caches = self.decoder.step(
                self.token_embedding(previous).unsqueeze(1),
                current_memory,
                local_positions,
                caches,
            )
            step_logits = self.output_head(hidden[:, 0])
            if teacher_targets is None:
                step_logits = step_logits.clone()
                eos_blocked = pointers < last_pointers
                step_logits[eos_blocked, self.config.eos_token_id] = torch.finfo(
                    step_logits.dtype
                ).min
                emitted = step_logits.argmax(dim=-1)
                active = torch.ones_like(pointers, dtype=torch.bool)
            else:
                emitted = teacher_targets[:, position]
                assert attention_mask is not None
                active = attention_mask[:, position]

            step_boundary_logits = self.boundary_head(
                hidden[:, 0],
                self.token_embedding(emitted),
                current_memory[:, 0],
                self.decoder.local_positions(local_positions),
            )
            predicted_boundary = (
                torch.sigmoid(step_boundary_logits)
                > self.config.decoder_boundary_threshold
            ) & active

            pointer_trace.append(pointers)
            local_position_trace.append(local_positions)
            logits.append(step_logits)
            generated.append(emitted)
            boundary_logits.append(step_boundary_logits)
            boundary_predictions.append(predicted_boundary)

            advance = predicted_boundary & (pointers < last_pointers)
            pointers = pointers + advance.long()
            next_local_positions = torch.where(
                advance,
                torch.zeros_like(local_positions),
                local_positions + 1,
            )
            local_positions = torch.where(
                active,
                next_local_positions,
                local_positions,
            )
            if teacher_targets is None:
                previous = emitted

        return {
            "logits": torch.stack(logits, dim=1),
            "generated": torch.stack(generated, dim=1),
            "boundary_logits": torch.stack(boundary_logits, dim=1),
            "boundary_predictions": torch.stack(boundary_predictions, dim=1),
            "pointer_trace": torch.stack(pointer_trace, dim=1),
            "local_position_trace": torch.stack(local_position_trace, dim=1),
            "final_pointers": pointers,
        }

    @torch.no_grad()
    def decode_with_predicted_pointers(
        self,
        latents: torch.Tensor,
        latent_mask: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.config.latent_routing != "monotonic_pointer":
            raise RuntimeError(
                "Predicted-pointer decoding requires monotonic_pointer routing."
            )
        return self._monotonic_pointer_decode(
            latents,
            latent_mask,
            max_length=targets.shape[1],
            teacher_targets=targets,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def free_running(
        self,
        latents: torch.Tensor,
        latent_mask: torch.Tensor,
        *,
        max_length: int,
        return_details: bool = False,
    ):
        if not 0 < max_length <= self.config.max_seq_len:
            raise ValueError("max_length is outside the configured sequence range.")
        if self.config.latent_routing == "monotonic_pointer":
            details = self._monotonic_pointer_decode(
                latents,
                latent_mask,
                max_length=max_length,
            )
            if return_details:
                return details
            return details["logits"], details["generated"]

        assert isinstance(self.decoder, SegmentalAutoregressiveDecoder)
        memory = self.decoder.prepare_memory(latents, latent_mask)
        batch_size = latents.shape[0]
        previous = torch.full(
            (batch_size,),
            self.config.bos_token_id,
            device=latents.device,
            dtype=torch.long,
        )
        caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None for _ in self.decoder.layers
        ]
        logits = []
        generated = []
        for _ in range(max_length):
            embedded = self.token_embedding(previous).unsqueeze(1)
            hidden, caches = self.decoder.step(
                embedded,
                memory,
                latent_mask,
                caches,
            )
            step_logits = self.output_head(hidden[:, 0])
            previous = step_logits.argmax(dim=-1)
            logits.append(step_logits)
            generated.append(previous)
        details = {
            "logits": torch.stack(logits, dim=1),
            "generated": torch.stack(generated, dim=1),
        }
        if return_details:
            return details
        return details["logits"], details["generated"]


def segmental_vqvae_losses(
    outputs: dict[str, torch.Tensor | bool | None],
    targets: torch.Tensor,
    attention_mask: torch.Tensor,
    config: SegmentalVQVAEConfig,
) -> dict[str, torch.Tensor]:
    valid_mask = attention_mask.to(device=targets.device, dtype=torch.bool)
    logits = outputs["logits"]
    assert isinstance(logits, torch.Tensor)
    reconstruction = F.cross_entropy(logits[valid_mask], targets[valid_mask])

    z_e = outputs["z_e"]
    z_q_raw = outputs["z_q_raw"]
    latent_mask = outputs["latent_mask"]
    assert isinstance(z_e, torch.Tensor)
    assert isinstance(z_q_raw, torch.Tensor)
    assert isinstance(latent_mask, torch.Tensor)
    if bool(outputs["quantizer_active"]):
        commitment_raw = (
            (z_e - z_q_raw.detach()).square()[latent_mask].mean()
        )
    else:
        commitment_raw = reconstruction.new_zeros(())
    commitment = config.commitment_beta * commitment_raw

    soft_ratio = outputs["soft_tokens_per_chunk"]
    gate_logit_l2 = outputs["gate_logit_l2"]
    assert isinstance(soft_ratio, torch.Tensor)
    assert isinstance(gate_logit_l2, torch.Tensor)
    compression_raw = (soft_ratio.mean() - config.compression_target).square()
    compression = config.compression_weight * compression_raw
    gate_regularization = config.gate_logit_l2_weight * gate_logit_l2
    decoder_boundary_logits = outputs.get("decoder_boundary_logits")
    if isinstance(decoder_boundary_logits, torch.Tensor):
        gate_probabilities = outputs["gate_probabilities"]
        assert isinstance(gate_probabilities, torch.Tensor)
        next_valid = F.pad(valid_mask[:, 1:], (0, 1), value=False)
        final_valid = valid_mask & ~next_valid
        boundary_targets = torch.where(
            final_valid,
            torch.ones_like(gate_probabilities),
            gate_probabilities,
        ).detach()
        decoder_boundary_raw = F.binary_cross_entropy_with_logits(
            decoder_boundary_logits[valid_mask],
            boundary_targets[valid_mask].to(decoder_boundary_logits.dtype),
        )
    else:
        decoder_boundary_raw = reconstruction.new_zeros(())
    decoder_boundary = (
        config.decoder_boundary_loss_weight * decoder_boundary_raw
    )
    total = (
        reconstruction
        + commitment
        + compression
        + gate_regularization
        + decoder_boundary
    )
    return {
        "loss": total,
        "reconstruction_loss": reconstruction,
        "commitment_loss": commitment_raw,
        "commitment_weighted_loss": commitment,
        "compression_loss": compression,
        "compression_loss_raw": compression_raw,
        "gate_logit_l2": gate_logit_l2,
        "gate_logit_l2_loss": gate_regularization,
        "decoder_boundary_loss": decoder_boundary_raw,
        "decoder_boundary_weighted_loss": decoder_boundary,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
