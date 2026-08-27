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


class TokenKeepGate(nn.Module):
    """Predict whether each continuous contextual latent should be transmitted."""

    def __init__(self, config: SegmentalVQVAEConfig):
        super().__init__()
        hidden_dim = config.d_model
        inner_dim = max(config.d_model // 2, 1)
        self.network = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, 1),
        )

    def forward(
        self,
        latents: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.network(latents).squeeze(-1)
        return torch.where(valid_mask, logits, torch.zeros_like(logits))


class TokenPruner(nn.Module):
    """Order-preserving hard pruning with a soft packed-sequence STE."""

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    @staticmethod
    def _soft_selection(
        probabilities: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Probability that token t becomes packed code ordinal k."""
        batch_size, seq_len = probabilities.shape
        state = probabilities.new_zeros(batch_size, seq_len)
        state[:, 0] = 1.0
        selections = []
        for position in range(seq_len):
            probability = probabilities[:, position].unsqueeze(1)
            selections.append(state * probability)
            shifted = F.pad(state[:, :-1], (1, 0))
            state = state * (1.0 - probability) + shifted * probability
        return torch.stack(selections, dim=1) * valid_mask.unsqueeze(-1)

    def forward(
        self,
        latents: torch.Tensor,
        gate_logits: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        sample_gates: bool,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len, _ = latents.shape
        valid_mask = valid_mask.to(device=latents.device, dtype=torch.bool)
        next_valid = F.pad(valid_mask[:, 1:], (0, 1), value=False)
        candidate_mask = valid_mask & next_valid
        final_mask = valid_mask & ~next_valid
        raw_probabilities = torch.sigmoid(gate_logits)
        probabilities = (
            raw_probabilities * candidate_mask.to(raw_probabilities.dtype)
            + final_mask.to(raw_probabilities.dtype)
        )
        if sample_gates:
            sampled = torch.bernoulli(raw_probabilities).bool()
        else:
            sampled = raw_probabilities > self.threshold
        keep_mask = (sampled & candidate_mask) | final_mask
        packed_ordinals = keep_mask.long().cumsum(dim=1) - 1
        hard_selection = F.one_hot(
            packed_ordinals.clamp(min=0, max=seq_len - 1),
            num_classes=seq_len,
        ).to(dtype=latents.dtype)
        hard_selection = hard_selection * keep_mask.unsqueeze(-1)
        soft_selection = self._soft_selection(probabilities, valid_mask)
        selection = soft_selection + (
            hard_selection - soft_selection
        ).detach()
        packed = torch.einsum("btk,btd->bkd", selection, latents)

        keep_counts = keep_mask.sum(dim=1)
        packed_mask = (
            torch.arange(seq_len, device=latents.device).unsqueeze(0)
            < keep_counts.unsqueeze(1)
        )
        packed = torch.where(
            packed_mask.unsqueeze(-1),
            packed,
            torch.zeros_like(packed),
        )
        segment_ids = keep_mask.long().cumsum(dim=1) - keep_mask.long()
        lengths = valid_mask.sum(dim=1)
        soft_keep_counts = probabilities.sum(dim=1)
        soft_ratio = lengths.to(latents.dtype) / soft_keep_counts.clamp_min(1.0)
        hard_ratio = lengths.to(latents.dtype) / keep_counts.clamp_min(1)
        candidate_logits = gate_logits[candidate_mask]
        logit_l2 = (
            candidate_logits.square().mean()
            if candidate_logits.numel()
            else gate_logits.sum() * 0.0
        )
        positions = torch.arange(
            seq_len,
            device=latents.device,
            dtype=latents.dtype,
        )
        packed_source_positions = torch.einsum(
            "btk,t->bk",
            hard_selection,
            positions,
        ).long()
        return {
            "pooled": packed,
            "chunk_mask": packed_mask,
            "chunk_counts": keep_counts,
            "hard_boundaries": keep_mask,
            "segment_ids": segment_ids,
            "gate_logits": gate_logits,
            "gate_probabilities": probabilities,
            "soft_chunk_counts": soft_keep_counts,
            "soft_tokens_per_chunk": soft_ratio,
            "hard_tokens_per_chunk": hard_ratio,
            "gate_logit_l2": logit_l2,
            "packed_source_positions": packed_source_positions,
            "token_pruning_active": torch.ones(
                (),
                device=latents.device,
                dtype=torch.bool,
            ),
        }


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


class MaskedBidirectionalRotarySelfAttention(nn.Module):
    """Bidirectional RoPE attention over an explicit sliding-window mask."""

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
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(tensor.shape[0], -1)
        angles = positions.float().unsqueeze(-1) * self.rope_inv_freq.float()
        cos = angles.cos().to(dtype=tensor.dtype)[:, None]
        sin = angles.sin().to(dtype=tensor.dtype)[:, None]
        pairs = tensor.reshape(*tensor.shape[:-1], self.head_dim // 2, 2)
        even, odd = pairs.unbind(dim=-1)
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos),
            dim=-1,
        ).flatten(-2)

    def forward(
        self,
        hidden: torch.Tensor,
        allowed: torch.Tensor,
        positions: torch.Tensor,
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
        query = self._apply_rope(query, positions)
        key = self._apply_rope(key, positions)
        if allowed.ndim != 3 or allowed.shape[:2] != (batch_size, seq_len):
            raise ValueError(
                "allowed must have shape [batch, length, local_window]."
            )
        window_size = allowed.shape[-1]
        if window_size % 2 != 1:
            raise ValueError("The local attention window must have odd width.")
        radius = window_size // 2
        padded_key = F.pad(key, (0, 0, radius, radius))
        padded_value = F.pad(value, (0, 0, radius, radius))
        key_windows = padded_key.unfold(2, window_size, 1).permute(
            0,
            1,
            2,
            4,
            3,
        )
        value_windows = padded_value.unfold(2, window_size, 1).permute(
            0,
            1,
            2,
            4,
            3,
        )
        scores = torch.einsum(
            "bhtd,bhtwd->bhtw",
            query,
            key_windows,
        ) / math.sqrt(self.head_dim)
        safe_allowed = allowed.clone()
        empty_queries = ~safe_allowed.any(dim=-1)
        safe_allowed[:, :, radius] |= empty_queries
        scores = scores.masked_fill(~safe_allowed[:, None], float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        attention = F.dropout(
            attention,
            p=self.attention_dropout,
            training=self.training,
        )
        attended = torch.einsum(
            "bhtw,bhtwd->bhtd",
            attention,
            value_windows,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch_size,
            seq_len,
            d_model,
        )
        return self.output_dropout(self.output(attended))


class MaskedBidirectionalBlock(nn.Module):
    def __init__(self, config: SegmentalVQVAEConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = MaskedBidirectionalRotarySelfAttention(config)
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
        allowed: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = hidden + self.attention(
            self.attention_norm(hidden),
            allowed,
            positions,
        )
        hidden = zero_padded_positions(hidden, padding_mask)
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return zero_padded_positions(hidden, padding_mask)


class LocalBoundaryEncoder(nn.Module):
    """Translation-equivariant encoder with a bounded total receptive field."""

    def __init__(self, config: SegmentalVQVAEConfig):
        super().__init__()
        base_radius, remainder = divmod(
            config.boundary_window_radius,
            config.boundary_encoder_layers,
        )
        self.layer_radii = [
            base_radius + int(index < remainder)
            for index in range(config.boundary_encoder_layers)
        ]
        self.layers = nn.ModuleList(
            MaskedBidirectionalBlock(config)
            for _ in range(config.boundary_encoder_layers)
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = token_embeddings.shape[1]
        positions = torch.arange(seq_len, device=token_embeddings.device)
        hidden = token_embeddings
        for layer, radius in zip(self.layers, self.layer_radii, strict=True):
            valid_windows = F.pad(
                valid_mask,
                (radius, radius),
                value=False,
            ).unfold(1, 2 * radius + 1, 1)
            allowed = (
                valid_mask.unsqueeze(-1)
                & valid_windows
            )
            hidden = layer(
                hidden,
                allowed,
                positions,
                ~valid_mask,
            )
        hidden = self.output_norm(hidden)
        return zero_padded_positions(hidden, ~valid_mask)


class SpanScorer(nn.Module):
    """Score all O(length * max_span_length) locally contextual spans."""

    def __init__(self, config: SegmentalVQVAEConfig):
        super().__init__()
        self.max_span_length = config.max_span_length
        self.d_model = config.d_model
        self.length_embedding = nn.Embedding(
            config.max_span_length + 1,
            config.d_model,
        )
        self.pooling_score = nn.Linear(config.d_model, 1)
        self.input_norm = nn.LayerNorm(8 * config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(8 * config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, max(config.d_model // 2, 1)),
            nn.GELU(),
            nn.Linear(max(config.d_model // 2, 1), 1),
        )
        lengths = torch.arange(
            config.max_span_length + 1,
            dtype=torch.float32,
        )
        scale = max(config.compression_target / 2.0, 1.0)
        initial_bias = -0.5 * (
            (lengths - config.compression_target) / scale
        ).square()
        self.length_bias = nn.Parameter(initial_bias)

    def forward(
        self,
        hidden: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden.shape
        lengths = valid_mask.sum(dim=1)
        salience_logits = self.pooling_score(hidden).squeeze(-1)
        salience_logits = salience_logits.masked_fill(~valid_mask, float("-inf"))
        salience = torch.exp(
            salience_logits
            - salience_logits.masked_fill(
                ~valid_mask,
                torch.finfo(salience_logits.dtype).min,
            ).max(dim=1, keepdim=True).values
        ) * valid_mask
        prefix_weight = F.pad(salience.cumsum(dim=1), (1, 0))
        prefix_weighted_hidden = F.pad(
            (salience.unsqueeze(-1) * hidden).cumsum(dim=1),
            (0, 0, 1, 0),
        )
        scores = hidden.new_full(
            (batch_size, seq_len, self.max_span_length),
            float("-inf"),
        )
        zero = torch.zeros_like(hidden[:, :1])
        for span_length in range(1, self.max_span_length + 1):
            candidate_count = seq_len - span_length + 1
            if candidate_count <= 0:
                break
            span_weight = (
                prefix_weight[:, span_length:]
                - prefix_weight[:, :-span_length]
            )
            span_weighted_hidden = (
                prefix_weighted_hidden[:, span_length:]
                - prefix_weighted_hidden[:, :-span_length]
            )
            pooled = span_weighted_hidden / span_weight.clamp_min(1e-12).unsqueeze(-1)
            left_inside = hidden[:, :candidate_count]
            right_inside = hidden[
                :,
                span_length - 1 : span_length - 1 + candidate_count,
            ]
            left_outside = torch.cat(
                (zero, hidden[:, : max(candidate_count - 1, 0)]),
                dim=1,
            )
            right_outside = torch.cat(
                (hidden[:, span_length:], zero),
                dim=1,
            )[:, :candidate_count]
            length_feature = self.length_embedding.weight[span_length].view(
                1,
                1,
                -1,
            ).expand(batch_size, candidate_count, -1)
            features = torch.cat(
                (
                    left_outside,
                    left_inside,
                    right_inside,
                    right_outside,
                    pooled,
                    right_inside - left_inside,
                    right_inside * left_inside,
                    length_feature,
                ),
                dim=-1,
            )
            span_scores = self.mlp(self.input_norm(features)).squeeze(-1)
            span_scores = span_scores + self.length_bias[span_length]
            starts = torch.arange(candidate_count, device=hidden.device)
            candidate_valid = (
                starts.unsqueeze(0) + span_length <= lengths.unsqueeze(1)
            )
            scores[:, :candidate_count, span_length - 1] = torch.where(
                candidate_valid,
                span_scores,
                torch.full_like(span_scores, float("-inf")),
            )
        return scores


class SemiMarkovSegmenter(nn.Module):
    """Bounded-duration segmental model with Viterbi forward and marginal STE."""

    def __init__(self, config: SegmentalVQVAEConfig):
        super().__init__()
        self.max_span_length = config.max_span_length
        self.compression_target = config.compression_target
        self.fixed_count = (
            config.segmentation_mode == "semi_markov_fixed_count"
        )
        self.scorer = SpanScorer(config)

    def _target_chunk_counts(self, lengths: torch.Tensor) -> torch.Tensor:
        rounded = torch.round(
            lengths.to(dtype=torch.float32) / self.compression_target
        ).long()
        minimum = torch.div(
            lengths + self.max_span_length - 1,
            self.max_span_length,
            rounding_mode="floor",
        )
        return torch.maximum(rounded, minimum).clamp(min=1).minimum(lengths)

    @staticmethod
    def _safe_logsumexp(
        values: torch.Tensor,
        *,
        dim: int,
    ) -> torch.Tensor:
        reachable = torch.isfinite(values).any(dim=dim)
        safe_values = torch.where(
            reachable.unsqueeze(dim),
            values,
            torch.zeros_like(values),
        )
        reduced = torch.logsumexp(safe_values, dim=dim)
        return torch.where(
            reachable,
            reduced,
            torch.full_like(reduced, float("-inf")),
        )

    def _forward_backward_fixed_count(
        self,
        span_scores: torch.Tensor,
        lengths: torch.Tensor,
        target_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = span_scores.shape
        max_count = int(target_counts.max().item())
        negative_infinity = float("-inf")
        initial = span_scores.new_full(
            (batch_size, max_count + 1),
            negative_infinity,
        )
        initial[:, 0] = 0.0
        alpha_columns = [initial]
        for end in range(1, seq_len + 1):
            candidates = []
            for span_length in range(
                1,
                min(self.max_span_length, end) + 1,
            ):
                previous = alpha_columns[end - span_length]
                shifted = F.pad(
                    previous[:, :-1],
                    (1, 0),
                    value=negative_infinity,
                )
                candidates.append(
                    shifted
                    + span_scores[
                        :, end - span_length, span_length - 1
                    ].unsqueeze(1)
                )
            value = self._safe_logsumexp(
                torch.stack(candidates, dim=1),
                dim=1,
            )
            alpha_columns.append(torch.where(
                (end <= lengths).unsqueeze(1),
                value,
                torch.full_like(value, negative_infinity),
            ))
        alpha = torch.stack(alpha_columns, dim=1)

        beta_columns: list[torch.Tensor | None] = [None] * (seq_len + 1)
        for start in range(seq_len, -1, -1):
            if start == seq_len:
                value = span_scores.new_full(
                    (batch_size, max_count + 1),
                    negative_infinity,
                )
            else:
                candidates = []
                for span_length in range(
                    1,
                    min(self.max_span_length, seq_len - start) + 1,
                ):
                    following = beta_columns[start + span_length]
                    assert following is not None
                    shifted = F.pad(
                        following[:, :-1],
                        (1, 0),
                        value=negative_infinity,
                    )
                    candidates.append(
                        shifted
                        + span_scores[:, start, span_length - 1].unsqueeze(1)
                    )
                value = self._safe_logsumexp(
                    torch.stack(candidates, dim=1),
                    dim=1,
                )
            at_sequence_end = lengths == start
            value = torch.where(
                (start <= lengths).unsqueeze(1),
                value,
                torch.full_like(value, negative_infinity),
            )
            value = value.clone()
            value[:, 0] = torch.where(
                at_sequence_end,
                torch.zeros_like(value[:, 0]),
                value[:, 0],
            )
            beta_columns[start] = value
        beta = torch.stack(
            [column for column in beta_columns if column is not None],
            dim=1,
        )
        batch_indices = torch.arange(batch_size, device=span_scores.device)
        log_partition = alpha[batch_indices, lengths, target_counts]
        if not torch.isfinite(log_partition).all():
            raise RuntimeError(
                "The fixed-count semi-Markov constraint has no feasible path."
            )
        return alpha, beta, log_partition

    def _boundary_marginals_fixed_count(
        self,
        span_scores: torch.Tensor,
        lengths: torch.Tensor,
        target_counts: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        log_partition: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = span_scores.shape
        max_count = alpha.shape[-1] - 1
        prefix_counts = torch.arange(max_count, device=span_scores.device)
        suffix_counts = (
            target_counts.unsqueeze(1) - prefix_counts.unsqueeze(0) - 1
        )
        valid_counts = (suffix_counts >= 0) & (suffix_counts <= max_count)
        safe_suffix_counts = suffix_counts.clamp(min=0, max=max_count)
        probabilities = span_scores.new_zeros(batch_size, seq_len)
        for end in range(1, seq_len + 1):
            terms = []
            for span_length in range(
                1,
                min(self.max_span_length, end) + 1,
            ):
                start = end - span_length
                suffix = beta[:, end].gather(
                    1,
                    safe_suffix_counts,
                )
                term = (
                    alpha[:, start, :max_count]
                    + span_scores[:, start, span_length - 1].unsqueeze(1)
                    + suffix
                    - log_partition.unsqueeze(1)
                )
                terms.append(torch.where(
                    valid_counts,
                    term,
                    torch.full_like(term, float("-inf")),
                ))
            probability = torch.exp(self._safe_logsumexp(
                torch.cat(terms, dim=1),
                dim=1,
            ))
            probabilities[:, end - 1] = torch.where(
                end <= lengths,
                probability,
                torch.zeros_like(probability),
            )
        return probabilities.clamp(0.0, 1.0)

    def _viterbi_fixed_count(
        self,
        span_scores: torch.Tensor,
        lengths: torch.Tensor,
        target_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = span_scores.shape
        max_count = int(target_counts.max().item())
        negative_infinity = float("-inf")
        initial = span_scores.new_full(
            (batch_size, max_count + 1),
            negative_infinity,
        )
        initial[:, 0] = 0.0
        delta_columns = [initial]
        back_lengths = torch.zeros(
            batch_size,
            seq_len + 1,
            max_count + 1,
            device=span_scores.device,
            dtype=torch.long,
        )
        for end in range(1, seq_len + 1):
            candidates = []
            for span_length in range(
                1,
                min(self.max_span_length, end) + 1,
            ):
                shifted = F.pad(
                    delta_columns[end - span_length][:, :-1],
                    (1, 0),
                    value=negative_infinity,
                )
                candidates.append(
                    shifted
                    + span_scores[
                        :, end - span_length, span_length - 1
                    ].unsqueeze(1)
                )
            value, choice = torch.stack(candidates, dim=1).max(dim=1)
            value = torch.where(
                (end <= lengths).unsqueeze(1),
                value,
                torch.full_like(value, negative_infinity),
            )
            delta_columns.append(value)
            back_lengths[:, end] = choice + 1

        delta = torch.stack(delta_columns, dim=1)
        batch_indices = torch.arange(batch_size, device=span_scores.device)
        path_scores = delta[batch_indices, lengths, target_counts]
        hard_boundaries = torch.zeros(
            batch_size,
            seq_len,
            device=span_scores.device,
            dtype=torch.bool,
        )
        current = lengths.clone()
        remaining = target_counts.clone()
        for _ in range(max_count):
            active = remaining > 0
            chosen_lengths = back_lengths[
                batch_indices,
                current.clamp(min=0),
                remaining.clamp(min=0),
            ]
            boundary_positions = (current - 1).clamp_min(0)
            hard_boundaries = hard_boundaries | (
                F.one_hot(boundary_positions, num_classes=seq_len).bool()
                & active.unsqueeze(1)
            )
            current = torch.where(active, current - chosen_lengths, current)
            remaining = torch.where(active, remaining - 1, remaining)
        return hard_boundaries, path_scores

    def _forward_backward(
        self,
        span_scores: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = span_scores.shape
        zero = span_scores.new_zeros(batch_size)
        alpha_columns = [zero]
        for end in range(1, seq_len + 1):
            max_length = min(self.max_span_length, end)
            candidates = torch.stack([
                alpha_columns[end - span_length]
                + span_scores[:, end - span_length, span_length - 1]
                for span_length in range(1, max_length + 1)
            ], dim=1)
            active = end <= lengths
            safe_candidates = torch.where(
                active.unsqueeze(1),
                candidates,
                torch.zeros_like(candidates),
            )
            value = torch.logsumexp(safe_candidates, dim=1)
            alpha_columns.append(
                torch.where(active, value, value.new_full((), float("-inf")))
            )
        alpha = torch.stack(alpha_columns, dim=1)

        beta_columns: list[torch.Tensor | None] = [None] * (seq_len + 1)
        beta_columns[seq_len] = zero
        for start in range(seq_len - 1, -1, -1):
            max_length = min(self.max_span_length, seq_len - start)
            candidates = torch.stack([
                span_scores[:, start, span_length - 1]
                + beta_columns[start + span_length]
                for span_length in range(1, max_length + 1)
            ], dim=1)
            active = start < lengths
            safe_candidates = torch.where(
                active.unsqueeze(1),
                candidates,
                torch.zeros_like(candidates),
            )
            value = torch.logsumexp(safe_candidates, dim=1)
            beta_columns[start] = torch.where(
                start < lengths,
                value,
                torch.where(start == lengths, zero, value.new_full((), float("-inf"))),
            )
        beta = torch.stack([column for column in beta_columns], dim=1)
        log_partition = alpha.gather(1, lengths[:, None]).squeeze(1)
        return alpha, beta, log_partition

    def _boundary_marginals(
        self,
        span_scores: torch.Tensor,
        lengths: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        log_partition: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = span_scores.shape
        probabilities = span_scores.new_zeros(batch_size, seq_len)
        for end in range(1, seq_len + 1):
            max_length = min(self.max_span_length, end)
            log_marginals = torch.stack([
                alpha[:, end - span_length]
                + span_scores[:, end - span_length, span_length - 1]
                + beta[:, end]
                - log_partition
                for span_length in range(1, max_length + 1)
            ], dim=1)
            active = end <= lengths
            safe_log_marginals = torch.where(
                active.unsqueeze(1),
                log_marginals,
                torch.zeros_like(log_marginals),
            )
            probability = torch.exp(
                torch.logsumexp(safe_log_marginals, dim=1)
            )
            probabilities[:, end - 1] = torch.where(
                active,
                probability,
                torch.zeros_like(probability),
            )
        final_positions = (lengths - 1).clamp_min(0)
        final_mask = F.one_hot(
            final_positions,
            num_classes=seq_len,
        ).to(dtype=torch.bool)
        return torch.where(
            final_mask,
            torch.ones_like(probabilities),
            probabilities.clamp(0.0, 1.0),
        )

    def _viterbi(
        self,
        span_scores: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = span_scores.shape
        zero = span_scores.new_zeros(batch_size)
        delta_columns = [zero]
        back_lengths = torch.zeros(
            batch_size,
            seq_len + 1,
            device=span_scores.device,
            dtype=torch.long,
        )
        for end in range(1, seq_len + 1):
            max_length = min(self.max_span_length, end)
            candidates = torch.stack([
                delta_columns[end - span_length]
                + span_scores[:, end - span_length, span_length - 1]
                for span_length in range(1, max_length + 1)
            ], dim=1)
            value, choice = candidates.max(dim=1)
            delta_columns.append(
                torch.where(end <= lengths, value, value.new_full((), float("-inf")))
            )
            back_lengths[:, end] = choice + 1

        hard_boundaries = torch.zeros(
            batch_size,
            seq_len,
            device=span_scores.device,
            dtype=torch.bool,
        )
        current = lengths.clone()
        for _ in range(seq_len):
            active = current > 0
            boundary_positions = (current - 1).clamp_min(0)
            hard_boundaries = hard_boundaries | (
                F.one_hot(boundary_positions, num_classes=seq_len).bool()
                & active.unsqueeze(1)
            )
            chosen_lengths = back_lengths.gather(1, current[:, None]).squeeze(1)
            current = torch.where(active, current - chosen_lengths, current)
        return hard_boundaries

    @staticmethod
    def _soft_assignment(
        boundary_probabilities: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = boundary_probabilities.shape
        next_valid = F.pad(valid_mask[:, 1:], (0, 1), value=False)
        candidate_mask = valid_mask & next_valid
        state = boundary_probabilities.new_zeros(batch_size, seq_len)
        state[:, 0] = 1.0
        rows = []
        for position in range(seq_len):
            rows.append(state)
            probability = (
                boundary_probabilities[:, position]
                * candidate_mask[:, position]
            ).unsqueeze(1)
            shifted = F.pad(state[:, :-1], (1, 0))
            state = state * (1.0 - probability) + shifted * probability
        return torch.stack(rows, dim=1) * valid_mask.unsqueeze(-1)

    def forward(
        self,
        hidden: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len, _ = hidden.shape
        lengths = valid_mask.sum(dim=1).long()
        span_scores = self.scorer(hidden, valid_mask)
        if self.fixed_count:
            target_chunk_counts = self._target_chunk_counts(lengths)
            alpha, beta, log_partition = self._forward_backward_fixed_count(
                span_scores,
                lengths,
                target_chunk_counts,
            )
            probabilities = self._boundary_marginals_fixed_count(
                span_scores,
                lengths,
                target_chunk_counts,
                alpha,
                beta,
                log_partition,
            )
            hard_boundaries, viterbi_scores = self._viterbi_fixed_count(
                span_scores,
                lengths,
                target_chunk_counts,
            )
        else:
            alpha, beta, log_partition = self._forward_backward(
                span_scores,
                lengths,
            )
            probabilities = self._boundary_marginals(
                span_scores,
                lengths,
                alpha,
                beta,
                log_partition,
            )
            hard_boundaries = self._viterbi(span_scores, lengths)
            target_chunk_counts = torch.zeros_like(lengths)
            viterbi_scores = log_partition.new_zeros(log_partition.shape)
        segment_ids = (
            hard_boundaries.long().cumsum(dim=1) - hard_boundaries.long()
        )
        hard_assignment = F.one_hot(
            segment_ids.clamp(min=0, max=seq_len - 1),
            num_classes=seq_len,
        ).to(dtype=hidden.dtype)
        hard_assignment = hard_assignment * valid_mask.unsqueeze(-1)
        soft_assignment = self._soft_assignment(probabilities, valid_mask)
        assignment = soft_assignment + (
            hard_assignment - soft_assignment
        ).detach()
        counts = assignment.sum(dim=1)
        soft_proxy = torch.einsum("btk,btd->bkd", assignment, hidden)
        soft_proxy = soft_proxy / counts.clamp_min(1e-6).unsqueeze(-1)

        chunk_counts = hard_boundaries.sum(dim=1)
        if not self.fixed_count:
            target_chunk_counts = chunk_counts
        chunk_mask = (
            torch.arange(seq_len, device=hidden.device).unsqueeze(0)
            < chunk_counts.unsqueeze(1)
        )
        soft_chunk_counts = probabilities.sum(dim=1)
        soft_ratio = lengths.to(hidden.dtype) / soft_chunk_counts.clamp_min(1.0)
        hard_ratio = lengths.to(hidden.dtype) / chunk_counts.clamp_min(1)
        target_ratio = lengths.to(hidden.dtype) / target_chunk_counts.clamp_min(1)
        candidate_mask = valid_mask & F.pad(
            valid_mask[:, 1:],
            (0, 1),
            value=False,
        )
        logits = torch.logit(probabilities.clamp(1e-6, 1.0 - 1e-6))
        candidate_logits = logits[candidate_mask]
        logit_l2 = (
            candidate_logits.square().mean()
            if candidate_logits.numel()
            else logits.sum() * 0.0
        )
        return {
            "pooled": soft_proxy,
            "hard_assignment": hard_assignment,
            "soft_assignment": soft_assignment,
            "chunk_mask": chunk_mask,
            "chunk_counts": chunk_counts,
            "hard_boundaries": hard_boundaries,
            "segment_ids": segment_ids,
            "gate_logits": logits * valid_mask,
            "gate_probabilities": probabilities * valid_mask,
            "soft_chunk_counts": soft_chunk_counts,
            "soft_tokens_per_chunk": soft_ratio,
            "hard_tokens_per_chunk": hard_ratio,
            "gate_logit_l2": logit_l2,
            "semi_markov_log_partition": log_partition,
            "target_chunk_counts": target_chunk_counts,
            "target_tokens_per_chunk": target_ratio,
            "chunk_count_constraint_violation": (
                chunk_counts - target_chunk_counts
            ).abs(),
            "hard_soft_ratio_gap": (hard_ratio - soft_ratio).abs(),
            "semi_markov_viterbi_score": viterbi_scores,
            "fixed_count_active": torch.tensor(
                self.fixed_count,
                device=hidden.device,
                dtype=torch.bool,
            ),
        }


class GreedySpanSegmenter(nn.Module):
    """Choose each next bounded span locally with an aligned soft backward pass."""

    def __init__(self, config: SegmentalVQVAEConfig):
        super().__init__()
        self.max_span_length = config.max_span_length
        self.scorer = SpanScorer(config)

    def _greedy_boundaries(
        self,
        span_scores: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Decode local argmax spans and retain local-softmax training statistics."""
        batch_size, seq_len, max_span_length = span_scores.shape
        if max_span_length != self.max_span_length:
            raise ValueError("span_scores has the wrong maximum span length.")
        batch_indices = torch.arange(batch_size, device=span_scores.device)
        candidate_lengths = torch.arange(
            1,
            self.max_span_length + 1,
            device=span_scores.device,
            dtype=span_scores.dtype,
        )
        current = torch.zeros_like(lengths)
        hard_boundaries = torch.zeros(
            batch_size,
            seq_len,
            device=span_scores.device,
            dtype=torch.bool,
        )
        expected_length_rows = []
        selected_probability_rows = []
        entropy_rows = []
        active_rows = []
        chosen_length_rows = []
        probability_rows = []
        for _ in range(seq_len):
            active = current < lengths
            safe_start = current.clamp(max=max(seq_len - 1, 0))
            local_scores = span_scores[batch_indices, safe_start]
            local_scores = torch.where(
                active.unsqueeze(1),
                local_scores,
                torch.zeros_like(local_scores),
            )
            probabilities = torch.softmax(local_scores, dim=1)
            chosen_indices = local_scores.argmax(dim=1)
            chosen_lengths = chosen_indices + 1
            chosen_lengths = torch.where(
                active,
                chosen_lengths,
                torch.zeros_like(chosen_lengths),
            )
            boundary_positions = (
                current + chosen_lengths - 1
            ).clamp(min=0, max=max(seq_len - 1, 0))
            hard_boundaries = hard_boundaries | (
                F.one_hot(boundary_positions, num_classes=seq_len).bool()
                & active.unsqueeze(1)
            )
            expected_length_rows.append(
                (probabilities * candidate_lengths.unsqueeze(0)).sum(dim=1)
            )
            selected_probability_rows.append(
                probabilities.gather(1, chosen_indices.unsqueeze(1)).squeeze(1)
            )
            entropy_rows.append(
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
            )
            active_rows.append(active)
            chosen_length_rows.append(chosen_lengths)
            probability_rows.append(probabilities)
            current = torch.where(active, current + chosen_lengths, current)

        if not torch.equal(current, lengths):
            raise RuntimeError("Greedy span decoding did not exactly cover the sequence.")
        return (
            hard_boundaries,
            torch.stack(expected_length_rows, dim=1),
            torch.stack(selected_probability_rows, dim=1),
            torch.stack(entropy_rows, dim=1),
            torch.stack(active_rows, dim=1),
            torch.stack(chosen_length_rows, dim=1),
            torch.stack(probability_rows, dim=1),
        )

    def _mask_illegal_endpoints(
        self,
        span_scores: torch.Tensor,
        legal_endpoints: torch.Tensor,
    ) -> torch.Tensor:
        """Restrict spans to legal ends, with a bounded emergency fallback."""
        if legal_endpoints.shape != span_scores.shape[:2]:
            raise ValueError("legal_endpoints must match hidden sequence dimensions.")
        _, seq_len, max_span_length = span_scores.shape
        starts = torch.arange(seq_len, device=span_scores.device).unsqueeze(1)
        candidate_lengths = torch.arange(
            1,
            max_span_length + 1,
            device=span_scores.device,
        ).unsqueeze(0)
        ends = starts + candidate_lengths - 1
        in_range = ends < seq_len
        endpoint_legal = legal_endpoints.gather(
            1,
            ends.clamp_max(seq_len - 1).reshape(1, -1).expand(
                legal_endpoints.shape[0], -1
            ),
        ).reshape(legal_endpoints.shape[0], seq_len, max_span_length)
        candidate_valid = torch.isfinite(span_scores) & in_range.unsqueeze(0)
        permitted = candidate_valid & endpoint_legal

        # A very long word can contain no legal endpoint within max_span_length.
        # Keep the partition feasible by allowing only the furthest valid span.
        missing = candidate_valid.any(dim=-1) & ~permitted.any(dim=-1)
        furthest = candidate_valid.long().sum(dim=-1).clamp_min(1) - 1
        fallback = F.one_hot(furthest, num_classes=max_span_length).bool()
        permitted = permitted | (fallback & missing.unsqueeze(-1))
        return span_scores.masked_fill(~permitted, float("-inf"))

    def _soft_span_proxies(
        self,
        hidden: torch.Tensor,
        span_scores: torch.Tensor,
        lengths: torch.Tensor,
        active_steps: torch.Tensor,
        chosen_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Mix candidate-span means at each hard-visited start for STE gradients."""
        batch_size, seq_len, hidden_dim = hidden.shape
        batch_indices = torch.arange(batch_size, device=hidden.device)
        candidate_lengths = torch.arange(
            1,
            self.max_span_length + 1,
            device=hidden.device,
        )
        prefix = F.pad(hidden.cumsum(dim=1), (0, 0, 1, 0))
        current = torch.zeros_like(lengths)
        rows = []
        for step in range(seq_len):
            active = active_steps[:, step]
            safe_start = current.clamp(max=max(seq_len - 1, 0))
            local_scores = span_scores[batch_indices, safe_start]
            local_scores = torch.where(
                active.unsqueeze(1),
                local_scores,
                torch.zeros_like(local_scores),
            )
            probabilities = torch.softmax(local_scores, dim=1)
            ends = (
                safe_start.unsqueeze(1) + candidate_lengths.unsqueeze(0)
            ).clamp(max=seq_len)
            end_values = prefix.gather(
                1,
                ends.unsqueeze(-1).expand(-1, -1, hidden_dim),
            )
            start_values = prefix.gather(
                1,
                safe_start[:, None, None].expand(-1, 1, hidden_dim),
            )
            candidate_means = (
                end_values - start_values
            ) / candidate_lengths.to(hidden.dtype)[None, :, None]
            rows.append(
                (probabilities.unsqueeze(-1) * candidate_means).sum(dim=1)
                * active.unsqueeze(1)
            )
            current = torch.where(
                active,
                current + chosen_lengths[:, step],
                current,
            )
        return torch.stack(rows, dim=1)

    def forward(
        self,
        hidden: torch.Tensor,
        valid_mask: torch.Tensor,
        legal_endpoints: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _, seq_len, _ = hidden.shape
        lengths = valid_mask.sum(dim=1).long()
        span_scores = self.scorer(hidden, valid_mask)
        if legal_endpoints is not None:
            span_scores = self._mask_illegal_endpoints(
                span_scores,
                legal_endpoints.to(device=hidden.device, dtype=torch.bool),
            )
        (
            hard_boundaries,
            expected_lengths,
            selected_probabilities,
            local_entropies,
            active_steps,
            chosen_lengths,
            length_probabilities,
        ) = self._greedy_boundaries(span_scores, lengths)
        segment_ids = (
            hard_boundaries.long().cumsum(dim=1) - hard_boundaries.long()
        )
        hard_assignment = F.one_hot(
            segment_ids.clamp(min=0, max=seq_len - 1),
            num_classes=seq_len,
        ).to(dtype=hidden.dtype)
        hard_assignment = hard_assignment * valid_mask.unsqueeze(-1)
        soft_proxy = self._soft_span_proxies(
            hidden,
            span_scores,
            lengths,
            active_steps,
            chosen_lengths,
        )

        chunk_counts = hard_boundaries.sum(dim=1)
        chunk_mask = (
            torch.arange(seq_len, device=hidden.device).unsqueeze(0)
            < chunk_counts.unsqueeze(1)
        )
        active_float = active_steps.to(hidden.dtype)
        local_expected_ratio = (
            expected_lengths * active_float
        ).sum(dim=1) / chunk_counts.clamp_min(1).to(hidden.dtype)
        hard_ratio = lengths.to(hidden.dtype) / chunk_counts.clamp_min(1)
        aligned_ratio = local_expected_ratio + (
            hard_ratio - local_expected_ratio
        ).detach()
        mean_selected_probability = (
            selected_probabilities * active_float
        ).sum(dim=1) / chunk_counts.clamp_min(1).to(hidden.dtype)
        mean_local_entropy = (
            local_entropies * active_float
        ).sum(dim=1) / chunk_counts.clamp_min(1).to(hidden.dtype)
        boundary_values = hard_boundaries.to(hidden.dtype)
        zero = span_scores[torch.isfinite(span_scores)].sum() * 0.0
        return {
            "pooled": soft_proxy,
            "hard_assignment": hard_assignment,
            "soft_assignment": hard_assignment.to(hidden.dtype),
            "chunk_mask": chunk_mask,
            "chunk_counts": chunk_counts,
            "hard_boundaries": hard_boundaries,
            "segment_ids": segment_ids,
            "gate_logits": boundary_values,
            "gate_probabilities": boundary_values,
            "soft_chunk_counts": lengths.to(hidden.dtype)
            / local_expected_ratio.clamp_min(1e-6),
            "soft_tokens_per_chunk": local_expected_ratio,
            "compression_tokens_per_chunk": aligned_ratio,
            "hard_tokens_per_chunk": hard_ratio,
            "gate_logit_l2": zero,
            "hard_soft_ratio_gap": (
                hard_ratio - local_expected_ratio
            ).abs(),
            "greedy_selected_probability": mean_selected_probability,
            "greedy_local_entropy": mean_local_entropy,
            "greedy_active_steps": active_steps,
            "greedy_chosen_lengths": chosen_lengths,
            "greedy_length_probabilities": length_probabilities,
            "fixed_count_active": torch.zeros(
                (), device=hidden.device, dtype=torch.bool
            ),
            "greedy_active": torch.ones(
                (), device=hidden.device, dtype=torch.bool
            ),
        }


class SegmentContentEncoder(nn.Module):
    """Encode selected spans bidirectionally without cross-span information."""

    def __init__(self, config: SegmentalVQVAEConfig):
        super().__init__()
        self.max_span_length = config.max_span_length
        self.local_position_embedding = nn.Embedding(
            config.max_span_length,
            config.d_model,
        )
        self.layers = nn.ModuleList(
            MaskedBidirectionalBlock(config)
            for _ in range(config.span_encoder_layers)
        )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.pooling_head = nn.Linear(config.d_model, 1)

    @staticmethod
    def local_positions(
        segment_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.arange(
            segment_ids.shape[1],
            device=segment_ids.device,
        ).unsqueeze(0).expand_as(segment_ids)
        starts = torch.zeros_like(valid_mask)
        starts[:, 0] = valid_mask[:, 0]
        starts[:, 1:] = (
            valid_mask[:, 1:]
            & (segment_ids[:, 1:] != segment_ids[:, :-1])
        )
        start_positions = torch.where(
            starts,
            positions,
            torch.zeros_like(positions),
        ).cummax(dim=1).values
        return torch.where(
            valid_mask,
            positions - start_positions,
            torch.zeros_like(positions),
        )

    def forward(
        self,
        token_embeddings: torch.Tensor,
        segment_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        hard_assignment: torch.Tensor,
    ) -> torch.Tensor:
        local_positions = self.local_positions(segment_ids, valid_mask)
        hidden = token_embeddings + self.local_position_embedding(
            local_positions.clamp_max(self.local_position_embedding.num_embeddings - 1)
        )
        radius = self.max_span_length - 1
        valid_windows = F.pad(
            valid_mask,
            (radius, radius),
            value=False,
        ).unfold(1, 2 * radius + 1, 1)
        segment_windows = F.pad(
            segment_ids,
            (radius, radius),
            value=-1,
        ).unfold(1, 2 * radius + 1, 1)
        allowed = (
            valid_mask.unsqueeze(-1)
            & valid_windows
            & (segment_ids.unsqueeze(-1) == segment_windows)
        )
        for layer in self.layers:
            hidden = layer(
                hidden,
                allowed,
                local_positions,
                ~valid_mask,
            )
        hidden = zero_padded_positions(self.output_norm(hidden), ~valid_mask)
        logits = self.pooling_head(hidden).squeeze(-1)
        stabilized = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
        weights = stabilized.unsqueeze(-1) * hard_assignment
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.einsum("btk,btd->bkd", weights, hidden)


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
        if config.segmentation_mode in {
            "semi_markov",
            "semi_markov_fixed_count",
            "semi_markov_greedy",
        }:
            self.encoder_layers = None
            self.encoder_norm = None
            self.gater = None
            self.segment_pooler = None
            self.token_pruner = None
            self.boundary_encoder: LocalBoundaryEncoder | None = (
                LocalBoundaryEncoder(config)
            )
            self.semi_markov_segmenter: (
                SemiMarkovSegmenter | GreedySpanSegmenter | None
            ) = (
                GreedySpanSegmenter(config)
                if config.segmentation_mode == "semi_markov_greedy"
                else SemiMarkovSegmenter(config)
            )
            self.span_content_encoder: SegmentContentEncoder | None = (
                SegmentContentEncoder(config)
            )
        else:
            self.encoder_layers = nn.ModuleList(
                RotaryResidualBlock(config) for _ in range(config.encoder_layers)
            )
            self.encoder_norm = nn.LayerNorm(config.d_model)
            if config.segmentation_mode == "token_pruning":
                self.gater = TokenKeepGate(config)
                self.segment_pooler = None
                self.token_pruner: TokenPruner | None = TokenPruner(
                    config.gate_threshold
                )
            else:
                self.gater = BoundaryGater(config)
                self.segment_pooler = SegmentPooler(config.gate_threshold)
                self.token_pruner = None
            self.boundary_encoder = None
            self.semi_markov_segmenter = None
            self.span_content_encoder = None
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

        token_embeddings = self.token_embedding(input_ids)
        if self.config.segmentation_mode in {
            "semi_markov",
            "semi_markov_fixed_count",
            "semi_markov_greedy",
        }:
            assert self.boundary_encoder is not None
            assert self.semi_markov_segmenter is not None
            assert self.span_content_encoder is not None
            boundary_hidden = self.boundary_encoder(token_embeddings, valid_mask)
            segmented = self.semi_markov_segmenter(
                boundary_hidden,
                valid_mask,
            )
            hard_pooled = self.span_content_encoder(
                token_embeddings,
                segmented["segment_ids"],
                valid_mask,
                segmented["hard_assignment"],
            )
            soft_proxy = segmented["pooled"]
            pooled = hard_pooled + (soft_proxy - soft_proxy.detach())
            segmented = {**segmented, "pooled": pooled}
        else:
            assert self.encoder_layers is not None
            assert self.encoder_norm is not None
            assert self.gater is not None
            hidden = token_embeddings
            for layer in self.encoder_layers:
                hidden = layer(hidden, padding_mask=~valid_mask)
                hidden = zero_padded_positions(hidden, ~valid_mask)
            hidden = self.encoder_norm(hidden)
            hidden = zero_padded_positions(hidden, ~valid_mask)
            should_sample = self.training if sample_gates is None else sample_gates
            if self.config.segmentation_mode == "token_pruning":
                assert isinstance(self.gater, TokenKeepGate)
                assert self.token_pruner is not None
                token_latents = self.latent_projection(hidden)
                token_latents = zero_padded_positions(token_latents, ~valid_mask)
                gate_logits = self.gater(token_latents, valid_mask)
                segmented = self.token_pruner(
                    token_latents,
                    gate_logits,
                    valid_mask,
                    sample_gates=should_sample,
                )
            else:
                assert self.segment_pooler is not None
                gate_logits = self.gater(hidden, valid_mask)
                segmented = self.segment_pooler(
                    hidden,
                    gate_logits,
                    valid_mask,
                    sample_gates=should_sample,
                )
        latents = (
            segmented["pooled"]
            if self.config.segmentation_mode == "token_pruning"
            else self.latent_projection(segmented["pooled"])
        )
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
        raw_logits = []
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
            raw_step_logits = step_logits
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
            raw_logits.append(raw_step_logits)
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
            "raw_logits": torch.stack(raw_logits, dim=1),
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

    soft_ratio = outputs.get(
        "compression_tokens_per_chunk",
        outputs["soft_tokens_per_chunk"],
    )
    gate_logit_l2 = outputs["gate_logit_l2"]
    assert isinstance(soft_ratio, torch.Tensor)
    assert isinstance(gate_logit_l2, torch.Tensor)
    if config.segmentation_mode == "semi_markov_fixed_count":
        compression_raw = reconstruction.new_zeros(())
        compression = reconstruction.new_zeros(())
    else:
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
