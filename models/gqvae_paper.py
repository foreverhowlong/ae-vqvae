"""Faithful model/loss implementation of GQ-VAE arXiv:2512.21913v1.

This module intentionally owns every operation that changes the optimization
dynamics: architecture, differentiable gate masks, VQ warmup/resampling, and
the five paper losses.  Data policy, optimizers, logging, and bistability
classification live in the training package.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import Encoder as XTransformerEncoder
from x_transformers.x_transformers import AttentionLayers, FeedForward

from common.gqvae_paper_config import GQVAEPaperModelConfig


@dataclass
class GQVAEPaperOutput:
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    codebook_loss: torch.Tensor | None
    commitment_loss: torch.Tensor | None
    compression_loss: torch.Tensor
    length_loss: torch.Tensor
    byte_logits: torch.Tensor
    predicted_masks: torch.Tensor
    reconstruction_masks: torch.Tensor
    reconstruction_targets: torch.Tensor
    gates: torch.Tensor
    code_indices: torch.Tensor | None
    codebook_utilization_batch: float | None
    quantizer_active: bool


class ReservoirSampler(nn.Module):
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity
        self.register_buffer("buffer", torch.empty(0), persistent=False)
        self.seen = 0

    @torch.no_grad()
    def reset(self) -> None:
        self.buffer = torch.empty(0, device=self.buffer.device)
        self.seen = 0

    @torch.no_grad()
    def add(self, samples: torch.Tensor) -> None:
        samples = samples.detach()
        if self.buffer.numel() == 0:
            self.buffer = torch.empty(
                self.capacity,
                samples.shape[-1],
                device=samples.device,
                dtype=samples.dtype,
            )
        if self.seen < self.capacity:
            count = min(self.capacity - self.seen, len(samples))
            self.buffer[self.seen : self.seen + count] = samples[:count]
            self.seen += count
            samples = samples[count:]
        if samples.numel() == 0:
            return
        stream_positions = torch.arange(
            self.seen,
            self.seen + len(samples),
            device=samples.device,
        )
        replacement = (stream_positions * torch.rand_like(stream_positions.float())).long()
        selected = replacement < self.capacity
        if selected.any():
            self.buffer[replacement[selected]] = samples[selected]
        self.seen += len(samples)

    def contents(self) -> torch.Tensor:
        return self.buffer[: min(self.seen, self.capacity)]


class PaperVectorQuantizer(nn.Module):
    def __init__(self, config: GQVAEPaperModelConfig):
        super().__init__()
        self.config = config
        self.codebook = nn.Embedding(config.codebook_size, config.codebook_dim)
        self.codebook.weight.data.uniform_(
            -1.0 / config.codebook_size,
            1.0 / config.codebook_size,
        )
        self.reservoir = ReservoirSampler(config.quantizer_reservoir_size)
        self.register_buffer("usage_ema", torch.zeros(config.codebook_size))

    def _nearest_indices(self, vectors: torch.Tensor) -> torch.Tensor:
        # The released implementation uses KeOps for this exact Euclidean
        # nearest-neighbour operation.  Dense distance is substantially faster
        # for tiny unit-test configurations and has identical assignments.
        if vectors.is_cuda and self.config.codebook_size >= 4096:
            from pykeops.torch import LazyTensor

            left = LazyTensor(vectors[:, None, :])
            right = LazyTensor(self.codebook.weight[None, :, :])
            distances = ((left - right) ** 2).sum(-1)
            return distances.argKmin(K=1, dim=1).reshape(-1).long()
        distances = (
            vectors.square().sum(dim=1, keepdim=True)
            - 2.0 * vectors @ self.codebook.weight.t()
            + self.codebook.weight.square().sum(dim=1).unsqueeze(0)
        )
        return distances.argmin(dim=1)

    @torch.no_grad()
    def _update_reservoir(self, vectors: torch.Tensor, indices: torch.Tensor) -> None:
        counts = torch.bincount(indices, minlength=self.config.codebook_size).to(
            dtype=self.usage_ema.dtype
        )
        probabilities = counts / max(len(indices), 1)
        decay = self.config.quantizer_usage_decay
        self.usage_ema.mul_(decay).add_(probabilities, alpha=1.0 - decay)
        # The source keeps at most 2500 high-distance vectors.  Sampling all
        # vectors while the reservoir is small is equivalent during warmup;
        # after activation we retain the exact high-error policy.
        selected_codes = self.codebook(indices)
        errors = (vectors - selected_codes).square().sum(dim=-1)
        if len(vectors) > 2500:
            keep = errors.topk(2500).indices
            vectors = vectors[keep]
        self.reservoir.add(vectors)

    @torch.no_grad()
    def _resample_unused(self) -> int:
        unused = self.usage_ema == 0
        count = int(unused.sum())
        candidates = self.reservoir.contents()
        if count == 0:
            return 0
        if len(candidates) < count:
            raise RuntimeError(
                f"Paper VQ resampling needs {count} vectors but reservoir has "
                f"only {len(candidates)}."
            )
        chosen = torch.randperm(len(candidates), device=candidates.device)[:count]
        self.codebook.weight.data[unused] = candidates[chosen]
        return count

    def forward(
        self,
        latents: torch.Tensor,
        *,
        step: int,
        update_state: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, int]:
        flat = latents.reshape(-1, latents.shape[-1])
        warmup = step < self.config.quantizer_warmup_steps
        if warmup:
            if update_state:
                self.reservoir.add(flat)
            return latents, None, None, None, 0

        resampled = 0
        due = (
            update_state
            and (
                step == self.config.quantizer_warmup_steps
                or (
                    step > self.config.quantizer_warmup_steps
                    and (step - self.config.quantizer_warmup_steps)
                    % self.config.quantizer_resample_every
                    == 0
                )
            )
        )
        if due:
            resampled = self._resample_unused()

        indices = self._nearest_indices(flat)
        raw = self.codebook(indices).view_as(latents)
        codebook_loss = (raw.detach() - latents).square().mean()
        commitment_loss = self.config.commitment_beta * (
            raw - latents.detach()
        ).square().mean()
        quantized = latents + (raw - latents).detach()
        if update_state:
            self._update_reservoir(flat, indices)
            if due:
                self.reservoir.reset()
                self.usage_ema.zero_()
        return quantized, codebook_loss, commitment_loss, indices.view(latents.shape[:2]), resampled


class PaperEncoder(nn.Module):
    def __init__(self, config: GQVAEPaperModelConfig):
        super().__init__()
        heads = config.codebook_dim // config.attention_head_dim
        self.layers = AttentionLayers(
            dim=config.codebook_dim,
            causal=False,
            depth=config.encoder_depth,
            heads=heads,
            attn_dim_head=config.attention_head_dim,
            alibi_pos_bias=True,
        )
        self.projection = FeedForward(
            dim=config.codebook_dim,
            dim_out=config.codebook_dim + 1,
        )

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(self.layers(embedded))
        latents, _unused_length = hidden.split([hidden.shape[-1] - 1, 1], dim=-1)
        return latents


class PaperGater(nn.Module):
    def __init__(self, config: GQVAEPaperModelConfig):
        super().__init__()
        dim = config.codebook_dim
        heads = dim // config.attention_head_dim
        width = config.decode_width
        self.attention = XTransformerEncoder(
            dim=dim,
            depth=config.gater_depth,
            heads=heads,
            alibi_pos_bias=True,
            use_simple_rmsnorm=True,
        )
        self.conv1 = nn.Conv1d(dim, dim // 2, width * 2, padding="same")
        self.conv2 = nn.Conv1d(dim // 2, dim // 4, width + width // 2, padding="same")
        self.conv3 = nn.Conv1d(dim // 4, 1, width, padding="same")

    def forward(self, quantized: torch.Tensor) -> torch.Tensor:
        hidden = self.attention(quantized).transpose(1, 2).contiguous()
        # No activation between convolutions: this mirrors the released model.
        return torch.sigmoid(self.conv3(self.conv2(self.conv1(hidden))).squeeze(1))


class PaperDecoder(nn.Module):
    def __init__(self, config: GQVAEPaperModelConfig):
        super().__init__()
        dim = config.codebook_dim
        self.alphabet_size = config.alphabet_size
        self.decode_width = config.decode_width
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(config.decoder_depth))
        self.feedforwards = nn.ModuleList(
            FeedForward(dim=dim, dim_out=dim) for _ in range(config.decoder_depth)
        )
        self.output = nn.ConvTranspose1d(
            dim,
            config.alphabet_size + 1,
            kernel_size=config.decode_width,
            stride=config.decode_width,
        )

    def forward(self, quantized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = quantized
        for norm, feedforward in zip(self.norms, self.feedforwards, strict=True):
            hidden = hidden + feedforward(norm(hidden))
        expanded = self.output(hidden.transpose(1, 2).contiguous())
        byte_logits, length_logits = expanded.split([self.alphabet_size, 1], dim=1)
        batch, _, sequence_x_width = byte_logits.shape
        byte_logits = byte_logits.reshape(
            batch,
            self.alphabet_size,
            sequence_x_width // self.decode_width,
            self.decode_width,
        )
        length_logits = length_logits.reshape(batch, -1, self.decode_width).transpose(1, 2)
        return byte_logits, length_logits


class GQVAEPaper(nn.Module):
    """Paper model implementing a small, typed boundary for the runner."""

    def __init__(self, config: GQVAEPaperModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.alphabet_size, config.embedding_dim)
        self.encoder = PaperEncoder(config)
        self.quantizer = PaperVectorQuantizer(config)
        self.gater = PaperGater(config)
        self.decoder = PaperDecoder(config)

    def _targets_and_masks(
        self,
        input_ids: torch.Tensor,
        gates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sequence = input_ids.shape
        width = self.config.decode_width
        flattened = input_ids.flatten()
        targets = torch.zeros(width, batch * sequence, device=input_ids.device)
        rolling = flattened
        for index in range(width):
            targets[width - index - 1] = rolling
            rolling = rolling.roll(1)
            rolling[(torch.arange(batch * sequence, device=input_ids.device) % sequence) == 0] = 0
        targets = targets.reshape(width, batch, sequence).permute(1, 2, 0).long()

        inverse = F.pad(1.0 - gates, (width - 1, 0), value=0.0)
        expanded = inverse.unfold(1, size=width, step=1).flip(2)
        expanded[:, :, 0] = 1.0
        masks = torch.cumprod(expanded, dim=2).flip(2)
        return targets, masks

    @staticmethod
    def _mask_approximation(length_logits: torch.Tensor) -> torch.Tensor:
        masks = length_logits.transpose(1, 2).contiguous()
        masks = masks - masks.max(dim=2, keepdim=True).values
        masks = masks.clamp(-50, 50).exp().add(1e-6)
        masks = masks.cumsum(dim=2)
        return masks / masks[:, :, -1:]

    def _hardset_gates(self, input_ids: torch.Tensor, width: int) -> torch.Tensor:
        first_zero = torch.argmax((input_ids == 0).float(), dim=1) - 1
        row = torch.zeros(self.config.input_len, device=input_ids.device)
        row[width - 1 :: width] = 1
        gates = row.repeat(len(input_ids), 1)
        end = (((first_zero // width) + 1) * width).clamp(max=self.config.input_len)
        positions = torch.arange(self.config.input_len, device=input_ids.device)[None]
        return gates * (positions < end[:, None])

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        step: int = 0,
        update_quantizer_state: bool = True,
    ) -> GQVAEPaperOutput:
        latents = self.encoder(self.embedding(input_ids.long()).float())
        quantized, codebook, commitment, indices, _resampled = self.quantizer(
            latents,
            step=step,
            update_state=update_quantizer_state,
        )
        gates = self.gater(quantized)
        if self.config.quantizer_hardset is not None:
            gates = self._hardset_gates(input_ids, self.config.quantizer_hardset)
        compression = self.config.compression_alpha * gates.mean()
        byte_logits, length_logits = self.decoder(quantized)
        targets, masks = self._targets_and_masks(input_ids, gates)
        predicted_masks = self._mask_approximation(length_logits).clamp(1e-7, 1 - 1e-7)
        length = F.binary_cross_entropy(predicted_masks, masks, reduction="none")
        length = (length.mean(dim=2) * gates.detach()).mean() * self.config.length_gamma
        reconstruction = F.cross_entropy(byte_logits, targets, reduction="none")
        reconstruction = (reconstruction * masks).mean()
        loss = reconstruction + compression + length
        if codebook is not None and commitment is not None:
            loss = loss + codebook + commitment
        utilization = None
        if indices is not None:
            utilization = float(indices.unique().numel() / self.config.codebook_size)
        return GQVAEPaperOutput(
            loss=loss,
            reconstruction_loss=reconstruction,
            codebook_loss=codebook,
            commitment_loss=commitment,
            compression_loss=compression,
            length_loss=length,
            byte_logits=byte_logits,
            predicted_masks=predicted_masks,
            reconstruction_masks=masks,
            reconstruction_targets=targets,
            gates=gates,
            code_indices=indices,
            codebook_utilization_batch=utilization,
            quantizer_active=indices is not None,
        )
