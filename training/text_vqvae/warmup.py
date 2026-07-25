"""Adaptive AE-warmup diagnostics based on latent dimensionality."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from training.text_vqvae.geometry import preserve_rng_state


def reverse_water_filling(
    eigenvalues: torch.Tensor,
    *,
    rate_bits: float,
) -> tuple[float, int]:
    """Return the Shannon water level and number of modes above it."""
    if rate_bits <= 0:
        raise ValueError("rate_bits must be positive.")
    values = torch.as_tensor(eigenvalues, dtype=torch.float64).flatten()
    values = values[torch.isfinite(values) & (values > 0)]
    if values.numel() == 0:
        return 0.0, 0
    values = values.sort(descending=True).values
    log_values = values.log()
    rate_nats = rate_bits * math.log(2.0)
    for active in range(1, len(values) + 1):
        log_level = (
            log_values[:active].mean()
            - 2.0 * rate_nats / active
        )
        level = float(log_level.exp())
        next_value = float(values[active]) if active < len(values) else 0.0
        if level >= next_value:
            return level, active
    raise RuntimeError("Could not resolve a reverse water-filling level.")


def latent_spectrum_metrics(
    vectors: torch.Tensor,
    *,
    codebook_size: int,
    variance_threshold: float,
) -> dict[str, object]:
    """Compute full PCA-spectrum and target-rate effective dimensions."""
    if vectors.ndim != 2 or len(vectors) < 2:
        raise ValueError("Adaptive warmup needs at least two latent vectors.")
    if codebook_size < 2:
        raise ValueError("Adaptive warmup needs a codebook with at least two entries.")
    if not 0 < variance_threshold < 1:
        raise ValueError("variance_threshold must be between 0 and 1.")

    centered = vectors.detach().to(device="cpu", dtype=torch.float64)
    centered = centered - centered.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (len(centered) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    total_variance = eigenvalues.sum()
    if float(total_variance) == 0.0:
        effective_dim = 0
        participation_ratio = 0.0
    else:
        cumulative = eigenvalues.cumsum(0) / total_variance
        effective_dim = int(
            torch.searchsorted(
                cumulative,
                torch.tensor(variance_threshold, dtype=cumulative.dtype),
            ).item()
            + 1
        )
        participation_ratio = float(
            total_variance.square()
            / eigenvalues.square().sum().clamp_min(1e-24)
        )

    rate_bits = math.log2(codebook_size)
    water_level, water_filling_dim = reverse_water_filling(
        eigenvalues,
        rate_bits=rate_bits,
    )
    return {
        "latent_points": int(len(vectors)),
        "latent_dimension": int(vectors.shape[1]),
        "variance_threshold": variance_threshold,
        "latent_effective_dim": effective_dim,
        "participation_ratio": participation_ratio,
        "rate_bits": rate_bits,
        "water_filling_level": water_level,
        "water_filling_effective_dim": water_filling_dim,
        "pca_eigenvalues": eigenvalues.tolist(),
    }


@torch.no_grad()
def evaluate_adaptive_warmup(
    model,
    probe_batches,
    *,
    codebook_size: int,
    variance_threshold: float,
) -> dict[str, object]:
    """Evaluate the AE latent spectrum on one fixed, PAD-filtered probe."""
    device = next(model.parameters()).device
    was_training = model.training
    chunks = []
    try:
        with preserve_rng_state():
            model.eval()
            for batch in probe_batches:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                latents, latent_mask = model.encode(
                    input_ids,
                    attention_mask=attention_mask,
                    return_mask=True,
                )
                chunks.append(latents[latent_mask].float().cpu())
    finally:
        model.train(was_training)
    valid_chunks = [chunk for chunk in chunks if len(chunk)]
    if not valid_chunks:
        raise ValueError("Adaptive warmup probe contains no valid latent slots.")
    return latent_spectrum_metrics(
        torch.cat(valid_chunks),
        codebook_size=codebook_size,
        variance_threshold=variance_threshold,
    )


@dataclass
class AdaptiveWarmupController:
    """Stop after both effective-dimension signals plateau, with a hard cap."""

    min_steps: int
    max_steps: int
    patience: int
    tolerance: int
    history: list[dict[str, int]] = field(default_factory=list)

    def observe(self, step: int, metrics: dict[str, object]) -> dict[str, object]:
        point = {
            "step": step,
            "water_filling_effective_dim": int(
                metrics["water_filling_effective_dim"]
            ),
            "latent_effective_dim": int(metrics["latent_effective_dim"]),
        }
        if step >= self.min_steps:
            self.history.append(point)

        decision = {
            "should_stop": False,
            "reason": None,
            "window_checks": min(len(self.history), self.patience + 1),
            "water_filling_dim_range": None,
            "latent_effective_dim_range": None,
        }
        if step >= self.max_steps:
            decision.update({"should_stop": True, "reason": "max_steps"})
            return decision
        if len(self.history) < self.patience + 1:
            return decision

        window = self.history[-(self.patience + 1):]
        water_dims = [item["water_filling_effective_dim"] for item in window]
        latent_dims = [item["latent_effective_dim"] for item in window]
        water_range = max(water_dims) - min(water_dims)
        latent_range = max(latent_dims) - min(latent_dims)
        plateau = water_range <= self.tolerance and latent_range <= self.tolerance
        decision.update({
            "should_stop": plateau,
            "reason": "dimension_plateau" if plateau else None,
            "water_filling_dim_range": water_range,
            "latent_effective_dim_range": latent_range,
        })
        return decision
