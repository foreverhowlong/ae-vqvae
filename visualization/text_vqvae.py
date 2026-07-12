"""Visual diagnostics for the text VQ-VAE."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


PCAFitMode = Literal["balanced", "all"]


@dataclass(frozen=True)
class EncoderVectorCollection:
    """Flattened encoder vectors together with per-slot PAD ratios."""

    vectors: torch.Tensor
    pad_ratios: torch.Tensor


@dataclass(frozen=True)
class PCAComparisonResult:
    """Projected distributions and diagnostics produced by a shared PCA basis."""

    encoder_2d: np.ndarray
    codebook_2d: np.ndarray
    explained_variance_ratio: np.ndarray
    original_dimension: int
    encoder_mean_norm: float
    encoder_norm_std: float
    codebook_mean_norm: float
    codebook_norm_std: float
    encoder_to_nearest_code_mean_distance: float
    encoder_pairwise_mean_distance: float
    encoder_pad_ratios: np.ndarray | None
    fit_mode: PCAFitMode
    fit_points_per_distribution: int | None

    def metadata(self) -> dict:
        explained = self.explained_variance_ratio
        return {
            "encoder_points": int(len(self.encoder_2d)),
            "codebook_points": int(len(self.codebook_2d)),
            "original_dimension": self.original_dimension,
            "fit_mode": self.fit_mode,
            "fit_points_per_distribution": self.fit_points_per_distribution,
            "explained_variance_ratio": explained.tolist(),
            "total_explained_variance": float(explained.sum()),
            "encoder_mean_norm": self.encoder_mean_norm,
            "encoder_norm_std": self.encoder_norm_std,
            "codebook_mean_norm": self.codebook_mean_norm,
            "codebook_norm_std": self.codebook_norm_std,
            "encoder_to_nearest_code_mean_distance": self.encoder_to_nearest_code_mean_distance,
            "encoder_pairwise_mean_distance": self.encoder_pairwise_mean_distance,
            "pca_centroid_distance": float(
                np.linalg.norm(self.encoder_2d.mean(axis=0) - self.codebook_2d.mean(axis=0))
            ),
        }


@torch.no_grad()
def collect_encoder_vectors(
    model, data_loader, *, max_points: int = 8192
) -> EncoderVectorCollection:
    """Collect flattened encoder outputs on CPU without changing model state."""
    if max_points < 1:
        raise ValueError("max_points must be positive.")

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    chunks = []
    pad_ratio_chunks = []
    collected = 0
    try:
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            z_e = model.encode(input_ids, attention_mask=attention_mask)
            flat = z_e.reshape(-1, z_e.shape[-1]).detach().float().cpu()
            remaining = max_points - collected
            chunks.append(flat[:remaining])
            if attention_mask is None:
                pad_mask = input_ids.eq(model.config.pad_token_id).float()
            else:
                pad_mask = (attention_mask == 0).float()
            slot_pad_ratios = _slot_pad_ratios(pad_mask, z_e.shape[1])
            pad_ratio_chunks.append(slot_pad_ratios.reshape(-1)[:remaining].cpu())
            collected += min(flat.shape[0], remaining)
            if collected >= max_points:
                break
    finally:
        model.train(was_training)

    if not chunks:
        raise ValueError("Cannot collect encoder vectors from an empty data loader.")
    return EncoderVectorCollection(
        vectors=torch.cat(chunks, dim=0),
        pad_ratios=torch.cat(pad_ratio_chunks, dim=0),
    )


def compare_vector_distributions_pca(
    encoder_vectors: torch.Tensor | np.ndarray,
    codebook_vectors: torch.Tensor | np.ndarray,
    *,
    encoder_pad_ratios: torch.Tensor | np.ndarray | None = None,
    fit_mode: PCAFitMode = "balanced",
    random_state: int = 0,
) -> PCAComparisonResult:
    """Project two vector distributions through a single fitted PCA basis.

    ``balanced`` fits PCA with the same number of vectors from each distribution,
    preventing an arbitrary sample-count ratio from determining the PCA axes.
    All input vectors are transformed and retained in the returned result.
    """
    encoder = _as_numpy_matrix(encoder_vectors, "encoder_vectors")
    codebook = _as_numpy_matrix(codebook_vectors, "codebook_vectors")
    pad_ratios = _as_pad_ratios(encoder_pad_ratios, len(encoder))
    if encoder.shape[1] != codebook.shape[1]:
        raise ValueError(
            "Encoder and codebook dimensions must match, got "
            f"{encoder.shape[1]} and {codebook.shape[1]}."
        )
    if encoder.shape[1] < 2 or len(encoder) + len(codebook) < 2:
        raise ValueError(
            "PCA visualization needs at least two vectors with at least two dimensions."
        )

    fit_points_per_distribution = None
    if fit_mode == "balanced":
        fit_points_per_distribution = min(len(encoder), len(codebook))
        rng = np.random.default_rng(random_state)
        encoder_fit = _sample_rows(encoder, fit_points_per_distribution, rng)
        codebook_fit = _sample_rows(codebook, fit_points_per_distribution, rng)
        fit_vectors = np.concatenate([encoder_fit, codebook_fit], axis=0)
    elif fit_mode == "all":
        fit_vectors = np.concatenate([encoder, codebook], axis=0)
    else:
        raise ValueError(f"Unknown PCA fit mode {fit_mode!r}; expected 'balanced' or 'all'.")

    pca = PCA(n_components=2)
    pca.fit(fit_vectors)
    return PCAComparisonResult(
        encoder_2d=pca.transform(encoder),
        codebook_2d=pca.transform(codebook),
        explained_variance_ratio=pca.explained_variance_ratio_,
        original_dimension=int(encoder.shape[1]),
        encoder_mean_norm=float(np.linalg.norm(encoder, axis=1).mean()),
        encoder_norm_std=float(np.linalg.norm(encoder, axis=1).std()),
        codebook_mean_norm=float(np.linalg.norm(codebook, axis=1).mean()),
        codebook_norm_std=float(np.linalg.norm(codebook, axis=1).std()),
        encoder_to_nearest_code_mean_distance=_mean_nearest_l2_distance(encoder, codebook),
        encoder_pairwise_mean_distance=_mean_pairwise_l2_distance(encoder),
        encoder_pad_ratios=pad_ratios,
        fit_mode=fit_mode,
        fit_points_per_distribution=fit_points_per_distribution,
    )


def render_pca_comparison(result: PCAComparisonResult, output_path: str | Path) -> Path:
    """Render a PCA comparison result to a PNG file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoder_2d = result.encoder_2d
    codebook_2d = result.codebook_2d

    fig, ax = plt.subplots(figsize=(9, 7))
    try:
        encoder_colors = result.encoder_pad_ratios
        if encoder_colors is None:
            encoder_colors = np.zeros(len(encoder_2d))
        encoder_scatter = ax.scatter(
            encoder_2d[:, 0],
            encoder_2d[:, 1],
            s=9,
            alpha=0.18,
            linewidths=0,
            label=f"Encoder outputs (n={len(encoder_2d):,})",
            c=encoder_colors,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        if result.encoder_pad_ratios is not None:
            colorbar = fig.colorbar(encoder_scatter, ax=ax, pad=0.02)
            colorbar.set_label("PAD ratio within slot")
        ax.scatter(
            codebook_2d[:, 0],
            codebook_2d[:, 1],
            s=24,
            alpha=0.75,
            linewidths=0.4,
            edgecolors="#8B1A1A",
            label=f"Codebook vectors (n={len(codebook_2d):,})",
            color="#D9534F",
            rasterized=True,
        )
        ax.scatter(
            *encoder_2d.mean(axis=0),
            s=150,
            marker="X",
            color="#174A70",
            edgecolors="white",
            linewidths=0.8,
            label="Encoder centroid",
            zorder=4,
        )
        ax.scatter(
            *codebook_2d.mean(axis=0),
            s=150,
            marker="P",
            color="#8B1A1A",
            edgecolors="white",
            linewidths=0.8,
            label="Codebook centroid",
            zorder=4,
        )
        explained = result.explained_variance_ratio
        ax.set_xlabel(f"PC1 ({explained[0]:.1%} explained variance)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1%} explained variance)")
        ax.set_title("Text VQ-VAE initialization: encoder outputs vs. codebook")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)
    return output_path


def save_pca_metadata(result: PCAComparisonResult, output_path: str | Path) -> Path:
    """Save PCA diagnostics as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result.metadata(), handle, indent=2)
    return output_path


@torch.no_grad()
def plot_initial_latent_codebook_pca(
    model,
    data_loader,
    output_path: str | Path,
    *,
    max_encoder_points: int = 8192,
    fit_mode: PCAFitMode = "balanced",
    random_state: int = 0,
) -> dict:
    """Backward-compatible convenience wrapper around the composable PCA steps."""
    encoder_vectors = collect_encoder_vectors(
        model, data_loader, max_points=max_encoder_points
    )
    result = compare_vector_distributions_pca(
        encoder_vectors.vectors,
        model.quantizer.codebook.weight,
        encoder_pad_ratios=encoder_vectors.pad_ratios,
        fit_mode=fit_mode,
        random_state=random_state,
    )
    output_path = render_pca_comparison(result, output_path)
    save_pca_metadata(result, output_path.with_suffix(".json"))
    return result.metadata()


def _as_numpy_matrix(vectors: torch.Tensor | np.ndarray, name: str) -> np.ndarray:
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().float().cpu().numpy()
    else:
        vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError(f"{name} must be a non-empty rank-2 matrix.")
    if not np.isfinite(vectors).all():
        raise ValueError(f"{name} contains non-finite values.")
    return vectors


def _as_pad_ratios(
    pad_ratios: torch.Tensor | np.ndarray | None, count: int
) -> np.ndarray | None:
    if pad_ratios is None:
        return None
    if isinstance(pad_ratios, torch.Tensor):
        pad_ratios = pad_ratios.detach().cpu().numpy()
    pad_ratios = np.asarray(pad_ratios, dtype=np.float32)
    if pad_ratios.ndim != 1 or len(pad_ratios) != count:
        raise ValueError(f"encoder_pad_ratios must have shape ({count},), got {pad_ratios.shape}.")
    if not np.isfinite(pad_ratios).all() or np.any((pad_ratios < 0) | (pad_ratios > 1)):
        raise ValueError("encoder_pad_ratios must contain finite values in [0, 1].")
    return pad_ratios


def _slot_pad_ratios(pad_mask: torch.Tensor, latent_slots: int) -> torch.Tensor:
    """Match adaptive_avg_pool1d's input bins and average PAD masks per bin."""
    seq_len = pad_mask.shape[1]
    ratios = []
    for slot in range(latent_slots):
        start = (slot * seq_len) // latent_slots
        end = ((slot + 1) * seq_len + latent_slots - 1) // latent_slots
        ratios.append(pad_mask[:, start:end].mean(dim=1))
    return torch.stack(ratios, dim=1)


def _mean_nearest_l2_distance(encoder: np.ndarray, codebook: np.ndarray) -> float:
    encoder_tensor = torch.from_numpy(encoder)
    codebook_tensor = torch.from_numpy(codebook)
    total = 0.0
    count = 0
    for start in range(0, len(encoder_tensor), 512):
        distances = torch.cdist(encoder_tensor[start : start + 512], codebook_tensor)
        total += float(distances.min(dim=1).values.sum())
        count += distances.shape[0]
    return total / count


def _mean_pairwise_l2_distance(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        raise ValueError("At least two encoder vectors are required for pairwise distance.")
    tensor = torch.from_numpy(vectors)
    total = 0.0
    pair_count = 0
    for start in range(0, len(tensor) - 1, 512):
        end = min(start + 512, len(tensor))
        if end < len(tensor):
            distances = torch.cdist(tensor[start:end], tensor[end:])
            total += float(distances.sum())
            pair_count += distances.numel()
        within = torch.pdist(tensor[start:end], p=2)
        total += float(within.sum())
        pair_count += within.numel()
    return total / pair_count


def _sample_rows(matrix: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(matrix) == count:
        return matrix
    return matrix[rng.choice(len(matrix), size=count, replace=False)]
