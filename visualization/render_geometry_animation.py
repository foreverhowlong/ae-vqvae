"""Render training geometry snapshots using one PCA basis shared by every frame."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio_ffmpeg
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from common.text_vqvae_config import GeometryRenderBasis


@dataclass(frozen=True)
class AnimationScales:
    """Axis limits and histogram bins shared by every animation frame."""

    pca_xlim: tuple[float, float]
    pca_ylim: tuple[float, float]
    norm_bins: np.ndarray
    norm_ylim: tuple[float, float]
    nearest_bins: np.ndarray
    nearest_ylim: tuple[float, float]
    rank_xlim: tuple[float, float]
    rank_ylim: tuple[float, float]


@dataclass(frozen=True)
class SnapshotEncoderView:
    """PAD-free arrays aligned to the encoder rows in one snapshot."""

    encoder: np.ndarray
    pad_ratios: np.ndarray | None
    slot_indices: np.ndarray | None
    assignments: np.ndarray | None
    nearest_distances: np.ndarray | None


_GEOMETRY_METRIC_PANELS = {
    "valid_probe_points": (
        "Valid latent points in fixed geometry probe",
        "Valid latent vectors (count)",
    ),
    "encoder_mean_norm": (
        "Mean encoder latent norm",
        "Mean vector L2 norm (arbitrary latent units)",
    ),
    "encoder_norm_std": (
        "Encoder latent norm spread",
        "L2-norm standard deviation (arbitrary latent units)",
    ),
    "encoder_pairwise_mean_distance": (
        "Mean pairwise encoder distance",
        "Mean pairwise L2 distance (arbitrary latent units)",
    ),
    "nearest_code_distance_p10": (
        "Encoder-to-assigned-code distance (10th percentile)",
        "Quantizer-space L2 distance (arbitrary latent units)",
    ),
    "nearest_code_distance_p50": (
        "Encoder-to-assigned-code distance (median)",
        "Quantizer-space L2 distance (arbitrary latent units)",
    ),
    "nearest_code_distance_p90": (
        "Encoder-to-assigned-code distance (90th percentile)",
        "Quantizer-space L2 distance (arbitrary latent units)",
    ),
    "win_count_gini": (
        "Code-assignment inequality on fixed probe",
        "Gini coefficient [0 = equal, 1 = concentrated]",
    ),
    "centroid_distance": (
        "Encoder-to-codebook centroid distance",
        "Centroid L2 distance (arbitrary latent units)",
    ),
}


def _pca_component_label(pca: PCA, component: int) -> str:
    explained = pca.explained_variance_ratio_[component]
    return (
        f"PC{component + 1} score "
        f"(arbitrary latent units; {explained:.1%} explained variance)"
    )


def _geometry_metric_panel_labels(key: str) -> tuple[str, str]:
    """Return a human-readable title and y-axis unit for a geometry metric."""
    return _GEOMETRY_METRIC_PANELS.get(
        key,
        (key.replace("_", " ").capitalize(), "Metric value (source-defined units)"),
    )


def _encoder_spectrum_metrics(
    encoder: np.ndarray,
    *,
    max_twonn_points: int = 2048,
) -> dict[str, object]:
    """Compute offline spectrum, smooth-rank, and TwoNN diagnostics."""
    points = np.asarray(encoder, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("Encoder spectrum diagnostics need a non-empty matrix.")

    centered = points - points.mean(axis=0, keepdims=True)
    covariance = (
        centered.T @ centered / (len(centered) - 1)
        if len(centered) > 1
        else np.zeros((points.shape[1], points.shape[1]), dtype=np.float64)
    )
    eigenvalues = np.linalg.eigvalsh(covariance).clip(min=0)[::-1]
    total_variance = eigenvalues.sum()
    participation_ratio = (
        float(total_variance**2 / np.square(eigenvalues).sum())
        if total_variance > 0
        else 0.0
    )

    singular_values = np.sqrt(eigenvalues)
    singular_sum = singular_values.sum()
    if singular_sum > 0:
        probabilities = singular_values[singular_values > 0] / singular_sum
        rankme = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    else:
        rankme = 0.0

    twonn_points = points
    if len(twonn_points) > max_twonn_points:
        sample_indices = np.linspace(
            0,
            len(twonn_points) - 1,
            num=max_twonn_points,
        ).round().astype(np.int64)
        twonn_points = twonn_points[sample_indices]
    twonn_intrinsic_dim = 0.0
    if len(twonn_points) >= 3:
        distances = NearestNeighbors(n_neighbors=3).fit(twonn_points).kneighbors(
            twonn_points,
            return_distance=True,
        )[0][:, 1:3]
        valid = (
            np.isfinite(distances).all(axis=1)
            & (distances[:, 0] > 0)
            & (distances[:, 1] > distances[:, 0])
        )
        if valid.any():
            mean_log_ratio = np.log(
                distances[valid, 1] / distances[valid, 0]
            ).mean()
            if np.isfinite(mean_log_ratio) and mean_log_ratio > 0:
                twonn_intrinsic_dim = float(1.0 / mean_log_ratio)

    return {
        "pca_eigenvalues": eigenvalues.tolist(),
        "participation_ratio": participation_ratio,
        "rankme": rankme,
        "twonn_intrinsic_dim": twonn_intrinsic_dim,
        "twonn_points": int(len(twonn_points)),
    }


def _snapshot_encoder_view(data) -> SnapshotEncoderView:
    """Load aligned PAD-free arrays from current and legacy snapshots.

    Format v2 snapshots are already filtered with the model's exact
    ``latent_mask``. Older snapshots stored every slot, so compatibility
    rendering uses ``pad_ratio <= 0.5`` as the explicitly approximate fallback.
    Legacy snapshots without PAD ratios remain readable and keep every row.
    """
    encoder = data["z_e"].astype(np.float32)
    original_count = len(encoder)
    format_version = (
        int(np.asarray(data["geometry_format_version"]).item())
        if "geometry_format_version" in data
        else 1
    )
    keep = np.ones(original_count, dtype=bool)
    if (
        format_version < 2
        and "pad_ratios" in data
        and len(data["pad_ratios"]) == original_count
    ):
        keep = data["pad_ratios"].astype(np.float32) <= 0.5
    if not keep.any():
        raise ValueError("Geometry snapshot contains no valid latent points.")

    def aligned(name: str, dtype):
        if name not in data:
            return None
        values = data[name].astype(dtype)
        return values[keep] if len(values) == original_count else values

    return SnapshotEncoderView(
        encoder=encoder[keep],
        pad_ratios=aligned("pad_ratios", np.float32),
        slot_indices=aligned("slot_indices", np.int64),
        assignments=aligned("assignments", np.int64),
        nearest_distances=aligned("nearest_distances", np.float32),
    )


def load_snapshots(run_dir: Path):
    paths = list((run_dir / "geometry").glob("step*.npz"))
    if not paths:
        raise FileNotFoundError(f"No geometry snapshots found under {run_dir / 'geometry'}")
    parsed = []
    suffix_order = {"": 0, "_pre_kmeans": 1, "_post_kmeans": 2}
    for path in paths:
        match = re.fullmatch(r"step(\d+)(.*)", path.stem)
        if match is None:
            continue
        parsed.append((int(match.group(1)), suffix_order.get(match.group(2), 3), path))
    return [(step, path) for step, _, path in sorted(parsed)]


def _snapshot_label(path: Path, *, has_codebook: bool) -> str:
    if path.stem.endswith("_pre_kmeans"):
        return "AE warmup end (before K-means)"
    if path.stem.endswith("_post_kmeans"):
        return "VQ initialization (after K-means)"
    return "VQ phase" if has_codebook else "AE warmup / continuous latent"


def fit_shared_pca(
    snapshots,
    basis: GeometryRenderBasis,
    random_state: int = 0,
    max_fit_points: int = 8192,
) -> PCA:
    if basis == "t0":
        selected = snapshots[:1]
    elif basis == "first_last":
        selected = snapshots[:1] if len(snapshots) == 1 else [snapshots[0], snapshots[-1]]
    elif basis == "pooled":
        selected = snapshots
    else:
        raise ValueError(f"Unknown basis {basis!r}")

    encoders, codebooks = [], []
    for _, path in selected:
        with np.load(path) as data:
            encoders.append(_snapshot_encoder_view(data).encoder)
            if "codebook" in data:
                codebooks.append(data["codebook"].astype(np.float32))
    encoder = np.concatenate(encoders)
    rng = np.random.default_rng(random_state)
    if codebooks:
        codebook = np.concatenate(codebooks)
        count = min(len(encoder), len(codebook), max_fit_points)
        encoder_fit = encoder[rng.choice(len(encoder), count, replace=False)]
        codebook_fit = codebook[rng.choice(len(codebook), count, replace=False)]
        fit_points = np.concatenate([encoder_fit, codebook_fit])
    else:
        count = min(len(encoder), max_fit_points)
        fit_points = encoder[rng.choice(len(encoder), count, replace=False)]
    return PCA(n_components=2).fit(fit_points)


def render_frame(
    step: int,
    path: Path,
    pca: PCA,
    output_path: Path,
    scales: AnimationScales,
    *,
    run_name: str | None = None,
    latent_phase_label: str | None = None,
) -> None:
    with np.load(path) as data:
        view = _snapshot_encoder_view(data)
        encoder = view.encoder
        if "codebook" not in data:
            if view.slot_indices is None or view.pad_ratios is None:
                raise ValueError(
                    "Continuous geometry snapshots require slot_indices and pad_ratios."
                )
            snapshot_label = _snapshot_label(path, has_codebook=False)
            if not path.stem.endswith("_pre_kmeans") and latent_phase_label:
                snapshot_label = latent_phase_label
            _render_continuous_frame(
                step,
                encoder,
                view.slot_indices,
                view.pad_ratios,
                pca,
                output_path,
                scales,
                run_name=run_name,
                phase_label=snapshot_label,
            )
            return
        codebook = data["codebook"].astype(np.float32)
        assignments = view.assignments
        nearest = view.nearest_distances
    if assignments is None:
        raise ValueError("VQ geometry snapshots require assignments.")
    wins = np.bincount(assignments, minlength=len(codebook))
    alive = wins > 0
    encoder_2d = pca.transform(encoder)
    codebook_2d = pca.transform(codebook)
    if nearest is None:
        # Compatibility fallback for pre-v2 snapshots that did not persist the
        # quantizer's own distance metric.
        nearest = np.linalg.norm(encoder - codebook[assignments], axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax = axes[0, 0]
    ax.scatter(
        encoder_2d[:, 0], encoder_2d[:, 1],
        color="#4C78A8", s=7, alpha=.18, linewidths=0,
        rasterized=True, label="encoder output", zorder=1,
    )
    ax.scatter(
        codebook_2d[~alive, 0], codebook_2d[~alive, 1],
        color="#3F3F3F", marker="x", s=22, alpha=.72, linewidths=.8,
        rasterized=True, label="0 assignments on fixed probe", zorder=3,
    )
    ax.scatter(
        codebook_2d[alive, 0], codebook_2d[alive, 1],
        color="#E45756", marker="*", s=72, alpha=.98, linewidths=.55,
        edgecolors="#5B1717", rasterized=True,
        label="≥1 assignment on fixed probe", zorder=5,
    )
    ax.set(
        xlim=scales.pca_xlim,
        ylim=scales.pca_ylim,
        title="Shared-basis PCA: encoder outputs and codebook state",
        xlabel=_pca_component_label(pca, 0),
        ylabel=_pca_component_label(pca, 1),
    )
    ax.legend(loc="upper right", fontsize=8)

    axes[0, 1].hist(
        np.linalg.norm(encoder, axis=1), bins=scales.norm_bins,
        color="#4C78A8", alpha=.62, density=True, label="encoder",
    )
    axes[0, 1].hist(
        np.linalg.norm(codebook, axis=1), bins=scales.norm_bins,
        color="#F58518", alpha=.58, density=True, label="codebook",
    )
    axes[0, 1].set(
        xlim=(scales.norm_bins[0], scales.norm_bins[-1]),
        ylim=scales.norm_ylim,
        title="Encoder and codebook vector-norm distributions",
        xlabel="Vector L2 norm (arbitrary latent units)",
        ylabel="Probability density (per arbitrary latent unit)",
    )
    axes[0, 1].legend(loc="upper right")
    axes[1, 0].hist(nearest, bins=scales.nearest_bins, color="#4C78A8", alpha=.8)
    axes[1, 0].set(
        xlim=(scales.nearest_bins[0], scales.nearest_bins[-1]),
        ylim=scales.nearest_ylim,
        title="Distance to assigned nearest code",
        xlabel="Quantizer-space L2 distance (arbitrary latent units)",
        ylabel="Valid probe points per bin (count)",
    )
    ranked = np.sort(wins)[::-1]
    axes[1, 1].plot(np.arange(1, len(ranked) + 1), ranked, color="#E45756")
    axes[1, 1].set(
        xlim=scales.rank_xlim,
        ylim=scales.rank_ylim,
        title="Fixed-probe code-assignment rank curve",
        xlabel="Code rank by probe assignments (1 = most used)",
        ylabel="Fixed-probe assignments per code (count)",
        yscale="symlog",
        xscale="log",
    )
    for axis in axes.flat:
        axis.grid(alpha=.2)
        axis.set_title(axis.get_title(), pad=9)
    fig.suptitle(
        f"{_snapshot_label(path, has_codebook=True)} — step {step:,} "
        f"— valid probe points {len(encoder):,}",
        fontsize=15,
        x=.5,
        y=.975,
    )
    if run_name:
        fig.text(
            .995, .995, f"run: {run_name}",
            ha="right", va="top", fontsize=7, color="0.35",
        )
    fig.subplots_adjust(left=.075, right=.975, bottom=.075, top=.91, wspace=.22, hspace=.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _render_continuous_frame(
    step: int,
    encoder: np.ndarray,
    slot_indices: np.ndarray,
    pad_ratios: np.ndarray,
    pca: PCA,
    output_path: Path,
    scales: AnimationScales,
    *,
    run_name: str | None = None,
    phase_label: str = "Continuous AE",
) -> None:
    """Render latent-only diagnostics for a continuous autoencoder."""
    encoder_2d = pca.transform(encoder)
    norms = np.linalg.norm(encoder, axis=1)
    unique_slots = np.unique(slot_indices)
    slot_mean_norms = [
        float(norms[slot_indices == slot].mean()) for slot in unique_slots
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    scatter = axes[0, 0].scatter(
        encoder_2d[:, 0],
        encoder_2d[:, 1],
        c=slot_indices,
        cmap="viridis",
        s=8,
        alpha=.28,
        linewidths=0,
        rasterized=True,
    )
    axes[0, 0].set(
        xlim=scales.pca_xlim,
        ylim=scales.pca_ylim,
        title="Shared-basis PCA of continuous encoder latents",
        xlabel=_pca_component_label(pca, 0),
        ylabel=_pca_component_label(pca, 1),
    )
    fig.colorbar(
        scatter,
        ax=axes[0, 0],
        label="Latent slot index (0-based)",
    )

    axes[0, 1].hist(
        norms,
        bins=scales.norm_bins,
        color="#4C78A8",
        alpha=.75,
        density=True,
    )
    axes[0, 1].set(
        xlim=(scales.norm_bins[0], scales.norm_bins[-1]),
        ylim=scales.norm_ylim,
        title="Continuous latent vector norms",
        xlabel="Vector L2 norm (arbitrary latent units)",
        ylabel="Probability density (per arbitrary latent unit)",
    )

    axes[1, 0].plot(unique_slots, slot_mean_norms, marker=".", color="#F58518")
    axes[1, 0].set(
        title="Mean latent norm by slot",
        xlabel="Latent slot index (0-based)",
        ylabel="Mean vector L2 norm (arbitrary latent units)",
    )

    axes[1, 1].scatter(
        pad_ratios,
        norms,
        s=8,
        alpha=.22,
        color="#54A24B",
        linewidths=0,
        rasterized=True,
    )
    axes[1, 1].set(
        title="Padding exposure versus latent norm",
        xlabel="PAD-token fraction in slot receptive segment [0, 1]",
        ylabel="Vector L2 norm (arbitrary latent units)",
    )
    for axis in axes.flat:
        axis.grid(alpha=.2)
        axis.set_title(axis.get_title(), pad=9)
    fig.suptitle(
        f"{phase_label}: latent geometry — step {step:,}",
        fontsize=15,
        x=.5,
        y=.975,
    )
    if run_name:
        fig.text(
            .995, .995, f"run: {run_name}",
            ha="right", va="top", fontsize=7, color="0.35",
        )
    fig.subplots_adjust(left=.075, right=.975, bottom=.075, top=.91, wspace=.28, hspace=.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def projection_limits(snapshots, pca: PCA):
    points = []
    for _, path in snapshots:
        with np.load(path) as data:
            points.append(pca.transform(_snapshot_encoder_view(data).encoder))
            if "codebook" in data:
                points.append(pca.transform(data["codebook"].astype(np.float32)))
    merged = np.concatenate(points)
    low = np.quantile(merged, .002, axis=0)
    high = np.quantile(merged, .998, axis=0)
    margin = np.maximum((high - low) * .05, 1e-3)
    return ((low[0] - margin[0], high[0] + margin[0]), (low[1] - margin[1], high[1] + margin[1]))


def compute_animation_scales(snapshots, pca: PCA) -> AnimationScales:
    """Precompute every axis scale once so video frames cannot visually jump."""
    pca_xlim, pca_ylim = projection_limits(snapshots, pca)
    encoder_norm_frames = []
    codebook_norm_frames = []
    nearest_distance_frames = []
    maximum_win_count = 0
    codebook_size = 0
    for _, path in snapshots:
        with np.load(path) as data:
            view = _snapshot_encoder_view(data)
            encoder = view.encoder
            encoder_norm_frames.append(np.linalg.norm(encoder, axis=1))
            if "codebook" in data:
                codebook = data["codebook"].astype(np.float32)
                assignments = view.assignments
                if assignments is None:
                    raise ValueError("VQ geometry snapshots require assignments.")
                codebook_norm_frames.append(np.linalg.norm(codebook, axis=1))
                nearest = view.nearest_distances
                if nearest is None:
                    nearest = np.linalg.norm(
                        encoder - codebook[assignments],
                        axis=1,
                    )
                nearest_distance_frames.append(nearest)
                wins = np.bincount(assignments, minlength=len(codebook))
                maximum_win_count = max(
                    maximum_win_count, int(wins.max(initial=0))
                )
                codebook_size = max(codebook_size, len(codebook))

    encoder_norms = np.concatenate(encoder_norm_frames)
    all_norms = encoder_norms
    if codebook_norm_frames:
        all_norms = np.concatenate([encoder_norms, *codebook_norm_frames])
    norm_bins = _fixed_bin_edges(all_norms, 50)
    norm_density_max = max(
        float(np.histogram(values, bins=norm_bins, density=True)[0].max(initial=0))
        for values in encoder_norm_frames + codebook_norm_frames
    )
    if nearest_distance_frames:
        nearest_distances = np.concatenate(nearest_distance_frames)
        nearest_bins = _fixed_bin_edges(nearest_distances, 60)
        nearest_count_max = max(
            int(np.histogram(values, bins=nearest_bins)[0].max(initial=0))
            for values in nearest_distance_frames
        )
    else:
        nearest_bins = np.linspace(0.0, 1.0, 61)
        nearest_count_max = 1
    return AnimationScales(
        pca_xlim=pca_xlim,
        pca_ylim=pca_ylim,
        norm_bins=norm_bins,
        norm_ylim=(0.0, max(float(norm_density_max) * 1.08, 1e-3)),
        nearest_bins=nearest_bins,
        nearest_ylim=(0.0, max(float(nearest_count_max) * 1.08, 1.0)),
        rank_xlim=(1.0, float(max(codebook_size, 2))),
        rank_ylim=(0.0, max(float(maximum_win_count) * 1.08, 1.0)),
    )


def _fixed_bin_edges(values: np.ndarray, count: int) -> np.ndarray:
    low = float(values.min())
    high = float(values.max())
    if np.isclose(low, high):
        margin = max(abs(low) * .05, 1e-3)
    else:
        margin = (high - low) * .025
    return np.linspace(low - margin, high + margin, count + 1)


def render_code_trajectories(
    snapshots,
    pca: PCA,
    output_path: Path,
    random_state: int = 0,
    *,
    run_name: str | None = None,
) -> None:
    with np.load(snapshots[-1][1]) as final:
        final_view = _snapshot_encoder_view(final)
        if final_view.assignments is None:
            raise ValueError("VQ geometry snapshots require assignments.")
        wins = np.bincount(
            final_view.assignments,
            minlength=len(final["codebook"]),
        )
    top = np.argsort(wins)[-min(16, len(wins)):][::-1]
    dead = np.setdiff1d(np.flatnonzero(wins == 0), top, assume_unique=False)
    rng = np.random.default_rng(random_state)
    sampled_dead = rng.choice(dead, min(16, len(dead)), replace=False) if len(dead) else np.array([], dtype=int)
    selected = np.concatenate([top, sampled_dead])
    tracks = []
    steps = []
    for step, path in snapshots:
        with np.load(path) as data:
            tracks.append(pca.transform(data["codebook"].astype(np.float32)[selected]))
        steps.append(step)
    tracks = np.stack(tracks)

    fig, ax = plt.subplots(figsize=(10, 8))
    for column, code_id in enumerate(selected):
        is_top = column < len(top)
        color = plt.cm.tab20(column % 20) if is_top else "#aaaaaa"
        ax.plot(tracks[:, column, 0], tracks[:, column, 1], color=color,
                alpha=.9 if is_top else .45, linewidth=1.6 if is_top else 1.0)
        ax.scatter(tracks[-1, column, 0], tracks[-1, column, 1], s=20, color=color)
        ax.annotate(str(code_id), tracks[-1, column], fontsize=7, color=color)
    ax.set(
        title=(
            "Code trajectories in the shared PCA basis\n"
            "(top-16 final probe winners + up to 16 codes unused by final probe)"
        ),
        xlabel=_pca_component_label(pca, 0),
        ylabel=_pca_component_label(pca, 1),
    )
    ax.grid(alpha=.2)
    if run_name:
        fig.text(
            .995, .995, f"run: {run_name}",
            ha="right", va="top", fontsize=7, color="0.35",
        )
    fig.tight_layout(rect=(0, 0, 1, .985))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_latent_centroid_trajectory(
    snapshots,
    pca: PCA,
    output_path: Path,
    *,
    run_name: str | None = None,
    phase_name: str = "Continuous AE",
) -> None:
    """Render continuous-latent center and spread dynamics across snapshots."""
    steps = []
    centroids = []
    mean_norms = []
    norm_stds = []
    for step, path in snapshots:
        with np.load(path) as data:
            encoder = _snapshot_encoder_view(data).encoder
        steps.append(step)
        centroids.append(encoder.mean(axis=0))
        norms = np.linalg.norm(encoder, axis=1)
        mean_norms.append(float(norms.mean()))
        norm_stds.append(float(norms.std()))
    projected = pca.transform(np.stack(centroids))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(projected[:, 0], projected[:, 1], marker="o", color="#4C78A8")
    for index in {0, len(steps) - 1}:
        axes[0].annotate(f"step {steps[index]:,}", projected[index], fontsize=8)
    axes[0].set(
        title=f"{phase_name} latent centroid trajectory in shared PCA basis",
        xlabel=_pca_component_label(pca, 0),
        ylabel=_pca_component_label(pca, 1),
    )
    axes[1].plot(steps, mean_norms, label="mean norm", color="#F58518")
    axes[1].plot(steps, norm_stds, label="norm std", color="#54A24B")
    axes[1].set(
        title=f"{phase_name} latent norm dynamics",
        xlabel="Optimizer step (parameter updates)",
        ylabel="Encoder latent L2 norm (arbitrary latent units)",
    )
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.2)
    if run_name:
        fig.text(
            .995, .995, f"run: {run_name}",
            ha="right", va="top", fontsize=7, color="0.35",
        )
    fig.tight_layout(rect=(0, 0, 1, .985))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_warmup_transition(
    snapshots,
    output_path: Path,
    *,
    run_name: str | None = None,
) -> None:
    pre = next(
        ((step, path) for step, path in snapshots if path.stem.endswith("_pre_kmeans")),
        None,
    )
    post = next(
        ((step, path) for step, path in snapshots if path.stem.endswith("_post_kmeans")),
        None,
    )
    if pre is None or post is None:
        raise ValueError("Warmup transition snapshots are incomplete.")
    with np.load(snapshots[0][1]) as start_data:
        start = _snapshot_encoder_view(start_data).encoder
    with np.load(pre[1]) as pre_data:
        end = _snapshot_encoder_view(pre_data).encoder
    with np.load(post[1]) as post_data:
        codebook = post_data["codebook"].astype(np.float32)

    fit_points = np.concatenate([start, end, codebook])
    pca = PCA(n_components=2).fit(fit_points)
    projected = [pca.transform(values) for values in (start, end, codebook)]
    merged = np.concatenate(projected)
    low = np.quantile(merged, .002, axis=0)
    high = np.quantile(merged, .998, axis=0)
    margin = np.maximum((high - low) * .06, 1e-3)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    panels = (
        ("AE warmup start", projected[0], False),
        ("AE warmup end, before K-means", projected[1], False),
        ("K-means C0 on warmup latents", projected[1], True),
    )
    for ax, (title, latent, show_codes) in zip(axes, panels):
        ax.scatter(latent[:, 0], latent[:, 1], s=7, alpha=.2, linewidths=0)
        if show_codes:
            ax.scatter(
                projected[2][:, 0],
                projected[2][:, 1],
                marker="*",
                s=55,
                color="#E45756",
                edgecolors="#5B1717",
                linewidths=.4,
                label="K-means C0",
            )
            ax.legend(fontsize=8)
        ax.set(
            title=title,
            xlabel=_pca_component_label(pca, 0),
            xlim=(low[0] - margin[0], high[0] + margin[0]),
            ylim=(low[1] - margin[1], high[1] + margin[1]),
        )
        ax.grid(alpha=.2)
    axes[0].set_ylabel(_pca_component_label(pca, 1))
    fig.suptitle(f"AE warmup → K-means transition at step {pre[0]:,}", fontsize=15)
    if run_name:
        fig.text(
            .995, .995, f"run: {run_name}",
            ha="right", va="top", fontsize=7, color="0.35",
        )
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_metric_series(
    run_dir: Path,
    output_path: Path,
    *,
    run_name: str | None = None,
) -> None:
    rows = []
    transition_step = None
    with (run_dir / "metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == "geometry":
                rows.append(row)
            elif row.get("split") == "phase_transition" and transition_step is None:
                transition_step = row["step"]
    if not rows:
        raise ValueError("metrics.jsonl contains no geometry rows")

    spectrum_rows = []
    for step, path in load_snapshots(run_dir):
        with np.load(path) as data:
            spectrum_rows.append({
                "step": step,
                **_encoder_spectrum_metrics(
                    _snapshot_encoder_view(data).encoder
                ),
            })

    excluded = {
        "split", "step", "elapsed_sec", "phase", "event", "quantizer_active",
        "participation_ratio",
        # Legacy geometry rows may contain this noncanonical count. Keep the
        # files readable, but never revive it as a current metric plot.
        "used_codes",
    }
    keys = sorted({
        key
        for row in rows
        for key, value in row.items()
        if key not in excluded and isinstance(value, (int, float))
    })
    columns = 3
    panel_count = len(keys) + 3
    rows_count = (panel_count + columns - 1) // columns
    fig, axes = plt.subplots(rows_count, columns, figsize=(15, 3.5 * rows_count), squeeze=False)
    steps = [row["step"] for row in rows]
    flat_axes = list(axes.flat)

    spectrum_axis = flat_axes[0]
    spectrum_steps = [row["step"] for row in spectrum_rows]
    step_min = min(spectrum_steps)
    step_max = max(spectrum_steps)
    if step_min == step_max:
        step_min -= 0.5
        step_max += 0.5
    normalization = matplotlib.colors.Normalize(step_min, step_max)
    colormap = matplotlib.colormaps["viridis"]
    for row in spectrum_rows:
        eigenvalues = np.asarray(row["pca_eigenvalues"])
        positive = eigenvalues > 0
        spectrum_axis.plot(
            np.arange(1, len(eigenvalues) + 1)[positive],
            eigenvalues[positive],
            color=colormap(normalization(row["step"])),
            alpha=.7,
            linewidth=1,
        )
    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=normalization, cmap=colormap),
        ax=spectrum_axis,
        pad=.02,
    )
    colorbar.set_label("Optimizer step (parameter updates)")
    spectrum_axis.set(
        title="Encoder covariance eigenvalue spectrum",
        xlabel="Covariance eigenvalue rank (1 = largest)",
        ylabel="Covariance eigenvalue (arbitrary latent units²)",
        yscale="log",
    )
    spectrum_axis.grid(alpha=.2)

    rank_axis = flat_axes[1]
    rank_axis.plot(
        spectrum_steps,
        [row["participation_ratio"] for row in spectrum_rows],
        marker=".",
        linewidth=1.2,
        label="participation ratio",
    )
    rank_axis.plot(
        spectrum_steps,
        [row["rankme"] for row in spectrum_rows],
        marker=".",
        linewidth=1.2,
        label="RankMe",
    )
    rank_axis.set(
        title="Participation ratio and RankMe",
        xlabel="Optimizer step (parameter updates)",
        ylabel="Effective rank (dimensions)",
    )
    rank_axis.legend()

    twonn_axis = flat_axes[2]
    twonn_axis.plot(
        spectrum_steps,
        [row["twonn_intrinsic_dim"] for row in spectrum_rows],
        marker=".",
        linewidth=1.2,
    )
    twonn_axis.set(
        title="TwoNN intrinsic dimension",
        xlabel="Optimizer step (parameter updates)",
        ylabel="Estimated intrinsic dimension (dimensions)",
    )

    for ax in (rank_axis, twonn_axis):
        if transition_step is not None:
            ax.axvspan(0, transition_step, color="0.92", alpha=.55, zorder=-10)
            ax.axvline(transition_step, color="0.35", linestyle=":", linewidth=1)
        ax.grid(alpha=.2)

    for ax, key in zip(flat_axes[3:], keys):
        ax.plot(steps, [row.get(key, np.nan) for row in rows], marker=".", linewidth=1.2)
        title, ylabel = _geometry_metric_panel_labels(key)
        ax.set(
            title=title,
            xlabel="Optimizer step (parameter updates)",
            ylabel=ylabel,
        )
        if transition_step is not None:
            ax.axvspan(0, transition_step, color="0.92", alpha=.55, zorder=-10)
            ax.axvline(transition_step, color="0.35", linestyle=":", linewidth=1)
        ax.grid(alpha=.2)
    for ax in flat_axes[panel_count:]:
        ax.axis("off")
    if run_name:
        fig.text(
            .995, .995, f"run: {run_name}",
            ha="right", va="top", fontsize=7, color="0.35",
        )
    fig.tight_layout(rect=(0, 0, 1, .985))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def assemble_animation(
    frame_paths,
    plots_dir: Path,
    fps: int,
    *,
    stem: str = "geometry_animation",
) -> Path:
    mp4_path = plots_dir / f"{stem}.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg:
        subprocess.run([
            ffmpeg, "-y", "-framerate", str(fps), "-i",
            str(frame_paths[0].parent / "frame%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4_path),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return mp4_path
    gif_path = plots_dir / f"{stem}.gif"
    images = [Image.open(path) for path in frame_paths]
    try:
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=1000 // fps, loop=0)
    finally:
        for image in images:
            image.close()
    return gif_path


def render_run(
    run_dir: Path,
    basis: GeometryRenderBasis = "first_last",
    fps: int = 8,
    *,
    keep_frames: bool = False,
) -> dict[str, Path]:
    snapshots = load_snapshots(run_dir)
    pca = fit_shared_pca(snapshots, basis)
    scales = compute_animation_scales(snapshots, pca)
    plots_dir = run_dir / "plots"
    run_name = run_dir.name
    plots_dir.mkdir(parents=True, exist_ok=True)

    def render_animation(
        snapshot_subset,
        frame_dir_name: str,
        stem: str,
        *,
        latent_phase_label: str | None = None,
    ) -> Path:
        frames_dir = plots_dir / frame_dir_name
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        for frame_index, (step, path) in enumerate(snapshot_subset):
            frame_path = frames_dir / f"frame{frame_index:06d}.png"
            render_frame(
                step,
                path,
                pca,
                frame_path,
                scales,
                run_name=run_name,
                latent_phase_label=latent_phase_label,
            )
            frame_paths.append(frame_path)
        animation = assemble_animation(frame_paths, plots_dir, fps, stem=stem)
        if not keep_frames:
            shutil.rmtree(frames_dir)
        return animation

    latent_snapshots = []
    vq_snapshots = []
    for snapshot in snapshots:
        with np.load(snapshot[1]) as data:
            (vq_snapshots if "codebook" in data else latent_snapshots).append(snapshot)

    outputs = {
        "animation": render_animation(
            snapshots,
            "geometry_frames",
            "geometry_animation",
            latent_phase_label=(
                "AE warmup" if vq_snapshots else "Continuous AE"
            ),
        ),
    }
    metrics_path = plots_dir / "geometry_metrics.png"
    outputs["metrics"] = metrics_path

    if latent_snapshots:
        latent_trajectory = plots_dir / (
            "ae_warmup_latent_trajectory.png"
            if vq_snapshots
            else "continuous_latent_trajectory.png"
        )
        render_latent_centroid_trajectory(
            latent_snapshots,
            pca,
            latent_trajectory,
            run_name=run_name,
            phase_name="AE warmup" if vq_snapshots else "Continuous AE",
        )
        outputs["latent_trajectory"] = latent_trajectory
    if vq_snapshots:
        code_trajectory = plots_dir / "vq_code_trajectories.png"
        render_code_trajectories(
            vq_snapshots,
            pca,
            code_trajectory,
            run_name=run_name,
        )
        outputs["code_trajectories"] = code_trajectory

    if latent_snapshots and vq_snapshots:
        outputs["ae_warmup_animation"] = render_animation(
            latent_snapshots,
            "ae_warmup_geometry_frames",
            "ae_warmup_latent_dynamics",
            latent_phase_label="AE warmup",
        )
        outputs["vq_animation"] = render_animation(
            vq_snapshots,
            "vq_geometry_frames",
            "vq_codebook_dynamics",
        )
        transition_path = plots_dir / "ae_warmup_to_kmeans_transition.png"
        render_warmup_transition(
            snapshots,
            transition_path,
            run_name=run_name,
        )
        outputs["transition"] = transition_path

    render_metric_series(run_dir, metrics_path, run_name=run_name)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--basis", choices=["t0", "first_last", "pooled"], default="first_last")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()
    outputs = render_run(
        args.run_dir, basis=args.basis, fps=args.fps, keep_frames=args.keep_frames
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
