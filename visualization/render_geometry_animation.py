"""Render training geometry snapshots using one PCA basis shared by every frame."""

from __future__ import annotations

import argparse
import json
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


def load_snapshots(run_dir: Path):
    paths = sorted((run_dir / "geometry").glob("step*.npz"))
    if not paths:
        raise FileNotFoundError(f"No geometry snapshots found under {run_dir / 'geometry'}")
    return [(int(path.stem.removeprefix("step")), path) for path in paths]


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
            encoders.append(data["z_e"].astype(np.float32))
            codebooks.append(data["codebook"].astype(np.float32))
    encoder = np.concatenate(encoders)
    codebook = np.concatenate(codebooks)
    count = min(len(encoder), len(codebook), max_fit_points)
    rng = np.random.default_rng(random_state)
    encoder_fit = encoder[rng.choice(len(encoder), count, replace=False)]
    codebook_fit = codebook[rng.choice(len(codebook), count, replace=False)]
    return PCA(n_components=2).fit(np.concatenate([encoder_fit, codebook_fit]))


def render_frame(
    step: int,
    path: Path,
    pca: PCA,
    output_path: Path,
    scales: AnimationScales,
) -> None:
    with np.load(path) as data:
        encoder = data["z_e"].astype(np.float32)
        codebook = data["codebook"].astype(np.float32)
        assignments = data["assignments"].astype(np.int64)
    wins = np.bincount(assignments, minlength=len(codebook))
    alive = wins > 0
    encoder_2d = pca.transform(encoder)
    codebook_2d = pca.transform(codebook)
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
        rasterized=True, label="dead code", zorder=3,
    )
    ax.scatter(
        codebook_2d[alive, 0], codebook_2d[alive, 1],
        color="#E45756", marker="*", s=72, alpha=.98, linewidths=.55,
        edgecolors="#5B1717", rasterized=True, label="winning code", zorder=5,
    )
    ax.set(
        xlim=scales.pca_xlim,
        ylim=scales.pca_ylim,
        title="Shared-basis PCA: encoder outputs and codebook state",
        xlabel="PC1",
        ylabel="PC2",
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
    axes[0, 1].set(xlim=(scales.norm_bins[0], scales.norm_bins[-1]), ylim=scales.norm_ylim)
    axes[0, 1].set_title("Vector norms")
    axes[0, 1].legend(loc="upper right")
    axes[1, 0].hist(nearest, bins=scales.nearest_bins, color="#4C78A8", alpha=.8)
    axes[1, 0].set(
        xlim=(scales.nearest_bins[0], scales.nearest_bins[-1]),
        ylim=scales.nearest_ylim,
        title="Distance to assigned nearest code",
        xlabel="L2 distance",
        ylabel="points",
    )
    ranked = np.sort(wins)[::-1]
    axes[1, 1].plot(np.arange(1, len(ranked) + 1), ranked, color="#E45756")
    axes[1, 1].set(
        xlim=scales.rank_xlim,
        ylim=scales.rank_ylim,
        title="Win-count rank curve",
        xlabel="code rank",
        ylabel="probe wins",
        yscale="symlog",
        xscale="log",
    )
    for axis in axes.flat:
        axis.grid(alpha=.2)
        axis.set_title(axis.get_title(), pad=9)
    fig.suptitle(
        f"Geometry step {step:,} — used codes {alive.sum():,}/{len(codebook):,}",
        fontsize=15,
        x=.5,
        y=.975,
    )
    fig.subplots_adjust(left=.075, right=.975, bottom=.075, top=.91, wspace=.22, hspace=.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def projection_limits(snapshots, pca: PCA):
    points = []
    for _, path in snapshots:
        with np.load(path) as data:
            points.append(pca.transform(data["z_e"].astype(np.float32)))
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
            encoder = data["z_e"].astype(np.float32)
            codebook = data["codebook"].astype(np.float32)
            assignments = data["assignments"].astype(np.int64)
        encoder_norm_frames.append(np.linalg.norm(encoder, axis=1))
        codebook_norm_frames.append(np.linalg.norm(codebook, axis=1))
        nearest_distance_frames.append(np.linalg.norm(encoder - codebook[assignments], axis=1))
        wins = np.bincount(assignments, minlength=len(codebook))
        maximum_win_count = max(maximum_win_count, int(wins.max(initial=0)))
        codebook_size = max(codebook_size, len(codebook))

    encoder_norms = np.concatenate(encoder_norm_frames)
    codebook_norms = np.concatenate(codebook_norm_frames)
    nearest_distances = np.concatenate(nearest_distance_frames)
    norm_bins = _fixed_bin_edges(np.concatenate([encoder_norms, codebook_norms]), 50)
    nearest_bins = _fixed_bin_edges(nearest_distances, 60)
    norm_density_max = max(
        float(np.histogram(values, bins=norm_bins, density=True)[0].max(initial=0))
        for values in encoder_norm_frames + codebook_norm_frames
    )
    nearest_count_max = max(
        int(np.histogram(values, bins=nearest_bins)[0].max(initial=0))
        for values in nearest_distance_frames
    )
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


def render_code_trajectories(snapshots, pca: PCA, output_path: Path, random_state: int = 0) -> None:
    with np.load(snapshots[-1][1]) as final:
        wins = np.bincount(final["assignments"].astype(np.int64), minlength=len(final["codebook"]))
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
    ax.set(title="Code trajectories in the shared PCA basis\n(top-16 final winners + up to 16 final dead codes)", xlabel="PC1", ylabel="PC2")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_metric_series(run_dir: Path, output_path: Path) -> None:
    rows = []
    with (run_dir / "metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == "geometry":
                rows.append(row)
    if not rows:
        raise ValueError("metrics.jsonl contains no geometry rows")
    keys = [key for key in rows[0] if key not in {"split", "step", "elapsed_sec"}]
    columns = 3
    rows_count = (len(keys) + columns - 1) // columns
    fig, axes = plt.subplots(rows_count, columns, figsize=(15, 3.5 * rows_count), squeeze=False)
    steps = [row["step"] for row in rows]
    for ax, key in zip(axes.flat, keys):
        ax.plot(steps, [row.get(key, np.nan) for row in rows], marker=".", linewidth=1.2)
        ax.set(title=key, xlabel="step")
        ax.grid(alpha=.2)
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def assemble_animation(frame_paths, plots_dir: Path, fps: int) -> Path:
    mp4_path = plots_dir / "geometry_animation.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg:
        subprocess.run([
            ffmpeg, "-y", "-framerate", str(fps), "-i", str(plots_dir / "geometry_frames" / "frame%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4_path),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return mp4_path
    gif_path = plots_dir / "geometry_animation.gif"
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
    frames_dir = plots_dir / "geometry_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for frame_index, (step, path) in enumerate(snapshots):
        frame_path = frames_dir / f"frame{frame_index:06d}.png"
        render_frame(step, path, pca, frame_path, scales)
        frame_paths.append(frame_path)
    trajectory_path = plots_dir / "geometry_code_trajectories.png"
    metrics_path = plots_dir / "geometry_metrics.png"
    render_code_trajectories(snapshots, pca, trajectory_path)
    render_metric_series(run_dir, metrics_path)
    animation_path = assemble_animation(frame_paths, plots_dir, fps)
    if not keep_frames:
        shutil.rmtree(frames_dir)
    return {"animation": animation_path, "trajectories": trajectory_path, "metrics": metrics_path}


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
