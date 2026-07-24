"""Raw, training-time geometry snapshots for offline diagnostics."""

from __future__ import annotations

import json
import random
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.text_vqvae_config import GeometryRenderBasis

@contextmanager
def preserve_rng_state():
    """Prevent diagnostics and DataLoader iterator setup from advancing training RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def materialize_geometry_probe(val_loader, *, latent_slots: int, max_points: int, run_dir: Path):
    """Copy a deterministic prefix of validation examples to CPU and write its metadata."""
    sample_target = max(1, (max_points + latent_slots - 1) // latent_slots)
    batches: list[dict[str, torch.Tensor]] = []
    sample_indices: list[int] = []
    dataset_indices = getattr(getattr(val_loader, "dataset", None), "indices", None)
    next_index = 0
    with preserve_rng_state():
        for batch in val_loader:
            take = min(len(batch["input_ids"]), sample_target - len(sample_indices))
            if take <= 0:
                break
            copied = {
                key: value[:take].detach().cpu().clone()
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            batches.append(copied)
            if dataset_indices is None:
                sample_indices.extend(range(next_index, next_index + take))
            else:
                sample_indices.extend(int(index) for index in dataset_indices[next_index : next_index + take])
            next_index += take
            if len(sample_indices) >= sample_target:
                break
    if not batches:
        raise ValueError("Cannot create a geometry probe from an empty validation loader.")

    pad_values = []
    for batch in batches:
        mask = batch.get("attention_mask")
        if mask is None:
            raise ValueError("Geometry probes require attention_mask to compute PAD ratios.")
        pad_values.append(
            F.adaptive_avg_pool1d((mask == 0).float().unsqueeze(1), latent_slots).squeeze(1)
        )
    pad_ratios = torch.cat(pad_values).reshape(-1)
    geometry_dir = Path(run_dir) / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "sample_indices": sample_indices,
        "samples": len(sample_indices),
        "latent_slots": latent_slots,
        "points": len(sample_indices) * latent_slots,
        "requested_points": max_points,
        "pad_ratio": float(pad_ratios.mean()),
        "pad_ratios": pad_ratios.tolist(),
    }
    with (geometry_dir / "probe_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return batches


def geometry_snapshot_due(step: int, *, dense_every: int, dense_until: int, sparse_every: int) -> bool:
    if step in (0, 1):
        return True
    if step <= dense_until:
        return step % dense_every == 0
    return step % sparse_every == 0


def finalize_geometry_artifacts(
    run_dir: Path,
    *,
    enabled: bool,
    basis: GeometryRenderBasis,
    fps: int,
    keep_snapshots: bool,
) -> dict[str, object]:
    """Render compact final artifacts and optionally remove raw snapshots.

    Snapshot deletion happens only after every renderer output exists, so a
    failed post-processing attempt remains recoverable.
    """
    if not enabled:
        return {"status": "disabled", "snapshots_retained": True}

    from visualization.render_geometry_animation import render_run

    run_dir = Path(run_dir)
    outputs = render_run(run_dir, basis=basis, fps=fps)
    missing = [str(path) for path in outputs.values() if not Path(path).is_file()]
    if missing:
        raise RuntimeError(f"Geometry renderer did not create expected outputs: {missing}")

    if not keep_snapshots:
        shutil.rmtree(run_dir / "geometry")

    return {
        "status": "completed",
        "basis": basis,
        "fps": fps,
        "snapshots_retained": keep_snapshots,
        "artifacts": {
            name: str(Path(path).relative_to(run_dir))
            for name, path in outputs.items()
        },
    }


@torch.no_grad()
def dump_geometry_snapshot(model, probe_batches, step: int, run_dir: Path) -> dict[str, float | int]:
    """Dump unprojected vectors/assignments and return full-dimensional metrics."""
    device = next(model.parameters()).device
    was_training = model.training
    z_chunks = []
    pad_chunks = []
    try:
        with preserve_rng_state():
            model.eval()
            for batch in probe_batches:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                z_e = model.encode(input_ids, attention_mask=attention_mask)
                z_chunks.append(z_e.reshape(-1, z_e.shape[-1]).float().cpu())
                if attention_mask is None:
                    pad_mask = input_ids.eq(model.config.pad_token_id).float()
                else:
                    pad_mask = (attention_mask == 0).float()
                pad_chunks.append(
                    F.adaptive_avg_pool1d(pad_mask.unsqueeze(1), z_e.shape[1]).squeeze(1).reshape(-1).cpu()
                )
            encoder = torch.cat(z_chunks)
            pad_ratios = torch.cat(pad_chunks)
            codebook = model.quantizer.codebook.weight.detach().float().cpu()
    finally:
        model.train(was_training)

    nearest_distances = []
    assignments = []
    for chunk in encoder.split(512):
        distances = torch.cdist(chunk, codebook)
        nearest, indices = distances.min(dim=1)
        nearest_distances.append(nearest)
        assignments.append(indices)
    nearest = torch.cat(nearest_distances)
    assigned = torch.cat(assignments)
    wins = torch.bincount(assigned, minlength=len(codebook))
    slot_indices = torch.arange(model.config.latent_slots, dtype=torch.int16).repeat(len(encoder) // model.config.latent_slots)

    geometry_dir = Path(run_dir) / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        geometry_dir / f"step{step:06d}.npz",
        z_e=encoder.numpy().astype(np.float16),
        codebook=codebook.numpy().astype(np.float16),
        assignments=assigned.numpy().astype(np.int32),
        pad_ratios=pad_ratios.numpy().astype(np.float16),
        slot_indices=slot_indices.numpy(),
    )

    norms = encoder.norm(dim=1)
    covariance = torch.cov(encoder.T) if len(encoder) > 1 else torch.zeros((encoder.shape[1], encoder.shape[1]))
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    eig_sum = eigenvalues.sum()
    participation_ratio = float(eig_sum.square() / eigenvalues.square().sum().clamp_min(1e-12))
    return {
        "encoder_mean_norm": float(norms.mean()),
        "encoder_norm_std": float(norms.std(unbiased=False)),
        "encoder_pairwise_mean_distance": _mean_pairwise_distance(encoder),
        "participation_ratio": participation_ratio,
        "nearest_code_distance_p10": float(torch.quantile(nearest, 0.1)),
        "nearest_code_distance_p50": float(torch.quantile(nearest, 0.5)),
        "nearest_code_distance_p90": float(torch.quantile(nearest, 0.9)),
        "used_codes": int((wins > 0).sum()),
        "win_count_gini": _gini(wins.float()),
        "centroid_distance": float(torch.linalg.vector_norm(encoder.mean(0) - codebook.mean(0))),
    }


def _mean_pairwise_distance(vectors: torch.Tensor) -> float:
    if len(vectors) < 2:
        return 0.0
    total = 0.0
    count = 0
    for start in range(0, len(vectors), 256):
        distances = torch.cdist(vectors[start : start + 256], vectors)
        rows = len(distances)
        total += float(distances.sum())
        count += rows * len(vectors)
    return total / (count - len(vectors))


def _gini(values: torch.Tensor) -> float:
    total = values.sum()
    if total == 0:
        return 0.0
    sorted_values = values.sort().values
    n = len(sorted_values)
    indices = torch.arange(1, n + 1, dtype=sorted_values.dtype)
    return float((2 * (indices * sorted_values).sum() / (n * total)) - (n + 1) / n)
