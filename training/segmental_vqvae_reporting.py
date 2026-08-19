"""Bounded artifacts and diagnostics for segmental VQ-VAE runs."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from training.text_vqvae.reporting import atomic_json_dump


def _atomic_torch_save(payload: object, path: Path) -> None:
    """Write a checkpoint atomically so interruption cannot corrupt its target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_metadata(model, *, step: int, epoch: int, phase: str) -> dict:
    return {
        "model_config": asdict(model.config),
        "step": step,
        "epoch": epoch,
        "phase": phase,
    }


def save_model_checkpoint(
    model,
    path: Path,
    *,
    step: int,
    epoch: int,
    phase: str,
) -> None:
    """Save a compact inference checkpoint without optimizer state."""
    _atomic_torch_save(
        {
            "model": model.state_dict(),
            **_checkpoint_metadata(model, step=step, epoch=epoch, phase=phase),
        },
        path,
    )


def save_resume_checkpoint(
    model,
    optimizer,
    path: Path,
    *,
    step: int,
    epoch: int,
    phase: str,
) -> None:
    """Save full training state to one rolling path."""
    _atomic_torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            **_checkpoint_metadata(model, step=step, epoch=epoch, phase=phase),
        },
        path,
    )


def rolling_checkpoint_due(*, step: int, every: int) -> bool:
    return every > 0 and step > 0 and step % every == 0


def finalize_checkpoints(
    model,
    optimizer,
    checkpoint_dir: Path,
    *,
    step: int,
    epoch: int,
    phase: str,
    save_last_resume: bool,
) -> list[str]:
    """Apply metrics-first retention after a successful run.

    The rolling checkpoint is useful only while a run is interruptible. On
    success it is removed; callers may explicitly retain one full final resume
    checkpoint in addition to the compact best model.
    """
    latest = checkpoint_dir / "latest.pt"
    last = checkpoint_dir / "last.pt"
    if save_last_resume:
        save_resume_checkpoint(
            model,
            optimizer,
            last,
            step=step,
            epoch=epoch,
            phase=phase,
        )
    else:
        last.unlink(missing_ok=True)
    latest.unlink(missing_ok=True)
    return sorted(path.name for path in checkpoint_dir.glob("*.pt"))


def _token_label(tokenizer, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    backend = getattr(tokenizer, "tokenizer", None)
    token = backend.id_to_token(token_id) if backend is not None else None
    if token is None:
        return str(token_id)
    return token.replace("Ġ", " ").replace("▁", " ")


def build_segmentation_snapshot(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    outputs: dict,
    *,
    tokenizer=None,
    max_examples: int = 8,
) -> dict:
    """Convert one already-computed intervention forward into bounded CPU data."""
    ids = input_ids.detach().cpu()
    valid = attention_mask.detach().cpu().bool()
    probabilities = outputs["gate_probabilities"].detach().cpu()
    boundaries = outputs["hard_boundaries"].detach().cpu().bool()
    segment_ids = outputs["segment_ids"].detach().cpu()

    all_probabilities: list[float] = []
    all_chunk_lengths: list[int] = []
    hard_candidate_boundaries = 0
    candidate_count = 0
    examples = []
    for row_index in range(ids.shape[0]):
        length = int(valid[row_index].sum())
        candidate_length = max(length - 1, 0)
        row_probabilities = probabilities[row_index, :candidate_length].tolist()
        row_boundaries = boundaries[row_index, :length].tolist()
        row_segments = segment_ids[row_index, :length].tolist()
        all_probabilities.extend(float(value) for value in row_probabilities)
        hard_candidate_boundaries += sum(row_boundaries[:candidate_length])
        candidate_count += candidate_length

        chunk_lengths = []
        if row_segments:
            chunk_lengths = torch.bincount(
                torch.tensor(row_segments, dtype=torch.long)
            ).tolist()
            all_chunk_lengths.extend(int(value) for value in chunk_lengths)

        if len(examples) < max_examples:
            token_ids = ids[row_index, :length].tolist()
            examples.append({
                "token_ids": token_ids,
                "tokens": [_token_label(tokenizer, token_id) for token_id in token_ids],
                "gate_probabilities": probabilities[row_index, :length].tolist(),
                "hard_boundaries": row_boundaries,
                "segment_ids": row_segments,
                "chunk_lengths": chunk_lengths,
            })

    probability_tensor = torch.tensor(all_probabilities, dtype=torch.float)
    length_tensor = torch.tensor(all_chunk_lengths, dtype=torch.float)
    return {
        "examples": examples,
        "gate_probabilities": all_probabilities,
        "chunk_lengths": all_chunk_lengths,
        "gate_probability_mean": (
            float(probability_tensor.mean()) if probability_tensor.numel() else 0.0
        ),
        "gate_probability_std": (
            float(probability_tensor.std(unbiased=False))
            if probability_tensor.numel()
            else 0.0
        ),
        "boundary_fraction": hard_candidate_boundaries / max(candidate_count, 1),
        "chunk_length_p50": (
            float(torch.quantile(length_tensor, 0.5)) if length_tensor.numel() else 0.0
        ),
        "chunk_length_p90": (
            float(torch.quantile(length_tensor, 0.9)) if length_tensor.numel() else 0.0
        ),
    }


def _add_run_label(fig, run_name: str | None) -> None:
    if run_name:
        fig.text(
            0.995,
            0.995,
            f"run: {run_name}",
            ha="right",
            va="top",
            fontsize=7,
            color="0.35",
        )


def _mark_no_data(axis, message: str) -> None:
    if not axis.lines and not axis.collections and not axis.patches:
        axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)


def _read_metrics(metrics_path: Path) -> list[dict]:
    rows = []
    if not metrics_path.is_file():
        return rows
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def plot_segmental_metrics(
    metrics_path: Path,
    plot_dir: Path,
    *,
    compression_target: float,
    run_name: str | None = None,
) -> None:
    """Overwrite compact run-level plots from scalar JSONL history."""
    rows = _read_metrics(metrics_path)
    if not rows:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    train_rows = [row for row in rows if row.get("split") == "train"]
    eval_rows = [row for row in rows if row.get("split") == "eval"]
    intervention_rows = [
        row for row in rows if row.get("split") == "latent_intervention"
    ]
    free_rows = [row for row in rows if row.get("split") == "free_running"]
    transition_rows = [row for row in rows if row.get("split") == "phase_transition"]
    transition_step = transition_rows[0]["step"] if transition_rows else None

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))

    for phase, label in (("ae_warmup", "train AE"), ("vq", "train VQ")):
        phase_rows = [row for row in train_rows if row.get("phase") == phase]
        if phase_rows:
            axes[0, 0].plot(
                [row["step"] for row in phase_rows],
                [row["loss"] for row in phase_rows],
                alpha=0.55,
                label=label,
            )
    if eval_rows:
        axes[0, 0].plot(
            [row["step"] for row in eval_rows],
            [row["loss"] for row in eval_rows],
            marker=".",
            label="eval",
        )
    axes[0, 0].set(title="Total objective", xlabel="step", ylabel="loss")

    if eval_rows:
        component_keys = (
            ("reconstruction_loss", "reconstruction"),
            ("commitment_weighted_loss", "commitment weighted"),
            ("compression_loss", "compression weighted"),
            ("gate_logit_l2_loss", "gate L2 weighted"),
        )
        for key, label in component_keys:
            if key in eval_rows[-1]:
                axes[0, 1].plot(
                    [row["step"] for row in eval_rows],
                    [row[key] for row in eval_rows],
                    marker=".",
                    label=label,
                )
    axes[0, 1].set(title="Eval loss components", xlabel="step", ylabel="loss")
    axes[0, 1].set_yscale("symlog", linthresh=1e-5)

    if eval_rows:
        axes[0, 2].plot(
            [row["step"] for row in eval_rows],
            [row["tokens_per_chunk_hard"] for row in eval_rows],
            marker=".",
            label="hard",
        )
        axes[0, 2].plot(
            [row["step"] for row in eval_rows],
            [row["tokens_per_chunk_soft_batch_mean"] for row in eval_rows],
            marker=".",
            label="soft expectation",
        )
    axes[0, 2].axhline(
        compression_target,
        color="0.3",
        linestyle=":",
        label=f"target {compression_target:g}",
    )
    axes[0, 2].set(title="Compression ratio", xlabel="step", ylabel="BPE tokens / chunk")

    rd_rows = [
        row for row in eval_rows
        if "rate_bits_per_bpe" in row and "distortion_bits_per_bpe" in row
    ]
    if rd_rows:
        axes[0, 3].plot(
            [row["rate_bits_per_bpe"] for row in rd_rows],
            [row["distortion_bits_per_bpe"] for row in rd_rows],
            marker="o",
        )
        last = rd_rows[-1]
        axes[0, 3].annotate(
            f"step {last['step']}",
            (last["rate_bits_per_bpe"], last["distortion_bits_per_bpe"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0, 3].set(
        title="Rate-distortion trajectory",
        xlabel="rate (bits / BPE token)",
        ylabel="teacher-forced distortion (bits / BPE token)",
    )

    codebook_rows = [row for row in eval_rows if "codebook_utilization" in row]
    if codebook_rows:
        axes[1, 0].plot(
            [row["step"] for row in codebook_rows],
            [row["codebook_utilization"] for row in codebook_rows],
            marker=".",
            label="utilization",
        )
        perplexity_axis = axes[1, 0].twinx()
        perplexity_axis.plot(
            [row["step"] for row in codebook_rows],
            [row["codebook_perplexity"] for row in codebook_rows],
            marker=".",
            color="#F58518",
            label="perplexity",
        )
        perplexity_axis.set_ylabel("codebook perplexity")
    axes[1, 0].set(title="Codebook health", xlabel="step", ylabel="used-code fraction")

    if intervention_rows:
        for key, label in (
            ("delta_rand_bits_per_bpe", "random latent"),
            ("delta_perm_bits_per_bpe", "permuted latent"),
            ("delta_null_bits_per_bpe", "no cross-attn"),
            ("rate_bits_per_bpe", "paid rate"),
        ):
            axes[1, 1].plot(
                [row["step"] for row in intervention_rows],
                [row[key] for row in intervention_rows],
                marker=".",
                label=label,
            )
    axes[1, 1].set(
        title="Length-matched latent interventions",
        xlabel="step",
        ylabel="CE increase / rate (bits per BPE)",
    )
    axes[1, 1].set_yscale("symlog", linthresh=1e-3)

    if free_rows:
        axes[1, 2].plot(
            [row["step"] for row in free_rows],
            [row["exposure_gap_bits_per_bpe"] for row in free_rows],
            marker="o",
            label="free - teacher CE",
        )
    axes[1, 2].axhline(0.0, color="0.3", linestyle=":")
    axes[1, 2].set(
        title="Teacher/free-running gap",
        xlabel="step",
        ylabel="exposure gap (bits per BPE)",
    )

    if free_rows:
        axes[1, 3].plot(
            [row["step"] for row in free_rows],
            [row["free_running_token_accuracy"] for row in free_rows],
            marker="o",
            label="token accuracy",
        )
        axes[1, 3].plot(
            [row["step"] for row in free_rows],
            [row["free_running_exact_match"] for row in free_rows],
            marker="o",
            label="exact match",
        )
        axes[1, 3].plot(
            [row["step"] for row in free_rows],
            [row["free_running_normalized_edit_distance"] for row in free_rows],
            marker="o",
            label="normalized edit distance",
        )
    axes[1, 3].set(title="Free-running reconstruction", xlabel="step", ylabel="metric value")

    for axis in axes.flat:
        if transition_step is not None and axis is not axes[0, 3]:
            axis.axvline(transition_step, color="0.5", linestyle=":", linewidth=1)
        axis.grid(alpha=0.2)
        _mark_no_data(axis, "Not available in this phase")
        handles, labels = axis.get_legend_handles_labels()
        if handles and labels:
            axis.legend(fontsize=7)
    _add_run_label(fig, run_name)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(plot_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    warmup_rows = [
        row for row in rows if row.get("split") == "ae_warmup_diagnostic"
    ]
    if warmup_rows:
        steps = [row["step"] for row in warmup_rows]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        axes[0].plot(
            steps,
            [row["water_filling_effective_dim"] for row in warmup_rows],
            marker="o",
            label="water-filling dimension",
        )
        axes[0].plot(
            steps,
            [row["latent_effective_dim"] for row in warmup_rows],
            marker=".",
            label="PCA effective dimension",
        )
        participation_axis = axes[1]
        water_axis = participation_axis.twinx()
        participation_axis.plot(
            steps,
            [row["participation_ratio"] for row in warmup_rows],
            marker=".",
            label="participation ratio",
        )
        water_axis.plot(
            steps,
            [row["water_filling_level"] for row in warmup_rows],
            marker=".",
            color="#F58518",
            label="water-filling level",
        )
        axes[0].set(title="Adaptive warmup dimensionality", xlabel="step", ylabel="dimensions")
        participation_axis.set(
            title="Adaptive warmup spectrum",
            xlabel="step",
            ylabel="participation ratio (dimensions)",
        )
        water_axis.set_ylabel("water-filling level (latent units squared)")
        for axis in axes:
            if transition_step is not None:
                axis.axvline(transition_step, color="0.5", linestyle=":")
            axis.grid(alpha=0.2)
            axis.legend()
        warmup_lines = participation_axis.lines + water_axis.lines
        participation_axis.legend(
            warmup_lines,
            [line.get_label() for line in warmup_lines],
            loc="best",
        )
        _add_run_label(fig, run_name)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(plot_dir / "ae_warmup_diagnostics.png", dpi=150)
        plt.close(fig)


def write_segmentation_visualization(
    snapshot: dict,
    plot_dir: Path,
    *,
    compression_target: float,
    run_name: str | None = None,
) -> None:
    """Overwrite the latest gate distributions and fixed segmentation examples."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(snapshot, plot_dir / "segmentation_latest.json")

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    probabilities = snapshot["gate_probabilities"]
    if probabilities:
        axes[0, 0].hist(probabilities, bins=30, range=(0.0, 1.0))
    axes[0, 0].axvline(0.5, color="0.3", linestyle=":", label="hard threshold")
    axes[0, 0].set(
        title="Gate probability distribution",
        xlabel="boundary probability",
        ylabel="candidate token positions",
    )

    chunk_lengths = snapshot["chunk_lengths"]
    if chunk_lengths:
        max_length = max(chunk_lengths)
        axes[0, 1].hist(
            chunk_lengths,
            bins=range(1, max_length + 2),
            align="left",
            rwidth=0.85,
        )
    axes[0, 1].axvline(
        compression_target,
        color="0.3",
        linestyle=":",
        label=f"target mean {compression_target:g}",
    )
    axes[0, 1].set(title="Hard chunk lengths", xlabel="BPE tokens / chunk", ylabel="chunks")

    examples = snapshot["examples"][:2]
    for example_index, axis in enumerate(axes[1]):
        if example_index >= len(examples):
            axis.text(0.5, 0.5, "No example", ha="center", va="center", transform=axis.transAxes)
            continue
        example = examples[example_index]
        token_count = len(example["tokens"])
        positions = list(range(token_count))
        for position, segment_id in enumerate(example["segment_ids"]):
            axis.axvspan(
                position - 0.5,
                position + 0.5,
                color=("#DCEAF7" if segment_id % 2 == 0 else "#FCE5CD"),
                alpha=0.65,
                zorder=-2,
            )
        axis.plot(positions, example["gate_probabilities"], marker=".", label="gate p")
        boundary_positions = [
            position
            for position, boundary in enumerate(example["hard_boundaries"])
            if boundary
        ]
        axis.scatter(
            boundary_positions,
            [1.02] * len(boundary_positions),
            marker="v",
            color="#C44E52",
            label="hard boundary",
        )
        stride = max(math.ceil(token_count / 24), 1)
        ticks = positions[::stride]
        axis.set_xticks(ticks, [example["tokens"][index][:12] for index in ticks], rotation=60, ha="right", fontsize=7)
        axis.set_ylim(-0.03, 1.1)
        axis.set(
            title=f"Fixed example {example_index + 1}: alternating colors are chunks",
            xlabel="BPE token",
            ylabel="boundary probability",
        )

    for axis in axes.flat:
        axis.grid(alpha=0.15)
        handles, labels = axis.get_legend_handles_labels()
        if handles and labels:
            axis.legend(fontsize=8)
    _add_run_label(fig, run_name)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(plot_dir / "segmentation_latest.png", dpi=150)
    plt.close(fig)
