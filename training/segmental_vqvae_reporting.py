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


def _position_dependence_score(
    rates: list[float | None],
    counts: list[int],
    global_rate: float,
) -> float:
    """Fraction of boundary-decision variance explained by position bins."""
    total = sum(counts)
    variance = global_rate * (1.0 - global_rate)
    if total == 0 or variance <= 1e-12:
        return 0.0
    between_bin_variance = sum(
        count / total * (rate - global_rate) ** 2
        for rate, count in zip(rates, counts, strict=True)
        if rate is not None and count > 0
    )
    return min(max(between_bin_variance / variance, 0.0), 1.0)


def build_segmentation_snapshot(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    outputs: dict,
    *,
    tokenizer=None,
    max_examples: int = 8,
    position_bin_count: int = 10,
    decoder_boundary_threshold: float = 0.5,
) -> dict:
    """Convert one already-computed intervention forward into bounded CPU data."""
    if position_bin_count <= 0:
        raise ValueError("position_bin_count must be positive.")
    ids = input_ids.detach().cpu()
    valid = attention_mask.detach().cpu().bool()
    probabilities = outputs["gate_probabilities"].detach().cpu()
    boundaries = outputs["hard_boundaries"].detach().cpu().bool()
    segment_ids = outputs["segment_ids"].detach().cpu()
    pruning_marker = outputs.get("token_pruning_active")
    token_pruning_active = (
        bool(pruning_marker.detach().cpu().item())
        if isinstance(pruning_marker, torch.Tensor)
        else bool(pruning_marker)
    )
    fixed_count_marker = outputs.get("fixed_count_active")
    fixed_count_active = (
        bool(fixed_count_marker.detach().cpu().item())
        if isinstance(fixed_count_marker, torch.Tensor)
        else bool(fixed_count_marker)
    )
    decoder_boundary_logits = outputs.get("decoder_boundary_logits")
    if isinstance(decoder_boundary_logits, torch.Tensor):
        decoder_probabilities = torch.sigmoid(
            decoder_boundary_logits.detach().cpu()
        )
        decoder_boundaries = (
            decoder_probabilities > decoder_boundary_threshold
        )
    else:
        decoder_probabilities = None
        decoder_boundaries = None

    all_probabilities: list[float] = []
    all_chunk_lengths: list[int] = []
    hard_candidate_boundaries = 0
    candidate_count = 0
    total_valid_tokens = 0
    total_kept_codes = 0
    position_candidate_counts = [0] * position_bin_count
    position_hard_boundary_counts = [0] * position_bin_count
    position_soft_boundary_sums = [0.0] * position_bin_count
    decoder_position_hard_boundary_counts = [0] * position_bin_count
    decoder_position_soft_boundary_sums = [0.0] * position_bin_count
    all_decoder_probabilities: list[float] = []
    examples = []
    for row_index in range(ids.shape[0]):
        length = int(valid[row_index].sum())
        total_valid_tokens += length
        candidate_length = max(length - 1, 0)
        row_probabilities = probabilities[row_index, :candidate_length].tolist()
        row_boundaries = boundaries[row_index, :length].tolist()
        row_segments = segment_ids[row_index, :length].tolist()
        all_probabilities.extend(float(value) for value in row_probabilities)
        if decoder_probabilities is not None:
            all_decoder_probabilities.extend(
                float(value)
                for value in decoder_probabilities[
                    row_index,
                    :candidate_length,
                ].tolist()
            )
        hard_candidate_boundaries += sum(row_boundaries[:candidate_length])
        total_kept_codes += sum(row_boundaries)
        candidate_count += candidate_length
        for position in range(candidate_length):
            normalized_position = (position + 0.5) / candidate_length
            bin_index = min(
                int(normalized_position * position_bin_count),
                position_bin_count - 1,
            )
            position_candidate_counts[bin_index] += 1
            position_hard_boundary_counts[bin_index] += int(
                row_boundaries[position]
            )
            position_soft_boundary_sums[bin_index] += float(
                row_probabilities[position]
            )
            if decoder_probabilities is not None and decoder_boundaries is not None:
                decoder_position_hard_boundary_counts[bin_index] += int(
                    decoder_boundaries[row_index, position]
                )
                decoder_position_soft_boundary_sums[bin_index] += float(
                    decoder_probabilities[row_index, position]
                )

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
                "decoder_boundary_probabilities": (
                    decoder_probabilities[row_index, :length].tolist()
                    if decoder_probabilities is not None
                    else None
                ),
                "hard_boundaries": row_boundaries,
                "segment_ids": row_segments,
                "chunk_lengths": chunk_lengths,
            })

    probability_tensor = torch.tensor(all_probabilities, dtype=torch.float)
    length_tensor = torch.tensor(all_chunk_lengths, dtype=torch.float)
    hard_boundary_rate_by_position = [
        hard_count / count if count else None
        for hard_count, count in zip(
            position_hard_boundary_counts,
            position_candidate_counts,
            strict=True,
        )
    ]
    soft_boundary_rate_by_position = [
        soft_sum / count if count else None
        for soft_sum, count in zip(
            position_soft_boundary_sums,
            position_candidate_counts,
            strict=True,
        )
    ]
    hard_boundary_rate = hard_candidate_boundaries / max(candidate_count, 1)
    soft_boundary_rate = (
        sum(all_probabilities) / candidate_count if candidate_count else 0.0
    )
    singleton_chunk_fraction = (
        sum(length == 1 for length in all_chunk_lengths) / len(all_chunk_lengths)
        if all_chunk_lengths
        else 0.0
    )
    mean_chunk_length = (
        sum(all_chunk_lengths) / len(all_chunk_lengths)
        if all_chunk_lengths
        else 0.0
    )
    memoryless_singleton_baseline = (
        1.0 / mean_chunk_length if mean_chunk_length > 0.0 else 0.0
    )
    early_bins = max(position_bin_count // 4, 1)
    late_start = max(position_bin_count - early_bins, 0)

    def _weighted_rate(
        rates: list[float | None],
        counts: list[int],
        start: int,
        stop: int,
    ) -> float:
        selected = [
            (rate, count)
            for rate, count in zip(
                rates[start:stop],
                counts[start:stop],
                strict=True,
            )
            if rate is not None and count > 0
        ]
        denominator = sum(count for _, count in selected)
        return (
            sum(float(rate) * count for rate, count in selected) / denominator
            if denominator
            else 0.0
        )

    longest_drop_run = max(all_chunk_lengths, default=1) - 1
    if decoder_probabilities is not None:
        decoder_hard_boundary_rate_by_position = [
            hard_count / count if count else None
            for hard_count, count in zip(
                decoder_position_hard_boundary_counts,
                position_candidate_counts,
                strict=True,
            )
        ]
        decoder_soft_boundary_rate_by_position = [
            soft_sum / count if count else None
            for soft_sum, count in zip(
                decoder_position_soft_boundary_sums,
                position_candidate_counts,
                strict=True,
            )
        ]
        decoder_hard_boundary_rate = (
            sum(decoder_position_hard_boundary_counts) / max(candidate_count, 1)
        )
        decoder_soft_boundary_rate = (
            sum(all_decoder_probabilities) / max(candidate_count, 1)
        )
        decoder_bpd_hard = _position_dependence_score(
            decoder_hard_boundary_rate_by_position,
            position_candidate_counts,
            decoder_hard_boundary_rate,
        )
        decoder_bpd_soft = _position_dependence_score(
            decoder_soft_boundary_rate_by_position,
            position_candidate_counts,
            decoder_soft_boundary_rate,
        )
    else:
        decoder_hard_boundary_rate_by_position = []
        decoder_soft_boundary_rate_by_position = []
        decoder_bpd_hard = None
        decoder_bpd_soft = None
    return {
        "selection_kind": "keep" if token_pruning_active else "boundary",
        "fixed_count_active": fixed_count_active,
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
        "boundary_fraction": hard_boundary_rate,
        "keep_fraction": (
            total_kept_codes / total_valid_tokens
            if token_pruning_active and total_valid_tokens
            else None
        ),
        "position_bin_centers": [
            (index + 0.5) / position_bin_count
            for index in range(position_bin_count)
        ],
        "position_bin_candidate_counts": position_candidate_counts,
        "hard_boundary_rate_by_position": hard_boundary_rate_by_position,
        "soft_boundary_rate_by_position": soft_boundary_rate_by_position,
        "boundary_position_dependence_hard": _position_dependence_score(
            hard_boundary_rate_by_position,
            position_candidate_counts,
            hard_boundary_rate,
        ),
        "boundary_position_dependence_soft": _position_dependence_score(
            soft_boundary_rate_by_position,
            position_candidate_counts,
            soft_boundary_rate,
        ),
        "keep_position_dependence_hard": (
            _position_dependence_score(
                hard_boundary_rate_by_position,
                position_candidate_counts,
                hard_boundary_rate,
            )
            if token_pruning_active
            else None
        ),
        "keep_position_dependence_soft": (
            _position_dependence_score(
                soft_boundary_rate_by_position,
                position_candidate_counts,
                soft_boundary_rate,
            )
            if token_pruning_active
            else None
        ),
        "early_keep_rate_hard": (
            _weighted_rate(
                hard_boundary_rate_by_position,
                position_candidate_counts,
                0,
                early_bins,
            )
            if token_pruning_active
            else None
        ),
        "late_keep_rate_hard": (
            _weighted_rate(
                hard_boundary_rate_by_position,
                position_candidate_counts,
                late_start,
                position_bin_count,
            )
            if token_pruning_active
            else None
        ),
        "early_keep_rate_soft": (
            _weighted_rate(
                soft_boundary_rate_by_position,
                position_candidate_counts,
                0,
                early_bins,
            )
            if token_pruning_active
            else None
        ),
        "late_keep_rate_soft": (
            _weighted_rate(
                soft_boundary_rate_by_position,
                position_candidate_counts,
                late_start,
                position_bin_count,
            )
            if token_pruning_active
            else None
        ),
        "decoder_hard_boundary_rate_by_position": (
            decoder_hard_boundary_rate_by_position
        ),
        "decoder_soft_boundary_rate_by_position": (
            decoder_soft_boundary_rate_by_position
        ),
        "decoder_boundary_position_dependence_hard": decoder_bpd_hard,
        "decoder_boundary_position_dependence_soft": decoder_bpd_soft,
        "singleton_chunk_fraction": singleton_chunk_fraction,
        "consecutive_keep_fraction": (
            singleton_chunk_fraction if token_pruning_active else None
        ),
        "longest_drop_run": longest_drop_run if token_pruning_active else None,
        "excess_singleton_fraction": (
            singleton_chunk_fraction - memoryless_singleton_baseline
        ),
        "chunk_length_p50": (
            float(torch.quantile(length_tensor, 0.5)) if length_tensor.numel() else 0.0
        ),
        "chunk_length_p90": (
            float(torch.quantile(length_tensor, 0.9)) if length_tensor.numel() else 0.0
        ),
        "keep_gap_p50": (
            float(torch.quantile(length_tensor, 0.5))
            if token_pruning_active and length_tensor.numel()
            else None
        ),
        "keep_gap_p90": (
            float(torch.quantile(length_tensor, 0.9))
            if token_pruning_active and length_tensor.numel()
            else None
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
    is_pruning_run = any("keep_fraction" in row for row in intervention_rows)
    is_fixed_count_run = any(
        "target_chunks" in row for row in intervention_rows
    )
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
            ("decoder_boundary_weighted_loss", "decoder boundary weighted"),
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
    axes[0, 2].set(
        title="Compression ratio",
        xlabel="step",
        ylabel=(
            "BPE tokens / kept code"
            if is_pruning_run
            else "BPE tokens / chunk"
        ),
    )

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
        pointer_gap_rows = [
            row
            for row in free_rows
            if "predicted_pointer_gap_bits_per_bpe" in row
        ]
        if pointer_gap_rows:
            axes[1, 2].plot(
                [row["step"] for row in pointer_gap_rows],
                [row["predicted_pointer_gap_bits_per_bpe"] for row in pointer_gap_rows],
                marker=".",
                label="predicted pointer - teacher pointer CE",
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

    segmentation_health_rows = [
        row
        for row in intervention_rows
        if "boundary_position_dependence_hard" in row
        and "boundary_position_dependence_soft" in row
    ]
    if segmentation_health_rows:
        steps = [row["step"] for row in segmentation_health_rows]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        hard_key = (
            "keep_position_dependence_hard"
            if is_pruning_run
            else "boundary_position_dependence_hard"
        )
        soft_key = (
            "keep_position_dependence_soft"
            if is_pruning_run
            else "boundary_position_dependence_soft"
        )
        axes[0].plot(
            steps,
            [row[hard_key] for row in segmentation_health_rows],
            marker="o",
            label="hard KPD" if is_pruning_run else "hard BPD",
        )
        axes[0].plot(
            steps,
            [row[soft_key] for row in segmentation_health_rows],
            marker=".",
            label="soft KPD" if is_pruning_run else "soft BPD",
        )
        decoder_bpd_rows = [
            row
            for row in segmentation_health_rows
            if "decoder_boundary_position_dependence_hard" in row
            and "decoder_boundary_position_dependence_soft" in row
        ]
        if decoder_bpd_rows:
            decoder_steps = [row["step"] for row in decoder_bpd_rows]
            axes[0].plot(
                decoder_steps,
                [
                    row["decoder_boundary_position_dependence_hard"]
                    for row in decoder_bpd_rows
                ],
                marker="o",
                linestyle="--",
                label="decoder hard BPD",
            )
            axes[0].plot(
                decoder_steps,
                [
                    row["decoder_boundary_position_dependence_soft"]
                    for row in decoder_bpd_rows
                ],
                marker=".",
                linestyle="--",
                label="decoder soft BPD",
            )
        axes[0].set(
            title=(
                "Keep-position dependence"
                if is_pruning_run
                else "Boundary-position dependence"
            ),
            xlabel="step",
            ylabel=(
                "position-explained keep variance"
                if is_pruning_run
                else "position-explained boundary variance"
            ),
            ylim=(-0.02, 1.02),
        )

        if is_pruning_run:
            pruning_rows = [
                row
                for row in segmentation_health_rows
                if "keep_gap_p50" in row
                and "keep_gap_p90" in row
                and "longest_drop_run" in row
            ]
            pruning_steps = [row["step"] for row in pruning_rows]
            axes[1].plot(
                pruning_steps,
                [row["keep_gap_p50"] for row in pruning_rows],
                marker="o",
                label="keep gap p50",
            )
            axes[1].plot(
                pruning_steps,
                [row["keep_gap_p90"] for row in pruning_rows],
                marker=".",
                label="keep gap p90",
            )
            axes[1].plot(
                pruning_steps,
                [row["longest_drop_run"] for row in pruning_rows],
                marker="x",
                label="longest drop run",
            )
            axes[1].set(
                title="Keep-gap concentration",
                xlabel="step",
                ylabel="BPE tokens",
            )
        else:
            singleton_rows = [
                row
                for row in segmentation_health_rows
                if "singleton_chunk_fraction" in row
                and "excess_singleton_fraction" in row
            ]
            if singleton_rows:
                singleton_steps = [row["step"] for row in singleton_rows]
                axes[1].plot(
                    singleton_steps,
                    [row["singleton_chunk_fraction"] for row in singleton_rows],
                    marker="o",
                    label="singleton chunks",
                )
                axes[1].plot(
                    singleton_steps,
                    [row["excess_singleton_fraction"] for row in singleton_rows],
                    marker=".",
                    label="excess vs geometric",
                )
            if is_fixed_count_run:
                churn_rows = [
                    row
                    for row in segmentation_health_rows
                    if row.get("viterbi_path_churn_available")
                ]
                axes[1].plot(
                    [row["step"] for row in churn_rows],
                    [row["viterbi_path_change_fraction"] for row in churn_rows],
                    marker="x",
                    label="Viterbi paths changed",
                )
                axes[1].plot(
                    [row["step"] for row in churn_rows],
                    [row["viterbi_boundary_churn"] for row in churn_rows],
                    marker="+",
                    label="boundary churn",
                )
            axes[1].axhline(0.0, color="0.3", linestyle=":")
            axes[1].set(
                title=(
                    "Fixed-count path stability"
                    if is_fixed_count_run
                    else "Singleton-chunk concentration"
                ),
                xlabel="step",
                ylabel="fraction",
            )
        for axis in axes:
            if transition_step is not None:
                axis.axvline(transition_step, color="0.5", linestyle=":")
            axis.grid(alpha=0.2)
            axis.legend()
        _add_run_label(fig, run_name)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(plot_dir / "segmentation_health.png", dpi=150)
        plt.close(fig)

    pointer_rows = [
        row for row in free_rows if "predicted_pointer_token_alignment" in row
    ]
    if pointer_rows:
        steps = [row["step"] for row in pointer_rows]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        axes[0, 0].plot(
            steps,
            [row["predicted_pointer_token_alignment"] for row in pointer_rows],
            marker="o",
            label="aligned token fraction",
        )
        axes[0, 0].plot(
            steps,
            [row["mean_first_pointer_drift_fraction"] for row in pointer_rows],
            marker=".",
            label="first drift position",
        )
        axes[0, 0].set(
            title="Predicted-pointer alignment",
            xlabel="step",
            ylabel="fraction",
            ylim=(-0.02, 1.02),
        )

        axes[0, 1].plot(
            steps,
            [row["predicted_pointer_mae"] for row in pointer_rows],
            marker="o",
            label="pointer MAE",
        )
        axes[0, 1].set(
            title="Pointer displacement",
            xlabel="step",
            ylabel="latent ordinals",
        )

        for key, label in (
            ("target_end_code_consumption", "codes consumed at target end"),
            ("free_running_code_consumption_at_eos", "codes consumed at free EOS"),
            ("premature_code_exhaustion_fraction", "premature exhaustion"),
        ):
            axes[1, 0].plot(
                steps,
                [row[key] for row in pointer_rows],
                marker=".",
                label=label,
            )
        axes[1, 0].set(
            title="Code-consumption health",
            xlabel="step",
            ylabel="fraction",
            ylim=(-0.02, 1.02),
        )

        boundary_f1_key = "predicted_pointer_boundary_f1"
        for key, label in (
            (boundary_f1_key, "boundary F1"),
            (
                "predicted_pointer_boundary_precision",
                "boundary precision",
            ),
            ("predicted_pointer_boundary_recall", "boundary recall"),
        ):
            axes[1, 1].plot(
                steps,
                [row[key] for row in pointer_rows],
                marker=".",
                label=label,
            )
        axes[1, 1].set(
            title="Boundary prediction under pointer rollout",
            xlabel="step",
            ylabel="fraction",
            ylim=(-0.02, 1.02),
        )

        for axis in axes.flat:
            if transition_step is not None:
                axis.axvline(transition_step, color="0.5", linestyle=":")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        _add_run_label(fig, run_name)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(plot_dir / "pointer_health.png", dpi=150)
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

    fig, axes = plt.subplot_mosaic(
        [
            ["gate", "length", "position"],
            ["example_1", "example_1", "example_2"],
        ],
        figsize=(18, 8),
    )
    is_pruning = snapshot.get("selection_kind") == "keep"
    decision_name = "keep" if is_pruning else "boundary"
    interval_name = "kept-code gap" if is_pruning else "chunk"
    probabilities = snapshot["gate_probabilities"]
    if probabilities:
        axes["gate"].hist(probabilities, bins=30, range=(0.0, 1.0))
    axes["gate"].axvline(0.5, color="0.3", linestyle=":", label="hard threshold")
    axes["gate"].set(
        title="Gate probability distribution",
        xlabel=f"{decision_name} probability",
        ylabel="candidate token positions",
    )

    chunk_lengths = snapshot["chunk_lengths"]
    if chunk_lengths:
        max_length = max(chunk_lengths)
        axes["length"].hist(
            chunk_lengths,
            bins=range(1, max_length + 2),
            align="left",
            rwidth=0.85,
        )
    axes["length"].axvline(
        compression_target,
        color="0.3",
        linestyle=":",
        label=f"target mean {compression_target:g}",
    )
    axes["length"].set(
        title=f"Hard {interval_name} lengths",
        xlabel=f"BPE tokens / {interval_name}",
        ylabel=f"{interval_name}s",
    )

    position_centers = snapshot.get("position_bin_centers", [])
    for key, label, marker, linestyle in (
        ("soft_boundary_rate_by_position", "encoder soft", ".", "-"),
        ("hard_boundary_rate_by_position", "encoder hard", "o", "-"),
        (
            "decoder_soft_boundary_rate_by_position",
            "decoder soft",
            ".",
            "--",
        ),
        (
            "decoder_hard_boundary_rate_by_position",
            "decoder hard",
            "o",
            "--",
        ),
    ):
        rates = snapshot.get(key, [])
        if not rates:
            continue
        plotted = [
            (position, rate)
            for position, rate in zip(position_centers, rates, strict=True)
            if rate is not None
        ]
        if plotted:
            axes["position"].plot(
                [position for position, _ in plotted],
                [rate for _, rate in plotted],
                marker=marker,
                linestyle=linestyle,
                label=label,
            )
    axes["position"].axhline(
        snapshot.get("boundary_fraction", 0.0),
        color="0.3",
        linestyle=":",
        label="global hard rate",
    )
    hard_bpd = snapshot.get("boundary_position_dependence_hard")
    soft_bpd = snapshot.get("boundary_position_dependence_soft")
    bpd_title = f"{decision_name.capitalize()} rate by normalized position"
    if hard_bpd is not None and soft_bpd is not None:
        bpd_title += f"\nencoder BPD H/S={hard_bpd:.3f}/{soft_bpd:.3f}"
    decoder_hard_bpd = snapshot.get(
        "decoder_boundary_position_dependence_hard"
    )
    decoder_soft_bpd = snapshot.get(
        "decoder_boundary_position_dependence_soft"
    )
    if decoder_hard_bpd is not None and decoder_soft_bpd is not None:
        bpd_title += (
            f"\ndecoder BPD H/S={decoder_hard_bpd:.3f}/"
            f"{decoder_soft_bpd:.3f}"
        )
    axes["position"].set(
        title=bpd_title,
        xlabel="normalized token position",
        ylabel=f"{decision_name} rate",
        xlim=(0.0, 1.0),
        ylim=(-0.03, 1.03),
    )

    examples = snapshot["examples"][:2]
    example_axes = [axes["example_1"], axes["example_2"]]
    for example_index, axis in enumerate(example_axes):
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
        axis.plot(
            positions,
            example["gate_probabilities"],
            marker=".",
            label=f"{decision_name} p",
        )
        decoder_example_probabilities = example.get(
            "decoder_boundary_probabilities"
        )
        if decoder_example_probabilities is not None:
            axis.plot(
                positions,
                decoder_example_probabilities,
                linestyle="--",
                label="decoder boundary p",
            )
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
            label=f"hard {decision_name}",
        )
        stride = max(math.ceil(token_count / 24), 1)
        ticks = positions[::stride]
        axis.set_xticks(ticks, [example["tokens"][index][:12] for index in ticks], rotation=60, ha="right", fontsize=7)
        axis.set_ylim(-0.03, 1.1)
        axis.set(
            title=(
                f"Fixed example {example_index + 1}: alternating colors are "
                f"{interval_name}s"
            ),
            xlabel="BPE token",
            ylabel=f"{decision_name} probability",
        )

    for axis in axes.values():
        axis.grid(alpha=0.15)
        handles, labels = axis.get_legend_handles_labels()
        if handles and labels:
            axis.legend(fontsize=8)
    _add_run_label(fig, run_name)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(plot_dir / "segmentation_latest.png", dpi=150)
    plt.close(fig)
