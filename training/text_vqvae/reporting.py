"""File I/O, plots, PCA diagnostics, and sample writing for text VQ-VAE."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from common.text_vqvae_config import PCAFitMode
from visualization.text_vqvae import (
    collect_encoder_vectors,
    compare_vector_distributions_pca,
    render_pca_comparison,
    save_pca_metadata,
)


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------

def atomic_json_dump(data, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    tmp_path.replace(path)


def append_jsonl(row, path: Path) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Sample writing
# ---------------------------------------------------------------------------

def build_reconstruction_rows(
    input_ids: torch.Tensor,
    pred_ids,
    lengths: torch.Tensor,
    tokenizer,
    *,
    max_items: int,
) -> list[dict[str, str]]:
    """Decode already-computed predictions without another model forward."""
    rows = []
    for original, reconstructed, length in zip(
        input_ids.detach().cpu(), pred_ids, lengths.detach().cpu()
    ):
        defined_length = int(length.item())
        rows.append({
            "original": tokenizer.decode(original[:defined_length].tolist()),
            "reconstruction": tokenizer.decode(
                reconstructed[:defined_length].tolist()
            ),
        })
        if len(rows) >= max_items:
            break
    return rows


def write_reconstruction_rows(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def write_reconstruction_samples(model, data_loader, device, model_config, tokenizer, path: Path, max_items: int = 16) -> None:
    was_training = model.training
    rows = []
    try:
        model.eval()
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            outputs = model.infer(input_ids, attention_mask)
            pred_ids = [logits.argmax(dim=-1).cpu() for logits in outputs["logits"]]
            rows.extend(build_reconstruction_rows(
                input_ids,
                pred_ids,
                outputs["lengths"],
                tokenizer,
                max_items=max_items - len(rows),
            ))
            if len(rows) >= max_items:
                break
    finally:
        model.train(was_training)
    write_reconstruction_rows(rows, path)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

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


def plot_training_curves(
    metrics_path: Path,
    plot_dir: Path,
    *,
    run_name: str | None = None,
) -> None:
    rows = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    if not rows:
        return

    train_rows = [r for r in rows if r["split"] == "train"]
    eval_rows = [r for r in rows if r["split"] == "eval"]
    train_window_rows = [r for r in rows if r["split"] == "train_window"]
    probe_rows = [r for r in rows if r["split"] == "codebook_probe"]
    geometry_rows = [r for r in rows if r["split"] == "geometry"]
    transition_rows = [r for r in rows if r["split"] == "phase_transition"]
    transition_step = transition_rows[0]["step"] if transition_rows else None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    if train_rows:
        for phase, label in (("ae_warmup", "train total (AE)"), ("vq", "train total (VQ)")):
            phase_rows = [r for r in train_rows if r.get("phase") == phase]
            if phase_rows:
                axes[0, 0].plot(
                    [r["step"] for r in phase_rows],
                    [r["loss"] for r in phase_rows],
                    label=label,
                )
        legacy_rows = [r for r in train_rows if "phase" not in r]
        if legacy_rows:
            axes[0, 0].plot(
                [r["step"] for r in legacy_rows],
                [r["loss"] for r in legacy_rows],
                label="train",
            )
    if eval_rows:
        axes[0, 0].plot([r["step"] for r in eval_rows], [r["loss"] for r in eval_rows], label="eval")
    axes[0, 0].set_title("Training objective by phase")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    if eval_rows:
        axes[0, 1].plot([r["step"] for r in eval_rows], [r["token_ppl"] for r in eval_rows])
    axes[0, 1].set_title("Eval token perplexity")
    axes[0, 1].grid(True, alpha=0.3)

    if train_rows:
        axes[1, 0].plot([r["step"] for r in train_rows], [r["token_accuracy"] for r in train_rows], label="train")
    if eval_rows:
        axes[1, 0].plot([r["step"] for r in eval_rows], [r["token_accuracy"] for r in eval_rows], label="eval")
    axes[1, 0].set_title("Token accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if probe_rows:
        axes[1, 1].plot(
            [r["step"] for r in probe_rows],
            [r["train_utilization"] for r in probe_rows],
            label="train probe, current codebook (eval mode)",
        )
        axes[1, 1].plot(
            [r["step"] for r in probe_rows],
            [r["eval_utilization"] for r in probe_rows],
            label="eval probe, current codebook (matched N)",
        )
    elif any("codebook_utilization" in r for r in train_rows + eval_rows):
        # Compatibility fallback for metrics written before matched probes.
        vq_train_rows = [r for r in train_rows if "codebook_utilization" in r]
        if vq_train_rows:
            axes[1, 1].plot(
                [r["step"] for r in vq_train_rows],
                [r["codebook_utilization"] for r in vq_train_rows],
                label="train batch, current codebook",
            )
        vq_eval_rows = [r for r in eval_rows if "codebook_utilization" in r]
        if vq_eval_rows:
            axes[1, 1].plot(
                [r["step"] for r in vq_eval_rows],
                [r["codebook_utilization"] for r in vq_eval_rows],
                label="eval full-set, current codebook",
            )

    eval_batch_rows = [
        r for r in eval_rows if "codebook_utilization_batch_mean" in r
    ]
    frozen_c0_rows = [
        r for r in eval_rows if "codebook_utilization_frozen_c0" in r
    ]
    if train_window_rows:
        axes[1, 1].plot(
            [r["step"] for r in train_window_rows],
            [r["codebook_utilization_batch_mean"] for r in train_window_rows],
            linestyle="--",
            alpha=0.7,
            label="train batch mean, current codebook",
        )
    if eval_batch_rows:
        axes[1, 1].plot(
            [r["step"] for r in eval_batch_rows],
            [r["codebook_utilization_batch_mean"] for r in eval_batch_rows],
            linestyle="--",
            alpha=0.7,
            label="eval batch mean, current codebook",
        )
    has_codebook_curves = bool(
        probe_rows or train_window_rows or eval_batch_rows or frozen_c0_rows
        or any("codebook_utilization" in r for r in train_rows + eval_rows)
    )
    if frozen_c0_rows:
        axes[1, 1].plot(
            [r["step"] for r in frozen_c0_rows],
            [r["codebook_utilization_frozen_c0"] for r in frozen_c0_rows],
            linestyle=":",
            linewidth=2,
            label="eval full-set, frozen K-means C0",
        )
        utilization_title = (
            "Codebook utilization: current codebook vs frozen K-means C0"
        )
    elif has_codebook_curves:
        utilization_title = "Codebook utilization: current codebook"
    else:
        latent_rows = [
            r for r in geometry_rows if "encoder_mean_norm" in r
        ]
        if latent_rows:
            axes[1, 1].plot(
                [r["step"] for r in latent_rows],
                [r["encoder_mean_norm"] for r in latent_rows],
                label="encoder latent mean norm",
            )
            axes[1, 1].plot(
                [r["step"] for r in latent_rows],
                [r["encoder_norm_std"] for r in latent_rows],
                label="encoder latent norm std",
            )
            utilization_title = "Continuous bottleneck: latent norm dynamics"
        else:
            utilization_title = "Codebook metrics not applicable (continuous bottleneck)"
    if transition_step is not None and has_codebook_curves:
        utilization_title += f" (VQ phase after step {transition_step})"
    axes[1, 1].set_title(utilization_title)
    if axes[1, 1].lines:
        axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    if transition_step is not None:
        for ax in axes.flat:
            ax.axvspan(0, transition_step, color="0.92", alpha=0.55, zorder=-10)
            ax.axvline(
                transition_step,
                color="0.35",
                linestyle=":",
                linewidth=1,
                label="K-means init" if ax is axes[0, 0] else None,
            )
        axes[0, 0].legend()

    _add_run_label(fig, run_name)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(plot_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    warmup_rows = [
        row for row in rows if row["split"] == "ae_warmup_diagnostic"
    ]
    if warmup_rows:
        steps = [row["step"] for row in warmup_rows]
        transition_step = next(
            (
                row["step"]
                for row in rows
                if row["split"] == "phase_transition"
            ),
            None,
        )
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        axes[0].plot(
            steps,
            [row["water_filling_effective_dim"] for row in warmup_rows],
            marker="o",
            label="target-rate water-filling dimension",
        )
        axes[0].plot(
            steps,
            [row["latent_effective_dim"] for row in warmup_rows],
            marker=".",
            label="PCA effective dimension",
        )
        threshold = warmup_rows[0]["variance_threshold"]
        axes[0].set(
            title=f"AE warmup dimensionality (PCA threshold {threshold:.0%})",
            xlabel="optimizer step",
            ylabel="dimensions",
        )
        axes[0].legend()
        axes[1].plot(
            steps,
            [row["water_filling_level"] for row in warmup_rows],
            marker=".",
            label="water level",
        )
        axes[1].plot(
            steps,
            [row["participation_ratio"] for row in warmup_rows],
            marker=".",
            label="participation ratio",
        )
        axes[1].set(
            title="Target-rate water level and latent participation ratio",
            xlabel="optimizer step",
        )
        axes[1].legend()
        for axis in axes:
            if transition_step is not None:
                axis.axvline(
                    transition_step,
                    color="0.35",
                    linestyle=":",
                    linewidth=1,
                    label="K-means transition",
                )
            axis.grid(alpha=0.2)
            axis.legend()
        _add_run_label(fig, run_name)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(plot_dir / "ae_warmup_diagnostics.png", dpi=160)
        plt.close(fig)


def plot_codebook_usage(
    counts,
    plot_dir: Path,
    *,
    run_name: str | None = None,
) -> None:
    counts_tensor = torch.tensor(counts, dtype=torch.float)
    sorted_counts = torch.sort(counts_tensor, descending=True).values.numpy()
    nonzero = counts_tensor[counts_tensor > 0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(sorted_counts)
    axes[0].set_title("Code usage counts, sorted")
    axes[0].set_xlabel("Code rank")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)

    if len(nonzero) > 0:
        axes[1].hist(nonzero, bins=50)
    axes[1].set_title("Nonzero code count histogram")
    axes[1].set_xlabel("Count")
    axes[1].set_ylabel("Codes")
    axes[1].grid(True, alpha=0.3)

    _add_run_label(fig, run_name)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(plot_dir / "codebook_usage.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Initial PCA diagnostic
# ---------------------------------------------------------------------------

def run_initial_pca(
    model,
    val_loader,
    run_dir: Path,
    train_cfg,
    config_payload: dict,
    *,
    enabled: bool,
    max_points: int,
    fit_mode: PCAFitMode,
    strict: bool,
    artifact_name: str = "initial_latent_codebook_pca.png",
    title: str = "Text VQ-VAE initialization: encoder outputs vs. codebook",
) -> None:
    if not enabled:
        return
    if model.quantizer is None:
        config_payload["diagnostics"]["initial_pca"].update({
            "status": "not_applicable",
            "reason": "continuous bottleneck has no codebook",
        })
        print("[Initial PCA] skipped: continuous bottleneck has no codebook.")
        return

    pca_path = run_dir / "plots" / artifact_name
    try:
        encoder_vectors = collect_encoder_vectors(model, val_loader, max_points=max_points)
        pca_result = compare_vector_distributions_pca(
            encoder_vectors.vectors,
            model.quantizer.codebook.weight,
            encoder_pad_ratios=encoder_vectors.pad_ratios,
            fit_mode=fit_mode,
            random_state=train_cfg.seed,
        )
        render_pca_comparison(
            pca_result,
            pca_path,
            run_name=train_cfg.run_name,
            title=title,
        )
        save_pca_metadata(pca_result, pca_path.with_suffix(".json"))
        pca_metadata = pca_result.metadata()
        config_payload["diagnostics"]["initial_pca"].update(
            {"status": "completed", "result": pca_metadata}
        )
        print(
            f"[Initial PCA] {pca_path} "
            f"explained={pca_metadata['total_explained_variance']:.1%}"
        )
    except Exception as exc:
        config_payload["diagnostics"]["initial_pca"].update(
            {"status": "failed", "error": repr(exc)}
        )
        if strict:
            raise
        print(f"[Initial PCA] warning: {exc!r}; training will continue.")
