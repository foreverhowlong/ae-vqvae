"""File I/O, plots, PCA diagnostics, and sample writing for text VQ-VAE."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

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

def plot_training_curves(metrics_path: Path, plot_dir: Path) -> None:
    rows = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    if not rows:
        return

    train_rows = [r for r in rows if r["split"] == "train"]
    eval_rows = [r for r in rows if r["split"] == "eval"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    if train_rows:
        axes[0, 0].plot([r["step"] for r in train_rows], [r["loss"] for r in train_rows], label="train")
    if eval_rows:
        axes[0, 0].plot([r["step"] for r in eval_rows], [r["loss"] for r in eval_rows], label="eval")
    axes[0, 0].set_title("Loss")
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

    if train_rows:
        axes[1, 1].plot([r["step"] for r in train_rows], [r["codebook_utilization"] for r in train_rows], label="train util")
    if eval_rows:
        axes[1, 1].plot([r["step"] for r in eval_rows], [r["codebook_utilization"] for r in eval_rows], label="eval util")
    axes[1, 1].set_title("Codebook utilization")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def plot_codebook_usage(counts, plot_dir: Path) -> None:
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

    fig.tight_layout()
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
    fit_mode: str,
    strict: bool,
) -> None:
    if not enabled:
        return

    pca_path = run_dir / "plots" / "initial_latent_codebook_pca.png"
    try:
        encoder_vectors = collect_encoder_vectors(model, val_loader, max_points=max_points)
        pca_result = compare_vector_distributions_pca(
            encoder_vectors.vectors,
            model.quantizer.codebook.weight,
            encoder_pad_ratios=encoder_vectors.pad_ratios,
            fit_mode=fit_mode,
            random_state=train_cfg.seed,
        )
        render_pca_comparison(pca_result, pca_path)
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
