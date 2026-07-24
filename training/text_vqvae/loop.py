"""Training loop, evaluation, and checkpoint helpers for text VQ-VAE."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from common.text_vqvae_config import (
    CollapseControlConfig,
    DataConfig,
    TextVQVAEConfig,
    TrainConfig,
)
from models.text_vqvae import codebook_stats, text_vqvae_losses
from training.text_vqvae.reporting import (
    append_jsonl,
    build_reconstruction_rows,
    plot_codebook_usage,
    plot_training_curves,
    run_initial_pca,
    write_reconstruction_rows,
    write_reconstruction_samples,
)
from training.text_vqvae.geometry import (
    dump_geometry_snapshot,
    finalize_geometry_artifacts,
    geometry_snapshot_due,
    materialize_geometry_probe,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    device,
    num_workers: int,
    *,
    persistent_workers: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=shuffle,
        persistent_workers=persistent_workers and num_workers > 0,
    )


def split_dataset(dataset, val_fraction: float, seed: int, max_eval_samples: int):
    if len(dataset) < 2:
        raise ValueError("Need at least 2 text examples to create train/eval splits.")
    val_size = max(1, int(len(dataset) * val_fraction))
    val_size = min(val_size, max_eval_samples, len(dataset) - 1)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def batch_to_device(batch, device):
    return {
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
    }


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def scheduled_commitment_beta(
    model_config: TextVQVAEConfig,
    collapse_config: CollapseControlConfig,
    step: int,
) -> float:
    target = model_config.commitment_beta
    if collapse_config.commitment_beta_start is None or collapse_config.commitment_beta_warmup_steps <= 0:
        return target
    progress = min(step / collapse_config.commitment_beta_warmup_steps, 1.0)
    return collapse_config.commitment_beta_start + progress * (target - collapse_config.commitment_beta_start)


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_token_id: int,
    attention_mask: torch.Tensor | None = None,
):
    preds = logits.argmax(dim=-1)
    if attention_mask is None:
        valid = targets != pad_token_id
    else:
        if attention_mask.shape != targets.shape:
            raise ValueError(
                f"attention_mask must have shape {targets.shape}, got {attention_mask.shape}."
            )
        valid = attention_mask.to(device=targets.device, dtype=torch.bool)
    correct = ((preds == targets) & valid).sum().item()
    total = valid.sum().item()
    return correct, total


def compute_bits_per_token(
    codebook_perplexity: float,
    latent_count: int,
    token_count: int,
) -> float:
    """Estimate zero-order entropy-coded latent bits per valid input token."""
    if latent_count <= 0 or token_count <= 0:
        return 0.0
    entropy_bits = math.log2(max(codebook_perplexity, 1.0))
    return latent_count * entropy_bits / token_count


def prune_checkpoints(checkpoint_dir: Path, keep_recent: int = 2) -> None:
    """Keep best.pt plus the most recently modified regular checkpoints."""
    regular_checkpoints = sorted(
        (path for path in checkpoint_dir.glob("*.pt") if path.name != "best.pt"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    for stale_checkpoint in regular_checkpoints[keep_recent:]:
        stale_checkpoint.unlink()


def save_checkpoint(model, optimizer, step: int, epoch: int, run_dir: Path, name: str) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    path = checkpoint_dir / name
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "epoch": epoch},
        path,
    )
    prune_checkpoints(checkpoint_dir)
    return path


# ---------------------------------------------------------------------------
# Train / eval steps
# ---------------------------------------------------------------------------

def optimizer_step(model, optimizer, batch, model_config, collapse_config, grad_clip: float, beta: float, step: int):
    outputs = model(batch["input_ids"], batch["attention_mask"])
    losses = text_vqvae_losses(
        outputs,
        batch["input_ids"],
        pad_token_id=model_config.pad_token_id,
        beta=beta,
        attention_mask=batch["attention_mask"],
        collapse_config=collapse_config,
        codebook_weight=model.quantizer.codebook.weight,
    )
    optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    reset_count = 0
    if (
        collapse_config.dead_code_reset_every > 0
        and step % collapse_config.dead_code_reset_every == 0
        and hasattr(model.quantizer, "reset_dead_codes")
    ):
        reset_count = model.quantizer.reset_dead_codes(
            outputs["z_e"],
            usage_threshold=collapse_config.dead_code_reset_usage_threshold,
            valid_mask=outputs["latent_mask"],
        )

    correct, total = compute_accuracy(
        outputs["logits"],
        batch["input_ids"],
        model_config.pad_token_id,
        attention_mask=batch["attention_mask"],
    )
    stats = codebook_stats(outputs["indices"], model_config.codebook_size, valid_mask=outputs["latent_mask"])
    return {
        "loss": losses["total"].item(),
        "recon_nll": losses["recon"].item(),
        "codebook_loss": losses["codebook"].item(),
        "commitment_loss": losses["commitment"].item(),
        "entropy_loss": losses["entropy"].item(),
        "assignment_entropy": losses["normalized_assignment_entropy"].item(),
        "diversity_loss": losses["diversity"].item(),
        "commitment_beta": beta,
        "token_accuracy": correct / max(total, 1),
        "grad_norm": float(grad_norm),
        "codebook_utilization": stats["utilization"],
        "codebook_perplexity": stats["codebook_perplexity"],
        "bits_per_token": compute_bits_per_token(
            stats["codebook_perplexity"],
            int(stats["counts"].sum().item()),
            total,
        ),
        "dead_code_resets": reset_count,
    }


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    device,
    model_config,
    collapse_config,
    beta: float,
    *,
    tokenizer=None,
    max_reconstruction_items: int = 16,
):
    was_training = model.training
    total_loss = 0.0
    total_recon = 0.0
    total_codebook = 0.0
    total_commit = 0.0
    total_entropy = 0.0
    total_assignment_entropy = 0.0
    total_diversity = 0.0
    total_correct = 0
    total_tokens = 0
    all_indices: list[torch.Tensor] = []
    all_latent_masks: list[torch.Tensor] = []
    reconstruction_rows: list[dict[str, str]] = []
    batches = 0

    try:
        model.eval()
        for batch in data_loader:
            batch = batch_to_device(batch, device)
            outputs = model(batch["input_ids"], batch["attention_mask"])
            losses = text_vqvae_losses(
                outputs,
                batch["input_ids"],
                pad_token_id=model_config.pad_token_id,
                beta=beta,
                attention_mask=batch["attention_mask"],
                collapse_config=collapse_config,
                codebook_weight=model.quantizer.codebook.weight,
            )
            correct, tokens = compute_accuracy(
                outputs["logits"],
                batch["input_ids"],
                model_config.pad_token_id,
                attention_mask=batch["attention_mask"],
            )

            total_loss += losses["total"].item()
            total_recon += losses["recon"].item()
            total_codebook += losses["codebook"].item()
            total_commit += losses["commitment"].item()
            total_entropy += losses["entropy"].item()
            total_assignment_entropy += losses["normalized_assignment_entropy"].item()
            total_diversity += losses["diversity"].item()
            total_correct += correct
            total_tokens += tokens
            all_indices.append(outputs["indices"].detach().cpu())
            all_latent_masks.append(outputs["latent_mask"].detach().cpu())
            batches += 1

            remaining = max_reconstruction_items - len(reconstruction_rows)
            if tokenizer is not None and remaining > 0:
                pred_ids = outputs["logits"].argmax(dim=-1).detach().cpu()
                reconstruction_rows.extend(build_reconstruction_rows(
                    batch["input_ids"],
                    pred_ids,
                    outputs["lengths"],
                    tokenizer,
                    max_items=remaining,
                ))
    finally:
        model.train(was_training)

    merged_indices = torch.cat(all_indices, dim=0)
    merged_latent_masks = torch.cat(all_latent_masks, dim=0)
    stats = codebook_stats(merged_indices, model_config.codebook_size, valid_mask=merged_latent_masks)
    avg_recon = total_recon / max(batches, 1)
    metrics = {
        "loss": total_loss / max(batches, 1),
        "recon_nll": avg_recon,
        "token_ppl": math.exp(min(avg_recon, 20.0)),
        "codebook_loss": total_codebook / max(batches, 1),
        "commitment_loss": total_commit / max(batches, 1),
        "entropy_loss": total_entropy / max(batches, 1),
        "assignment_entropy": total_assignment_entropy / max(batches, 1),
        "diversity_loss": total_diversity / max(batches, 1),
        "commitment_beta": beta,
        "token_accuracy": total_correct / max(total_tokens, 1),
        "codebook_utilization": stats["utilization"],
        "codebook_perplexity": stats["codebook_perplexity"],
        "bits_per_token": compute_bits_per_token(
            stats["codebook_perplexity"],
            int(stats["counts"].sum().item()),
            total_tokens,
        ),
        "used_codes": stats["used_codes"],
        "dead_codes": stats["dead_codes"],
        "code_counts": stats["counts"].tolist(),
    }
    return metrics, reconstruction_rows


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run(
    model,
    optimizer,
    train_loader,
    val_loader,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    model_config: TextVQVAEConfig,
    collapse_config: CollapseControlConfig,
    run_dir: Path,
    run_name: str,
    tokenizer,
    device,
    config_payload: dict,
    tracker,
    initial_pca_opts: dict,
    geometry_snapshot_opts: dict | None = None,
):
    from training.text_vqvae.reporting import atomic_json_dump
    import shutil

    initialization_started = time.time()
    try:
        run_initial_pca(
            model, val_loader, run_dir, train_cfg, config_payload, **initial_pca_opts
        )
    except Exception as exc:
        atomic_json_dump(config_payload, run_dir / "config.json")
        atomic_json_dump(
            {
                "run_name": run_name,
                "status": "failed",
                "steps": 0,
                "error": repr(exc),
                "elapsed_sec": time.time() - initialization_started,
            },
            run_dir / "summary.json",
        )
        raise
    atomic_json_dump(config_payload, run_dir / "config.json")

    metrics_path = run_dir / "metrics.jsonl"
    best_eval_loss = float("inf")
    best_step = 0
    global_step = 0
    last_eval = None
    start_time = time.time()
    geometry_opts = geometry_snapshot_opts or {"enabled": False}
    probe_batches = None

    def take_geometry_snapshot(step: int) -> None:
        if probe_batches is None:
            return
        try:
            geometry_metrics = dump_geometry_snapshot(model, probe_batches, step, run_dir)
            append_jsonl(
                {"split": "geometry", "step": step,
                 "elapsed_sec": time.time() - start_time, **geometry_metrics},
                metrics_path,
            )
            tracker.log({f"geometry/{k}": v for k, v in geometry_metrics.items()}, step=step)
            print(f"[Geometry] step={step} used_codes={geometry_metrics['used_codes']}")
        except Exception as exc:
            if geometry_opts.get("strict", False):
                raise
            print(f"[Geometry] warning at step {step}: {exc!r}; training will continue.")

    if geometry_opts.get("enabled", False):
        try:
            probe_batches = materialize_geometry_probe(
                val_loader,
                latent_slots=model_config.latent_slots,
                max_points=geometry_opts["probe_points"],
                run_dir=run_dir,
            )
            take_geometry_snapshot(0)
        except Exception as exc:
            probe_batches = None
            if geometry_opts.get("strict", False):
                raise
            print(f"[Geometry] warning during probe setup: {exc!r}; snapshots disabled.")

    if config_payload["diagnostics"]["initial_pca"].get("status") == "completed":
        initial_metrics = {
            key: config_payload["diagnostics"]["initial_pca"]["result"][key]
            for key in (
                "encoder_mean_norm",
                "encoder_norm_std",
                "codebook_mean_norm",
                "codebook_norm_std",
                "encoder_to_nearest_code_mean_distance",
                "encoder_pairwise_mean_distance",
            )
        }
        append_jsonl(
            {"split": "initialization", "step": 0, "elapsed_sec": 0.0, **initial_metrics},
            metrics_path,
        )
        tracker.log({f"initial/{k}": v for k, v in initial_metrics.items()}, step=0)

    try:
        for epoch in range(1, train_cfg.epochs + 1):
            model.train()
            for batch in train_loader:
                global_step += 1
                batch = batch_to_device(batch, device)
                beta = scheduled_commitment_beta(model_config, collapse_config, global_step)
                train_metrics = optimizer_step(
                    model, optimizer, batch, model_config, collapse_config,
                    train_cfg.grad_clip, beta, global_step,
                )
                append_jsonl(
                    {"split": "train", "epoch": epoch, "step": global_step,
                     "elapsed_sec": time.time() - start_time, **train_metrics},
                    metrics_path,
                )
                tracker.log({f"train/{k}": v for k, v in train_metrics.items()}, step=global_step)

                if probe_batches is not None and geometry_snapshot_due(
                    global_step,
                    dense_every=geometry_opts["dense_every"],
                    dense_until=geometry_opts["dense_until"],
                    sparse_every=geometry_opts["sparse_every"],
                ):
                    take_geometry_snapshot(global_step)

                if global_step == 1 or global_step % train_cfg.eval_every == 0:
                    last_eval, reconstruction_rows = evaluate(
                        model,
                        val_loader,
                        device,
                        model_config,
                        collapse_config,
                        beta,
                        tokenizer=tokenizer,
                    )
                    append_jsonl(
                        {"split": "eval", "epoch": epoch, "step": global_step,
                         "elapsed_sec": time.time() - start_time,
                         **{k: v for k, v in last_eval.items() if k != "code_counts"}},
                        metrics_path,
                    )
                    tracker.log(
                        {f"eval/{k}": v for k, v in last_eval.items() if k != "code_counts"},
                        step=global_step,
                    )
                    write_reconstruction_rows(
                        reconstruction_rows,
                        run_dir / "samples" / f"recon_step{global_step}.jsonl",
                    )
                    if last_eval["loss"] < best_eval_loss:
                        best_eval_loss = last_eval["loss"]
                        best_step = global_step
                        save_checkpoint(model, optimizer, global_step, epoch, run_dir, "best.pt")
                    print(
                        f"[Eval] step={global_step} loss={last_eval['loss']:.4f} "
                        f"ppl={last_eval['token_ppl']:.2f} acc={last_eval['token_accuracy']:.3f} "
                        f"util={last_eval['codebook_utilization']:.3f}"
                    )

                if global_step % train_cfg.save_every == 0:
                    save_checkpoint(model, optimizer, global_step, epoch, run_dir, f"step{global_step}.pt")

        if last_eval is None:
            beta = scheduled_commitment_beta(model_config, collapse_config, global_step)
            last_eval, _ = evaluate(
                model, val_loader, device, model_config, collapse_config, beta
            )

        save_checkpoint(model, optimizer, global_step, train_cfg.epochs, run_dir, "last.pt")
        if probe_batches is not None and not geometry_snapshot_due(
            global_step,
            dense_every=geometry_opts["dense_every"],
            dense_until=geometry_opts["dense_until"],
            sparse_every=geometry_opts["sparse_every"],
        ):
            take_geometry_snapshot(global_step)
        write_reconstruction_samples(
            model, val_loader, device, model_config, tokenizer,
            run_dir / "samples" / "recon_final.jsonl",
        )
        plot_training_curves(metrics_path, run_dir / "plots")
        plot_codebook_usage(last_eval["code_counts"], run_dir / "plots")

        geometry_render = {"status": "disabled", "snapshots_retained": True}
        if probe_batches is not None:
            try:
                geometry_render = finalize_geometry_artifacts(
                    run_dir,
                    enabled=geometry_opts.get("render_enabled", False),
                    basis=geometry_opts.get("render_basis", "first_last"),
                    fps=geometry_opts.get("render_fps", 8),
                    keep_snapshots=geometry_opts.get("keep_snapshots", True),
                )
                if geometry_render["status"] == "completed":
                    print(
                        "[Geometry] rendered final artifacts; "
                        f"snapshots_retained={geometry_render['snapshots_retained']}"
                    )
            except Exception as exc:
                geometry_render = {
                    "status": "failed",
                    "error": repr(exc),
                    "snapshots_retained": True,
                }
                print(
                    f"[Geometry] final rendering failed: {exc!r}; "
                    "raw snapshots were retained."
                )

        config_payload["diagnostics"]["geometry"].update({
            "render_status": geometry_render["status"],
            "render_result": geometry_render,
        })
        atomic_json_dump(config_payload, run_dir / "config.json")

        summary = {
            "run_name": run_name,
            "status": "completed",
            "steps": global_step,
            "epochs": train_cfg.epochs,
            "best_eval_loss": best_eval_loss,
            "best_step": best_step,
            "final_eval": {k: v for k, v in last_eval.items() if k != "code_counts"},
            "parameter_count": config_payload.get("parameter_count"),
            "geometry_render": geometry_render,
            "elapsed_sec": time.time() - start_time,
        }
        atomic_json_dump(summary, run_dir / "summary.json")
        shutil.copy2(run_dir / "summary.json", run_dir / "latest_summary.json")
        tracker.summary.update({
            "best_eval_loss": best_eval_loss,
            "best_step": best_step,
            "steps": global_step,
        })

    except Exception as exc:
        atomic_json_dump(
            {
                "run_name": run_name,
                "status": "failed",
                "steps": global_step,
                "error": repr(exc),
                "elapsed_sec": time.time() - start_time,
            },
            run_dir / "summary.json",
        )
        raise
