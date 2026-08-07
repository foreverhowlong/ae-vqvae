"""Train the gated, variable-length GQ-VAE tokenizer on UTF-8 bytes."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from common import ROOT, enable_tf32, get_device
from common.gqvae_config import GQVAEConfig, GQVAEDataConfig, GQVAETrainConfig
from common.text_data import ByteTokenizer, build_text_dataset
from common.tracking import wandb_run
from models.gqvae import GQVAE
from training.text_vqvae.codebook_init import initialize_codebook_kmeans
from training.text_vqvae.loop import batch_to_device, make_loader, split_dataset
from training.text_vqvae.reporting import append_jsonl, atomic_json_dump
from training.text_vqvae.warmup import AdaptiveWarmupController, evaluate_adaptive_warmup


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-name")
    parser.add_argument("--ablation")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--ae-warmup-min-steps", type=int)
    parser.add_argument("--ae-warmup-max-steps", type=int)
    parser.add_argument("--ae-warmup-check-every", type=int)
    parser.add_argument("--ae-warmup-patience", type=int)
    parser.add_argument("--ae-warmup-dim-tolerance", type=int)
    parser.add_argument("--ae-warmup-probe-points", type=int)
    parser.add_argument("--ae-warmup-variance-threshold", type=float)
    parser.add_argument("--compression-warmup-steps", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--dataset-config")
    parser.add_argument("--split")
    parser.add_argument("--text-field")
    parser.add_argument("--data-file")
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument(
        "--continuous-truncation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--code-dim", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--encoder-layers", type=int)
    parser.add_argument("--gater-layers", type=int)
    parser.add_argument("--ffn-mult", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--codebook-size", type=int)
    parser.add_argument("--decode-width", type=int)
    parser.add_argument("--commitment-beta", type=float)
    parser.add_argument("--compression-weight", type=float)
    parser.add_argument("--length-weight", type=float)
    parser.add_argument("--gate-threshold", type=float)
    parser.add_argument("--ema-decay", type=float)
    parser.add_argument("--ema-eps", type=float)


def _override(config, args, fields: tuple[str, ...]) -> None:
    for field in fields:
        value = getattr(args, field, None)
        if value is not None:
            setattr(config, field, value)


def build_configs(args) -> tuple[GQVAETrainConfig, GQVAEDataConfig, GQVAEConfig]:
    train = GQVAETrainConfig()
    data = GQVAEDataConfig()
    model = GQVAEConfig()
    _override(train, args, tuple(asdict(train)))
    _override(data, args, tuple(asdict(data)))
    _override(model, args, tuple(asdict(model)))
    if model.codebook_size != 8192:
        raise ValueError("GQ-VAE experiments lock --codebook-size 8192.")
    if train.ae_warmup_min_steps < 0:
        raise ValueError("--ae-warmup-min-steps must be non-negative.")
    if train.ae_warmup_max_steps <= train.ae_warmup_min_steps:
        raise ValueError("--ae-warmup-max-steps must exceed the minimum.")
    if train.ae_warmup_check_every < 1 or train.ae_warmup_patience < 1:
        raise ValueError("Warmup check interval and patience must be positive.")
    if train.compression_warmup_steps < 1:
        raise ValueError("--compression-warmup-steps must be positive.")
    if model.compression_weight < 0 or model.commitment_beta < 0:
        raise ValueError("Loss weights must be non-negative.")
    return train, data, model


def _make_run_dir(run_name: str) -> Path:
    path = ROOT / "outputs" / "gqvae" / run_name
    if path.exists():
        raise FileExistsError(f"Run directory already exists: {path}")
    (path / "checkpoints").mkdir(parents=True)
    return path


def _materialize_probe(loader, max_points: int) -> list[dict[str, torch.Tensor]]:
    batches = []
    points = 0
    for batch in loader:
        cpu_batch = {key: value.detach().cpu() for key, value in batch.items()}
        batches.append(cpu_batch)
        points += int(cpu_batch["attention_mask"].sum())
        if points >= max_points:
            break
    if not batches:
        raise ValueError("AE warmup probe is empty.")
    return batches


def _compression_weight(config: GQVAEConfig, train: GQVAETrainConfig, vq_step: int) -> float:
    progress = min(vq_step / train.compression_warmup_steps, 1.0)
    return config.compression_weight * progress


@torch.no_grad()
def evaluate(model, loader, device, compression_weight: float, use_quantizer: bool):
    was_training = model.training
    totals: dict[str, float] = {}
    batches = 0
    valid_bytes = 0
    selected_tokens = 0
    current_correct = 0
    gated_tokens = 0
    exact_gated_tokens = 0
    try:
        model.eval()
        for batch in loader:
            batch = batch_to_device(batch, device)
            outputs = model(
                batch["input_ids"],
                batch["attention_mask"],
                use_quantizer=use_quantizer,
                compression_weight=compression_weight,
            )
            for key in (
                "loss",
                "reconstruction_loss",
                "compression_loss",
                "compression_gate_mean",
                "length_loss",
                "commitment_loss",
            ):
                totals[key] = totals.get(key, 0.0) + float(outputs[key])
            valid = batch["attention_mask"].bool()
            predictions = outputs["byte_logits"].argmax(dim=-1)
            current_correct += int(((predictions[:, :, 0] == batch["input_ids"]) & valid).sum())
            valid_bytes += int(valid.sum())
            hard_gates = (outputs["gates"] > model.config.gate_threshold) & valid
            selected_tokens += int(hard_gates.sum())
            lengths = outputs["length_logits"].argmax(dim=-1) + 1
            reconstruction_mask = outputs["reconstruction_mask"] > 0.5
            positions = torch.arange(model.config.decode_width, device=device)
            predicted_mask = positions[None, None, :] < lengths.unsqueeze(-1)
            correct = predictions == model._reconstruction_targets(
                batch["input_ids"], valid, outputs["gates"]
            )[0]
            token_correct = torch.logical_or(~predicted_mask, correct).all(dim=-1)
            # A selected token is exact only when its predicted span is valid and
            # every predicted byte matches the source.
            token_correct &= torch.logical_or(~predicted_mask, reconstruction_mask).all(dim=-1)
            gated_tokens += int(hard_gates.sum())
            exact_gated_tokens += int((token_correct & hard_gates).sum())
            batches += 1
    finally:
        model.train(was_training)
    metrics = {key: value / max(batches, 1) for key, value in totals.items()}
    metrics.update({
        "byte_accuracy": current_correct / max(valid_bytes, 1),
        "exact_gated_token_accuracy": exact_gated_tokens / max(gated_tokens, 1),
        "bytes_per_token": valid_bytes / max(selected_tokens, 1),
        "selected_tokens": selected_tokens,
        "valid_bytes": valid_bytes,
    })
    return metrics


def _save_checkpoint(model, optimizer, path: Path, step: int, epoch: int) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(model.config),
            "step": step,
            "epoch": epoch,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    train_cfg, data_cfg, model_cfg = build_configs(args)
    payload = {
        "train": asdict(train_cfg),
        "data": asdict(data_cfg),
        "model": asdict(model_cfg),
    }
    if args.print_config:
        print(json.dumps(payload, indent=2))
        return

    run_name = train_cfg.run_name or time.strftime("gqvae_%Y%m%d_%H%M%S")
    train_cfg.run_name = run_name
    run_dir = _make_run_dir(run_name)
    device = get_device()
    enable_tf32(device)
    torch.manual_seed(train_cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_cfg.seed)

    tokenizer = ByteTokenizer()
    dataset = build_text_dataset(
        max_seq_len=model_cfg.max_seq_len,
        max_samples=data_cfg.max_train_samples,
        data_file=data_cfg.data_file,
        dataset_name=data_cfg.dataset,
        dataset_config=data_cfg.dataset_config,
        split=data_cfg.split,
        text_field=data_cfg.text_field,
        cache_dir=data_cfg.cache_dir,
        tokenizer=tokenizer,
        continuous_truncation=data_cfg.continuous_truncation,
    )
    train_dataset, val_dataset = split_dataset(
        dataset,
        val_fraction=data_cfg.val_fraction,
        seed=train_cfg.seed,
        max_eval_samples=data_cfg.max_eval_samples,
    )
    train_loader = make_loader(
        train_dataset,
        train_cfg.batch_size,
        shuffle=True,
        device=device,
        num_workers=train_cfg.num_workers,
    )
    init_loader = make_loader(
        train_dataset,
        train_cfg.batch_size,
        shuffle=False,
        device=device,
        num_workers=0,
    )
    val_loader = make_loader(
        val_dataset,
        train_cfg.batch_size,
        shuffle=False,
        device=device,
        num_workers=train_cfg.num_workers,
        persistent_workers=True,
    )
    model = GQVAE(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    payload["train"] = asdict(train_cfg)
    payload["parameter_count"] = sum(p.numel() for p in model.parameters())
    payload["data"].update({
        "train_examples": len(train_dataset),
        "eval_examples": len(val_dataset),
    })
    atomic_json_dump(payload, run_dir / "config.json")

    total_steps = train_cfg.epochs * len(train_loader)
    if train_cfg.ae_warmup_max_steps >= total_steps:
        raise ValueError("Adaptive AE warmup must leave at least one VQ step.")
    probe = _materialize_probe(init_loader, train_cfg.ae_warmup_probe_points)
    controller = AdaptiveWarmupController(
        min_steps=train_cfg.ae_warmup_min_steps,
        max_steps=train_cfg.ae_warmup_max_steps,
        patience=train_cfg.ae_warmup_patience,
        tolerance=train_cfg.ae_warmup_dim_tolerance,
    )
    metrics_path = run_dir / "metrics.jsonl"
    global_step = 0
    transition_step = None
    stop_reason = None
    best_loss = math.inf
    best_step = 0
    last_eval = None
    started = time.time()

    print(f"[Run] {run_name} [Device] {device} [Params] {payload['parameter_count']:,}")
    print("[Codebook] locked at 8192; adaptive AE warmup enabled")
    with wandb_run(
        run_name,
        group="gqvae",
        tags=["text", "gqvae", "variable-length"],
        config=payload,
    ) as tracker:
        for epoch in range(1, train_cfg.epochs + 1):
            model.train()
            for batch in train_loader:
                if stop_reason is not None and transition_step is None:
                    init_result = initialize_codebook_kmeans(
                        model,
                        init_loader,
                        device,
                        seed=train_cfg.seed,
                    )
                    transition_step = global_step
                    payload["codebook_initialization"] = {
                        "status": "completed",
                        "transition_step": transition_step,
                        "warmup_stop_reason": stop_reason,
                        **init_result,
                    }
                    atomic_json_dump(payload, run_dir / "config.json")
                    print(
                        f"[Phase transition] step={global_step} K-means complete; "
                        "entering gated VQ phase"
                    )

                global_step += 1
                use_quantizer = transition_step is not None
                vq_step = global_step - transition_step if use_quantizer else 0
                alpha = (
                    _compression_weight(model_cfg, train_cfg, vq_step)
                    if use_quantizer
                    else 0.0
                )
                batch = batch_to_device(batch, device)
                outputs = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    use_quantizer=use_quantizer,
                    compression_weight=alpha,
                )
                optimizer.zero_grad(set_to_none=True)
                outputs["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_cfg.grad_clip,
                )
                optimizer.step()
                row = {
                    "split": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "vq_step": vq_step,
                    "phase": "gated_vq" if use_quantizer else "ae_warmup",
                    "compression_weight": alpha,
                    "loss": float(outputs["loss"]),
                    "reconstruction_loss": float(outputs["reconstruction_loss"]),
                    "compression_gate_mean": float(outputs["compression_gate_mean"]),
                    "length_loss": float(outputs["length_loss"]),
                    "commitment_loss": float(outputs["commitment_loss"]),
                    "grad_norm": float(grad_norm),
                    "elapsed_sec": time.time() - started,
                }
                append_jsonl(row, metrics_path)
                tracker.log({f"train/{k}": v for k, v in row.items() if isinstance(v, (int, float))}, step=global_step)

                if (
                    transition_step is None
                    and (
                        global_step % train_cfg.ae_warmup_check_every == 0
                        or global_step >= train_cfg.ae_warmup_max_steps
                    )
                ):
                    spectrum = evaluate_adaptive_warmup(
                        model,
                        probe,
                        codebook_size=model_cfg.codebook_size,
                        variance_threshold=train_cfg.ae_warmup_variance_threshold,
                    )
                    decision = controller.observe(global_step, spectrum)
                    append_jsonl(
                        {
                            "split": "ae_warmup_diagnostic",
                            "step": global_step,
                            **spectrum,
                            **decision,
                        },
                        metrics_path,
                    )
                    if decision["should_stop"]:
                        stop_reason = str(decision["reason"])

                eval_due = global_step == 1 or global_step % train_cfg.eval_every == 0
                if eval_due:
                    last_eval = evaluate(
                        model,
                        val_loader,
                        device,
                        alpha,
                        use_quantizer,
                    )
                    eval_row = {
                        "split": "eval",
                        "epoch": epoch,
                        "step": global_step,
                        "vq_step": vq_step,
                        "phase": "gated_vq" if use_quantizer else "ae_warmup",
                        "compression_weight": alpha,
                        **last_eval,
                    }
                    append_jsonl(eval_row, metrics_path)
                    tracker.log({f"eval/{k}": v for k, v in last_eval.items()}, step=global_step)
                    if use_quantizer and last_eval["loss"] < best_loss:
                        best_loss = last_eval["loss"]
                        best_step = global_step
                        _save_checkpoint(
                            model,
                            optimizer,
                            run_dir / "checkpoints" / "best.pt",
                            global_step,
                            epoch,
                        )
                    print(
                        f"[Eval] step={global_step} loss={last_eval['loss']:.4f} "
                        f"bytes/token={last_eval['bytes_per_token']:.3f} "
                        f"byte_acc={last_eval['byte_accuracy']:.3f}"
                    )
                if global_step % train_cfg.save_every == 0:
                    _save_checkpoint(
                        model,
                        optimizer,
                        run_dir / "checkpoints" / f"step{global_step}.pt",
                        global_step,
                        epoch,
                    )

        if transition_step is None:
            raise RuntimeError("Training ended before the adaptive AE warmup transitioned.")
        final_alpha = _compression_weight(
            model_cfg,
            train_cfg,
            global_step - transition_step,
        )
        last_eval = evaluate(model, val_loader, device, final_alpha, True)
        _save_checkpoint(
            model,
            optimizer,
            run_dir / "checkpoints" / "last.pt",
            global_step,
            train_cfg.epochs,
        )

    atomic_json_dump(
        {
            "run_name": run_name,
            "status": "completed",
            "steps": global_step,
            "actual_ae_warmup_steps": transition_step,
            "ae_warmup_stop_reason": stop_reason,
            "best_eval_loss": best_loss,
            "best_step": best_step,
            "final_eval": last_eval,
            "elapsed_sec": time.time() - started,
        },
        run_dir / "summary.json",
    )


if __name__ == "__main__":
    main()

