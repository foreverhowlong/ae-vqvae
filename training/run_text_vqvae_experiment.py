"""Train a non-autoregressive VQ-VAE on TinyStories byte sequences."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split

from common import ROOT, enable_tf32, get_device
from common.text_data import BYTE_PAD, BYTE_VOCAB_SIZE, ByteTokenizer, build_text_dataset
from models.text_vqvae import (
    CollapseControlConfig,
    TextVQVAE,
    TextVQVAEConfig,
    codebook_stats,
    count_parameters,
    text_vqvae_losses,
)


@dataclass
class TrainConfig:
    run_name: str
    seed: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float
    eval_every: int
    save_every: int
    max_train_samples: int | None
    max_eval_samples: int
    val_fraction: float
    num_workers: int
    data_file: str | None
    tinystories_split: str
    dataset_cache_dir: str | None
    streaming: bool
    ablation: str | None


def parse_args():
    parser = argparse.ArgumentParser(description="TinyStories byte-level VQ-VAE experiment")

    parser.add_argument("--run-name", default=None, help="Output run name. Defaults to timestamp.")
    parser.add_argument("--ablation", default=None, help="Free-form ablation label stored in config/logs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--data-file", default=None, help="Optional local .txt or .jsonl file.")
    parser.add_argument("--tinystories-split", default="train")
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--max-eval-samples", type=int, default=2048)
    parser.add_argument("--val-fraction", type=float, default=0.02)

    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--latent-slots", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=448)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--commitment-beta", type=float, default=0.25)

    parser.add_argument(
        "--collapse-preset",
        choices=["none", "anti"],
        default="none",
        help="Use `anti` to enable common anti-collapse engineering measures.",
    )
    parser.add_argument("--use-ema-codebook", dest="use_ema_codebook", action="store_true", default=None)
    parser.add_argument("--no-ema-codebook", dest="use_ema_codebook", action="store_false")
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--ema-eps", type=float, default=None)
    parser.add_argument("--entropy-weight", type=float, default=None)
    parser.add_argument("--entropy-temperature", type=float, default=None)
    parser.add_argument("--diversity-weight", type=float, default=None)
    parser.add_argument("--code-dropout", type=float, default=None)
    parser.add_argument("--stochastic-code-sampling", dest="stochastic_code_sampling", action="store_true", default=None)
    parser.add_argument("--no-stochastic-code-sampling", dest="stochastic_code_sampling", action="store_false")
    parser.add_argument("--sampling-temperature", type=float, default=None)
    parser.add_argument("--sampling-topk", type=int, default=None)
    parser.add_argument("--dead-code-reset-every", type=int, default=None)
    parser.add_argument("--dead-code-reset-usage-threshold", type=float, default=None)
    parser.add_argument("--normalize-latents", dest="normalize_latents", action="store_true", default=None)
    parser.add_argument("--no-normalize-latents", dest="normalize_latents", action="store_false")
    parser.add_argument("--commitment-beta-start", type=float, default=None)
    parser.add_argument("--commitment-beta-warmup-steps", type=int, default=None)

    return parser.parse_args()


def build_collapse_config(args):
    if args.collapse_preset == "anti":
        config = CollapseControlConfig(
            enabled=True,
            use_ema_codebook=True,
            ema_decay=0.99,
            ema_eps=1e-5,
            entropy_weight=0.05,
            entropy_temperature=1.0,
            diversity_weight=0.001,
            code_dropout=0.01,
            stochastic_code_sampling=True,
            sampling_temperature=0.5,
            sampling_topk=8,
            dead_code_reset_every=500,
            dead_code_reset_usage_threshold=1.0,
            normalize_latents=True,
            commitment_beta_start=0.05,
            commitment_beta_warmup_steps=2000,
        )
    else:
        config = CollapseControlConfig()

    overrides = {
        "use_ema_codebook": args.use_ema_codebook,
        "ema_decay": args.ema_decay,
        "ema_eps": args.ema_eps,
        "entropy_weight": args.entropy_weight,
        "entropy_temperature": args.entropy_temperature,
        "diversity_weight": args.diversity_weight,
        "code_dropout": args.code_dropout,
        "stochastic_code_sampling": args.stochastic_code_sampling,
        "sampling_temperature": args.sampling_temperature,
        "sampling_topk": args.sampling_topk,
        "dead_code_reset_every": args.dead_code_reset_every,
        "dead_code_reset_usage_threshold": args.dead_code_reset_usage_threshold,
        "normalize_latents": args.normalize_latents,
        "commitment_beta_start": args.commitment_beta_start,
        "commitment_beta_warmup_steps": args.commitment_beta_warmup_steps,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    config.enabled = any(
        [
            config.use_ema_codebook,
            config.entropy_weight > 0,
            config.diversity_weight > 0,
            config.code_dropout > 0,
            config.stochastic_code_sampling,
            config.dead_code_reset_every > 0,
            config.normalize_latents,
            config.commitment_beta_start is not None,
        ]
    )
    return config


def scheduled_commitment_beta(model_config, collapse_config, step):
    target = model_config.commitment_beta
    if collapse_config.commitment_beta_start is None or collapse_config.commitment_beta_warmup_steps <= 0:
        return target

    progress = min(step / collapse_config.commitment_beta_warmup_steps, 1.0)
    return collapse_config.commitment_beta_start + progress * (target - collapse_config.commitment_beta_start)


def make_run_dir(run_name: str | None):
    if run_name is None:
        run_name = time.strftime("text_vqvae_%Y%m%d_%H%M%S")
    run_dir = ROOT / "outputs" / "text_vqvae" / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    for child in ["checkpoints", "plots", "samples"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir, run_name


def atomic_json_dump(data, path: Path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    tmp_path.replace(path)


def append_jsonl(row, path: Path):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def make_loader(dataset, batch_size, shuffle, device, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=shuffle,
    )


def split_dataset(dataset, val_fraction, seed, max_eval_samples):
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


def compute_accuracy(logits, targets, pad_token_id):
    preds = logits.argmax(dim=-1)
    valid = targets != pad_token_id
    correct = ((preds == targets) & valid).sum().item()
    total = valid.sum().item()
    return correct, total


def optimizer_step(model, optimizer, batch, model_config, collapse_config, grad_clip, beta, step):
    outputs = model(batch["input_ids"], batch["attention_mask"])
    losses = text_vqvae_losses(
        outputs,
        batch["input_ids"],
        pad_token_id=model_config.pad_token_id,
        beta=beta,
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
        )

    correct, total = compute_accuracy(outputs["logits"], batch["input_ids"], model_config.pad_token_id)
    stats = codebook_stats(outputs["indices"], model_config.codebook_size)
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
        "dead_code_resets": reset_count,
    }


@torch.no_grad()
def evaluate(model, data_loader, device, model_config, collapse_config, beta):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_codebook = 0.0
    total_commit = 0.0
    total_entropy = 0.0
    total_assignment_entropy = 0.0
    total_diversity = 0.0
    total_correct = 0
    total_tokens = 0
    all_indices = []
    batches = 0

    for batch in data_loader:
        batch = batch_to_device(batch, device)
        outputs = model(batch["input_ids"], batch["attention_mask"])
        losses = text_vqvae_losses(
            outputs,
            batch["input_ids"],
            pad_token_id=model_config.pad_token_id,
            beta=beta,
            collapse_config=collapse_config,
            codebook_weight=model.quantizer.codebook.weight,
        )
        correct, tokens = compute_accuracy(outputs["logits"], batch["input_ids"], model_config.pad_token_id)

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
        batches += 1

    merged_indices = torch.cat(all_indices, dim=0)
    stats = codebook_stats(merged_indices, model_config.codebook_size)
    avg_recon = total_recon / max(batches, 1)
    return {
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
        "used_codes": stats["used_codes"],
        "dead_codes": stats["dead_codes"],
        "code_counts": stats["counts"].tolist(),
    }


@torch.no_grad()
def write_reconstruction_samples(model, data_loader, device, model_config, path: Path, max_items=16):
    model.eval()
    tokenizer = ByteTokenizer()
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for batch in data_loader:
            batch = batch_to_device(batch, device)
            outputs = model(batch["input_ids"], batch["attention_mask"])
            pred_ids = outputs["logits"].argmax(dim=-1).cpu()
            input_ids = batch["input_ids"].cpu()
            for original, reconstructed in zip(input_ids, pred_ids):
                row = {
                    "original": tokenizer.decode(original.tolist()),
                    "reconstruction": tokenizer.decode(reconstructed.tolist()),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                if written >= max_items:
                    return


def save_checkpoint(model, optimizer, step, epoch, run_dir, name):
    path = run_dir / "checkpoints" / name
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
        },
        path,
    )
    return path


def plot_training_curves(metrics_path: Path, plot_dir: Path):
    rows = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    if not rows:
        return

    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "eval"]

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


def plot_codebook_usage(counts, plot_dir: Path):
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


def main():
    args = parse_args()
    run_dir, run_name = make_run_dir(args.run_name)

    device = get_device()
    enable_tf32(device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_config = TrainConfig(
        run_name=run_name,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_every=args.eval_every,
        save_every=args.save_every,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        data_file=args.data_file,
        tinystories_split=args.tinystories_split,
        dataset_cache_dir=args.dataset_cache_dir,
        streaming=args.streaming,
        ablation=args.ablation,
    )
    model_config = TextVQVAEConfig(
        vocab_size=BYTE_VOCAB_SIZE,
        max_seq_len=args.max_seq_len,
        latent_slots=args.latent_slots,
        d_model=args.d_model,
        n_heads=args.n_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        codebook_size=args.codebook_size,
        commitment_beta=args.commitment_beta,
        pad_token_id=BYTE_PAD,
    )
    collapse_config = build_collapse_config(args)

    config_payload = {
        "train": asdict(train_config),
        "model": model_config.to_dict(),
        "collapse_control": collapse_config.to_dict(),
        "device": str(device),
        "output_dir": str(run_dir),
    }
    atomic_json_dump(config_payload, run_dir / "config.json")

    dataset = build_text_dataset(
        max_seq_len=model_config.max_seq_len,
        max_samples=train_config.max_train_samples,
        data_file=train_config.data_file,
        split=train_config.tinystories_split,
        cache_dir=train_config.dataset_cache_dir,
        streaming=train_config.streaming,
    )
    train_dataset, val_dataset = split_dataset(
        dataset,
        val_fraction=train_config.val_fraction,
        seed=train_config.seed,
        max_eval_samples=train_config.max_eval_samples,
    )

    train_loader = make_loader(
        train_dataset, train_config.batch_size, shuffle=True,
        device=device, num_workers=train_config.num_workers,
    )
    val_loader = make_loader(
        val_dataset, train_config.batch_size, shuffle=False,
        device=device, num_workers=train_config.num_workers,
    )

    model = TextVQVAE(model_config, collapse_config=collapse_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    parameter_count = count_parameters(model)
    config_payload["parameter_count"] = parameter_count
    config_payload["compression"] = {
        "tokens_per_example": model_config.max_seq_len,
        "latent_slots": model_config.latent_slots,
        "nominal_token_to_latent_ratio": model_config.max_seq_len / model_config.latent_slots,
    }
    config_payload["dataset"] = {
        "train_examples": len(train_dataset),
        "eval_examples": len(val_dataset),
    }
    atomic_json_dump(config_payload, run_dir / "config.json")

    print(f"[Run] {run_name}")
    print(f"[Device] {device}")
    print(f"[Params] {parameter_count:,}")
    print(f"[Data] train={len(train_dataset)} eval={len(val_dataset)}")
    print(f"[Output] {run_dir}")

    metrics_path = run_dir / "metrics.jsonl"
    best_eval_loss = float("inf")
    best_step = 0
    global_step = 0
    last_eval = None
    start_time = time.time()

    try:
        for epoch in range(1, train_config.epochs + 1):
            model.train()
            for batch in train_loader:
                global_step += 1
                batch = batch_to_device(batch, device)
                beta = scheduled_commitment_beta(model_config, collapse_config, global_step)
                train_metrics = optimizer_step(
                    model,
                    optimizer,
                    batch,
                    model_config,
                    collapse_config,
                    train_config.grad_clip,
                    beta,
                    global_step,
                )
                train_row = {
                    "split": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "elapsed_sec": time.time() - start_time,
                    **train_metrics,
                }
                append_jsonl(train_row, metrics_path)

                if global_step == 1 or global_step % train_config.eval_every == 0:
                    last_eval = evaluate(model, val_loader, device, model_config, collapse_config, beta)
                    eval_row = {
                        "split": "eval",
                        "epoch": epoch,
                        "step": global_step,
                        "elapsed_sec": time.time() - start_time,
                        **{k: v for k, v in last_eval.items() if k != "code_counts"},
                    }
                    append_jsonl(eval_row, metrics_path)
                    write_reconstruction_samples(
                        model,
                        val_loader,
                        device,
                        model_config,
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

                if global_step % train_config.save_every == 0:
                    save_checkpoint(
                        model, optimizer, global_step, epoch, run_dir, f"step{global_step}.pt"
                    )

        if last_eval is None:
            beta = scheduled_commitment_beta(model_config, collapse_config, global_step)
            last_eval = evaluate(model, val_loader, device, model_config, collapse_config, beta)

        save_checkpoint(model, optimizer, global_step, train_config.epochs, run_dir, "last.pt")
        write_reconstruction_samples(
            model,
            val_loader,
            device,
            model_config,
            run_dir / "samples" / "recon_final.jsonl",
        )
        plot_training_curves(metrics_path, run_dir / "plots")
        plot_codebook_usage(last_eval["code_counts"], run_dir / "plots")

        summary = {
            "run_name": run_name,
            "status": "completed",
            "steps": global_step,
            "epochs": train_config.epochs,
            "best_eval_loss": best_eval_loss,
            "best_step": best_step,
            "final_eval": {k: v for k, v in last_eval.items() if k != "code_counts"},
            "parameter_count": parameter_count,
            "elapsed_sec": time.time() - start_time,
        }
        atomic_json_dump(summary, run_dir / "summary.json")
        shutil.copy2(run_dir / "summary.json", run_dir / "latest_summary.json")

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


if __name__ == "__main__":
    main()
