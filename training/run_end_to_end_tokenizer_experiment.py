"""Train the greedy VQ tokenizer directly on a two-part causal codelength."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict

import torch
from torch.utils.data import Subset

from common import ROOT, enable_tf32, get_device
from common.end_to_end_tokenizer_config import (
    EndToEndTokenizerConfig,
    EndToEndTokenizerDataConfig,
    EndToEndTokenizerTrainConfig,
)
from common.end_to_end_tokenizer_data import build_end_to_end_tokenizer_dataset
from common.text_data import BPETokenizer
from common.tracking import wandb_run
from models.end_to_end_tokenizer import (
    EndToEndTokenizerModel,
    end_to_end_tokenizer_losses,
)
from models.text_vqvae import codebook_stats
from training.text_vqvae.loop import make_loader, split_dataset
from training.text_vqvae.reporting import append_jsonl, atomic_json_dump
from training.segmental_vqvae_reporting import (
    build_segmentation_snapshot,
    write_segmentation_visualization,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-name")
    parser.add_argument("--ablation")
    parser.add_argument("--print-config", action="store_true")
    for name, value_type in (
        ("seed", int),
        ("epochs", int),
        ("batch-size", int),
        ("gradient-accumulation-steps", int),
        ("learning-rate", float),
        ("weight-decay", float),
        ("grad-clip", float),
        ("eval-every", int),
        ("num-workers", int),
        ("target-prior-parameters", int),
        ("parameter-tolerance", float),
        ("vq-warmup-steps", int),
        ("prior-anneal-steps", int),
        ("max-train-samples", int),
        ("max-eval-samples", int),
        ("val-fraction", float),
        ("max-seq-len", int),
        ("segmenter-d-model", int),
        ("segmenter-n-heads", int),
        ("boundary-encoder-layers", int),
        ("boundary-window-radius", int),
        ("max-span-length", int),
        ("span-encoder-layers", int),
        ("segmenter-ffn-mult", int),
        ("latent-dim", int),
        ("codebook-size", int),
        ("commitment-beta", float),
        ("ema-decay", float),
        ("ema-eps", float),
        ("code-target-topk", int),
        ("code-target-temperature", float),
        ("prior-layers", int),
        ("prior-heads", int),
        ("prior-d-model", int),
        ("prior-dropout", float),
        ("text-decoder-layers", int),
        ("text-decoder-heads", int),
        ("text-decoder-d-model", int),
        ("text-decoder-dropout", float),
        ("compression-target", float),
        ("rate-dual-initial", float),
        ("rate-dual-lr", float),
        ("rate-dual-max-abs", float),
        ("dropout", float),
    ):
        parser.add_argument(f"--{name}", type=value_type)
    for name in (
        "tokenizer-path",
        "dataset",
        "dataset-config",
        "split",
        "text-field",
        "data-file",
        "cache-dir",
    ):
        parser.add_argument(f"--{name}")
    for name in (
        "continuous-truncation",
        "word-boundary-only",
        "prior-bias",
        "text-decoder-bias",
        "save-last-resume",
    ):
        parser.add_argument(
            f"--{name}",
            action=argparse.BooleanOptionalAction,
            default=None,
        )


def _override(config, args) -> None:
    for field_name in asdict(config):
        value = getattr(args, field_name, None)
        if value is not None:
            setattr(config, field_name, value)


def build_configs(
    args,
    tokenizer: BPETokenizer,
) -> tuple[
    EndToEndTokenizerTrainConfig,
    EndToEndTokenizerDataConfig,
    EndToEndTokenizerConfig,
]:
    train = EndToEndTokenizerTrainConfig()
    data = EndToEndTokenizerDataConfig()
    model = EndToEndTokenizerConfig()
    _override(train, args)
    _override(data, args)
    _override(model, args)
    if tokenizer.bos_token_id is None:
        raise ValueError("The end-to-end causal decoder requires a BPE <bos> token.")
    model.vocab_size = tokenizer.vocab_size
    model.pad_token_id = tokenizer.pad_token_id
    model.bos_token_id = tokenizer.bos_token_id
    model.eos_token_id = tokenizer.eos_token_id
    model.validate()
    if data.continuous_truncation:
        raise ValueError("This experiment intentionally forbids continuous truncation.")
    if min(
        train.epochs,
        train.batch_size,
        train.gradient_accumulation_steps,
        train.eval_every,
    ) < 1:
        raise ValueError("Training counts and cadence must be positive.")
    if train.vq_warmup_steps < 0 or train.prior_anneal_steps < 0:
        raise ValueError("Warmup and annealing steps must be non-negative.")
    if not 0 <= data.val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1).")
    return train, data, model


def prior_objective_weight(
    optimizer_step: int,
    *,
    vq_warmup_steps: int,
    prior_anneal_steps: int,
) -> float:
    """Keep the latent prior off during VQ warmup, then introduce it linearly."""
    if optimizer_step <= vq_warmup_steps:
        return 0.0
    if prior_anneal_steps == 0:
        return 1.0
    return min(
        (optimizer_step - vq_warmup_steps) / prior_anneal_steps,
        1.0,
    )


def _set_prior_trainable(
    model: EndToEndTokenizerModel,
    trainable: bool,
) -> None:
    model.chunk_prior.requires_grad_(trainable)
    model.chunk_prior.train(trainable)


def _batch_to_device(batch, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


@torch.no_grad()
def write_segmentation_diagnostics(
    model: EndToEndTokenizerModel,
    cpu_batch: dict[str, torch.Tensor],
    device: torch.device,
    tokenizer: BPETokenizer,
    run_dir,
    run_name: str,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    try:
        batch = _batch_to_device(cpu_batch, device)
        outputs = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["legal_endpoints"],
        )
        snapshot = build_segmentation_snapshot(
            batch["input_ids"],
            batch["attention_mask"],
            outputs,
            tokenizer=tokenizer,
        )
        write_segmentation_visualization(
            snapshot,
            run_dir / "plots",
            compression_target=model.config.compression_target,
            run_name=run_name,
        )
        return {
            key: float(snapshot[key])
            for key in (
                "singleton_chunk_fraction",
                "excess_singleton_fraction",
                "chunk_length_p50",
                "chunk_length_p90",
                "boundary_position_dependence_hard",
            )
        }
    finally:
        model.train(was_training)


@torch.no_grad()
def evaluate(
    model: EndToEndTokenizerModel,
    loader,
    device: torch.device,
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    totals = {
        "generative_nll_sum": 0.0,
        "length_nll_sum": 0.0,
        "code_nll_sum": 0.0,
        "text_nll_sum": 0.0,
        "tokens": 0,
        "chunks": 0,
        "raw_bytes": 0,
        "text_correct": 0,
        "length_correct": 0,
        "code_correct": 0,
    }
    all_indices = []
    all_masks = []
    try:
        for cpu_batch in loader:
            batch = _batch_to_device(cpu_batch, device)
            outputs = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["legal_endpoints"],
            )
            losses = end_to_end_tokenizer_losses(
                outputs,
                batch["input_ids"],
                batch["attention_mask"],
                model,
            )
            for key in (
                "generative_nll_sum",
                "length_nll_sum",
                "code_nll_sum",
                "text_nll_sum",
            ):
                totals[key] += float(losses[key])
            valid = batch["attention_mask"].bool()
            chunks = outputs["latent_mask"].bool()
            totals["tokens"] += int(valid.sum())
            totals["chunks"] += int(chunks.sum())
            totals["raw_bytes"] += int(batch["raw_byte_count"].sum())
            totals["text_correct"] += int(
                ((outputs["text_logits"].argmax(dim=-1) == batch["input_ids"]) & valid).sum()
            )
            totals["length_correct"] += int(
                (
                    outputs["length_logits"].argmax(dim=-1)
                    == outputs["chunk_lengths"] - 1
                )[chunks].sum()
            )
            totals["code_correct"] += int(
                (outputs["code_logits"].argmax(dim=-1) == outputs["indices"])[chunks].sum()
            )
            all_indices.append(outputs["indices"].detach().cpu())
            all_masks.append(chunks.detach().cpu())
    finally:
        model.train(was_training)

    tokens = max(int(totals["tokens"]), 1)
    chunks = max(int(totals["chunks"]), 1)
    raw_bytes = max(int(totals["raw_bytes"]), 1)
    stats = codebook_stats(
        torch.cat(all_indices),
        model.config.codebook_size,
        torch.cat(all_masks),
    )
    return {
        "bits_per_raw_byte": totals["generative_nll_sum"] / (math.log(2.0) * raw_bytes),
        "generative_nll_per_bpe": totals["generative_nll_sum"] / tokens,
        "length_nll_per_bpe": totals["length_nll_sum"] / tokens,
        "code_nll_per_bpe": totals["code_nll_sum"] / tokens,
        "text_nll_per_bpe": totals["text_nll_sum"] / tokens,
        "tokens_per_chunk": totals["tokens"] / chunks,
        "chunks_per_token": totals["chunks"] / tokens,
        "text_token_accuracy": totals["text_correct"] / tokens,
        "length_accuracy": totals["length_correct"] / chunks,
        "code_accuracy": totals["code_correct"] / chunks,
        "tokens": int(totals["tokens"]),
        "chunks": int(totals["chunks"]),
        "raw_utf8_bytes": int(totals["raw_bytes"]),
        "used_codes": stats["used_codes"],
        "codebook_utilization": stats["utilization"],
        "codebook_perplexity": stats["codebook_perplexity"],
        "rate_dual": float(model.rate_dual),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    tokenizer_path = args.tokenizer_path or EndToEndTokenizerTrainConfig.tokenizer_path
    tokenizer = BPETokenizer(tokenizer_path)
    train_cfg, data_cfg, model_cfg = build_configs(args, tokenizer)
    model = EndToEndTokenizerModel(model_cfg)
    prior_parameters = model.prior_parameter_count()
    prior_error = abs(
        prior_parameters - train_cfg.target_prior_parameters
    ) / train_cfg.target_prior_parameters
    if prior_error > train_cfg.parameter_tolerance:
        raise ValueError(
            f"Chunk prior has {prior_parameters:,} parameters, outside the "
            f"{train_cfg.parameter_tolerance:.1%} tolerance around "
            f"{train_cfg.target_prior_parameters:,}."
        )
    payload = {
        "train": asdict(train_cfg),
        "data": asdict(data_cfg),
        "model": asdict(model_cfg),
        "prior_parameter_count": prior_parameters,
        "total_parameter_count": model.parameter_count(),
        "prior_parameter_error_fraction": prior_error,
        "metric_contract": {
            "bits_per_raw_byte": "(length NLL + code NLL + text NLL) / raw UTF-8 bytes / ln(2)",
            "excluded_from_bpb": ["commitment loss", "rate dual constraint"],
        },
    }
    if args.print_config:
        print(json.dumps(payload, indent=2))
        return

    run_name = train_cfg.run_name or time.strftime("end_to_end_tokenizer_%Y%m%d_%H%M%S")
    train_cfg.run_name = run_name
    run_dir = ROOT / "outputs" / "end_to_end_tokenizer" / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    device = get_device()
    enable_tf32(device)
    torch.manual_seed(train_cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_cfg.seed)

    dataset = build_end_to_end_tokenizer_dataset(
        tokenizer=tokenizer,
        max_seq_len=model_cfg.max_seq_len,
        max_samples=data_cfg.max_train_samples,
        word_boundary_only=model_cfg.word_boundary_only,
        data_file=data_cfg.data_file,
        dataset_name=data_cfg.dataset,
        dataset_config=data_cfg.dataset_config,
        split=data_cfg.split,
        text_field=data_cfg.text_field,
        cache_dir=data_cfg.cache_dir,
    )
    train_dataset, validation_dataset = split_dataset(
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
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / train_cfg.gradient_accumulation_steps
    )
    total_optimizer_steps = train_cfg.epochs * optimizer_steps_per_epoch
    if (
        train_cfg.vq_warmup_steps + train_cfg.prior_anneal_steps
        >= total_optimizer_steps
    ):
        raise ValueError(
            "VQ warmup plus prior annealing must leave at least one full-objective step."
        )
    validation_loader = make_loader(
        validation_dataset,
        train_cfg.batch_size,
        shuffle=False,
        device=device,
        num_workers=train_cfg.num_workers,
    )
    diagnostic_loader = make_loader(
        Subset(
            validation_dataset,
            range(min(32, len(validation_dataset))),
        ),
        min(32, len(validation_dataset)),
        shuffle=False,
        device=device,
        num_workers=0,
    )
    diagnostic_batch = next(iter(diagnostic_loader))
    model = model.to(device)
    _set_prior_trainable(model, train_cfg.vq_warmup_steps == 0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    payload["train"] = asdict(train_cfg)
    payload["data"].update({
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
    })
    atomic_json_dump(payload, run_dir / "config.json")

    print(
        f"[Run] {run_name} [Device] {device} "
        f"[Prior params] {prior_parameters:,} [Total params] {model.parameter_count():,}"
    )
    print(
        "[Objective] length NLL + code NLL + causal text NLL; "
        f"hard rate target={model_cfg.compression_target:.3f} BPE/chunk"
    )
    if train_cfg.vq_warmup_steps:
        print(
            "[VQ warmup] "
            f"text+commitment+rate for {train_cfg.vq_warmup_steps} steps; "
            f"length/code NLL annealed over {train_cfg.prior_anneal_steps} steps"
        )
    metrics_path = run_dir / "metrics.jsonl"
    global_step = 0
    best_bpb = math.inf
    best_step = 0
    final_validation = None
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    accumulated_rate = 0.0

    with wandb_run(
        run_name,
        group="end-to-end-tokenizer",
        tags=["end-to-end", "greedy", "vq", "nanogpt", "bpb"],
        config=payload,
    ) as tracker:
        for epoch in range(1, train_cfg.epochs + 1):
            model.train()
            for batch_index, cpu_batch in enumerate(train_loader):
                prior_weight = prior_objective_weight(
                    global_step + 1,
                    vq_warmup_steps=train_cfg.vq_warmup_steps,
                    prior_anneal_steps=train_cfg.prior_anneal_steps,
                )
                _set_prior_trainable(model, prior_weight > 0.0)
                batch = _batch_to_device(cpu_batch, device)
                outputs = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["legal_endpoints"],
                )
                losses = end_to_end_tokenizer_losses(
                    outputs,
                    batch["input_ids"],
                    batch["attention_mask"],
                    model,
                    prior_weight=prior_weight,
                )
                (losses["loss"] / train_cfg.gradient_accumulation_steps).backward()
                accumulated += 1
                accumulated_rate += float(losses["hard_chunks_per_token"])
                epoch_finished = batch_index + 1 == len(train_loader)
                if (
                    accumulated < train_cfg.gradient_accumulation_steps
                    and not epoch_finished
                ):
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_cfg.grad_clip,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                hard_rate = accumulated_rate / accumulated
                model.update_rate_dual(hard_rate)
                train_row = {
                    "split": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "loss": float(losses["loss"].detach()),
                    "training_nll_per_bpe": float(losses["training_nll_per_bpe"].detach()),
                    "generative_nll_per_bpe": float(losses["generative_nll_per_bpe"].detach()),
                    "length_nll_per_bpe": float(losses["length_nll_per_bpe"].detach()),
                    "code_nll_per_bpe": float(losses["code_nll_per_bpe"].detach()),
                    "text_nll_per_bpe": float(losses["text_nll_per_bpe"].detach()),
                    "commitment_loss": float(losses["commitment_loss"].detach()),
                    "rate_constraint_loss": float(losses["rate_constraint_loss"].detach()),
                    "tokens_per_chunk": 1.0 / max(hard_rate, 1e-12),
                    "rate_dual": float(model.rate_dual),
                    "prior_weight": prior_weight,
                    "phase": (
                        "vq_warmup"
                        if prior_weight == 0.0
                        else "prior_anneal"
                        if prior_weight < 1.0
                        else "joint"
                    ),
                    "grad_norm": float(grad_norm),
                    "elapsed_sec": time.time() - started,
                }
                append_jsonl(train_row, metrics_path)
                tracker.log(
                    {f"train/{key}": value for key, value in train_row.items() if isinstance(value, (int, float))},
                    step=global_step,
                )
                accumulated = 0
                accumulated_rate = 0.0

                if global_step % train_cfg.eval_every == 0:
                    final_validation = evaluate(model, validation_loader, device)
                    final_validation.update(write_segmentation_diagnostics(
                        model,
                        diagnostic_batch,
                        device,
                        tokenizer,
                        run_dir,
                        run_name,
                    ))
                    validation_row = {
                        "split": "validation",
                        "step": global_step,
                        **final_validation,
                    }
                    append_jsonl(validation_row, metrics_path)
                    tracker.log(
                        {f"validation/{key}": value for key, value in final_validation.items()},
                        step=global_step,
                    )
                    if final_validation["bits_per_raw_byte"] < best_bpb:
                        best_bpb = float(final_validation["bits_per_raw_byte"])
                        best_step = global_step
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "model_config": asdict(model_cfg),
                                "step": global_step,
                            },
                            run_dir / "checkpoints" / "best.pt",
                        )
                    print(
                        f"[Validation] step={global_step} "
                        f"BPB={final_validation['bits_per_raw_byte']:.5f} "
                        f"rate={final_validation['tokens_per_chunk']:.3f}"
                    )

        final_validation = evaluate(model, validation_loader, device)
        final_validation.update(write_segmentation_diagnostics(
            model,
            diagnostic_batch,
            device,
            tokenizer,
            run_dir,
            run_name,
        ))
        if final_validation["bits_per_raw_byte"] < best_bpb:
            best_bpb = float(final_validation["bits_per_raw_byte"])
            best_step = global_step
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": asdict(model_cfg),
                    "step": global_step,
                },
                run_dir / "checkpoints" / "best.pt",
            )
        if train_cfg.save_last_resume:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": asdict(model_cfg),
                    "step": global_step,
                },
                run_dir / "checkpoints" / "last.pt",
            )

    atomic_json_dump(
        {
            "run_name": run_name,
            "status": "completed",
            "steps": global_step,
            "best_validation_bpb": best_bpb,
            "best_step": best_step,
            "final_validation": final_validation,
            "prior_parameter_count": prior_parameters,
            "total_parameter_count": model.parameter_count(),
            "elapsed_sec": time.time() - started,
        },
        run_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
