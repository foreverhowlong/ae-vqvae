"""Train an approximately 18M-parameter nanoGPT and report validation BPB."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from common import ROOT, enable_tf32, get_device
from common.tracking import wandb_run
from models.nanogpt import NanoGPT, NanoGPTConfig
from training.text_vqvae.reporting import append_jsonl, atomic_json_dump


@dataclass
class NanoGPTTrainConfig:
    run_name: str = ""
    seed: int = 42
    epochs: int = 1
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 1000
    eval_stride: int = 128
    eval_batch_size: int = 16
    eval_max_documents: int | None = None
    target_parameters: int = 18_000_000
    parameter_tolerance: float = 0.05
    ablation: str | None = None


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-name")
    parser.add_argument("--ablation")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--min-lr", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--eval-stride", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--eval-max-documents", type=int)
    parser.add_argument("--target-parameters", type=int)
    parser.add_argument("--parameter-tolerance", type=float)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--n-layer", type=int)
    parser.add_argument("--n-head", type=int)
    parser.add_argument("--n-embd", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=None)


def build_configs(args, metadata: dict) -> tuple[NanoGPTTrainConfig, NanoGPTConfig]:
    train = NanoGPTTrainConfig()
    model = NanoGPTConfig(vocab_size=int(metadata["vocab_size"]))
    for field in asdict(train):
        value = getattr(args, field, None)
        if value is not None:
            setattr(train, field, value)
    for field in asdict(model):
        value = getattr(args, field, None)
        if value is not None:
            setattr(model, field, value)
    if train.epochs < 1 or train.batch_size < 1 or train.gradient_accumulation_steps < 1:
        raise ValueError("Epoch, batch, and accumulation counts must be positive.")
    if not 0 < train.eval_stride <= model.block_size:
        raise ValueError("--eval-stride must be in [1, block-size].")
    if model.n_embd % model.n_head:
        raise ValueError("--n-embd must be divisible by --n-head.")
    return train, model


def _lr(step: int, max_steps: int, config: NanoGPTTrainConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(config.warmup_steps, 1)
    if step >= max_steps:
        return config.min_lr
    ratio = (step - config.warmup_steps) / max(max_steps - config.warmup_steps, 1)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return config.min_lr + coefficient * (config.learning_rate - config.min_lr)


def _validation_windows(
    tokens: np.memmap,
    offsets: np.ndarray,
    *,
    block_size: int,
    stride: int,
    max_documents: int | None,
):
    document_count = len(offsets) - 1
    if max_documents is not None:
        document_count = min(document_count, max_documents)
    for document in range(document_count):
        start = int(offsets[document])
        end = int(offsets[document + 1])
        document_tokens = tokens[start:end]
        target_start = 1
        while target_start < len(document_tokens):
            target_end = min(target_start + stride, len(document_tokens))
            context_start = max(0, target_end - block_size - 1)
            inputs = np.asarray(
                document_tokens[context_start : target_end - 1],
                dtype=np.int64,
            )
            targets = np.asarray(
                document_tokens[context_start + 1 : target_end],
                dtype=np.int64,
            )
            score_from = target_start - context_start - 1
            targets[:score_from] = -1
            yield inputs, targets
            target_start = target_end


@torch.no_grad()
def evaluate_bpb(
    model: NanoGPT,
    data_dir: Path,
    metadata: dict,
    config: NanoGPTTrainConfig,
    device: torch.device,
) -> dict[str, float | int]:
    tokens = np.memmap(data_dir / "validation.bin", dtype=np.uint16, mode="r")
    offsets = np.fromfile(data_dir / "validation.idx", dtype=np.uint64)
    byte_counts = np.fromfile(data_dir / "validation.bytes", dtype=np.uint64)
    document_count = len(offsets) - 1
    if config.eval_max_documents is not None:
        document_count = min(document_count, config.eval_max_documents)
    raw_bytes = int(byte_counts[:document_count].sum())
    loss_nats = 0.0
    predicted_tokens = 0
    batch_inputs = []
    batch_targets = []
    was_training = model.training
    model.eval()

    def flush() -> None:
        nonlocal loss_nats, predicted_tokens
        if not batch_inputs:
            return
        max_length = max(len(values) for values in batch_inputs)
        inputs = torch.full(
            (len(batch_inputs), max_length),
            int(metadata["pad_token_id"]),
            dtype=torch.long,
            device=device,
        )
        targets = torch.full_like(inputs, -1)
        for row, (input_values, target_values) in enumerate(
            zip(batch_inputs, batch_targets, strict=True)
        ):
            inputs[row, : len(input_values)] = torch.from_numpy(input_values).to(device)
            targets[row, : len(target_values)] = torch.from_numpy(target_values).to(device)
        _, loss = model(inputs, targets, reduction="sum")
        assert loss is not None
        loss_nats += float(loss)
        predicted_tokens += int((targets != -1).sum())
        batch_inputs.clear()
        batch_targets.clear()

    try:
        for inputs, targets in _validation_windows(
            tokens,
            offsets,
            block_size=model.config.block_size,
            stride=config.eval_stride,
            max_documents=config.eval_max_documents,
        ):
            batch_inputs.append(inputs)
            batch_targets.append(targets)
            if len(batch_inputs) >= config.eval_batch_size:
                flush()
        flush()
    finally:
        model.train(was_training)
    return {
        "loss_nats": loss_nats,
        "predicted_tokens": predicted_tokens,
        "raw_utf8_bytes": raw_bytes,
        "token_nll": loss_nats / max(predicted_tokens, 1),
        "token_ppl": math.exp(min(loss_nats / max(predicted_tokens, 1), 20.0)),
        "bits_per_raw_byte": loss_nats / (math.log(2.0) * max(raw_bytes, 1)),
        "documents": document_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    metadata = json.loads((args.data_dir / "meta.json").read_text(encoding="utf-8"))
    train_cfg, model_cfg = build_configs(args, metadata)
    model = NanoGPT(model_cfg)
    parameter_count = model.count_parameters()
    parameter_error = abs(parameter_count - train_cfg.target_parameters) / train_cfg.target_parameters
    if parameter_error > train_cfg.parameter_tolerance:
        raise ValueError(
            f"NanoGPT has {parameter_count:,} parameters, outside the "
            f"{train_cfg.parameter_tolerance:.1%} tolerance around "
            f"{train_cfg.target_parameters:,}."
        )
    payload = {
        "train": asdict(train_cfg),
        "model": asdict(model_cfg),
        "corpus": metadata,
        "parameter_count": parameter_count,
        "parameter_error_fraction": parameter_error,
    }
    if args.print_config:
        print(json.dumps(payload, indent=2))
        return

    run_name = train_cfg.run_name or time.strftime("nanogpt_%Y%m%d_%H%M%S")
    train_cfg.run_name = run_name
    run_dir = ROOT / "outputs" / "nanogpt" / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    payload["train"] = asdict(train_cfg)
    atomic_json_dump(payload, run_dir / "config.json")
    device = get_device()
    enable_tf32(device)
    torch.manual_seed(train_cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_cfg.seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay,
    )
    train_tokens = np.memmap(args.data_dir / "train.bin", dtype=np.uint16, mode="r")
    block_count = (len(train_tokens) - 1) // model_cfg.block_size
    if block_count < 1:
        raise ValueError(
            "Training corpus is too short: need at least "
            f"{model_cfg.block_size + 1} tokens, found {len(train_tokens)}."
        )
    microbatches_per_epoch = math.ceil(block_count / train_cfg.batch_size)
    steps_per_epoch = math.ceil(
        microbatches_per_epoch / train_cfg.gradient_accumulation_steps
    )
    max_steps = train_cfg.epochs * steps_per_epoch
    metrics_path = run_dir / "metrics.jsonl"
    global_step = 0
    best_bpb = math.inf
    best_step = 0
    final_eval = None
    started = time.time()
    generator = torch.Generator().manual_seed(train_cfg.seed)

    print(f"[Run] {run_name} [Device] {device} [Params] {parameter_count:,}")
    print(f"[Corpus] tokenizer={metadata['tokenizer']} train_tokens={len(train_tokens):,}")
    with wandb_run(
        run_name,
        group="nanogpt-tokenizer-comparison",
        tags=["nanogpt", metadata["tokenizer"], "bpb"],
        config=payload,
    ) as tracker:
        for epoch in range(1, train_cfg.epochs + 1):
            permutation = torch.randperm(block_count, generator=generator).tolist()
            optimizer.zero_grad(set_to_none=True)
            accumulation = 0
            accumulated_loss = 0.0
            accumulated_tokens = 0
            for batch_start in range(0, block_count, train_cfg.batch_size):
                block_ids = permutation[batch_start : batch_start + train_cfg.batch_size]
                inputs = torch.stack([
                    torch.from_numpy(
                        np.asarray(
                            train_tokens[
                                block_id * model_cfg.block_size :
                                block_id * model_cfg.block_size + model_cfg.block_size
                            ],
                            dtype=np.int64,
                        )
                    )
                    for block_id in block_ids
                ]).to(device)
                targets = torch.stack([
                    torch.from_numpy(
                        np.asarray(
                            train_tokens[
                                block_id * model_cfg.block_size + 1 :
                                block_id * model_cfg.block_size + model_cfg.block_size + 1
                            ],
                            dtype=np.int64,
                        )
                    )
                    for block_id in block_ids
                ]).to(device)
                _, loss = model(inputs, targets)
                assert loss is not None
                (loss / train_cfg.gradient_accumulation_steps).backward()
                accumulation += 1
                accumulated_loss += float(loss)
                accumulated_tokens += targets.numel()
                epoch_finished = batch_start + train_cfg.batch_size >= block_count
                if accumulation < train_cfg.gradient_accumulation_steps and not epoch_finished:
                    continue

                global_step += 1
                learning_rate = _lr(global_step - 1, max_steps, train_cfg)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_cfg.grad_clip,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_row = {
                    "split": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "token_nll": accumulated_loss / max(accumulation, 1),
                    "tokens": accumulated_tokens,
                    "learning_rate": learning_rate,
                    "grad_norm": float(grad_norm),
                    "elapsed_sec": time.time() - started,
                }
                append_jsonl(train_row, metrics_path)
                tracker.log({f"train/{k}": v for k, v in train_row.items() if isinstance(v, (int, float))}, step=global_step)
                accumulation = 0
                accumulated_loss = 0.0
                accumulated_tokens = 0

                if global_step % train_cfg.eval_every == 0:
                    final_eval = evaluate_bpb(
                        model,
                        args.data_dir,
                        metadata,
                        train_cfg,
                        device,
                    )
                    append_jsonl({"split": "validation", "step": global_step, **final_eval}, metrics_path)
                    tracker.log({f"validation/{k}": v for k, v in final_eval.items()}, step=global_step)
                    if final_eval["bits_per_raw_byte"] < best_bpb:
                        best_bpb = float(final_eval["bits_per_raw_byte"])
                        best_step = global_step
                        torch.save(
                            {"model": model.state_dict(), "model_config": asdict(model_cfg), "step": global_step},
                            run_dir / "checkpoints" / "best.pt",
                        )
                    print(f"[Validation] step={global_step} BPB={final_eval['bits_per_raw_byte']:.5f}")

        final_eval = evaluate_bpb(
            model,
            args.data_dir,
            metadata,
            train_cfg,
            device,
        )
        if final_eval["bits_per_raw_byte"] < best_bpb:
            best_bpb = float(final_eval["bits_per_raw_byte"])
            best_step = global_step
            torch.save(
                {"model": model.state_dict(), "model_config": asdict(model_cfg), "step": global_step},
                run_dir / "checkpoints" / "best.pt",
            )
        torch.save(
            {"model": model.state_dict(), "model_config": asdict(model_cfg), "step": global_step},
            run_dir / "checkpoints" / "last.pt",
        )

    atomic_json_dump(
        {
            "run_name": run_name,
            "status": "completed",
            "steps": global_step,
            "parameter_count": parameter_count,
            "best_validation_bpb": best_bpb,
            "best_step": best_step,
            "final_validation": final_eval,
            "train_raw_utf8_bytes": metadata["train"]["raw_utf8_bytes"],
            "train_tokens": metadata["train"]["tokens"],
            "elapsed_sec": time.time() - started,
        },
        run_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
