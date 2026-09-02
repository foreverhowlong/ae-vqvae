"""Train one fixed-backbone Engram Phase-0 variant on a frozen token stream."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from common import ROOT, enable_tf32, get_device
from common.tracking import wandb_run
from models.engram_phase0 import (
    EngramPhase0LM,
    Phase0BackboneConfig,
    Phase0EngramConfig,
    analytical_backbone_parameter_count,
    analytical_engram_counts,
    consecutive_unique_primes,
)
from training.text_vqvae.reporting import append_jsonl, atomic_json_dump


VARIANTS = ("baseline", "engram_s", "engram_m", "engram_l")
EXPECTED_TABLE_TARGETS = {
    "baseline": 0,
    "engram_s": 32_768,
    "engram_m": 131_072,
    "engram_l": 524_288,
}
EXPECTED_FINEWEB_REVISION = "05c1931294b0d1379055d1f802d369f2c3bb2f4b"
EXPECTED_GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--profile", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "engram_phase0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_config(raw: dict[str, Any], variant: str, profile: str) -> dict[str, Any]:
    if variant not in raw["variants"]:
        raise ValueError(f"Config has no variant {variant!r}.")
    if profile not in raw["profiles"]:
        raise ValueError(f"Config has no profile {profile!r}.")
    backbone = Phase0BackboneConfig(**raw["backbone"])
    variant_config = raw["variants"][variant]
    engram = Phase0EngramConfig(**variant_config)
    backbone.validate()
    engram.validate(backbone)
    if backbone != Phase0BackboneConfig():
        raise ValueError("Phase-0 backbone architecture is fixed and may not be changed.")
    if engram.table_rows_target != EXPECTED_TABLE_TARGETS[variant]:
        raise ValueError(f"Phase-0 table target is fixed for {variant}.")
    if engram.enabled != (variant != "baseline"):
        raise ValueError(f"Unexpected Engram enablement for {variant}.")
    optimization = raw["optimization"]
    for key, expected in {
        "peak_lr": 6e-4,
        "weight_decay": 0.1,
        "gradient_clipping": 1.0,
        "engram_lr_multiplier": 5.0,
    }.items():
        if float(optimization[key]) != expected:
            raise ValueError(f"Phase-0 fixes {key}={expected}.")
    selected_profile = raw["profiles"][profile]
    if profile == "final":
        if int(selected_profile["training_tokens"]) != 2_500_000_000:
            raise ValueError("Final Phase-0 training budget must be exactly 2.5B tokens.")
        if int(selected_profile["validation_tokens"]) < 10_000_000:
            raise ValueError("Final Phase-0 validation must score at least 10M tokens.")
        if not 20_000_000 <= int(selected_profile["eval_every_tokens"]) <= 50_000_000:
            raise ValueError("Final validation cadence must be between 20M and 50M tokens.")
    return {
        "format_version": int(raw["format_version"]),
        "experiment_name": str(raw["experiment_name"]),
        "variant": variant,
        "profile_name": profile,
        "backbone": asdict(backbone),
        "engram": asdict(engram),
        "optimization": dict(optimization),
        "profile": dict(selected_profile),
        "go_criteria": dict(raw["go_criteria"]),
        "sources": dict(raw.get("sources", {})),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_data(data_dir: Path, resolved: dict[str, Any]) -> dict[str, Any]:
    metadata = _load_json(data_dir / "meta.json")
    required = {
        "train.bin": metadata["train"]["sha256"],
        "validation.bin": metadata["validation"]["sha256"],
        "canonical_projection.npy": metadata["tokenizer"][
            "canonical_projection_sha256"
        ],
    }
    for filename, expected in required.items():
        actual = _sha256(data_dir / filename)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {filename}: {actual} != {expected}")
    if int(metadata["tokenizer"]["vocab_size"]) != resolved["backbone"]["vocab_size"]:
        raise ValueError("Corpus tokenizer vocabulary does not match the backbone.")
    if resolved["profile_name"] == "final":
        tokenizer_identity = {
            "name": metadata["tokenizer"].get("name"),
            "revision": metadata["tokenizer"].get("revision"),
        }
        if tokenizer_identity != {
            "name": "openai-community/gpt2",
            "revision": EXPECTED_GPT2_REVISION,
        }:
            raise ValueError(f"Final Phase-0 requires pinned GPT-2: {tokenizer_identity}")
    if int(metadata["train"]["prediction_tokens"]) < int(
        resolved["profile"]["training_tokens"]
    ):
        raise ValueError("Prepared training stream is shorter than the selected token budget.")
    if int(metadata["validation"]["prediction_tokens"]) < int(
        resolved["profile"]["validation_tokens"]
    ):
        raise ValueError("Prepared validation stream is shorter than the selected eval budget.")
    if resolved["profile_name"] == "final":
        source = metadata.get("source", {})
        actual_source = {
            "type": source.get("type"),
            "dataset": source.get("dataset"),
            "config": source.get("config"),
            "revision": source.get("revision"),
        }
        expected_source = {
            "type": "huggingface",
            "dataset": "HuggingFaceFW/fineweb-edu",
            "config": "sample-10BT",
            "revision": EXPECTED_FINEWEB_REVISION,
        }
        if actual_source != expected_source:
            raise ValueError(
                f"Final Phase-0 requires the pinned FineWeb-Edu source: {actual_source}"
            )
    return metadata


def make_batch(
    stream: np.memmap,
    start: int,
    prediction_tokens: int,
    batch_size: int,
    context_length: int,
    eot_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    capacity = batch_size * context_length
    valid = min(capacity, prediction_tokens)
    inputs = np.full((batch_size, context_length), eot_id, dtype=np.int64)
    targets = np.full((batch_size, context_length), -1, dtype=np.int64)
    remaining = valid
    for row in range(batch_size):
        count = min(context_length, remaining)
        if count <= 0:
            break
        row_start = start + row * context_length
        inputs[row, :count] = np.asarray(stream[row_start : row_start + count], dtype=np.int64)
        targets[row, :count] = np.asarray(
            stream[row_start + 1 : row_start + count + 1], dtype=np.int64
        )
        remaining -= count
    return (
        torch.from_numpy(inputs).to(device, non_blocking=True),
        torch.from_numpy(targets).to(device, non_blocking=True),
        valid,
    )


def learning_rate_at(tokens_seen: int, total_tokens: int, settings: dict[str, Any]) -> float:
    peak = float(settings["peak_lr"])
    minimum = float(settings["min_lr"])
    warmup = int(settings["warmup_tokens"])
    if tokens_seen <= warmup:
        return peak * tokens_seen / max(warmup, 1)
    progress = (tokens_seen - warmup) / max(total_tokens - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def global_clip_grad_norm(parameters, max_norm: float) -> float:
    """Clip a mixture of dense and sparse gradients to one global L2 norm."""
    gradients: list[torch.Tensor] = []
    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.coalesce() if parameter.grad.is_sparse else parameter.grad
        if parameter.grad.is_sparse:
            parameter.grad = gradient
            values = gradient.values()
        else:
            values = gradient
        gradients.append(values)
        contribution = values.detach().float().pow(2).sum()
        squared = contribution if squared is None else squared + contribution
    if squared is None:
        return 0.0
    norm = float(torch.sqrt(squared))
    coefficient = min(1.0, max_norm / (norm + 1e-6))
    if coefficient < 1.0:
        for gradient in gradients:
            gradient.mul_(coefficient)
    return norm


def build_optimizers(
    model: EngramPhase0LM, settings: dict[str, Any], device: torch.device
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer | None]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    sparse_id = id(model.engram.embedding.weight) if model.engram is not None else None
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) == sparse_id:
            continue
        if parameter.ndim < 2 or name.endswith("bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    dense = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(settings["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(settings["peak_lr"]),
        betas=(float(settings["beta1"]), float(settings["beta2"])),
        fused=device.type == "cuda",
    )
    sparse = None
    if model.engram is not None:
        table = model.engram.embedding.weight
        multiplier = float(settings["engram_lr_multiplier"])
        if model.engram_config.sparse_gradients:
            sparse = torch.optim.SparseAdam(
                [table],
                lr=float(settings["peak_lr"]) * multiplier,
                betas=(float(settings["beta1"]), float(settings["beta2"])),
            )
        else:
            sparse = torch.optim.AdamW(
                [{"params": [table], "weight_decay": 0.0}],
                lr=float(settings["peak_lr"]) * multiplier,
                betas=(float(settings["beta1"]), float(settings["beta2"])),
                fused=device.type == "cuda",
            )
    return dense, sparse


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate(
    model: EngramPhase0LM,
    stream: np.memmap,
    prediction_tokens: int,
    batch_size: int,
    device: torch.device,
    eot_id: int,
) -> dict[str, float | int | None]:
    was_training = model.training
    model.eval()
    cursor = 0
    total_loss = 0.0
    gate_weight = 0
    gate_sum = 0.0
    gate_square_sum = 0.0
    while cursor < prediction_tokens:
        inputs, targets, valid = make_batch(
            stream,
            cursor,
            prediction_tokens - cursor,
            batch_size,
            model.backbone_config.context_length,
            eot_id,
            device,
        )
        with _autocast(device):
            _, loss, auxiliary = model(inputs, targets, reduction="sum")
        assert loss is not None
        total_loss += float(loss)
        if auxiliary:
            mean = float(auxiliary["engram_gate_mean"])
            std = float(auxiliary["engram_gate_std"])
            gate_sum += mean * valid
            gate_square_sum += (std * std + mean * mean) * valid
            gate_weight += valid
        cursor += valid
    model.train(was_training)
    nll = total_loss / prediction_tokens
    gate_mean = gate_sum / gate_weight if gate_weight else None
    gate_variance = gate_square_sum / gate_weight - gate_mean**2 if gate_weight else None
    return {
        "validation_nll": nll,
        "perplexity": math.exp(min(nll, 20.0)),
        "validation_tokens": prediction_tokens,
        "engram_gate_mean": gate_mean,
        "engram_gate_std": math.sqrt(max(gate_variance, 0.0)) if gate_variance is not None else None,
    }


def _format_bytes(value: int) -> str:
    return f"{value / (1024 ** 3):.3f} GiB"


def _print_counts(backbone: Phase0BackboneConfig, engram: Phase0EngramConfig) -> dict[str, int]:
    backbone_count = analytical_backbone_parameter_count(backbone)
    engram_counts = analytical_engram_counts(backbone, engram)
    counts = {"backbone": backbone_count, **engram_counts}
    counts["total"] = backbone_count + counts["dense_engram"] + counts["sparse_tables"]
    print(f"[Params] backbone={counts['backbone']:,}")
    print(f"[Params] dense_engram={counts['dense_engram']:,}")
    print(f"[Params] sparse_tables={counts['sparse_tables']:,}")
    print(f"[Params] total={counts['total']:,}")
    print(
        "[Storage] table_theoretical_bf16="
        f"{counts['table_theoretical_bf16_bytes']:,} bytes "
        f"({_format_bytes(counts['table_theoretical_bf16_bytes'])})"
    )
    if engram.enabled:
        primes = consecutive_unique_primes(engram.table_rows_target, engram.route_count)
        print(f"[Tables] target_M={engram.table_rows_target:,} primes={list(primes)}")
    return counts


def save_checkpoint(
    path: Path,
    model: EngramPhase0LM,
    dense_optimizer: torch.optim.Optimizer,
    sparse_optimizer: torch.optim.Optimizer | None,
    tokens_seen: int,
    step: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "dense_optimizer": dense_optimizer.state_dict(),
            "sparse_optimizer": sparse_optimizer.state_dict() if sparse_optimizer else None,
            "tokens_seen": tokens_seen,
            "step": step,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    raw = _load_json(args.config)
    resolved = resolve_config(raw, args.variant, args.profile)
    data_dir = args.data_dir.expanduser().resolve()
    metadata = validate_data(data_dir, resolved)
    backbone = Phase0BackboneConfig(**resolved["backbone"])
    engram = Phase0EngramConfig(**resolved["engram"])
    counts = _print_counts(backbone, engram)
    resolved["parameter_counts"] = counts
    resolved["corpus"] = metadata
    if counts["backbone"] != int(raw["expected_backbone_parameters"]):
        raise ValueError("Analytical backbone parameter count violates the experiment contract.")
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return

    run_dir = (
        args.output_root.expanduser().resolve()
        / resolved["experiment_name"]
        / args.profile
        / args.variant
    )
    checkpoint_dir = run_dir / "checkpoints"
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(resolved, run_dir / "config.json")

    device = get_device()
    if device.type != "cuda":
        raise RuntimeError("The 125M Phase-0 training path requires a CUDA device.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support bfloat16 training.")
    enable_tf32(device)
    canonical_projection = torch.from_numpy(
        np.load(data_dir / "canonical_projection.npy")
    )
    model = EngramPhase0LM(
        backbone,
        engram,
        canonical_projection if engram.enabled else None,
        backbone_init_seed=int(resolved["optimization"]["initialization_seed"]),
    ).to(device)
    actual_counts = model.parameter_counts()
    if actual_counts != counts:
        raise RuntimeError(f"Analytical/actual parameter mismatch: {counts} != {actual_counts}")
    dense_optimizer, sparse_optimizer = build_optimizers(
        model, resolved["optimization"], device
    )

    train_stream = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    validation_stream = np.memmap(
        data_dir / "validation.bin", dtype=np.uint16, mode="r"
    )
    total_tokens = int(resolved["profile"]["training_tokens"])
    validation_tokens = int(resolved["profile"]["validation_tokens"])
    micro_batch_size = int(resolved["profile"]["micro_batch_size"])
    accumulation_steps = int(resolved["profile"]["gradient_accumulation_steps"])
    eval_every_tokens = int(resolved["profile"]["eval_every_tokens"])
    log_every_steps = int(resolved["profile"]["log_every_steps"])
    eot_id = int(metadata["tokenizer"]["eot_token_id"])
    settings = {**resolved["optimization"], **resolved["profile"]}
    metrics_path = run_dir / "metrics.jsonl"
    tokens_seen = 0
    step = 0
    best_validation = math.inf
    best_tokens = 0

    last_checkpoint = checkpoint_dir / "last.pt"
    if args.resume:
        state = torch.load(last_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        dense_optimizer.load_state_dict(state["dense_optimizer"])
        if sparse_optimizer is not None and state["sparse_optimizer"] is not None:
            sparse_optimizer.load_state_dict(state["sparse_optimizer"])
        tokens_seen = int(state["tokens_seen"])
        step = int(state["step"])
    next_eval = ((tokens_seen // eval_every_tokens) + 1) * eval_every_tokens
    started = time.perf_counter()
    interval_started = started
    interval_tokens = 0
    peak_gpu_memory = 0
    evaluations_completed = 0

    print(
        f"[Run] {resolved['experiment_name']}/{args.profile}/{args.variant} "
        f"device={device} precision=bf16 tokens={total_tokens:,}"
    )
    with wandb_run(
        f"{resolved['experiment_name']}-{args.profile}-{args.variant}",
        group=f"{resolved['experiment_name']}-{args.profile}",
        tags=["engram", "phase0", args.profile, args.variant],
        config=resolved,
    ) as tracker:
        while tokens_seen < total_tokens:
            step += 1
            step_budget = min(
                micro_batch_size
                * backbone.context_length
                * accumulation_steps,
                total_tokens - tokens_seen,
            )
            dense_optimizer.zero_grad(set_to_none=True)
            if sparse_optimizer is not None:
                sparse_optimizer.zero_grad(set_to_none=True)
            step_consumed = 0
            step_loss_sum = 0.0
            gate_means: list[torch.Tensor] = []
            gate_stds: list[torch.Tensor] = []
            while step_consumed < step_budget:
                inputs, targets, valid = make_batch(
                    train_stream,
                    tokens_seen + step_consumed,
                    step_budget - step_consumed,
                    micro_batch_size,
                    backbone.context_length,
                    eot_id,
                    device,
                )
                with _autocast(device):
                    _, loss_sum, auxiliary = model(inputs, targets, reduction="sum")
                assert loss_sum is not None
                (loss_sum / step_budget).backward()
                step_loss_sum += float(loss_sum.detach())
                if auxiliary:
                    gate_means.append(auxiliary["engram_gate_mean"])
                    gate_stds.append(auxiliary["engram_gate_std"])
                step_consumed += valid

            new_tokens_seen = tokens_seen + step_budget
            lr = learning_rate_at(new_tokens_seen, total_tokens, settings)
            for group in dense_optimizer.param_groups:
                group["lr"] = lr
            if sparse_optimizer is not None:
                for group in sparse_optimizer.param_groups:
                    group["lr"] = lr * float(settings["engram_lr_multiplier"])
            grad_norm = global_clip_grad_norm(
                model.parameters(), float(settings["gradient_clipping"])
            )
            dense_optimizer.step()
            if sparse_optimizer is not None:
                sparse_optimizer.step()
            tokens_seen = new_tokens_seen
            interval_tokens += step_budget
            peak_gpu_memory = max(peak_gpu_memory, torch.cuda.max_memory_allocated(device))

            if step % log_every_steps == 0 or tokens_seen == total_tokens:
                now = time.perf_counter()
                row = {
                    "split": "train",
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "train_nll": step_loss_sum / step_budget,
                    "learning_rate": lr,
                    "engram_learning_rate": lr
                    * float(settings["engram_lr_multiplier"])
                    if engram.enabled
                    else None,
                    "tokens_per_sec": interval_tokens / max(now - interval_started, 1e-9),
                    "peak_gpu_memory_bytes": peak_gpu_memory,
                    "grad_norm": grad_norm,
                    "engram_gate_mean": float(torch.stack(gate_means).mean())
                    if gate_means
                    else None,
                    "engram_gate_std": float(torch.stack(gate_stds).mean())
                    if gate_stds
                    else None,
                    "engram_table_parameter_count": counts["sparse_tables"],
                }
                append_jsonl(row, metrics_path)
                tracker.log(
                    {f"train/{key}": value for key, value in row.items() if isinstance(value, (int, float))},
                    step=step,
                )
                print(
                    f"[Train] step={step:,} tokens={tokens_seen:,} "
                    f"nll={row['train_nll']:.5f} tok/s={row['tokens_per_sec']:.0f}"
                )
                interval_started = now
                interval_tokens = 0

            if tokens_seen >= next_eval or tokens_seen == total_tokens:
                evaluation_started = time.perf_counter()
                evaluation = evaluate(
                    model,
                    validation_stream,
                    validation_tokens,
                    int(resolved["profile"]["eval_batch_size"]),
                    device,
                    eot_id,
                )
                row = {
                    "split": "validation",
                    "step": step,
                    "tokens_seen": tokens_seen,
                    **evaluation,
                    "peak_gpu_memory_bytes": peak_gpu_memory,
                    "engram_table_parameter_count": counts["sparse_tables"],
                }
                append_jsonl(row, metrics_path)
                tracker.log(
                    {f"validation/{key}": value for key, value in row.items() if isinstance(value, (int, float))},
                    step=step,
                )
                print(
                    f"[Validation] tokens={tokens_seen:,} "
                    f"nll={evaluation['validation_nll']:.5f} ppl={evaluation['perplexity']:.3f}"
                )
                evaluations_completed += 1
                if float(evaluation["validation_nll"]) < best_validation:
                    best_validation = float(evaluation["validation_nll"])
                    best_tokens = tokens_seen
                checkpoint_every = int(
                    resolved["profile"].get("checkpoint_every_evals", 4)
                )
                if evaluations_completed % checkpoint_every == 0 or tokens_seen == total_tokens:
                    save_checkpoint(
                        last_checkpoint,
                        model,
                        dense_optimizer,
                        sparse_optimizer,
                        tokens_seen,
                        step,
                    )
                while next_eval <= tokens_seen:
                    next_eval += eval_every_tokens
                # Do not charge fixed validation time to the training throughput interval.
                interval_started += time.perf_counter() - evaluation_started

    summary = {
        "status": "completed",
        "experiment_name": resolved["experiment_name"],
        "profile": args.profile,
        "variant": args.variant,
        "tokens_seen": tokens_seen,
        "steps": step,
        "best_validation_nll": best_validation,
        "best_tokens": best_tokens,
        "parameter_counts": counts,
        "elapsed_seconds": time.perf_counter() - started,
        "data_sha256": {
            "train": metadata["train"]["sha256"],
            "validation": metadata["validation"]["sha256"],
        },
    }
    atomic_json_dump(summary, run_dir / "summary.json")


if __name__ == "__main__":
    main()
