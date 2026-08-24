"""Train the segmental BPE VQ-VAE with adaptive AE warmup."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from common import ROOT, enable_tf32, get_device
from common.segmental_vqvae_config import (
    LATENT_ROUTING_MODES,
    SEGMENTATION_MODES,
    SegmentalVQVAEConfig,
    SegmentalVQVAEDataConfig,
    SegmentalVQVAETrainConfig,
)
from common.text_data import BPETokenizer, build_text_dataset
from common.tracking import wandb_run
from models.segmental_vqvae import (
    SegmentalVQVAE,
    count_parameters,
    segmental_vqvae_losses,
)
from models.text_vqvae import codebook_stats
from training.text_vqvae.codebook_init import initialize_codebook_kmeans
from training.text_vqvae.geometry import preserve_rng_state
from training.text_vqvae.loop import batch_to_device, make_loader, split_dataset
from training.text_vqvae.reporting import append_jsonl, atomic_json_dump
from training.text_vqvae.warmup import (
    AdaptiveWarmupController,
    evaluate_adaptive_warmup,
)
from training.segmental_vqvae_reporting import (
    build_segmentation_snapshot,
    finalize_checkpoints,
    plot_segmental_metrics,
    rolling_checkpoint_due,
    save_model_checkpoint,
    save_resume_checkpoint,
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
        ("lr", float),
        ("weight-decay", float),
        ("grad-clip", float),
        ("eval-every", int),
        ("save-every", int),
        ("num-workers", int),
        ("ae-warmup-min-steps", int),
        ("ae-warmup-max-steps", int),
        ("ae-warmup-check-every", int),
        ("ae-warmup-patience", int),
        ("ae-warmup-dim-tolerance", int),
        ("ae-warmup-probe-points", int),
        ("ae-warmup-variance-threshold", float),
        ("intervention-probe-examples", int),
        ("free-running-every", int),
        ("free-running-samples", int),
        ("max-train-samples", int),
        ("max-eval-samples", int),
        ("val-fraction", float),
        ("max-seq-len", int),
        ("d-model", int),
        ("latent-dim", int),
        ("n-heads", int),
        ("encoder-layers", int),
        ("decoder-layers", int),
        ("boundary-encoder-layers", int),
        ("boundary-window-radius", int),
        ("max-span-length", int),
        ("span-encoder-layers", int),
        ("ffn-mult", int),
        ("dropout", float),
        ("codebook-size", int),
        ("commitment-beta", float),
        ("compression-target", float),
        ("compression-weight", float),
        ("gate-logit-l2-weight", float),
        ("gate-threshold", float),
        ("decoder-boundary-loss-weight", float),
        ("decoder-boundary-threshold", float),
        ("ema-decay", float),
        ("ema-eps", float),
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
    parser.add_argument(
        "--latent-routing",
        choices=LATENT_ROUTING_MODES,
    )
    parser.add_argument(
        "--segmentation-mode",
        choices=SEGMENTATION_MODES,
    )
    parser.add_argument(
        "--continuous-truncation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--save-last-resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Retain a full model+optimizer last.pt after successful training.",
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
    SegmentalVQVAETrainConfig,
    SegmentalVQVAEDataConfig,
    SegmentalVQVAEConfig,
]:
    train = SegmentalVQVAETrainConfig()
    data = SegmentalVQVAEDataConfig()
    model = SegmentalVQVAEConfig()
    _override(train, args)
    _override(data, args)
    _override(model, args)
    if tokenizer.bos_token_id is None:
        raise ValueError("Segmental AR decoding requires a BPE <bos> token.")
    model.vocab_size = tokenizer.vocab_size
    model.pad_token_id = tokenizer.pad_token_id
    model.bos_token_id = tokenizer.bos_token_id
    model.eos_token_id = tokenizer.eos_token_id
    model.validate()
    if not 0 <= train.ae_warmup_min_steps < train.ae_warmup_max_steps:
        raise ValueError("Adaptive warmup requires 0 <= min_steps < max_steps.")
    if train.ae_warmup_check_every < 1 or train.ae_warmup_patience < 1:
        raise ValueError("Adaptive warmup cadence and patience must be positive.")
    if train.intervention_probe_examples < 1:
        raise ValueError("intervention_probe_examples must be positive.")
    if train.free_running_every < 1 or train.free_running_samples < 1:
        raise ValueError("Free-running cadence and sample count must be positive.")
    if train.save_every < 0:
        raise ValueError("save_every must be non-negative; zero disables rolling saves.")
    if train.batch_size < 1 or train.epochs < 1:
        raise ValueError("Training batch size and epochs must be positive.")
    return train, data, model


def _make_run_dir(run_name: str) -> Path:
    path = ROOT / "outputs" / "segmental_vqvae" / run_name
    if path.exists():
        raise FileExistsError(f"Run directory already exists: {path}")
    (path / "checkpoints").mkdir(parents=True)
    return path


def _materialize_warmup_probe(loader, max_points: int):
    batches = []
    points = 0
    for batch in loader:
        cpu_batch = {key: value.detach().cpu() for key, value in batch.items()}
        batches.append(cpu_batch)
        points += int(cpu_batch["attention_mask"].sum())
        if points >= max_points:
            break
    if not batches:
        raise ValueError("Adaptive warmup probe is empty.")
    return batches


def _materialize_fixed_batch(loader, max_examples: int) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = {}
    examples = 0
    for batch in loader:
        take = min(max_examples - examples, int(batch["input_ids"].shape[0]))
        for key, value in batch.items():
            chunks.setdefault(key, []).append(value[:take].detach().cpu())
        examples += take
        if examples >= max_examples:
            break
    if examples == 0:
        raise ValueError("Fixed evaluation probe is empty.")
    return {key: torch.cat(values) for key, values in chunks.items()}


def _masked_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(logits[valid_mask], targets[valid_mask])


def _boundary_candidate_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    next_valid = F.pad(valid_mask[:, 1:], (0, 1), value=False)
    return valid_mask & next_valid


def _boundary_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, int]:
    candidate_mask = _boundary_candidate_mask(valid_mask)
    predictions = torch.sigmoid(logits) > threshold
    target_boundaries = targets.bool()
    return {
        "decoder_boundary_tp": int(
            (predictions & target_boundaries & candidate_mask).sum()
        ),
        "decoder_boundary_fp": int(
            (predictions & ~target_boundaries & candidate_mask).sum()
        ),
        "decoder_boundary_fn": int(
            (~predictions & target_boundaries & candidate_mask).sum()
        ),
        "decoder_boundary_tn": int(
            (~predictions & ~target_boundaries & candidate_mask).sum()
        ),
    }


def _boundary_metrics_from_counts(counts: dict[str, float]) -> dict[str, float]:
    true_positive = counts.get("decoder_boundary_tp", 0.0)
    false_positive = counts.get("decoder_boundary_fp", 0.0)
    false_negative = counts.get("decoder_boundary_fn", 0.0)
    true_negative = counts.get("decoder_boundary_tn", 0.0)
    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    return {
        "decoder_boundary_accuracy": (
            (true_positive + true_negative)
            / max(
                true_positive + false_positive + false_negative + true_negative,
                1.0,
            )
        ),
        "decoder_boundary_precision": precision,
        "decoder_boundary_recall": recall,
        "decoder_boundary_f1": (
            2.0 * precision * recall / max(precision + recall, 1e-12)
        ),
    }


@torch.no_grad()
def evaluate(
    model: SegmentalVQVAE,
    loader,
    device: torch.device,
    *,
    use_quantizer: bool,
) -> dict[str, float | int]:
    was_training = model.training
    totals: dict[str, float] = {}
    batches = 0
    total_tokens = 0
    total_chunks = 0
    all_indices = []
    all_masks = []
    try:
        model.eval()
        for batch in loader:
            batch = batch_to_device(batch, device)
            outputs = model(
                batch["input_ids"],
                batch["attention_mask"],
                use_quantizer=use_quantizer,
                sample_gates=False,
            )
            losses = segmental_vqvae_losses(
                outputs,
                batch["input_ids"],
                batch["attention_mask"],
                model.config,
            )
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            valid = batch["attention_mask"].bool()
            totals["reconstruction_nll_sum"] = totals.get(
                "reconstruction_nll_sum", 0.0
            ) + float(F.cross_entropy(
                outputs["logits"][valid],
                batch["input_ids"][valid],
                reduction="sum",
            ))
            tokens = int(valid.sum())
            chunks = int(outputs["chunk_counts"].sum())
            total_tokens += tokens
            total_chunks += chunks
            predictions = outputs["logits"].argmax(dim=-1)
            totals["correct_tokens"] = totals.get("correct_tokens", 0.0) + int(
                ((predictions == batch["input_ids"]) & valid).sum()
            )
            totals["soft_ratio"] = totals.get("soft_ratio", 0.0) + float(
                outputs["soft_tokens_per_chunk"].mean()
            )
            totals["hard_ratio"] = totals.get("hard_ratio", 0.0) + float(
                outputs["hard_tokens_per_chunk"].mean()
            )
            boundary_logits = outputs.get("decoder_boundary_logits")
            if isinstance(boundary_logits, torch.Tensor):
                boundary_counts = _boundary_counts(
                    boundary_logits,
                    outputs["hard_boundaries"],
                    valid,
                    threshold=model.config.decoder_boundary_threshold,
                )
                for key, value in boundary_counts.items():
                    totals[key] = totals.get(key, 0.0) + value
            if use_quantizer:
                all_indices.append(outputs["indices"].detach().cpu())
                all_masks.append(outputs["latent_mask"].detach().cpu())
            batches += 1
    finally:
        model.train(was_training)
    metrics: dict[str, float | int] = {
        key: value / max(batches, 1)
        for key, value in totals.items()
        if key not in {
            "correct_tokens",
            "soft_ratio",
            "hard_ratio",
            "reconstruction_nll_sum",
            "decoder_boundary_tp",
            "decoder_boundary_fp",
            "decoder_boundary_fn",
            "decoder_boundary_tn",
        }
    }
    metrics.update({
        "token_accuracy": totals.get("correct_tokens", 0.0) / max(total_tokens, 1),
        "tokens": total_tokens,
        "chunks": total_chunks,
        "tokens_per_chunk_hard": total_tokens / max(total_chunks, 1),
        "tokens_per_chunk_soft_batch_mean": totals.get("soft_ratio", 0.0)
        / max(batches, 1),
        "rate_bits_per_bpe": (
            total_chunks * math.log2(model.config.codebook_size)
            / max(total_tokens, 1)
        ),
    })
    metrics["distortion_nats_per_bpe"] = (
        totals.get("reconstruction_nll_sum", 0.0) / max(total_tokens, 1)
    )
    metrics["distortion_bits_per_bpe"] = (
        float(metrics["distortion_nats_per_bpe"]) / math.log(2.0)
    )
    if "decoder_boundary_tp" in totals:
        metrics.update(_boundary_metrics_from_counts(totals))
    if use_quantizer and all_indices:
        stats = codebook_stats(
            torch.cat(all_indices),
            model.config.codebook_size,
            valid_mask=torch.cat(all_masks),
        )
        metrics.update({
            "codebook_utilization": stats["utilization"],
            "codebook_perplexity": stats["codebook_perplexity"],
            "used_codes": stats["used_codes"],
            "dead_codes": stats["dead_codes"],
        })
    return metrics


@torch.no_grad()
def evaluate_interventions(
    model: SegmentalVQVAE,
    probe_batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    use_quantizer: bool,
    seed: int,
    tokenizer=None,
) -> tuple[dict[str, float | int], dict]:
    """Run four length-matched teacher-forced latent interventions."""
    was_training = model.training
    try:
        model.eval()
        batch = batch_to_device(probe_batch, device)
        outputs = model(
            batch["input_ids"],
            batch["attention_mask"],
            use_quantizer=use_quantizer,
            sample_gates=False,
        )
        valid = batch["attention_mask"].bool()
        latent_mask = outputs["latent_mask"]
        latents = outputs["z_latent"]
        ce_correct = _masked_ce(outputs["logits"], batch["input_ids"], valid)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        random_indices = torch.randint(
            model.config.codebook_size,
            latent_mask.shape,
            generator=generator,
        ).to(device)
        random_latents = model.quantizer.codebook(random_indices)
        random_latents = torch.where(
            latent_mask.unsqueeze(-1),
            random_latents,
            torch.zeros_like(random_latents),
        )
        ce_random = _masked_ce(
            model.decode_teacher_forced(
                random_latents,
                latent_mask,
                batch["input_ids"],
                valid,
                segment_ids=outputs["segment_ids"],
            ),
            batch["input_ids"],
            valid,
        )

        permuted_latents = latents.clone()
        for batch_index, chunk_count in enumerate(outputs["chunk_counts"].tolist()):
            order = torch.randperm(chunk_count, generator=generator).to(device)
            permuted_latents[batch_index, :chunk_count] = latents[
                batch_index, order
            ]
        ce_permuted = _masked_ce(
            model.decode_teacher_forced(
                permuted_latents,
                latent_mask,
                batch["input_ids"],
                valid,
                segment_ids=outputs["segment_ids"],
            ),
            batch["input_ids"],
            valid,
        )

        ce_null = _masked_ce(
            model.decode_teacher_forced(
                latents,
                latent_mask,
                batch["input_ids"],
                valid,
                segment_ids=outputs["segment_ids"],
                disable_cross_attention=True,
            ),
            batch["input_ids"],
            valid,
        )

        token_count = int(valid.sum())
        chunk_count = int(outputs["chunk_counts"].sum())
        rate = (
            chunk_count * math.log2(model.config.codebook_size)
            / max(token_count, 1)
        )
        inv_ln2 = 1.0 / math.log(2.0)
        delta_random = float(ce_random - ce_correct) * inv_ln2
        delta_permuted = float(ce_permuted - ce_correct) * inv_ln2
        delta_null = float(ce_null - ce_correct) * inv_ln2
        snapshot = build_segmentation_snapshot(
            batch["input_ids"],
            valid,
            outputs,
            tokenizer=tokenizer,
            decoder_boundary_threshold=model.config.decoder_boundary_threshold,
        )
        metrics = {
            "examples": int(batch["input_ids"].shape[0]),
            "tokens": token_count,
            "chunks": chunk_count,
            "ce_0_nats_per_bpe": float(ce_correct),
            "ce_rand_nats_per_bpe": float(ce_random),
            "ce_perm_nats_per_bpe": float(ce_permuted),
            "ce_null_nats_per_bpe": float(ce_null),
            "delta_rand_bits_per_bpe": delta_random,
            "delta_perm_bits_per_bpe": delta_permuted,
            "delta_null_bits_per_bpe": delta_null,
            "rate_bits_per_bpe": rate,
            "channel_utilization_ratio": delta_null / max(rate, 1e-12),
            "gate_probability_mean": snapshot["gate_probability_mean"],
            "gate_probability_std": snapshot["gate_probability_std"],
            "boundary_fraction": snapshot["boundary_fraction"],
            "boundary_position_dependence_hard": snapshot[
                "boundary_position_dependence_hard"
            ],
            "boundary_position_dependence_soft": snapshot[
                "boundary_position_dependence_soft"
            ],
            "singleton_chunk_fraction": snapshot["singleton_chunk_fraction"],
            "excess_singleton_fraction": snapshot["excess_singleton_fraction"],
            "chunk_length_p50": snapshot["chunk_length_p50"],
            "chunk_length_p90": snapshot["chunk_length_p90"],
        }
        if model.config.segmentation_mode == "token_pruning":
            metrics.update({
                "keep_fraction": snapshot["keep_fraction"],
                "keep_position_dependence_hard": snapshot[
                    "keep_position_dependence_hard"
                ],
                "keep_position_dependence_soft": snapshot[
                    "keep_position_dependence_soft"
                ],
                "early_keep_rate_hard": snapshot["early_keep_rate_hard"],
                "late_keep_rate_hard": snapshot["late_keep_rate_hard"],
                "early_keep_rate_soft": snapshot["early_keep_rate_soft"],
                "late_keep_rate_soft": snapshot["late_keep_rate_soft"],
                "consecutive_keep_fraction": snapshot[
                    "consecutive_keep_fraction"
                ],
                "longest_drop_run": snapshot["longest_drop_run"],
                "keep_gap_p50": snapshot["keep_gap_p50"],
                "keep_gap_p90": snapshot["keep_gap_p90"],
            })
        boundary_logits = outputs.get("decoder_boundary_logits")
        if isinstance(boundary_logits, torch.Tensor):
            metrics.update({
                "decoder_boundary_position_dependence_hard": snapshot[
                    "decoder_boundary_position_dependence_hard"
                ],
                "decoder_boundary_position_dependence_soft": snapshot[
                    "decoder_boundary_position_dependence_soft"
                ],
            })
            metrics.update(_boundary_metrics_from_counts(_boundary_counts(
                boundary_logits,
                outputs["hard_boundaries"],
                valid,
                threshold=model.config.decoder_boundary_threshold,
            )))
        return metrics, snapshot
    finally:
        model.train(was_training)


def _trim_at_eos(tokens: list[int], eos_token_id: int) -> list[int]:
    try:
        return tokens[: tokens.index(eos_token_id) + 1]
    except ValueError:
        return tokens


def _edit_distance(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_token != right_token),
            ))
        previous = current
    return previous[-1]


@torch.no_grad()
def evaluate_free_running(
    model: SegmentalVQVAE,
    probe_batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    use_quantizer: bool,
) -> dict[str, float | int]:
    was_training = model.training
    try:
        model.eval()
        batch = batch_to_device(probe_batch, device)
        outputs = model(
            batch["input_ids"],
            batch["attention_mask"],
            use_quantizer=use_quantizer,
            sample_gates=False,
        )
        free_details = model.free_running(
            outputs["z_latent"],
            outputs["latent_mask"],
            max_length=batch["input_ids"].shape[1],
            return_details=True,
        )
        free_logits = free_details.get("raw_logits", free_details["logits"])
        generated = free_details["generated"]
        valid = batch["attention_mask"].bool()
        teacher_ce = _masked_ce(outputs["logits"], batch["input_ids"], valid)
        free_ce = _masked_ce(free_logits, batch["input_ids"], valid)
        token_accuracy = float(
            ((generated == batch["input_ids"]) & valid).sum()
        ) / max(int(valid.sum()), 1)
        exact = 0
        normalized_edit = 0.0
        for target_row, generated_row, length in zip(
            batch["input_ids"].tolist(),
            generated.tolist(),
            valid.sum(dim=1).tolist(),
            strict=True,
        ):
            target = target_row[:length]
            prediction = _trim_at_eos(
                generated_row[: model.config.max_seq_len],
                model.config.eos_token_id,
            )
            exact += prediction == target
            normalized_edit += _edit_distance(prediction, target) / max(len(target), 1)
        examples = int(batch["input_ids"].shape[0])
        metrics = {
            "examples": examples,
            "teacher_forced_ce_nats_per_bpe": float(teacher_ce),
            "free_running_ce_nats_per_bpe": float(free_ce),
            "exposure_gap_nats_per_bpe": float(free_ce - teacher_ce),
            "exposure_gap_bits_per_bpe": float(free_ce - teacher_ce) / math.log(2.0),
            "free_running_token_accuracy": token_accuracy,
            "free_running_exact_match": exact / max(examples, 1),
            "free_running_normalized_edit_distance": normalized_edit
            / max(examples, 1),
        }
        if model.config.latent_routing == "monotonic_pointer":
            predicted_pointer = model.decode_with_predicted_pointers(
                outputs["z_latent"],
                outputs["latent_mask"],
                batch["input_ids"],
                valid,
            )
            predicted_pointer_ce = _masked_ce(
                predicted_pointer["logits"],
                batch["input_ids"],
                valid,
            )
            teacher_pointers = outputs["segment_ids"]
            predicted_pointers = predicted_pointer["pointer_trace"]
            pointer_error = (
                predicted_pointers - teacher_pointers
            ).abs()
            pointer_tokens = max(int(valid.sum()), 1)
            pointer_exact = float(
                ((predicted_pointers == teacher_pointers) & valid).sum()
            ) / pointer_tokens
            pointer_mae = float(pointer_error[valid].float().mean())

            chunk_counts = outputs["chunk_counts"].long()
            lengths = valid.sum(dim=1).long()
            final_target_positions = (lengths - 1).clamp_min(0)
            batch_indices = torch.arange(
                examples,
                device=predicted_pointers.device,
            )
            target_end_pointers = predicted_pointers[
                batch_indices,
                final_target_positions,
            ]
            target_consumption = (
                (target_end_pointers + 1).float() / chunk_counts.clamp_min(1)
            ).clamp(max=1.0)
            teacher_last_pointers = chunk_counts - 1
            premature_exhaustion = (
                (predicted_pointers >= teacher_last_pointers.unsqueeze(1))
                & (teacher_pointers < teacher_last_pointers.unsqueeze(1))
                & valid
            ).any(dim=1)

            free_pointers = free_details["pointer_trace"]
            free_consumption = []
            first_drift_fractions = []
            for row_index, length in enumerate(lengths.tolist()):
                pointer_matches = (
                    predicted_pointers[row_index, :length]
                    == teacher_pointers[row_index, :length]
                )
                drift = (~pointer_matches).nonzero(as_tuple=False)
                first_drift_fractions.append(
                    float(drift[0, 0]) / max(length, 1)
                    if drift.numel()
                    else 1.0
                )
                generated_row = generated[row_index].tolist()
                try:
                    stop_position = generated_row.index(model.config.eos_token_id)
                except ValueError:
                    stop_position = len(generated_row) - 1
                free_pointer = free_pointers[row_index, stop_position]
                free_consumption.append(float(
                    ((free_pointer + 1).float() / chunk_counts[row_index]).clamp(
                        max=1.0
                    )
                ))

            boundary_metrics = _boundary_metrics_from_counts(_boundary_counts(
                predicted_pointer["boundary_logits"],
                outputs["hard_boundaries"],
                valid,
                threshold=model.config.decoder_boundary_threshold,
            ))
            metrics.update({
                "predicted_pointer_ce_nats_per_bpe": float(predicted_pointer_ce),
                "predicted_pointer_gap_nats_per_bpe": float(
                    predicted_pointer_ce - teacher_ce
                ),
                "predicted_pointer_gap_bits_per_bpe": float(
                    predicted_pointer_ce - teacher_ce
                ) / math.log(2.0),
                "predicted_pointer_token_alignment": pointer_exact,
                "predicted_pointer_mae": pointer_mae,
                "mean_first_pointer_drift_fraction": sum(first_drift_fractions)
                / max(examples, 1),
                "target_end_code_consumption": float(target_consumption.mean()),
                "target_end_unconsumed_code_fraction": float(
                    (1.0 - target_consumption).mean()
                ),
                "premature_code_exhaustion_fraction": float(
                    premature_exhaustion.float().mean()
                ),
                "free_running_code_consumption_at_eos": sum(free_consumption)
                / max(examples, 1),
                **{
                    key.replace(
                        "decoder_boundary_",
                        "predicted_pointer_boundary_",
                    ): value
                    for key, value in boundary_metrics.items()
                },
            })
        return metrics
    finally:
        model.train(was_training)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    tokenizer_path = args.tokenizer_path or SegmentalVQVAETrainConfig.tokenizer_path
    tokenizer = BPETokenizer(tokenizer_path)
    train_cfg, data_cfg, model_cfg = build_configs(args, tokenizer)
    payload = {
        "train": asdict(train_cfg),
        "data": asdict(data_cfg),
        "model": asdict(model_cfg),
    }
    if args.print_config:
        print(json.dumps(payload, indent=2))
        return

    run_name = train_cfg.run_name or time.strftime("segmental_vqvae_%Y%m%d_%H%M%S")
    train_cfg.run_name = run_name
    run_dir = _make_run_dir(run_name)
    device = get_device()
    enable_tf32(device)
    torch.manual_seed(train_cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_cfg.seed)

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
    intervention_loader = make_loader(
        Subset(
            val_dataset,
            range(min(train_cfg.intervention_probe_examples, len(val_dataset))),
        ),
        train_cfg.intervention_probe_examples,
        shuffle=False,
        device=device,
        num_workers=0,
    )
    free_running_loader = make_loader(
        Subset(
            val_dataset,
            range(min(train_cfg.free_running_samples, len(val_dataset))),
        ),
        train_cfg.free_running_samples,
        shuffle=False,
        device=device,
        num_workers=0,
    )
    warmup_probe = _materialize_warmup_probe(
        init_loader,
        train_cfg.ae_warmup_probe_points,
    )
    intervention_probe = _materialize_fixed_batch(
        intervention_loader,
        train_cfg.intervention_probe_examples,
    )
    free_running_probe = _materialize_fixed_batch(
        free_running_loader,
        train_cfg.free_running_samples,
    )

    model = SegmentalVQVAE(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    total_steps = train_cfg.epochs * len(train_loader)
    if train_cfg.ae_warmup_max_steps >= total_steps:
        raise ValueError("Adaptive AE warmup must leave at least one VQ step.")
    controller = AdaptiveWarmupController(
        min_steps=train_cfg.ae_warmup_min_steps,
        max_steps=train_cfg.ae_warmup_max_steps,
        patience=train_cfg.ae_warmup_patience,
        tolerance=train_cfg.ae_warmup_dim_tolerance,
    )
    payload["train"] = asdict(train_cfg)
    payload["parameter_count"] = count_parameters(model)
    payload["data"].update({
        "train_examples": len(train_dataset),
        "eval_examples": len(val_dataset),
    })
    payload["codebook_initialization"] = {
        "method": "kmeans",
        "status": "pending_ae_warmup",
    }
    atomic_json_dump(payload, run_dir / "config.json")

    metrics_path = run_dir / "metrics.jsonl"
    global_step = 0
    transition_step = None
    stop_reason = None
    best_eval_loss = math.inf
    best_step = 0
    last_eval = None
    last_eval_step = None
    last_interventions = None
    last_free_running = None
    started = time.time()

    print(
        f"[Run] {run_name} [Device] {device} "
        f"[Params] {payload['parameter_count']:,}"
    )
    architecture_middle = (
        "contextual latents -> VQ-pre token pruning -> packed EMA VQ"
        if model_cfg.segmentation_mode == "token_pruning"
        else f"{model_cfg.segmentation_mode} segmentation -> EMA VQ"
    )
    print(
        f"[Architecture] BPE -> {architecture_middle} -> "
        f"{model_cfg.latent_routing} AR decoder"
    )
    rate_unit = (
        "kept code"
        if model_cfg.segmentation_mode == "token_pruning"
        else "chunk"
    )
    print(
        f"[Rate target] {model_cfg.compression_target:.2f} BPE/{rate_unit} "
        f"with K={model_cfg.codebook_size}"
    )
    if model_cfg.segmentation_mode == "semi_markov":
        print(
            "[Semi-Markov] "
            f"boundary_layers={model_cfg.boundary_encoder_layers} "
            f"receptive_radius={model_cfg.boundary_window_radius} "
            f"max_span={model_cfg.max_span_length} "
            f"span_layers={model_cfg.span_encoder_layers}"
        )

    with wandb_run(
        run_name,
        group="segmental-vqvae",
        tags=[
            "text",
            "bpe",
            "segmental",
            "vqvae",
            "autoregressive",
            model_cfg.segmentation_mode,
            model_cfg.latent_routing,
        ],
        config=payload,
    ) as tracker:
        for epoch in range(1, train_cfg.epochs + 1):
            model.train()
            for batch in train_loader:
                if stop_reason is not None and transition_step is None:
                    with preserve_rng_state():
                        init_result = initialize_codebook_kmeans(
                            model,
                            init_loader,
                            device,
                            seed=train_cfg.seed,
                        )
                    transition_step = global_step
                    payload["codebook_initialization"] = {
                        "method": "kmeans",
                        "status": "completed",
                        "transition_step": transition_step,
                        "warmup_stop_reason": stop_reason,
                        **init_result,
                    }
                    atomic_json_dump(payload, run_dir / "config.json")
                    append_jsonl({
                        "split": "phase_transition",
                        "event": "kmeans_initialized",
                        "step": global_step,
                        "phase": "vq",
                        "warmup_stop_reason": stop_reason,
                    }, metrics_path)
                    transition_free = evaluate_free_running(
                        model,
                        free_running_probe,
                        device,
                        use_quantizer=True,
                    )
                    append_jsonl({
                        "split": "free_running",
                        "event": "post_kmeans_transition",
                        "step": global_step,
                        "phase": "vq",
                        **transition_free,
                    }, metrics_path)
                    tracker.log(
                        {f"free_running/{key}": value for key, value in transition_free.items()},
                        step=global_step,
                    )
                    print(
                        f"[Phase transition] step={global_step} K-means initialized "
                        f"from {init_result['encoder_vectors']:,} chunks"
                    )

                global_step += 1
                use_quantizer = transition_step is not None
                vq_step = global_step - transition_step if use_quantizer else 0
                batch = batch_to_device(batch, device)
                outputs = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    use_quantizer=use_quantizer,
                )
                losses = segmental_vqvae_losses(
                    outputs,
                    batch["input_ids"],
                    batch["attention_mask"],
                    model_cfg,
                )
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_cfg.grad_clip,
                )
                optimizer.step()
                train_row = {
                    "split": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "vq_step": vq_step,
                    "phase": "vq" if use_quantizer else "ae_warmup",
                    "quantizer_active": int(use_quantizer),
                    "grad_norm": float(grad_norm),
                    "tokens_per_chunk_soft": float(
                        outputs["soft_tokens_per_chunk"].mean().detach()
                    ),
                    "tokens_per_chunk_hard": float(
                        outputs["hard_tokens_per_chunk"].mean().detach()
                    ),
                    "elapsed_sec": time.time() - started,
                    **{
                        key: float(value.detach())
                        for key, value in losses.items()
                    },
                }
                append_jsonl(train_row, metrics_path)
                tracker.log(
                    {
                        f"train/{key}": value
                        for key, value in train_row.items()
                        if isinstance(value, (int, float))
                    },
                    step=global_step,
                )

                if (
                    transition_step is None
                    and (
                        global_step % train_cfg.ae_warmup_check_every == 0
                        or global_step >= train_cfg.ae_warmup_max_steps
                    )
                ):
                    spectrum = evaluate_adaptive_warmup(
                        model,
                        warmup_probe,
                        codebook_size=model_cfg.codebook_size,
                        variance_threshold=train_cfg.ae_warmup_variance_threshold,
                    )
                    decision = controller.observe(global_step, spectrum)
                    warmup_row = {
                        "split": "ae_warmup_diagnostic",
                        "step": global_step,
                        "phase": "ae_warmup",
                        **spectrum,
                        **decision,
                    }
                    append_jsonl(warmup_row, metrics_path)
                    tracker.log(
                        {
                            f"ae_warmup/{key}": value
                            for key, value in warmup_row.items()
                            if isinstance(value, (int, float))
                        },
                        step=global_step,
                    )
                    if decision["should_stop"]:
                        stop_reason = str(decision["reason"])

                eval_due = global_step == 1 or global_step % train_cfg.eval_every == 0
                if eval_due:
                    last_eval = evaluate(
                        model,
                        val_loader,
                        device,
                        use_quantizer=use_quantizer,
                    )
                    interventions, segmentation_snapshot = evaluate_interventions(
                        model,
                        intervention_probe,
                        device,
                        use_quantizer=use_quantizer,
                        seed=train_cfg.seed + 17,
                        tokenizer=tokenizer,
                    )
                    eval_row = {
                        "split": "eval",
                        "epoch": epoch,
                        "step": global_step,
                        "vq_step": vq_step,
                        "phase": "vq" if use_quantizer else "ae_warmup",
                        **last_eval,
                    }
                    intervention_row = {
                        "split": "latent_intervention",
                        "epoch": epoch,
                        "step": global_step,
                        "vq_step": vq_step,
                        "phase": "vq" if use_quantizer else "ae_warmup",
                        **interventions,
                    }
                    append_jsonl(eval_row, metrics_path)
                    append_jsonl(intervention_row, metrics_path)
                    last_eval_step = global_step
                    last_interventions = interventions
                    write_segmentation_visualization(
                        segmentation_snapshot,
                        run_dir / "plots",
                        compression_target=model_cfg.compression_target,
                        run_name=run_name,
                    )
                    plot_segmental_metrics(
                        metrics_path,
                        run_dir / "plots",
                        compression_target=model_cfg.compression_target,
                        run_name=run_name,
                    )
                    tracker.log(
                        {f"eval/{key}": value for key, value in last_eval.items()},
                        step=global_step,
                    )
                    tracker.log(
                        {
                            f"intervention/{key}": value
                            for key, value in interventions.items()
                        },
                        step=global_step,
                    )
                    if use_quantizer and last_eval["loss"] < best_eval_loss:
                        best_eval_loss = float(last_eval["loss"])
                        best_step = global_step
                        save_model_checkpoint(
                            model,
                            run_dir / "checkpoints" / "best.pt",
                            step=global_step,
                            epoch=epoch,
                            phase="vq",
                        )
                    print(
                        f"[Eval] step={global_step} loss={last_eval['loss']:.4f} "
                        f"r={last_eval['tokens_per_chunk_hard']:.3f} "
                        f"rate={last_eval['rate_bits_per_bpe']:.3f} "
                        f"latent_gain={interventions['delta_null_bits_per_bpe']:.3f}"
                    )

                free_running_due = (
                    use_quantizer
                    and vq_step > 0
                    and vq_step % train_cfg.free_running_every == 0
                )
                if free_running_due:
                    last_free_running = evaluate_free_running(
                        model,
                        free_running_probe,
                        device,
                        use_quantizer=True,
                    )
                    free_row = {
                        "split": "free_running",
                        "epoch": epoch,
                        "step": global_step,
                        "vq_step": vq_step,
                        "phase": "vq",
                        **last_free_running,
                    }
                    append_jsonl(free_row, metrics_path)
                    tracker.log(
                        {
                            f"free_running/{key}": value
                            for key, value in last_free_running.items()
                        },
                        step=global_step,
                    )

                if rolling_checkpoint_due(
                    step=global_step,
                    every=train_cfg.save_every,
                ):
                    save_resume_checkpoint(
                        model,
                        optimizer,
                        run_dir / "checkpoints" / "latest.pt",
                        step=global_step,
                        epoch=epoch,
                        phase="vq" if use_quantizer else "ae_warmup",
                    )

        if transition_step is None:
            raise RuntimeError("Training ended before adaptive AE warmup transitioned.")
        if last_eval_step != global_step:
            last_eval = evaluate(model, val_loader, device, use_quantizer=True)
            last_interventions, segmentation_snapshot = evaluate_interventions(
                model,
                intervention_probe,
                device,
                use_quantizer=True,
                seed=train_cfg.seed + 17,
                tokenizer=tokenizer,
            )
            append_jsonl({
                "split": "eval",
                "event": "final",
                "step": global_step,
                "vq_step": global_step - transition_step,
                "phase": "vq",
                **last_eval,
            }, metrics_path)
            append_jsonl({
                "split": "latent_intervention",
                "event": "final",
                "step": global_step,
                "vq_step": global_step - transition_step,
                "phase": "vq",
                **last_interventions,
            }, metrics_path)
            tracker.log(
                {f"eval/{key}": value for key, value in last_eval.items()},
                step=global_step,
            )
            tracker.log(
                {
                    f"intervention/{key}": value
                    for key, value in last_interventions.items()
                },
                step=global_step,
            )
            write_segmentation_visualization(
                segmentation_snapshot,
                run_dir / "plots",
                compression_target=model_cfg.compression_target,
                run_name=run_name,
            )
        last_free_running = evaluate_free_running(
            model,
            free_running_probe,
            device,
            use_quantizer=True,
        )
        append_jsonl({
            "split": "free_running",
            "event": "final",
            "step": global_step,
            "vq_step": global_step - transition_step,
            "phase": "vq",
            **last_free_running,
        }, metrics_path)
        if last_eval["loss"] < best_eval_loss:
            best_eval_loss = float(last_eval["loss"])
            best_step = global_step
            save_model_checkpoint(
                model,
                run_dir / "checkpoints" / "best.pt",
                step=global_step,
                epoch=train_cfg.epochs,
                phase="vq",
            )
        retained_checkpoints = finalize_checkpoints(
            model,
            optimizer,
            run_dir / "checkpoints",
            step=global_step,
            epoch=train_cfg.epochs,
            phase="vq",
            save_last_resume=train_cfg.save_last_resume,
        )
        plot_segmental_metrics(
            metrics_path,
            run_dir / "plots",
            compression_target=model_cfg.compression_target,
            run_name=run_name,
        )

    atomic_json_dump({
        "run_name": run_name,
        "status": "completed",
        "steps": global_step,
        "actual_ae_warmup_steps": transition_step,
        "ae_warmup_stop_reason": stop_reason,
        "best_eval_loss": best_eval_loss,
        "best_step": best_step,
        "final_eval": last_eval,
        "final_interventions": last_interventions,
        "final_free_running": last_free_running,
        "retained_checkpoints": retained_checkpoints,
        "elapsed_sec": time.time() - started,
    }, run_dir / "summary.json")


if __name__ == "__main__":
    main()
