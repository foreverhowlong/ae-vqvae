"""Data, optimization, and diagnostics for the GQ-VAE paper reproduction."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import regex
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common.gqvae_paper_config import (
    GQVAEPaperDataConfig,
    GQVAEPaperTrainConfig,
    GateBistabilityConfig,
)
from models.gqvae_paper import GQVAEPaper, GQVAEPaperOutput


GPT2_REGEX = regex.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


class ASCIIPieceDataset(Dataset):
    """Paper preprocessing: concatenated TinyStories, GPT-2 regex, ASCII, pad 0."""

    def __init__(self, rows: torch.Tensor):
        if rows.ndim != 2:
            raise ValueError("Prepared paper rows must have shape [examples, input_len].")
        self.rows = rows.to(dtype=torch.uint8, device="cpu")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.rows[index].long()


def preprocess_ascii_pieces(texts: list[str], *, input_len: int) -> torch.Tensor:
    # Joining before regex is a meaningful part of the released preprocessing.
    pieces = GPT2_REGEX.findall("".join(texts))
    rows: list[list[int]] = []
    for piece in pieces:
        if len(piece) > input_len:
            continue
        values = [ord(character) for character in piece if ord(character) < 128]
        values.extend([0] * (input_len - len(values)))
        rows.append(values)
    if not rows:
        raise ValueError("GQ-VAE paper preprocessing produced no ASCII regex pieces.")
    return torch.tensor(rows, dtype=torch.uint8)


def load_or_prepare_dataset(
    config: GQVAEPaperDataConfig,
    *,
    input_len: int,
    prepared_output: Path,
) -> ASCIIPieceDataset:
    if config.prepared_data is not None:
        rows = torch.load(config.prepared_data, map_location="cpu", weights_only=True)
        return ASCIIPieceDataset(rows)

    from datasets import load_dataset

    dataset = load_dataset(
        config.dataset,
        config.dataset_config,
        split=config.split,
        cache_dir=config.cache_dir,
    )
    if config.max_source_documents is not None:
        dataset = dataset.select(range(min(config.max_source_documents, len(dataset))))
    rows = preprocess_ascii_pieces(list(dataset[config.text_field]), input_len=input_len)
    prepared_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, prepared_output)
    return ASCIIPieceDataset(rows)


def split_paper_dataset(
    dataset: ASCIIPieceDataset,
    *,
    train_fraction: float,
) -> tuple[ASCIIPieceDataset, ASCIIPieceDataset]:
    boundary = int(len(dataset) * train_fraction)
    if boundary == 0 or boundary == len(dataset):
        raise ValueError("Paper dataset split must contain both train and validation rows.")
    # No randomized split: the official code performs one contiguous 90/10 split.
    return ASCIIPieceDataset(dataset.rows[:boundary]), ASCIIPieceDataset(dataset.rows[boundary:])


def make_paper_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        num_workers=num_workers,
    )


def build_paper_optimizer(
    model: GQVAEPaper,
    config: GQVAEPaperTrainConfig,
) -> torch.optim.Optimizer:
    codebook_parameters = list(model.quantizer.parameters())
    codebook_ids = {id(parameter) for parameter in codebook_parameters}
    base_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in codebook_ids
    ]
    return torch.optim.Adam(
        [
            {
                "params": base_parameters,
                "lr": config.learning_rate,
                "amsgrad": config.adam_amsgrad,
                "weight_decay": config.weight_decay,
            },
            {
                "params": codebook_parameters,
                "lr": config.learning_rate * config.codebook_learning_rate_multiplier,
                "amsgrad": config.adam_amsgrad,
                "weight_decay": config.weight_decay,
            },
        ]
    )


def build_paper_scheduler(
    optimizer: torch.optim.Optimizer,
    config: GQVAEPaperTrainConfig,
    *,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    if total_steps <= config.lr_warmup_steps:
        raise ValueError(
            f"Paper LR schedule needs more than {config.lr_warmup_steps} steps; "
            f"the dataset supplies {total_steps}."
        )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=config.lr_warmup_start_factor,
        total_iters=config.lr_warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - config.lr_warmup_steps,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[config.lr_warmup_steps],
    )


def gate_statistics(
    gates: torch.Tensor,
    config: GateBistabilityConfig,
) -> dict[str, float | int | list[int] | str]:
    values = gates.detach().float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("Cannot diagnose an empty gate tensor.")
    low = float((values < config.low_threshold).float().mean())
    high = float((values > config.high_threshold).float().mean())
    hard_on = float((values > 0.5).float().mean())
    sequence_low = (gates.detach() < config.low_threshold).float().mean(dim=1)
    sequence_high = (gates.detach() > config.high_threshold).float().mean(dim=1)
    zero_sequences = float((sequence_low >= config.collapse_fraction).float().mean())
    one_sequences = float((sequence_high >= config.collapse_fraction).float().mean())
    histogram = torch.histc(values.cpu(), bins=config.histogram_bins, min=0.0, max=1.0)
    if low >= config.collapse_fraction:
        state = "collapsed_zero"
    elif high >= config.collapse_fraction:
        state = "collapsed_one"
    elif low + high >= config.collapse_fraction:
        state = "polarized"
    else:
        state = "interior"
    return {
        "gate_mean": float(values.mean()),
        "gate_std": float(values.std(unbiased=False)),
        "gate_min": float(values.min()),
        "gate_max": float(values.max()),
        "gate_hard_on_fraction": hard_on,
        "gate_low_fraction": low,
        "gate_high_fraction": high,
        "gate_mid_fraction": max(0.0, 1.0 - low - high),
        "gate_zero_sequence_fraction": zero_sequences,
        "gate_one_sequence_fraction": one_sequences,
        "gate_histogram": histogram.long().tolist(),
        "gate_state": state,
    }


def output_metrics(
    output: GQVAEPaperOutput,
    input_ids: torch.Tensor,
    diagnostics: GateBistabilityConfig,
) -> dict[str, float | int | list[int] | str]:
    hard_gates = output.gates > 0.5
    correct = output.byte_logits.argmax(dim=1) == output.reconstruction_targets
    predicted = output.predicted_masks > 0.5
    selected = hard_gates.sum()
    correct_masked_characters = torch.logical_and(
        output.reconstruction_masks > 0.5,
        correct,
    )[hard_gates]
    masked_character_count = (output.reconstruction_masks > 0.5)[hard_gates]
    exact_mask_tokens = torch.logical_or(
        output.reconstruction_masks < 0.5,
        correct,
    ).all(dim=2)[hard_gates]
    exact_predicted_tokens = torch.logical_or(~predicted, correct).all(dim=2)[hard_gates]
    metrics: dict[str, float | int | list[int] | str] = {
        "loss": float(output.loss.detach()),
        "reconstruction_loss": float(output.reconstruction_loss.detach()),
        "compression_loss": float(output.compression_loss.detach()),
        "length_loss": float(output.length_loss.detach()),
        "codebook_loss": (
            math.nan if output.codebook_loss is None else float(output.codebook_loss.detach())
        ),
        "commitment_loss": (
            math.nan
            if output.commitment_loss is None
            else float(output.commitment_loss.detach())
        ),
        "codebook_utilization_batch": (
            math.nan
            if output.codebook_utilization_batch is None
            else output.codebook_utilization_batch
        ),
        "correct_masked_character": float(
            correct_masked_characters.float().sum()
            / masked_character_count.float().sum().clamp_min(1)
        ),
        "correct_masked_token": float(exact_mask_tokens.float().mean())
        if exact_mask_tokens.numel()
        else math.nan,
        "correct_predicted_token": float(exact_predicted_tokens.float().mean())
        if exact_predicted_tokens.numel()
        else math.nan,
        "bytes_per_token": float((input_ids != 0).sum() / selected.clamp_min(1)),
        "selected_tokens": int(selected),
    }
    metrics.update(gate_statistics(output.gates, diagnostics))
    return metrics


def numeric_tracker_metrics(metrics: dict[str, object]) -> dict[str, float | int]:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }


def reproduction_manifest(
    train: GQVAEPaperTrainConfig,
    data: GQVAEPaperDataConfig,
) -> dict[str, object]:
    return {
        "faithful_training": {
            "optimizer": "Adam",
            "parameter_groups": "quantizer_lr=10x_base_lr",
            "scheduler": "1000-step LinearLR(0.1) then cosine",
            "quantizer": "500-step identity warmup; unused-code resample every 250 steps",
            "objective": "reconstruction + alpha*mean(gate) + gamma*length + VQ losses",
            "gate_threshold": 0.5,
            "epoch_semantics": "one full loader pass, matching released executable code",
            "validation_mode": "train mode, matching released validate()",
        },
        "paper_code_ambiguities": {
            "declared_epochs": 15,
            "executed_epochs": train.epochs,
            "released_data_partitions_loaded": 1,
            "dataset_split": data.split,
        },
        "diagnostics_do_not_affect_gradients": True,
        "train_config": asdict(train),
        "data_config": asdict(data),
    }
