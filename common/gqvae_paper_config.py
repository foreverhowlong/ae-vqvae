"""Configuration for the faithful GQ-VAE v1 paper reproduction.

The defaults mirror the executable defaults in the authors' public repository
at revision ``9366387cafc3aeaa16fb33506762698b077d28d8``.  This profile is kept
separate from :mod:`common.gqvae_config`: the latter is the project's scalable
GQ-inspired implementation, while this module describes the paper model.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, TypeVar

from common.text_data import DEFAULT_HF_DATASET_CACHE, DEFAULT_TEXT_DATASET


PAPER_REPOSITORY = "https://github.com/Theo-Datta-115/gq-vae"
PAPER_REVISION = "9366387cafc3aeaa16fb33506762698b077d28d8"
PAPER_ARXIV = "https://arxiv.org/abs/2512.21913"


@dataclass(frozen=True)
class GQVAEPaperModelConfig:
    input_len: int = 16
    embedding_dim: int = 1024
    codebook_dim: int = 1024
    alphabet_size: int = 128
    codebook_size: int = 50_000
    decode_width: int = 10
    encoder_depth: int = 4
    gater_depth: int = 2
    decoder_depth: int = 4
    attention_head_dim: int = 64
    compression_alpha: float = 3.0
    commitment_beta: float = 0.25
    length_gamma: float = 1.0
    gate_threshold: float = 0.5
    quantizer_reservoir_size: int = 131_072
    quantizer_warmup_steps: int = 500
    quantizer_resample_every: int = 250
    quantizer_usage_decay: float = 0.99
    quantizer_hardset: int | None = None

    def validate(self) -> None:
        if self.embedding_dim != self.codebook_dim:
            raise ValueError(
                "The paper implementation requires embedding_dim == codebook_dim."
            )
        if self.codebook_dim % self.attention_head_dim:
            raise ValueError("codebook_dim must be divisible by attention_head_dim.")
        if self.input_len < 1 or self.decode_width < 1:
            raise ValueError("input_len and decode_width must be positive.")
        if self.alphabet_size != 128:
            raise ValueError("The TinyStories paper profile uses a 128-symbol ASCII alphabet.")
        if self.codebook_size < 1:
            raise ValueError("codebook_size must be positive.")
        if not 0.0 < self.gate_threshold < 1.0:
            raise ValueError("gate_threshold must be strictly between zero and one.")


@dataclass(frozen=True)
class GQVAEPaperTrainConfig:
    # The released main.py has an --epochs default of 15, but its epoch loop is
    # commented out and the executable performs exactly one loader pass.
    epochs: int = 1
    batch_size: int = 1024
    learning_rate: float = 1e-4
    codebook_learning_rate_multiplier: float = 10.0
    weight_decay: float = 1e-4
    adam_amsgrad: bool = True
    lr_warmup_steps: int = 1000
    lr_warmup_start_factor: float = 0.1
    log_every: int = 50
    eval_every: int = 1001
    eval_batches: int = 50
    save_every: int = 5000
    final_checkpoint: str = "none"
    num_workers: int = 0

    def validate(self) -> None:
        if self.epochs != 1:
            raise ValueError(
                "The faithful paper-code profile performs one training-loader pass."
            )
        if self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("batch_size and learning_rate must be positive.")
        if self.lr_warmup_steps < 1:
            raise ValueError("lr_warmup_steps must be positive.")
        if self.save_every < 0:
            raise ValueError("save_every must be non-negative.")
        if self.final_checkpoint not in {"none", "model", "resume"}:
            raise ValueError(
                "final_checkpoint must be one of: none, model, resume."
            )


@dataclass(frozen=True)
class GQVAEPaperDataConfig:
    dataset: str = DEFAULT_TEXT_DATASET
    dataset_config: str | None = None
    split: str = "train[:10%]"
    text_field: str = "text"
    cache_dir: str = str(DEFAULT_HF_DATASET_CACHE)
    regex_rule: str = "gpt2"
    ascii_only: bool = True
    max_piece_length: int = 16
    train_fraction: float = 0.9
    max_source_documents: int | None = None
    prepared_data: str | None = (
        "data/prepared/gqvae-paper-v1-tinystories-train-10pct-ascii-gpt2-len16.pt"
    )

    def validate(self) -> None:
        if self.regex_rule != "gpt2":
            raise ValueError("The paper profile requires the GPT-2 regex.")
        if not self.ascii_only:
            raise ValueError("The TinyStories paper profile uses ASCII inputs.")
        if self.max_piece_length != 16:
            raise ValueError("The paper profile filters regex pieces at length 16.")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be strictly between zero and one.")


@dataclass(frozen=True)
class GateBistabilityConfig:
    histogram_bins: int = 20
    low_threshold: float = 0.1
    high_threshold: float = 0.9
    collapse_fraction: float = 0.95

    def validate(self) -> None:
        if self.histogram_bins < 2:
            raise ValueError("histogram_bins must be at least two.")
        if not 0.0 < self.low_threshold < self.high_threshold < 1.0:
            raise ValueError("Gate diagnostic thresholds must satisfy 0 < low < high < 1.")
        if not 0.5 < self.collapse_fraction <= 1.0:
            raise ValueError("collapse_fraction must be in (0.5, 1].")


T = TypeVar("T")


def dataclass_from_dict(cls: type[T], payload: dict[str, Any] | None) -> T:
    """Construct a strict dataclass so config typos fail before a long run."""
    values = {} if payload is None else dict(payload)
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {', '.join(unknown)}")
    result = cls(**values)
    result.validate()
    return result
