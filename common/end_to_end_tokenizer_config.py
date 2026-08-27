"""Configuration for the end-to-end greedy VQ tokenizer experiment."""

from dataclasses import dataclass

from common.text_data import (
    DEFAULT_BPE_TOKENIZER_PATH,
    DEFAULT_HF_DATASET_CACHE,
    DEFAULT_TEXT_DATASET,
)


@dataclass
class EndToEndTokenizerTrainConfig:
    run_name: str = ""
    ablation: str | None = None
    seed: int = 42
    epochs: int = 10
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 200
    num_workers: int = 0
    tokenizer_path: str = str(DEFAULT_BPE_TOKENIZER_PATH)
    target_prior_parameters: int = 18_000_000
    parameter_tolerance: float = 0.05
    save_last_resume: bool = False


@dataclass
class EndToEndTokenizerDataConfig:
    dataset: str | None = DEFAULT_TEXT_DATASET
    dataset_config: str | None = None
    split: str = "train"
    text_field: str = "text"
    data_file: str | None = None
    cache_dir: str | None = str(DEFAULT_HF_DATASET_CACHE)
    max_train_samples: int | None = 50000
    max_eval_samples: int = 2048
    val_fraction: float = 0.02
    continuous_truncation: bool = False


@dataclass
class EndToEndTokenizerConfig:
    vocab_size: int = 8192
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3
    max_seq_len: int = 256

    segmenter_d_model: int = 256
    segmenter_n_heads: int = 8
    boundary_encoder_layers: int = 4
    boundary_window_radius: int = 16
    max_span_length: int = 16
    span_encoder_layers: int = 2
    segmenter_ffn_mult: int = 4

    latent_dim: int = 32
    codebook_size: int = 8192
    commitment_beta: float = 0.25
    ema_decay: float = 0.99
    ema_eps: float = 1e-5
    code_target_topk: int = 32
    code_target_temperature: float = 1.0

    prior_layers: int = 8
    prior_heads: int = 6
    prior_d_model: int = 384
    prior_dropout: float = 0.0
    prior_bias: bool = False

    text_decoder_layers: int = 2
    text_decoder_heads: int = 8
    text_decoder_d_model: int = 256
    text_decoder_dropout: float = 0.1
    text_decoder_bias: bool = False

    compression_target: float = 1.67
    rate_dual_initial: float = 0.0
    rate_dual_lr: float = 0.05
    rate_dual_max_abs: float = 20.0
    word_boundary_only: bool = True
    dropout: float = 0.1

    def validate(self) -> None:
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least two.")
        if not 1 <= self.max_span_length <= self.max_seq_len:
            raise ValueError("max_span_length must be in [1, max_seq_len].")
        if self.compression_target < 1.0:
            raise ValueError("compression_target must be at least one.")
        if self.codebook_size < 2 or self.latent_dim < 1:
            raise ValueError("codebook_size and latent_dim must be valid.")
        if not 1 <= self.code_target_topk <= self.codebook_size:
            raise ValueError("code_target_topk must be in [1, codebook_size].")
        if self.code_target_temperature <= 0:
            raise ValueError("code_target_temperature must be positive.")
        for width, heads, name in (
            (self.segmenter_d_model, self.segmenter_n_heads, "segmenter"),
            (self.prior_d_model, self.prior_heads, "prior"),
            (self.text_decoder_d_model, self.text_decoder_heads, "text decoder"),
        ):
            if width < 1 or heads < 1 or width % heads:
                raise ValueError(f"{name} width must be divisible by its head count.")
        if min(
            self.boundary_encoder_layers,
            self.span_encoder_layers,
            self.prior_layers,
            self.text_decoder_layers,
        ) < 1:
            raise ValueError("All model depths must be positive.")
        if self.rate_dual_lr < 0 or self.rate_dual_max_abs <= 0:
            raise ValueError("Rate-dual learning rate and bound must be valid.")
        if not 0 <= self.commitment_beta:
            raise ValueError("commitment_beta must be non-negative.")
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        if len(special_ids) != 3 or not all(
            0 <= token_id < self.vocab_size for token_id in special_ids
        ):
            raise ValueError("PAD, BOS, and EOS ids must be distinct and in range.")
