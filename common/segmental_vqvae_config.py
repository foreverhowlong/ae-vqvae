"""Single-source configuration for segmental BPE VQ-VAE experiments."""

from dataclasses import dataclass

from common.text_data import (
    DEFAULT_BPE_TOKENIZER_PATH,
    DEFAULT_HF_DATASET_CACHE,
    DEFAULT_TEXT_DATASET,
)

LATENT_ROUTING_MODES = (
    "global_cross_attention",
    "monotonic_pointer",
)

SEGMENTATION_MODES = (
    "bernoulli",
    "semi_markov",
    "token_pruning",
)


@dataclass
class SegmentalVQVAETrainConfig:
    run_name: str = ""
    seed: int = 42
    epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 200
    save_every: int = 1000
    save_last_resume: bool = False
    num_workers: int = 0
    tokenizer_path: str = str(DEFAULT_BPE_TOKENIZER_PATH)
    ae_warmup_min_steps: int = 1000
    ae_warmup_max_steps: int = 6000
    ae_warmup_check_every: int = 200
    ae_warmup_patience: int = 5
    ae_warmup_dim_tolerance: int = 1
    ae_warmup_probe_points: int = 8192
    ae_warmup_variance_threshold: float = 0.99
    intervention_probe_examples: int = 32
    free_running_every: int = 2000
    free_running_samples: int = 32
    ablation: str | None = None


@dataclass
class SegmentalVQVAEDataConfig:
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
class SegmentalVQVAEConfig:
    vocab_size: int = 8192
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3
    max_seq_len: int = 256
    d_model: int = 448
    latent_dim: int = 32
    n_heads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 6
    latent_routing: str = "global_cross_attention"
    segmentation_mode: str = "bernoulli"
    boundary_encoder_layers: int = 4
    boundary_window_radius: int = 16
    max_span_length: int = 16
    span_encoder_layers: int = 2
    ffn_mult: int = 4
    dropout: float = 0.1
    codebook_size: int = 8192
    commitment_beta: float = 0.25
    compression_target: float = 1.67
    compression_weight: float = 10.0
    gate_logit_l2_weight: float = 1e-4
    gate_threshold: float = 0.5
    decoder_boundary_loss_weight: float = 1.0
    decoder_boundary_threshold: float = 0.5
    ema_decay: float = 0.99
    ema_eps: float = 1e-5

    def validate(self) -> None:
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least two.")
        if self.d_model < 1 or self.latent_dim < 1:
            raise ValueError("d_model and latent_dim must be positive.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension.")
        if self.encoder_layers < 1 or self.decoder_layers < 1:
            raise ValueError("Encoder and decoder depths must be positive.")
        if self.boundary_encoder_layers < 1 or self.span_encoder_layers < 1:
            raise ValueError("Boundary and span encoder depths must be positive.")
        if self.latent_routing not in LATENT_ROUTING_MODES:
            raise ValueError(
                "latent_routing must be global_cross_attention or monotonic_pointer."
            )
        if self.segmentation_mode not in SEGMENTATION_MODES:
            raise ValueError(
                "segmentation_mode must be bernoulli, semi_markov, or "
                "token_pruning."
            )
        if (
            self.segmentation_mode == "token_pruning"
            and self.latent_routing != "global_cross_attention"
        ):
            raise ValueError(
                "token_pruning requires global_cross_attention latent routing."
            )
        if self.boundary_window_radius < 1:
            raise ValueError("boundary_window_radius must be positive.")
        if not 1 <= self.max_span_length <= self.max_seq_len:
            raise ValueError(
                "max_span_length must be between one and max_seq_len."
            )
        if self.codebook_size < 2:
            raise ValueError("codebook_size must be at least two.")
        if not 0.0 < self.gate_threshold < 1.0:
            raise ValueError("gate_threshold must be strictly between zero and one.")
        if not 0.0 < self.decoder_boundary_threshold < 1.0:
            raise ValueError(
                "decoder_boundary_threshold must be strictly between zero and one."
            )
        if self.compression_target < 1.0:
            raise ValueError("compression_target must be at least one token per chunk.")
        if min(
            self.commitment_beta,
            self.compression_weight,
            self.gate_logit_l2_weight,
            self.decoder_boundary_loss_weight,
        ) < 0:
            raise ValueError("Loss weights must be non-negative.")
        special_ids = {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
        }
        if len(special_ids) != 3 or not all(
            0 <= token_id < self.vocab_size for token_id in special_ids
        ):
            raise ValueError("PAD, BOS, and EOS ids must be distinct vocabulary ids.")
