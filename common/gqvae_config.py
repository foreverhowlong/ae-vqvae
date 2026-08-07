"""Single-source configuration for gated, variable-length text VQ-VAE runs."""

from dataclasses import dataclass

from common.text_data import DEFAULT_HF_DATASET_CACHE, DEFAULT_TEXT_DATASET


@dataclass
class GQVAETrainConfig:
    run_name: str = ""
    seed: int = 42
    epochs: int = 10
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 200
    save_every: int = 1000
    num_workers: int = 0
    ae_warmup_min_steps: int = 1000
    ae_warmup_max_steps: int = 6000
    ae_warmup_check_every: int = 200
    ae_warmup_patience: int = 5
    ae_warmup_dim_tolerance: int = 1
    ae_warmup_probe_points: int = 8192
    ae_warmup_variance_threshold: float = 0.99
    compression_warmup_steps: int = 4000
    ablation: str | None = None


@dataclass
class GQVAEDataConfig:
    dataset: str = DEFAULT_TEXT_DATASET
    dataset_config: str | None = None
    split: str = "train"
    text_field: str = "text"
    data_file: str | None = None
    cache_dir: str = str(DEFAULT_HF_DATASET_CACHE)
    max_train_samples: int | None = None
    max_eval_samples: int = 2048
    val_fraction: float = 0.02
    continuous_truncation: bool = True


@dataclass
class GQVAEConfig:
    vocab_size: int = 258
    pad_token_id: int = 257
    max_seq_len: int = 256
    d_model: int = 256
    code_dim: int = 128
    n_heads: int = 8
    encoder_layers: int = 4
    gater_layers: int = 2
    ffn_mult: int = 4
    dropout: float = 0.1
    codebook_size: int = 8192
    decode_width: int = 10
    commitment_beta: float = 0.25
    compression_weight: float = 3.0
    length_weight: float = 1.0
    gate_threshold: float = 0.5
    use_ema_codebook: bool = True
    ema_decay: float = 0.99
    ema_eps: float = 1e-5

