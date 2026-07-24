"""All text VQ-VAE configuration types and defaults.

Categorical text options use ``Literal``. Free-form identifiers and paths remain
``str`` because their valid values are supplied by users or external datasets.
"""

from dataclasses import dataclass
from typing import Literal

from common.text_data import (
    DEFAULT_BPE_TOKENIZER_PATH,
    DEFAULT_HF_DATASET_CACHE,
    DEFAULT_TEXT_DATASET,
)


EncoderType = Literal["absolute", "rope", "vqgans"]
DecoderType = Literal["cross_attention", "memory_trunk", "vqgans"]
TokenizerType = Literal["bpe", "byte"]
CodebookInitialization = Literal["random", "kmeans"]
DataSource = Literal["huggingface", "file"]
PCAFitMode = Literal["balanced", "all"]
GeometryRenderBasis = Literal["t0", "first_last", "pooled"]
CollapsePreset = Literal["none", "anti"]


@dataclass
class TrainConfig:
    run_name: str = ""
    seed: int = 42
    epochs: int = 5
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 200
    save_every: int = 1000
    num_workers: int = 0
    tokenizer: TokenizerType = "bpe"
    tokenizer_path: str | None = str(DEFAULT_BPE_TOKENIZER_PATH)
    # Research flag: "random" is the control; "kmeans" uses an encoder pre-pass.
    codebook_init: CodebookInitialization = "kmeans"
    # Free-form experiment label, not a categorical option.
    ablation: str | None = None


@dataclass
class DataConfig:
    source: DataSource = "huggingface"
    dataset: str | None = DEFAULT_TEXT_DATASET
    dataset_config: str | None = None
    split: str | None = "train"
    text_field: str = "text"
    data_file: str | None = None
    cache_dir: str | None = str(DEFAULT_HF_DATASET_CACHE)
    streaming: bool | None = False
    max_train_samples: int | None = 50000
    max_eval_samples: int = 2048
    val_fraction: float = 0.02


@dataclass
class TextVQVAEConfig:
    vocab_size: int = 258
    max_seq_len: int = 256
    latent_slots: int = 128
    d_model: int = 448
    n_heads: int = 8
    encoder_layers: int = 4
    encoder_type: EncoderType = "rope"
    decoder_layers: int = 6
    decoder_type: DecoderType = "memory_trunk"
    memory_decoder_latent_layers: int = 4
    memory_decoder_output_layers: int = 2
    ffn_mult: int = 4
    dropout: float = 0.1
    codebook_size: int = 3072
    commitment_beta: float = 0.25
    pad_token_id: int = 257
    slot_pad_ratio_threshold: float = 0.5
    l2_normalize_before_vq: bool = False

@dataclass
class CollapseControlConfig:
    """Engineering controls commonly used to reduce codebook collapse."""

    enabled: bool = False
    use_ema_codebook: bool = False
    ema_decay: float = 0.99
    ema_eps: float = 1e-5
    entropy_weight: float = 0.0
    entropy_temperature: float = 1.0
    diversity_weight: float = 0.0
    code_dropout: float = 0.0
    stochastic_code_sampling: bool = False
    sampling_temperature: float = 0.5
    sampling_topk: int = 8
    dead_code_reset_every: int = 0
    dead_code_reset_usage_threshold: float = 1.0
    normalize_latents: bool = False
    commitment_beta_start: float | None = None
    commitment_beta_warmup_steps: int = 0

@dataclass
class DiagnosticsConfig:
    initial_pca_enabled: bool = True
    initial_pca_max_points: int = 8192
    initial_pca_fit_mode: PCAFitMode = "balanced"
    initial_pca_strict: bool = False
    geometry_snapshot_enabled: bool = True
    geometry_dense_every: int = 50
    geometry_dense_until: int = 1500
    geometry_sparse_every: int = 500
    geometry_probe_points: int = 4096
    geometry_render_enabled: bool = True
    geometry_render_basis: GeometryRenderBasis = "first_last"
    geometry_render_fps: int = 8
    geometry_keep_snapshots: bool = True
