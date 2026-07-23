"""Configuration dataclasses and CLI parser for text VQ-VAE experiments.

Flag discipline
---------------
* True defaults live only in the dataclass fields below.  argparse uses
  ``default=None`` everywhere so that "flag not passed" is distinguishable
  from "flag passed with the default value."  ``build_configs()`` merges
  non-None overrides on top of the dataclass defaults. CLI help annotates its
  actions from those same dataclass instances instead of copying defaults.

* Bug-fix / metric-validity changes → make default, delete old code path,
  record in CHANGELOG with git tag.

* Research-question flags (controlled ablations) → keep as CLI flags with a
  comment stating which hypothesis the flag tests.  Default = current best
  configuration.  Delete the flag once the ablation is consumed.

* Impossible / meaningless combinations → assert in ``build_configs()``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from common.text_data import DEFAULT_BPE_TOKENIZER_PATH, DEFAULT_HF_DATASET_CACHE, DEFAULT_TEXT_DATASET
from models.text_vqvae import (
    DECODER_TYPES,
    ENCODER_TYPES,
    CollapseControlConfig,
    TextVQVAEConfig,
)


# ---------------------------------------------------------------------------
# Dataclasses – single source of truth for defaults
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    run_name: str = ""                  # filled in at runtime from timestamp
    seed: int = 42
    epochs: int = 5
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 200
    save_every: int = 1000
    num_workers: int = 0
    tokenizer: str = "bpe"
    tokenizer_path: str | None = str(DEFAULT_BPE_TOKENIZER_PATH)
    # Research flag: codebook initialisation strategy.
    # "random"  → normal_(std=d**-0.5); keep as control group, do not modify.
    # "kmeans"  → MiniBatch KMeans over one encoder pre-pass.
    # Hypothesis tested: does data-driven init eliminate the early high-active-
    # code spike and reduce/delay collapse? (see technical summary §2.3)
    codebook_init: str = "kmeans"
    ablation: str | None = None


@dataclass
class DataConfig:
    source: str = "huggingface"
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
class DiagnosticsConfig:
    initial_pca_enabled: bool = True
    initial_pca_max_points: int = 8192
    initial_pca_fit_mode: str = "balanced"
    initial_pca_strict: bool = False
    geometry_snapshot_enabled: bool = True
    geometry_dense_every: int = 50
    geometry_dense_until: int = 1500
    geometry_sparse_every: int = 500
    geometry_probe_points: int = 4096
    geometry_render_enabled: bool = True
    geometry_render_basis: str = "first_last"
    geometry_render_fps: int = 8
    geometry_keep_snapshots: bool = True


class ConfigDefaultsHelpFormatter(argparse.HelpFormatter):
    """Show effective dataclass defaults while argparse still stores ``None``."""

    def _get_help_string(self, action):
        help_text = action.help or ""
        if hasattr(action, "effective_default"):
            default = _format_help_default(action.effective_default)
            separator = " " if help_text else ""
            return f"{help_text}{separator}[default: {default}]"
        return help_text


def _format_help_default(value: Any) -> str:
    if value is None:
        return "<unset>"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value == "":
        return '""'
    return str(value)


def _annotate_effective_defaults(parser) -> None:
    """Attach display-only defaults sourced from the configuration dataclasses."""
    defaults: dict[str, Any] = {}
    for config in (
        TrainConfig(),
        DataConfig(),
        TextVQVAEConfig(),
        CollapseControlConfig(),
        DiagnosticsConfig(),
    ):
        for field in dataclasses.fields(config):
            value = getattr(config, field.name)
            if field.name in defaults and defaults[field.name] != value:
                raise ValueError(f"Ambiguous CLI default for field {field.name!r}.")
            defaults[field.name] = value

    diagnostics = DiagnosticsConfig()
    defaults.update({
        "collapse_preset": None,
        "skip_initial_pca": not diagnostics.initial_pca_enabled,
        "strict_initial_pca": diagnostics.initial_pca_strict,
        "print_config": False,
    })
    for action in parser._actions:
        if action.dest in defaults:
            action.effective_default = defaults[action.dest]
            if action.help is None:
                action.help = f"{action.dest.replace('_', ' ').capitalize()}."


# ---------------------------------------------------------------------------
# argparse builder
# ---------------------------------------------------------------------------

def add_arguments(parser) -> None:
    """Add all CLI flags to *parser* with default=None (real defaults in dataclasses)."""
    parser.formatter_class = ConfigDefaultsHelpFormatter
    parser.add_argument(
        "--print-config",
        action="store_true",
        default=None,
        help="Print the resolved configuration as JSON to stdout and exit.",
    )
    # ---- training ----
    g = parser.add_argument_group("training")
    g.add_argument("--run-name", default=None, help="Output run name. Defaults to timestamp.")
    g.add_argument("--ablation", default=None, help="Free-form ablation label stored in config/logs.")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--epochs", type=int, default=None)
    g.add_argument("--batch-size", type=int, default=None)
    g.add_argument("--lr", type=float, default=None)
    g.add_argument("--weight-decay", type=float, default=None)
    g.add_argument("--grad-clip", type=float, default=None)
    g.add_argument("--eval-every", type=int, default=None)
    g.add_argument("--save-every", type=int, default=None)
    g.add_argument("--num-workers", type=int, default=None)
    g.add_argument(
        "--tokenizer", choices=["bpe", "byte"], default=None,
        help="Tokenizer to use.",
    )
    g.add_argument(
        "--tokenizer-path", default=None,
        help="Saved tokenizer.json for --tokenizer bpe.",
    )
    g.add_argument(
        "--codebook-init", choices=["random", "kmeans"], default=None,
        help="Codebook initialisation strategy.",
    )

    # ---- data ----
    g = parser.add_argument_group("data")
    g.add_argument(
        "--dataset", default=None,
        help="Hugging Face dataset name.",
    )
    g.add_argument("--dataset-config", default=None, help="Optional Hugging Face dataset config.")
    g.add_argument("--split", default=None, help="Dataset split.")
    g.add_argument("--text-field", default=None, help="Dataset/JSONL text field.")
    g.add_argument("--data-file", default=None, help="Optional local .txt or .jsonl file.")
    g.add_argument(
        "--cache-dir", default=None,
        help="Hugging Face dataset cache.",
    )
    g.add_argument("--streaming", action="store_true", default=None)
    g.add_argument("--max-train-samples", type=int, default=None)
    g.add_argument("--max-eval-samples", type=int, default=None)
    g.add_argument("--val-fraction", type=float, default=None)

    # ---- model ----
    g = parser.add_argument_group("model")
    g.add_argument("--max-seq-len", type=int, default=None)
    g.add_argument("--latent-slots", type=int, default=None)
    g.add_argument("--slot-pad-ratio-threshold", type=float, default=None)
    g.add_argument("--d-model", type=int, default=None)
    g.add_argument("--n-heads", type=int, default=None)
    g.add_argument("--encoder-layers", type=int, default=None)
    g.add_argument(
        "--encoder-type",
        choices=ENCODER_TYPES,
        default=None,
        help="Encoder position encoding.",
    )
    g.add_argument("--decoder-layers", type=int, default=None)
    g.add_argument(
        "--decoder-type", choices=DECODER_TYPES, default=None,
        help="Decoder backbone.",
    )
    g.add_argument("--memory-decoder-latent-layers", type=int, default=None)
    g.add_argument("--memory-decoder-output-layers", type=int, default=None)
    g.add_argument("--ffn-mult", type=int, default=None)
    g.add_argument("--dropout", type=float, default=None)
    g.add_argument("--codebook-size", type=int, default=None)
    g.add_argument("--commitment-beta", type=float, default=None)
    g.add_argument(
        "--l2-normalize-before-vq",
        dest="l2_normalize_before_vq",
        action="store_true",
        default=None,
        help="L2-normalize valid encoder latents immediately before vector quantization.",
    )
    g.add_argument(
        "--no-l2-normalize-before-vq",
        dest="l2_normalize_before_vq",
        action="store_false",
    )

    # ---- collapse control ----
    g = parser.add_argument_group("collapse control")
    g.add_argument(
        "--collapse-preset", choices=["none", "anti"], default=None,
        help="'anti' enables all common anti-collapse measures.",
    )
    g.add_argument("--use-ema-codebook", dest="use_ema_codebook", action="store_true", default=None)
    g.add_argument("--no-ema-codebook", dest="use_ema_codebook", action="store_false")
    g.add_argument("--ema-decay", type=float, default=None)
    g.add_argument("--ema-eps", type=float, default=None)
    g.add_argument("--entropy-weight", type=float, default=None)
    g.add_argument("--entropy-temperature", type=float, default=None)
    g.add_argument("--diversity-weight", type=float, default=None)
    g.add_argument("--code-dropout", type=float, default=None)
    g.add_argument("--stochastic-code-sampling", dest="stochastic_code_sampling", action="store_true", default=None)
    g.add_argument("--no-stochastic-code-sampling", dest="stochastic_code_sampling", action="store_false")
    g.add_argument("--sampling-temperature", type=float, default=None)
    g.add_argument("--sampling-topk", type=int, default=None)
    g.add_argument("--dead-code-reset-every", type=int, default=None)
    g.add_argument("--dead-code-reset-usage-threshold", type=float, default=None)
    g.add_argument("--normalize-latents", dest="normalize_latents", action="store_true", default=None)
    g.add_argument("--no-normalize-latents", dest="normalize_latents", action="store_false")
    g.add_argument("--commitment-beta-start", type=float, default=None)
    g.add_argument("--commitment-beta-warmup-steps", type=int, default=None)

    # ---- diagnostics ----
    g = parser.add_argument_group("diagnostics")
    g.add_argument(
        "--initial-pca-max-points", type=int, default=None,
        help="Maximum encoder latent vectors in the initialisation PCA plot.",
    )
    g.add_argument(
        "--initial-pca-fit-mode", choices=["balanced", "all"], default=None,
        help="Fit PCA with equal group sizes or all collected vectors.",
    )
    g.add_argument(
        "--skip-initial-pca", action="store_true", default=None,
        help="Do not generate the initialisation encoder/codebook PCA plot.",
    )
    g.add_argument(
        "--strict-initial-pca", action="store_true", default=None,
        help="Fail the run instead of warning if the initialisation PCA diagnostic fails.",
    )
    g.add_argument(
        "--geometry-snapshot-enabled", type=_parse_bool, default=None,
        help="Enable periodic raw geometry snapshots (true/false).",
    )
    g.add_argument("--geometry-dense-every", type=int, default=None)
    g.add_argument("--geometry-dense-until", type=int, default=None)
    g.add_argument("--geometry-sparse-every", type=int, default=None)
    g.add_argument("--geometry-probe-points", type=int, default=None)
    g.add_argument(
        "--geometry-render-enabled", type=_parse_bool, default=None,
        help="Render geometry plots and animation when training completes (true/false).",
    )
    g.add_argument(
        "--geometry-render-basis", choices=["t0", "first_last", "pooled"], default=None,
        help="PCA basis used by every animation frame.",
    )
    g.add_argument("--geometry-render-fps", type=int, default=None)
    g.add_argument(
        "--geometry-keep-snapshots", type=_parse_bool, default=None,
        help="Keep raw geometry NPZ snapshots after a successful render (true/false).",
    )
    _annotate_effective_defaults(parser)


def _override(obj, attrs: dict[str, Any]) -> None:
    """Apply non-None values from *attrs* onto dataclass *obj* in place."""
    for key, value in attrs.items():
        if value is not None:
            setattr(obj, key, value)


def _parse_bool(value: str) -> bool:
    value = value.lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def build_train_config(args) -> TrainConfig:
    """Resolve training CLI overrides before tokenizer construction."""
    config = TrainConfig()
    _override(config, {
        "run_name": getattr(args, "run_name", None),
        "seed": getattr(args, "seed", None),
        "epochs": getattr(args, "epochs", None),
        "batch_size": getattr(args, "batch_size", None),
        "lr": getattr(args, "lr", None),
        "weight_decay": getattr(args, "weight_decay", None),
        "grad_clip": getattr(args, "grad_clip", None),
        "eval_every": getattr(args, "eval_every", None),
        "save_every": getattr(args, "save_every", None),
        "num_workers": getattr(args, "num_workers", None),
        "tokenizer": getattr(args, "tokenizer", None),
        "tokenizer_path": getattr(args, "tokenizer_path", None),
        "codebook_init": getattr(args, "codebook_init", None),
        "ablation": getattr(args, "ablation", None),
    })
    if config.tokenizer == "bpe" and not config.tokenizer_path:
        raise ValueError("--tokenizer-path is required when --tokenizer bpe is selected.")
    return config


def build_diagnostics_config(args) -> DiagnosticsConfig:
    config = DiagnosticsConfig()
    if getattr(args, "skip_initial_pca", None):
        config.initial_pca_enabled = False
    _override(config, {
        "initial_pca_max_points": getattr(args, "initial_pca_max_points", None),
        "initial_pca_fit_mode": getattr(args, "initial_pca_fit_mode", None),
        "initial_pca_strict": getattr(args, "strict_initial_pca", None),
        "geometry_snapshot_enabled": getattr(args, "geometry_snapshot_enabled", None),
        "geometry_dense_every": getattr(args, "geometry_dense_every", None),
        "geometry_dense_until": getattr(args, "geometry_dense_until", None),
        "geometry_sparse_every": getattr(args, "geometry_sparse_every", None),
        "geometry_probe_points": getattr(args, "geometry_probe_points", None),
        "geometry_render_enabled": getattr(args, "geometry_render_enabled", None),
        "geometry_render_basis": getattr(args, "geometry_render_basis", None),
        "geometry_render_fps": getattr(args, "geometry_render_fps", None),
        "geometry_keep_snapshots": getattr(args, "geometry_keep_snapshots", None),
    })
    if config.geometry_dense_every < 1 or config.geometry_sparse_every < 1:
        raise ValueError("Geometry snapshot intervals must be positive.")
    if config.geometry_dense_until < 0 or config.geometry_probe_points < 1:
        raise ValueError("Geometry dense-until must be non-negative and probe-points positive.")
    if config.geometry_render_fps < 1:
        raise ValueError("Geometry render FPS must be positive.")
    if not config.geometry_snapshot_enabled:
        if getattr(args, "geometry_render_enabled", None):
            raise ValueError("Geometry rendering requires geometry snapshots.")
        config.geometry_render_enabled = False
    return config


def build_collapse_config(args) -> CollapseControlConfig:
    if getattr(args, "collapse_preset", None) == "anti":
        config = CollapseControlConfig(
            enabled=True,
            use_ema_codebook=True,
            ema_decay=0.99,
            ema_eps=1e-5,
            entropy_weight=0.05,
            entropy_temperature=1.0,
            diversity_weight=0.001,
            code_dropout=0.01,
            stochastic_code_sampling=True,
            sampling_temperature=0.5,
            sampling_topk=8,
            dead_code_reset_every=500,
            dead_code_reset_usage_threshold=1.0,
            normalize_latents=True,
            commitment_beta_start=0.05,
            commitment_beta_warmup_steps=2000,
        )
    else:
        config = CollapseControlConfig()

    _override(config, {
        "use_ema_codebook": getattr(args, "use_ema_codebook", None),
        "ema_decay": getattr(args, "ema_decay", None),
        "ema_eps": getattr(args, "ema_eps", None),
        "entropy_weight": getattr(args, "entropy_weight", None),
        "entropy_temperature": getattr(args, "entropy_temperature", None),
        "diversity_weight": getattr(args, "diversity_weight", None),
        "code_dropout": getattr(args, "code_dropout", None),
        "stochastic_code_sampling": getattr(args, "stochastic_code_sampling", None),
        "sampling_temperature": getattr(args, "sampling_temperature", None),
        "sampling_topk": getattr(args, "sampling_topk", None),
        "dead_code_reset_every": getattr(args, "dead_code_reset_every", None),
        "dead_code_reset_usage_threshold": getattr(args, "dead_code_reset_usage_threshold", None),
        "normalize_latents": getattr(args, "normalize_latents", None),
        "commitment_beta_start": getattr(args, "commitment_beta_start", None),
        "commitment_beta_warmup_steps": getattr(args, "commitment_beta_warmup_steps", None),
    })

    config.enabled = any([
        config.use_ema_codebook,
        config.entropy_weight > 0,
        config.diversity_weight > 0,
        config.code_dropout > 0,
        config.stochastic_code_sampling,
        config.dead_code_reset_every > 0,
        config.normalize_latents,
        config.commitment_beta_start is not None,
    ])
    return config


def build_configs(args, tokenizer, train_cfg: TrainConfig | None = None):
    """Build all config objects from parsed args, applying overrides onto dataclass defaults."""
    train_cfg = train_cfg or build_train_config(args)

    data_file = getattr(args, "data_file", None)
    data_cfg = DataConfig()
    if data_file:
        data_cfg.source = "file"
        data_cfg.dataset = None
        data_cfg.dataset_config = None
        data_cfg.split = None
        data_cfg.cache_dir = None
        data_cfg.streaming = None
        data_cfg.data_file = data_file
    else:
        _override(data_cfg, {
            "dataset": getattr(args, "dataset", None),
            "dataset_config": getattr(args, "dataset_config", None),
            "split": getattr(args, "split", None),
            "text_field": getattr(args, "text_field", None),
            "cache_dir": getattr(args, "cache_dir", None),
            "streaming": True if getattr(args, "streaming", None) else None,
        })
    _override(data_cfg, {
        "text_field": getattr(args, "text_field", None),
        "max_train_samples": getattr(args, "max_train_samples", None),
        "max_eval_samples": getattr(args, "max_eval_samples", None),
        "val_fraction": getattr(args, "val_fraction", None),
    })

    model_cfg = TextVQVAEConfig()
    model_cfg.vocab_size = tokenizer.vocab_size
    model_cfg.pad_token_id = tokenizer.pad_token_id
    _override(model_cfg, {
        "max_seq_len": getattr(args, "max_seq_len", None),
        "latent_slots": getattr(args, "latent_slots", None),
        "slot_pad_ratio_threshold": getattr(args, "slot_pad_ratio_threshold", None),
        "d_model": getattr(args, "d_model", None),
        "n_heads": getattr(args, "n_heads", None),
        "encoder_layers": getattr(args, "encoder_layers", None),
        "encoder_type": getattr(args, "encoder_type", None),
        "decoder_layers": getattr(args, "decoder_layers", None),
        "decoder_type": getattr(args, "decoder_type", None),
        "memory_decoder_latent_layers": getattr(args, "memory_decoder_latent_layers", None),
        "memory_decoder_output_layers": getattr(args, "memory_decoder_output_layers", None),
        "ffn_mult": getattr(args, "ffn_mult", None),
        "dropout": getattr(args, "dropout", None),
        "codebook_size": getattr(args, "codebook_size", None),
        "commitment_beta": getattr(args, "commitment_beta", None),
        "l2_normalize_before_vq": getattr(args, "l2_normalize_before_vq", None),
    })

    collapse_cfg = build_collapse_config(args)
    return train_cfg, data_cfg, model_cfg, collapse_cfg


# ---------------------------------------------------------------------------
# Config payload helpers
# ---------------------------------------------------------------------------

def _git_info() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {"git_commit": None, "git_dirty": None}


def build_config_payload(
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    model_cfg: TextVQVAEConfig,
    collapse_cfg: CollapseControlConfig,
    run_dir,
    device,
    initial_pca_enabled: bool,
    initial_pca_max_points: int,
    initial_pca_fit_mode: str,
    initial_pca_strict: bool,
    codebook_init_method: str,
    geometry_config: DiagnosticsConfig | None = None,
) -> dict[str, Any]:
    geometry_config = geometry_config or DiagnosticsConfig()
    return {
        "config_version": 1,
        **_git_info(),
        "train": asdict(train_cfg),
        "data": asdict(data_cfg),
        "model": model_cfg.to_dict(),
        "collapse_control": collapse_cfg.to_dict(),
        "device": str(device),
        "output_dir": str(run_dir),
        "codebook_initialization": {
            "method": codebook_init_method,
            "status": "completed" if codebook_init_method == "random" else "pending",
        },
        "diagnostics": {
            "initial_pca": {
                "enabled": initial_pca_enabled,
                "max_encoder_points": initial_pca_max_points,
                "fit_mode": initial_pca_fit_mode,
                "strict": initial_pca_strict,
                "status": "disabled" if not initial_pca_enabled else "pending",
            },
            "geometry": {
                "snapshot_enabled": geometry_config.geometry_snapshot_enabled,
                "dense_every": geometry_config.geometry_dense_every,
                "dense_until": geometry_config.geometry_dense_until,
                "sparse_every": geometry_config.geometry_sparse_every,
                "probe_points": geometry_config.geometry_probe_points,
                "render_enabled": geometry_config.geometry_render_enabled,
                "render_basis": geometry_config.geometry_render_basis,
                "render_fps": geometry_config.geometry_render_fps,
                "keep_snapshots": geometry_config.geometry_keep_snapshots,
                "render_status": "pending" if geometry_config.geometry_render_enabled else "disabled",
            },
        },
    }


# ---------------------------------------------------------------------------
# Load a historical run's config.json back into dataclasses
# ---------------------------------------------------------------------------

def load_run_config(path: str | Path) -> tuple[TrainConfig, DataConfig, TextVQVAEConfig, CollapseControlConfig]:
    """Reconstruct config objects from a saved config.json.

    Missing keys are filled from current dataclass defaults. Missing and
    unknown keys both emit warnings so compatibility decisions remain visible.
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    def _fill(default_obj, saved: Any, label: str):
        if not isinstance(saved, dict):
            warnings.warn(
                f"[load_run_config] {label}: expected an object; using all defaults",
                stacklevel=2,
            )
            saved = {}
        defaults = dataclasses.asdict(default_obj)
        missing = sorted(defaults.keys() - saved.keys())
        unknown = sorted(saved.keys() - defaults.keys())
        if missing:
            warnings.warn(
                f"[load_run_config] {label}: filling missing keys with defaults: {missing}",
                stacklevel=2,
            )
        if unknown:
            warnings.warn(
                f"[load_run_config] {label}: ignoring unknown keys: {unknown}",
                stacklevel=2,
            )
        kwargs = {key: saved.get(key, value) for key, value in defaults.items()}
        return type(default_obj)(**kwargs)

    train_cfg = _fill(TrainConfig(), payload.get("train", {}), "TrainConfig")
    data_cfg = _fill(DataConfig(), payload.get("data", {}), "DataConfig")
    model_cfg = _fill(TextVQVAEConfig(), payload.get("model", {}), "TextVQVAEConfig")
    collapse_cfg = _fill(
        CollapseControlConfig(), payload.get("collapse_control", {}), "CollapseControlConfig"
    )

    return train_cfg, data_cfg, model_cfg, collapse_cfg
