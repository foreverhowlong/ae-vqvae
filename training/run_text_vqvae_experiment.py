"""Train a non-autoregressive VQ-VAE on tokenized text sequences."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch

from common import ROOT, enable_tf32, get_device
from common.text_data import BPETokenizer, ByteTokenizer, build_text_dataset
from common.tracking import init_wandb
from models.text_vqvae import TextVQVAE, count_parameters
from training.text_vqvae.codebook_init import initialize_codebook_kmeans
from training.text_vqvae.config import add_arguments, build_config_payload, build_configs
from training.text_vqvae.loop import make_loader, run, split_dataset
from training.text_vqvae.reporting import atomic_json_dump


def _load_tokenizer(name: str, path: str | None):
    if name == "byte":
        return ByteTokenizer(), None
    if not path:
        raise ValueError("--tokenizer-path is required when --tokenizer bpe is selected.")
    tokenizer = BPETokenizer(path)
    return tokenizer, str(tokenizer.path)


def _make_run_dir(run_name: str | None):
    if run_name is None:
        run_name = time.strftime("text_vqvae_%Y%m%d_%H%M%S")
    run_dir = ROOT / "outputs" / "text_vqvae" / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    for child in ["checkpoints", "plots", "samples"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir, run_name


def main():
    parser = argparse.ArgumentParser(description="Text VQ-VAE experiment")
    add_arguments(parser)
    args = parser.parse_args()

    tokenizer, tokenizer_path = _load_tokenizer(
        args.tokenizer or "bpe", args.tokenizer_path
    )
    run_dir, run_name = _make_run_dir(args.run_name)
    device = get_device()
    enable_tf32(device)

    train_cfg, data_cfg, model_cfg, collapse_cfg = build_configs(args, tokenizer)
    train_cfg.run_name = run_name
    train_cfg.tokenizer_path = tokenizer_path

    torch.manual_seed(train_cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_cfg.seed)

    dataset = build_text_dataset(
        max_seq_len=model_cfg.max_seq_len,
        max_samples=data_cfg.max_train_samples,
        data_file=data_cfg.data_file,
        dataset_name=data_cfg.dataset,
        dataset_config=data_cfg.dataset_config,
        split=data_cfg.split or "train",
        text_field=data_cfg.text_field,
        cache_dir=data_cfg.cache_dir,
        streaming=bool(data_cfg.streaming),
        tokenizer=tokenizer,
    )
    train_dataset, val_dataset = split_dataset(
        dataset,
        val_fraction=data_cfg.val_fraction,
        seed=train_cfg.seed,
        max_eval_samples=data_cfg.max_eval_samples,
    )
    train_loader = make_loader(train_dataset, train_cfg.batch_size, shuffle=True, device=device, num_workers=train_cfg.num_workers)
    val_loader = make_loader(val_dataset, train_cfg.batch_size, shuffle=False, device=device, num_workers=train_cfg.num_workers)

    model = TextVQVAE(model_cfg, collapse_config=collapse_cfg).to(device)

    skip_pca = getattr(args, "skip_initial_pca", False)
    pca_max_points = getattr(args, "initial_pca_max_points", None) or 8192
    pca_fit_mode = getattr(args, "initial_pca_fit_mode", None) or "balanced"
    pca_strict = getattr(args, "strict_initial_pca", False)

    config_payload = build_config_payload(
        train_cfg, data_cfg, model_cfg, collapse_cfg,
        run_dir=run_dir,
        device=device,
        initial_pca_enabled=not skip_pca,
        initial_pca_max_points=pca_max_points,
        initial_pca_fit_mode=pca_fit_mode,
        initial_pca_strict=pca_strict,
        codebook_init_method=train_cfg.codebook_init,
    )
    atomic_json_dump(config_payload, run_dir / "config.json")

    if train_cfg.codebook_init == "kmeans":
        print("[Codebook init] Running encoder pass and fitting MiniBatch K-Means...")
        init_result = initialize_codebook_kmeans(model, train_loader, device, seed=train_cfg.seed)
        config_payload["codebook_initialization"].update({"status": "completed", **init_result})
        atomic_json_dump(config_payload, run_dir / "config.json")
        print(f"[Codebook init] K-means completed from {init_result['encoder_vectors']:,} encoder vectors")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    param_count = count_parameters(model)
    config_payload["parameter_count"] = param_count
    config_payload["compression"] = {
        "tokens_per_example": model_cfg.max_seq_len,
        "latent_slots": model_cfg.latent_slots,
        "nominal_token_to_latent_ratio": model_cfg.max_seq_len / model_cfg.latent_slots,
    }
    config_payload["data"]["train_examples"] = len(train_dataset)
    config_payload["data"]["eval_examples"] = len(val_dataset)
    atomic_json_dump(config_payload, run_dir / "config.json")

    print(f"[Run] {run_name}")
    print(f"[Device] {device}")
    print(f"[Params] {param_count:,}")
    print(f"[Tokenizer] {train_cfg.tokenizer} vocab={tokenizer.vocab_size} pad={tokenizer.pad_token_id}")
    print(f"[Data] train={len(train_dataset)} eval={len(val_dataset)}")
    print(f"[Output] {run_dir}")

    tracker = init_wandb(run_name, group="text-vqvae", tags=["text", "vqvae"], config=config_payload)

    run(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        model_config=model_cfg,
        collapse_config=collapse_cfg,
        run_dir=run_dir,
        run_name=run_name,
        tokenizer=tokenizer,
        device=device,
        config_payload=config_payload,
        tracker=tracker,
        initial_pca_opts={
            "enabled": not skip_pca,
            "max_points": pca_max_points,
            "fit_mode": pca_fit_mode,
            "strict": pca_strict,
        },
    )


if __name__ == "__main__":
    main()
