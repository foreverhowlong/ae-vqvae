"""Export a trained GQ-VAE decoder dictionary as a lossless tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.gqvae_config import GQVAEConfig
from common.learned_tokenizer import LearnedByteFallbackTokenizer
from models.gqvae import GQVAE


def load_gqvae_checkpoint(path: str | Path, device: torch.device) -> GQVAE:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config_payload = checkpoint.get("model_config")
    if not isinstance(config_payload, dict):
        raise ValueError("GQ-VAE checkpoint is missing model_config.")
    model = GQVAE(GQVAEConfig(**config_payload)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def export_tokenizer(model: GQVAE, output: str | Path) -> dict[str, object]:
    tokenizer = LearnedByteFallbackTokenizer(model.decoded_codebook())
    tokenizer.save(output)
    unique_tokens = len({token for token in tokenizer.learned_tokens if token})
    return {
        "output": str(Path(output)),
        "codebook_size": tokenizer.codebook_size,
        "unique_nonempty_learned_tokens": unique_tokens,
        "vocab_size_with_fallback": tokenizer.vocab_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    result = export_tokenizer(
        load_gqvae_checkpoint(args.checkpoint, device),
        args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

