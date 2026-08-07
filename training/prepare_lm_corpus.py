"""Prepare identical raw-text splits for BPE, GQ-VAE, or fixed VQ token streams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from common.learned_tokenizer import LearnedByteFallbackTokenizer
from common.text_data import BPETokenizer, iter_hf_texts, iter_texts_from_file
from common.vq_block_tokenizer import VQBlockTokenizer


class BPEAdapter:
    def __init__(self, path: Path):
        self.tokenizer = BPETokenizer(path)
        self.vocab_size = self.tokenizer.vocab_size
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        bos = self.tokenizer.tokenizer.token_to_id("<bos>")
        if bos is None:
            raise ValueError("BPE tokenizer is missing <bos>.")
        self.bos_token_id = bos

    def encode(self, text: str) -> tuple[list[int], dict[str, int | bool]]:
        ids = [self.bos_token_id, *self.tokenizer.tokenizer.encode(
            text,
            add_special_tokens=False,
        ).ids]
        ids.append(self.eos_token_id)
        return ids, {
            "used_vq": False,
            "fallback_bytes": 0,
            "raw_bytes": len(text.encode("utf-8")),
        }


class LearnedAdapter:
    def __init__(self, path: Path):
        self.tokenizer = LearnedByteFallbackTokenizer.load(path)
        self.vocab_size = self.tokenizer.vocab_size
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id

    def encode(self, text: str) -> tuple[list[int], dict[str, int | bool]]:
        raw = text.encode("utf-8")
        ids = self.tokenizer.encode_bytes(raw)
        fallback = sum(
            self.tokenizer.byte_fallback_offset
            <= token
            < self.tokenizer.byte_fallback_offset + 256
            for token in ids
        )
        if self.tokenizer.decode_bytes(ids) != raw:
            raise ValueError("Learned tokenizer failed its lossless round trip.")
        return ids, {
            "used_vq": True,
            "fallback_bytes": fallback,
            "raw_bytes": len(raw),
        }


def _texts(
    *,
    data_file: Path | None,
    dataset: str,
    dataset_config: str | None,
    split: str,
    text_field: str,
    cache_dir: str,
    max_documents: int | None,
) -> Iterable[str]:
    if data_file is not None:
        return iter_texts_from_file(
            data_file,
            text_field=text_field,
            max_samples=max_documents,
        )
    return iter_hf_texts(
        dataset_name=dataset,
        dataset_config=dataset_config,
        split=split,
        text_field=text_field,
        cache_dir=cache_dir,
        max_samples=max_documents,
    )


def write_split(texts: Iterable[str], tokenizer, output_dir: Path, split: str):
    token_path = output_dir / f"{split}.bin"
    index_path = output_dir / f"{split}.idx"
    byte_count_path = output_dir / f"{split}.bytes"
    documents = 0
    raw_bytes = 0
    token_count = 0
    fallback_bytes = 0
    vq_documents = 0
    with (
        token_path.open("wb") as token_file,
        index_path.open("wb") as index_file,
        byte_count_path.open("wb") as byte_count_file,
    ):
        np.asarray([0], dtype=np.uint64).tofile(index_file)
        for text in texts:
            ids, stats = tokenizer.encode(text)
            if not ids:
                continue
            if max(ids) >= 2**16:
                raise ValueError("Corpus token ids exceed uint16 capacity.")
            np.asarray(ids, dtype=np.uint16).tofile(token_file)
            token_count += len(ids)
            np.asarray([token_count], dtype=np.uint64).tofile(index_file)
            documents += 1
            raw_bytes += int(stats["raw_bytes"])
            np.asarray([int(stats["raw_bytes"])], dtype=np.uint64).tofile(
                byte_count_file
            )
            fallback_bytes += int(stats.get("fallback_bytes", 0))
            vq_documents += int(bool(stats.get("used_vq", False)))
    return {
        "documents": documents,
        "raw_utf8_bytes": raw_bytes,
        "tokens": token_count,
        "bytes_per_token": raw_bytes / max(token_count, 1),
        "fallback_bytes": fallback_bytes,
        "fallback_byte_fraction": fallback_bytes / max(raw_bytes, 1),
        "vq_documents": vq_documents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", help="Sequence-runner label; output-dir remains authoritative.")
    parser.add_argument("--ablation", help="Sequence-runner label stored in command history.")
    parser.add_argument("--tokenizer", choices=("bpe", "gqvae", "vqvae"), required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--vq-checkpoint", type=Path)
    parser.add_argument("--vq-config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--dataset-config")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--cache-dir", default="data/huggingface")
    parser.add_argument("--train-data-file", type=Path)
    parser.add_argument("--validation-data-file", type=Path)
    parser.add_argument("--max-train-documents", type=int)
    parser.add_argument("--max-validation-documents", type=int)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.tokenizer == "bpe":
        if args.tokenizer_path is None:
            raise ValueError("--tokenizer-path is required for BPE.")
        tokenizer = BPEAdapter(args.tokenizer_path)
    elif args.tokenizer == "gqvae":
        if args.tokenizer_path is None:
            raise ValueError("--tokenizer-path is required for GQ-VAE.")
        tokenizer = LearnedAdapter(args.tokenizer_path)
    else:
        if args.vq_checkpoint is None or args.vq_config is None:
            raise ValueError("--vq-checkpoint and --vq-config are required for VQ-VAE.")
        tokenizer = VQBlockTokenizer.load(
            args.vq_checkpoint,
            args.vq_config,
            torch.device(args.device),
        )

    train_texts = _texts(
        data_file=args.train_data_file,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.train_split,
        text_field=args.text_field,
        cache_dir=args.cache_dir,
        max_documents=args.max_train_documents,
    )
    validation_texts = _texts(
        data_file=args.validation_data_file,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.validation_split,
        text_field=args.text_field,
        cache_dir=args.cache_dir,
        max_documents=args.max_validation_documents,
    )
    metadata = {
        "format": "nanogpt-token-corpus-v1",
        "tokenizer": args.tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "train": write_split(train_texts, tokenizer, args.output_dir, "train"),
        "validation": write_split(
            validation_texts,
            tokenizer,
            args.output_dir,
            "validation",
        ),
    }
    (args.output_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
