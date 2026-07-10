"""Train a byte-level BPE tokenizer on TinyStories."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from common import ROOT
from common.text_data import (
    DEFAULT_HF_DATASET_CACHE,
    iter_texts_from_file,
    iter_tinystories_texts,
)


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
DEFAULT_VOCAB_SIZE = 8192
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "tokenizers" / "tinystories_bpe_8k"
VALIDATION_TEXT = "Once upon a time, a tiny dragon said: 你好!\nThe end."


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer on TinyStories."
    )
    parser.add_argument("--split", default="train", help="TinyStories dataset split.")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional sample limit for smoke tests or subset training.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        default=str(DEFAULT_HF_DATASET_CACHE),
        help=f"Hugging Face dataset cache (default: {DEFAULT_HF_DATASET_CACHE}).",
    )
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stream TinyStories without building the reusable local Arrow cache (default: false).",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Optional local .txt or .jsonl file instead of TinyStories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Tokenizer output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def validate_args(args):
    if args.vocab_size < len(SPECIAL_TOKENS) + 256:
        raise ValueError(
            f"--vocab-size must be at least {len(SPECIAL_TOKENS) + 256} "
            "to hold the byte alphabet and special tokens."
        )
    if args.min_frequency < 1:
        raise ValueError("--min-frequency must be at least 1.")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1 when provided.")
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists():
        if not args.output_dir.is_dir():
            raise FileExistsError(f"Output path is not a directory: {args.output_dir}")
        if any(args.output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {args.output_dir}")


def build_text_iterator(args) -> Iterable[str]:
    if args.data_file:
        return iter_texts_from_file(Path(args.data_file), max_samples=args.max_samples)
    return iter_tinystories_texts(
        split=args.split,
        max_samples=args.max_samples,
        cache_dir=args.dataset_cache_dir,
        streaming=args.streaming,
    )


def count_texts(texts: Iterable[str], counter: dict[str, int]) -> Iterator[str]:
    for text in texts:
        counter["samples"] += 1
        yield text


def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train_tokenizer(texts: Iterable[str], vocab_size: int, min_frequency: int) -> Tokenizer:
    tokenizer = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def validate_tokenizer(tokenizer: Tokenizer, requested_vocab_size: int):
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > requested_vocab_size:
        raise RuntimeError(
            f"Trained vocabulary has {vocab_size} entries, exceeding {requested_vocab_size}."
        )

    special_token_ids = {
        token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS
    }
    expected_ids = dict(zip(SPECIAL_TOKENS, range(len(SPECIAL_TOKENS))))
    if special_token_ids != expected_ids:
        raise RuntimeError(
            f"Unexpected special token IDs: {special_token_ids}; expected {expected_ids}."
        )

    encoding = tokenizer.encode(VALIDATION_TEXT)
    if tokenizer.token_to_id("<unk>") in encoding.ids:
        raise RuntimeError("Validation text unexpectedly produced an <unk> token.")
    decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
    if decoded != VALIDATION_TEXT:
        raise RuntimeError(
            f"Tokenizer round trip failed: expected {VALIDATION_TEXT!r}, got {decoded!r}."
        )
    return special_token_ids, encoding


def save_tokenizer(tokenizer: Tokenizer, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    saved_model_files = tokenizer.model.save(str(output_dir))
    expected_files = {"vocab.json", "merges.txt"}
    actual_files = {Path(path).name for path in saved_model_files}
    if not expected_files.issubset(actual_files):
        raise RuntimeError(f"BPE model did not save the expected files: {actual_files}")


def main():
    args = parse_args()
    validate_args(args)

    counter = {"samples": 0}
    tokenizer = train_tokenizer(
        count_texts(build_text_iterator(args), counter),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    if counter["samples"] == 0:
        raise ValueError("No texts were loaded for tokenizer training.")

    special_token_ids, validation_encoding = validate_tokenizer(
        tokenizer, requested_vocab_size=args.vocab_size
    )
    output_dir = args.output_dir
    save_tokenizer(tokenizer, output_dir)

    config = {
        "dataset": {
            "name": "local" if args.data_file else "roneneldan/TinyStories",
            "data_file": args.data_file,
            "split": args.split,
            "streaming": args.streaming if not args.data_file else None,
            "cache_dir": args.dataset_cache_dir,
            "max_samples": args.max_samples,
            "training_samples": counter["samples"],
        },
        "tokenizer": {
            "type": "byte-level BPE",
            "requested_vocab_size": args.vocab_size,
            "actual_vocab_size": tokenizer.get_vocab_size(),
            "min_frequency": args.min_frequency,
            "special_token_ids": special_token_ids,
            "normalizer": None,
            "automatic_bos_eos": False,
        },
        "validation": {
            "text": VALIDATION_TEXT,
            "token_ids": validation_encoding.ids,
            "tokens": validation_encoding.tokens,
            "round_trip_ok": True,
        },
        "output_dir": str(output_dir),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"[Data] samples={counter['samples']}")
    print(f"[Vocab] size={tokenizer.get_vocab_size()}")
    print(f"[Special tokens] {special_token_ids}")
    print(f"[Validation tokens] {validation_encoding.tokens}")
    print(f"[Validation decoded] {tokenizer.decode(validation_encoding.ids)!r}")
    print(f"[Output] {output_dir}")


if __name__ == "__main__":
    main()
