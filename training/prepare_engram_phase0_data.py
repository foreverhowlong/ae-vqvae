"""Prepare a frozen GPT-2 token stream for the Engram Phase-0 experiment.

The default source is the pinned FineWeb-Edu sample-10BT revision.  A single
deterministically shuffled stream is split by whole-document assignment:
validation documents are consumed first, the final partial validation document
is discarded, and subsequent documents form training.  The emitted files and
their SHA256 digests are then the source of truth shared by all four runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from datasets import load_dataset
from tokenizers import Regex, Tokenizer, normalizers

from common.text_data import get_hf_token


DEFAULT_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_DATASET_CONFIG = "sample-10BT"
DEFAULT_DATASET_REVISION = "05c1931294b0d1379055d1f802d369f2c3bb2f4b"
DEFAULT_TOKENIZER = "openai-community/gpt2"
DEFAULT_TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
GPT2_VOCAB_SIZE = 50_257
GPT2_EOT_ID = 50_256


def canonical_normalizer():
    """Normalizer copied from DeepSeek-AI/Engram's CompressedTokenizer."""
    sentinel = "\uE000"
    return normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )


def build_canonical_projection(tokenizer: Tokenizer) -> np.ndarray:
    """Build the official raw-ID -> normalized-equivalence-class projection."""
    vocabulary_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if vocabulary_size != GPT2_VOCAB_SIZE:
        raise ValueError(
            f"Expected GPT-2 vocabulary size {GPT2_VOCAB_SIZE}, got {vocabulary_size}."
        )
    normalize = canonical_normalizer()
    key_to_id: dict[str, int] = {}
    projection = np.empty(vocabulary_size, dtype=np.int64)
    for token_id in range(vocabulary_size):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        raw_token = tokenizer.id_to_token(token_id)
        if raw_token is None:
            raise RuntimeError(f"Tokenizer has no token for ID {token_id}.")
        key = raw_token if "�" in text else normalize.normalize_str(text)
        if not key:
            key = text
        projection[token_id] = key_to_id.setdefault(key, len(key_to_id))
    return projection


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document_fingerprint(document_id: str) -> bytes:
    return hashlib.sha256(document_id.encode("utf-8", errors="replace")).digest()


def local_documents(path: Path, text_field: str) -> Iterator[dict]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                yield {
                    "text": str(row[text_field]),
                    "id": str(row.get("id", f"line-{line_number}")),
                }
    else:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                yield {"text": line, "id": f"line-{line_number}"}


def write_token_stream(
    documents: Iterator[dict],
    tokenizer: Tokenizer,
    output_path: Path,
    prediction_tokens: int,
    *,
    text_field: str = "text",
) -> dict[str, int | str]:
    """Write prediction_tokens + 1 uint16 IDs, assigning only whole documents."""
    required = prediction_tokens + 1
    written = 0
    document_count = 0
    document_digest = hashlib.sha256()
    with output_path.open("xb") as handle:
        while written < required:
            try:
                document = next(documents)
            except StopIteration as error:
                raise ValueError(
                    f"Dataset ended after {written:,}/{required:,} required tokens."
                ) from error
            text = str(document[text_field])
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            ids.append(GPT2_EOT_ID)
            take = min(len(ids), required - written)
            if take:
                np.asarray(ids[:take], dtype=np.uint16).tofile(handle)
                written += take
            document_count += 1
            document_digest.update(
                _document_fingerprint(str(document.get("id", document_count)))
            )
    return {
        "stored_tokens": written,
        "prediction_tokens": prediction_tokens,
        "documents_assigned": document_count,
        "document_id_digest": document_digest.hexdigest(),
        "sha256": _sha256(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--shuffle-seed", type=int, default=12_345)
    parser.add_argument("--shuffle-buffer", type=int, default=100_000)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--train-prediction-tokens", type=int, default=2_500_000_000)
    parser.add_argument("--validation-prediction-tokens", type=int, default=10_000_000)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--data-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_prediction_tokens < 1 or args.validation_prediction_tokens < 1:
        raise ValueError("Token counts must be positive.")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision, token=get_hf_token()
    )
    if tokenizer.get_vocab_size(with_added_tokens=True) != GPT2_VOCAB_SIZE:
        raise ValueError("Phase-0 requires the unmodified 50,257-entry GPT-2 tokenizer.")
    tokenizer.save(str(output_dir / "tokenizer.json"))
    projection = build_canonical_projection(tokenizer)
    np.save(output_dir / "canonical_projection.npy", projection)

    if args.data_file is not None:
        documents: Iterable[dict] = local_documents(args.data_file, args.text_field)
        source = {"type": "local", "path": str(args.data_file.expanduser().resolve())}
    else:
        dataset = load_dataset(
            args.dataset,
            name=args.dataset_config,
            split=args.split,
            revision=args.dataset_revision,
            streaming=True,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            token=get_hf_token(),
        )
        documents = dataset.shuffle(
            seed=args.shuffle_seed, buffer_size=args.shuffle_buffer
        )
        source = {
            "type": "huggingface",
            "dataset": args.dataset,
            "config": args.dataset_config,
            "revision": args.dataset_revision,
            "split": args.split,
            "shuffle_seed": args.shuffle_seed,
            "shuffle_buffer": args.shuffle_buffer,
        }
    iterator = iter(documents)
    validation = write_token_stream(
        iterator,
        tokenizer,
        output_dir / "validation.bin",
        args.validation_prediction_tokens,
        text_field=args.text_field,
    )
    train = write_token_stream(
        iterator,
        tokenizer,
        output_dir / "train.bin",
        args.train_prediction_tokens,
        text_field=args.text_field,
    )
    metadata = {
        "format_version": 1,
        "source": source,
        "split_policy": (
            "validation receives the first whole documents from the deterministic stream; "
            "the unused suffix of its last document is discarded; training uses later documents"
        ),
        "tokenizer": {
            "name": args.tokenizer,
            "revision": args.tokenizer_revision,
            "vocab_size": GPT2_VOCAB_SIZE,
            "eot_token_id": GPT2_EOT_ID,
            "tokenizer_json_sha256": _sha256(output_dir / "tokenizer.json"),
            "canonical_projection_sha256": _sha256(
                output_dir / "canonical_projection.npy"
            ),
            "canonical_vocab_size": int(projection.max()) + 1,
            "canonicalization": "DeepSeek-AI/Engram CompressedTokenizer logic",
        },
        "validation": validation,
        "train": train,
    }
    (output_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
