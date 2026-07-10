"""Byte-level text datasets for language compression experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import torch
from dotenv import dotenv_values
from torch.utils.data import Dataset

from . import ROOT


BYTE_EOS = 256
BYTE_PAD = 257
BYTE_VOCAB_SIZE = 258
DEFAULT_HF_DATASET_CACHE = ROOT / "data" / "huggingface"
HF_TOKEN_ENV_VAR = "HF_TOKEN"
DEFAULT_DOTENV_PATH = ROOT / ".env"


def get_hf_token(dotenv_path: Path = DEFAULT_DOTENV_PATH) -> str | None:
    """Read HF_TOKEN from the environment, falling back to the ignored root .env."""
    environment_token = os.environ.get(HF_TOKEN_ENV_VAR)
    if environment_token:
        return environment_token

    if not dotenv_path.is_file():
        return None
    dotenv_token = dotenv_values(dotenv_path).get(HF_TOKEN_ENV_VAR)
    return str(dotenv_token) if dotenv_token else None


class ByteTokenizer:
    eos_token_id = BYTE_EOS
    pad_token_id = BYTE_PAD
    vocab_size = BYTE_VOCAB_SIZE

    def encode(self, text: str, max_length: int):
        byte_ids = list(text.encode("utf-8", errors="ignore"))
        byte_ids = byte_ids[: max_length - 1] + [self.eos_token_id]
        attention_mask = [1] * len(byte_ids)

        pad_len = max_length - len(byte_ids)
        if pad_len > 0:
            byte_ids.extend([self.pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)

        return byte_ids, attention_mask

    def decode(self, ids: Iterable[int]):
        byte_values = []
        for idx in ids:
            idx = int(idx)
            if idx == self.eos_token_id:
                break
            if 0 <= idx < 256:
                byte_values.append(idx)
        return bytes(byte_values).decode("utf-8", errors="replace")


class ByteTextDataset(Dataset):
    def __init__(self, texts: list[str], max_seq_len: int):
        self.texts = texts
        self.max_seq_len = max_seq_len
        self.tokenizer = ByteTokenizer()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids, attention_mask = self.tokenizer.encode(self.texts[idx], self.max_seq_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def read_texts_from_file(path: Path, text_field: str = "text", max_samples: int | None = None):
    return list(iter_texts_from_file(path, text_field=text_field, max_samples=max_samples))


def iter_texts_from_file(
    path: Path,
    text_field: str = "text",
    max_samples: int | None = None,
):
    """Yield texts from a local .txt or .jsonl file without materializing them."""
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        yielded = 0
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if suffix == ".jsonl":
                item = json.loads(line)
                text = str(item[text_field])
            else:
                text = line
            yield text
            yielded += 1
            if max_samples is not None and yielded >= max_samples:
                break


def load_tinystories_texts(
    split: str,
    max_samples: int | None,
    cache_dir: str | None = None,
    streaming: bool = False,
):
    return list(
        iter_tinystories_texts(
            split=split,
            max_samples=max_samples,
            cache_dir=cache_dir,
            streaming=streaming,
        )
    )


def iter_tinystories_texts(
    split: str = "train",
    max_samples: int | None = None,
    cache_dir: str | None = None,
    streaming: bool = False,
):
    """Yield TinyStories texts lazily from Hugging Face Datasets."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "TinyStories loading requires the optional `datasets` package. "
            "Install it or pass --data-file with a local .txt/.jsonl file."
        ) from exc

    resolved_cache_dir = Path(cache_dir).expanduser() if cache_dir else DEFAULT_HF_DATASET_CACHE
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "roneneldan/TinyStories",
        split=split,
        cache_dir=str(resolved_cache_dir),
        streaming=streaming,
        token=get_hf_token(),
    )

    for index, row in enumerate(dataset):
        yield str(row["text"])
        if max_samples is not None and index + 1 >= max_samples:
            break


def build_text_dataset(
    *,
    max_seq_len: int,
    max_samples: int | None,
    data_file: str | None = None,
    split: str = "train",
    cache_dir: str | None = None,
    streaming: bool = False,
):
    if data_file:
        texts = read_texts_from_file(Path(data_file), max_samples=max_samples)
    else:
        texts = load_tinystories_texts(
            split=split,
            max_samples=max_samples,
            cache_dir=cache_dir,
            streaming=streaming,
        )

    if not texts:
        raise ValueError("No texts were loaded for the language compression experiment.")

    return ByteTextDataset(texts, max_seq_len=max_seq_len)
