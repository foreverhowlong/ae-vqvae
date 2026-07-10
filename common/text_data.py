"""Tokenizer adapters and text datasets for language compression experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Protocol

import torch
from dotenv import dotenv_values
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from . import ROOT


BYTE_EOS = 256
BYTE_PAD = 257
BYTE_VOCAB_SIZE = 258
DEFAULT_HF_DATASET_CACHE = ROOT / "data" / "huggingface"
DEFAULT_TEXT_DATASET = "roneneldan/TinyStories"
HF_TOKEN_ENV_VAR = "HF_TOKEN"
DEFAULT_DOTENV_PATH = ROOT / ".env"
DEFAULT_BPE_TOKENIZER_PATH = (
    ROOT / "outputs" / "tokenizers" / "tinystories_bpe_8k" / "tokenizer.json"
)


def get_hf_token(dotenv_path: Path = DEFAULT_DOTENV_PATH) -> str | None:
    """Read HF_TOKEN from the environment, falling back to the ignored root .env."""
    environment_token = os.environ.get(HF_TOKEN_ENV_VAR)
    if environment_token:
        return environment_token

    if not dotenv_path.is_file():
        return None
    dotenv_token = dotenv_values(dotenv_path).get(HF_TOKEN_ENV_VAR)
    return str(dotenv_token) if dotenv_token else None


class TextTokenizer(Protocol):
    vocab_size: int
    pad_token_id: int
    eos_token_id: int

    def encode(self, text: str, max_length: int): ...

    def decode(self, ids: Iterable[int]): ...


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


class BPETokenizer:
    """Adapter for a saved Hugging Face `tokenizers` BPE tokenizer."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"BPE tokenizer file does not exist: {self.path}")

        self.tokenizer = Tokenizer.from_file(str(self.path))
        self.pad_token_id = self._required_token_id("<pad>")
        self.eos_token_id = self._required_token_id("<eos>")
        self.vocab_size = self.tokenizer.get_vocab_size()

    def _required_token_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"BPE tokenizer is missing required special token {token!r}: {self.path}")
        return token_id

    def encode(self, text: str, max_length: int):
        token_ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        token_ids = token_ids[: max_length - 1] + [self.eos_token_id]
        attention_mask = [1] * len(token_ids)

        pad_len = max_length - len(token_ids)
        if pad_len > 0:
            token_ids.extend([self.pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)

        return token_ids, attention_mask

    def decode(self, ids: Iterable[int]):
        content_ids = []
        for idx in ids:
            idx = int(idx)
            if idx == self.eos_token_id:
                break
            if idx != self.pad_token_id:
                content_ids.append(idx)
        return self.tokenizer.decode(content_ids, skip_special_tokens=True)


class TextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        max_seq_len: int,
        tokenizer: TextTokenizer | None = None,
    ):
        self.texts = texts
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer or ByteTokenizer()

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


def load_hf_texts(
    dataset_name: str,
    split: str,
    max_samples: int | None,
    text_field: str = "text",
    dataset_config: str | None = None,
    cache_dir: str | None = None,
    streaming: bool = False,
):
    return list(
        iter_hf_texts(
            dataset_name=dataset_name,
            split=split,
            max_samples=max_samples,
            text_field=text_field,
            dataset_config=dataset_config,
            cache_dir=cache_dir,
            streaming=streaming,
        )
    )


def iter_hf_texts(
    dataset_name: str,
    split: str = "train",
    max_samples: int | None = None,
    text_field: str = "text",
    dataset_config: str | None = None,
    cache_dir: str | None = None,
    streaming: bool = False,
):
    """Yield text rows lazily from any Hugging Face dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face dataset loading requires the optional `datasets` package. "
            "Install it or pass --data-file with a local .txt/.jsonl file."
        ) from exc

    resolved_cache_dir = Path(cache_dir).expanduser() if cache_dir else DEFAULT_HF_DATASET_CACHE
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        path=dataset_name,
        name=dataset_config,
        split=split,
        cache_dir=str(resolved_cache_dir),
        streaming=streaming,
        token=get_hf_token(),
    )

    for index, row in enumerate(dataset):
        if text_field not in row:
            raise KeyError(
                f"Text field {text_field!r} was not found in dataset {dataset_name!r}; "
                f"available fields: {list(row)}"
            )
        yield str(row[text_field])
        if max_samples is not None and index + 1 >= max_samples:
            break


def build_text_dataset(
    *,
    max_seq_len: int,
    max_samples: int | None,
    data_file: str | None = None,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    split: str = "train",
    text_field: str = "text",
    cache_dir: str | None = None,
    streaming: bool = False,
    tokenizer: TextTokenizer | None = None,
):
    if data_file:
        texts = read_texts_from_file(
            Path(data_file), text_field=text_field, max_samples=max_samples
        )
    else:
        if not dataset_name:
            raise ValueError("dataset_name is required when data_file is not provided.")
        texts = load_hf_texts(
            dataset_name=dataset_name,
            split=split,
            max_samples=max_samples,
            text_field=text_field,
            dataset_config=dataset_config,
            cache_dir=cache_dir,
            streaming=streaming,
        )

    if not texts:
        raise ValueError("No texts were loaded for the language compression experiment.")

    return TextDataset(texts, max_seq_len=max_seq_len, tokenizer=tokenizer)
