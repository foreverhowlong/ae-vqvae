"""BPE text data with lossless byte counts and legal word-boundary endpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from common.text_data import BPETokenizer, load_hf_texts, read_texts_from_file


def _is_word_boundary(text: str, offset: int) -> bool:
    """A boundary is illegal only when it splits two word characters."""
    if offset <= 0 or offset >= len(text):
        return True
    return not (text[offset - 1].isalnum() and text[offset].isalnum())


class EndToEndTokenizerDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        *,
        tokenizer: BPETokenizer,
        max_seq_len: int,
        word_boundary_only: bool,
    ):
        if not texts:
            raise ValueError("End-to-end tokenizer dataset cannot be empty.")
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.word_boundary_only = word_boundary_only
        self.continuous_truncation = False

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        text = self.texts[index]
        encoded = self.tokenizer.tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        content_ids = encoded.ids[: self.max_seq_len - 1]
        offsets = encoded.offsets[: self.max_seq_len - 1]
        input_ids = content_ids + [self.tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        legal_endpoints = [
            (
                not self.word_boundary_only
                or _is_word_boundary(text, int(end))
            )
            for _, end in offsets
        ] + [True]
        covered_characters = int(offsets[-1][1]) if offsets else 0
        raw_byte_count = len(text[:covered_characters].encode("utf-8"))

        padding = self.max_seq_len - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * padding)
        attention_mask.extend([0] * padding)
        legal_endpoints.extend([False] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "legal_endpoints": torch.tensor(legal_endpoints, dtype=torch.bool),
            "raw_byte_count": torch.tensor(raw_byte_count, dtype=torch.long),
        }


def build_end_to_end_tokenizer_dataset(
    *,
    tokenizer: BPETokenizer,
    max_seq_len: int,
    max_samples: int | None,
    word_boundary_only: bool,
    data_file: str | None = None,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    split: str = "train",
    text_field: str = "text",
    cache_dir: str | None = None,
) -> EndToEndTokenizerDataset:
    if data_file:
        texts = read_texts_from_file(
            Path(data_file),
            text_field=text_field,
            max_samples=max_samples,
        )
    else:
        if not dataset_name:
            raise ValueError("dataset_name is required without data_file.")
        texts = load_hf_texts(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            max_samples=max_samples,
            text_field=text_field,
            cache_dir=cache_dir,
        )
    return EndToEndTokenizerDataset(
        texts,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        word_boundary_only=word_boundary_only,
    )
