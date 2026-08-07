"""Lossless document tokenizer backed by a fixed-slot text VQ-VAE.

The existing autoencoder is block-contextual rather than a standalone token
dictionary. A document is therefore represented by VQ blocks only when every
valid BPE token reconstructs exactly; otherwise the whole document uses the
explicit byte fallback. This conservative contract prevents lossy codecs from
obtaining artificially low downstream bits-per-byte.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import torch

from common.text_data import BPETokenizer
from common.text_vqvae_config import CollapseControlConfig, TextVQVAEConfig
from models.text_encoders import pad_aware_adaptive_pool1d
from models.text_vqvae import TextVQVAE


def _dataclass_payload(cls, payload: dict) -> dict:
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in payload.items() if key in names}


class VQBlockTokenizer:
    def __init__(
        self,
        model: TextVQVAE,
        bpe_tokenizer: BPETokenizer,
        device: torch.device,
    ):
        if model.config.codebook_size != 8192:
            raise ValueError("The downstream VQ tokenizer locks codebook_size=8192.")
        self.model = model.to(device).eval()
        self.bpe = bpe_tokenizer
        self.device = device
        self.codebook_size = model.config.codebook_size
        self.byte_fallback_offset = self.codebook_size
        self.vq_block_token_id = self.byte_fallback_offset + 256
        self.byte_document_token_id = self.vq_block_token_id + 1
        self.bos_token_id = self.byte_document_token_id + 1
        self.eos_token_id = self.bos_token_id + 1
        self.pad_token_id = self.eos_token_id + 1
        self.vocab_size = self.pad_token_id + 1

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        config_path: str | Path,
        device: torch.device,
    ) -> "VQBlockTokenizer":
        import json

        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
        model_config = TextVQVAEConfig(
            **_dataclass_payload(TextVQVAEConfig, payload["model"])
        )
        collapse = CollapseControlConfig(
            **_dataclass_payload(
                CollapseControlConfig,
                payload.get("collapse_control", {}),
            )
        )
        model = TextVQVAE(model_config, collapse_config=collapse)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        tokenizer_path = payload["train"].get("tokenizer_path")
        if not tokenizer_path:
            raise ValueError("VQ run config is missing the BPE tokenizer path.")
        path = Path(tokenizer_path)
        if not path.is_file():
            # Historical configs may contain a path from another checkout.
            path = (
                Path(config_path).resolve().parents[3]
                / "outputs"
                / "tokenizers"
                / "tinystories_bpe_8k"
                / "tokenizer.json"
            )
        if not path.is_file():
            from common.text_data import DEFAULT_BPE_TOKENIZER_PATH

            path = DEFAULT_BPE_TOKENIZER_PATH
        return cls(model, BPETokenizer(path), device)

    def _length_tokens(self, length: int) -> list[int]:
        if not 0 <= length <= self.model.config.max_seq_len:
            raise ValueError(f"Invalid VQ block length {length}.")
        return [
            self.byte_fallback_offset + (length & 0xFF),
            self.byte_fallback_offset + ((length >> 8) & 0xFF),
        ]

    @torch.no_grad()
    def encode(self, text: str) -> tuple[list[int], dict[str, int | bool]]:
        chunks = self.bpe.encode_chunks(text, self.model.config.max_seq_len)
        encoded_blocks: list[int] = []
        exact = True
        valid_bpe_tokens = 0
        for input_ids, attention_mask in chunks:
            inputs = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
            outputs = self.model(inputs, mask)
            valid = mask.bool()
            predictions = outputs["logits"].argmax(dim=-1)
            if not torch.equal(predictions[valid], inputs[valid]):
                exact = False
                break
            indices = outputs["indices"][0]
            if (indices < 0).any():
                # Keep a fixed-size block; invalid slots are ignored by the
                # reconstructed latent mask and use code zero as serialization padding.
                indices = indices.clamp_min(0)
            length = int(valid.sum())
            valid_bpe_tokens += length
            encoded_blocks.extend([
                self.vq_block_token_id,
                *self._length_tokens(length),
                *[int(value) for value in indices.tolist()],
            ])

        raw = text.encode("utf-8")
        if exact:
            ids = [self.bos_token_id, *encoded_blocks]
        else:
            ids = [
                self.bos_token_id,
                self.byte_document_token_id,
                *[self.byte_fallback_offset + value for value in raw],
            ]
        ids.append(self.eos_token_id)
        return ids, {
            "used_vq": exact,
            "fallback_bytes": 0 if exact else len(raw),
            "raw_bytes": len(raw),
            "valid_bpe_tokens": valid_bpe_tokens,
        }

    @torch.no_grad()
    def decode_bytes(self, ids: list[int]) -> bytes:
        if not ids:
            return b""
        position = 1 if ids and ids[0] == self.bos_token_id else 0
        if position >= len(ids):
            return b""
        if ids[position] == self.byte_document_token_id:
            raw = []
            for token_id in ids[position + 1:]:
                if token_id == self.eos_token_id:
                    break
                if not self.byte_fallback_offset <= token_id < self.vq_block_token_id:
                    raise ValueError("Malformed byte-fallback VQ document.")
                raw.append(token_id - self.byte_fallback_offset)
            return bytes(raw)

        bpe_ids: list[int] = []
        while position < len(ids) and ids[position] != self.eos_token_id:
            if ids[position] != self.vq_block_token_id:
                raise ValueError("Malformed VQ block stream.")
            low, high = ids[position + 1 : position + 3]
            length = (
                low - self.byte_fallback_offset
                + ((high - self.byte_fallback_offset) << 8)
            )
            start = position + 3
            end = start + self.model.config.latent_slots
            code_ids = torch.tensor([ids[start:end]], device=self.device)
            if code_ids.shape[1] != self.model.config.latent_slots:
                raise ValueError("Truncated VQ block stream.")
            codes = self.model.quantizer.codebook(code_ids)
            output_mask = torch.arange(
                self.model.config.max_seq_len,
                device=self.device,
            )[None] < length
            dummy = torch.zeros(
                1,
                self.model.config.max_seq_len,
                1,
                device=self.device,
            )
            _, latent_mask = pad_aware_adaptive_pool1d(
                dummy,
                output_mask,
                self.model.config.latent_slots,
                slot_pad_ratio_threshold=self.model.config.slot_pad_ratio_threshold,
            )
            logits = self.model.decode(
                codes,
                self.model.config.max_seq_len,
                latent_mask=latent_mask,
                output_mask=output_mask,
            )
            bpe_ids.extend(logits.argmax(dim=-1)[0, :length].tolist())
            position = end
        return self.bpe.decode(bpe_ids).encode("utf-8")
