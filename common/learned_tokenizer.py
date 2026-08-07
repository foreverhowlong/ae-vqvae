"""Deterministic, byte-fallback tokenizers exported by learned codecs."""

from __future__ import annotations

import base64
import json
from pathlib import Path


class LearnedByteFallbackTokenizer:
    """Greedy learned vocabulary with a lossless byte fallback alphabet."""

    FORMAT = "learned-byte-fallback-v1"

    def __init__(self, learned_tokens: list[bytes]):
        self.learned_tokens = learned_tokens
        self.codebook_size = len(learned_tokens)
        self.byte_fallback_offset = self.codebook_size
        self.bos_token_id = self.byte_fallback_offset + 256
        self.eos_token_id = self.bos_token_id + 1
        self.pad_token_id = self.eos_token_id + 1
        self.vocab_size = self.pad_token_id + 1
        self._candidates: dict[int, list[tuple[bytes, int]]] = {}
        seen: set[bytes] = set()
        for code_id, token in enumerate(learned_tokens):
            if not token or token in seen:
                continue
            seen.add(token)
            self._candidates.setdefault(token[0], []).append((token, code_id))
        for candidates in self._candidates.values():
            candidates.sort(key=lambda item: (-len(item[0]), item[1]))

    def encode_bytes(
        self,
        raw: bytes,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        ids: list[int] = [self.bos_token_id] if add_bos else []
        position = 0
        while position < len(raw):
            match = None
            for token, code_id in self._candidates.get(raw[position], ()):
                if raw.startswith(token, position):
                    match = (token, code_id)
                    break
            if match is None:
                ids.append(self.byte_fallback_offset + raw[position])
                position += 1
            else:
                token, code_id = match
                ids.append(code_id)
                position += len(token)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        return self.encode_bytes(
            text.encode("utf-8"),
            add_bos=add_bos,
            add_eos=add_eos,
        )

    def decode_bytes(self, ids: list[int]) -> bytes:
        pieces: list[bytes] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id == self.eos_token_id:
                break
            if token_id in {self.bos_token_id, self.pad_token_id}:
                continue
            if 0 <= token_id < self.codebook_size:
                pieces.append(self.learned_tokens[token_id])
            elif self.byte_fallback_offset <= token_id < self.eos_token_id:
                pieces.append(bytes([token_id - self.byte_fallback_offset]))
            else:
                raise ValueError(f"Unknown learned-tokenizer id {token_id}.")
        return b"".join(pieces)

    def decode(self, ids: list[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def to_payload(self) -> dict[str, object]:
        return {
            "format": self.FORMAT,
            "codebook_size": self.codebook_size,
            "byte_fallback_offset": self.byte_fallback_offset,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "vocab_size": self.vocab_size,
            "learned_tokens_base64": [
                base64.b64encode(token).decode("ascii")
                for token in self.learned_tokens
            ],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_payload(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "LearnedByteFallbackTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != cls.FORMAT:
            raise ValueError(f"Unsupported tokenizer format: {payload.get('format')!r}.")
        tokens = [
            base64.b64decode(value)
            for value in payload["learned_tokens_base64"]
        ]
        tokenizer = cls(tokens)
        for field in (
            "codebook_size",
            "byte_fallback_offset",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "vocab_size",
        ):
            if getattr(tokenizer, field) != payload[field]:
                raise ValueError(f"Inconsistent tokenizer metadata for {field}.")
        return tokenizer
