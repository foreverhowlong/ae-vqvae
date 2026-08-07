"""Small GPT-2 architecture adapted from the public nanoGPT design."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NanoGPTConfig:
    block_size: int = 256
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(inputs, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        if config.n_embd % config.n_head:
            raise ValueError("n_embd must be divisible by n_head.")
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = inputs.shape
        q, k, v = self.c_attn(inputs).split(self.n_embd, dim=2)
        head_dim = channels // self.n_head
        q = q.view(batch, seq_len, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, head_dim).transpose(1, 2)
        if self.flash:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
            attention = attention.masked_fill(
                self.bias[:, :, :seq_len, :seq_len] == 0,
                float("-inf"),
            )
            attention = self.attn_dropout(F.softmax(attention, dim=-1))
            output = attention @ v
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_dropout(self.c_proj(output))


class MLP(nn.Module):
    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(inputs))))


class Block(nn.Module):
    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attn(self.ln_1(inputs))
        return inputs + self.mlp(self.ln_2(inputs))


class NanoGPT(nn.Module):
    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": LayerNorm(config.n_embd, config.bias),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer),
                )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        seq_len = input_ids.shape[1]
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block_size={self.config.block_size}."
            )
        positions = torch.arange(seq_len, device=input_ids.device)
        hidden = self.transformer.drop(
            self.transformer.wte(input_ids) + self.transformer.wpe(positions)
        )
        for block in self.transformer.h:
            hidden = block(hidden)
        logits = self.lm_head(self.transformer.ln_f(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-1,
                reduction=reduction,
            )
        return logits, loss

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

