"""Phase-0 decoder-only Transformer and paper-faithful Engram memory.

The Engram path follows equations (1)--(5) of arXiv:2601.07372v2.  The
single-stream adaptation has one key projection and one value projection.
"Layer 2" means that Engram is residual-added immediately before the second
Transformer block (zero-based ``blocks[1]``), matching the official demo.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Phase0BackboneConfig:
    vocab_size: int = 50_257
    context_length: int = 1_024
    layers: int = 12
    d_model: int = 768
    attention_heads: int = 12
    d_ff: int = 2_048
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5

    def validate(self) -> None:
        if self.layers < 2:
            raise ValueError("Layer-2 Engram injection requires at least two blocks.")
        if self.d_model % self.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads.")
        if self.context_length < 1 or self.d_ff < 1 or self.vocab_size < 1:
            raise ValueError("Backbone dimensions must be positive.")


@dataclass(frozen=True)
class Phase0EngramConfig:
    enabled: bool = False
    table_rows_target: int = 0
    ngram_orders: tuple[int, ...] = (2, 3)
    hash_heads_per_order: int = 8
    memory_dim: int = 256
    injection_layer: int = 2  # one-based paper layer number
    kernel_size: int = 4
    dilation: int = 3
    hash_seed: int = 0
    init_seed: int = 10_042
    pad_token_id: int = 50_256
    sparse_gradients: bool = True
    # FP32 master parameters + BF16 autocast avoid BF16 Adam moment accumulation.
    table_dtype: str = "float32"

    def validate(self, backbone: Phase0BackboneConfig) -> None:
        if not self.enabled:
            return
        routes = len(self.ngram_orders) * self.hash_heads_per_order
        if self.ngram_orders != (2, 3):
            raise ValueError("Phase-0 fixes ngram_orders to (2, 3).")
        if routes != 16 or self.memory_dim % routes:
            raise ValueError("Phase-0 requires 16 routes evenly dividing memory_dim.")
        if self.memory_dim != 256:
            raise ValueError("Phase-0 fixes memory_dim to 256.")
        if self.injection_layer != 2:
            raise ValueError("Phase-0 fixes Engram injection to paper Layer 2.")
        if not 1 <= self.injection_layer <= backbone.layers:
            raise ValueError("Engram injection layer is outside the backbone.")
        if self.table_rows_target < 2:
            raise ValueError("Enabled Engram requires table_rows_target >= 2.")
        if not 0 <= self.pad_token_id < backbone.vocab_size:
            raise ValueError("pad_token_id must be inside the backbone vocabulary.")
        if self.kernel_size != 4 or self.dilation != max(self.ngram_orders):
            raise ValueError("Phase-0 fixes convolution kernel=4 and dilation=max n=3.")
        if self.table_dtype not in {"float32", "bfloat16"}:
            raise ValueError("table_dtype must be float32 or bfloat16.")

    @property
    def route_count(self) -> int:
        return len(self.ngram_orders) * self.hash_heads_per_order

    @property
    def embedding_dim_per_route(self) -> int:
        return self.memory_dim // self.route_count


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = inputs.float() * torch.rsqrt(
            inputs.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(inputs.dtype) * self.weight


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small_primes:
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        shifts += 1
        exponent //= 2
    # Deterministic Miller-Rabin bases for unsigned 64-bit integers.
    for base in (2, 325, 9_375, 28_178, 450_775, 9_780_504, 1_795_265_022):
        if base % value == 0:
            continue
        x = pow(base, exponent, value)
        if x in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            x = pow(x, 2, value)
            if x == value - 1:
                break
        else:
            return False
    return True


def consecutive_unique_primes(start: int, count: int) -> tuple[int, ...]:
    """Official-table convention: successive unique primes at or above target M."""
    candidate = max(2, int(start))
    output: list[int] = []
    while len(output) < count:
        if _is_prime(candidate):
            output.append(candidate)
        candidate += 1
    return tuple(output)


def official_hash_multipliers(
    *, vocabulary_size: int, max_ngram_order: int, layer_id_zero_based: int, seed: int
) -> torch.Tensor:
    """Reproduce ``NgramHashMapping`` multiplier generation from DeepSeek-AI/Engram."""
    max_long = np.iinfo(np.int64).max
    half_bound = max(1, int(max_long // vocabulary_size) // 2)
    generator = np.random.default_rng(seed + 10_007 * layer_id_zero_based)
    values = generator.integers(
        low=0, high=half_bound, size=(max_ngram_order,), dtype=np.int64
    )
    return torch.from_numpy(values * 2 + 1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, context_length: int, theta: float):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even.")
        inverse = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(context_length, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse)
        self.register_buffer("cos", frequencies.cos(), persistent=False)
        self.register_buffer("sin", frequencies.sin(), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [batch, heads, time, head_dim]
        length = inputs.shape[-2]
        cos = self.cos[:length].to(dtype=inputs.dtype)[None, None, :, :]
        sin = self.sin[:length].to(dtype=inputs.dtype)[None, None, :, :]
        even, odd = inputs[..., 0::2], inputs[..., 1::2]
        return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Phase0BackboneConfig):
        super().__init__()
        self.heads = config.attention_heads
        self.width = config.d_model
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(
            config.d_model // config.attention_heads,
            config.context_length,
            config.rope_theta,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, length, width = inputs.shape
        head_dim = width // self.heads
        query, key, value = self.qkv(inputs).chunk(3, dim=-1)
        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, length, self.heads, head_dim).transpose(1, 2)
        query = self.rope(split_heads(query))
        key = self.rope(split_heads(key))
        value = split_heads(value)
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=True
        )
        return self.output(attended.transpose(1, 2).contiguous().view(batch, length, width))


class SwiGLU(nn.Module):
    def __init__(self, config: Phase0BackboneConfig):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(inputs)) * self.up(inputs))


class TransformerBlock(nn.Module):
    def __init__(self, config: Phase0BackboneConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attention(self.attention_norm(inputs))
        return inputs + self.ffn(self.ffn_norm(inputs))


class EngramMemory(nn.Module):
    """Equations (1)--(5), adapted from mHC to one residual stream."""

    def __init__(
        self,
        backbone: Phase0BackboneConfig,
        config: Phase0EngramConfig,
        canonical_projection: torch.Tensor,
    ):
        super().__init__()
        if canonical_projection.shape != (backbone.vocab_size,):
            raise ValueError(
                "canonical_projection must contain exactly one ID per backbone token."
            )
        self.config = config
        self.table_row_counts = consecutive_unique_primes(
            config.table_rows_target, config.route_count
        )
        offsets = [0]
        for rows in self.table_row_counts[:-1]:
            offsets.append(offsets[-1] + rows)
        self.register_buffer("table_offsets", torch.tensor(offsets, dtype=torch.long))
        self.register_buffer("table_moduli", torch.tensor(self.table_row_counts, dtype=torch.long))
        self.register_buffer("canonical_projection", canonical_projection.long())
        self.register_buffer(
            "canonical_pad_id",
            canonical_projection.long()[config.pad_token_id].clone(),
        )
        canonical_vocab_size = int(canonical_projection.max()) + 1
        self.register_buffer(
            "hash_multipliers",
            official_hash_multipliers(
                vocabulary_size=canonical_vocab_size,
                max_ngram_order=max(config.ngram_orders),
                layer_id_zero_based=config.injection_layer - 1,
                seed=config.hash_seed,
            ),
        )
        table_dtype = torch.bfloat16 if config.table_dtype == "bfloat16" else torch.float32
        self.embedding = nn.Embedding(
            sum(self.table_row_counts),
            config.embedding_dim_per_route,
            sparse=config.sparse_gradients,
            dtype=table_dtype,
        )
        self.key_projection = nn.Linear(config.memory_dim, backbone.d_model)
        self.value_projection = nn.Linear(config.memory_dim, backbone.d_model)
        self.query_norm = RMSNorm(backbone.d_model, backbone.rms_norm_eps)
        self.key_norm = RMSNorm(backbone.d_model, backbone.rms_norm_eps)
        self.value_norm = RMSNorm(backbone.d_model, backbone.rms_norm_eps)
        self.short_conv = nn.Conv1d(
            backbone.d_model,
            backbone.d_model,
            kernel_size=config.kernel_size,
            dilation=config.dilation,
            padding=(config.kernel_size - 1) * config.dilation,
            groups=backbone.d_model,
            bias=False,
        )
        self._initialize(config.init_seed)

    def _initialize(self, seed: int) -> None:
        devices = []
        if torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for projection in (self.key_projection, self.value_projection):
                nn.init.normal_(projection.weight, mean=0.0, std=0.02)
                nn.init.zeros_(projection.bias)
            # Paper Section 5 / Table 5: zero-init the convolutional branch.
            nn.init.zeros_(self.short_conv.weight)
        # Keep dense Engram initialization identical across the M sweep.  Table
        # draws use a separate stream so changing its row count cannot advance
        # the RNG used by W_K/W_V.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 1)
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def hash_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        compressed = self.canonical_projection[input_ids]
        shifts = [compressed]
        for shift in range(1, max(self.config.ngram_orders)):
            shifted = torch.full_like(compressed, self.canonical_pad_id)
            shifted[:, shift:] = compressed[:, :-shift]
            shifts.append(shifted)
        hashes: list[torch.Tensor] = []
        route = 0
        for order in self.config.ngram_orders:
            mixed = shifts[0] * self.hash_multipliers[0]
            for position in range(1, order):
                mixed = torch.bitwise_xor(
                    mixed, shifts[position] * self.hash_multipliers[position]
                )
            for _ in range(self.config.hash_heads_per_order):
                hashes.append(
                    torch.remainder(mixed, self.table_moduli[route])
                    + self.table_offsets[route]
                )
                route += 1
        return torch.stack(hashes, dim=-1)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        memory = self.embedding(self.hash_ids(input_ids)).flatten(start_dim=-2)
        key = self.key_projection(memory)
        value = self.value_projection(memory)
        gate_logits = (
            self.query_norm(hidden_states) * self.key_norm(key)
        ).sum(dim=-1) / math.sqrt(hidden_states.shape[-1])
        # Eq. (4) in v2 is used directly.  The older demo's signed-sqrt transform
        # is intentionally not applied because it is absent from the paper equation.
        gate = torch.sigmoid(gate_logits)
        gated_value = gate.unsqueeze(-1) * value
        convolved = self.short_conv(
            self.value_norm(gated_value).transpose(1, 2)
        )[..., : hidden_states.shape[1]].transpose(1, 2)
        output = gated_value + F.silu(convolved)
        return output, {
            "engram_gate_mean": gate.detach().float().mean(),
            "engram_gate_std": gate.detach().float().std(unbiased=False),
        }

    @property
    def sparse_parameter_count(self) -> int:
        return self.embedding.weight.numel()

    @property
    def theoretical_bf16_bytes(self) -> int:
        return self.sparse_parameter_count * 2


class EngramPhase0LM(nn.Module):
    def __init__(
        self,
        backbone_config: Phase0BackboneConfig,
        engram_config: Phase0EngramConfig,
        canonical_projection: torch.Tensor | None = None,
        *,
        backbone_init_seed: int = 42,
    ):
        super().__init__()
        backbone_config.validate()
        engram_config.validate(backbone_config)
        self.backbone_config = backbone_config
        self.engram_config = engram_config
        self.token_embedding = nn.Embedding(
            backbone_config.vocab_size, backbone_config.d_model
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(backbone_config) for _ in range(backbone_config.layers)
        )
        self.final_norm = RMSNorm(
            backbone_config.d_model, backbone_config.rms_norm_eps
        )
        self._initialize_backbone(backbone_init_seed)
        self.engram: EngramMemory | None = None
        if engram_config.enabled:
            if canonical_projection is None:
                raise ValueError("Enabled Engram requires a canonical token-ID projection.")
            self.engram = EngramMemory(
                backbone_config, engram_config, canonical_projection
            )

    def _initialize_backbone(self, seed: int) -> None:
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
            residual_std = 0.02 / math.sqrt(2 * self.backbone_config.layers)
            for block in self.blocks:
                nn.init.normal_(block.attention.output.weight, mean=0.0, std=residual_std)
                nn.init.normal_(block.ffn.down.weight, mean=0.0, std=residual_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time].")
        if input_ids.shape[1] > self.backbone_config.context_length:
            raise ValueError("Input exceeds configured context length.")
        hidden = self.token_embedding(input_ids)
        auxiliary: dict[str, torch.Tensor] = {}
        injection_index = self.engram_config.injection_layer - 1
        for index, block in enumerate(self.blocks):
            if self.engram is not None and index == injection_index:
                memory_output, auxiliary = self.engram(hidden, input_ids)
                hidden = hidden + memory_output
            hidden = block(hidden)
        logits = F.linear(self.final_norm(hidden), self.token_embedding.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-1,
                reduction=reduction,
            )
        return logits, loss, auxiliary

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if not name.startswith("engram."):
                yield parameter

    def dense_engram_parameters(self) -> Iterable[nn.Parameter]:
        if self.engram is None:
            return
        for name, parameter in self.engram.named_parameters():
            if name != "embedding.weight":
                yield parameter

    def sparse_engram_parameters(self) -> Iterable[nn.Parameter]:
        if self.engram is not None:
            yield self.engram.embedding.weight

    def parameter_counts(self) -> dict[str, int]:
        backbone = sum(parameter.numel() for parameter in self.backbone_parameters())
        dense = sum(parameter.numel() for parameter in self.dense_engram_parameters())
        sparse = sum(parameter.numel() for parameter in self.sparse_engram_parameters())
        return {
            "backbone": backbone,
            "dense_engram": dense,
            "sparse_tables": sparse,
            "total": backbone + dense + sparse,
            "table_theoretical_bf16_bytes": sparse * 2,
        }


def analytical_backbone_parameter_count(config: Phase0BackboneConfig) -> int:
    """Count tied vocabulary, attention/FFN matrices, and RMSNorm scales."""
    embedding = config.vocab_size * config.d_model
    per_layer = (
        4 * config.d_model * config.d_model
        + 3 * config.d_model * config.d_ff
        + 2 * config.d_model
    )
    return embedding + config.layers * per_layer + config.d_model


def analytical_engram_counts(
    backbone: Phase0BackboneConfig, config: Phase0EngramConfig
) -> dict[str, int]:
    if not config.enabled:
        return {"dense_engram": 0, "sparse_tables": 0, "table_theoretical_bf16_bytes": 0}
    rows = consecutive_unique_primes(config.table_rows_target, config.route_count)
    sparse = sum(rows) * config.embedding_dim_per_route
    dense = (
        2 * (config.memory_dim * backbone.d_model + backbone.d_model)
        + 3 * backbone.d_model
        + backbone.d_model * config.kernel_size
    )
    return {
        "dense_engram": dense,
        "sparse_tables": sparse,
        "table_theoretical_bf16_bytes": sparse * 2,
    }
