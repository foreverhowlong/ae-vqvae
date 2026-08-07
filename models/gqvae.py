"""Gated quantized VAE for learning variable-length byte tokens.

The model follows the GQ-VAE decomposition: a bidirectional encoder produces
one latent per input byte, a VQ bottleneck discretizes those latents, a gater
learns token boundaries, and a decoder predicts a bounded backwards byte span
plus its length.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.gqvae_config import GQVAEConfig
from common.text_vqvae_config import CollapseControlConfig
from models.text_vqvae import VectorQuantizer


def _transformer_stack(
    d_model: int,
    n_heads: int,
    layers: int,
    ffn_mult: int,
    dropout: float,
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_model * ffn_mult,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


class GQVAE(nn.Module):
    def __init__(self, config: GQVAEConfig):
        super().__init__()
        if config.codebook_size != 8192:
            raise ValueError("The research baseline locks codebook_size=8192.")
        if config.decode_width < 1:
            raise ValueError("decode_width must be positive.")
        if not 0.0 < config.gate_threshold < 1.0:
            raise ValueError("gate_threshold must be between zero and one.")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.encoder = _transformer_stack(
            config.d_model,
            config.n_heads,
            config.encoder_layers,
            config.ffn_mult,
            config.dropout,
        )
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.latent_proj = nn.Linear(config.d_model, config.code_dim)
        collapse = CollapseControlConfig(
            use_ema_codebook=config.use_ema_codebook,
            ema_decay=config.ema_decay,
            ema_eps=config.ema_eps,
        )
        self.collapse_config = collapse
        self.quantizer = VectorQuantizer(
            config.codebook_size,
            config.code_dim,
            collapse,
        )
        self.gater_input = nn.Linear(config.code_dim, config.d_model)
        self.gater = _transformer_stack(
            config.d_model,
            config.n_heads,
            config.gater_layers,
            config.ffn_mult,
            config.dropout,
        )
        self.gater_norm = nn.LayerNorm(config.d_model)
        self.gate_head = nn.Linear(config.d_model, 1)
        self.decoder_expand = nn.Linear(
            config.code_dim,
            config.decode_width * config.d_model,
        )
        self.byte_head = nn.Linear(config.d_model, config.vocab_size)
        self.length_head = nn.Linear(config.code_dim, config.decode_width)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_mask: bool = False,
    ):
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {input_ids.shape[1]} exceeds "
                f"max_seq_len={self.config.max_seq_len}."
            )
        if attention_mask is None:
            attention_mask = input_ids != self.config.pad_token_id
        else:
            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None]
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask)
        latents = self.latent_proj(self.encoder_norm(hidden))
        latents = torch.where(
            attention_mask.unsqueeze(-1),
            latents,
            torch.zeros_like(latents),
        )
        if return_mask:
            return latents, attention_mask
        return latents

    def decode_codes(self, codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expanded = self.decoder_expand(codes)
        expanded = expanded.view(
            *codes.shape[:-1],
            self.config.decode_width,
            self.config.d_model,
        )
        return self.byte_head(F.gelu(expanded)), self.length_head(codes)

    def _gates(
        self,
        quantized: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.gater_input(quantized)
        hidden = self.gater(hidden, src_key_padding_mask=~attention_mask)
        gates = torch.sigmoid(self.gate_head(self.gater_norm(hidden)).squeeze(-1))
        return gates * attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        use_quantizer: bool = True,
        compression_weight: float | None = None,
    ) -> dict[str, torch.Tensor | float | bool]:
        z_e, valid_mask = self.encode(
            input_ids,
            attention_mask,
            return_mask=True,
        )
        if use_quantizer:
            quantized = self.quantizer(z_e, valid_mask=valid_mask)
            z_q_raw = quantized["z_q_raw"]
            z_q = quantized["z_q_st"]
            indices = quantized["indices"]
        else:
            z_q_raw = z_e
            z_q = z_e
            indices = torch.full(
                z_e.shape[:2],
                -1,
                dtype=torch.long,
                device=z_e.device,
            )
        gates = self._gates(z_q, valid_mask)
        byte_logits, length_logits = self.decode_codes(z_q)
        losses = self.losses(
            input_ids,
            valid_mask,
            z_e,
            z_q_raw,
            gates,
            byte_logits,
            length_logits,
            use_quantizer=use_quantizer,
            compression_weight=(
                self.config.compression_weight
                if compression_weight is None
                else compression_weight
            ),
        )
        return {
            "z_e": z_e,
            "z_q_raw": z_q_raw,
            "z_q_st": z_q,
            "indices": indices,
            "latent_mask": valid_mask,
            "gates": gates,
            "byte_logits": byte_logits,
            "length_logits": length_logits,
            "quantizer_active": use_quantizer,
            **losses,
        }

    def _reconstruction_targets(
        self,
        input_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        gates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len = input_ids.shape
        width = self.config.decode_width
        targets = torch.full(
            (batch, seq_len, width),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        masks = torch.zeros(
            (batch, seq_len, width),
            dtype=gates.dtype,
            device=gates.device,
        )
        for offset in range(width):
            if offset >= seq_len:
                break
            targets[:, offset:, offset] = input_ids[:, : seq_len - offset]
            target_valid = valid_mask[:, : seq_len - offset].to(gates.dtype)
            survival = torch.ones_like(gates)
            for previous in range(1, offset + 1):
                shifted = torch.ones_like(gates)
                shifted[:, previous:] = 1.0 - gates[:, : seq_len - previous]
                survival = survival * shifted
            masks[:, offset:, offset] = (
                survival[:, offset:]
                * target_valid
                * valid_mask[:, offset:].to(gates.dtype)
            )
        return targets, masks

    def losses(
        self,
        input_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        z_e: torch.Tensor,
        z_q_raw: torch.Tensor,
        gates: torch.Tensor,
        byte_logits: torch.Tensor,
        length_logits: torch.Tensor,
        *,
        use_quantizer: bool,
        compression_weight: float,
    ) -> dict[str, torch.Tensor]:
        targets, reconstruction_mask = self._reconstruction_targets(
            input_ids,
            valid_mask,
            gates,
        )
        token_losses = F.cross_entropy(
            byte_logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
            reduction="none",
        ).view_as(reconstruction_mask)
        reconstruction = (
            (token_losses * reconstruction_mask).sum()
            / reconstruction_mask.sum().clamp_min(1.0)
        )
        valid = valid_mask.to(gates.dtype)
        compression_raw = (gates * valid).sum() / valid.sum().clamp_min(1.0)
        compression = compression_weight * compression_raw
        length_targets = reconstruction_mask.detach()
        length_losses = F.binary_cross_entropy_with_logits(
            length_logits,
            length_targets,
            reduction="none",
        ).mean(dim=-1)
        length_weights = gates.detach() * valid
        length = (
            (length_losses * length_weights).sum()
            / length_weights.sum().clamp_min(1.0)
        )

        if use_quantizer:
            selected = valid_mask.unsqueeze(-1)
            commitment_raw = F.mse_loss(
                z_e[selected.expand_as(z_e)],
                z_q_raw.detach()[selected.expand_as(z_q_raw)],
            )
            if self.config.use_ema_codebook:
                codebook = reconstruction.new_zeros(())
            else:
                codebook = F.mse_loss(
                    z_q_raw[selected.expand_as(z_q_raw)],
                    z_e.detach()[selected.expand_as(z_e)],
                )
            commitment = self.config.commitment_beta * commitment_raw
        else:
            codebook = reconstruction.new_zeros(())
            commitment_raw = reconstruction.new_zeros(())
            commitment = reconstruction.new_zeros(())
        total = (
            reconstruction
            + compression
            + self.config.length_weight * length
            + codebook
            + commitment
        )
        return {
            "loss": total,
            "reconstruction_loss": reconstruction,
            "compression_loss": compression,
            "compression_gate_mean": compression_raw,
            "length_loss": length,
            "codebook_loss": codebook,
            "commitment_loss": commitment_raw,
            "commitment_weighted_loss": commitment,
            "reconstruction_mask": reconstruction_mask,
        }

    @torch.no_grad()
    def decoded_codebook(self) -> list[bytes]:
        was_training = self.training
        self.eval()
        try:
            byte_logits, length_logits = self.decode_codes(self.quantizer.codebook.weight)
            predicted = byte_logits.argmax(dim=-1)
            lengths = length_logits.argmax(dim=-1) + 1
            tokens: list[bytes] = []
            for code, length in zip(predicted, lengths, strict=True):
                backwards = [
                    int(value)
                    for value in code[: int(length)].tolist()
                    if 0 <= int(value) < 256
                ]
                tokens.append(bytes(reversed(backwards)))
            return tokens
        finally:
            self.train(was_training)
