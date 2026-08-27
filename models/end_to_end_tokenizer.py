"""End-to-end greedy chunk tokenizer with a code prior and causal text decoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.end_to_end_tokenizer_config import EndToEndTokenizerConfig
from common.segmental_vqvae_config import SegmentalVQVAEConfig
from common.text_vqvae_config import CollapseControlConfig
from models.nanogpt import Block, LayerNorm, NanoGPTConfig
from models.segmental_vqvae import (
    GreedySpanSegmenter,
    LocalBoundaryEncoder,
    SegmentContentEncoder,
)
from models.text_vqvae import VectorQuantizer


def _gpt_config(
    *,
    block_size: int,
    vocab_size: int,
    layers: int,
    heads: int,
    width: int,
    dropout: float,
    bias: bool,
) -> NanoGPTConfig:
    return NanoGPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=layers,
        n_head=heads,
        n_embd=width,
        dropout=dropout,
        bias=bias,
    )


class ChunkPrior(nn.Module):
    """Factor p(length, code | previous chunks) with an 18M GPT-2 trunk."""

    def __init__(self, config: EndToEndTokenizerConfig):
        super().__init__()
        self.codebook_size = config.codebook_size
        self.max_span_length = config.max_span_length
        gpt = _gpt_config(
            block_size=config.max_seq_len,
            vocab_size=config.codebook_size + 1,
            layers=config.prior_layers,
            heads=config.prior_heads,
            width=config.prior_d_model,
            dropout=config.prior_dropout,
            bias=config.prior_bias,
        )
        self.code_embedding = nn.Embedding(config.codebook_size + 1, gpt.n_embd)
        self.length_embedding = nn.Embedding(config.max_span_length + 1, gpt.n_embd)
        self.position_embedding = nn.Embedding(config.max_seq_len, gpt.n_embd)
        self.dropout = nn.Dropout(gpt.dropout)
        self.blocks = nn.ModuleList(Block(gpt) for _ in range(gpt.n_layer))
        self.output_norm = LayerNorm(gpt.n_embd, gpt.bias)
        self.code_condition_norm = LayerNorm(gpt.n_embd, gpt.bias)
        self.length_head = nn.Linear(
            gpt.n_embd,
            config.max_span_length,
            bias=gpt.bias,
        )
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * gpt.n_layer),
                )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        codes: torch.Tensor,
        lengths: torch.Tensor,
        chunk_mask: torch.Tensor,
        length_probabilities: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, chunk_slots = codes.shape
        if lengths.shape != codes.shape or chunk_mask.shape != codes.shape:
            raise ValueError("codes, lengths, and chunk_mask must have the same shape.")
        if length_probabilities.shape != (
            batch_size,
            chunk_slots,
            self.max_span_length,
        ):
            raise ValueError("length_probabilities has the wrong shape.")

        bos_code = self.codebook_size
        previous_codes = torch.full_like(codes, bos_code)
        previous_lengths = torch.zeros_like(lengths)
        if chunk_slots > 1:
            previous_codes[:, 1:] = codes[:, :-1].clamp_min(0)
            previous_lengths[:, 1:] = lengths[:, :-1]
        positions = torch.arange(chunk_slots, device=codes.device)
        hidden = self.dropout(
            self.code_embedding(previous_codes)
            + self.length_embedding(previous_lengths)
            + self.position_embedding(positions)
        )
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.output_norm(hidden)
        length_logits = self.length_head(hidden)

        hard_current_length = self.length_embedding(lengths.clamp_min(0))
        soft_current_length = length_probabilities @ self.length_embedding.weight[1:]
        current_length = soft_current_length + (
            hard_current_length - soft_current_length
        ).detach()
        code_hidden = self.code_condition_norm(hidden + current_length)
        code_logits = F.linear(
            code_hidden,
            self.code_embedding.weight[: self.codebook_size],
        )
        return {
            "length_logits": length_logits,
            "code_logits": code_logits,
        }


class CausalTextDecoder(nn.Module):
    """Predict BPE text from its prefix and the currently consumed chunk code."""

    def __init__(self, config: EndToEndTokenizerConfig):
        super().__init__()
        gpt = _gpt_config(
            block_size=config.max_seq_len,
            vocab_size=config.vocab_size,
            layers=config.text_decoder_layers,
            heads=config.text_decoder_heads,
            width=config.text_decoder_d_model,
            dropout=config.text_decoder_dropout,
            bias=config.text_decoder_bias,
        )
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            gpt.n_embd,
            padding_idx=config.pad_token_id,
        )
        self.position_embedding = nn.Embedding(config.max_seq_len, gpt.n_embd)
        self.local_position_embedding = nn.Embedding(
            config.max_span_length,
            gpt.n_embd,
        )
        self.length_embedding = nn.Embedding(
            config.max_span_length + 1,
            gpt.n_embd,
        )
        self.code_projection = nn.Linear(config.latent_dim, gpt.n_embd)
        self.dropout = nn.Dropout(gpt.dropout)
        self.blocks = nn.ModuleList(Block(gpt) for _ in range(gpt.n_layer))
        self.output_norm = LayerNorm(gpt.n_embd, gpt.bias)
        self.apply(ChunkPrior._init_weights)

    def forward(
        self,
        targets: torch.Tensor,
        segment_ids: torch.Tensor,
        quantized_codes: torch.Tensor,
        chunk_lengths: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        decoder_inputs = torch.full_like(targets, self.pad_token_id)
        decoder_inputs[:, 0] = self.bos_token_id
        decoder_inputs[:, 1:] = targets[:, :-1]
        gather_latent = segment_ids.unsqueeze(-1).expand(
            -1,
            -1,
            quantized_codes.shape[-1],
        )
        current_codes = quantized_codes.gather(1, gather_latent)
        current_lengths = chunk_lengths.gather(1, segment_ids)
        local_positions = SegmentContentEncoder.local_positions(
            segment_ids,
            attention_mask,
        )
        positions = torch.arange(targets.shape[1], device=targets.device)
        hidden = self.dropout(
            self.token_embedding(decoder_inputs)
            + self.position_embedding(positions)
            + self.local_position_embedding(local_positions)
            + self.length_embedding(current_lengths)
            + self.code_projection(current_codes)
        )
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.output_norm(hidden)
        return F.linear(hidden, self.token_embedding.weight)


class EndToEndTokenizerModel(nn.Module):
    """Greedy word-aware segmentation, VQ, chunk prior, and residual text model."""

    def __init__(self, config: EndToEndTokenizerConfig):
        super().__init__()
        config.validate()
        self.config = config
        segmental = SegmentalVQVAEConfig(
            vocab_size=config.vocab_size,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            max_seq_len=config.max_seq_len,
            d_model=config.segmenter_d_model,
            latent_dim=config.latent_dim,
            n_heads=config.segmenter_n_heads,
            encoder_layers=1,
            decoder_layers=1,
            segmentation_mode="semi_markov_greedy",
            boundary_encoder_layers=config.boundary_encoder_layers,
            boundary_window_radius=config.boundary_window_radius,
            max_span_length=config.max_span_length,
            span_encoder_layers=config.span_encoder_layers,
            ffn_mult=config.segmenter_ffn_mult,
            dropout=config.dropout,
            codebook_size=config.codebook_size,
            commitment_beta=config.commitment_beta,
            compression_target=config.compression_target,
            ema_decay=config.ema_decay,
            ema_eps=config.ema_eps,
        )
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.segmenter_d_model,
            padding_idx=config.pad_token_id,
        )
        self.boundary_encoder = LocalBoundaryEncoder(segmental)
        self.segmenter = GreedySpanSegmenter(segmental)
        self.span_encoder = SegmentContentEncoder(segmental)
        self.latent_projection = nn.Linear(config.segmenter_d_model, config.latent_dim)
        self.collapse_config = CollapseControlConfig(
            use_ema_codebook=True,
            ema_decay=config.ema_decay,
            ema_eps=config.ema_eps,
        )
        self.quantizer = VectorQuantizer(
            config.codebook_size,
            config.latent_dim,
            self.collapse_config,
        )
        self.chunk_prior = ChunkPrior(config)
        self.text_decoder = CausalTextDecoder(config)
        self.register_buffer(
            "rate_dual",
            torch.tensor(float(config.rate_dual_initial)),
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        legal_endpoints: torch.Tensor | None = None,
        *,
        return_mask: bool = False,
    ):
        valid_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        token_embeddings = self.token_embedding(input_ids)
        boundary_hidden = self.boundary_encoder(token_embeddings, valid_mask)
        segmented = self.segmenter(
            boundary_hidden,
            valid_mask,
            legal_endpoints=legal_endpoints,
        )
        hard_pooled = self.span_encoder(
            token_embeddings,
            segmented["segment_ids"],
            valid_mask,
            segmented["hard_assignment"],
        )
        soft_proxy = segmented["pooled"]
        pooled = hard_pooled + (soft_proxy - soft_proxy.detach())
        latents = self.latent_projection(pooled)
        latents = torch.where(
            segmented["chunk_mask"].unsqueeze(-1),
            latents,
            torch.zeros_like(latents),
        )
        if return_mask:
            return latents, segmented["chunk_mask"]
        return latents

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        legal_endpoints: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        valid_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        token_embeddings = self.token_embedding(input_ids)
        boundary_hidden = self.boundary_encoder(token_embeddings, valid_mask)
        segmented = self.segmenter(
            boundary_hidden,
            valid_mask,
            legal_endpoints=legal_endpoints,
        )
        hard_pooled = self.span_encoder(
            token_embeddings,
            segmented["segment_ids"],
            valid_mask,
            segmented["hard_assignment"],
        )
        soft_proxy = segmented["pooled"]
        pooled = hard_pooled + (soft_proxy - soft_proxy.detach())
        z_e = self.latent_projection(pooled)
        chunk_mask = segmented["chunk_mask"]
        z_e = torch.where(chunk_mask.unsqueeze(-1), z_e, torch.zeros_like(z_e))
        # Preserve the exact codebook used for hard assignments. EMA mutates the
        # shared table inside VectorQuantizer before returning.
        codebook_snapshot = self.quantizer.codebook.weight.detach().clone()
        quantized = self.quantizer(z_e, valid_mask=chunk_mask)
        quantized.pop("distances", None)
        # Recompute from that immutable snapshot so soft code targets match the
        # hard assignment and can differentiate through z_e safely.
        code_distances = (
            z_e.square().sum(dim=-1, keepdim=True)
            - 2.0 * z_e @ codebook_snapshot.t()
            + codebook_snapshot.square().sum(dim=-1)
        )
        code_distances = torch.where(
            chunk_mask.unsqueeze(-1),
            code_distances,
            torch.zeros_like(code_distances),
        )
        chunk_lengths = segmented["greedy_chosen_lengths"]
        length_probabilities = segmented["greedy_length_probabilities"]
        prior = self.chunk_prior(
            quantized["indices"],
            chunk_lengths,
            chunk_mask,
            length_probabilities,
        )
        text_logits = self.text_decoder(
            input_ids,
            segmented["segment_ids"],
            quantized["z_q_st"],
            chunk_lengths,
            valid_mask,
        )
        return {
            **segmented,
            **prior,
            "text_logits": text_logits,
            "z_e": z_e,
            "z_q_raw": quantized["z_q_raw"],
            "z_q_st": quantized["z_q_st"],
            "indices": quantized["indices"],
            "code_distances": code_distances,
            "chunk_lengths": chunk_lengths,
            "length_probabilities": length_probabilities,
            "latent_mask": chunk_mask,
        }

    @torch.no_grad()
    def update_rate_dual(self, hard_chunks_per_token: float) -> float:
        target = 1.0 / self.config.compression_target
        self.rate_dual.add_(
            self.config.rate_dual_lr * (hard_chunks_per_token - target)
        )
        self.rate_dual.clamp_(
            -self.config.rate_dual_max_abs,
            self.config.rate_dual_max_abs,
        )
        return float(self.rate_dual)

    def prior_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.chunk_prior.parameters())

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def end_to_end_tokenizer_losses(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    attention_mask: torch.Tensor,
    model: EndToEndTokenizerModel,
    *,
    prior_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Hard-forward codelength with local-softmax gradients for discrete choices."""
    if not 0.0 <= prior_weight <= 1.0:
        raise ValueError("prior_weight must be in [0, 1].")
    valid_mask = attention_mask.to(device=targets.device, dtype=torch.bool)
    chunk_mask = outputs["latent_mask"].bool()
    token_count = valid_mask.sum().clamp_min(1)

    length_targets = outputs["chunk_lengths"] - 1
    hard_length_sum = F.cross_entropy(
        outputs["length_logits"][chunk_mask],
        length_targets[chunk_mask],
        reduction="sum",
    )
    length_log_probabilities = F.log_softmax(outputs["length_logits"], dim=-1)
    soft_length_sum = -(
        outputs["length_probabilities"]
        * length_log_probabilities
    ).sum(dim=-1)[chunk_mask].sum()
    length_nll_sum = soft_length_sum + (
        hard_length_sum - soft_length_sum
    ).detach()

    hard_code_sum = F.cross_entropy(
        outputs["code_logits"][chunk_mask],
        outputs["indices"][chunk_mask],
        reduction="sum",
    )
    valid_distances = outputs["code_distances"][chunk_mask]
    nearest_distances, nearest_indices = torch.topk(
        valid_distances,
        k=model.config.code_target_topk,
        dim=-1,
        largest=False,
    )
    posterior = torch.softmax(
        -nearest_distances / model.config.code_target_temperature,
        dim=-1,
    )
    code_log_probabilities = F.log_softmax(outputs["code_logits"][chunk_mask], dim=-1)
    candidate_log_probabilities = code_log_probabilities.gather(
        1,
        nearest_indices,
    )
    soft_code_sum = -(posterior * candidate_log_probabilities).sum()
    code_nll_sum = soft_code_sum + (hard_code_sum - soft_code_sum).detach()

    text_nll_sum = F.cross_entropy(
        outputs["text_logits"][valid_mask],
        targets[valid_mask],
        reduction="sum",
    )
    commitment_raw = (
        (outputs["z_e"] - outputs["z_q_raw"].detach()).square()[chunk_mask].mean()
    )

    hard_chunk_count = chunk_mask.sum().to(outputs["z_e"].dtype)
    soft_chunk_count = outputs["soft_chunk_counts"].sum()
    aligned_chunk_count = soft_chunk_count + (
        hard_chunk_count - soft_chunk_count
    ).detach()
    hard_chunks_per_token = hard_chunk_count / token_count
    aligned_chunks_per_token = aligned_chunk_count / token_count
    target_chunks_per_token = 1.0 / model.config.compression_target
    rate_constraint = model.rate_dual.detach() * (
        aligned_chunks_per_token - target_chunks_per_token
    )

    generative_nll_sum = length_nll_sum + code_nll_sum + text_nll_sum
    generative_nll_per_bpe = generative_nll_sum / token_count
    prior_nll_sum = length_nll_sum + code_nll_sum
    training_nll_per_bpe = (
        text_nll_sum + prior_weight * prior_nll_sum
    ) / token_count
    commitment = model.config.commitment_beta * commitment_raw
    loss = training_nll_per_bpe + commitment + rate_constraint
    return {
        "loss": loss,
        "training_nll_per_bpe": training_nll_per_bpe,
        "generative_nll_sum": generative_nll_sum,
        "generative_nll_per_bpe": generative_nll_per_bpe,
        "length_nll_sum": length_nll_sum,
        "length_nll_per_bpe": length_nll_sum / token_count,
        "code_nll_sum": code_nll_sum,
        "code_nll_per_bpe": code_nll_sum / token_count,
        "text_nll_sum": text_nll_sum,
        "text_nll_per_bpe": text_nll_sum / token_count,
        "commitment_loss": commitment_raw,
        "commitment_weighted_loss": commitment,
        "rate_constraint_loss": rate_constraint,
        "hard_chunks_per_token": hard_chunks_per_token,
        "hard_tokens_per_chunk": token_count / hard_chunk_count.clamp_min(1.0),
        "aligned_chunks_per_token": aligned_chunks_per_token,
        "rate_dual": model.rate_dual.detach().clone(),
        "prior_weight": text_nll_sum.new_tensor(prior_weight),
        "token_count": token_count,
        "chunk_count": hard_chunk_count,
    }
