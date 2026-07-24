"""Non-autoregressive VQ-VAE for byte-level text compression."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.text_vqvae_config import CollapseControlConfig, TextVQVAEConfig
from models.text_decoders import (
    DECODER_TYPES,
    CrossAttentionTextDecoder,
    MemoryTrunkTextDecoder,
    SubPixelSequenceUpsampler,
    TextDecoder,
    VQGANPreAttentionTextDecoder,
    VQGANTextDecoder,
    build_text_decoder,
)
from models.text_encoders import (
    ENCODER_TYPES,
    AbsoluteTextEncoder,
    RotaryTextEncoder,
    TextEncoder,
    VQGANPreAttentionTextEncoder,
    VQGANTextEncoder,
    build_text_encoder,
    pad_aware_adaptive_pool1d,
)
# Keep architecture classes importable from this module for existing callers.
from models.text_layers import (
    PlainSelfAttention,
    RotaryResidualBlock,
    RotarySelfAttention,
)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, d_model: int, collapse_config: CollapseControlConfig | None = None):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.collapse_config = collapse_config or CollapseControlConfig()
        self.codebook = nn.Embedding(codebook_size, d_model)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=d_model**-0.5)

        if self.collapse_config.use_ema_codebook:
            self.codebook.weight.requires_grad_(False)
            self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
            self.register_buffer("ema_embed_sum", self.codebook.weight.detach().clone())

    def forward(self, z_e: torch.Tensor, valid_mask: torch.Tensor | None = None):
        flat = z_e.reshape(-1, self.d_model)
        if valid_mask is None:
            valid_mask = torch.ones(z_e.shape[:2], dtype=torch.bool, device=z_e.device)
        elif valid_mask.shape != z_e.shape[:2]:
            raise ValueError(
                f"valid_mask must have shape {z_e.shape[:2]}, got {valid_mask.shape}."
            )
        valid_mask = valid_mask.to(device=z_e.device, dtype=torch.bool)
        flat_valid_mask = valid_mask.reshape(-1)
        weights = self.codebook.weight

        # Keep invalid slots out of nearest-neighbour assignment altogether.
        # Full-shaped tensors are reconstructed below for API/checkpoint-tooling
        # compatibility, but invalid rows never gather a codebook vector.
        flat_indices = torch.full(
            (flat.shape[0],), -1, dtype=torch.long, device=z_e.device
        )
        flat_z_q_raw = flat.detach().clone()
        distances = flat.new_zeros((flat.shape[0], self.codebook_size))

        if flat_valid_mask.any():
            distance_flat = flat[flat_valid_mask]
            distance_weights = weights
            if self.collapse_config.normalize_latents:
                distance_flat = F.normalize(distance_flat, dim=-1)
                distance_weights = F.normalize(distance_weights, dim=-1)

            valid_distances = (
                distance_flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * distance_flat @ distance_weights.t()
                + distance_weights.pow(2).sum(dim=1).unsqueeze(0)
            )
            valid_indices = self._select_codes(valid_distances)
            flat_indices[flat_valid_mask] = valid_indices
            flat_z_q_raw[flat_valid_mask] = self.codebook(valid_indices)
            distances[flat_valid_mask] = valid_distances

            if self.training and self.collapse_config.use_ema_codebook:
                self._ema_update(
                    flat[flat_valid_mask].detach(),
                    valid_indices.detach(),
                )

        z_q_raw = flat_z_q_raw.view_as(z_e)

        z_q_st = z_e + (z_q_raw - z_e).detach()
        indices = flat_indices.view(z_e.shape[0], z_e.shape[1])
        return {
            "z_q_raw": z_q_raw,
            "z_q_st": z_q_st,
            "indices": indices,
            "distances": distances.view(z_e.shape[0], z_e.shape[1], self.codebook_size),
        }

    def _select_codes(self, distances: torch.Tensor):
        if self.training and self.collapse_config.stochastic_code_sampling:
            k = min(self.collapse_config.sampling_topk, self.codebook_size)
            top_distances, top_indices = torch.topk(distances, k=k, dim=-1, largest=False)
            temperature = max(self.collapse_config.sampling_temperature, 1e-6)
            logits = -top_distances / temperature
            gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-12)).clamp_min(1e-12))
            sampled_pos = (logits + gumbel).argmax(dim=-1)
            indices = top_indices.gather(1, sampled_pos.unsqueeze(1)).squeeze(1)
        else:
            indices = distances.argmin(dim=-1)

        if self.training and self.collapse_config.code_dropout > 0:
            drop_mask = torch.rand(indices.shape, device=indices.device) < self.collapse_config.code_dropout
            random_indices = torch.randint(0, self.codebook_size, indices.shape, device=indices.device)
            indices = torch.where(drop_mask, random_indices, indices)

        return indices

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, indices: torch.Tensor):
        one_hot = F.one_hot(indices, num_classes=self.codebook_size).type_as(flat)
        cluster_size = one_hot.sum(dim=0)
        embed_sum = one_hot.t() @ flat

        decay = self.collapse_config.ema_decay
        self.ema_cluster_size.mul_(decay).add_(cluster_size, alpha=1 - decay)
        self.ema_embed_sum.mul_(decay).add_(embed_sum, alpha=1 - decay)

        n = self.ema_cluster_size.sum()
        smoothed_cluster_size = (
            (self.ema_cluster_size + self.collapse_config.ema_eps)
            / (n + self.codebook_size * self.collapse_config.ema_eps)
            * n.clamp_min(1.0)
        )
        normalized_embed = self.ema_embed_sum / smoothed_cluster_size.unsqueeze(1).clamp_min(1e-12)
        if self.collapse_config.normalize_latents:
            normalized_embed = F.normalize(normalized_embed, dim=-1)
        self.codebook.weight.data.copy_(normalized_embed)

    @torch.no_grad()
    def reset_dead_codes(
        self,
        z_e: torch.Tensor,
        usage_threshold: float = 1.0,
        valid_mask: torch.Tensor | None = None,
    ):
        flat = z_e.reshape(-1, self.d_model).detach()
        if valid_mask is not None:
            flat = flat[valid_mask.reshape(-1).to(device=flat.device, dtype=torch.bool)]
        if flat.numel() == 0:
            return 0

        if hasattr(self, "ema_cluster_size"):
            dead_mask = self.ema_cluster_size <= usage_threshold
        else:
            return 0

        dead_indices = dead_mask.nonzero(as_tuple=False).flatten()
        if dead_indices.numel() == 0:
            return 0

        sample_indices = torch.randint(0, flat.shape[0], (dead_indices.numel(),), device=flat.device)
        replacements = flat[sample_indices]
        if self.collapse_config.normalize_latents:
            replacements = F.normalize(replacements, dim=-1)

        self.codebook.weight.data[dead_indices] = replacements
        self.ema_embed_sum.data[dead_indices] = replacements
        self.ema_cluster_size.data[dead_indices] = usage_threshold + 1.0
        return int(dead_indices.numel())


class TextVQVAE(nn.Module):
    """Compresses a byte sequence into discrete latent slots and decodes in parallel."""

    def __init__(self, config: TextVQVAEConfig, collapse_config: CollapseControlConfig | None = None):
        super().__init__()
        if not 0.0 <= config.slot_pad_ratio_threshold < 1.0:
            raise ValueError(
                "slot_pad_ratio_threshold must be in [0, 1), got "
                f"{config.slot_pad_ratio_threshold}."
            )
        self.config = config
        self.collapse_config = collapse_config or CollapseControlConfig()

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder: TextEncoder = build_text_encoder(config)
        self.latent_proj = nn.Linear(config.d_model, config.d_model)
        self.quantizer = VectorQuantizer(config.codebook_size, config.d_model, self.collapse_config)
        self.decoder_impl: TextDecoder = build_text_decoder(config)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_mask: bool = False,
    ):
        hidden = self.token_embedding(input_ids)
        if attention_mask is None:
            attention_mask = input_ids != self.config.pad_token_id
        else:
            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        pooled, latent_mask = self.encoder(hidden, attention_mask)
        latents = self.latent_proj(pooled)
        # A fully padded segment is represented by a fixed zero vector, including
        # after the projection bias, so it cannot become a trainable PAD prototype.
        latents = torch.where(latent_mask.unsqueeze(-1), latents, torch.zeros_like(latents))
        if self.config.l2_normalize_before_vq:
            latents = F.normalize(latents, p=2, dim=-1)
        if return_mask:
            return latents, latent_mask
        return latents

    def decode(self, z_q_st: torch.Tensor, seq_len: int):
        return self.output_head(self.decoder_impl(z_q_st, seq_len))

    @property
    def encoder_pos_embedding(self):
        """Compatibility view of the absolute encoder's position embedding."""
        return getattr(self.encoder, "position_embedding", None)

    @property
    def encoder_norm(self):
        """Compatibility view of the normalization now owned by each encoder."""
        return self.encoder.norm

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load current checkpoints and migrate pre-registry encoder/decoder keys."""
        legacy_prefixes = {"encoder_norm.": "encoder.norm."}
        if isinstance(self.encoder, AbsoluteTextEncoder):
            legacy_prefixes.update(
                {
                    "encoder_pos_embedding.": "encoder.position_embedding.",
                    "encoder.layers.": "encoder.transformer.layers.",
                }
            )
        if isinstance(self.decoder_impl, CrossAttentionTextDecoder):
            legacy_prefixes.update(
                {
                    "decoder_pos_embedding.": "decoder_impl.position_embedding.",
                    "decoder.": "decoder_impl.transformer.",
                    "decoder_norm.": "decoder_impl.norm.",
                }
            )

        if any(
            key.startswith(prefix)
            for key in state_dict
            for prefix in legacy_prefixes
        ):
            metadata = getattr(state_dict, "_metadata", None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata
            for key in list(state_dict):
                for old_prefix, new_prefix in legacy_prefixes.items():
                    if key.startswith(old_prefix):
                        state_dict[new_prefix + key.removeprefix(old_prefix)] = state_dict.pop(key)
                        break
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        if attention_mask is None:
            resolved_attention_mask = input_ids != self.config.pad_token_id
        else:
            resolved_attention_mask = attention_mask.to(
                device=input_ids.device, dtype=torch.bool
            )
        lengths = resolved_attention_mask.sum(dim=-1)
        z_e, latent_mask = self.encode(
            input_ids,
            attention_mask=resolved_attention_mask,
            return_mask=True,
        )
        quantized = self.quantizer(z_e, valid_mask=latent_mask)
        z_q_raw = quantized["z_q_raw"]
        z_q_st = quantized["z_q_st"]
        indices = quantized["indices"]
        logits = self.decode(z_q_st, seq_len=input_ids.shape[1])
        return {
            "logits": logits,
            "z_e": z_e,
            "z_q_raw": z_q_raw,
            "z_q_st": z_q_st,
            "indices": indices,
            "distances": quantized["distances"],
            "latent_mask": latent_mask,
            "lengths": lengths,
        }

    @torch.no_grad()
    def infer(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """Run reconstruction and expose only defined, per-example logits.

        Training keeps a dense ``[B, L, V]`` tensor. Inference returns a tuple
        of ``[length, V]`` tensors so callers cannot accidentally consume the
        undefined PAD region.
        """
        was_training = self.training
        self.eval()
        try:
            outputs = self.forward(input_ids, attention_mask=attention_mask)
            dense_logits = outputs["logits"]
            outputs["logits"] = tuple(
                sample_logits[:length]
                for sample_logits, length in zip(
                    dense_logits, outputs["lengths"].tolist()
                )
            )
            return outputs
        finally:
            self.train(was_training)


def assignment_entropy_loss(outputs, temperature: float, codebook_size: int):
    distances = outputs["distances"]
    probs = torch.softmax(-distances / max(temperature, 1e-6), dim=-1)
    latent_mask = outputs.get("latent_mask")
    if latent_mask is not None:
        probs = probs[latent_mask]
    else:
        probs = probs.reshape(-1, probs.shape[-1])
    if probs.numel() == 0:
        zero = distances.sum() * 0.0
        return zero, zero.detach()
    avg_probs = probs.mean(dim=0)
    entropy = -(avg_probs * torch.log(avg_probs + 1e-12)).sum()
    normalized_entropy = entropy / torch.log(torch.tensor(float(codebook_size), device=distances.device))
    return -normalized_entropy, normalized_entropy.detach()


def codebook_diversity_loss(codebook_weight: torch.Tensor):
    normalized = F.normalize(codebook_weight, dim=-1)
    gram = normalized @ normalized.t()
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=torch.bool)
    off_diag = gram.masked_select(~eye)
    return off_diag.pow(2).mean()


def _masked_vector_mse(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    squared_error = (inputs - targets).pow(2)
    if valid_mask is None:
        return squared_error.mean()
    valid_mask = valid_mask.to(device=inputs.device, dtype=torch.bool)
    if not valid_mask.any():
        return inputs.sum() * 0.0
    return squared_error[valid_mask].mean()


def text_vqvae_losses(
    outputs,
    targets,
    pad_token_id: int,
    beta: float,
    collapse_config: CollapseControlConfig | None = None,
    codebook_weight: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
):
    collapse_config = collapse_config or CollapseControlConfig()
    logits = outputs["logits"]
    if attention_mask is None:
        # Compatibility path for external/legacy callers. The main training
        # path passes the independently constructed token-level mask.
        valid_tokens = targets != pad_token_id
    else:
        if attention_mask.shape != targets.shape:
            raise ValueError(
                f"attention_mask must have shape {targets.shape}, got {attention_mask.shape}."
            )
        valid_tokens = attention_mask.to(device=targets.device, dtype=torch.bool)
    if valid_tokens.any():
        recon_loss = F.cross_entropy(logits[valid_tokens], targets[valid_tokens])
    else:
        recon_loss = logits.sum() * 0.0
    z_e = outputs["z_e"]
    z_q_raw = outputs["z_q_raw"]
    latent_mask = outputs.get("latent_mask")
    if collapse_config.use_ema_codebook:
        codebook_loss = torch.zeros((), device=z_e.device)
    else:
        codebook_loss = _masked_vector_mse(z_q_raw, z_e.detach(), latent_mask)
    commitment_loss = _masked_vector_mse(z_e, z_q_raw.detach(), latent_mask)
    entropy_loss = torch.zeros((), device=z_e.device)
    normalized_entropy = torch.zeros((), device=z_e.device)
    if collapse_config.entropy_weight > 0:
        entropy_loss, normalized_entropy = assignment_entropy_loss(
            outputs,
            temperature=collapse_config.entropy_temperature,
            codebook_size=outputs["distances"].shape[-1],
        )

    diversity_loss = torch.zeros((), device=z_e.device)
    if collapse_config.diversity_weight > 0 and codebook_weight is not None:
        diversity_loss = codebook_diversity_loss(codebook_weight)

    total = (
        recon_loss
        + codebook_loss
        + beta * commitment_loss
        + collapse_config.entropy_weight * entropy_loss
        + collapse_config.diversity_weight * diversity_loss
    )
    return {
        "total": total,
        "recon": recon_loss,
        "codebook": codebook_loss,
        "commitment": commitment_loss,
        "entropy": entropy_loss,
        "normalized_assignment_entropy": normalized_entropy,
        "diversity": diversity_loss,
    }


@torch.no_grad()
def codebook_stats(
    indices: torch.Tensor,
    codebook_size: int,
    valid_mask: torch.Tensor | None = None,
):
    if valid_mask is None:
        flat = indices.reshape(-1)
    else:
        flat = indices[valid_mask.to(device=indices.device, dtype=torch.bool)]
    flat = flat[flat >= 0].detach().cpu()
    counts = torch.bincount(flat, minlength=codebook_size).float()
    if flat.numel() == 0:
        return {
            "used_codes": 0,
            "dead_codes": codebook_size,
            "utilization": 0.0,
            "codebook_perplexity": 0.0,
            "counts": counts,
        }
    probs = counts / (counts.sum() + 1e-12)
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    perplexity = torch.exp(entropy).item()
    used = int((counts > 0).sum().item())
    return {
        "used_codes": used,
        "dead_codes": int(codebook_size - used),
        "utilization": used / codebook_size,
        "codebook_perplexity": perplexity,
        "counts": counts,
    }


def count_parameters(model: nn.Module):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
