"""Non-autoregressive VQ-VAE for byte-level text compression."""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TextVQVAEConfig:
    vocab_size: int = 258
    max_seq_len: int = 256
    latent_slots: int = 128
    d_model: int = 448
    n_heads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 6
    ffn_mult: int = 4
    dropout: float = 0.1
    codebook_size: int = 1024
    commitment_beta: float = 0.25
    pad_token_id: int = 257

    def to_dict(self):
        return asdict(self)


@dataclass
class CollapseControlConfig:
    """Engineering controls commonly used to reduce codebook collapse."""

    enabled: bool = False
    use_ema_codebook: bool = False
    ema_decay: float = 0.99
    ema_eps: float = 1e-5
    entropy_weight: float = 0.0
    entropy_temperature: float = 1.0
    diversity_weight: float = 0.0
    code_dropout: float = 0.0
    stochastic_code_sampling: bool = False
    sampling_temperature: float = 0.5
    sampling_topk: int = 8
    dead_code_reset_every: int = 0
    dead_code_reset_usage_threshold: float = 1.0
    normalize_latents: bool = False
    commitment_beta_start: float | None = None
    commitment_beta_warmup_steps: int = 0

    def to_dict(self):
        return asdict(self)


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

    def forward(self, z_e: torch.Tensor):
        flat = z_e.reshape(-1, self.d_model)
        weights = self.codebook.weight

        distance_flat = flat
        distance_weights = weights
        if self.collapse_config.normalize_latents:
            distance_flat = F.normalize(distance_flat, dim=-1)
            distance_weights = F.normalize(distance_weights, dim=-1)

        distances = (
            distance_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * distance_flat @ distance_weights.t()
            + distance_weights.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = self._select_codes(distances)
        z_q_raw = self.codebook(indices).view_as(z_e)

        if self.training and self.collapse_config.use_ema_codebook:
            self._ema_update(flat.detach(), indices.detach())

        z_q_st = z_e + (z_q_raw - z_e).detach()
        return {
            "z_q_raw": z_q_raw,
            "z_q_st": z_q_st,
            "indices": indices.view(z_e.shape[0], z_e.shape[1]),
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
    def reset_dead_codes(self, z_e: torch.Tensor, usage_threshold: float = 1.0):
        flat = z_e.reshape(-1, self.d_model).detach()
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
        self.config = config
        self.collapse_config = collapse_config or CollapseControlConfig()
        ffn_dim = config.d_model * config.ffn_mult

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder_pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.decoder_pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            enable_nested_tensor=False,
        )
        self.encoder_norm = nn.LayerNorm(config.d_model)

        self.latent_proj = nn.Linear(config.d_model, config.d_model)
        self.quantizer = VectorQuantizer(config.codebook_size, config.d_model, self.collapse_config)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)
        self.decoder_norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.token_embedding(input_ids) + self.encoder_pos_embedding(positions)

        padding_mask = None
        if attention_mask is not None:
            padding_mask = attention_mask == 0

        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        hidden = self.encoder_norm(hidden)

        latents = F.adaptive_avg_pool1d(
            hidden.transpose(1, 2), self.config.latent_slots
        ).transpose(1, 2)
        return self.latent_proj(latents)

    def decode(self, z_q_st: torch.Tensor, seq_len: int):
        batch_size = z_q_st.shape[0]
        positions = torch.arange(seq_len, device=z_q_st.device).unsqueeze(0).expand(batch_size, -1)
        queries = self.decoder_pos_embedding(positions)
        decoded = self.decoder(tgt=queries, memory=z_q_st)
        decoded = self.decoder_norm(decoded)
        return self.output_head(decoded)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        z_e = self.encode(input_ids, attention_mask=attention_mask)
        quantized = self.quantizer(z_e)
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
        }


def assignment_entropy_loss(outputs, temperature: float, codebook_size: int):
    distances = outputs["distances"]
    probs = torch.softmax(-distances / max(temperature, 1e-6), dim=-1)
    avg_probs = probs.mean(dim=(0, 1))
    entropy = -(avg_probs * torch.log(avg_probs + 1e-12)).sum()
    normalized_entropy = entropy / torch.log(torch.tensor(float(codebook_size), device=distances.device))
    return -normalized_entropy, normalized_entropy.detach()


def codebook_diversity_loss(codebook_weight: torch.Tensor):
    normalized = F.normalize(codebook_weight, dim=-1)
    gram = normalized @ normalized.t()
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=torch.bool)
    off_diag = gram.masked_select(~eye)
    return off_diag.pow(2).mean()


def text_vqvae_losses(
    outputs,
    targets,
    pad_token_id: int,
    beta: float,
    collapse_config: CollapseControlConfig | None = None,
    codebook_weight: torch.Tensor | None = None,
):
    collapse_config = collapse_config or CollapseControlConfig()
    logits = outputs["logits"]
    recon_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_token_id,
    )
    z_e = outputs["z_e"]
    z_q_raw = outputs["z_q_raw"]
    if collapse_config.use_ema_codebook:
        codebook_loss = torch.zeros((), device=z_e.device)
    else:
        codebook_loss = F.mse_loss(z_q_raw, z_e.detach())
    commitment_loss = F.mse_loss(z_e, z_q_raw.detach())
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
def codebook_stats(indices: torch.Tensor, codebook_size: int):
    flat = indices.reshape(-1).detach().cpu()
    counts = torch.bincount(flat, minlength=codebook_size).float()
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
