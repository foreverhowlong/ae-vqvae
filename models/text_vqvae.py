"""Non-autoregressive VQ-VAE for byte-level text compression."""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TextVQVAEConfig:
    vocab_size: int = 258
    max_seq_len: int = 256
    latent_slots: int = 32
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
    """Reserved extension hooks for later anti-collapse experiments."""

    enabled: bool = False
    entropy_weight: float = 0.0
    code_dropout: float = 0.0
    reset_dead_codes: bool = False
    ema_codebook: bool = False

    def to_dict(self):
        return asdict(self)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, d_model: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.codebook = nn.Embedding(codebook_size, d_model)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=d_model**-0.5)

    def forward(self, z_e: torch.Tensor):
        flat = z_e.reshape(-1, self.d_model)
        weights = self.codebook.weight

        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ weights.t()
            + weights.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = distances.argmin(dim=-1)
        z_q_raw = self.codebook(indices).view_as(z_e)
        z_q_st = z_e + (z_q_raw - z_e).detach()
        return z_q_raw, z_q_st, indices.view(z_e.shape[0], z_e.shape[1])


class TextVQVAE(nn.Module):
    """Compresses a byte sequence into discrete latent slots and decodes in parallel."""

    def __init__(self, config: TextVQVAEConfig):
        super().__init__()
        self.config = config
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
        self.quantizer = VectorQuantizer(config.codebook_size, config.d_model)

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
        z_q_raw, z_q_st, indices = self.quantizer(z_e)
        logits = self.decode(z_q_st, seq_len=input_ids.shape[1])
        return {
            "logits": logits,
            "z_e": z_e,
            "z_q_raw": z_q_raw,
            "z_q_st": z_q_st,
            "indices": indices,
        }


def text_vqvae_losses(outputs, targets, pad_token_id: int, beta: float):
    logits = outputs["logits"]
    recon_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_token_id,
    )
    z_e = outputs["z_e"]
    z_q_raw = outputs["z_q_raw"]
    codebook_loss = F.mse_loss(z_q_raw, z_e.detach())
    commitment_loss = F.mse_loss(z_e, z_q_raw.detach())
    total = recon_loss + codebook_loss + beta * commitment_loss
    return total, recon_loss, codebook_loss, commitment_loss


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
