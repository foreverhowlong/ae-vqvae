"""Codebook initialisation strategies for text VQ-VAE."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


@torch.no_grad()
def initialize_codebook_kmeans(model, data_loader, device, seed: int) -> dict:
    """Fit MiniBatch KMeans on one encoder pass and copy centres to the codebook.

    Only content (non-PAD) latent slots are included in the KMeans fit so that
    PAD-dominated segments do not anchor a cluster centre at zero.

    Returns a dict with metadata suitable for storing in config.json.
    """
    codebook_size = model.config.codebook_size
    kmeans = MiniBatchKMeans(
        n_clusters=codebook_size,
        init="random",
        n_init=1,
        random_state=seed,
        reassignment_ratio=0.0,
    )
    was_training = model.training
    model.eval()
    pending: list[np.ndarray] = []
    pending_count = 0
    vectors_seen = 0
    fitted = False

    try:
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            z_e, latent_mask = model.encode(
                input_ids, attention_mask=attention_mask, return_mask=True
            )
            vectors = z_e[latent_mask].detach().float().cpu().numpy()
            vectors_seen += len(vectors)

            if len(vectors) == 0:
                continue

            if not fitted:
                pending.append(vectors)
                pending_count += len(vectors)
                if pending_count < codebook_size:
                    continue
                vectors = np.concatenate(pending, axis=0)
                pending.clear()
                fitted = True

            kmeans.partial_fit(vectors)
    finally:
        model.train(was_training)

    if not fitted:
        raise ValueError(
            "KMeans codebook initialisation needs at least as many encoder vectors as "
            f"codes, but the first pass produced {vectors_seen} vectors for "
            f"{codebook_size} codes."
        )

    centers = torch.as_tensor(
        kmeans.cluster_centers_,
        device=model.quantizer.codebook.weight.device,
        dtype=model.quantizer.codebook.weight.dtype,
    )
    if model.collapse_config.normalize_latents:
        centers = torch.nn.functional.normalize(centers, dim=-1)
    model.quantizer.codebook.weight.copy_(centers)

    # Keep EMA buffers consistent so the first update does not restore old random values.
    if hasattr(model.quantizer, "ema_embed_sum"):
        model.quantizer.ema_embed_sum.copy_(centers)
        model.quantizer.ema_cluster_size.fill_(1.0)

    return {"method": "kmeans", "encoder_vectors": vectors_seen}
