"""Visualize VAE reconstructions and 2D latent space."""

import torch

from common import ROOT, get_device
from models.vae import VAE
from visualization.common import (
    collect_latents,
    denormalize,
    first_test_batch,
    load_state_dict,
    plot_latent_scatter,
    show_reconstruction_grid,
)

model_path = ROOT / "outputs/vae2.pth"


def load_model(pth_path=model_path):
    device = get_device()
    model = load_state_dict(VAE().to(device), pth_path, device)
    return model, device


def show_reconstruction(pth_path=model_path, num_pairs=8):
    model, device = load_model(pth_path)
    images, labels = first_test_batch(batch_size=64, num_items=num_pairs)
    images = images.to(device)

    with torch.no_grad():
        reconstructed, _, _ = model(images)

    show_reconstruction_grid(
        denormalize(images),
        denormalize(reconstructed),
        labels=labels,
        title="VAE Reconstruction Comparison",
    )


def plot_latent_space(pth_path=model_path):
    model, device = load_model(pth_path)
    latents, labels = collect_latents(
        model, device, encode_fn=lambda m, x: m.encoder(x)[0],
    )
    plot_latent_scatter(latents, labels, "VAE Latent Space (2D) - MNIST Test Set")


if __name__ == "__main__":
    show_reconstruction()
    plot_latent_space()
