"""Shared helpers for MNIST visualization scripts."""

import matplotlib.pyplot as plt
import torch

from common.data import get_test_loader


def load_state_dict(model, pth_path, device):
    model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
    model.eval()
    return model


def first_test_batch(batch_size=64, num_items=None):
    images, labels = next(iter(get_test_loader(batch_size=batch_size)))
    if num_items is not None:
        images = images[:num_items]
        labels = labels[:num_items]
    return images, labels


def denormalize(images):
    return ((images + 1) / 2).cpu().clamp(0, 1)


def show_reconstruction_grid(images, reconstructed, labels=None, title=None):
    num_pairs = images.size(0)
    fig, axes = plt.subplots(num_pairs, 2, figsize=(4, num_pairs * 2))

    if num_pairs == 1:
        axes = axes.reshape(1, 2)

    for i in range(num_pairs):
        ax_orig = axes[i, 0]
        ax_orig.imshow(images[i].squeeze(), cmap="gray")
        if labels is None:
            ax_orig.set_title("Original")
        else:
            ax_orig.set_title(f"Original (label={labels[i].item()})")
        ax_orig.axis("off")

        ax_recon = axes[i, 1]
        ax_recon.imshow(reconstructed[i].squeeze(), cmap="gray")
        ax_recon.set_title("Reconstructed")
        ax_recon.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def collect_latents(model, device, encode_fn, batch_size=256):
    all_latents = []
    all_labels = []
    for images, labels in get_test_loader(batch_size=batch_size):
        latents = encode_fn(model, images.to(device))
        all_latents.append(latents.cpu())
        all_labels.append(labels)

    return torch.cat(all_latents, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()


def plot_latent_scatter(latents, labels, title):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        latents[:, 0], latents[:, 1],
        c=labels, cmap="tab10", s=2, alpha=0.7,
    )
    plt.colorbar(scatter, ticks=range(10), label="Digit class")
    plt.clim(-0.5, 9.5)
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.title(title)
    plt.tight_layout()
    plt.show()
