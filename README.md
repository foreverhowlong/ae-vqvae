# AE / VAE / VQ-VAE on MNIST

PyTorch implementations of Autoencoder, Variational Autoencoder, and Vector Quantized VAE, trained on MNIST with a 2D latent space for visualization.

## Structure

- `models/` — model definitions (`ae.py`, `vae.py`, `vqvae.py`)
- `training/` — training scripts for each model
- `visualization/` — five visualization scripts:
  - `visualize_ae.py` / `visualize_vae.py` / `visualize_vqvae.py` — reconstruction comparison (original vs. decoded) and 2D latent space visualization
  - `explore_ae_latent.py` / `explore_vae_latent.py` / `visualize_vqvae.py`— interactive 2D latent space exploration with draggable knob and real-time decoding
- `outputs/` — saved model checkpoints

## Usage

```bash
# train
python main.py

# visualize
python -m visualization.visualize_vqvae
python -m visualization.explore_vae_latent
```

## Dependencies

Managed via `uv` (`pyproject.toml`). Core: `torch`, `torchvision`, `matplotlib`.