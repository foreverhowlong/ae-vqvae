# AE / VAE / VQ-VAE on MNIST

PyTorch implementations of Autoencoder, Variational Autoencoder, and Vector Quantized VAE, trained on MNIST with a 2D latent space for visualization.

## Structure

- `models/` — model definitions (`ae.py`, `vae.py`, `vqvae.py`)
- `common/` — shared project paths, device detection, MNIST loaders, VQ-VAE experiment helpers
- `training/` — training scripts for each model
- `training/run_text_vqvae_experiment.py` — TinyStories byte-level language compression experiment with a non-autoregressive VQ-VAE decoder
- `visualization/` — five visualization scripts:
  - `visualize_ae.py` / `visualize_vae.py` / `visualize_vqvae.py` — reconstruction comparison (original vs. decoded) and 2D latent space visualization
  - `explore_ae_latent.py` / `explore_vae_latent.py` — interactive 2D latent space exploration with draggable knob and real-time decoding
- `outputs/` — saved model checkpoints

## Usage

```bash
# train
python -m training.trainAE
python -m training.trainVAE
python -m training.trainVQVAE

# run VQ-VAE collapse experiments
python -m training.run_collapse_experiments --dry-run

# run TinyStories language compression experiment
# Requires the optional HuggingFace `datasets` package unless --data-file is used.
python -m training.run_text_vqvae_experiment --run-name tinystories_vqvae_baseline

# quick local smoke run from a .txt/.jsonl file
python -m training.run_text_vqvae_experiment --data-file path/to/text.txt --max-train-samples 2000 --epochs 1

# visualize
python -m visualization.visualize_ae
python -m visualization.visualize_vae
python -m visualization.visualize_vqvae
python -m visualization.explore_ae_latent
python -m visualization.explore_vae_latent
```

## Dependencies

Managed via `uv` (`pyproject.toml`). Core: `torch`, `torchvision`, `matplotlib`.
