# AE / VAE / VQ-VAE on MNIST

PyTorch implementations of Autoencoder, Variational Autoencoder, and Vector Quantized VAE, trained on MNIST with a 2D latent space for visualization.

## Structure

- `models/` — model definitions (`ae.py`, `vae.py`, `vqvae.py`)
- `common/` — shared project paths, device detection, MNIST loaders, VQ-VAE experiment helpers
- `training/` — training scripts for each model
- `training/run_text_vqvae_experiment.py` — TinyStories byte-level language compression experiment with a non-autoregressive VQ-VAE decoder
- `training/train_tokenizer.py` — trains a byte-level BPE tokenizer on TinyStories
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
python -m training.run_text_vqvae_experiment --run-name tinystories_vqvae_baseline

# train an 8K byte-level BPE tokenizer on the full TinyStories train split
# The first run downloads it to data/huggingface; later runs reuse that cache.
python -m training.train_tokenizer

# authenticated download (enter the token at the hidden prompt; it is not logged)
read -s HF_TOKEN
export HF_TOKEN
python -m training.train_tokenizer
unset HF_TOKEN

# alternatively, put this entry in the ignored project-root .env file:
# HF_TOKEN=hf_your_read_token
# python -m training.train_tokenizer will load it automatically

# quick tokenizer smoke run without downloading the complete dataset
python -m training.train_tokenizer --streaming --max-samples 1000 --output-dir outputs/tokenizers/tinystories_bpe_smoke

# offline tokenizer smoke run from a local .txt/.jsonl file
python -m training.train_tokenizer --data-file path/to/text.txt --max-samples 1000 --output-dir outputs/tokenizers/local_bpe_smoke

# run TinyStories language compression with common anti-collapse measures enabled
python -m training.run_text_vqvae_experiment --run-name tinystories_vqvae_anti --collapse-preset anti

# quick local smoke run from a .txt/.jsonl file
python -m training.run_text_vqvae_experiment --data-file path/to/text.txt --max-train-samples 2000 --epochs 1

# sync outputs from the Tailscale host configured as `mech` in ~/.ssh/config
scripts/sync_outputs_from_mech.sh

# sync all outputs, but only pull best.pt among *.pt model files
scripts/sync_outputs_from_mech.sh --best-only

# sync outputs without model/checkpoint weight files
scripts/sync_outputs_from_mech.sh --no-models

# sync only the newest output run directory
scripts/sync_outputs_from_mech.sh --latest-only

# visualize
python -m visualization.visualize_ae
python -m visualization.visualize_vae
python -m visualization.visualize_vqvae
python -m visualization.explore_ae_latent
python -m visualization.explore_vae_latent
```

Both TinyStories training entries read `HF_TOKEN` from the process environment first,
then fall back to the ignored project-root `.env` file. The token is never written to
run configs, tokenizer artifacts, or repository files.

## Dependencies

Managed via `uv` (`pyproject.toml`). Core: `torch`, `torchvision`, `matplotlib`.
