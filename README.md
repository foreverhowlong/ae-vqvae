# Text VQ-VAE — TinyStories Compression Research

Non-autoregressive VQ-VAE for text compression, with a focus on studying codebook collapse dynamics. Trained on TinyStories using a dataset-specific BPE tokenizer.

## Structure

### Active research line (text-vqvae)

- `models/text_vqvae.py` — model definition: transformer encoder → VectorQuantizer → memory-trunk decoder
- `common/` — shared utilities (paths, device detection, wandb tracking, text data loading)
- `training/run_text_vqvae_experiment.py` — main training entry point
- `training/text_vqvae/` — sub-package: config, training loop, codebook init, reporting
- `training/train_tokenizer.py` — trains a byte-level BPE tokenizer on a text dataset
- `training/run_experiment_sequence.py` — runs a JSON-configured sequence of experiments
- `visualization/text_vqvae.py` — PCA diagnostics and encoder/codebook distribution plots
- `tests/` — unit and integration tests
- `configs/` — example experiment sequence JSON configs

### Legacy (AE / VAE / VQ-VAE on MNIST)

`legacy/` contains the original learning-project code. Entry points are renamed:

```bash
python -m legacy.training.trainAE
python -m legacy.training.trainVAE
python -m legacy.training.trainVQVAE
python -m legacy.training.run_collapse_experiments
```

## Usage

All training entry points report configs and metrics to Weights & Biases. Put the API key
in the ignored root `.env` file (project/entity settings are optional):

```dotenv
WANDB_API_KEY=your_api_key
WANDB_PROJECT=ae-vqvae
# WANDB_ENTITY=your_team_or_username
```

### Train

```bash
# run TinyStories compression with the trained 8K BPE tokenizer (default)
python -m training.run_text_vqvae_experiment --run-name my_run

# choose decoder type: cross_attention (original) or memory_trunk (current best)
python -m training.run_text_vqvae_experiment --decoder-type memory_trunk

# use kmeans codebook initialization instead of random
python -m training.run_text_vqvae_experiment --codebook-init kmeans

# enable all anti-collapse measures (EMA, dead-code reset, entropy loss, …)
python -m training.run_text_vqvae_experiment --collapse-preset anti

# quick offline smoke test from a local file
python -m training.run_text_vqvae_experiment \
    --data-file path/to/text.txt --tokenizer byte \
    --epochs 1 --max-train-samples 200 --batch-size 8

# run a configured sequence of experiments
python -m training.run_experiment_sequence --config configs/text_vqvae_experiments.example.json
python -m training.run_experiment_sequence --config configs/text_vqvae_experiments.example.json --dry-run
```

### Train tokenizer

```bash
# train an 8K byte-level BPE tokenizer on the full TinyStories train split
python -m training.train_tokenizer

# quick streaming smoke run
python -m training.train_tokenizer --streaming --max-samples 1000 --output-dir outputs/tokenizers/smoke
```

### Visualize

```bash
python -m visualization.text_vqvae   # initial PCA diagnostic (also run automatically at training start)
```

### Sync outputs from remote host

```bash
scripts/sync_outputs_from_mech.sh               # sync all outputs
scripts/sync_outputs_from_mech.sh --best-only   # only pull best.pt among *.pt files
scripts/sync_outputs_from_mech.sh --no-models   # skip weight files
scripts/sync_outputs_from_mech.sh --latest-only # sync only the newest run directory
```

Both text-data training entries read `HF_TOKEN` from the process environment first,
then fall back to the ignored project-root `.env` file. The token is never written to
run configs, tokenizer artifacts, or repository files.

## Dependencies

Managed via `uv` (`pyproject.toml`). Core: `torch`, `torchvision`, `matplotlib`, `tokenizers`, `datasets`, `wandb`, `scikit-learn`.
