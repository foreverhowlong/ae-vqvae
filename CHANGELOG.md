# Changelog

## 2026-07-12 — Text-VQVAE主线化重构

### Breaking changes (internal)

* `initialize_codebook_from_first_encoder_pass` renamed to
  `initialize_codebook_kmeans` and moved to
  `training.text_vqvae.codebook_init`. Tests updated accordingly.
* `compute_accuracy` moved to `training.text_vqvae.loop`.
* `atomic_json_dump`, `append_jsonl`, `plot_training_curves`,
  `plot_codebook_usage`, `write_reconstruction_samples` moved to
  `training.text_vqvae.reporting`.

### Default value alignment

Two `TextVQVAEConfig` dataclass defaults were wrong (they differed from the
CLI values that all real runs have used):

| Field | Old dataclass default | New default | Actual runs used |
|---|---|---|---|
| `latent_slots` | 128 | **32** | 32 (all runs) |
| `codebook_size` | 1024 | **3072** | 3072 (all runs) |

The dataclass defaults now match reality. The CLI `argparse` defaults are all
`None` so the dataclass values are the single source of truth.

**Note**: the `latent_slots=32` vs `128` choice is a research decision (see
technical summary §2.4 on capacity), not resolved by this refactor. To run
with S=128 pass `--latent-slots 128` explicitly.

### Legacy isolation

MNIST/image-line code moved to `legacy/` package:
- `legacy/models/{ae,vae,vqvae}.py`
- `legacy/training/{trainAE,trainVAE,trainVQVAE,...}.py`
- `legacy/visualization/...`
- `legacy/analysis/intrinsic_dimension.py`
- `legacy/common/{data,experiment}.py`

Entry points now: `python -m legacy.training.trainVQVAE` etc.

### PAD masking (committed just before this refactor)

* `pad_aware_adaptive_pool1d`: pools only valid (non-PAD) tokens per bin.
* VQ encoder `latent_mask` propagates through quantizer, losses, and
  codebook stats — PAD slots excluded from all training signals.
* Recon CE uses boolean index on valid tokens (not `ignore_index`).
* Codebook/commitment MSE uses `_masked_vector_mse`.

### Config provenance

`config.json` now includes:
* `config_version: 1`
* `git_commit` (SHA of HEAD at run time, or null if not in a git repo)
* `git_dirty` (bool)

`training.text_vqvae.config.load_run_config(path)` reconstructs config
dataclasses from any historical `config.json`, filling missing keys from
current defaults with a printed warning.

### Flag discipline documented

See `training/text_vqvae/config.py` module docstring for the full
make-default vs. keep-as-flag decision rules.
