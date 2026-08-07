# Learned-tokenizer research pipelines

All primary VQ experiments in this set lock `codebook_size=8192`. Expensive
commands are intentionally separate: first train/export tokenizers, then build
immutable token corpora, and only then train the matched nanoGPT models.

## 1. Fixed VQ and Top-k curriculum

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m training.run_experiment_sequence \
  --config configs/topk-curriculum-k8192-20260807.json --dry-run
```

Remove `--dry-run` to launch the nearest-code control followed by the
SAE-inspired `8 -> 4 -> 2 -> 1` curriculum. Both use adaptive AE warmup and
K-means initialization. The curriculum occupies the first 60% of the observed
VQ phase; validation and the last 40% use hard top-1.

## 2. Variable-length GQ-VAE

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m training.run_experiment_sequence \
  --config configs/gqvae-compression-curriculum-k8192-20260807.json --dry-run
```

The four runs sweep the final compression weight `alpha` while keeping the
adaptive AE warmup and codebook fixed. After choosing a checkpoint on the
reconstruction/compression frontier, export its learned dictionary:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m training.export_gqvae_tokenizer \
  --checkpoint outputs/gqvae/<run>/checkpoints/best.pt \
  --output outputs/tokenizers/gqvae-k8192/tokenizer.json
```

The exported vocabulary always includes a byte fallback and round-trips UTF-8
bytes exactly.

## 3. Commitment-beta sweep

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m training.run_experiment_sequence \
  --config configs/commitment-beta-sweep-k8192-20260807.json --dry-run
```

This isolates fixed post-warmup beta values `0.05, 0.1, 0.25, 0.5, 1.0`.
Every cell shares the same adaptive AE warmup; beta is inactive until the VQ
phase begins.

## 4. Matched raw-text corpora

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m training.run_experiment_sequence \
  --config configs/lm-corpus-k8192-20260807.json --dry-run
```

Each corpus contains `train.bin`, `validation.bin`, document offsets, per-doc
raw byte counts, and `meta.json`. The fixed-slot VQ tokenizer uses VQ codes only
for documents that reconstruct exactly; otherwise it records a byte fallback.
This conservative rule prevents lossy token streams from receiving artificially
low bits-per-byte.

## 5. Approximately 18M-parameter nanoGPT comparison

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m training.run_experiment_sequence \
  --config configs/nanogpt-tokenizer-comparison-18m-20260807.json --dry-run
```

The primary scalar is `validation/bits_per_raw_byte`:

```text
sum(next-token NLL in nats) / (ln(2) * validation UTF-8 bytes)
```

The evaluator scores each document with a sliding token context, sums NLL over
all content/EOS predictions, and uses the immutable raw-byte count as the
denominator. Token NLL and perplexity remain diagnostics and must not be used
to rank different tokenizers.

