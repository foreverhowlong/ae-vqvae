# Learned-tokenizer research pipelines

All primary VQ experiments in this set lock `codebook_size=8192`. Expensive
commands are intentionally separate: first train/export tokenizers, then build
immutable token corpora, and only then train the matched nanoGPT models.
The fixed VQ, Top-k, GQ-VAE, and commitment-beta cells explicitly disable
continuous truncation: each source story contributes at most one
`max_seq_len=256` sample, so longer stories do not silently expand into extra
training examples. Every tokenizer-training cell uses one full-data epoch; the
adaptive AE warmup is capped at 6,000 steps, leaving most of that epoch for the
quantized phase without multiplying the sweep cost by ten.

## One-command full pipeline

The master config composes all five focused configs into 17 configured jobs,
bootstraps the BPE tokenizer when it is missing, selects the GQ-VAE Pareto knee,
exports its learned tokenizer, builds date-scoped corpora, and finally trains
the three matched nanoGPT models.

Inspect the complete command plan without writing outputs:

```bash
MPLCONFIGDIR=/tmp/ae-vqvae-mpl UV_CACHE_DIR=/tmp/uv-cache \
  uv run python -m training.run_research_pipeline \
  --config configs/full-research-pipeline-k8192-18m-20260807.json \
  --dry-run
```

Launch everything sequentially with one command:

```bash
MPLCONFIGDIR=/tmp/ae-vqvae-mpl UV_CACHE_DIR=/tmp/uv-cache \
  uv run python -m training.run_research_pipeline \
  --config configs/full-research-pipeline-k8192-18m-20260807.json
```

The generated date is part of every run name. To continue a pipeline on a
later day, rerun it with the original suffix, for example `--run-date
20260807`. Completed outputs are skipped. An existing incomplete training
directory stops the pipeline with its exact path because individual trainers
do not yet implement checkpoint resume. If the process is stopped with
`Ctrl-C`, `state.json` records `status="interrupted"`; it is not a live-process
indicator, so use `ps` or the tmux pane when checking whether training is still
running.

By default the GQ-VAE choice is the Pareto-frontier point closest to the ideal
of minimum validation reconstruction loss and maximum validation bytes/token.
The candidates, frontier, score, and chosen checkpoint are recorded in
`outputs/research_pipeline/<pipeline>__<date>/gqvae_selection.json`. To pin a
specific cell instead, pass its exact label, for example:

```bash
--gqvae-ablation gqvae-k8192-alpha2
```

The final pipeline state is stored alongside that selection in `state.json`.
Corpus outputs are date-scoped so a new run cannot silently reuse token streams
from a different learned checkpoint.

### Automatic visualizations

After all nanoGPT runs finish, the one-command pipeline automatically runs
`visualization.render_research_pipeline` and writes the following artifacts to
`outputs/research_pipeline/<pipeline>__<date>/plots/`:

```text
paper_fig1_gqvae_architecture.png
paper_fig2_decoder_head.png
paper_fig3_compression_vocabulary.png
paper_fig5_language_modeling.png
paper_fig7_token_frequencies.png
topk_curriculum.png
topk_vs_nearest.png
gqvae_rate_distortion.png
gqvae_training_dynamics.png
commitment_beta_sweep.png
nanogpt_bpb_comparison.png
tokenizer_compression_stats.png
results_summary.json
results_table.csv
manifest.json
```

The GQ-VAE paper's Figure 4 is not emitted because this experiment design has
only one fixed-length VQ baseline, not a fixed-length sweep. Figure 6 is not
emitted because the locked design has BPE-8192 rather than a second
compression-matched BPE tokenizer. These omissions and reasons are recorded in
`manifest.json`; the renderer does not invent unsupported points.

The paper-aligned language-model plot uses validation bits per raw UTF-8 byte
instead of cross-tokenizer token loss. This preserves the paper's comparison
question while keeping the vertical axis comparable across different token
boundaries. Top-k runs additionally log sparse-mixture entropy and
`effective_k = exp(entropy)` for the curriculum plot.

Completed visualization manifests are skipped when the same pipeline is run
again. To render an already-completed pipeline manually:

```bash
MPLCONFIGDIR=/tmp/ae-vqvae-mpl UV_CACHE_DIR=/tmp/uv-cache \
  uv run python -m visualization.render_research_pipeline \
  --pipeline-dir outputs/research_pipeline/full-k8192-18m__20260807
```

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
