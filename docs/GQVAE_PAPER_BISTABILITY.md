# GQ-VAE v1 gate-bistability reproduction

This experiment is pinned to arXiv:2512.21913v1 and the authors' public code at
commit `9366387cafc3aeaa16fb33506762698b077d28d8`.

It deliberately does not reuse the scalable `models/gqvae.py` architecture.
The paper model owns the encoder, quantizer lifecycle, gater, decoder, and five
loss terms. The training package owns TinyStories preprocessing, optimizer and
scheduler construction, metric logging, and read-only bistability diagnosis.

The official command-line parser declares 15 epochs, but the released epoch
loop is commented out: the executable performs one pass over the first compiled
10% TinyStories partition. The faithful config records both facts and follows
the behavior that actually runs.

The released CLI also accepts `--seed` without applying it. This experiment
deliberately seeds Python, NumPy, and PyTorch so initialization sensitivity can
be measured reproducibly. That is a diagnostic-control extension, not a change
to the architecture or objective, and it is recorded in every run manifest.

Dry-run the five-seed experiment:

```bash
uv run python -m training.run_gqvae_paper_bistability_sweep \
  --config configs/gqvae-paper-v1-bistability-seeds.json \
  --run-date 20260813 \
  --gpus 0,1,2 \
  --dry-run
```

Run it by removing `--dry-run`. One seed occupies one GPU at a time; remaining
seeds are queued. Each run writes `metrics.jsonl`, `summary.json`, and
checkpoints below `outputs/gqvae_paper_bistability/`. The sweep writes
`summary__<run-date>.json` with terminal state counts and a conservative
`observed_zero_one_bistability` verdict.

Gate states are defined only for diagnosis and do not enter the loss:

- `collapsed_zero`: at least 95% of gate values are below 0.1.
- `collapsed_one`: at least 95% are above 0.9.
- `polarized`: at least 95% lie in the two extremes, but neither extreme alone
  occupies 95%.
- `interior`: none of the above.

The logs also retain a 20-bin gate histogram, hard-on fraction at the paper's
0.5 threshold, per-sequence zero/one collapse fractions, and state transitions.
Seeing both terminal collapse states across seeds is evidence for the proposed
zero/one bistability; one collapsed seed by itself is not.
