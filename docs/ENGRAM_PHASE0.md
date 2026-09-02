# Engram Phase-0

This path answers one question only: with a fixed 123,551,232-parameter active
decoder-only backbone, does increasing passive Engram table capacity produce a
stable validation-NLL improvement on a frozen FineWeb-Edu token stream?

## Fixed contract

- Backbone: 12 layers, width 768, 12 heads, SwiGLU width 2048, RMSNorm, RoPE,
  causal PyTorch SDPA, tied 50,257-token GPT-2 input/output weights, context 1024.
- Engram: suffix 2/3-grams, official canonical token projection and
  multiplicative-XOR hash construction, 8 prime-modulus heads per order,
  16-dimensional embeddings per route, 256-dimensional concatenated memory,
  paper Eq. (3)-(5) gating/fusion, zero-initialized depthwise causal convolution.
- Injection: residual-add immediately before zero-based `blocks[1]`. This is
  paper "Layer 2": one attention/FFN block has already contextualized the query.
- Precision/optimizers: BF16 autocast with FP32 master parameters, AdamW for
  dense parameters, and row-sparse Adam for memory rows with no weight decay
  and 5x the backbone learning rate. The sparse update is an engineering choice
  that leaves lookup and fusion mathematics unchanged. Reported table storage
  is the requested theoretical BF16 footprint, separate from optimizer state.

The paper v2 Eq. (4) is the source of truth for the scalar sigmoid gate. The
official demo contains an additional signed-square-root transform not present
in Eq. (4); Phase-0 intentionally does not apply that transform.

## Prepare one immutable corpus

Final data (about 5 GB of token IDs plus validation):

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m training.prepare_engram_phase0_data \
  --output-dir /absolute/path/to/fineweb_edu_engram_phase0
```

The default source is the pinned FineWeb-Edu `sample-10BT` revision. The command
writes `train.bin`, `validation.bin`, `tokenizer.json`,
`canonical_projection.npy`, and `meta.json`. The metadata records dataset and
tokenizer revisions, split policy, assigned-document digests, and file SHA256s.
Every training launch verifies those hashes before allocating the model.

For a small pipeline check, prepare only the pilot budget:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m training.prepare_engram_phase0_data \
  --output-dir /absolute/path/to/fineweb_edu_engram_pilot \
  --train-prediction-tokens 100000000 \
  --validation-prediction-tokens 1000000
```

## Validate configuration without allocating tables

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m training.run_engram_phase0_sweep \
  --config configs/engram-phase0.json \
  --data-dir /absolute/path/to/fineweb_edu_engram_phase0 \
  --profile final \
  --output-root outputs/engram_phase0 \
  --dry-run
```

## Run

Use `--profile pilot` for 100M tokens. It can expose implementation and systems
problems but its report is always labeled `PILOT_ONLY` and can never emit GO.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m training.run_engram_phase0_sweep \
  --config configs/engram-phase0.json \
  --data-dir /absolute/path/to/fineweb_edu_engram_phase0 \
  --profile final \
  --output-root outputs/engram_phase0
```

The sequential sweep creates one resolved `config.json` per run and then emits
`val_loss_vs_tokens.png`, `final_val_loss_vs_log_table_size.png`, `results.csv`,
`phase0_judgement.json`, and `phase0_judgement.md` in the profile directory.
Individual variants can instead be scheduled independently with
`training.run_engram_phase0`; run the analysis module after all four finish.

GO requires the full 2.5B-token profile, Engram-L at least 0.01 nats/token below
baseline, a negative fitted NLL-vs-log(M) slope with L better than S, and a
persistent Engram-L advantage in at least two of the final three matched
validation evaluations. Local S/M swaps are allowed.
