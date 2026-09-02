import json
import math

import numpy as np
import torch

from analysis.engram_phase0_results import (
    create_results_csv,
    judge,
    load_runs,
    plot_capacity,
    plot_curves,
)
from models.engram_phase0 import (
    EngramPhase0LM,
    Phase0BackboneConfig,
    Phase0EngramConfig,
    analytical_backbone_parameter_count,
    analytical_engram_counts,
    consecutive_unique_primes,
)
from training.run_engram_phase0 import global_clip_grad_norm, make_batch


def tiny_configs(enabled: bool = True):
    backbone = Phase0BackboneConfig(
        vocab_size=32,
        context_length=8,
        layers=2,
        d_model=32,
        attention_heads=4,
        d_ff=64,
    )
    engram = Phase0EngramConfig(
        enabled=enabled,
        table_rows_target=29 if enabled else 0,
        table_dtype="float32",
        pad_token_id=31,
    )
    return backbone, engram


def test_phase0_backbone_has_expected_125m_parameter_count():
    assert analytical_backbone_parameter_count(Phase0BackboneConfig()) == 123_551_232


def test_prime_tables_are_unique_and_near_target():
    primes = consecutive_unique_primes(32_768, 16)
    assert len(primes) == len(set(primes)) == 16
    assert primes[0] >= 32_768
    assert primes[-1] < 33_000


def test_actual_and_analytical_counts_match_and_backbone_init_is_fixed():
    backbone, baseline_config = tiny_configs(enabled=False)
    _, engram_config = tiny_configs(enabled=True)
    projection = torch.arange(backbone.vocab_size)
    baseline = EngramPhase0LM(backbone, baseline_config, backbone_init_seed=7)
    augmented = EngramPhase0LM(
        backbone, engram_config, projection, backbone_init_seed=7
    )
    baseline_state = baseline.state_dict()
    for name, value in augmented.state_dict().items():
        if not name.startswith("engram."):
            assert torch.equal(value, baseline_state[name])
    analytical = analytical_engram_counts(backbone, engram_config)
    actual = augmented.parameter_counts()
    assert actual["backbone"] == analytical_backbone_parameter_count(backbone)
    assert actual["dense_engram"] == analytical["dense_engram"]
    assert actual["sparse_tables"] == analytical["sparse_tables"]
    assert torch.count_nonzero(augmented.engram.short_conv.weight) == 0


def test_dense_engram_initialization_does_not_depend_on_table_capacity():
    backbone, small_config = tiny_configs(enabled=True)
    large_config = Phase0EngramConfig(
        enabled=True,
        table_rows_target=101,
        table_dtype="float32",
        pad_token_id=31,
    )
    projection = torch.arange(backbone.vocab_size)
    small = EngramPhase0LM(backbone, small_config, projection, backbone_init_seed=9)
    large = EngramPhase0LM(backbone, large_config, projection, backbone_init_seed=9)
    assert torch.equal(
        small.engram.key_projection.weight, large.engram.key_projection.weight
    )
    assert torch.equal(
        small.engram.value_projection.weight, large.engram.value_projection.weight
    )


def test_hashing_matches_official_numpy_order_and_forward_is_finite():
    backbone, engram_config = tiny_configs(enabled=True)
    projection = torch.tensor([value // 2 for value in range(backbone.vocab_size)])
    model = EngramPhase0LM(backbone, engram_config, projection, backbone_init_seed=3)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    actual = model.engram.hash_ids(input_ids).numpy()

    compressed = projection[input_ids].numpy().astype(np.int64)
    pad = int(projection[engram_config.pad_token_id])
    shifts = [compressed]
    for shift in (1, 2):
        shifts.append(
            np.pad(compressed, ((0, 0), (shift, 0)), constant_values=pad)[:, :8]
        )
    expected = []
    route = 0
    multipliers = model.engram.hash_multipliers.numpy()
    for order in (2, 3):
        mixed = shifts[0] * multipliers[0]
        for position in range(1, order):
            mixed = np.bitwise_xor(mixed, shifts[position] * multipliers[position])
        for _ in range(8):
            expected.append(
                mixed % model.engram.table_row_counts[route]
                + int(model.engram.table_offsets[route])
            )
            route += 1
    expected = np.stack(expected, axis=-1)
    assert np.array_equal(actual, expected)

    logits, loss, auxiliary = model(input_ids, input_ids)
    assert logits.shape == (1, 8, backbone.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    assert 0.0 < float(auxiliary["engram_gate_mean"]) < 1.0


def test_fixed_stream_batch_masks_only_final_extra_positions(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(21, dtype=np.uint16).tofile(path)
    stream = np.memmap(path, dtype=np.uint16, mode="r")
    inputs, targets, valid = make_batch(
        stream,
        start=0,
        prediction_tokens=13,
        batch_size=2,
        context_length=8,
        eot_id=31,
        device=torch.device("cpu"),
    )
    assert valid == 13
    assert int((targets != -1).sum()) == 13
    assert torch.equal(inputs[0], torch.arange(8))
    assert torch.equal(targets[0], torch.arange(1, 9))


def test_sparse_and_dense_gradients_share_one_clip_norm():
    backbone, engram_config = tiny_configs(enabled=True)
    model = EngramPhase0LM(
        backbone, engram_config, torch.arange(backbone.vocab_size), backbone_init_seed=5
    )
    ids = torch.arange(8).view(1, 8)
    _, loss, _ = model(ids, ids)
    assert loss is not None
    loss.backward()
    assert model.engram.embedding.weight.grad.is_sparse
    before = global_clip_grad_norm(model.parameters(), 0.1)
    assert math.isfinite(before) and before > 0.1
    after = global_clip_grad_norm(model.parameters(), 0.1)
    assert after <= 0.10001


def test_complete_final_sweep_generates_artifacts_and_go_verdict(tmp_path):
    final_nll = {
        "baseline": 2.00,
        "engram_s": 1.995,
        "engram_m": 1.990,
        "engram_l": 1.985,
    }
    table_rows = {"baseline": 0, "engram_s": 32_768, "engram_m": 131_072, "engram_l": 524_288}
    for variant, nll in final_nll.items():
        run_dir = tmp_path / variant
        run_dir.mkdir()
        enabled = variant != "baseline"
        config = {
            "variant": variant,
            "profile_name": "final",
            "backbone": {"layers": 12},
            "optimization": {"peak_lr": 6e-4},
            "profile": {"training_tokens": 2_500_000_000},
            "go_criteria": {
                "large_min_nll_improvement": 0.01,
                "late_tail_points": 3,
                "late_tail_min_positive_fraction": 2 / 3,
            },
            "corpus": {
                "train": {"sha256": "train"},
                "validation": {"sha256": "validation"},
            },
            "engram": {
                "enabled": enabled,
                "table_rows_target": table_rows[variant],
                "memory_dim": 256,
                "ngram_orders": [2, 3],
            },
            "parameter_counts": {
                "backbone": 123_551_232,
                "dense_engram": 400_128 if enabled else 0,
                "sparse_tables": table_rows[variant] * 16 * 16,
                "total": 123_551_232,
            },
        }
        (run_dir / "config.json").write_text(json.dumps(config))
        (run_dir / "summary.json").write_text(
            json.dumps({"status": "completed"})
        )
        with (run_dir / "metrics.jsonl").open("w") as handle:
            for index, tokens in enumerate((2_450_000_000, 2_475_000_000, 2_500_000_000)):
                point_nll = nll + 0.002 * (2 - index)
                handle.write(
                    json.dumps(
                        {
                            "split": "validation",
                            "tokens_seen": tokens,
                            "validation_nll": point_nll,
                            "perplexity": math.exp(point_nll),
                        }
                    )
                    + "\n"
                )
    runs = load_runs(tmp_path)
    rows = create_results_csv(runs, tmp_path / "results.csv")
    plot_curves(runs, tmp_path / "val_loss_vs_tokens.png")
    plot_capacity(rows, tmp_path / "final_val_loss_vs_log_table_size.png")
    verdict = judge(runs, rows)
    assert verdict["decision"] == "GO"
    assert (tmp_path / "results.csv").is_file()
    assert (tmp_path / "val_loss_vs_tokens.png").stat().st_size > 0
    assert (tmp_path / "final_val_loss_vs_log_table_size.png").stat().st_size > 0
