import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from common.segmental_vqvae_config import (
    SegmentalVQVAEConfig,
    SegmentalVQVAEDataConfig,
)
from models.segmental_vqvae import (
    LocalBoundaryEncoder,
    SemiMarkovSegmenter,
    SegmentContentEncoder,
    SegmentPooler,
    SegmentalVQVAE,
    TokenPruner,
    segmental_vqvae_losses,
)
from training.run_segmental_vqvae_experiment import (
    _viterbi_path_churn,
    evaluate,
    evaluate_free_running,
    evaluate_interventions,
)
from training.segmental_vqvae_reporting import (
    build_segmentation_snapshot,
    finalize_checkpoints,
    plot_segmental_metrics,
    rolling_checkpoint_due,
    save_model_checkpoint,
    save_resume_checkpoint,
    write_segmentation_visualization,
)


def _config(**overrides) -> SegmentalVQVAEConfig:
    values = {
        "vocab_size": 32,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "max_seq_len": 8,
        "d_model": 16,
        "latent_dim": 4,
        "n_heads": 4,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "boundary_encoder_layers": 1,
        "boundary_window_radius": 2,
        "max_span_length": 8,
        "span_encoder_layers": 1,
        "ffn_mult": 2,
        "dropout": 0.0,
        "codebook_size": 16,
    }
    values.update(overrides)
    return SegmentalVQVAEConfig(**values)


def _batch():
    input_ids = torch.tensor([
        [3, 4, 5, 2, 0, 0, 0, 0],
        [6, 7, 8, 9, 10, 11, 12, 2],
    ])
    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(0).long(),
    }


def test_segment_pooler_makes_disjoint_contiguous_chunks_not_prefixes():
    pooler = SegmentPooler(threshold=0.5)
    hidden = torch.arange(1, 7, dtype=torch.float32).view(1, 6, 1)
    logits = torch.tensor([[-10.0, 10.0, -10.0, 10.0, -10.0, 0.0]])
    outputs = pooler(
        hidden,
        logits,
        torch.ones(1, 6, dtype=torch.bool),
        sample_gates=False,
    )
    assert outputs["segment_ids"].tolist() == [[0, 0, 1, 1, 2, 2]]
    assert outputs["chunk_counts"].tolist() == [3]
    torch.testing.assert_close(
        outputs["pooled"][0, :3, 0],
        torch.tensor([1.5, 3.5, 5.5]),
    )
    assert math.isclose(float(outputs["hard_tokens_per_chunk"]), 2.0)


def test_token_pruner_packs_only_kept_latents_in_source_order():
    pruner = TokenPruner(threshold=0.5)
    latents = torch.arange(1, 6, dtype=torch.float32).view(1, 5, 1)
    logits = torch.tensor([[-10.0, 10.0, -10.0, 10.0, -10.0]])
    outputs = pruner(
        latents,
        logits,
        torch.ones(1, 5, dtype=torch.bool),
        sample_gates=False,
    )

    assert outputs["hard_boundaries"].tolist() == [
        [False, True, False, True, True]
    ]
    assert outputs["chunk_counts"].tolist() == [3]
    assert outputs["packed_source_positions"][0, :3].tolist() == [1, 3, 4]
    torch.testing.assert_close(
        outputs["pooled"][0, :3, 0],
        torch.tensor([2.0, 4.0, 5.0]),
    )
    assert torch.count_nonzero(outputs["pooled"][0, 3:]) == 0


def test_token_pruning_forces_final_code_and_quantizes_only_packed_slots():
    model = SegmentalVQVAE(_config(segmentation_mode="token_pruning")).eval()
    assert model.gater is not None
    for parameter in model.gater.parameters():
        parameter.data.zero_()
    model.gater.network[-1].bias.data.fill_(-10.0)

    batch = _batch()
    outputs = model(
        batch["input_ids"],
        batch["attention_mask"],
        use_quantizer=True,
        sample_gates=False,
    )

    expected_final = torch.tensor([3, 7])
    assert outputs["chunk_counts"].tolist() == [1, 1]
    assert outputs["packed_source_positions"][:, 0].tolist() == [3, 7]
    assert torch.equal(
        outputs["hard_boundaries"].gather(1, expected_final[:, None]),
        torch.ones(2, 1, dtype=torch.bool),
    )
    assert torch.all(outputs["indices"][:, 0] >= 0)
    assert torch.all(outputs["indices"][:, 1:] == -1)


def test_token_pruning_gate_backpropagates_before_vq():
    torch.manual_seed(5)
    model = SegmentalVQVAE(
        _config(
            segmentation_mode="token_pruning",
            compression_weight=0.0,
            gate_logit_l2_weight=0.0,
        )
    )
    batch = _batch()
    outputs = model(
        batch["input_ids"],
        batch["attention_mask"],
        use_quantizer=False,
        sample_gates=False,
    )
    losses = segmental_vqvae_losses(
        outputs,
        batch["input_ids"],
        batch["attention_mask"],
        model.config,
    )
    losses["loss"].backward()

    assert model.gater is not None
    gradient = model.gater.network[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_token_pruning_rejects_monotonic_pointer_decoder():
    with pytest.raises(
        ValueError,
        match="token_pruning requires global_cross_attention",
    ):
        _config(
            segmentation_mode="token_pruning",
            latent_routing="monotonic_pointer",
        ).validate()


def test_warmup_forward_keeps_gate_losses_active_and_backpropagates():
    torch.manual_seed(7)
    model = SegmentalVQVAE(_config())
    batch = _batch()
    outputs = model(
        batch["input_ids"],
        batch["attention_mask"],
        use_quantizer=False,
    )
    losses = segmental_vqvae_losses(
        outputs,
        batch["input_ids"],
        batch["attention_mask"],
        model.config,
    )
    assert float(losses["commitment_loss"].detach()) == 0.0
    assert outputs["decoder_boundary_logits"] is None
    assert float(losses["decoder_boundary_loss"].detach()) == 0.0
    assert float(losses["compression_loss"].detach()) >= 0.0
    assert float(losses["gate_logit_l2_loss"].detach()) >= 0.0
    losses["loss"].backward()
    gradient = model.gater.mlp[-1].weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0


def test_semi_markov_segmentation_covers_sequence_with_bounded_spans():
    torch.manual_seed(11)
    model = SegmentalVQVAE(
        _config(
            segmentation_mode="semi_markov",
            max_span_length=3,
        )
    )
    batch = _batch()
    outputs = model(
        batch["input_ids"],
        batch["attention_mask"],
        use_quantizer=False,
        sample_gates=False,
    )

    valid = batch["attention_mask"].bool()
    assert torch.equal(outputs["hard_boundaries"].sum(dim=1), outputs["chunk_counts"])
    for row, length in zip(outputs["segment_ids"], valid.sum(dim=1), strict=True):
        segment_ids = row[:length]
        assert segment_ids[0] == 0
        assert torch.equal(
            segment_ids.unique_consecutive(),
            torch.arange(int(segment_ids[-1]) + 1),
        )
        assert int(torch.bincount(segment_ids).max()) <= 3
    assert torch.allclose(
        outputs["gate_probabilities"].gather(
            1,
            (valid.sum(dim=1) - 1).unsqueeze(1),
        ),
        torch.ones(valid.shape[0], 1),
    )


def test_semi_markov_forward_backward_matches_enumerated_partitions():
    config = _config(
        segmentation_mode="semi_markov",
        max_seq_len=3,
        max_span_length=2,
    )
    segmenter = SemiMarkovSegmenter(config)
    span_scores = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, -torch.inf]]])
    lengths = torch.tensor([3])

    alpha, beta, log_partition = segmenter._forward_backward(
        span_scores,
        lengths,
    )
    probabilities = segmenter._boundary_marginals(
        span_scores,
        lengths,
        alpha,
        beta,
        log_partition,
    )

    assert math.isclose(float(log_partition), math.log(3.0), rel_tol=1e-6)
    torch.testing.assert_close(
        probabilities,
        torch.tensor([[2.0 / 3.0, 2.0 / 3.0, 1.0]]),
    )


def test_fixed_count_semi_markov_matches_enumerated_two_segment_paths():
    config = _config(
        segmentation_mode="semi_markov_fixed_count",
        compression_target=1.5,
        max_seq_len=3,
        max_span_length=2,
    )
    segmenter = SemiMarkovSegmenter(config)
    span_scores = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, -torch.inf]]])
    lengths = torch.tensor([3])
    target_counts = segmenter._target_chunk_counts(lengths)

    alpha, beta, log_partition = segmenter._forward_backward_fixed_count(
        span_scores,
        lengths,
        target_counts,
    )
    probabilities = segmenter._boundary_marginals_fixed_count(
        span_scores,
        lengths,
        target_counts,
        alpha,
        beta,
        log_partition,
    )
    hard_boundaries, path_scores = segmenter._viterbi_fixed_count(
        span_scores,
        lengths,
        target_counts,
    )

    assert target_counts.tolist() == [2]
    assert math.isclose(float(log_partition), math.log(2.0), rel_tol=1e-6)
    torch.testing.assert_close(
        probabilities,
        torch.tensor([[0.5, 0.5, 1.0]]),
    )
    assert hard_boundaries.sum().item() == 2
    assert float(path_scores) == 0.0


def test_fixed_count_semi_markov_enforces_rate_and_disables_compression_loss():
    torch.manual_seed(23)
    model = SegmentalVQVAE(
        _config(segmentation_mode="semi_markov_fixed_count")
    )
    batch = _batch()
    outputs = model(
        batch["input_ids"],
        batch["attention_mask"],
        use_quantizer=False,
        sample_gates=False,
    )
    losses = segmental_vqvae_losses(
        outputs,
        batch["input_ids"],
        batch["attention_mask"],
        model.config,
    )

    assert outputs["target_chunk_counts"].tolist() == [2, 5]
    assert torch.equal(
        outputs["chunk_counts"],
        outputs["target_chunk_counts"],
    )
    torch.testing.assert_close(
        outputs["soft_chunk_counts"],
        outputs["target_chunk_counts"].float(),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.count_nonzero(outputs["chunk_count_constraint_violation"]) == 0
    assert float(losses["compression_loss"]) == 0.0
    losses["loss"].backward()
    assert model.semi_markov_segmenter is not None
    gradient = model.semi_markov_segmenter.scorer.mlp[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_fixed_count_interventions_report_constraint_health():
    model = SegmentalVQVAE(
        _config(segmentation_mode="semi_markov_fixed_count")
    ).eval()
    metrics, snapshot = evaluate_interventions(
        model,
        _batch(),
        torch.device("cpu"),
        use_quantizer=True,
        seed=59,
    )

    assert snapshot["fixed_count_active"] is True
    assert metrics["target_chunks"] == metrics["chunks"]
    assert metrics["chunk_count_constraint_violations"] == 0
    assert metrics["hard_soft_ratio_gap"] < 1e-5

    eval_metrics = evaluate(
        model,
        [_batch()],
        torch.device("cpu"),
        use_quantizer=False,
    )
    assert eval_metrics["target_chunks"] == eval_metrics["chunks"]
    assert eval_metrics["chunk_count_constraint_violations"] == 0
    assert eval_metrics["hard_soft_ratio_gap"] < 1e-5


def test_viterbi_path_churn_compares_the_same_fixed_examples():
    first = {
        "examples": [{
            "token_ids": [3, 4, 5, 2],
            "hard_boundaries": [False, True, False, True],
        }]
    }
    second = {
        "examples": [{
            "token_ids": [3, 4, 5, 2],
            "hard_boundaries": [True, False, False, True],
        }]
    }

    unavailable = _viterbi_path_churn(None, first)
    changed = _viterbi_path_churn(first, second)

    assert unavailable["viterbi_path_churn_available"] == 0
    assert changed["viterbi_path_churn_available"] == 1
    assert changed["viterbi_path_change_fraction"] == 1.0
    assert math.isclose(changed["viterbi_boundary_churn"], 2.0 / 3.0)


def test_semi_markov_reconstruction_gradient_reaches_span_scorer():
    torch.manual_seed(13)
    model = SegmentalVQVAE(
        _config(
            segmentation_mode="semi_markov",
            max_span_length=4,
        )
    )
    batch = _batch()
    outputs = model(
        batch["input_ids"],
        batch["attention_mask"],
        use_quantizer=False,
        sample_gates=False,
    )
    losses = segmental_vqvae_losses(
        outputs,
        batch["input_ids"],
        batch["attention_mask"],
        model.config,
    )
    losses["loss"].backward()

    assert model.semi_markov_segmenter is not None
    gradient = model.semi_markov_segmenter.scorer.mlp[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_local_boundary_encoder_cannot_read_beyond_its_receptive_field():
    torch.manual_seed(17)
    config = _config(
        segmentation_mode="semi_markov",
        boundary_encoder_layers=1,
        boundary_window_radius=1,
    )
    embedding = torch.nn.Embedding(config.vocab_size, config.d_model)
    encoder = LocalBoundaryEncoder(config).eval()
    first = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 2]])
    second = first.clone()
    second[:, -1] = 10
    valid = torch.ones_like(first, dtype=torch.bool)

    first_hidden = encoder(embedding(first), valid)
    second_hidden = encoder(embedding(second), valid)

    torch.testing.assert_close(first_hidden[:, :-2], second_hidden[:, :-2])


def test_span_content_encoder_does_not_mix_selected_spans():
    torch.manual_seed(19)
    config = _config(
        segmentation_mode="semi_markov",
        max_span_length=5,
    )
    embedding = torch.nn.Embedding(config.vocab_size, config.d_model)
    encoder = SegmentContentEncoder(config).eval()
    first = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 2]])
    second = first.clone()
    second[:, -1] = 10
    valid = torch.ones_like(first, dtype=torch.bool)
    segment_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    assignment = torch.nn.functional.one_hot(
        segment_ids,
        num_classes=first.shape[1],
    ).float()

    first_pooled = encoder(embedding(first), segment_ids, valid, assignment)
    second_pooled = encoder(embedding(second), segment_ids, valid, assignment)

    torch.testing.assert_close(first_pooled[:, 0], second_pooled[:, 0])
    assert not torch.equal(first_pooled[:, 1], second_pooled[:, 1])


def test_teacher_forced_decoder_is_causal():
    model = SegmentalVQVAE(_config()).eval()
    latent = torch.randn(1, 8, model.config.latent_dim)
    latent_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    attention_mask = torch.ones(1, 8, dtype=torch.bool)
    first = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 2]])
    second = first.clone()
    second[:, 4] = 11
    first_logits = model.decode_teacher_forced(
        latent, latent_mask, first, attention_mask
    )
    second_logits = model.decode_teacher_forced(
        latent, latent_mask, second, attention_mask
    )
    # target[4] is shifted into decoder input position 5.
    torch.testing.assert_close(first_logits[:, :5], second_logits[:, :5])


def test_monotonic_decoder_cannot_read_future_latents():
    model = SegmentalVQVAE(
        _config(latent_routing="monotonic_pointer")
    ).eval()
    latents = torch.randn(1, 3, model.config.latent_dim)
    changed = latents.clone()
    changed[:, 2] += 100.0
    latent_mask = torch.ones(1, 3, dtype=torch.bool)
    targets = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 2]])
    attention_mask = torch.ones_like(targets, dtype=torch.bool)
    segment_ids = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2]])

    original_logits = model.decode_teacher_forced(
        latents,
        latent_mask,
        targets,
        attention_mask,
        segment_ids=segment_ids,
    )
    changed_logits = model.decode_teacher_forced(
        changed,
        latent_mask,
        targets,
        attention_mask,
        segment_ids=segment_ids,
    )

    torch.testing.assert_close(original_logits[:, :5], changed_logits[:, :5])
    assert not torch.equal(original_logits[:, 5:], changed_logits[:, 5:])


def test_monotonic_boundary_head_is_shallow_and_backpropagates():
    model = SegmentalVQVAE(_config(latent_routing="monotonic_pointer"))
    batch = _batch()
    outputs = model(
        batch["input_ids"],
        batch["attention_mask"],
        use_quantizer=False,
        sample_gates=False,
    )
    losses = segmental_vqvae_losses(
        outputs,
        batch["input_ids"],
        batch["attention_mask"],
        model.config,
    )

    assert outputs["decoder_boundary_logits"].shape == batch["input_ids"].shape
    assert float(losses["decoder_boundary_loss"].detach()) > 0.0
    losses["loss"].backward()
    assert model.boundary_head is not None
    gradient = model.boundary_head.mlp[-1].weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0


def test_monotonic_predicted_boundaries_advance_and_clamp_pointer():
    model = SegmentalVQVAE(
        _config(latent_routing="monotonic_pointer")
    ).eval()
    assert model.boundary_head is not None
    for parameter in model.boundary_head.parameters():
        parameter.data.zero_()
    model.boundary_head.mlp[-1].bias.data.fill_(10.0)
    latents = torch.randn(1, 3, model.config.latent_dim)
    latent_mask = torch.ones(1, 3, dtype=torch.bool)
    targets = torch.tensor([[3, 4, 5, 6, 2]])
    attention_mask = torch.ones_like(targets, dtype=torch.bool)

    decoded = model.decode_with_predicted_pointers(
        latents,
        latent_mask,
        targets,
        attention_mask,
    )

    assert decoded["pointer_trace"].tolist() == [[0, 1, 2, 2, 2]]
    assert decoded["local_position_trace"].tolist() == [[0, 0, 0, 1, 2]]
    assert decoded["final_pointers"].tolist() == [2]


def test_chunk_ordinal_embedding_is_added_after_vq_latents():
    model = SegmentalVQVAE(_config()).eval()
    latent = torch.zeros(1, 8, model.config.latent_dim)
    latent_mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    memory = model.decoder.prepare_memory(latent, latent_mask)
    assert not torch.equal(memory[:, 0], memory[:, 1])
    assert torch.count_nonzero(memory[:, 2:]) == 0


def test_interventions_are_length_matched_and_report_rate_aligned_gains():
    model = SegmentalVQVAE(_config()).eval()
    metrics, snapshot = evaluate_interventions(
        model,
        _batch(),
        torch.device("cpu"),
        use_quantizer=True,
        seed=59,
    )
    assert metrics["examples"] == 2
    assert metrics["chunks"] >= 2
    assert metrics["rate_bits_per_bpe"] > 0
    assert math.isfinite(metrics["channel_utilization_ratio"])
    assert snapshot["examples"]
    assert snapshot["chunk_lengths"]
    assert 0.0 <= metrics["boundary_fraction"] <= 1.0
    assert 0.0 <= metrics["boundary_position_dependence_hard"] <= 1.0
    assert 0.0 <= metrics["boundary_position_dependence_soft"] <= 1.0
    assert 0.0 <= metrics["singleton_chunk_fraction"] <= 1.0
    assert math.isfinite(metrics["excess_singleton_fraction"])
    for key in (
        "ce_0_nats_per_bpe",
        "ce_rand_nats_per_bpe",
        "ce_perm_nats_per_bpe",
        "ce_null_nats_per_bpe",
    ):
        assert math.isfinite(metrics[key])


def test_token_pruning_interventions_report_keep_and_gap_health():
    model = SegmentalVQVAE(
        _config(segmentation_mode="token_pruning")
    ).eval()
    metrics, snapshot = evaluate_interventions(
        model,
        _batch(),
        torch.device("cpu"),
        use_quantizer=True,
        seed=59,
    )

    assert snapshot["selection_kind"] == "keep"
    assert 0.0 < metrics["keep_fraction"] <= 1.0
    assert 0.0 <= metrics["keep_position_dependence_hard"] <= 1.0
    assert 0.0 <= metrics["keep_position_dependence_soft"] <= 1.0
    assert 0.0 <= metrics["early_keep_rate_hard"] <= 1.0
    assert 0.0 <= metrics["late_keep_rate_hard"] <= 1.0
    assert metrics["longest_drop_run"] >= 0
    assert metrics["keep_gap_p50"] >= 1.0
    assert metrics["keep_gap_p90"] >= metrics["keep_gap_p50"]

    with TemporaryDirectory() as temporary:
        plot_dir = Path(temporary) / "plots"
        write_segmentation_visualization(
            snapshot,
            plot_dir,
            compression_target=model.config.compression_target,
            run_name="token-pruning-test",
        )
        metrics_path = Path(temporary) / "metrics.jsonl"
        metrics_path.write_text(
            json.dumps({
                "split": "latent_intervention",
                "step": 1,
                **metrics,
            }) + "\n",
            encoding="utf-8",
        )
        plot_segmental_metrics(
            metrics_path,
            plot_dir,
            compression_target=model.config.compression_target,
            run_name="token-pruning-test",
        )
        assert (plot_dir / "segmentation_latest.png").is_file()
        assert (plot_dir / "segmentation_health.png").is_file()


def test_segmentation_snapshot_detects_position_determined_boundaries():
    token_count = 21
    input_ids = torch.arange(3, 3 + token_count).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    hard_boundaries = torch.tensor(
        [[False] * 10 + [True] * 11],
        dtype=torch.bool,
    )
    segment_ids = (
        hard_boundaries.long().cumsum(dim=1) - hard_boundaries.long()
    )
    gate_probabilities = torch.tensor(
        [[0.0] * 10 + [1.0] * 10 + [0.0]],
        dtype=torch.float,
    )
    snapshot = build_segmentation_snapshot(
        input_ids,
        attention_mask,
        {
            "gate_probabilities": gate_probabilities,
            "hard_boundaries": hard_boundaries,
            "segment_ids": segment_ids,
        },
    )

    assert snapshot["position_bin_candidate_counts"] == [2] * 10
    assert snapshot["hard_boundary_rate_by_position"] == [0.0] * 5 + [1.0] * 5
    assert snapshot["soft_boundary_rate_by_position"] == [0.0] * 5 + [1.0] * 5
    assert math.isclose(snapshot["boundary_position_dependence_hard"], 1.0)
    assert math.isclose(snapshot["boundary_position_dependence_soft"], 1.0)
    assert snapshot["excess_singleton_fraction"] > 0.0


def test_cached_greedy_free_running_reports_exposure_metrics():
    model = SegmentalVQVAE(_config()).eval()
    metrics = evaluate_free_running(
        model,
        _batch(),
        torch.device("cpu"),
        use_quantizer=True,
    )
    assert metrics["examples"] == 2
    assert 0.0 <= metrics["free_running_token_accuracy"] <= 1.0
    assert 0.0 <= metrics["free_running_exact_match"] <= 1.0
    assert metrics["free_running_normalized_edit_distance"] >= 0.0
    assert math.isfinite(metrics["exposure_gap_bits_per_bpe"])


def test_monotonic_free_running_reports_pointer_health():
    model = SegmentalVQVAE(
        _config(latent_routing="monotonic_pointer")
    ).eval()
    metrics = evaluate_free_running(
        model,
        _batch(),
        torch.device("cpu"),
        use_quantizer=True,
    )

    assert 0.0 <= metrics["predicted_pointer_token_alignment"] <= 1.0
    assert metrics["predicted_pointer_mae"] >= 0.0
    assert 0.0 <= metrics["target_end_code_consumption"] <= 1.0
    assert 0.0 <= metrics["free_running_code_consumption_at_eos"] <= 1.0
    assert 0.0 <= metrics["premature_code_exhaustion_fraction"] <= 1.0
    assert math.isfinite(metrics["predicted_pointer_gap_bits_per_bpe"])
    assert math.isfinite(metrics["free_running_ce_nats_per_bpe"])


def test_monotonic_free_running_keeps_unmasked_logits_for_ce():
    model = SegmentalVQVAE(
        _config(latent_routing="monotonic_pointer")
    ).eval()
    assert model.boundary_head is not None
    for parameter in model.boundary_head.parameters():
        parameter.data.zero_()
    model.boundary_head.mlp[-1].bias.data.fill_(-10.0)
    latents = torch.randn(1, 2, model.config.latent_dim)
    latent_mask = torch.ones(1, 2, dtype=torch.bool)

    details = model.free_running(
        latents,
        latent_mask,
        max_length=4,
        return_details=True,
    )

    assert torch.isfinite(details["raw_logits"]).all()
    assert torch.all(
        details["logits"][:, :, model.config.eos_token_id]
        < details["raw_logits"][:, :, model.config.eos_token_id]
    )


def test_research_defaults_match_the_confirmed_architecture():
    config = SegmentalVQVAEConfig()
    data_config = SegmentalVQVAEDataConfig()
    assert config.codebook_size == 8192
    assert config.d_model == 448
    assert config.n_heads == 8
    assert config.encoder_layers == 4
    assert config.decoder_layers == 6
    assert config.commitment_beta == 0.25
    assert config.compression_target == 1.67
    assert config.compression_weight == 10.0
    assert config.gate_logit_l2_weight == 1e-4
    assert data_config.continuous_truncation is False
    assert config.latent_routing == "global_cross_attention"
    assert config.segmentation_mode == "bernoulli"
    assert config.decoder_boundary_loss_weight == 1.0


def test_monotonic_experiment_config_is_paired_with_comparable_global_decoder():
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "segmental-vqvae-bpe-k8192-monotonic.json"
    )
    experiments = json.loads(config_path.read_text())["experiments"]
    assert len(experiments) == 2
    monotonic, global_decoder = experiments
    assert monotonic["continuous-truncation"] is False
    assert global_decoder["continuous-truncation"] is False
    assert monotonic["latent-routing"] == "monotonic_pointer"
    assert global_decoder["latent-routing"] == "global_cross_attention"
    assert monotonic["decoder-boundary-loss-weight"] == 1.0
    differing_keys = {
        key
        for key in monotonic
        if monotonic[key] != global_decoder[key]
    }
    assert differing_keys == {"ablation", "latent-routing"}


def test_semi_markov_experiment_is_paired_with_bernoulli_control():
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "segmental-vqvae-bpe-k8192-semimarkov.json"
    )
    experiments = json.loads(config_path.read_text())["experiments"]
    assert len(experiments) == 2
    semi_markov, bernoulli = experiments
    assert semi_markov["continuous-truncation"] is False
    assert bernoulli["continuous-truncation"] is False
    assert semi_markov["latent-routing"] == "global_cross_attention"
    assert bernoulli["latent-routing"] == "global_cross_attention"
    assert semi_markov["encoder-layers"] == 6
    assert bernoulli["encoder-layers"] == 6
    assert semi_markov["segmentation-mode"] == "semi_markov"
    assert bernoulli["segmentation-mode"] == "bernoulli"
    differing_keys = {
        key
        for key in semi_markov
        if semi_markov[key] != bernoulli[key]
    }
    assert differing_keys == {"ablation", "segmentation-mode"}


def test_token_pruning_experiment_contains_only_the_pruning_run():
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "segmental-vqvae-bpe-k8192-token-pruning.json"
    )
    experiments = json.loads(config_path.read_text())["experiments"]
    assert len(experiments) == 1
    token_pruning = experiments[0]
    assert token_pruning["continuous-truncation"] is False
    assert token_pruning["latent-routing"] == "global_cross_attention"
    assert token_pruning["segmentation-mode"] == "token_pruning"


def test_fixed_count_semi_markov_config_contains_only_the_constrained_run():
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "segmental-vqvae-bpe-k8192-semimarkov-fixed-count.json"
    )
    experiments = json.loads(config_path.read_text())["experiments"]
    assert len(experiments) == 1
    fixed_count = experiments[0]
    assert fixed_count["continuous-truncation"] is False
    assert fixed_count["latent-routing"] == "global_cross_attention"
    assert fixed_count["segmentation-mode"] == "semi_markov_fixed_count"
    assert fixed_count["compression-weight"] == 0.0


def test_metrics_first_checkpoints_overwrite_and_drop_resume_state():
    model = SegmentalVQVAE(_config())
    optimizer = torch.optim.AdamW(model.parameters())
    with TemporaryDirectory() as temporary:
        checkpoint_dir = Path(temporary) / "checkpoints"
        checkpoint_dir.mkdir()
        latest = checkpoint_dir / "latest.pt"
        save_resume_checkpoint(
            model,
            optimizer,
            latest,
            step=10,
            epoch=1,
            phase="vq",
        )
        save_resume_checkpoint(
            model,
            optimizer,
            latest,
            step=20,
            epoch=2,
            phase="vq",
        )
        assert [path.name for path in checkpoint_dir.glob("*.pt")] == ["latest.pt"]
        assert torch.load(latest, weights_only=False)["step"] == 20

        best = checkpoint_dir / "best.pt"
        save_model_checkpoint(model, best, step=20, epoch=2, phase="vq")
        assert "optimizer" not in torch.load(best, weights_only=False)
        retained = finalize_checkpoints(
            model,
            optimizer,
            checkpoint_dir,
            step=20,
            epoch=2,
            phase="vq",
            save_last_resume=False,
        )
        assert retained == ["best.pt"]
        assert not latest.exists()
        assert rolling_checkpoint_due(step=1000, every=1000)
        assert not rolling_checkpoint_due(step=1000, every=0)


def test_segmental_visualizations_overwrite_bounded_artifacts():
    model = SegmentalVQVAE(
        _config(latent_routing="monotonic_pointer")
    ).eval()
    metrics, snapshot = evaluate_interventions(
        model,
        _batch(),
        torch.device("cpu"),
        use_quantizer=True,
        seed=59,
    )
    free_metrics = evaluate_free_running(
        model,
        _batch(),
        torch.device("cpu"),
        use_quantizer=True,
    )
    with TemporaryDirectory() as temporary:
        run_dir = Path(temporary)
        metrics_path = run_dir / "metrics.jsonl"
        rows = [
            {
                "split": "ae_warmup_diagnostic",
                "step": 1,
                "water_filling_effective_dim": 3,
                "latent_effective_dim": 4,
                "participation_ratio": 3.5,
                "water_filling_level": 0.2,
            },
            {"split": "phase_transition", "step": 2},
            {
                "split": "eval",
                "step": 2,
                "phase": "vq",
                "loss": 2.0,
                "reconstruction_loss": 1.5,
                "commitment_weighted_loss": 0.1,
                "compression_loss": 0.3,
                "gate_logit_l2_loss": 0.1,
                "tokens_per_chunk_hard": 1.7,
                "tokens_per_chunk_soft_batch_mean": 1.67,
                "rate_bits_per_bpe": 7.8,
                "distortion_bits_per_bpe": 2.1,
                "codebook_utilization": 0.1,
                "codebook_perplexity": 12.0,
            },
            {"split": "latent_intervention", "step": 2, **metrics},
            {
                "split": "free_running",
                "step": 2,
                **free_metrics,
            },
        ]
        metrics_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        plot_dir = run_dir / "plots"
        plot_segmental_metrics(
            metrics_path,
            plot_dir,
            compression_target=1.67,
            run_name="test",
        )
        write_segmentation_visualization(
            snapshot,
            plot_dir,
            compression_target=1.67,
            run_name="test",
        )
        assert (plot_dir / "training_curves.png").is_file()
        assert (plot_dir / "ae_warmup_diagnostics.png").is_file()
        assert (plot_dir / "segmentation_latest.png").is_file()
        assert (plot_dir / "segmentation_latest.json").is_file()
        assert (plot_dir / "segmentation_health.png").is_file()
        assert (plot_dir / "pointer_health.png").is_file()
        assert sorted(path.name for path in plot_dir.iterdir()) == [
            "ae_warmup_diagnostics.png",
            "pointer_health.png",
            "segmentation_health.png",
            "segmentation_latest.json",
            "segmentation_latest.png",
            "training_curves.png",
        ]
