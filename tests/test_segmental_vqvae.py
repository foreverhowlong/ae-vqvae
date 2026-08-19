import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from common.segmental_vqvae_config import SegmentalVQVAEConfig
from models.segmental_vqvae import (
    SegmentPooler,
    SegmentalVQVAE,
    segmental_vqvae_losses,
)
from training.run_segmental_vqvae_experiment import (
    evaluate_free_running,
    evaluate_interventions,
)
from training.segmental_vqvae_reporting import (
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
    assert float(losses["compression_loss"].detach()) >= 0.0
    assert float(losses["gate_logit_l2_loss"].detach()) >= 0.0
    losses["loss"].backward()
    gradient = model.gater.mlp[-1].weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0


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
    for key in (
        "ce_0_nats_per_bpe",
        "ce_rand_nats_per_bpe",
        "ce_perm_nats_per_bpe",
        "ce_null_nats_per_bpe",
    ):
        assert math.isfinite(metrics[key])


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


def test_research_defaults_match_the_confirmed_architecture():
    config = SegmentalVQVAEConfig()
    assert config.codebook_size == 8192
    assert config.d_model == 448
    assert config.n_heads == 8
    assert config.encoder_layers == 4
    assert config.decoder_layers == 6
    assert config.commitment_beta == 0.25
    assert config.compression_target == 1.67
    assert config.compression_weight == 10.0
    assert config.gate_logit_l2_weight == 1e-4


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
    model = SegmentalVQVAE(_config()).eval()
    metrics, snapshot = evaluate_interventions(
        model,
        _batch(),
        torch.device("cpu"),
        use_quantizer=True,
        seed=59,
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
                "exposure_gap_bits_per_bpe": 0.5,
                "free_running_token_accuracy": 0.2,
                "free_running_exact_match": 0.0,
                "free_running_normalized_edit_distance": 0.8,
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
        assert sorted(path.name for path in plot_dir.iterdir()) == [
            "ae_warmup_diagnostics.png",
            "segmentation_latest.json",
            "segmentation_latest.png",
            "training_curves.png",
        ]
