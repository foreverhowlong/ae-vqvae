from collections import OrderedDict
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn

from models.text_vqvae import (
    CollapseControlConfig,
    CrossAttentionTextDecoder,
    DECODER_TYPES,
    ENCODER_TYPES,
    MemoryTrunkTextDecoder,
    PlainSelfAttention,
    RotarySelfAttention,
    RotaryTextEncoder,
    SubPixelSequenceUpsampler,
    TextAttnBlock,
    TextResBlock,
    TextVQVAE,
    TextVQVAEConfig,
    VQGANPreAttentionTextDecoder,
    VQGANPreAttentionTextEncoder,
    VQGANRTextDecoder,
    VQGANRTextEncoder,
    VQGANTextDecoder,
    VQGANTextEncoder,
    VectorQuantizer,
    codebook_stats,
    pad_aware_adaptive_pool1d,
    text_vqvae_losses,
)
from training.text_vqvae.codebook_init import initialize_codebook_kmeans
from training.text_vqvae.loop import (
    compute_accuracy,
    compute_bits_per_token,
    evaluate,
    evaluate_codebook_usage,
    make_loader,
    optimizer_step,
    save_checkpoint,
)
from training.text_vqvae.warmup import (
    AdaptiveWarmupController,
    evaluate_adaptive_warmup,
    latent_spectrum_metrics,
    reverse_water_filling,
)
from training.text_vqvae.reporting import plot_codebook_usage, plot_training_curves
from training.text_vqvae.geometry import dump_geometry_snapshot, finalize_geometry_artifacts
from visualization.text_vqvae import (
    collect_encoder_vectors,
    compare_vector_distributions_pca,
    render_pca_comparison,
    save_pca_metadata,
)
from visualization.render_geometry_animation import (
    _encoder_spectrum_metrics,
    _geometry_metric_panel_labels,
    _pca_component_label,
    _snapshot_encoder_view,
    compute_animation_scales,
    fit_shared_pca,
    load_snapshots,
    render_frame,
    render_run,
)


def small_config(**overrides):
    values = {
        "vocab_size": 32,
        "max_seq_len": 12,
        "latent_slots": 4,
        "d_model": 16,
        "n_heads": 4,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "memory_decoder_latent_layers": 2,
        "memory_decoder_output_layers": 1,
        "ffn_mult": 2,
        "dropout": 0.0,
        "codebook_size": 8,
        "pad_token_id": 31,
    }
    values.update(overrides)
    return TextVQVAEConfig(**values)


class EvaluationPipelineTest(unittest.TestCase):
    def test_continuous_optimizer_step_omits_codebook_metrics(self):
        config = small_config(bottleneck_type="continuous")
        collapse_config = CollapseControlConfig(use_ema_codebook=False)
        model = TextVQVAE(config, collapse_config=collapse_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch = {
            "input_ids": torch.randint(0, 31, (2, 12)),
            "attention_mask": torch.ones(2, 12, dtype=torch.long),
        }

        metrics = optimizer_step(
            model,
            optimizer,
            batch,
            config,
            collapse_config,
            grad_clip=1.0,
            beta=config.commitment_beta,
            step=1,
        )

        self.assertEqual(
            set(metrics),
            {"loss", "recon_nll", "token_accuracy", "grad_norm"},
        )

    def test_vq_bypass_behaves_as_ae_without_updating_codebook(self):
        config = small_config()
        collapse_config = CollapseControlConfig(use_ema_codebook=True)
        model = TextVQVAE(config, collapse_config=collapse_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        batch = {
            "input_ids": torch.randint(0, 31, (2, 12)),
            "attention_mask": torch.ones(2, 12, dtype=torch.long),
        }
        codebook_before = model.quantizer.codebook.weight.detach().clone()
        cluster_before = model.quantizer.ema_cluster_size.detach().clone()

        metrics = optimizer_step(
            model,
            optimizer,
            batch,
            config,
            collapse_config,
            grad_clip=1.0,
            beta=config.commitment_beta,
            step=1,
            use_quantizer=False,
        )

        self.assertEqual(
            set(metrics),
            {"loss", "recon_nll", "token_accuracy", "grad_norm"},
        )
        torch.testing.assert_close(model.quantizer.codebook.weight, codebook_before)
        torch.testing.assert_close(model.quantizer.ema_cluster_size, cluster_before)

    def test_evaluate_collects_reconstructions_in_one_pass_and_restores_mode(self):
        config = small_config()
        collapse_config = CollapseControlConfig()
        model = TextVQVAE(config, collapse_config=collapse_config)
        model.train()
        batches = [{
            "input_ids": torch.tensor([
                [1, 2, 3, 4, 5, 6, 31, 31, 31, 31, 31, 31],
                [7, 8, 9, 10, 11, 12, 13, 14, 31, 31, 31, 31],
            ]),
            "attention_mask": torch.tensor([
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            ]),
        }]

        class CountingLoader:
            iterations = 0

            def __iter__(self):
                self.iterations += 1
                return iter(batches)

        loader = CountingLoader()
        tokenizer = SimpleNamespace(
            decode=lambda ids: " ".join(str(token_id) for token_id in ids)
        )
        frozen_codebook_c0 = model.quantizer.codebook.weight.detach().clone()

        metrics, rows = evaluate(
            model,
            loader,
            torch.device("cpu"),
            config,
            collapse_config,
            beta=config.commitment_beta,
            tokenizer=tokenizer,
            frozen_codebook_c0=frozen_codebook_c0,
        )

        self.assertEqual(loader.iterations, 1)
        self.assertTrue(model.training)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["original"], "1 2 3 4 5 6")
        self.assertEqual(len(rows[0]["reconstruction"].split()), 6)
        self.assertIn("loss", metrics)
        self.assertEqual(len(metrics["code_counts"]), config.codebook_size)
        self.assertEqual(
            metrics["codebook_utilization"],
            metrics["codebook_utilization_full"],
        )
        self.assertIn("codebook_utilization_batch_mean", metrics)
        self.assertGreater(metrics["codebook_assignment_count"], 0)
        self.assertEqual(
            metrics["codebook_utilization_frozen_c0"],
            metrics["codebook_utilization_full"],
        )

    def test_continuous_evaluation_reports_only_common_metrics(self):
        config = small_config(bottleneck_type="continuous")
        collapse_config = CollapseControlConfig(use_ema_codebook=False)
        model = TextVQVAE(config, collapse_config=collapse_config)
        batch = {
            "input_ids": torch.randint(0, 31, (2, 12)),
            "attention_mask": torch.ones(2, 12, dtype=torch.long),
        }

        metrics, rows = evaluate(
            model,
            [batch],
            torch.device("cpu"),
            config,
            collapse_config,
            beta=config.commitment_beta,
        )

        self.assertEqual(rows, [])
        self.assertEqual(
            set(metrics),
            {"examples", "loss", "recon_nll", "token_ppl", "token_accuracy"},
        )
        with self.assertRaisesRegex(ValueError, "not applicable"):
            evaluate_codebook_usage(
                model,
                [batch],
                torch.device("cpu"),
                config,
            )

    def test_continuous_bottleneck_runs_through_training_pipeline(self):
        from common.text_data import ByteTokenizer
        from common.text_vqvae_config import DataConfig, TrainConfig
        from training.text_vqvae.loop import run

        config = small_config(bottleneck_type="continuous")
        collapse_config = CollapseControlConfig(use_ema_codebook=False)
        model = TextVQVAE(config, collapse_config=collapse_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch = {
            "input_ids": torch.randint(0, 31, (2, 12)),
            "attention_mask": torch.ones(2, 12, dtype=torch.long),
        }
        tracker = SimpleNamespace(
            log=lambda *args, **kwargs: None,
            summary={},
        )

        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            for child in ("checkpoints", "plots", "samples"):
                (run_dir / child).mkdir()
            payload = {
                "diagnostics": {
                    "initial_pca": {
                        "status": "not_applicable",
                        "reason": "continuous bottleneck has no codebook",
                    },
                    "geometry": {},
                },
            }
            run(
                model=model,
                optimizer=optimizer,
                train_loader=[batch],
                val_loader=[batch],
                train_cfg=TrainConfig(
                    epochs=1,
                    eval_every=1,
                    save_every=100,
                    tokenizer="byte",
                    tokenizer_path=None,
                ),
                data_cfg=DataConfig(),
                model_config=config,
                collapse_config=collapse_config,
                run_dir=run_dir,
                run_name="continuous-smoke",
                tokenizer=ByteTokenizer(),
                device=torch.device("cpu"),
                config_payload=payload,
                tracker=tracker,
                initial_pca_opts={
                    "enabled": True,
                    "max_points": 8,
                    "fit_mode": "balanced",
                    "strict": True,
                },
                geometry_snapshot_opts={
                    "enabled": True,
                    "dense_every": 1,
                    "dense_until": 1,
                    "sparse_every": 1,
                    "probe_points": 8,
                    "strict": True,
                    "render_enabled": False,
                    "render_basis": "first_last",
                    "render_fps": 8,
                    "keep_snapshots": True,
                },
            )

            self.assertTrue((run_dir / "summary.json").is_file())
            self.assertTrue((run_dir / "plots" / "training_curves.png").is_file())
            self.assertFalse((run_dir / "plots" / "codebook_usage.png").exists())
            self.assertTrue((run_dir / "geometry" / "step000000.npz").is_file())

    def test_codebook_probe_is_eval_only_and_preserves_rng_ema_and_mode(self):
        config = small_config()
        collapse_config = CollapseControlConfig(use_ema_codebook=True)
        model = TextVQVAE(config, collapse_config=collapse_config)
        model.train()
        loader = [{
            "input_ids": torch.tensor([
                [1, 2, 3, 4, 5, 6, 31, 31, 31, 31, 31, 31],
                [7, 8, 9, 10, 11, 12, 13, 14, 31, 31, 31, 31],
            ]),
            "attention_mask": torch.tensor([
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            ]),
        }]
        codebook_before = model.quantizer.codebook.weight.detach().clone()
        cluster_size_before = model.quantizer.ema_cluster_size.detach().clone()
        rng_before = torch.random.get_rng_state().clone()

        metrics = evaluate_codebook_usage(
            model,
            loader,
            torch.device("cpu"),
            config,
        )

        self.assertTrue(model.training)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        self.assertTrue(torch.equal(
            model.quantizer.codebook.weight,
            codebook_before,
        ))
        self.assertTrue(torch.equal(
            model.quantizer.ema_cluster_size,
            cluster_size_before,
        ))
        self.assertEqual(metrics["examples"], 2)
        self.assertGreater(metrics["codebook_assignment_count"], 0)

    def test_eval_batch_mean_excludes_the_partial_final_batch(self):
        config = small_config()
        collapse_config = CollapseControlConfig()
        model = TextVQVAE(config, collapse_config=collapse_config)
        samples = [
            {
                "input_ids": torch.tensor(
                    [1, 2, 3, 4, 5, 6, 31, 31, 31, 31, 31, 31]
                ),
                "attention_mask": torch.tensor(
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
                ),
            }
            for _ in range(3)
        ]
        loader = make_loader(
            samples,
            batch_size=2,
            shuffle=False,
            device=torch.device("cpu"),
            num_workers=0,
        )

        metrics, _ = evaluate(
            model,
            loader,
            torch.device("cpu"),
            config,
            collapse_config,
            beta=config.commitment_beta,
            max_reconstruction_items=0,
        )

        self.assertEqual(metrics["examples"], 3)
        self.assertEqual(metrics["codebook_batch_size"], 2)
        self.assertEqual(metrics["codebook_batch_count"], 1)
        self.assertNotIn("used_codes", metrics)
        self.assertNotIn("dead_codes", metrics)
        self.assertIn("compat_full_eval_used_codes", metrics)
        self.assertIn("compat_full_eval_dead_codes", metrics)

    def test_training_plot_prefers_matched_probes_and_adds_compact_run_label(self):
        import json

        with TemporaryDirectory() as temp_dir:
            plot_dir = Path(temp_dir)
            metrics_path = plot_dir / "metrics.jsonl"
            rows = [
                {
                    "split": "train",
                    "step": 1,
                    "loss": 1.0,
                    "token_accuracy": 0.5,
                    "codebook_utilization": 0.25,
                },
                {
                    "split": "train_window",
                    "step": 1,
                    "codebook_utilization_batch_mean": 0.25,
                },
                {
                    "split": "eval",
                    "step": 1,
                    "loss": 1.1,
                    "token_ppl": 3.0,
                    "token_accuracy": 0.4,
                    "codebook_utilization": 0.5,
                    "codebook_utilization_batch_mean": 0.3,
                    "codebook_utilization_frozen_c0": 0.2,
                },
                {
                    "split": "codebook_probe",
                    "step": 1,
                    "train_utilization": 0.4,
                    "eval_utilization": 0.5,
                },
            ]
            metrics_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            with (
                patch("matplotlib.figure.Figure.text") as figure_text,
                patch("matplotlib.axes.Axes.set_xlabel") as set_xlabel,
                patch("matplotlib.axes.Axes.set_ylabel") as set_ylabel,
                patch("matplotlib.axes.Axes.set_title") as set_title,
            ):
                plot_training_curves(
                    metrics_path,
                    plot_dir,
                    run_name="compact-run-name",
                )

            self.assertTrue((plot_dir / "training_curves.png").is_file())
            self.assertEqual(
                figure_text.call_args.args[2],
                "run: compact-run-name",
            )
            self.assertIn(
                "Codebook utilization: current vs frozen K-means C0",
                [call.args[0] for call in set_title.call_args_list],
            )
            self.assertEqual(
                {
                    "Optimizer step (parameter updates)",
                },
                {call.args[0] for call in set_xlabel.call_args_list},
            )
            self.assertTrue({
                "Mean total loss (composite objective units)",
                "Perplexity (dimensionless)",
                "Correct-token fraction [0, 1]",
                "Used-code fraction [0, 1]",
            }.issubset({call.args[0] for call in set_ylabel.call_args_list}))

    def test_codebook_usage_panels_label_counts_and_rank_semantics(self):
        with TemporaryDirectory() as temp_dir:
            plot_dir = Path(temp_dir)
            with (
                patch("matplotlib.axes.Axes.set_xlabel") as set_xlabel,
                patch("matplotlib.axes.Axes.set_ylabel") as set_ylabel,
            ):
                plot_codebook_usage([4, 2, 0], plot_dir)

            self.assertTrue((plot_dir / "codebook_usage.png").is_file())
            self.assertEqual(
                {
                    "Code rank by assignments (1 = most used)",
                    "Assignments per active code (count)",
                },
                {call.args[0] for call in set_xlabel.call_args_list},
            )
            self.assertEqual(
                {
                    "Validation assignments (count)",
                    "Active codes per histogram bin (count)",
                },
                {call.args[0] for call in set_ylabel.call_args_list},
            )

    def test_persistent_workers_are_opt_in_and_require_workers(self):
        device = torch.device("cpu")
        train_loader = make_loader(
            [0, 1], 1, shuffle=True, device=device, num_workers=1
        )
        val_loader = make_loader(
            [0, 1],
            1,
            shuffle=False,
            device=device,
            num_workers=1,
            persistent_workers=True,
        )
        single_process_loader = make_loader(
            [0, 1],
            1,
            shuffle=False,
            device=device,
            num_workers=0,
            persistent_workers=True,
        )

        self.assertFalse(train_loader.persistent_workers)
        self.assertTrue(val_loader.persistent_workers)
        self.assertFalse(single_process_loader.persistent_workers)


class GeometrySnapshotTest(unittest.TestCase):
    def test_snapshot_fields_shapes_mode_and_rng(self):
        model = TextVQVAE(
            small_config(),
            collapse_config=CollapseControlConfig(use_ema_codebook=True),
        )
        model.train()
        probe = [{
            "input_ids": torch.tensor([
                [1, 2, 3, 4, 5, 6, 7, 8, 31, 31, 31, 31],
                [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 31, 31],
            ]),
            "attention_mask": torch.tensor([
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            ]),
        }]
        codebook_before = model.quantizer.codebook.weight.detach().clone()
        cluster_size_before = model.quantizer.ema_cluster_size.detach().clone()
        rng_before = torch.random.get_rng_state().clone()

        with TemporaryDirectory() as temp_dir:
            metrics = dump_geometry_snapshot(model, probe, 7, Path(temp_dir))
            with np.load(Path(temp_dir) / "geometry" / "step000007.npz") as snapshot:
                self.assertEqual(
                    set(snapshot.files),
                    {
                        "geometry_format_version",
                        "z_e",
                        "codebook",
                        "assignments",
                        "nearest_distances",
                        "pad_ratios",
                        "slot_indices",
                    },
                )
                self.assertEqual(int(snapshot["geometry_format_version"]), 2)
                self.assertEqual(snapshot["z_e"].shape, (6, 16))
                self.assertEqual(snapshot["z_e"].dtype, np.float16)
                self.assertEqual(snapshot["codebook"].shape, (8, 16))
                self.assertEqual(snapshot["codebook"].dtype, np.float16)
                self.assertEqual(snapshot["assignments"].shape, (6,))
                self.assertEqual(snapshot["assignments"].dtype, np.int32)
                self.assertEqual(snapshot["nearest_distances"].shape, (6,))
                self.assertEqual(snapshot["pad_ratios"].shape, (6,))
                self.assertEqual(
                    snapshot["slot_indices"].tolist(),
                    [0, 1, 2, 0, 1, 2],
                )

        self.assertTrue(model.training)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
        torch.testing.assert_close(model.quantizer.codebook.weight, codebook_before)
        torch.testing.assert_close(model.quantizer.ema_cluster_size, cluster_size_before)
        self.assertEqual(metrics["valid_probe_points"], 6)
        self.assertNotIn("used_codes", metrics)
        self.assertIn("participation_ratio", metrics)
        self.assertIn("win_count_gini", metrics)

    def test_continuous_snapshot_contains_only_latent_geometry(self):
        model = TextVQVAE(
            small_config(bottleneck_type="continuous"),
            collapse_config=CollapseControlConfig(use_ema_codebook=False),
        )
        probe = [{
            "input_ids": torch.randint(0, 31, (2, 12)),
            "attention_mask": torch.ones(2, 12, dtype=torch.long),
        }]

        with TemporaryDirectory() as temp_dir:
            metrics = dump_geometry_snapshot(model, probe, 3, Path(temp_dir))
            snapshot_path = Path(temp_dir) / "geometry" / "step000003.npz"
            with np.load(snapshot_path) as snapshot:
                self.assertEqual(
                    set(snapshot.files),
                    {
                        "geometry_format_version",
                        "z_e",
                        "pad_ratios",
                        "slot_indices",
                    },
                )
                self.assertEqual(snapshot["z_e"].shape, (8, 16))

        self.assertIn("encoder_mean_norm", metrics)
        self.assertIn("participation_ratio", metrics)
        self.assertNotIn("used_codes", metrics)
        self.assertNotIn("nearest_code_distance_p50", metrics)

    def test_legacy_snapshot_rendering_filters_pad_rows_explicitly(self):
        with TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "legacy.npz"
            np.savez_compressed(
                snapshot_path,
                z_e=np.arange(8, dtype=np.float32).reshape(4, 2),
                pad_ratios=np.array([0.0, 0.5, 0.75, 1.0], dtype=np.float32),
                slot_indices=np.arange(4, dtype=np.int16),
                assignments=np.arange(4, dtype=np.int32),
            )
            with np.load(snapshot_path) as snapshot:
                view = _snapshot_encoder_view(snapshot)

        self.assertEqual(view.encoder.shape, (2, 2))
        self.assertEqual(view.pad_ratios.tolist(), [0.0, 0.5])
        self.assertEqual(view.slot_indices.tolist(), [0, 1])
        self.assertEqual(view.assignments.tolist(), [0, 1])

    def test_vq_snapshot_can_omit_codebook_and_use_transition_suffix(self):
        model = TextVQVAE(small_config())
        probe = [{
            "input_ids": torch.randint(0, 31, (2, 12)),
            "attention_mask": torch.ones(2, 12, dtype=torch.long),
        }]

        with TemporaryDirectory() as temp_dir:
            metrics = dump_geometry_snapshot(
                model,
                probe,
                5,
                Path(temp_dir),
                include_codebook=False,
                filename_suffix="_pre_kmeans",
            )
            snapshot_path = (
                Path(temp_dir) / "geometry" / "step000005_pre_kmeans.npz"
            )
            with np.load(snapshot_path) as snapshot:
                self.assertNotIn("codebook", snapshot.files)
                self.assertNotIn("assignments", snapshot.files)

        self.assertNotIn("used_codes", metrics)

    def test_successful_finalization_removes_raw_snapshots(self):
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            geometry_dir = run_dir / "geometry"
            geometry_dir.mkdir()
            (geometry_dir / "step000000.npz").write_bytes(b"raw snapshot")

            def fake_render(received_run_dir, basis, fps):
                self.assertEqual(received_run_dir, run_dir)
                self.assertEqual(basis, "first_last")
                self.assertEqual(fps, 8)
                plots_dir = run_dir / "plots"
                plots_dir.mkdir()
                outputs = {
                    "animation": plots_dir / "geometry_animation.mp4",
                    "trajectories": plots_dir / "geometry_code_trajectories.png",
                    "metrics": plots_dir / "geometry_metrics.png",
                }
                for path in outputs.values():
                    path.write_bytes(b"artifact")
                return outputs

            with patch(
                "visualization.render_geometry_animation.render_run",
                side_effect=fake_render,
            ):
                result = finalize_geometry_artifacts(
                    run_dir,
                    enabled=True,
                    basis="first_last",
                    fps=8,
                    keep_snapshots=False,
                )

            self.assertEqual(result["status"], "completed")
            self.assertFalse(result["snapshots_retained"])
            self.assertFalse(geometry_dir.exists())
            self.assertEqual(
                result["artifacts"]["animation"],
                "plots/geometry_animation.mp4",
            )

    def test_failed_finalization_preserves_raw_snapshots(self):
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            geometry_dir = run_dir / "geometry"
            geometry_dir.mkdir()
            snapshot = geometry_dir / "step000000.npz"
            snapshot.write_bytes(b"raw snapshot")

            with patch(
                "visualization.render_geometry_animation.render_run",
                side_effect=RuntimeError("render failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "render failed"):
                    finalize_geometry_artifacts(
                        run_dir,
                        enabled=True,
                        basis="first_last",
                        fps=8,
                        keep_snapshots=False,
                    )

            self.assertTrue(snapshot.exists())


class GeometryAnimationTest(unittest.TestCase):
    def test_geometry_metric_panels_define_meaning_and_units(self):
        current_metrics = {
            "valid_probe_points",
            "encoder_mean_norm",
            "encoder_norm_std",
            "encoder_pairwise_mean_distance",
            "nearest_code_distance_p10",
            "nearest_code_distance_p50",
            "nearest_code_distance_p90",
            "win_count_gini",
            "centroid_distance",
        }

        for key in current_metrics:
            title, ylabel = _geometry_metric_panel_labels(key)
            self.assertNotEqual(title, key)
            self.assertNotIn("source-defined", ylabel)

        pca = PCA(n_components=2).fit(
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        )
        self.assertIn("arbitrary latent units", _pca_component_label(pca, 0))
        self.assertIn("variance", _pca_component_label(pca, 1))

    def test_encoder_spectrum_metrics_include_rankme_and_twonn(self):
        equal_spectrum = np.concatenate([np.eye(4), -np.eye(4)], axis=0)

        metrics = _encoder_spectrum_metrics(equal_spectrum)

        self.assertEqual(len(metrics["pca_eigenvalues"]), 4)
        self.assertAlmostEqual(metrics["participation_ratio"], 4.0)
        self.assertAlmostEqual(metrics["rankme"], 4.0)
        self.assertEqual(metrics["twonn_points"], 8)

        line_metrics = _encoder_spectrum_metrics(
            np.array([[0.0], [1.0], [3.0]])
        )
        expected = 1.0 / (
            (np.log(3.0) + np.log(2.0) + np.log(1.5)) / 3.0
        )
        self.assertAlmostEqual(line_metrics["twonn_intrinsic_dim"], expected)

    def test_mixed_warmup_and_vq_snapshots_render_phase_artifacts(self):
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            geometry_dir = run_dir / "geometry"
            plots_dir = run_dir / "plots"
            geometry_dir.mkdir()
            plots_dir.mkdir()
            latent = np.array(
                [[0.0, 0.0], [1.0, .5], [2.0, 1.0], [3.0, 1.5]],
                dtype=np.float32,
            )
            common = {
                "pad_ratios": np.array([0.0, .25, .5, 1.0], dtype=np.float32),
                "slot_indices": np.array([0, 1, 2, 3], dtype=np.int16),
            }
            np.savez_compressed(
                geometry_dir / "step000000.npz",
                z_e=latent,
                **common,
            )
            np.savez_compressed(
                geometry_dir / "step000002_pre_kmeans.npz",
                z_e=latent + 1,
                **common,
            )
            codebook = latent + 1
            for name, offset in (
                ("step000002_post_kmeans.npz", 1.0),
                ("step000003.npz", 1.25),
            ):
                encoder = latent + offset
                np.savez_compressed(
                    geometry_dir / name,
                    z_e=encoder,
                    codebook=codebook,
                    assignments=np.arange(4, dtype=np.int32),
                    **common,
                )
            (run_dir / "metrics.jsonl").write_text(
                "\n".join([
                    json.dumps({
                        "split": "geometry", "step": 0,
                        "encoder_mean_norm": 1.0,
                    }),
                    json.dumps({
                        "split": "geometry", "step": 2,
                        "event": "pre_kmeans", "encoder_mean_norm": 2.0,
                    }),
                    json.dumps({
                        "split": "phase_transition", "step": 2,
                    }),
                    json.dumps({
                        "split": "geometry", "step": 2,
                        "event": "post_kmeans", "encoder_mean_norm": 2.0,
                        "used_codes": 4,
                    }),
                    json.dumps({
                        "split": "geometry", "step": 3,
                        "encoder_mean_norm": 2.5, "used_codes": 4,
                    }),
                ]) + "\n"
            )

            def fake_assemble(frame_paths, received_plots_dir, fps, *, stem):
                path = received_plots_dir / f"{stem}.mp4"
                path.write_bytes(b"video")
                return path

            with patch(
                "visualization.render_geometry_animation.assemble_animation",
                side_effect=fake_assemble,
            ):
                outputs = render_run(run_dir, keep_frames=False)

            ordered = [path.name for _, path in load_snapshots(run_dir)]
            self.assertEqual(
                ordered,
                [
                    "step000000.npz",
                    "step000002_pre_kmeans.npz",
                    "step000002_post_kmeans.npz",
                    "step000003.npz",
                ],
            )
            for key in (
                "animation",
                "ae_warmup_animation",
                "vq_animation",
                "latent_trajectory",
                "code_trajectories",
                "transition",
                "metrics",
            ):
                self.assertTrue(outputs[key].is_file(), key)

    def test_frames_use_global_scales_without_pad_coloring(self):
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            geometry_dir = run_dir / "geometry"
            geometry_dir.mkdir()
            snapshots = []
            payloads = [
                (
                    np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]], dtype=np.float32),
                    np.array([[0.0, 0.0], [2.0, 1.0], [4.0, 2.0], [6.0, 3.0]], dtype=np.float32),
                    np.array([0, 1, 1], dtype=np.int32),
                ),
                (
                    np.array([[8.0, 4.0], [10.0, 5.0], [12.0, 6.0]], dtype=np.float32),
                    np.array([[1.0, 0.0], [5.0, 2.0], [9.0, 4.0], [13.0, 6.0]], dtype=np.float32),
                    np.array([2, 2, 3], dtype=np.int32),
                ),
            ]
            for step, (encoder, codebook, assignments) in enumerate(payloads):
                path = geometry_dir / f"step{step:06d}.npz"
                # Deliberately omit pad_ratios: animation color must not depend on it.
                np.savez_compressed(
                    path,
                    z_e=encoder,
                    codebook=codebook,
                    assignments=assignments,
                )
                snapshots.append((step, path))

            pca = fit_shared_pca(snapshots, "first_last")
            scales = compute_animation_scales(snapshots, pca)
            self.assertEqual(scales.rank_xlim, (1.0, 4.0))
            self.assertGreaterEqual(scales.norm_bins[-1], np.linalg.norm(payloads[-1][1], axis=1).max())
            per_frame_nearest_max = max(
                np.histogram(
                    np.linalg.norm(encoder - codebook[assignments], axis=1),
                    bins=scales.nearest_bins,
                )[0].max()
                for encoder, codebook, assignments in payloads
            )
            self.assertAlmostEqual(scales.nearest_ylim[1], per_frame_nearest_max * 1.08)

            for step, path in snapshots:
                output = run_dir / f"frame{step}.png"
                render_frame(step, path, pca, output, scales)
                self.assertTrue(output.is_file())

    def test_continuous_frames_use_latent_only_snapshots(self):
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            geometry_dir = run_dir / "geometry"
            geometry_dir.mkdir()
            snapshots = []
            for step, offset in ((0, 0.0), (10, 1.0)):
                path = geometry_dir / f"step{step:06d}.npz"
                encoder = np.array(
                    [[0.0, 0.0], [1.0, .5], [2.0, 1.0], [3.0, 1.5]],
                    dtype=np.float32,
                ) + offset
                np.savez_compressed(
                    path,
                    z_e=encoder,
                    pad_ratios=np.array([0.0, .25, .5, 1.0], dtype=np.float32),
                    slot_indices=np.array([0, 1, 2, 3], dtype=np.int16),
                )
                snapshots.append((step, path))

            pca = fit_shared_pca(snapshots, "first_last")
            scales = compute_animation_scales(snapshots, pca)
            output = run_dir / "continuous_frame.png"
            render_frame(
                snapshots[-1][0],
                snapshots[-1][1],
                pca,
                output,
                scales,
                run_name="continuous-control",
            )

            self.assertTrue(output.is_file())
            self.assertEqual(scales.rank_xlim, (1.0, 2.0))


class CheckpointRetentionTest(unittest.TestCase):
    def test_keeps_best_and_two_most_recent_regular_checkpoints(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())

        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            (run_dir / "checkpoints").mkdir()

            save_checkpoint(model, optimizer, 1, 1, run_dir, "best.pt")
            save_checkpoint(model, optimizer, 10, 1, run_dir, "step10.pt")
            save_checkpoint(model, optimizer, 20, 1, run_dir, "step20.pt")
            save_checkpoint(model, optimizer, 30, 1, run_dir, "step30.pt")

            self.assertEqual(
                {path.name for path in (run_dir / "checkpoints").glob("*.pt")},
                {"best.pt", "step20.pt", "step30.pt"},
            )

    def test_final_checkpoint_counts_as_one_of_the_two_recent_files(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())

        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            (run_dir / "checkpoints").mkdir()

            save_checkpoint(model, optimizer, 10, 1, run_dir, "step10.pt")
            save_checkpoint(model, optimizer, 20, 1, run_dir, "step20.pt")
            save_checkpoint(model, optimizer, 20, 1, run_dir, "last.pt")

            self.assertEqual(
                {path.name for path in (run_dir / "checkpoints").glob("*.pt")},
                {"step20.pt", "last.pt"},
            )


class TextVQVAEDecoderTest(unittest.TestCase):
    def test_embedding_and_latent_dimensions_can_be_controlled_independently(self):
        for bottleneck_type in ("vq", "continuous"):
            with self.subTest(bottleneck_type=bottleneck_type):
                model = TextVQVAE(
                    small_config(
                        bottleneck_type=bottleneck_type,
                        latent_dim=7,
                    )
                )
                input_ids = torch.randint(0, 31, (2, 12))
                outputs = model(input_ids)

                self.assertEqual(model.token_embedding.embedding_dim, 16)
                self.assertEqual(model.latent_dim, 7)
                self.assertEqual(model.latent_proj.in_features, 16)
                self.assertEqual(model.latent_proj.out_features, 7)
                self.assertIsInstance(model.decoder_input_proj, nn.Linear)
                self.assertEqual(model.decoder_input_proj.in_features, 7)
                self.assertEqual(model.decoder_input_proj.out_features, 16)
                self.assertEqual(outputs["z_e"].shape, (2, 4, 7))
                self.assertEqual(outputs["z_latent"].shape, (2, 4, 7))
                self.assertEqual(outputs["logits"].shape, (2, 12, 32))
                if bottleneck_type == "vq":
                    self.assertEqual(
                        model.quantizer.codebook.weight.shape,
                        (8, 7),
                    )
                else:
                    self.assertIsNone(model.quantizer)

                outputs["logits"].sum().backward()
                self.assertIsNotNone(model.latent_proj.weight.grad)
                self.assertIsNotNone(model.decoder_input_proj.weight.grad)

    def test_default_latent_dimension_still_follows_d_model(self):
        model = TextVQVAE(small_config())

        self.assertEqual(model.latent_dim, 16)
        self.assertIsInstance(model.decoder_input_proj, nn.Identity)
        self.assertEqual(model.quantizer.codebook.weight.shape, (8, 16))

    def test_continuous_bottleneck_bypasses_quantization_and_vq_losses(self):
        model = TextVQVAE(small_config(bottleneck_type="continuous"))
        input_ids = torch.randint(0, 31, (2, 12))
        attention_mask = torch.ones_like(input_ids)

        outputs = model(input_ids, attention_mask)
        losses = text_vqvae_losses(
            outputs,
            input_ids,
            pad_token_id=model.config.pad_token_id,
            beta=model.config.commitment_beta,
            attention_mask=attention_mask,
        )

        self.assertIsNone(model.quantizer)
        self.assertEqual(outputs["bottleneck_type"], "continuous")
        self.assertIs(outputs["z_latent"], outputs["z_e"])
        self.assertNotIn("indices", outputs)
        self.assertNotIn("z_q_raw", outputs)
        self.assertEqual(set(losses), {"total", "recon"})
        torch.testing.assert_close(losses["total"], losses["recon"])
        losses["total"].backward()
        self.assertIsNotNone(model.latent_proj.weight.grad)
        self.assertIsNotNone(model.output_head.weight.grad)

    def test_continuous_bottleneck_rejects_vq_only_controls(self):
        with self.assertRaisesRegex(ValueError, "require bottleneck_type='vq'"):
            TextVQVAE(
                small_config(bottleneck_type="continuous"),
                collapse_config=CollapseControlConfig(use_ema_codebook=True),
            )

    def test_unknown_bottleneck_type_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "Unknown bottleneck_type"):
            TextVQVAE(small_config(bottleneck_type="unknown"))

    def test_nonpositive_latent_dimension_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "latent_dim must be positive"):
            TextVQVAE(small_config(latent_dim=0))

    def test_all_decoders_forward_and_backward(self):
        for decoder_type in DECODER_TYPES:
            with self.subTest(decoder_type=decoder_type):
                overrides = {"decoder_type": decoder_type}
                if decoder_type == "vqganr":
                    overrides["max_seq_len"] = 16
                model = TextVQVAE(small_config(**overrides))
                memory = torch.randn(2, 4, 16, requires_grad=True)
                logits = model.decode(memory, seq_len=9)
                self.assertEqual(logits.shape, (2, 9, 32))
                logits.sum().backward()
                self.assertIsNotNone(memory.grad)

                outputs = model(torch.randint(0, 31, (2, 12)))
                self.assertEqual(outputs["logits"].shape, (2, 12, 32))
                outputs["logits"].sum().backward()

    def test_all_decoders_ignore_masked_latent_values_and_zero_masked_outputs(self):
        latent_mask = torch.tensor(
            [[True, True, False, False], [True, False, True, False]]
        )
        output_mask = torch.tensor(
            [
                [True, True, True, True, True, False, False, False, False],
                [True, True, True, True, True, True, True, False, False],
            ]
        )
        first_memory = torch.randn(2, 4, 16)
        second_memory = first_memory.clone()
        second_memory[~latent_mask] = torch.randn_like(second_memory[~latent_mask])

        for decoder_type in DECODER_TYPES:
            with self.subTest(decoder_type=decoder_type):
                overrides = {"decoder_type": decoder_type}
                if decoder_type == "vqganr":
                    overrides["max_seq_len"] = 16
                model = TextVQVAE(small_config(**overrides))
                model.eval()
                with torch.no_grad():
                    first = model.decode(
                        first_memory,
                        seq_len=9,
                        latent_mask=latent_mask,
                        output_mask=output_mask,
                    )
                    second = model.decode(
                        second_memory,
                        seq_len=9,
                        latent_mask=latent_mask,
                        output_mask=output_mask,
                    )
                    decoder_hidden = model.decoder_impl(
                        model.decoder_input_proj(first_memory),
                        seq_len=9,
                        latent_mask=latent_mask,
                        output_mask=output_mask,
                    )

                torch.testing.assert_close(first, second)
                torch.testing.assert_close(
                    decoder_hidden[~output_mask],
                    torch.zeros_like(decoder_hidden[~output_mask]),
                )

    def test_all_decoders_keep_fully_masked_samples_finite(self):
        memory = torch.randn(2, 4, 16)
        latent_mask = torch.zeros(2, 4, dtype=torch.bool)
        output_mask = torch.zeros(2, 9, dtype=torch.bool)

        for decoder_type in DECODER_TYPES:
            with self.subTest(decoder_type=decoder_type):
                overrides = {"decoder_type": decoder_type}
                if decoder_type == "vqganr":
                    overrides["max_seq_len"] = 16
                model = TextVQVAE(small_config(**overrides))
                model.eval()
                with torch.no_grad():
                    hidden = model.decoder_impl(
                        model.decoder_input_proj(memory),
                        seq_len=9,
                        latent_mask=latent_mask,
                        output_mask=output_mask,
                    )

                self.assertTrue(torch.isfinite(hidden).all())
                torch.testing.assert_close(hidden, torch.zeros_like(hidden))

    def test_memory_trunk_is_default(self):
        model = TextVQVAE(small_config())
        self.assertIsInstance(model.decoder_impl, MemoryTrunkTextDecoder)

    def test_memory_trunk_uses_rope_without_cross_attention_or_position_embedding(self):
        model = TextVQVAE(small_config(decoder_type="memory_trunk"))
        decoder = model.decoder_impl

        self.assertIsInstance(decoder, MemoryTrunkTextDecoder)
        self.assertFalse(any(isinstance(module, nn.TransformerDecoder) for module in decoder.modules()))
        self.assertFalse(any(isinstance(module, nn.Embedding) for module in decoder.modules()))
        self.assertTrue(
            all(isinstance(block.attention, RotarySelfAttention) for block in decoder.latent_blocks)
        )
        self.assertTrue(
            all(isinstance(block.attention, RotarySelfAttention) for block in decoder.output_blocks)
        )

    def test_subpixel_sequence_order(self):
        upsampler = SubPixelSequenceUpsampler(d_model=2, upscale_factor=2)
        with torch.no_grad():
            upsampler.projection.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [10.0, 0.0],
                        [0.0, 10.0],
                    ]
                )
            )
            upsampler.projection.bias.zero_()

        result = upsampler(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]))
        expected = torch.tensor([[[1.0, 2.0], [10.0, 20.0], [3.0, 4.0], [30.0, 40.0]]])
        torch.testing.assert_close(result, expected)

    def test_non_default_integer_upscale_factor(self):
        model = TextVQVAE(
            small_config(max_seq_len=20, latent_slots=4, decoder_type="memory_trunk")
        )
        self.assertEqual(model.decoder_impl.upsampler.upscale_factor, 5)
        self.assertEqual(model.decode(torch.randn(2, 4, 16), seq_len=20).shape, (2, 20, 32))

    def test_invalid_memory_trunk_ratio_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            TextVQVAE(
                small_config(max_seq_len=10, latent_slots=4, decoder_type="memory_trunk")
            )

    def test_invalid_decoder_type_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "Unknown decoder_type"):
            TextVQVAE(small_config(decoder_type="unknown"))

    def test_vqgans_decoder_is_symmetric_and_uses_dynamic_stride(self):
        decoder = TextVQVAE(
            small_config(decoder_type="vqgans")
        ).decoder_impl

        self.assertIsInstance(decoder, VQGANTextDecoder)
        self.assertEqual(decoder.compression_factor, 3)
        self.assertEqual(decoder.transposed_conv.kernel_size, (3,))
        self.assertEqual(decoder.transposed_conv.stride, (3,))
        self.assertEqual(len(decoder.attention_blocks), 2)

    def test_decode_rejects_length_above_configured_maximum(self):
        for decoder_type in DECODER_TYPES:
            with self.subTest(decoder_type=decoder_type):
                overrides = {"decoder_type": decoder_type}
                if decoder_type == "vqganr":
                    overrides["max_seq_len"] = 8
                model = TextVQVAE(small_config(**overrides))
                with self.assertRaisesRegex(ValueError, "seq_len"):
                    model.decode(torch.randn(2, 4, 16), seq_len=13)

    def test_inference_returns_logits_truncated_by_side_channel_lengths(self):
        model = TextVQVAE(small_config(decoder_type="memory_trunk"))
        input_ids = torch.randint(0, 31, (2, 12))
        attention_mask = torch.zeros(2, 12, dtype=torch.long)
        attention_mask[0, :5] = 1
        attention_mask[1, :9] = 1

        dense_outputs = model(input_ids, attention_mask)
        inference_outputs = model.infer(input_ids, attention_mask)

        torch.testing.assert_close(dense_outputs["lengths"], torch.tensor([5, 9]))
        self.assertEqual(
            [tuple(logits.shape) for logits in inference_outputs["logits"]],
            [(5, 32), (9, 32)],
        )

    def test_original_cross_attention_checkpoint_keys_are_migrated(self):
        model = TextVQVAE(small_config(decoder_type="cross_attention"))
        legacy_state = OrderedDict()
        for key, value in model.state_dict().items():
            key = key.replace("decoder_impl.position_embedding.", "decoder_pos_embedding.")
            key = key.replace("decoder_impl.transformer.", "decoder.")
            key = key.replace("decoder_impl.norm.", "decoder_norm.")
            legacy_state[key] = value

        model.load_state_dict(legacy_state, strict=True)


class TextVQVAEEncoderTest(unittest.TestCase):
    def test_rope_encoder_remains_default(self):
        model = TextVQVAE(small_config())

        self.assertEqual(model.config.encoder_type, "rope")
        self.assertIsInstance(model.encoder, RotaryTextEncoder)
        self.assertIsNone(model.encoder_pos_embedding)

    def test_rope_encoder_has_no_absolute_position_embedding(self):
        model = TextVQVAE(small_config(encoder_type="rope"))

        self.assertIsInstance(model.encoder, RotaryTextEncoder)
        self.assertIsNone(model.encoder_pos_embedding)
        self.assertTrue(
            all(
                isinstance(layer.attention, RotarySelfAttention)
                for layer in model.encoder.layers
            )
        )

    def test_all_encoders_forward_and_backward(self):
        for encoder_type in ENCODER_TYPES:
            with self.subTest(encoder_type=encoder_type):
                overrides = {"encoder_type": encoder_type}
                if encoder_type == "vqganr":
                    overrides["max_seq_len"] = 16
                model = TextVQVAE(small_config(**overrides))
                outputs = model(torch.randint(0, 31, (2, 12)))

                self.assertEqual(outputs["logits"].shape, (2, 12, 32))
                outputs["logits"].sum().backward()
                self.assertIsNotNone(model.token_embedding.weight.grad)

    def test_all_encoder_decoder_combinations_forward_and_backward(self):
        for encoder_type in ENCODER_TYPES:
            for decoder_type in DECODER_TYPES:
                with self.subTest(
                    encoder_type=encoder_type,
                    decoder_type=decoder_type,
                ):
                    model = TextVQVAE(
                        small_config(
                            encoder_type=encoder_type,
                            decoder_type=decoder_type,
                            max_seq_len=(
                                16
                                if "vqganr" in (encoder_type, decoder_type)
                                else 12
                            ),
                        )
                    )
                    outputs = model(torch.randint(0, 31, (2, 12)))
                    self.assertEqual(outputs["z_e"].shape, (2, 4, 16))
                    self.assertEqual(outputs["logits"].shape, (2, 12, 32))
                    outputs["logits"].sum().backward()
                    self.assertIsNotNone(model.token_embedding.weight.grad)

    def test_rope_encoder_masks_padding_keys(self):
        model = TextVQVAE(small_config(encoder_type="rope"))
        model.eval()
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]
        )
        first = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
        second = first.clone()
        second[:, 6:] = torch.tensor([13, 14, 15, 16, 17, 18])

        with torch.no_grad():
            first_latents = model.encode(first, attention_mask=attention_mask)
            second_latents = model.encode(second, attention_mask=attention_mask)

        torch.testing.assert_close(first_latents, second_latents)

    def test_invalid_encoder_type_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "Unknown encoder_type"):
            TextVQVAE(small_config(encoder_type="unknown"))

    def test_pre_registry_encoder_checkpoint_keys_are_migrated(self):
        for encoder_type in ("absolute", "rope", "vqgans"):
            with self.subTest(encoder_type=encoder_type):
                model = TextVQVAE(small_config(encoder_type=encoder_type))
                legacy_state = OrderedDict()
                for key, value in model.state_dict().items():
                    key = key.replace("encoder.norm.", "encoder_norm.")
                    if encoder_type == "absolute":
                        key = key.replace(
                            "encoder.position_embedding.",
                            "encoder_pos_embedding.",
                        )
                        key = key.replace(
                            "encoder.transformer.layers.",
                            "encoder.layers.",
                        )
                    legacy_state[key] = value

                model.load_state_dict(legacy_state, strict=True)

    def test_vqgans_encoder_decoder_are_symmetric(self):
        model = TextVQVAE(
            small_config(encoder_type="vqgans", decoder_type="vqgans")
        )

        self.assertIsInstance(model.encoder, VQGANTextEncoder)
        self.assertIsInstance(model.decoder_impl, VQGANTextDecoder)
        self.assertIsNone(model.encoder_pos_embedding)
        self.assertEqual(model.encoder.compression_factor, 3)
        self.assertEqual(model.encoder.strided_conv.kernel_size, (3,))
        self.assertEqual(model.encoder.strided_conv.stride, (3,))
        self.assertEqual(model.decoder_impl.transposed_conv.kernel_size, (3,))
        self.assertEqual(model.decoder_impl.transposed_conv.stride, (3,))
        self.assertEqual(len(model.encoder.attention_blocks), 2)
        self.assertEqual(len(model.decoder_impl.attention_blocks), 2)
        self.assertTrue(
            all(
                isinstance(block.attention, PlainSelfAttention)
                for block in (
                    *model.encoder.attention_blocks,
                    *model.decoder_impl.attention_blocks,
                )
            )
        )

        outputs = model(torch.randint(0, 31, (2, 12)))
        self.assertEqual(outputs["z_e"].shape, (2, 4, 16))
        self.assertEqual(outputs["logits"].shape, (2, 12, 32))
        outputs["logits"].sum().backward()

        shorter_outputs = model(torch.randint(0, 31, (2, 9)))
        self.assertEqual(shorter_outputs["z_e"].shape, (2, 4, 16))
        self.assertEqual(shorter_outputs["logits"].shape, (2, 9, 32))

    def test_vqganpa_adds_symmetric_full_resolution_attention(self):
        model = TextVQVAE(
            small_config(encoder_type="vqganpa", decoder_type="vqganpa")
        )

        self.assertIsInstance(model.encoder, VQGANPreAttentionTextEncoder)
        self.assertIsInstance(model.decoder_impl, VQGANPreAttentionTextDecoder)
        self.assertIsInstance(model.encoder.pre_attention.attention, PlainSelfAttention)
        self.assertIsInstance(model.decoder_impl.post_attention.attention, PlainSelfAttention)
        self.assertEqual(len(model.encoder.attention_blocks), 2)
        self.assertEqual(len(model.decoder_impl.attention_blocks), 2)
        self.assertEqual(model.encoder.strided_conv.stride, (3,))
        self.assertEqual(model.decoder_impl.transposed_conv.stride, (3,))

        encoder_attention_params = sum(
            parameter.numel() for parameter in model.encoder.pre_attention.parameters()
        )
        decoder_attention_params = sum(
            parameter.numel() for parameter in model.decoder_impl.post_attention.parameters()
        )
        self.assertEqual(encoder_attention_params, decoder_attention_params)

        encoder_order = []
        decoder_order = []
        encoder_modules = (
            ("pre", model.encoder.pre_attention),
            ("conv", model.encoder.strided_conv),
            ("bottleneck0", model.encoder.attention_blocks[0]),
            ("bottleneck1", model.encoder.attention_blocks[1]),
        )
        decoder_modules = (
            ("bottleneck0", model.decoder_impl.attention_blocks[0]),
            ("bottleneck1", model.decoder_impl.attention_blocks[1]),
            ("conv", model.decoder_impl.transposed_conv),
            ("post", model.decoder_impl.post_attention),
        )
        handles = [
            module.register_forward_hook(
                lambda _module, _inputs, _output, name=name: encoder_order.append(name)
            )
            for name, module in encoder_modules
        ]
        handles.extend(
            module.register_forward_hook(
                lambda _module, _inputs, _output, name=name: decoder_order.append(name)
            )
            for name, module in decoder_modules
        )
        try:
            model.encoder(torch.randn(2, 12, 16), torch.ones(2, 12, dtype=torch.bool))
            model.decoder_impl(torch.randn(2, 4, 16), seq_len=12)
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(encoder_order, ["pre", "conv", "bottleneck0", "bottleneck1"])
        self.assertEqual(decoder_order, ["bottleneck0", "bottleneck1", "conv", "post"])

        outputs = model(torch.randint(0, 31, (2, 12)))
        self.assertEqual(outputs["z_e"].shape, (2, 4, 16))
        self.assertEqual(outputs["logits"].shape, (2, 12, 32))
        outputs["logits"].sum().backward()

    def test_vqganr_builds_one_residual_attention_stage_per_compression_level(self):
        model = TextVQVAE(
            small_config(
                max_seq_len=16,
                latent_slots=4,
                latent_dim=7,
                encoder_type="vqganr",
                decoder_type="vqganr",
                vqganr_num_res_blocks=1,
            )
        )

        self.assertIsInstance(model.encoder, VQGANRTextEncoder)
        self.assertIsInstance(model.decoder_impl, VQGANRTextDecoder)
        self.assertEqual(model.encoder.num_levels, 2)
        self.assertEqual(model.decoder_impl.num_levels, 2)
        self.assertEqual(model.encoder.conv_out.out_channels, 7)
        self.assertEqual(model.decoder_impl.conv_in.in_channels, 7)
        self.assertIsInstance(model.latent_proj, nn.Identity)
        self.assertIsInstance(model.decoder_input_proj, nn.Identity)
        self.assertTrue(
            all(
                len(level.res_blocks) == 1
                and isinstance(level.res_blocks[0], TextResBlock)
                and isinstance(level.attention, TextAttnBlock)
                and isinstance(level.attention.attention, RotarySelfAttention)
                and not hasattr(level.attention, "ffn")
                and level.downsample.kernel_size == (4,)
                and level.downsample.stride == (2,)
                and level.downsample.padding == (1,)
                for level in model.encoder.levels
            )
        )
        self.assertTrue(
            all(
                len(level.res_blocks) == 2
                and all(
                    isinstance(block, TextResBlock)
                    for block in level.res_blocks
                )
                and isinstance(level.attention.attention, RotarySelfAttention)
                and level.upsample.kernel_size == (2,)
                and level.upsample.stride == (2,)
                for level in model.decoder_impl.levels
            )
        )

        outputs = model(torch.randint(0, 31, (2, 13)))
        self.assertEqual(outputs["z_e"].shape, (2, 4, 7))
        self.assertEqual(outputs["logits"].shape, (2, 13, 32))
        outputs["logits"].sum().backward()
        self.assertIsNotNone(model.encoder.conv_out.weight.grad)
        self.assertIsNotNone(model.decoder_impl.conv_in.weight.grad)

    def test_vqganr_preserves_independent_latent_dim_in_mixed_architectures(self):
        encoder_native = TextVQVAE(
            small_config(
                max_seq_len=16,
                latent_dim=7,
                encoder_type="vqganr",
                decoder_type="memory_trunk",
            )
        )
        decoder_native = TextVQVAE(
            small_config(
                max_seq_len=16,
                latent_dim=7,
                encoder_type="rope",
                decoder_type="vqganr",
            )
        )

        self.assertIsInstance(encoder_native.latent_proj, nn.Identity)
        self.assertIsInstance(encoder_native.decoder_input_proj, nn.Linear)
        self.assertEqual(
            encoder_native(torch.randint(0, 31, (2, 12)))["z_e"].shape,
            (2, 4, 7),
        )
        self.assertIsInstance(decoder_native.latent_proj, nn.Linear)
        self.assertIsInstance(decoder_native.decoder_input_proj, nn.Identity)
        self.assertEqual(
            decoder_native(torch.randint(0, 31, (2, 12)))["logits"].shape,
            (2, 12, 32),
        )

    def test_vqganr_parameter_budgets_across_compression_levels(self):
        cases = (
            (128, 3, 9_253_440, 9_857_792),
            (64, 1, 9_655_296, 11_065_152),
            (32, 1, 12_470_976, 14_686_336),
            (16, 1, 15_286_656, 18_307_520),
        )
        for latent_slots, num_res_blocks, expected_encoder, expected_decoder in cases:
            with self.subTest(
                latent_slots=latent_slots,
                num_res_blocks=num_res_blocks,
            ):
                config = TextVQVAEConfig(
                    max_seq_len=256,
                    latent_slots=latent_slots,
                    d_model=448,
                    n_heads=8,
                    vqganr_num_res_blocks=num_res_blocks,
                )
                encoder = VQGANRTextEncoder(config)
                decoder = VQGANRTextDecoder(config)
                self.assertEqual(
                    sum(parameter.numel() for parameter in encoder.parameters()),
                    expected_encoder,
                )
                self.assertEqual(
                    sum(parameter.numel() for parameter in decoder.parameters()),
                    expected_decoder,
                )

    def test_vqganr_downsamples_mask_at_every_level(self):
        encoder = VQGANRTextEncoder(
            small_config(
                max_seq_len=16,
                latent_slots=4,
                vqganr_num_res_blocks=1,
            )
        )
        valid_mask = torch.zeros(1, 16, dtype=torch.bool)
        valid_mask[:, :9] = True
        hidden = torch.randn(1, 16, 16)
        hidden = encoder.conv_in(hidden.transpose(1, 2)).transpose(1, 2)

        hidden, valid_mask = encoder.levels[0](hidden, valid_mask)
        self.assertEqual(valid_mask.tolist(), [[True] * 5 + [False] * 3])
        torch.testing.assert_close(
            hidden[:, 5:],
            torch.zeros_like(hidden[:, 5:]),
        )
        hidden, valid_mask = encoder.levels[1](hidden, valid_mask)
        self.assertEqual(valid_mask.tolist(), [[True, True, True, False]])
        torch.testing.assert_close(
            hidden[:, 3:],
            torch.zeros_like(hidden[:, 3:]),
        )

    def test_vqgans_stride_tracks_compression_ratio(self):
        model = TextVQVAE(
            small_config(
                max_seq_len=20,
                latent_slots=4,
                encoder_type="vqgans",
                decoder_type="vqgans",
            )
        )

        self.assertEqual(model.encoder.compression_factor, 5)
        self.assertEqual(model.decoder_impl.compression_factor, 5)
        self.assertEqual(model(torch.randint(0, 31, (2, 20)))["logits"].shape, (2, 20, 32))

    def test_vqgans_requires_integer_compression_ratio(self):
        for architecture_type in ("vqgans", "vqganpa"):
            for field in ("encoder_type", "decoder_type"):
                with self.subTest(
                    architecture_type=architecture_type,
                    field=field,
                ):
                    overrides = {
                        field: architecture_type,
                        "max_seq_len": 10,
                        "latent_slots": 4,
                    }
                    with self.assertRaisesRegex(ValueError, "integer multiple"):
                        TextVQVAE(small_config(**overrides))

    def test_vqganr_requires_power_of_two_compression_ratio(self):
        for field in ("encoder_type", "decoder_type"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "integer multiple"):
                    TextVQVAE(
                        small_config(
                            **{
                                field: "vqganr",
                                "max_seq_len": 10,
                                "latent_slots": 4,
                            }
                        )
                    )
                with self.assertRaisesRegex(ValueError, "power of two"):
                    TextVQVAE(
                        small_config(
                            **{
                                field: "vqganr",
                                "max_seq_len": 12,
                                "latent_slots": 4,
                            }
                        )
                    )

    def test_vqganr_requires_positive_residual_block_count(self):
        for field in ("encoder_type", "decoder_type"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ValueError,
                    "vqganr_num_res_blocks must be positive",
                ):
                    TextVQVAE(
                        small_config(
                            **{
                                field: "vqganr",
                                "max_seq_len": 16,
                                "vqganr_num_res_blocks": 0,
                            }
                        )
                    )

    def test_vqgan_encoders_mask_padding_before_strided_conv(self):
        for encoder_type in ("vqgans", "vqganpa", "vqganr"):
            with self.subTest(encoder_type=encoder_type):
                overrides = {"encoder_type": encoder_type}
                if encoder_type == "vqganr":
                    overrides["max_seq_len"] = 16
                model = TextVQVAE(small_config(**overrides))
                model.eval()
                attention_mask = torch.tensor(
                    [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]
                )
                first = torch.tensor(
                    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
                )
                second = first.clone()
                second[:, 6:] = torch.tensor([13, 14, 15, 16, 17, 18])

                with torch.no_grad():
                    first_latents = model.encode(
                        first,
                        attention_mask=attention_mask,
                    )
                    second_latents = model.encode(
                        second,
                        attention_mask=attention_mask,
                    )

                torch.testing.assert_close(first_latents, second_latents)


class TextVQVAEPaddingTest(unittest.TestCase):
    def test_pad_aware_pool_excludes_pad_heavy_segments(self):
        hidden = torch.tensor(
            [
                [[1.0], [2.0], [100.0], [4.0], [100.0], [100.0]],
                [[9.0], [9.0], [9.0], [8.0], [8.0], [8.0]],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        pooled, latent_mask = pad_aware_adaptive_pool1d(
            hidden,
            attention_mask,
            output_size=2,
        )

        torch.testing.assert_close(
            pooled,
            torch.tensor([[[1.5], [0.0]], [[0.0], [0.0]]]),
        )
        torch.testing.assert_close(
            latent_mask,
            torch.tensor([[True, False], [False, False]]),
        )

    def test_pad_ratio_at_threshold_remains_a_content_slot(self):
        hidden = torch.tensor([[[1.0], [100.0], [3.0], [5.0]]])
        attention_mask = torch.tensor([[1, 0, 1, 1]])

        pooled, latent_mask = pad_aware_adaptive_pool1d(
            hidden,
            attention_mask,
            output_size=2,
            slot_pad_ratio_threshold=0.5,
        )

        torch.testing.assert_close(pooled, torch.tensor([[[1.0], [4.0]]]))
        self.assertTrue(latent_mask.all())

    def test_slot_pad_ratio_threshold_is_validated(self):
        with self.assertRaisesRegex(ValueError, "slot_pad_ratio_threshold"):
            TextVQVAE(small_config(slot_pad_ratio_threshold=1.0))

    def test_pad_aware_pool_matches_adaptive_pool_when_all_tokens_are_valid(self):
        hidden = torch.randn(2, 11, 3)
        pooled, latent_mask = pad_aware_adaptive_pool1d(
            hidden,
            torch.ones(2, 11, dtype=torch.long),
            output_size=4,
        )
        expected = torch.nn.functional.adaptive_avg_pool1d(
            hidden.transpose(1, 2), 4
        ).transpose(1, 2)

        torch.testing.assert_close(pooled, expected)
        self.assertTrue(latent_mask.all())

    def test_fully_padded_latent_slots_remain_fixed_zero_through_quantization(self):
        model = TextVQVAE(small_config())
        input_ids = torch.full((1, 12), 31, dtype=torch.long)
        input_ids[:, :3] = torch.randint(0, 31, (1, 3))
        attention_mask = torch.zeros(1, 12, dtype=torch.long)
        attention_mask[:, :3] = 1

        outputs = model(input_ids, attention_mask)

        torch.testing.assert_close(
            outputs["latent_mask"],
            torch.tensor([[True, False, False, False]]),
        )
        torch.testing.assert_close(outputs["z_e"][:, 1:], torch.zeros(1, 3, 16))
        torch.testing.assert_close(outputs["z_q_raw"][:, 1:], torch.zeros(1, 3, 16))
        torch.testing.assert_close(outputs["z_q_st"][:, 1:], torch.zeros(1, 3, 16))
        torch.testing.assert_close(outputs["indices"][:, 1:], -torch.ones(1, 3, dtype=torch.long))

    def test_losses_accuracy_and_codebook_stats_ignore_padding(self):
        targets = torch.tensor([[1, 3, 3]])
        logits = torch.tensor(
            [[[0.0, 4.0, 0.0, 0.0], [9.0, 0.0, 0.0, 0.0], [9.0, 0.0, 0.0, 0.0]]]
        )
        base_outputs = {
            "logits": logits,
            "z_e": torch.tensor([[[1.0, 1.0], [100.0, 100.0]]]),
            "z_q_raw": torch.tensor([[[3.0, 3.0], [-100.0, -100.0]]]),
            "latent_mask": torch.tensor([[True, False]]),
            "distances": torch.zeros(1, 2, 4),
        }

        losses = text_vqvae_losses(
            base_outputs,
            targets,
            pad_token_id=3,
            beta=0.25,
            collapse_config=CollapseControlConfig(use_ema_codebook=False),
        )
        correct, total = compute_accuracy(logits, targets, pad_token_id=3)
        stats = codebook_stats(
            torch.tensor([[2, -1]]),
            codebook_size=4,
            valid_mask=base_outputs["latent_mask"],
        )

        self.assertAlmostEqual(losses["codebook"].item(), 4.0)
        self.assertAlmostEqual(losses["commitment"].item(), 4.0)
        self.assertEqual((correct, total), (1, 1))
        self.assertEqual(stats["used_codes"], 1)
        self.assertEqual(stats["counts"].tolist(), [0.0, 0.0, 1.0, 0.0])

    def test_bits_per_token_uses_code_entropy_and_valid_counts(self):
        self.assertAlmostEqual(
            compute_bits_per_token(
                codebook_perplexity=2.0,
                latent_count=4,
                token_count=8,
            ),
            0.5,
        )
        self.assertEqual(compute_bits_per_token(0.0, 0, 8), 0.0)

    def test_reconstruction_loss_and_accuracy_prefer_attention_mask(self):
        targets = torch.tensor([[1, 3, 3]])
        logits = torch.tensor(
            [[[0.0, 4.0, 0.0, 0.0], [9.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 9.0]]]
        )
        outputs = {
            "logits": logits,
            "z_e": torch.zeros(1, 1, 2),
            "z_q_raw": torch.zeros(1, 1, 2),
            "latent_mask": torch.ones(1, 1, dtype=torch.bool),
            "distances": torch.zeros(1, 1, 4),
        }
        attention_mask = torch.tensor([[1, 1, 0]])

        losses = text_vqvae_losses(
            outputs,
            targets,
            pad_token_id=3,
            beta=0.25,
            attention_mask=attention_mask,
        )
        expected = torch.nn.functional.cross_entropy(logits[:, :2].reshape(-1, 4), targets[:, :2].reshape(-1))
        correct, total = compute_accuracy(
            logits,
            targets,
            pad_token_id=3,
            attention_mask=attention_mask,
        )

        torch.testing.assert_close(losses["recon"], expected)
        self.assertEqual((correct, total), (1, 2))


class VectorQuantizerMaskingTest(unittest.TestCase):
    def test_only_valid_slots_are_sent_to_code_assignment(self):
        quantizer = VectorQuantizer(codebook_size=4, d_model=2)
        z_e = torch.randn(2, 3, 2)
        valid_mask = torch.tensor([[True, False, True], [False, False, True]])

        with patch.object(quantizer, "_select_codes", wraps=quantizer._select_codes) as select:
            outputs = quantizer(z_e, valid_mask=valid_mask)

        self.assertEqual(select.call_count, 1)
        self.assertEqual(select.call_args.args[0].shape, (3, 4))
        torch.testing.assert_close(outputs["indices"][~valid_mask], -torch.ones(3, dtype=torch.long))
        torch.testing.assert_close(outputs["distances"][~valid_mask], torch.zeros(3, 4))

    def test_all_invalid_slots_skip_code_assignment(self):
        quantizer = VectorQuantizer(codebook_size=4, d_model=2)
        z_e = torch.randn(1, 2, 2)
        valid_mask = torch.zeros(1, 2, dtype=torch.bool)

        with patch.object(quantizer, "_select_codes", wraps=quantizer._select_codes) as select:
            outputs = quantizer(z_e, valid_mask=valid_mask)

        select.assert_not_called()
        torch.testing.assert_close(outputs["z_q_raw"], z_e)
        torch.testing.assert_close(outputs["indices"], -torch.ones(1, 2, dtype=torch.long))


class PreVQL2NormalizationTest(unittest.TestCase):
    def test_normalizes_valid_latents_and_preserves_invalid_zero_slots(self):
        model = TextVQVAE(small_config(l2_normalize_before_vq=True))
        input_ids = torch.tensor([[1, 2, 3, 31, 31, 31, 31, 31, 31, 31, 31, 31]])
        attention_mask = input_ids != 31

        latents, latent_mask = model.encode(
            input_ids,
            attention_mask=attention_mask,
            return_mask=True,
        )

        valid_norms = latents[latent_mask].norm(dim=-1)
        torch.testing.assert_close(valid_norms, torch.ones_like(valid_norms))
        torch.testing.assert_close(latents[~latent_mask], torch.zeros_like(latents[~latent_mask]))


class TextVQVAEVisualizationTest(unittest.TestCase):
    def test_initial_pca_steps_are_composable_and_balanced(self):
        model = TextVQVAE(small_config())
        model.train()
        batch = {
            "input_ids": torch.randint(0, 31, (3, 12)),
            "attention_mask": torch.ones(3, 12, dtype=torch.long),
        }
        batch["attention_mask"][0, 6:] = 0

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "initial_pca.png"
            encoder_vectors = collect_encoder_vectors(
                model,
                [batch],
                max_points=7,
            )
            result = compare_vector_distributions_pca(
                encoder_vectors.vectors,
                model.quantizer.codebook.weight,
                encoder_pad_ratios=encoder_vectors.pad_ratios,
                fit_mode="balanced",
            )
            render_pca_comparison(result, output_path)
            save_pca_metadata(result, output_path.with_suffix(".json"))
            metadata = result.metadata()

            self.assertTrue(output_path.is_file())
            self.assertTrue(output_path.with_suffix(".json").is_file())
            self.assertEqual(metadata["encoder_points"], 7)
            self.assertEqual(metadata["codebook_points"], 8)
            self.assertEqual(metadata["original_dimension"], 16)
            self.assertEqual(metadata["fit_mode"], "balanced")
            self.assertEqual(metadata["fit_points_per_distribution"], 7)
            self.assertIn("encoder_norm_std", metadata)
            self.assertIn("codebook_norm_std", metadata)
            self.assertIn("encoder_to_nearest_code_mean_distance", metadata)
            self.assertIn("encoder_pairwise_mean_distance", metadata)
            self.assertEqual(len(result.encoder_pad_ratios), 7)
            torch.testing.assert_close(
                encoder_vectors.pad_ratios,
                torch.zeros(7),
            )
            self.assertEqual(len(metadata["explained_variance_ratio"]), 2)
            self.assertTrue(model.training)

    def test_pca_rejects_mismatched_dimensions(self):
        with self.assertRaisesRegex(ValueError, "dimensions must match"):
            compare_vector_distributions_pca(
                torch.randn(3, 4),
                torch.randn(3, 5),
            )

    def test_initial_distance_metrics_use_euclidean_distance(self):
        result = compare_vector_distributions_pca(
            torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]),
            torch.tensor([[0.0, 0.0]]),
            fit_mode="all",
        )

        self.assertAlmostEqual(result.encoder_to_nearest_code_mean_distance, 7.0 / 3.0)
        self.assertAlmostEqual(result.encoder_pairwise_mean_distance, 4.0)


class AdaptiveWarmupDiagnosticsTest(unittest.TestCase):
    def test_reverse_water_filling_on_equal_spectrum(self):
        level, active = reverse_water_filling(
            torch.ones(4),
            rate_bits=2.0,
        )

        self.assertAlmostEqual(level, 0.5)
        self.assertEqual(active, 4)

    def test_latent_spectrum_metrics_reports_full_pca_and_dimensions(self):
        vectors = torch.cat([torch.eye(4), -torch.eye(4)], dim=0)

        metrics = latent_spectrum_metrics(
            vectors,
            codebook_size=4,
            variance_threshold=0.99,
        )

        self.assertEqual(metrics["latent_points"], 8)
        self.assertEqual(metrics["latent_dimension"], 4)
        self.assertEqual(len(metrics["pca_eigenvalues"]), 4)
        self.assertEqual(metrics["latent_effective_dim"], 4)
        self.assertEqual(metrics["water_filling_effective_dim"], 4)

    def test_adaptive_probe_excludes_invalid_pad_slots(self):
        model = TextVQVAE(small_config(codebook_size=4))
        probe = [{
            "input_ids": torch.tensor([
                [1] * 12,
                [31] * 12,
            ]),
            "attention_mask": torch.tensor([
                [1] * 12,
                [0] * 12,
            ]),
        }]

        metrics = evaluate_adaptive_warmup(
            model,
            probe,
            codebook_size=4,
            variance_threshold=0.99,
        )

        self.assertEqual(metrics["latent_points"], 4)

    def test_controller_stops_on_plateau_and_has_max_step_fallback(self):
        controller = AdaptiveWarmupController(
            min_steps=2,
            max_steps=10,
            patience=2,
            tolerance=1,
        )
        decisions = [
            controller.observe(step, {
                "water_filling_effective_dim": water_dim,
                "latent_effective_dim": latent_dim,
            })
            for step, water_dim, latent_dim in (
                (2, 5, 7),
                (4, 6, 8),
                (6, 5, 7),
            )
        ]
        self.assertFalse(decisions[0]["should_stop"])
        self.assertFalse(decisions[1]["should_stop"])
        self.assertTrue(decisions[2]["should_stop"])
        self.assertEqual(decisions[2]["reason"], "dimension_plateau")

        max_controller = AdaptiveWarmupController(
            min_steps=2,
            max_steps=4,
            patience=5,
            tolerance=0,
        )
        decision = max_controller.observe(4, {
            "water_filling_effective_dim": 3,
            "latent_effective_dim": 4,
        })
        self.assertTrue(decision["should_stop"])
        self.assertEqual(decision["reason"], "max_steps")


class TextVQVAECodebookInitializationTest(unittest.TestCase):
    def test_kmeans_initialization_updates_codebook_and_ema_state(self):
        collapse_config = CollapseControlConfig(use_ema_codebook=True)
        model = TextVQVAE(small_config(), collapse_config=collapse_config)
        model.train()
        original_codebook = model.quantizer.codebook.weight.detach().clone()
        batches = [
            {
                "input_ids": torch.randint(0, 31, (3, 12)),
                "attention_mask": torch.ones(3, 12, dtype=torch.long),
            }
            for _ in range(2)
        ]

        result = initialize_codebook_kmeans(
            model,
            batches,
            torch.device("cpu"),
            seed=7,
        )

        self.assertEqual(result, {"method": "kmeans", "encoder_vectors": 24})
        self.assertTrue(model.training)
        self.assertFalse(torch.equal(model.quantizer.codebook.weight, original_codebook))
        self.assertTrue(torch.isfinite(model.quantizer.codebook.weight).all())
        torch.testing.assert_close(
            model.quantizer.ema_embed_sum,
            model.quantizer.codebook.weight,
        )
        torch.testing.assert_close(
            model.quantizer.ema_cluster_size,
            torch.ones(8),
        )

    def test_kmeans_initialization_requires_at_least_one_vector_per_code(self):
        model = TextVQVAE(small_config(codebook_size=16))
        batch = {
            "input_ids": torch.randint(0, 31, (1, 12)),
            "attention_mask": torch.ones(1, 12, dtype=torch.long),
        }

        with self.assertRaisesRegex(ValueError, "produced 4 vectors for 16 codes"):
            initialize_codebook_kmeans(
                model,
                [batch],
                torch.device("cpu"),
                seed=7,
            )


class ConfigDefaultsTest(unittest.TestCase):
    """Ensure CLI defaults and dataclass defaults are in sync (no double-source drift)."""

    def test_all_configuration_dataclasses_live_in_one_module(self):
        from common.text_vqvae_config import (
            CollapseControlConfig,
            DataConfig,
            DiagnosticsConfig,
            TextVQVAEConfig,
            TrainConfig,
        )

        config_classes = (
            TrainConfig,
            DataConfig,
            TextVQVAEConfig,
            CollapseControlConfig,
            DiagnosticsConfig,
        )
        self.assertTrue(all(cls.__module__ == "common.text_vqvae_config" for cls in config_classes))

    def test_categorical_text_configuration_fields_use_literal(self):
        from typing import Literal, get_origin, get_type_hints

        from common.text_vqvae_config import (
            DataConfig,
            DiagnosticsConfig,
            TextVQVAEConfig,
            TrainConfig,
        )

        categorical_fields = {
            TrainConfig: ("tokenizer", "codebook_init", "ae_warmup_mode"),
            DataConfig: ("source",),
            TextVQVAEConfig: ("bottleneck_type", "encoder_type", "decoder_type"),
            DiagnosticsConfig: ("initial_pca_fit_mode", "geometry_render_basis"),
        }
        for config_class, field_names in categorical_fields.items():
            hints = get_type_hints(config_class)
            for field_name in field_names:
                with self.subTest(
                    config_class=config_class.__name__,
                    field=field_name,
                ):
                    self.assertIs(get_origin(hints[field_name]), Literal)

    def _parser(self):
        import argparse
        from training.text_vqvae.config import add_arguments
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        return parser

    def _parse(self, *argv):
        return self._parser().parse_args(list(argv))

    def test_empty_cli_gives_none_for_all_overrideable_flags(self):
        """With no flags, all overrideable args should be None so dataclass defaults win."""
        args = self._parse()
        for attr, value in vars(args).items():
            self.assertIsNone(value, msg=f"--{attr} should default to None")

    def test_help_shows_dataclass_defaults_for_every_config_flag(self):
        parser = self._parser()
        for action in parser._actions:
            if action.dest != "help":
                self.assertTrue(
                    hasattr(action, "effective_default"),
                    msg=f"{action.option_strings} has no displayed effective default",
                )
        help_text = parser.format_help()
        self.assertIn("--batch-size BATCH_SIZE", help_text)
        self.assertIn("Batch size. [default: 32]", help_text)
        self.assertIn("Latent slots. [default: 128]", help_text)
        self.assertIn("Commitment beta start. [default: <unset>]", help_text)
        self.assertIn("Geometry render fps. [default: 8]", help_text)

    def test_print_config_writes_resolved_json_to_stdout_without_creating_a_run(self):
        import json
        import subprocess
        import sys
        import uuid

        repo_root = Path(__file__).resolve().parents[1]
        run_name = f"print_config_test_{uuid.uuid4().hex}"
        run_dir = repo_root / "outputs" / "text_vqvae" / run_name
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "training.run_text_vqvae_experiment",
                "--print-config",
                "--tokenizer",
                "byte",
                "--batch-size",
                "17",
                "--collapse-preset",
                "anti",
                "--run-name",
                run_name,
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(result.stderr, "")
        self.assertEqual(payload["train"]["batch_size"], 17)
        self.assertEqual(payload["train"]["run_name"], run_name)
        self.assertTrue(payload["collapse_control"]["use_ema_codebook"])
        self.assertTrue(payload["collapse_control"]["enabled"])
        self.assertEqual(payload["model"]["vocab_size"], 258)
        self.assertFalse(run_dir.exists())

    def test_dataclass_defaults_match_current_mainline(self):
        """Dataclass defaults describe the configuration recommended for new runs."""
        from training.text_vqvae.config import DataConfig, DiagnosticsConfig, TrainConfig
        train = TrainConfig()
        data = DataConfig()
        diagnostics = DiagnosticsConfig()
        model = TextVQVAEConfig()

        self.assertEqual(train.seed, 42)
        self.assertEqual(train.batch_size, 32)
        self.assertEqual(train.ae_warmup_mode, "fixed")
        self.assertEqual(train.ae_warmup_steps, 0)
        self.assertEqual(train.ae_warmup_min_steps, 1000)
        self.assertIsNone(train.ae_warmup_max_steps)
        self.assertAlmostEqual(train.lr, 3e-4)
        self.assertEqual(data.max_train_samples, 50000)
        self.assertEqual(data.val_fraction, 0.02)
        self.assertEqual(model.latent_slots, 128)
        self.assertEqual(model.slot_pad_ratio_threshold, 0.5)
        self.assertEqual(model.codebook_size, 3072)
        self.assertEqual(model.d_model, 448)
        self.assertIsNone(model.latent_dim)
        self.assertEqual(model.resolved_latent_dim, 448)
        self.assertEqual(model.max_seq_len, 256)
        self.assertEqual(model.bottleneck_type, "vq")
        self.assertEqual(model.encoder_type, "rope")
        self.assertEqual(model.vqganr_num_res_blocks, 1)
        self.assertFalse(model.l2_normalize_before_vq)
        self.assertEqual(diagnostics.initial_pca_max_points, 8192)
        self.assertEqual(diagnostics.initial_pca_fit_mode, "balanced")
        self.assertTrue(diagnostics.geometry_snapshot_enabled)
        self.assertEqual(diagnostics.geometry_dense_every, 50)
        self.assertEqual(diagnostics.geometry_dense_until, 1500)
        self.assertEqual(diagnostics.geometry_sparse_every, 500)
        self.assertEqual(diagnostics.geometry_probe_points, 4096)
        self.assertTrue(diagnostics.geometry_render_enabled)
        self.assertEqual(diagnostics.geometry_render_basis, "first_last")
        self.assertEqual(diagnostics.geometry_render_fps, 8)
        self.assertTrue(diagnostics.geometry_keep_snapshots)

    def test_geometry_snapshot_can_be_disabled_explicitly(self):
        from training.text_vqvae.config import build_diagnostics_config

        diagnostics = build_diagnostics_config(
            self._parse("--geometry-snapshot-enabled", "false")
        )
        self.assertFalse(diagnostics.geometry_snapshot_enabled)
        self.assertFalse(diagnostics.geometry_render_enabled)

    def test_encoder_type_can_be_selected_from_cli(self):
        from training.text_vqvae.config import build_configs

        tokenizer = SimpleNamespace(vocab_size=123, pad_token_id=0)
        _, _, absolute, _ = build_configs(
            self._parse("--encoder-type", "absolute"),
            tokenizer,
        )
        _, _, rope, _ = build_configs(
            self._parse("--encoder-type", "rope"),
            tokenizer,
        )

        self.assertEqual(absolute.encoder_type, "absolute")
        self.assertEqual(rope.encoder_type, "rope")

    def test_pre_vq_l2_normalization_can_be_selected_from_cli(self):
        from training.text_vqvae.config import build_configs

        tokenizer = SimpleNamespace(vocab_size=123, pad_token_id=0)
        _, _, enabled, _ = build_configs(
            self._parse("--l2-normalize-before-vq"),
            tokenizer,
        )
        _, _, disabled, _ = build_configs(
            self._parse("--no-l2-normalize-before-vq"),
            tokenizer,
        )

        self.assertTrue(enabled.l2_normalize_before_vq)
        self.assertFalse(disabled.l2_normalize_before_vq)

    def test_embedding_and_latent_dimensions_can_be_selected_from_cli(self):
        from training.text_vqvae.config import build_configs

        tokenizer = SimpleNamespace(vocab_size=123, pad_token_id=0)
        _, _, model, _ = build_configs(
            self._parse(
                "--d-model",
                "96",
                "--latent-dim",
                "24",
            ),
            tokenizer,
        )

        self.assertEqual(model.d_model, 96)
        self.assertEqual(model.latent_dim, 24)
        self.assertEqual(model.resolved_latent_dim, 24)

    def test_continuous_bottleneck_can_be_selected_from_cli(self):
        from training.text_vqvae.config import build_configs

        tokenizer = SimpleNamespace(vocab_size=123, pad_token_id=0)
        _, _, model, collapse = build_configs(
            self._parse(
                "--bottleneck-type",
                "continuous",
                "--no-ema-codebook",
            ),
            tokenizer,
        )

        self.assertEqual(model.bottleneck_type, "continuous")
        self.assertFalse(collapse.is_active)

    def test_ae_warmup_requires_vq_and_kmeans(self):
        from training.text_vqvae.config import build_configs

        tokenizer = SimpleNamespace(vocab_size=123, pad_token_id=0)
        train, _, model, _ = build_configs(
            self._parse("--ae-warmup-steps", "12"),
            tokenizer,
        )
        self.assertEqual(train.ae_warmup_steps, 12)
        self.assertEqual(model.bottleneck_type, "vq")

        with self.assertRaisesRegex(ValueError, "requires --bottleneck-type vq"):
            build_configs(
                self._parse(
                    "--bottleneck-type", "continuous",
                    "--ae-warmup-steps", "12",
                ),
                tokenizer,
            )
        with self.assertRaisesRegex(ValueError, "requires --codebook-init kmeans"):
            build_configs(
                self._parse(
                    "--codebook-init", "random",
                    "--ae-warmup-steps", "12",
                ),
                tokenizer,
            )

    def test_adaptive_ae_warmup_configuration(self):
        from training.text_vqvae.config import build_configs

        tokenizer = SimpleNamespace(vocab_size=123, pad_token_id=0)
        train, _, _, _ = build_configs(
            self._parse(
                "--ae-warmup-mode", "adaptive",
                "--ae-warmup-min-steps", "100",
                "--ae-warmup-max-steps", "1000",
                "--ae-warmup-check-every", "50",
            ),
            tokenizer,
        )
        self.assertEqual(train.ae_warmup_mode, "adaptive")
        self.assertEqual(train.ae_warmup_min_steps, 100)
        self.assertEqual(train.ae_warmup_max_steps, 1000)
        self.assertEqual(train.ae_warmup_check_every, 50)

        with self.assertRaisesRegex(ValueError, "max-steps is required"):
            build_configs(
                self._parse("--ae-warmup-mode", "adaptive"),
                tokenizer,
            )
        with self.assertRaisesRegex(ValueError, "only valid.*fixed"):
            build_configs(
                self._parse(
                    "--ae-warmup-mode", "adaptive",
                    "--ae-warmup-steps", "500",
                    "--ae-warmup-max-steps", "1000",
                ),
                tokenizer,
            )

    def test_continuous_bottleneck_rejects_vq_only_cli_options(self):
        from training.text_vqvae.config import build_configs

        tokenizer = SimpleNamespace(vocab_size=123, pad_token_id=0)
        with self.assertRaisesRegex(ValueError, "require --bottleneck-type vq"):
            build_configs(
                self._parse(
                    "--bottleneck-type",
                    "continuous",
                    "--use-ema-codebook",
                ),
                tokenizer,
            )
        with self.assertRaisesRegex(ValueError, "requires --bottleneck-type vq"):
            build_configs(
                self._parse(
                    "--bottleneck-type",
                    "continuous",
                    "--l2-normalize-before-vq",
                ),
                tokenizer,
            )

    def test_vqgan_encoder_and_decoder_variants_can_be_selected_from_cli(self):
        from training.text_vqvae.config import build_configs

        tokenizer = SimpleNamespace(vocab_size=123, pad_token_id=0)
        for architecture_type in ("vqgans", "vqganpa", "vqganr"):
            with self.subTest(architecture_type=architecture_type):
                _, _, model, _ = build_configs(
                    self._parse(
                        "--encoder-type",
                        architecture_type,
                        "--decoder-type",
                        architecture_type,
                    ),
                    tokenizer,
                )

                self.assertEqual(model.encoder_type, architecture_type)
                self.assertEqual(model.decoder_type, architecture_type)

    def test_vqganr_residual_block_count_can_be_selected_from_cli(self):
        from training.text_vqvae.config import build_configs

        tokenizer = SimpleNamespace(vocab_size=123, pad_token_id=0)
        _, _, model, _ = build_configs(
            self._parse("--vqganr-num-res-blocks", "3"),
            tokenizer,
        )

        self.assertEqual(model.vqganr_num_res_blocks, 3)

    def test_geometry_snapshots_can_be_retained_after_rendering(self):
        from training.text_vqvae.config import build_diagnostics_config

        diagnostics = build_diagnostics_config(
            self._parse("--geometry-keep-snapshots", "true")
        )
        self.assertTrue(diagnostics.geometry_keep_snapshots)

    def test_empty_cli_builds_each_dataclass_from_its_defaults(self):
        from training.text_vqvae.config import (
            DataConfig,
            DiagnosticsConfig,
            TrainConfig,
            build_configs,
            build_diagnostics_config,
            build_train_config,
        )

        args = self._parse()
        tokenizer = SimpleNamespace(vocab_size=123, pad_token_id=0)
        train_cfg = build_train_config(args)
        train_cfg, data_cfg, model_cfg, collapse_cfg = build_configs(
            args, tokenizer, train_cfg=train_cfg
        )

        expected_model = TextVQVAEConfig(vocab_size=123, pad_token_id=0)
        self.assertEqual(asdict(train_cfg), asdict(TrainConfig()))
        self.assertEqual(asdict(data_cfg), asdict(DataConfig()))
        self.assertEqual(asdict(model_cfg), asdict(expected_model))
        self.assertEqual(asdict(collapse_cfg), asdict(CollapseControlConfig()))
        self.assertNotIn("enabled", asdict(collapse_cfg))
        self.assertTrue(collapse_cfg.is_active)
        self.assertEqual(
            asdict(build_diagnostics_config(args)), asdict(DiagnosticsConfig())
        )

    def test_collapse_activity_is_derived_from_behavioral_fields(self):
        inactive = CollapseControlConfig(use_ema_codebook=False)
        self.assertFalse(inactive.is_active)

        inactive.dead_code_reset_every = 500
        self.assertTrue(inactive.is_active)

    def test_default_tokenizer_is_resolved_from_train_config(self):
        from training.run_text_vqvae_experiment import _resolve_tokenizer
        from training.text_vqvae.config import TrainConfig

        args = self._parse()
        with patch("training.run_text_vqvae_experiment.BPETokenizer") as tokenizer_cls:
            tokenizer_cls.return_value.path = Path("resolved-tokenizer.json")
            train_cfg, tokenizer, resolved_path = _resolve_tokenizer(args)

        tokenizer_cls.assert_called_once_with(TrainConfig().tokenizer_path)
        self.assertIs(tokenizer, tokenizer_cls.return_value)
        self.assertEqual(resolved_path, "resolved-tokenizer.json")


class LoadRunConfigTest(unittest.TestCase):
    """load_run_config should round-trip a real saved config.json."""

    def test_round_trip_versioned_legacy_fixture(self):
        from training.text_vqvae.config import load_run_config
        real_config = Path(__file__).parent / "fixtures" / "legacy_run_config.json"

        with self.assertWarnsRegex(UserWarning, "ignoring unknown keys"):
            train_cfg, data_cfg, model_cfg, collapse_cfg = load_run_config(real_config)

        self.assertEqual(model_cfg.codebook_size, 3072)
        self.assertEqual(model_cfg.latent_slots, 32)
        self.assertEqual(model_cfg.d_model, 448)
        self.assertEqual(train_cfg.seed, 12)
        self.assertEqual(data_cfg.max_train_samples, 50000)
        self.assertFalse(collapse_cfg.use_ema_codebook)

    def test_missing_keys_fill_defaults(self):
        """A minimal config.json (missing most fields) fills defaults without crashing."""
        import json, tempfile, os
        from training.text_vqvae.config import load_run_config
        minimal = {
            "train": {"run_name": "test_run", "seed": 99},
            "model": {"vocab_size": 256},
            "data": {},
            "collapse_control": {},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(minimal, f)
            path = f.name
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                train_cfg, data_cfg, model_cfg, collapse_cfg = load_run_config(path)
            self.assertEqual(train_cfg.seed, 99)
            self.assertEqual(model_cfg.codebook_size, 3072)   # default from dataclass
            messages = "\n".join(str(item.message) for item in caught)
            self.assertIn("TextVQVAEConfig", messages)
            self.assertIn("CollapseControlConfig", messages)
        finally:
            os.unlink(path)

    def test_stale_legacy_enabled_value_is_ignored(self):
        import json, os, tempfile

        from training.text_vqvae.config import load_run_config

        payload = {
            "train": {},
            "model": {},
            "data": {},
            "collapse_control": {
                "enabled": False,
                "use_ema_codebook": True,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            path = f.name
        try:
            with self.assertWarnsRegex(UserWarning, "ignoring stale derived enabled=False"):
                _, _, _, collapse_cfg = load_run_config(path)
            self.assertTrue(collapse_cfg.is_active)
        finally:
            os.unlink(path)


class TrainingLifecycleTest(unittest.TestCase):
    def test_best_checkpoint_prefers_reconstruction_nll_over_total_loss(self):
        from common.text_data import ByteTokenizer
        from training.text_vqvae.config import DataConfig, TrainConfig
        from training.text_vqvae.loop import run

        config = small_config()
        collapse_config = CollapseControlConfig()
        model = TextVQVAE(config, collapse_config=collapse_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batches = [{
            "input_ids": torch.randint(0, 31, (2, 12)),
            "attention_mask": torch.ones(2, 12, dtype=torch.long),
        } for _ in range(2)]
        tracker = SimpleNamespace(log=lambda *args, **kwargs: None, summary={})
        eval_results = [
            (
                {
                    "loss": 1.0,
                    "recon_nll": 2.0,
                    "token_ppl": 2.0,
                    "token_accuracy": 0.8,
                },
                [],
            ),
            (
                {
                    "loss": 2.0,
                    "recon_nll": 1.0,
                    "token_ppl": 2.0,
                    "token_accuracy": 0.7,
                },
                [],
            ),
        ]

        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            for child in ("checkpoints", "plots", "samples"):
                (run_dir / child).mkdir()
            payload = {
                "diagnostics": {
                    "initial_pca": {"status": "disabled"},
                    "geometry": {},
                },
            }
            with (
                patch(
                    "training.text_vqvae.loop.evaluate",
                    side_effect=eval_results,
                ),
                patch("training.text_vqvae.loop.write_reconstruction_samples"),
            ):
                run(
                    model=model,
                    optimizer=optimizer,
                    train_loader=batches,
                    val_loader=[batches[0]],
                    train_cfg=TrainConfig(
                        epochs=1,
                        eval_every=1,
                        save_every=100,
                        tokenizer="byte",
                        tokenizer_path=None,
                    ),
                    data_cfg=DataConfig(),
                    model_config=config,
                    collapse_config=collapse_config,
                    run_dir=run_dir,
                    run_name="best-by-nll",
                    tokenizer=ByteTokenizer(),
                    device=torch.device("cpu"),
                    config_payload=payload,
                    tracker=tracker,
                    initial_pca_opts={
                        "enabled": False,
                        "max_points": 8,
                        "fit_mode": "balanced",
                        "strict": True,
                    },
                    geometry_snapshot_opts={
                        "enabled": False,
                        "strict": True,
                        "render_enabled": False,
                    },
                )

            summary = json.loads((run_dir / "summary.json").read_text())
            checkpoint = torch.load(
                run_dir / "checkpoints" / "best.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(summary["best_selection_metric"], "recon_nll")
            self.assertEqual(summary["best_eval_nll"], 1.0)
            self.assertEqual(
                summary["compat_best_eval_loss_at_best_nll"],
                2.0,
            )
            self.assertEqual(summary["best_step"], 2)
            self.assertEqual(checkpoint["step"], 2)

    def test_adaptive_ae_warmup_transitions_after_plateau(self):
        from common.text_data import ByteTokenizer
        from training.text_vqvae.config import DataConfig, TrainConfig
        from training.text_vqvae.loop import run

        config = small_config(codebook_size=4)
        collapse_config = CollapseControlConfig()
        model = TextVQVAE(config, collapse_config=collapse_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batches = [{
            "input_ids": torch.randint(0, 31, (2, 12)),
            "attention_mask": torch.ones(2, 12, dtype=torch.long),
        } for _ in range(4)]
        tracker = SimpleNamespace(log=lambda *args, **kwargs: None, summary={})
        constant_spectrum = {
            "latent_points": 8,
            "latent_dimension": 16,
            "variance_threshold": 0.99,
            "latent_effective_dim": 6,
            "participation_ratio": 5.5,
            "rate_bits": 2.0,
            "water_filling_level": 0.1,
            "water_filling_effective_dim": 5,
            "pca_eigenvalues": [1.0] * 16,
        }

        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            for child in ("checkpoints", "plots", "samples"):
                (run_dir / child).mkdir()
            payload = {
                "codebook_initialization": {
                    "method": "kmeans",
                    "status": "deferred",
                    "mode": "adaptive",
                },
                "diagnostics": {
                    "initial_pca": {"status": "disabled"},
                    "geometry": {},
                },
            }
            with patch(
                "training.text_vqvae.loop.evaluate_adaptive_warmup",
                return_value=constant_spectrum,
            ):
                run(
                    model=model,
                    optimizer=optimizer,
                    train_loader=batches,
                    val_loader=[batches[0]],
                    codebook_init_loader=batches,
                    train_probe_loader=[batches[0]],
                    train_cfg=TrainConfig(
                        epochs=1,
                        eval_every=100,
                        save_every=100,
                        ae_warmup_mode="adaptive",
                        ae_warmup_min_steps=1,
                        ae_warmup_max_steps=3,
                        ae_warmup_check_every=1,
                        ae_warmup_patience=1,
                        ae_warmup_dim_tolerance=0,
                        ae_warmup_probe_points=8,
                        tokenizer="byte",
                        tokenizer_path=None,
                    ),
                    data_cfg=DataConfig(),
                    model_config=config,
                    collapse_config=collapse_config,
                    run_dir=run_dir,
                    run_name="adaptive-ae-warmup-smoke",
                    tokenizer=ByteTokenizer(),
                    device=torch.device("cpu"),
                    config_payload=payload,
                    tracker=tracker,
                    initial_pca_opts={
                        "enabled": False,
                        "max_points": 8,
                        "fit_mode": "balanced",
                        "strict": True,
                    },
                    geometry_snapshot_opts={
                        "enabled": False,
                        "strict": True,
                        "render_enabled": False,
                    },
                )

            rows = [
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text().splitlines()
            ]
            train_rows = [row for row in rows if row["split"] == "train"]
            self.assertEqual(
                [row["phase"] for row in train_rows],
                ["ae_warmup", "ae_warmup", "vq", "vq"],
            )
            diagnostics = [
                row for row in rows
                if row["split"] == "ae_warmup_diagnostic"
            ]
            self.assertEqual([row["step"] for row in diagnostics], [1, 2])
            transitions = [
                row for row in rows if row["split"] == "phase_transition"
            ]
            self.assertEqual(transitions[0]["step"], 2)
            self.assertEqual(
                transitions[0]["warmup_stop_reason"],
                "dimension_plateau",
            )
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["actual_ae_warmup_steps"], 2)
            self.assertEqual(summary["best_step"], 3)
            self.assertTrue(
                (run_dir / "plots" / "ae_warmup_diagnostics.png").is_file()
            )

    def test_ae_warmup_switches_to_kmeans_vq_before_next_step(self):
        from common.text_data import ByteTokenizer
        from training.text_vqvae.config import DataConfig, TrainConfig
        from training.text_vqvae.loop import run

        config = small_config(codebook_size=4)
        collapse_config = CollapseControlConfig()
        model = TextVQVAE(config, collapse_config=collapse_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batches = [{
            "input_ids": torch.randint(0, 31, (2, 12)),
            "attention_mask": torch.ones(2, 12, dtype=torch.long),
        } for _ in range(3)]
        tracker = SimpleNamespace(log=lambda *args, **kwargs: None, summary={})

        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            for child in ("checkpoints", "plots", "samples"):
                (run_dir / child).mkdir()
            payload = {
                "codebook_initialization": {
                    "method": "kmeans",
                    "status": "deferred",
                    "scheduled_after_step": 2,
                },
                "diagnostics": {
                    "initial_pca": {"status": "disabled"},
                    "geometry": {},
                },
            }
            run(
                model=model,
                optimizer=optimizer,
                train_loader=batches,
                val_loader=[batches[0]],
                codebook_init_loader=batches,
                train_probe_loader=[batches[0]],
                train_cfg=TrainConfig(
                    epochs=1,
                    eval_every=100,
                    save_every=100,
                    ae_warmup_steps=2,
                    tokenizer="byte",
                    tokenizer_path=None,
                ),
                data_cfg=DataConfig(),
                model_config=config,
                collapse_config=collapse_config,
                run_dir=run_dir,
                run_name="ae-warmup-smoke",
                tokenizer=ByteTokenizer(),
                device=torch.device("cpu"),
                config_payload=payload,
                tracker=tracker,
                initial_pca_opts={
                    "enabled": False,
                    "max_points": 8,
                    "fit_mode": "balanced",
                    "strict": True,
                },
                geometry_snapshot_opts={
                    "enabled": True,
                    "dense_every": 1,
                    "dense_until": 3,
                    "sparse_every": 1,
                    "probe_points": 8,
                    "strict": True,
                    "render_enabled": False,
                    "render_basis": "first_last",
                    "render_fps": 8,
                    "keep_snapshots": True,
                },
            )

            rows = [
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text().splitlines()
            ]
            train_rows = [row for row in rows if row["split"] == "train"]
            self.assertEqual(
                [row["phase"] for row in train_rows],
                ["ae_warmup", "ae_warmup", "vq"],
            )
            self.assertNotIn("codebook_utilization_batch", train_rows[0])
            self.assertNotIn("codebook_utilization_batch", train_rows[1])
            self.assertIn("codebook_utilization_batch", train_rows[2])
            transitions = [
                row for row in rows if row["split"] == "phase_transition"
            ]
            self.assertEqual(len(transitions), 1)
            self.assertEqual(transitions[0]["step"], 2)
            probe_rows = [row for row in rows if row["split"] == "codebook_probe"]
            self.assertEqual([row["step"] for row in probe_rows], [3])
            self.assertIn("train_used_codes", probe_rows[0])
            self.assertIn("eval_used_codes", probe_rows[0])
            eval_rows = [row for row in rows if row["split"] == "eval"]
            self.assertNotIn("used_codes", eval_rows[-1])
            self.assertIn("compat_full_eval_used_codes", eval_rows[-1])
            self.assertTrue((run_dir / "codebook_c0_kmeans.pt").is_file())
            self.assertTrue(
                (run_dir / "geometry" / "step000002_pre_kmeans.npz").is_file()
            )
            self.assertTrue(
                (run_dir / "geometry" / "step000002_post_kmeans.npz").is_file()
            )
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["best_step"], 3)
            self.assertEqual(summary["final_phase"], "vq")
            self.assertEqual(
                summary["final_codebook_probe"]["eval_used_codes"],
                probe_rows[-1]["eval_used_codes"],
            )

    def test_strict_initial_pca_failure_writes_failed_summary(self):
        import json
        from training.text_vqvae.config import DataConfig, TrainConfig
        from training.text_vqvae.loop import run

        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            payload = {"diagnostics": {"initial_pca": {"status": "pending"}}}
            with patch(
                "training.text_vqvae.loop.run_initial_pca",
                side_effect=RuntimeError("strict PCA failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "strict PCA failed"):
                    run(
                        model=None,
                        optimizer=None,
                        train_loader=None,
                        val_loader=None,
                        train_cfg=TrainConfig(run_name="pca_failure"),
                        data_cfg=DataConfig(),
                        model_config=TextVQVAEConfig(),
                        collapse_config=CollapseControlConfig(),
                        run_dir=run_dir,
                        run_name="pca_failure",
                        tokenizer=None,
                        device=torch.device("cpu"),
                        config_payload=payload,
                        tracker=SimpleNamespace(),
                        initial_pca_opts={
                            "enabled": True,
                            "max_points": 8,
                            "fit_mode": "balanced",
                            "strict": True,
                        },
                    )

            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["steps"], 0)
            self.assertIn("strict PCA failed", summary["error"])


if __name__ == "__main__":
    unittest.main()
