from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch
import torch.nn as nn

from models.text_vqvae import (
    CollapseControlConfig,
    CrossAttentionTextDecoder,
    MemoryTrunkTextDecoder,
    RotarySelfAttention,
    SubPixelSequenceUpsampler,
    TextVQVAE,
    TextVQVAEConfig,
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
    make_loader,
    save_checkpoint,
)
from training.text_vqvae.geometry import dump_geometry_snapshot, finalize_geometry_artifacts
from visualization.text_vqvae import (
    collect_encoder_vectors,
    compare_vector_distributions_pca,
    render_pca_comparison,
    save_pca_metadata,
)
from visualization.render_geometry_animation import (
    compute_animation_scales,
    fit_shared_pca,
    render_frame,
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

        metrics, rows = evaluate(
            model,
            loader,
            torch.device("cpu"),
            config,
            collapse_config,
            beta=config.commitment_beta,
            tokenizer=tokenizer,
        )

        self.assertEqual(loader.iterations, 1)
        self.assertTrue(model.training)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["original"], "1 2 3 4 5 6")
        self.assertEqual(len(rows[0]["reconstruction"].split()), 6)
        self.assertIn("loss", metrics)
        self.assertEqual(len(metrics["code_counts"]), config.codebook_size)

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
        model = TextVQVAE(small_config())
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
        rng_before = torch.random.get_rng_state().clone()

        with TemporaryDirectory() as temp_dir:
            metrics = dump_geometry_snapshot(model, probe, 7, Path(temp_dir))
            with np.load(Path(temp_dir) / "geometry" / "step000007.npz") as snapshot:
                self.assertEqual(
                    set(snapshot.files),
                    {"z_e", "codebook", "assignments", "pad_ratios", "slot_indices"},
                )
                self.assertEqual(snapshot["z_e"].shape, (8, 16))
                self.assertEqual(snapshot["z_e"].dtype, np.float16)
                self.assertEqual(snapshot["codebook"].shape, (8, 16))
                self.assertEqual(snapshot["codebook"].dtype, np.float16)
                self.assertEqual(snapshot["assignments"].shape, (8,))
                self.assertEqual(snapshot["assignments"].dtype, np.int32)
                self.assertEqual(snapshot["pad_ratios"].shape, (8,))
                self.assertEqual(snapshot["slot_indices"].tolist(), [0, 1, 2, 3, 0, 1, 2, 3])

        self.assertTrue(model.training)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
        self.assertGreaterEqual(metrics["used_codes"], 1)
        self.assertLessEqual(metrics["used_codes"], 8)
        self.assertIn("participation_ratio", metrics)
        self.assertIn("win_count_gini", metrics)

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
    def test_both_decoders_forward_and_backward(self):
        for decoder_type in ("cross_attention", "memory_trunk"):
            with self.subTest(decoder_type=decoder_type):
                model = TextVQVAE(small_config(decoder_type=decoder_type))
                memory = torch.randn(2, 4, 16, requires_grad=True)
                logits = model.decode(memory, seq_len=9)
                self.assertEqual(logits.shape, (2, 9, 32))
                logits.sum().backward()
                self.assertIsNotNone(memory.grad)

                outputs = model(torch.randint(0, 31, (2, 12)))
                self.assertEqual(outputs["logits"].shape, (2, 12, 32))
                outputs["logits"].sum().backward()

    def test_cross_attention_is_default(self):
        model = TextVQVAE(small_config())
        self.assertIsInstance(model.decoder_impl, CrossAttentionTextDecoder)

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

    def test_decode_rejects_length_above_configured_maximum(self):
        for decoder_type in ("cross_attention", "memory_trunk"):
            with self.subTest(decoder_type=decoder_type):
                model = TextVQVAE(small_config(decoder_type=decoder_type))
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
        model = TextVQVAE(small_config())
        legacy_state = OrderedDict()
        for key, value in model.state_dict().items():
            key = key.replace("decoder_impl.position_embedding.", "decoder_pos_embedding.")
            key = key.replace("decoder_impl.transformer.", "decoder.")
            key = key.replace("decoder_impl.norm.", "decoder_norm.")
            legacy_state[key] = value

        model.load_state_dict(legacy_state, strict=True)


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
                encoder_vectors.pad_ratios[:4], torch.tensor([0.0, 0.0, 1.0, 1.0])
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
        self.assertAlmostEqual(train.lr, 3e-4)
        self.assertEqual(data.max_train_samples, 50000)
        self.assertEqual(data.val_fraction, 0.02)
        self.assertEqual(model.latent_slots, 128)
        self.assertEqual(model.slot_pad_ratio_threshold, 0.5)
        self.assertEqual(model.codebook_size, 3072)
        self.assertEqual(model.d_model, 448)
        self.assertEqual(model.max_seq_len, 256)
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
        self.assertEqual(
            asdict(build_diagnostics_config(args)), asdict(DiagnosticsConfig())
        )

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


class TrainingLifecycleTest(unittest.TestCase):
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
