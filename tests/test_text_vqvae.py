from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

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
    codebook_stats,
    pad_aware_adaptive_pool1d,
    text_vqvae_losses,
)
from training.run_text_vqvae_experiment import (
    compute_accuracy,
    initialize_codebook_from_first_encoder_pass,
)
from visualization.text_vqvae import (
    collect_encoder_vectors,
    compare_vector_distributions_pca,
    render_pca_comparison,
    save_pca_metadata,
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
    def test_pad_aware_pool_excludes_pad_tokens_and_zeros_empty_segments(self):
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
            torch.tensor([[[1.5], [4.0]], [[0.0], [0.0]]]),
        )
        torch.testing.assert_close(
            latent_mask,
            torch.tensor([[True, True], [False, False]]),
        )

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

        result = initialize_codebook_from_first_encoder_pass(
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
            initialize_codebook_from_first_encoder_pass(
                model,
                [batch],
                torch.device("cpu"),
                seed=7,
            )


if __name__ == "__main__":
    unittest.main()
