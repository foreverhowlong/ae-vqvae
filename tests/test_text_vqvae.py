from collections import OrderedDict
import unittest

import torch
import torch.nn as nn

from models.text_vqvae import (
    CrossAttentionTextDecoder,
    MemoryTrunkTextDecoder,
    RotarySelfAttention,
    SubPixelSequenceUpsampler,
    TextVQVAE,
    TextVQVAEConfig,
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


if __name__ == "__main__":
    unittest.main()
