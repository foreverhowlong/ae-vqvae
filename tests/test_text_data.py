import argparse
from types import SimpleNamespace
import unittest

import torch

from common.text_data import BYTE_EOS, BYTE_PAD, ByteTokenizer, TextDataset
from common.text_vqvae_config import DataConfig
from training.text_vqvae.config import add_arguments, build_configs
from training.text_vqvae.loop import split_dataset


class ContinuousTruncationTest(unittest.TestCase):
    def test_default_mode_preserves_single_truncated_sample(self):
        dataset = TextDataset(["a" * 256], max_seq_len=64, tokenizer=ByteTokenizer())

        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        self.assertEqual(sample["input_ids"].shape, (64,))
        self.assertEqual(sample["input_ids"][-1].item(), BYTE_EOS)
        self.assertTrue(torch.all(sample["attention_mask"] == 1))

    def test_continuous_mode_splits_exact_multiple_without_extra_sample(self):
        dataset = TextDataset(
            ["a" * 256],
            max_seq_len=64,
            tokenizer=ByteTokenizer(),
            continuous_truncation=True,
        )

        self.assertEqual(len(dataset), 4)
        for sample in dataset:
            self.assertEqual(sample["input_ids"].tolist(), [ord("a")] * 64)
            self.assertEqual(sample["attention_mask"].tolist(), [1] * 64)

    def test_continuous_mode_keeps_and_pads_final_remainder(self):
        dataset = TextDataset(
            ["a" * 65],
            max_seq_len=64,
            tokenizer=ByteTokenizer(),
            continuous_truncation=True,
        )

        self.assertEqual(len(dataset), 2)
        final = dataset[1]
        self.assertEqual(final["input_ids"][:2].tolist(), [ord("a"), BYTE_EOS])
        self.assertEqual(final["input_ids"][2:].tolist(), [BYTE_PAD] * 62)
        self.assertEqual(final["attention_mask"][:2].tolist(), [1, 1])
        self.assertEqual(final["attention_mask"][2:].tolist(), [0] * 62)

    def test_cli_flag_is_opt_in_and_resolves_into_data_config(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        tokenizer = SimpleNamespace(vocab_size=258, pad_token_id=BYTE_PAD)

        default_data = build_configs(parser.parse_args([]), tokenizer)[1]
        enabled_data = build_configs(
            parser.parse_args(["--continuous-truncation"]), tokenizer
        )[1]

        self.assertEqual(default_data, DataConfig())
        self.assertFalse(default_data.continuous_truncation)
        self.assertTrue(enabled_data.continuous_truncation)

    def test_split_keeps_chunks_from_each_source_text_together(self):
        dataset = TextDataset(
            ["a" * 130, "b" * 130, "c" * 130],
            max_seq_len=64,
            tokenizer=ByteTokenizer(),
            continuous_truncation=True,
        )

        train, val = split_dataset(
            dataset,
            val_fraction=1 / 3,
            seed=42,
            max_eval_samples=10,
        )
        train_sources = {
            dataset.sample_text_indices[index] for index in train.indices
        }
        val_sources = {
            dataset.sample_text_indices[index] for index in val.indices
        }

        self.assertFalse(train_sources & val_sources)
        self.assertEqual(train_sources | val_sources, {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
