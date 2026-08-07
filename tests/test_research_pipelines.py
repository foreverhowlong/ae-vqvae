import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from common.gqvae_config import GQVAEConfig
from common.learned_tokenizer import LearnedByteFallbackTokenizer
from common.text_vqvae_config import CollapseControlConfig
from models.gqvae import GQVAE
from models.nanogpt import NanoGPT, NanoGPTConfig
from models.text_vqvae import VectorQuantizer
from training.run_nanogpt_experiment import NanoGPTTrainConfig, evaluate_bpb
from training.text_vqvae.config import add_arguments, build_configs


def test_topk_quantizer_anneals_8_4_2_1_and_eval_is_hard():
    quantizer = VectorQuantizer(
        16,
        4,
        CollapseControlConfig(
            use_ema_codebook=False,
            quantizer_mode="topk",
            topk_start=8,
            topk_hard_fraction=0.4,
            topk_temperature_start=1.0,
            topk_temperature_end=0.1,
        ),
    )
    latents = torch.randn(2, 3, 4, requires_grad=True)
    observed = []
    for progress in (0.0, 0.21, 0.41, 0.61):
        outputs = quantizer(latents, curriculum_progress=progress)
        observed.append(outputs["quantizer_topk"])
    assert observed == [8, 4, 2, 1]
    outputs = quantizer(latents, curriculum_progress=0.0)
    outputs["z_q_st"].square().mean().backward()
    assert latents.grad is not None
    assert torch.count_nonzero(latents.grad) > 0

    quantizer.eval()
    hard = quantizer(latents.detach(), curriculum_progress=0.0)
    assert hard["quantizer_topk"] == 1
    expected = quantizer.codebook(hard["indices"])
    torch.testing.assert_close(hard["z_q_raw"], expected)


def test_topk_cli_is_opt_in_and_full_data_removes_sample_cap():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([
        "--quantizer-mode",
        "topk",
        "--topk-start",
        "8",
        "--full-train-data",
        "--ae-warmup-mode",
        "adaptive",
        "--ae-warmup-max-steps",
        "6000",
    ])
    train, data, model, collapse = build_configs(
        args,
        SimpleNamespace(vocab_size=8192, pad_token_id=0),
    )
    assert train.ae_warmup_mode == "adaptive"
    assert data.max_train_samples is None
    assert model.codebook_size == 3072
    assert collapse.quantizer_mode == "topk"
    assert collapse.topk_start == 8


def test_learned_tokenizer_is_greedy_and_lossless_with_byte_fallback():
    tokenizer = LearnedByteFallbackTokenizer([b"ab", b"a"] + [b""] * 8190)
    raw = b"ab az\x00"
    ids = tokenizer.encode_bytes(raw)
    assert ids[0] == tokenizer.bos_token_id
    assert ids[1] == 0
    assert tokenizer.decode_bytes(ids) == raw
    with TemporaryDirectory() as directory:
        path = Path(directory) / "tokenizer.json"
        tokenizer.save(path)
        restored = LearnedByteFallbackTokenizer.load(path)
        assert restored.decode_bytes(restored.encode_bytes(raw)) == raw


def test_gqvae_forward_has_gate_length_and_vq_losses():
    model = GQVAE(GQVAEConfig(
        max_seq_len=8,
        d_model=16,
        code_dim=8,
        n_heads=4,
        encoder_layers=1,
        gater_layers=1,
        ffn_mult=2,
        decode_width=4,
    ))
    inputs = torch.randint(0, 256, (2, 8))
    mask = torch.ones_like(inputs)
    outputs = model(inputs, mask, compression_weight=2.0)
    assert outputs["byte_logits"].shape == (2, 8, 4, 258)
    assert outputs["length_logits"].shape == (2, 8, 4)
    assert outputs["gates"].shape == (2, 8)
    assert outputs["quantizer_active"] is True
    outputs["loss"].backward()
    assert math.isfinite(float(outputs["loss"].detach()))


def test_nanogpt_validation_bpb_uses_summed_nll_and_raw_bytes():
    with TemporaryDirectory() as directory:
        data_dir = Path(directory)
        # Two documents with BOS/content/EOS. A zeroed model predicts a uniform
        # distribution over eight tokens for five scored next-token positions.
        np.asarray([6, 1, 2, 7, 6, 3, 7], dtype=np.uint16).tofile(
            data_dir / "validation.bin"
        )
        np.asarray([0, 4, 7], dtype=np.uint64).tofile(
            data_dir / "validation.idx"
        )
        np.asarray([2, 1], dtype=np.uint64).tofile(
            data_dir / "validation.bytes"
        )
        model = NanoGPT(NanoGPTConfig(
            block_size=4,
            vocab_size=8,
            n_layer=1,
            n_head=2,
            n_embd=8,
        ))
        for parameter in model.parameters():
            parameter.data.zero_()
        metadata = {"pad_token_id": 0}
        metrics = evaluate_bpb(
            model,
            data_dir,
            metadata,
            NanoGPTTrainConfig(
                eval_stride=2,
                eval_batch_size=2,
                eval_max_documents=None,
            ),
            torch.device("cpu"),
        )
        assert metrics["predicted_tokens"] == 5
        assert metrics["raw_utf8_bytes"] == 3
        assert math.isclose(metrics["bits_per_raw_byte"], 5.0, rel_tol=1e-6)


def test_nanogpt_18m_shape_stays_within_declared_parameter_tolerance():
    model = NanoGPT(NanoGPTConfig(vocab_size=8192))
    error = abs(model.count_parameters() - 18_000_000) / 18_000_000
    assert error < 0.05

