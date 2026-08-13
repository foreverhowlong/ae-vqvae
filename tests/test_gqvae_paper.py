import math

import torch

from common.gqvae_paper_config import (
    GQVAEPaperModelConfig,
    GateBistabilityConfig,
)
from models.gqvae_paper import GQVAEPaper
from training.gqvae_paper import gate_statistics, preprocess_ascii_pieces


def small_config(**overrides):
    values = {
        "embedding_dim": 16,
        "codebook_dim": 16,
        "codebook_size": 32,
        "encoder_depth": 1,
        "gater_depth": 1,
        "decoder_depth": 1,
        "attention_head_dim": 8,
        "decode_width": 4,
        "quantizer_reservoir_size": 64,
        "quantizer_warmup_steps": 2,
        "quantizer_resample_every": 2,
    }
    values.update(overrides)
    return GQVAEPaperModelConfig(**values)


def test_paper_preprocessing_joins_regexes_filters_and_pads_ascii():
    rows = preprocess_ascii_pieces(["Hi", " there!", " café"], input_len=16)
    assert rows.ndim == 2
    assert rows.shape[1] == 16
    decoded = [bytes(row[row != 0].tolist()).decode("ascii") for row in rows]
    assert "Hi" in decoded
    assert " there" in decoded
    assert all(len(value) <= 16 for value in decoded)


def test_paper_model_matches_decoder_and_loss_contract_during_vq_warmup():
    model = GQVAEPaper(small_config())
    input_ids = torch.randint(1, 128, (2, 16))
    output = model(input_ids, step=0)
    assert output.byte_logits.shape == (2, 128, 16, 4)
    assert output.reconstruction_masks.shape == (2, 16, 4)
    assert output.gates.shape == (2, 16)
    assert output.quantizer_active is False
    assert output.codebook_loss is None
    output.loss.backward()
    assert math.isfinite(float(output.loss.detach()))


def test_paper_model_activates_quantizer_after_500_step_equivalent():
    model = GQVAEPaper(small_config(quantizer_warmup_steps=1))
    input_ids = torch.randint(1, 128, (4, 16))
    model(input_ids, step=0)
    output = model(input_ids, step=1)
    assert output.quantizer_active is True
    assert output.code_indices is not None
    assert output.commitment_loss is not None


def test_read_only_diagnostic_forward_preserves_quantizer_warmup_state():
    model = GQVAEPaper(small_config())
    input_ids = torch.randint(1, 128, (2, 16))
    before = model.quantizer.reservoir.seen
    output = model(input_ids, step=0, update_quantizer_state=False)
    assert output.quantizer_active is False
    assert model.quantizer.reservoir.seen == before


def test_gate_bistability_classification_separates_collapses_and_polarization():
    config = GateBistabilityConfig()
    assert gate_statistics(torch.full((2, 16), 0.01), config)["gate_state"] == "collapsed_zero"
    assert gate_statistics(torch.full((2, 16), 0.99), config)["gate_state"] == "collapsed_one"
    polarized = torch.tensor([[0.01] * 8 + [0.99] * 8])
    assert gate_statistics(polarized, config)["gate_state"] == "polarized"
