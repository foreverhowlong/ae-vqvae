import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common.end_to_end_tokenizer_config import EndToEndTokenizerConfig
from common.end_to_end_tokenizer_data import _is_word_boundary
from models.end_to_end_tokenizer import (
    EndToEndTokenizerModel,
    end_to_end_tokenizer_losses,
)
from models.segmental_vqvae import GreedySpanSegmenter
from training.run_experiment_sequence import load_config
from training.run_end_to_end_tokenizer_experiment import (
    _set_prior_trainable,
    prior_objective_weight,
)


def _config(**overrides) -> EndToEndTokenizerConfig:
    values = {
        "vocab_size": 32,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "max_seq_len": 8,
        "segmenter_d_model": 16,
        "segmenter_n_heads": 4,
        "boundary_encoder_layers": 1,
        "boundary_window_radius": 2,
        "max_span_length": 4,
        "span_encoder_layers": 1,
        "segmenter_ffn_mult": 2,
        "latent_dim": 4,
        "codebook_size": 16,
        "code_target_topk": 4,
        "prior_layers": 1,
        "prior_heads": 4,
        "prior_d_model": 16,
        "text_decoder_layers": 1,
        "text_decoder_heads": 4,
        "text_decoder_d_model": 16,
        "dropout": 0.0,
        "prior_dropout": 0.0,
        "text_decoder_dropout": 0.0,
    }
    values.update(overrides)
    return EndToEndTokenizerConfig(**values)


def _batch():
    input_ids = torch.tensor([
        [3, 4, 5, 2, 0, 0, 0, 0],
        [6, 7, 8, 9, 10, 11, 12, 2],
    ])
    attention_mask = input_ids.ne(0)
    legal_endpoints = torch.zeros_like(attention_mask)
    legal_endpoints[0, [1, 3]] = True
    legal_endpoints[1, [1, 3, 5, 7]] = True
    return input_ids, attention_mask, legal_endpoints


def test_word_boundary_only_rejects_boundaries_inside_words():
    text = "small fox"
    assert not _is_word_boundary(text, 2)
    assert _is_word_boundary(text, 5)
    assert _is_word_boundary(text, len(text))


def test_greedy_endpoint_mask_keeps_partition_legal():
    torch.manual_seed(3)
    model = EndToEndTokenizerModel(_config())
    input_ids, attention_mask, legal_endpoints = _batch()
    outputs = model(input_ids, attention_mask, legal_endpoints)
    assert not (outputs["hard_boundaries"] & ~legal_endpoints).any()
    assert torch.equal(
        outputs["hard_boundaries"].sum(dim=1),
        outputs["chunk_counts"],
    )


def test_endpoint_mask_has_bounded_fallback_for_long_words():
    model = EndToEndTokenizerModel(_config(max_span_length=2))
    segmenter = model.segmenter
    assert isinstance(segmenter, GreedySpanSegmenter)
    scores = torch.tensor([[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, -torch.inf]]])
    legal = torch.tensor([[False, False, False, True]])
    masked = segmenter._mask_illegal_endpoints(scores, legal)
    boundaries, *_ = segmenter._greedy_boundaries(masked, torch.tensor([4]))
    assert boundaries.tolist() == [[False, True, False, True]]


def test_hard_codelength_has_soft_gradients_for_segmenter_and_encoder():
    torch.manual_seed(5)
    model = EndToEndTokenizerModel(_config()).train()
    input_ids, attention_mask, legal_endpoints = _batch()
    outputs = model(input_ids, attention_mask, legal_endpoints)
    losses = end_to_end_tokenizer_losses(
        outputs,
        input_ids,
        attention_mask,
        model,
    )
    chunks = outputs["latent_mask"].bool()
    hard_length = F.cross_entropy(
        outputs["length_logits"][chunks],
        (outputs["chunk_lengths"] - 1)[chunks],
        reduction="sum",
    )
    hard_code = F.cross_entropy(
        outputs["code_logits"][chunks],
        outputs["indices"][chunks],
        reduction="sum",
    )
    torch.testing.assert_close(losses["length_nll_sum"], hard_length)
    torch.testing.assert_close(losses["code_nll_sum"], hard_code)
    torch.testing.assert_close(
        losses["generative_nll_sum"],
        losses["length_nll_sum"]
        + losses["code_nll_sum"]
        + losses["text_nll_sum"],
    )
    losses["loss"].backward()
    scorer_gradient = model.segmenter.scorer.mlp[-1].weight.grad
    encoder_gradient = model.latent_projection.weight.grad
    prior_gradient = model.chunk_prior.length_head.weight.grad
    for gradient in (scorer_gradient, encoder_gradient, prior_gradient):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_rate_dual_uses_observed_hard_chunk_rate():
    model = EndToEndTokenizerModel(_config(rate_dual_lr=0.5))
    target_rate = 1.0 / model.config.compression_target
    increased = model.update_rate_dual(target_rate + 0.2)
    assert increased > 0
    decreased = model.update_rate_dual(target_rate - 0.4)
    assert decreased < increased


def test_vq_warmup_schedule_delays_and_then_anneals_prior_objective():
    weights = [
        prior_objective_weight(
            step,
            vq_warmup_steps=3,
            prior_anneal_steps=2,
        )
        for step in range(1, 7)
    ]
    assert weights == [0.0, 0.0, 0.0, 0.5, 1.0, 1.0]


def test_vq_warmup_trains_reconstruction_path_without_prior_parameters():
    torch.manual_seed(13)
    model = EndToEndTokenizerModel(_config()).train()
    _set_prior_trainable(model, False)
    input_ids, attention_mask, legal_endpoints = _batch()
    outputs = model(input_ids, attention_mask, legal_endpoints)
    losses = end_to_end_tokenizer_losses(
        outputs,
        input_ids,
        attention_mask,
        model,
        prior_weight=0.0,
    )
    expected = (
        losses["text_nll_per_bpe"]
        + losses["commitment_weighted_loss"]
        + losses["rate_constraint_loss"]
    )
    torch.testing.assert_close(losses["loss"], expected)
    losses["loss"].backward()
    assert all(parameter.grad is None for parameter in model.chunk_prior.parameters())
    gradient = model.latent_projection.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0


def test_end_to_end_experiment_config_is_single_runnable_experiment():
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "end-to-end-greedy-vq-k8192-18m.json"
    )
    module, experiments = load_config(path)
    assert module == "training.run_end_to_end_tokenizer_experiment"
    assert len(experiments) == 1
    experiment = experiments[0]
    assert experiment["codebook-size"] == 8192
    assert experiment["target-prior-parameters"] == 18_000_000
    assert experiment["continuous-truncation"] is False
    assert experiment["word-boundary-only"] is True
    json.loads(path.read_text(encoding="utf-8"))


def test_vq_warmup_config_preserves_original_experiment_shape():
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "end-to-end-greedy-vq-k8192-18m-vq-warmup.json"
    )
    module, experiments = load_config(path)
    assert module == "training.run_end_to_end_tokenizer_experiment"
    assert len(experiments) == 1
    experiment = experiments[0]
    assert experiment["vq-warmup-steps"] == 3000
    assert experiment["prior-anneal-steps"] == 2000
    assert experiment["max-train-samples"] == 50000
    assert experiment["continuous-truncation"] is False
