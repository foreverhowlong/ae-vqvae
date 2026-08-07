import json
from pathlib import Path

import numpy as np

from common import ROOT
from training.run_research_pipeline import (
    _dynamic_corpus_experiments,
    _dynamic_nanogpt_experiments,
    load_pipeline_definition,
    select_gqvae,
    select_pareto_knee,
    validate_locked_experiments,
)
from common.learned_tokenizer import LearnedByteFallbackTokenizer
from visualization.render_research_pipeline import render_pipeline


MASTER_CONFIG = ROOT / "configs" / "full-research-pipeline-k8192-18m-20260807.json"


def test_master_pipeline_locks_8192_and_contains_all_17_experiments():
    definition = load_pipeline_definition(MASTER_CONFIG)

    assert definition["codebook_size"] == 8192
    assert validate_locked_experiments(definition) == 17


def test_pareto_knee_rejects_dominated_point_and_selects_balanced_frontier_point():
    candidates = [
        {"ablation": "reconstruction", "reconstruction_loss": 1.0, "bytes_per_token": 1.0},
        {"ablation": "knee", "reconstruction_loss": 1.3, "bytes_per_token": 2.5},
        {"ablation": "compression", "reconstruction_loss": 2.0, "bytes_per_token": 3.0},
        {"ablation": "dominated", "reconstruction_loss": 1.5, "bytes_per_token": 2.0},
    ]

    selected, frontier = select_pareto_knee(candidates)

    assert selected["ablation"] == "knee"
    assert frontier == ["reconstruction", "knee", "compression"]
    assert selected["selection_score"] < 0.4


def test_explicit_gqvae_selection_overrides_pareto_rule():
    candidates = [
        {"ablation": "alpha1", "reconstruction_loss": 1.0, "bytes_per_token": 1.0},
        {"ablation": "alpha4", "reconstruction_loss": 2.0, "bytes_per_token": 4.0},
    ]

    selected, manifest = select_gqvae(candidates, "alpha4")

    assert selected["ablation"] == "alpha4"
    assert manifest["method"] == "explicit_ablation"


def test_downstream_paths_are_date_scoped_and_generated_from_selected_artifacts():
    definition = load_pipeline_definition(MASTER_CONFIG)
    bpe_path = Path("/tmp/tokenizers/bpe.json")
    gqvae_path = Path("/tmp/tokenizers/gqvae.json")

    corpus_experiments, corpora = _dynamic_corpus_experiments(
        definition,
        "20990102",
        bpe_path,
        gqvae_path,
    )
    by_tokenizer = {item["tokenizer"]: item for item in corpus_experiments}

    assert by_tokenizer["bpe"]["tokenizer-path"] == str(bpe_path)
    assert by_tokenizer["gqvae"]["tokenizer-path"] == str(gqvae_path)
    assert by_tokenizer["vqvae"]["device"] == "auto"
    assert "vq-k8192-nearest-adaptive-full__20990102" in by_tokenizer["vqvae"][
        "vq-checkpoint"
    ]
    assert all("full-k8192-18m__20990102" in str(path) for path in corpora.values())

    nano = _dynamic_nanogpt_experiments(definition, corpora)
    assert {Path(item["data-dir"]).name for item in nano} == {
        "bpe8k",
        "gqvae-k8192",
        "vqvae-k8192",
    }


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_complete_pipeline_renderer_emits_supported_paper_and_sweep_figures(tmp_path):
    pipeline_dir = tmp_path / "pipeline"
    stages = {name: [] for name in ("topk", "gqvae", "commitment_beta", "lm_corpus", "nanogpt")}

    common_eval = {
        "recon_nll": 1.2,
        "token_accuracy": 0.75,
        "codebook_utilization": 0.4,
        "bits_per_token": 3.1,
        "commitment_loss": 0.2,
    }
    for name, mode in (
        ("vq-k8192-nearest-adaptive-full__20990102", "nearest"),
        ("vq-k8192-topk8-to-1-adaptive-full__20990102", "topk"),
    ):
        target = tmp_path / name
        train = [{
            "split": "train",
            "step": step,
            "quantizer_topk": 8 if mode == "topk" and step == 1 else 1,
            "quantizer_temperature": 1.0 if step == 1 else 0.1,
            "quantizer_mixture_entropy": 1.5 if mode == "topk" and step == 1 else 0.0,
            "quantizer_effective_k": 4.5 if mode == "topk" and step == 1 else 1.0,
        } for step in (1, 2)]
        evaluation = [
            {"split": "eval", "step": 1, **common_eval},
            {"split": "eval", "step": 2, **{**common_eval, "recon_nll": 1.0}},
        ]
        _write_jsonl(target / "metrics.jsonl", [*train, *evaluation])
        _write_json(target / "summary.json", {"final_eval": common_eval})
        stages["topk"].append({"target": str(target), "run_name": name, "status": "completed"})

    candidates = []
    for index, alpha in enumerate((1.0, 2.0, 3.0, 4.0), start=1):
        name = f"gqvae-k8192-alpha{index}__20990102"
        target = tmp_path / name
        gq_eval = {
            "loss": 2.0,
            "reconstruction_loss": 1.0 + 0.1 * index,
            "compression_loss": 0.2,
            "compression_gate_mean": 0.3,
            "length_loss": 0.1,
            "commitment_loss": 0.05,
            "byte_accuracy": 0.9,
            "exact_gated_token_accuracy": 0.8,
            "bytes_per_token": 1.5 + 0.4 * index,
            "selected_tokens": 10,
            "valid_bytes": 20,
        }
        _write_jsonl(target / "metrics.jsonl", [
            {"split": "train", "step": 1, "compression_weight": 0.0},
            {"split": "train", "step": 2, "compression_weight": alpha},
            {"split": "eval", "step": 1, **gq_eval},
            {"split": "eval", "step": 2, **gq_eval},
        ])
        _write_json(target / "config.json", {"model": {"compression_weight": alpha}})
        _write_json(target / "summary.json", {"final_eval": gq_eval})
        stages["gqvae"].append({"target": str(target), "run_name": name, "status": "completed"})
        candidates.append({
            "ablation": f"gqvae-k8192-alpha{index}",
            "run_name": name,
            "checkpoint": str(target / "checkpoints" / "best.pt"),
            "best_step": 2,
            "reconstruction_loss": gq_eval["reconstruction_loss"],
            "bytes_per_token": gq_eval["bytes_per_token"],
            "byte_accuracy": 0.9,
            "exact_gated_token_accuracy": 0.8,
        })
    _write_json(pipeline_dir / "gqvae_selection.json", {
        "method": "pareto_knee",
        "pareto_frontier": [item["ablation"] for item in candidates],
        "selected": candidates[1],
        "candidates": candidates,
    })
    LearnedByteFallbackTokenizer([b"a", b"the", b"story", b""]).save(
        pipeline_dir / "gqvae_tokenizer.json"
    )

    for index, beta in enumerate((0.05, 0.1, 0.25, 0.5, 1.0)):
        name = f"vq-k8192-beta{index}__20990102"
        target = tmp_path / name
        _write_json(target / "config.json", {"model": {"commitment_beta": beta}})
        _write_json(target / "summary.json", {"final_eval": common_eval})
        stages["commitment_beta"].append({"target": str(target), "run_name": name, "status": "completed"})

    corpus_specs = {
        "bpe": (16, 13, 14, 15, [1, 2, 3, 3, 4, 5]),
        "gqvae": (263, 260, 261, 262, [260, 0, 1, 1, 2, 261]),
        "vqvae": (16, 13, 14, 15, [13, 3, 4, 4, 5, 14]),
    }
    for tokenizer, (vocab, bos, eos, pad, token_ids) in corpus_specs.items():
        target = tmp_path / f"corpus-{tokenizer}"
        target.mkdir(parents=True)
        np.asarray(token_ids, dtype=np.uint16).tofile(target / "train.bin")
        _write_json(target / "meta.json", {
            "tokenizer": tokenizer,
            "vocab_size": vocab,
            "bos_token_id": bos,
            "eos_token_id": eos,
            "pad_token_id": pad,
            "train": {
                "documents": 2,
                "raw_utf8_bytes": 20,
                "tokens": len(token_ids),
                "bytes_per_token": 20 / len(token_ids),
                "fallback_bytes": 2 if tokenizer != "bpe" else 0,
                "fallback_byte_fraction": 0.1 if tokenizer != "bpe" else 0.0,
                "vq_documents": 1,
            },
        })
        stages["lm_corpus"].append({"target": str(target), "run_name": tokenizer, "status": "completed"})

    for tokenizer in corpus_specs:
        name = f"nanogpt18m-{tokenizer}__20990102"
        target = tmp_path / name
        _write_json(target / "config.json", {"corpus": {"tokenizer": tokenizer}})
        final = {"bits_per_raw_byte": 2.0 + 0.1 * len(tokenizer)}
        _write_jsonl(target / "metrics.jsonl", [
            {"split": "validation", "step": 1, "bits_per_raw_byte": final["bits_per_raw_byte"] + 0.2}
        ])
        _write_json(target / "summary.json", {
            "parameter_count": 18_000_000,
            "best_validation_bpb": final["bits_per_raw_byte"],
            "steps": 2,
            "elapsed_sec": 10.0,
            "final_validation": final,
        })
        stages["nanogpt"].append({"target": str(target), "run_name": name, "status": "completed"})

    _write_json(pipeline_dir / "state.json", {
        "pipeline": "synthetic",
        "run_date": "20990102",
        "config": str(MASTER_CONFIG),
        "stages": stages,
    })

    manifest = render_pipeline(pipeline_dir)

    assert manifest["status"] == "completed"
    assert manifest["paper_figures"]["figure_4"] is None
    assert manifest["paper_figures"]["figure_6"] is None
    assert len(list((pipeline_dir / "plots").glob("*.png"))) == 12
    assert (pipeline_dir / "plots" / "results_table.csv").is_file()
