import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from common import ROOT
import training.run_research_pipeline as research_pipeline
from training.run_research_pipeline import (
    ExperimentJob,
    _dynamic_corpus_experiments,
    _dynamic_nanogpt_experiments,
    _gpu_slots,
    _launch_logged_job,
    _parse_gpu_ids,
    _preflight_gpus,
    _run_parallel_jobs,
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

    for stage in ("topk", "gqvae", "commitment_beta"):
        config_path = ROOT / definition["configs"][stage]
        experiments = json.loads(config_path.read_text(encoding="utf-8"))["experiments"]
        assert all(experiment["epochs"] == 10 for experiment in experiments)
        assert all(
            experiment["max-train-samples"] == 50000
            for experiment in experiments
        )
        assert all(
            not experiment.get("full-train-data", False)
            for experiment in experiments
        )
        assert all(
            experiment.get("continuous-truncation", False) is False
            for experiment in experiments
        )


def test_gpu_pool_cli_parsing_and_slot_expansion():
    assert _parse_gpu_ids("0, 1,2") == ("0", "1", "2")
    assert _gpu_slots(("0", "1", "2"), 2) == ("0", "1", "2", "0", "1", "2")

    with pytest.raises(argparse.ArgumentTypeError, match="duplicate"):
        _parse_gpu_ids("0,1,1")
    with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
        _parse_gpu_ids("0,gpu2")
    with pytest.raises(ValueError, match="positive"):
        _gpu_slots(("0",), 0)


def _synthetic_jobs(tmp_path: Path, count: int) -> list[ExperimentJob]:
    return [
        ExperimentJob(
            stage="topk" if index < 2 else "gqvae",
            module="training.run_text_vqvae_experiment",
            parameters={"ablation": f"job-{index}"},
            run_name=f"job-{index}",
            target=tmp_path / f"job-{index}",
            command=["python", "-m", "training.fake", "--run-name", f"job-{index}"],
        )
        for index in range(count)
    ]


def test_parallel_dry_run_assigns_jobs_round_robin_without_writes(tmp_path, capsys):
    state = {"stages": {}}
    stages = _run_parallel_jobs(
        _synthetic_jobs(tmp_path, 5),
        ("0", "1", "2"),
        1,
        tmp_path / "pipeline",
        state,
        tmp_path / "pipeline" / "state.json",
        dry_run=True,
    )

    records = [*stages["topk"], *stages["gqvae"]]
    assert [record["gpu"] for record in records] == ["0", "1", "2", "0", "1"]
    assert all(record["status"] == "planned" for record in records)
    assert not (tmp_path / "pipeline").exists()
    assert "CUDA_VISIBLE_DEVICES=2" in capsys.readouterr().out


def test_parallel_pool_refills_each_gpu_and_persists_completion(tmp_path, monkeypatch):
    jobs = _synthetic_jobs(tmp_path, 7)
    completed: set[Path] = set()
    active_by_gpu = {"0": 0, "1": 0, "2": 0}
    max_by_gpu = {"0": 0, "1": 0, "2": 0}
    used_gpus: set[str] = set()
    lock = threading.Lock()

    monkeypatch.setattr(
        research_pipeline,
        "_experiment_completed",
        lambda _module, target: target in completed,
    )

    def fake_launcher(job, gpu, _log_path, on_started):
        on_started(10_000 + int(job.run_name.rsplit("-", 1)[1]))
        with lock:
            active_by_gpu[gpu] += 1
            max_by_gpu[gpu] = max(max_by_gpu[gpu], active_by_gpu[gpu])
            used_gpus.add(gpu)
        time.sleep(0.01)
        completed.add(job.target)
        with lock:
            active_by_gpu[gpu] -= 1

    state = {"status": "running", "stages": {}}
    state_path = tmp_path / "pipeline" / "state.json"
    stages = _run_parallel_jobs(
        jobs,
        ("0", "1", "2"),
        1,
        tmp_path / "pipeline",
        state,
        state_path,
        dry_run=False,
        launcher=fake_launcher,
    )

    records = [record for values in stages.values() for record in values]
    assert used_gpus == {"0", "1", "2"}
    assert max_by_gpu == {"0": 1, "1": 1, "2": 1}
    assert all(record["status"] == "completed" for record in records)
    assert all("pid" in record and "started_at" in record for record in records)
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert sum(len(values) for values in persisted["stages"].values()) == 7


def test_parallel_pool_spreads_short_stage_before_reusing_gpu(tmp_path, monkeypatch):
    jobs = _synthetic_jobs(tmp_path, 3)
    completed: set[Path] = set()
    assignments: dict[str, str] = {}
    lock = threading.Lock()

    monkeypatch.setattr(
        research_pipeline,
        "_experiment_completed",
        lambda _module, target: target in completed,
    )

    def fake_launcher(job, gpu, _log_path, on_started):
        on_started(20_000 + int(job.run_name.rsplit("-", 1)[1]))
        with lock:
            assignments[job.run_name] = gpu
            completed.add(job.target)

    _run_parallel_jobs(
        jobs,
        ("0", "1", "2"),
        2,
        tmp_path / "pipeline",
        {"status": "running", "stages": {}},
        tmp_path / "pipeline" / "state.json",
        dry_run=False,
        launcher=fake_launcher,
    )

    assert assignments == {"job-0": "0", "job-1": "1", "job-2": "2"}


def test_parallel_failure_is_persisted_and_reports_log(tmp_path, monkeypatch):
    job = _synthetic_jobs(tmp_path, 1)[0]
    monkeypatch.setattr(
        research_pipeline,
        "_experiment_completed",
        lambda _module, _target: False,
    )

    def failing_launcher(_job, _gpu, _log_path, on_started):
        on_started(999)
        raise subprocess.CalledProcessError(7, _job.command)

    state = {"status": "running", "stages": {}}
    state_path = tmp_path / "pipeline" / "state.json"
    with pytest.raises(RuntimeError, match="Parallel job failed") as error:
        _run_parallel_jobs(
            [job],
            ("0",),
            1,
            tmp_path / "pipeline",
            state,
            state_path,
            dry_run=False,
            launcher=failing_launcher,
        )

    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    record = persisted["stages"][job.stage][0]
    assert record["status"] == "failed"
    assert record["gpu"] == "0"
    assert record["pid"] == 999
    assert record["log"] in str(error.value)


def test_logged_job_pins_one_visible_gpu_and_unbuffers_output(tmp_path, monkeypatch):
    captured = {}

    class FakeProcess:
        pid = 4321

        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured.update(kwargs)

        def wait(self):
            return 0

    monkeypatch.setattr(research_pipeline.subprocess, "Popen", FakeProcess)
    job = _synthetic_jobs(tmp_path, 1)[0]
    pids = []
    _launch_logged_job(job, "2", tmp_path / "job.log", pids.append)

    assert pids == [4321]
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert captured["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert captured["env"]["PYTHONUNBUFFERED"] == "1"
    assert captured["stderr"] == research_pipeline.subprocess.STDOUT


def test_gpu_preflight_checks_each_gpu_with_the_current_interpreter(monkeypatch):
    calls = []

    class Result:
        returncode = 0
        stdout = "torch=2.6.0+cu118 cuda=11.8 available=True count=1 device=RTX 3090"
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Result()

    monkeypatch.setattr(research_pipeline.subprocess, "run", fake_run)
    _preflight_gpus(("0", "2"))

    assert [call[1]["env"]["CUDA_VISIBLE_DEVICES"] for call in calls] == ["0", "2"]
    assert all(call[0][0] == research_pipeline.sys.executable for call in calls)
    assert all(call[1]["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" for call in calls)


def test_gpu_preflight_rejects_cpu_fallback(monkeypatch):
    class Result:
        returncode = 1
        stdout = "torch=2.11.0 cuda=13.0 available=False count=0 device=None"
        stderr = ""

    monkeypatch.setattr(
        research_pipeline.subprocess,
        "run",
        lambda *_args, **_kwargs: Result(),
    )

    with pytest.raises(RuntimeError, match="CUDA preflight failed for GPU 1"):
        _preflight_gpus(("1",))


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
    assert "vq-k8192-nearest-adaptive-n50k-e10__20990102" in by_tokenizer["vqvae"][
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
