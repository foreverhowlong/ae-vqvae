"""Run the complete K=8192 tokenizer-research pipeline with one command.

The orchestrator reuses the five focused experiment configs, supplies dynamic
artifact paths between stages, and skips stages that already have a completed
output. It intentionally does not delete or silently resume incomplete model
directories because the underlying trainers do not support checkpoint resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from common import ROOT
from training.run_experiment_sequence import build_command, load_config, make_run_name
from training.text_vqvae.reporting import atomic_json_dump


TRAIN_OUTPUT_ROOTS = {
    "training.run_text_vqvae_experiment": ROOT / "outputs" / "text_vqvae",
    "training.run_gqvae_experiment": ROOT / "outputs" / "gqvae",
    "training.run_nanogpt_experiment": ROOT / "outputs" / "nanogpt",
}
STAGE_ORDER = ("topk", "gqvae", "commitment_beta")


def _root_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_pipeline_definition(path: str | Path) -> dict[str, Any]:
    config_path = _root_path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("pipeline_version") != 1:
        raise ValueError("One-click pipeline requires pipeline_version=1.")
    if not isinstance(payload.get("name"), str) or not payload["name"]:
        raise ValueError("Pipeline config requires a non-empty name.")
    if payload.get("codebook_size") != 8192:
        raise ValueError("This research pipeline is locked to codebook_size=8192.")
    configs = payload.get("configs")
    required = {*STAGE_ORDER, "lm_corpus", "nanogpt"}
    if not isinstance(configs, dict) or set(configs) != required:
        raise ValueError(f"Pipeline configs must contain exactly {sorted(required)}.")
    for value in configs.values():
        if not _root_path(value).is_file():
            raise FileNotFoundError(f"Referenced experiment config does not exist: {value}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or "bpe_tokenizer" not in artifacts:
        raise ValueError("Pipeline artifacts must define bpe_tokenizer.")
    selection = payload.get("gqvae_selection")
    if not isinstance(selection, dict) or selection.get("method") != "pareto_knee":
        raise ValueError("gqvae_selection.method must be 'pareto_knee'.")
    return payload


def _load_stage(definition: dict[str, Any], stage: str):
    return load_config(_root_path(definition["configs"][stage]))


def validate_locked_experiments(definition: dict[str, Any]) -> int:
    total = 0
    for stage in (*STAGE_ORDER, "lm_corpus", "nanogpt"):
        _, experiments = _load_stage(definition, stage)
        total += len(experiments)
        if stage in STAGE_ORDER:
            for experiment in experiments:
                if experiment.get("codebook-size") != definition["codebook_size"]:
                    raise ValueError(
                        f"{stage} experiment {experiment.get('ablation')!r} is not locked "
                        f"to codebook_size={definition['codebook_size']}."
                    )
                if experiment.get("continuous-truncation") is not False:
                    raise ValueError(
                        f"{stage} experiment {experiment.get('ablation')!r} must explicitly "
                        "set continuous-truncation=false in this matched pipeline."
                    )
    return total


def _pipeline_fingerprint(definition: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(definition, sort_keys=True).encode("utf-8"))
    for stage in (*STAGE_ORDER, "lm_corpus", "nanogpt"):
        digest.update(_root_path(definition["configs"][stage]).read_bytes())
    return digest.hexdigest()


def _completed_summary(run_dir: Path) -> bool:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return summary.get("status") == "completed"


def _experiment_target(module: str, parameters: dict[str, Any], run_name: str) -> Path:
    if module in TRAIN_OUTPUT_ROOTS:
        return TRAIN_OUTPUT_ROOTS[module] / run_name
    if module == "training.prepare_lm_corpus":
        return _root_path(parameters["output-dir"])
    raise ValueError(f"No completion rule is defined for module {module!r}.")


def _experiment_completed(module: str, target: Path) -> bool:
    if module in TRAIN_OUTPUT_ROOTS:
        return (
            _completed_summary(target)
            and (target / "checkpoints" / "best.pt").is_file()
        )
    required = (
        "meta.json",
        "train.bin",
        "train.idx",
        "train.bytes",
        "validation.bin",
        "validation.idx",
        "validation.bytes",
    )
    return all((target / name).is_file() for name in required)


def _run_command(label: str, command: list[str], *, dry_run: bool) -> None:
    print(f"\n[{label}]\n  {shlex.join(command)}")
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def _run_experiment(
    stage: str,
    module: str,
    parameters: dict[str, Any],
    run_date: str,
    *,
    dry_run: bool,
) -> dict[str, str]:
    run_name = make_run_name(parameters, run_date)
    target = _experiment_target(module, parameters, run_name)
    command = build_command(module, parameters, run_name)
    if _experiment_completed(module, target):
        print(f"\n[{stage}] READY, skipping {run_name}\n  {target}")
        return {"run_name": run_name, "target": str(target), "status": "skipped_completed"}
    if not dry_run and target.exists():
        raise RuntimeError(
            f"Incomplete output blocks {stage}/{run_name}: {target}. "
            "Inspect or move that directory, then rerun with the same --run-date."
        )
    _run_command(f"{stage}: {run_name}", command, dry_run=dry_run)
    if not dry_run and not _experiment_completed(module, target):
        raise RuntimeError(f"{stage}/{run_name} exited without complete artifacts: {target}")
    return {"run_name": run_name, "target": str(target), "status": "planned" if dry_run else "completed"}


def _run_stage(
    definition: dict[str, Any],
    stage: str,
    run_date: str,
    *,
    dry_run: bool,
    experiments: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    module, configured = _load_stage(definition, stage)
    selected = configured if experiments is None else experiments
    return [
        _run_experiment(stage, module, dict(parameters), run_date, dry_run=dry_run)
        for parameters in selected
    ]


def _gqvae_eval_at_best(run_dir: Path, best_step: int) -> dict[str, Any]:
    selected = None
    with (run_dir / "metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == "eval" and row.get("step") == best_step:
                selected = row
    if selected is None:
        raise ValueError(f"No GQ-VAE eval row exists at best_step={best_step}: {run_dir}")
    return selected


def load_gqvae_candidates(
    definition: dict[str, Any], run_date: str
) -> list[dict[str, Any]]:
    module, experiments = _load_stage(definition, "gqvae")
    if module != "training.run_gqvae_experiment":
        raise ValueError("The gqvae stage must use training.run_gqvae_experiment.")
    candidates = []
    for parameters in experiments:
        run_name = make_run_name(parameters, run_date)
        run_dir = TRAIN_OUTPUT_ROOTS[module] / run_name
        if not _experiment_completed(module, run_dir):
            raise RuntimeError(f"GQ-VAE candidate is incomplete: {run_dir}")
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        best_step = int(summary["best_step"])
        metrics = _gqvae_eval_at_best(run_dir, best_step)
        candidates.append({
            "ablation": parameters["ablation"],
            "run_name": run_name,
            "checkpoint": str(run_dir / "checkpoints" / "best.pt"),
            "best_step": best_step,
            "reconstruction_loss": float(metrics["reconstruction_loss"]),
            "bytes_per_token": float(metrics["bytes_per_token"]),
            "byte_accuracy": float(metrics["byte_accuracy"]),
            "exact_gated_token_accuracy": float(metrics["exact_gated_token_accuracy"]),
        })
    return candidates


def select_pareto_knee(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Select the frontier point nearest ideal reconstruction and compression."""
    if not candidates:
        raise ValueError("At least one GQ-VAE candidate is required.")
    for candidate in candidates:
        for metric in ("reconstruction_loss", "bytes_per_token"):
            if not math.isfinite(float(candidate[metric])):
                raise ValueError(f"Non-finite {metric} for {candidate.get('ablation')!r}.")
    frontier = []
    for candidate in candidates:
        dominated = any(
            other is not candidate
            and other["reconstruction_loss"] <= candidate["reconstruction_loss"]
            and other["bytes_per_token"] >= candidate["bytes_per_token"]
            and (
                other["reconstruction_loss"] < candidate["reconstruction_loss"]
                or other["bytes_per_token"] > candidate["bytes_per_token"]
            )
            for other in candidates
        )
        if not dominated:
            frontier.append(candidate)
    reconstruction = [float(item["reconstruction_loss"]) for item in frontier]
    compression = [float(item["bytes_per_token"]) for item in frontier]
    recon_min, recon_max = min(reconstruction), max(reconstruction)
    comp_min, comp_max = min(compression), max(compression)

    def score(item: dict[str, Any]) -> tuple[float, float, float, str]:
        recon_cost = (
            (float(item["reconstruction_loss"]) - recon_min) / (recon_max - recon_min)
            if recon_max > recon_min
            else 0.0
        )
        compression_cost = (
            (comp_max - float(item["bytes_per_token"])) / (comp_max - comp_min)
            if comp_max > comp_min
            else 0.0
        )
        return (
            math.hypot(recon_cost, compression_cost),
            float(item["reconstruction_loss"]),
            -float(item["bytes_per_token"]),
            str(item["ablation"]),
        )

    chosen = dict(min(frontier, key=score))
    chosen["selection_score"] = score(chosen)[0]
    return chosen, [str(item["ablation"]) for item in frontier]


def select_gqvae(
    candidates: list[dict[str, Any]], explicit_ablation: str | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    if explicit_ablation is not None:
        matches = [item for item in candidates if item["ablation"] == explicit_ablation]
        if len(matches) != 1:
            available = ", ".join(str(item["ablation"]) for item in candidates)
            raise ValueError(
                f"Unknown --gqvae-ablation {explicit_ablation!r}; available: {available}"
            )
        chosen = dict(matches[0])
        method = "explicit_ablation"
        frontier = None
    else:
        chosen, frontier = select_pareto_knee(candidates)
        method = "pareto_knee"
    return chosen, {
        "method": method,
        "objective": "minimize reconstruction_loss and maximize bytes_per_token",
        "pareto_frontier": frontier,
        "selected": chosen,
        "candidates": candidates,
    }


def _prepare_bpe(
    definition: dict[str, Any], *, dry_run: bool
) -> Path:
    tokenizer_path = _root_path(definition["artifacts"]["bpe_tokenizer"])
    if tokenizer_path.is_file():
        print(f"\n[bpe_tokenizer] READY, skipping\n  {tokenizer_path}")
        return tokenizer_path
    output_dir = tokenizer_path.parent
    if not dry_run and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"BPE tokenizer directory is non-empty but incomplete: {output_dir}"
        )
    command = [
        sys.executable,
        "-m",
        "training.train_tokenizer",
        "--vocab-size",
        str(definition["codebook_size"]),
        "--output-dir",
        str(output_dir),
    ]
    _run_command("bpe_tokenizer", command, dry_run=dry_run)
    if not dry_run and not tokenizer_path.is_file():
        raise RuntimeError(f"BPE tokenizer was not created: {tokenizer_path}")
    return tokenizer_path


def _export_gqvae(
    chosen: dict[str, Any], output: Path, selection_path: Path, selection: dict[str, Any],
    *, dry_run: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        "training.export_gqvae_tokenizer",
        "--checkpoint",
        str(chosen["checkpoint"]),
        "--output",
        str(output),
    ]
    if output.is_file() and selection_path.is_file():
        recorded = json.loads(selection_path.read_text(encoding="utf-8"))
        if recorded.get("selected", {}).get("checkpoint") == chosen["checkpoint"]:
            print(f"\n[gqvae_export] READY, skipping\n  {output}")
            return
    if not dry_run and output.exists():
        raise RuntimeError(f"GQ-VAE tokenizer output conflicts with this selection: {output}")
    _run_command("gqvae_export", command, dry_run=dry_run)
    if not dry_run:
        if not output.is_file():
            raise RuntimeError(f"GQ-VAE tokenizer was not created: {output}")
        atomic_json_dump(selection, selection_path)


def _render_visualizations(pipeline_dir: Path, *, dry_run: bool) -> dict[str, str]:
    manifest_path = pipeline_dir / "plots" / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = {}
        if manifest.get("status") == "completed":
            print(f"\n[visualization] READY, skipping\n  {manifest_path}")
            return {"status": "skipped_completed", "manifest": str(manifest_path)}
    command = [
        sys.executable,
        "-m",
        "visualization.render_research_pipeline",
        "--pipeline-dir",
        str(pipeline_dir),
    ]
    _run_command("visualization", command, dry_run=dry_run)
    if not dry_run and not manifest_path.is_file():
        raise RuntimeError(f"Visualization manifest was not created: {manifest_path}")
    return {
        "status": "planned" if dry_run else "completed",
        "manifest": str(manifest_path),
    }


def _dynamic_corpus_experiments(
    definition: dict[str, Any], run_date: str, bpe_path: Path, gqvae_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    _, configured = _load_stage(definition, "lm_corpus")
    corpus_root = (
        ROOT / "outputs" / "lm_corpora" / f"{definition['name']}__{run_date}"
    )
    topk_module, topk_experiments = _load_stage(definition, "topk")
    reference_label = definition["vq_reference_ablation"]
    references = [item for item in topk_experiments if item.get("ablation") == reference_label]
    if len(references) != 1:
        raise ValueError(f"Expected one VQ reference ablation named {reference_label!r}.")
    reference_name = make_run_name(references[0], run_date)
    reference_dir = TRAIN_OUTPUT_ROOTS[topk_module] / reference_name
    by_tokenizer: dict[str, Path] = {}
    dynamic = []
    for original in configured:
        parameters = dict(original)
        tokenizer = str(parameters["tokenizer"])
        output_dir = corpus_root / Path(str(parameters["output-dir"])).name
        parameters["output-dir"] = str(output_dir)
        if tokenizer == "bpe":
            parameters["tokenizer-path"] = str(bpe_path)
        elif tokenizer == "gqvae":
            parameters["tokenizer-path"] = str(gqvae_path)
        elif tokenizer == "vqvae":
            parameters["vq-checkpoint"] = str(reference_dir / "checkpoints" / "best.pt")
            parameters["vq-config"] = str(reference_dir / "config.json")
            parameters["device"] = definition.get("vq_tokenizer_device", "auto")
        else:
            raise ValueError(f"Unsupported corpus tokenizer {tokenizer!r}.")
        by_tokenizer[tokenizer] = output_dir
        dynamic.append(parameters)
    return dynamic, by_tokenizer


def _dynamic_nanogpt_experiments(
    definition: dict[str, Any], corpora: dict[str, Path]
) -> list[dict[str, Any]]:
    _, configured = _load_stage(definition, "nanogpt")
    dynamic = []
    for original in configured:
        parameters = dict(original)
        ablation = str(parameters.get("ablation", ""))
        matches = [name for name in corpora if name in ablation]
        if len(matches) != 1:
            raise ValueError(f"Cannot map nanoGPT ablation to one corpus: {ablation!r}")
        parameters["data-dir"] = str(corpora[matches[0]])
        dynamic.append(parameters)
    return dynamic


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--run-date",
        default=datetime.now().astimezone().strftime("%Y%m%d"),
        help="Stable YYYYMMDD suffix. Reuse it when restarting an interrupted pipeline.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--gqvae-ablation",
        help="Override Pareto-knee selection with one exact GQ-VAE ablation label.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if len(args.run_date) != 8 or not args.run_date.isdigit():
        raise ValueError("--run-date must use YYYYMMDD format.")
    definition = load_pipeline_definition(args.config)
    experiment_count = validate_locked_experiments(definition)
    pipeline_dir = (
        ROOT / "outputs" / "research_pipeline" / f"{definition['name']}__{args.run_date}"
    )
    selection_path = pipeline_dir / "gqvae_selection.json"
    gqvae_tokenizer = pipeline_dir / "gqvae_tokenizer.json"
    fingerprint = _pipeline_fingerprint(definition)
    state_path = pipeline_dir / "state.json"
    state: dict[str, Any] = {
        "pipeline": definition["name"],
        "run_date": args.run_date,
        "status": "dry_run" if args.dry_run else "running",
        "config": str(_root_path(args.config)),
        "config_fingerprint": fingerprint,
        "configured_experiments": experiment_count,
        "stages": {},
    }
    if not args.dry_run:
        if state_path.is_file():
            previous = json.loads(state_path.read_text(encoding="utf-8"))
            if previous.get("config_fingerprint") != fingerprint:
                raise RuntimeError(
                    f"Pipeline config changed for existing run date: {pipeline_dir}. "
                    "Choose a new --run-date or restore the original configs."
                )
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(state, state_path)

    print(
        f"[Pipeline] {definition['name']} date={args.run_date} "
        f"configured_experiments={experiment_count} dry_run={args.dry_run}"
    )
    try:
        bpe_path = _prepare_bpe(definition, dry_run=args.dry_run)
        for stage in STAGE_ORDER:
            state["current_stage"] = stage
            state["stages"][stage] = _run_stage(
                definition, stage, args.run_date, dry_run=args.dry_run
            )
            if not args.dry_run:
                atomic_json_dump(state, state_path)

        if args.dry_run:
            selection_plan = (
                f"explicit ablation {args.gqvae_ablation!r}"
                if args.gqvae_ablation
                else "Pareto knee of reconstruction_loss vs bytes_per_token"
            )
            print(f"\n[gqvae_selection] after training: {selection_plan}")
            print(f"[gqvae_export] planned output: {gqvae_tokenizer}")
        else:
            candidates = load_gqvae_candidates(definition, args.run_date)
            chosen, selection = select_gqvae(candidates, args.gqvae_ablation)
            selection.update({
                "pipeline": definition["name"],
                "run_date": args.run_date,
                "tokenizer_output": str(gqvae_tokenizer),
            })
            _export_gqvae(
                chosen,
                gqvae_tokenizer,
                selection_path,
                selection,
                dry_run=False,
            )
            state["gqvae_selection"] = selection["selected"]
            atomic_json_dump(state, state_path)

        corpus_experiments, corpora = _dynamic_corpus_experiments(
            definition,
            args.run_date,
            bpe_path,
            gqvae_tokenizer,
        )
        state["current_stage"] = "lm_corpus"
        state["stages"]["lm_corpus"] = _run_stage(
            definition,
            "lm_corpus",
            args.run_date,
            dry_run=args.dry_run,
            experiments=corpus_experiments,
        )
        if not args.dry_run:
            atomic_json_dump(state, state_path)
        nano_experiments = _dynamic_nanogpt_experiments(definition, corpora)
        state["current_stage"] = "nanogpt"
        state["stages"]["nanogpt"] = _run_stage(
            definition,
            "nanogpt",
            args.run_date,
            dry_run=args.dry_run,
            experiments=nano_experiments,
        )
        if not args.dry_run:
            atomic_json_dump(state, state_path)
        state["current_stage"] = "visualization"
        state["stages"]["visualization"] = _render_visualizations(
            pipeline_dir,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        if not args.dry_run:
            state["status"] = "interrupted"
            state["error"] = "KeyboardInterrupt"
            atomic_json_dump(state, state_path)
        raise
    except Exception as error:
        if not args.dry_run:
            state["status"] = "failed"
            state["error"] = repr(error)
            atomic_json_dump(state, state_path)
        raise

    if not args.dry_run:
        state["status"] = "completed"
        state.pop("current_stage", None)
        atomic_json_dump(state, state_path)
    print(f"\n[Pipeline] {'DRY RUN complete' if args.dry_run else 'COMPLETED'}: {pipeline_dir}")


if __name__ == "__main__":
    main()
