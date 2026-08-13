"""Run the faithful GQ-VAE v1 setup and diagnose gate bistability."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from common import ROOT, enable_tf32, get_device
from common.gqvae_paper_config import (
    PAPER_ARXIV,
    PAPER_REPOSITORY,
    PAPER_REVISION,
    GQVAEPaperDataConfig,
    GQVAEPaperModelConfig,
    GQVAEPaperTrainConfig,
    GateBistabilityConfig,
    dataclass_from_dict,
)
from common.tracking import wandb_run
from models.gqvae_paper import GQVAEPaper
from training.gqvae_paper import (
    build_paper_optimizer,
    build_paper_scheduler,
    gate_statistics,
    load_or_prepare_dataset,
    make_paper_loader,
    numeric_tracker_metrics,
    output_metrics,
    reproduction_manifest,
    split_paper_dataset,
)
from training.text_vqvae.reporting import append_jsonl, atomic_json_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def load_config(path: Path):
    with path.expanduser().open(encoding="utf-8") as handle:
        payload = json.load(handle)
    allowed = {"paper", "model", "train", "data", "bistability"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown root config fields: {', '.join(unknown)}")
    paper = payload.get("paper", {})
    if paper.get("revision", PAPER_REVISION) != PAPER_REVISION:
        raise ValueError(f"This implementation is pinned to paper revision {PAPER_REVISION}.")
    return (
        dataclass_from_dict(GQVAEPaperModelConfig, payload.get("model")),
        dataclass_from_dict(GQVAEPaperTrainConfig, payload.get("train")),
        dataclass_from_dict(GQVAEPaperDataConfig, payload.get("data")),
        dataclass_from_dict(GateBistabilityConfig, payload.get("bistability")),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, *, step: int, batches: int, diagnostics):
    was_training = model.training
    numeric_totals: dict[str, float] = {}
    histogram = torch.zeros(diagnostics.histogram_bins, dtype=torch.long)
    gate_values: list[torch.Tensor] = []
    count = 0
    try:
        # The released validate() does not call eval(); preserve its train-mode
        # layer behavior while no_grad keeps this diagnostic read-only.
        model.train()
        for input_ids in loader:
            if count >= batches:
                break
            input_ids = input_ids.to(device, non_blocking=True)
            output = model(input_ids, step=step, update_quantizer_state=False)
            metrics = output_metrics(output, input_ids, diagnostics)
            for key, value in numeric_tracker_metrics(metrics).items():
                numeric_totals[key] = numeric_totals.get(key, 0.0) + float(value)
            histogram += torch.tensor(metrics["gate_histogram"], dtype=torch.long)
            gate_values.append(output.gates.detach().cpu())
            count += 1
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("Validation loader yielded no full batches.")
    result = {key: value / count for key, value in numeric_totals.items()}
    result.update(gate_statistics(torch.cat(gate_values), diagnostics))
    result["gate_histogram"] = histogram.tolist()
    result["batches"] = count
    return result


def classify_trajectory(eval_rows: list[dict[str, object]]) -> dict[str, object]:
    states = [str(row["gate_state"]) for row in eval_rows]
    terminal = states[-1]
    switches = sum(left != right for left, right in zip(states, states[1:]))
    return {
        "terminal_gate_state": terminal,
        "gate_state_switches": switches,
        "visited_gate_states": list(dict.fromkeys(states)),
        "collapsed": terminal in {"collapsed_zero", "collapsed_one"},
    }


def main() -> None:
    args = parse_args()
    model_config, train_config, data_config, diagnostics = load_config(args.config)
    payload = {
        "paper": {
            "arxiv": PAPER_ARXIV,
            "repository": PAPER_REPOSITORY,
            "revision": PAPER_REVISION,
        },
        "model": asdict(model_config),
        "train": asdict(train_config),
        "data": asdict(data_config),
        "bistability": asdict(diagnostics),
        "seed": args.seed,
        "reproduction": reproduction_manifest(train_config, data_config),
        "diagnostic_extensions": {
            "seed_is_applied": True,
            "note": (
                "The released CLI accepts --seed but never applies it. This runner "
                "applies the requested seed solely to make initialization sensitivity "
                "and cross-run gate bistability measurable."
            ),
        },
    }
    if args.print_config:
        print(json.dumps(payload, indent=2))
        return

    run_dir = ROOT / "outputs" / "gqvae_paper_bistability" / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    seed_everything(args.seed)
    device = get_device()
    enable_tf32(device)
    dataset = load_or_prepare_dataset(
        data_config,
        input_len=model_config.input_len,
        prepared_output=run_dir / "prepared_ascii_gpt2.pt",
    )
    train_dataset, validation_dataset = split_paper_dataset(
        dataset,
        train_fraction=data_config.train_fraction,
    )
    train_loader = make_paper_loader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )
    validation_loader = make_paper_loader(
        validation_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,  # mirrors the released validation DataLoader
        num_workers=train_config.num_workers,
    )
    if len(train_loader) == 0 or len(validation_loader) == 0:
        raise ValueError(
            "Paper DataLoaders need at least one full batch in both splits; "
            "prepare more data or preserve the paper batch_size=1024."
        )
    model = GQVAEPaper(model_config).to(device)
    optimizer = build_paper_optimizer(model, train_config)
    total_steps = train_config.epochs * len(train_loader)
    scheduler = build_paper_scheduler(optimizer, train_config, total_steps=total_steps)
    payload.update({
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "total_steps": total_steps,
        "device": str(device),
    })
    atomic_json_dump(payload, run_dir / "config.json")
    metrics_path = run_dir / "metrics.jsonl"
    eval_rows: list[dict[str, object]] = []
    step = 0
    started = time.time()
    print(
        f"[GQ-VAE paper v1] run={args.run_name} seed={args.seed} "
        f"device={device} params={payload['parameter_count']:,} steps={total_steps}"
    )
    with wandb_run(
        args.run_name,
        group="gqvae-paper-bistability",
        tags=["gqvae", "paper-v1", "gate-bistability", f"seed-{args.seed}"],
        config=payload,
    ) as tracker:
        for epoch in range(1, train_config.epochs + 1):
            model.train()
            for input_ids in train_loader:
                input_ids = input_ids.to(device, non_blocking=True)
                output = model(input_ids, step=step)
                optimizer.zero_grad(set_to_none=True)
                output.loss.backward()
                optimizer.step()
                scheduler.step()
                train_metrics = output_metrics(output, input_ids, diagnostics)
                row = {
                    "split": "train",
                    "epoch": epoch,
                    "step": step,
                    "lr": optimizer.param_groups[0]["lr"],
                    "quantizer_lr": optimizer.param_groups[1]["lr"],
                    "quantizer_active": output.quantizer_active,
                    "elapsed_sec": time.time() - started,
                    **train_metrics,
                }
                if step % train_config.log_every == 0:
                    append_jsonl(row, metrics_path)
                    tracker.log(
                        {f"train/{key}": value for key, value in numeric_tracker_metrics(row).items()},
                        step=step,
                    )
                eval_due = step == 0 or (
                    step != 0 and step % train_config.eval_every == 0
                )
                if eval_due:
                    validation = evaluate(
                        model,
                        validation_loader,
                        device,
                        step=step,
                        batches=train_config.eval_batches,
                        diagnostics=diagnostics,
                    )
                    eval_row = {"split": "eval", "epoch": epoch, "step": step, **validation}
                    append_jsonl(eval_row, metrics_path)
                    eval_rows.append(eval_row)
                    tracker.log(
                        {f"eval/{key}": value for key, value in numeric_tracker_metrics(validation).items()},
                        step=step,
                    )
                    print(
                        f"[Eval] step={step} gate={validation['gate_mean']:.4f} "
                        f"hard_on={validation['gate_hard_on_fraction']:.4f} "
                        f"state={validation['gate_state']}"
                    )
                if step % train_config.save_every == 0:
                    torch.save(
                        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
                        run_dir / "checkpoints" / f"step{step}.pt",
                    )
                step += 1

        final_eval = evaluate(
            model,
            validation_loader,
            device,
            step=step,
            batches=train_config.eval_batches,
            diagnostics=diagnostics,
        )
        final_row = {"split": "eval", "epoch": train_config.epochs, "step": step, **final_eval}
        append_jsonl(final_row, metrics_path)
        eval_rows.append(final_row)
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
            run_dir / "checkpoints" / "last.pt",
        )

    summary = {
        "status": "completed",
        "run_name": args.run_name,
        "seed": args.seed,
        "steps": step,
        "elapsed_sec": time.time() - started,
        "final_eval": final_eval,
        **classify_trajectory(eval_rows),
    }
    atomic_json_dump(summary, run_dir / "summary.json")
    print(
        f"[Complete] seed={args.seed} terminal={summary['terminal_gate_state']} "
        f"switches={summary['gate_state_switches']}"
    )


if __name__ == "__main__":
    main()
