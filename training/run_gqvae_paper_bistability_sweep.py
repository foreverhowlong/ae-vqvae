"""Launch the pinned GQ-VAE paper run across seeds and aggregate gate states."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import ROOT
from training.text_vqvae.reporting import atomic_json_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_seeds(config: Path) -> list[int]:
    with config.expanduser().open(encoding="utf-8") as handle:
        payload = json.load(handle)
    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 3 or not all(isinstance(x, int) for x in seeds):
        raise ValueError("Bistability config must provide at least three integer seeds.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Bistability seeds must be unique.")
    return seeds


def model_config_path(config: Path) -> Path:
    with config.expanduser().open(encoding="utf-8") as handle:
        payload = json.load(handle)
    relative = payload.get("experiment-config")
    if not isinstance(relative, str):
        raise ValueError("Bistability config requires an experiment-config path.")
    return (ROOT / relative).resolve()


def aggregate(run_names: list[str], output: Path) -> None:
    rows = []
    for run_name in run_names:
        summary_path = ROOT / "outputs" / "gqvae_paper_bistability" / run_name / "summary.json"
        with summary_path.open(encoding="utf-8") as handle:
            rows.append(json.load(handle))
    states: dict[str, int] = {}
    for row in rows:
        state = row["terminal_gate_state"]
        states[state] = states.get(state, 0) + 1
    atomic_json_dump(
        {
            "runs": rows,
            "terminal_state_counts": states,
            "observed_zero_one_bistability": (
                states.get("collapsed_zero", 0) > 0 and states.get("collapsed_one", 0) > 0
            ),
        },
        output,
    )


def main() -> None:
    args = parse_args()
    seeds = load_seeds(args.config)
    experiment = model_config_path(args.config)
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    run_names = [f"gqvae-paper-v1-alpha3-seed{seed}__{args.run_date}" for seed in seeds]
    commands = []
    for index, (seed, run_name) in enumerate(zip(seeds, run_names, strict=True)):
        command = [
            sys.executable,
            "-m",
            "training.run_gqvae_paper_bistability",
            "--config",
            str(experiment),
            "--run-name",
            run_name,
            "--seed",
            str(seed),
        ]
        environment = None
        if gpus:
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = gpus[index % len(gpus)]
        commands.append((run_name, command, environment))

    print(f"[GQ-VAE paper bistability] seeds={seeds} gpus={gpus or ['inherited']}")
    if args.dry_run:
        for run_name, command, environment in commands:
            prefix = ""
            if environment is not None:
                prefix = f"CUDA_VISIBLE_DEVICES={environment['CUDA_VISIBLE_DEVICES']} "
            print(f"[{run_name}]\n  {prefix}{' '.join(command)}")
        return

    # Run one process per assigned GPU concurrently. With no explicit GPU list,
    # run sequentially to avoid accidental device oversubscription.
    if not gpus:
        for _run_name, command, environment in commands:
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
    else:
        pending = list(commands)
        active: list[tuple[str, subprocess.Popen, str]] = []
        free_gpus = list(gpus)
        while pending or active:
            while pending and free_gpus:
                run_name, command, environment = pending.pop(0)
                gpu = environment["CUDA_VISIBLE_DEVICES"]
                if gpu not in free_gpus:
                    pending.append((run_name, command, environment))
                    break
                free_gpus.remove(gpu)
                process = subprocess.Popen(command, cwd=ROOT, env=environment)
                active.append((run_name, process, gpu))
            for item in list(active):
                run_name, process, gpu = item
                return_code = process.poll()
                if return_code is None:
                    continue
                active.remove(item)
                free_gpus.append(gpu)
                if return_code:
                    raise subprocess.CalledProcessError(return_code, process.args)
            if active:
                time.sleep(1)

    output = ROOT / "outputs" / "gqvae_paper_bistability" / f"summary__{args.run_date}.json"
    aggregate(run_names, output)
    print(f"[Aggregate] {output}")


if __name__ == "__main__":
    main()
