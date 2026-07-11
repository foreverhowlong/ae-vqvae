"""Run a sequence of training experiments described by a JSON file.

Example:
    python -m training.run_experiment_sequence \
        --config configs/text_vqvae_experiments.example.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from common import ROOT


DEFAULT_MODULE = "training.run_text_vqvae_experiment"
_PARAMETER_NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")
_UNSAFE_FILENAME_CHARS = re.compile(r"[^a-zA-Z0-9._-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run configured experiments sequentially, preserving training defaults."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to the JSON experiment config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated commands without starting training.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later experiments if one command fails.",
    )
    return parser.parse_args()


def load_config(path: Path) -> tuple[str, list[dict[str, Any]]]:
    with path.expanduser().open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("The config root must be a JSON object.")

    module = payload.get("module", DEFAULT_MODULE)
    experiments = payload.get("experiments")
    if not isinstance(module, str) or not module.startswith("training."):
        raise ValueError("'module' must name a module inside the training package.")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("'experiments' must be a non-empty list.")
    if not all(isinstance(item, dict) and item for item in experiments):
        raise ValueError("Each experiment must be a non-empty JSON object of CLI parameters.")

    return module, experiments


def parameter_to_cli(name: str, value: Any) -> list[str]:
    """Convert one JSON entry to argparse-style CLI tokens."""
    if not _PARAMETER_NAME.fullmatch(name):
        raise ValueError(f"Invalid parameter name: {name!r}")
    if name == "run-name":
        raise ValueError("'run-name' is generated automatically and must not appear in the config.")

    option = f"--{name.replace('_', '-')}"
    if isinstance(value, bool):
        return [option] if value else [f"--no-{option[2:]}"]
    if value is None or isinstance(value, (dict, list)):
        raise ValueError(f"Parameter {name!r} must be a string, number, or boolean.")
    return [option, str(value)]


def _filename_value(value: Any) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    text = _UNSAFE_FILENAME_CHARS.sub("-", text).strip("-._") or "empty"
    return text[:40]


def make_run_name(parameters: dict[str, Any], date: str) -> str:
    """Include every explicitly configured parameter and the launch date."""
    parts = [
        f"{name.replace('_', '-')}-{_filename_value(value)}"
        for name, value in sorted(parameters.items())
    ]
    return "__".join([*parts, date])


def build_command(module: str, parameters: dict[str, Any], run_name: str) -> list[str]:
    command = [sys.executable, "-m", module, "--run-name", run_name]
    for name, value in parameters.items():
        command.extend(parameter_to_cli(name, value))
    return command


def main() -> None:
    args = parse_args()
    module, experiments = load_config(args.config)
    launch_date = datetime.now().astimezone().strftime("%Y%m%d")

    commands: list[tuple[str, list[str]]] = []
    seen_names: set[str] = set()
    for index, parameters in enumerate(experiments, start=1):
        run_name = make_run_name(parameters, launch_date)
        if run_name in seen_names:
            raise ValueError(f"Experiments must be unique; duplicate generated name: {run_name}")
        seen_names.add(run_name)
        commands.append((run_name, build_command(module, parameters, run_name)))

    print(f"Loaded {len(commands)} experiments from {args.config}")
    for index, (run_name, command) in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] {run_name}")
        print("  " + " ".join(command))
        if args.dry_run:
            continue

        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            print(f"Experiment {index} failed with exit code {error.returncode}.", file=sys.stderr)
            if not args.continue_on_error:
                raise SystemExit(error.returncode) from error


if __name__ == "__main__":
    main()
