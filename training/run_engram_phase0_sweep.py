"""Run the four Phase-0 variants sequentially, then aggregate their results."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from training.run_engram_phase0 import VARIANTS, _load_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for variant in VARIANTS:
        command = [
            sys.executable,
            "-m",
            "training.run_engram_phase0",
            "--config",
            str(args.config),
            "--data-dir",
            str(args.data_dir),
            "--variant",
            variant,
            "--profile",
            args.profile,
            "--output-root",
            str(args.output_root),
        ]
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, check=True)
    if not args.dry_run:
        experiment = _load_json(args.config)["experiment_name"]
        sweep_dir = args.output_root.expanduser().resolve() / experiment / args.profile
        subprocess.run(
            [
                sys.executable,
                "-m",
                "analysis.engram_phase0_results",
                "--sweep-dir",
                str(sweep_dir),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
