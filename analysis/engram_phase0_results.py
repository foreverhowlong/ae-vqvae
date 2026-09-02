"""Aggregate the four Engram Phase-0 runs, plot curves, and issue the verdict."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ("baseline", "engram_s", "engram_m", "engram_l")
LABELS = {
    "baseline": "baseline",
    "engram_s": "M≈32K",
    "engram_m": "M≈128K",
    "engram_l": "M≈512K",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_validation(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == "validation":
                rows.append(row)
    if not rows:
        raise ValueError(f"No validation metrics in {path}")
    return sorted(rows, key=lambda row: int(row["tokens_seen"]))


def load_runs(sweep_dir: Path):
    runs = {}
    reference = None
    for variant in VARIANTS:
        run_dir = sweep_dir / variant
        config = _read_json(run_dir / "config.json")
        summary = _read_json(run_dir / "summary.json")
        if summary.get("status") != "completed":
            raise ValueError(f"Run {variant} is not completed.")
        if config["variant"] != variant:
            raise ValueError(f"Variant mismatch in {run_dir}")
        comparison_contract = {
            "profile_name": config["profile_name"],
            "backbone": config["backbone"],
            "optimization": config["optimization"],
            "profile": config["profile"],
            "corpus_train_sha": config["corpus"]["train"]["sha256"],
            "corpus_validation_sha": config["corpus"]["validation"]["sha256"],
            "go_criteria": config["go_criteria"],
            "engram_fixed": {
                key: value
                for key, value in config["engram"].items()
                if key not in {"enabled", "table_rows_target"}
            },
        }
        if reference is None:
            reference = comparison_contract
        elif comparison_contract != reference:
            raise ValueError(f"Run {variant} violates the fixed comparison contract.")
        runs[variant] = {
            "dir": run_dir,
            "config": config,
            "summary": summary,
            "validation": _read_validation(run_dir / "metrics.jsonl"),
        }
    return runs


def create_results_csv(runs, output: Path) -> list[dict[str, Any]]:
    baseline = float(runs["baseline"]["validation"][-1]["validation_nll"])
    rows = []
    for variant in VARIANTS:
        run = runs[variant]
        final = run["validation"][-1]
        engram = run["config"]["engram"]
        counts = run["config"]["parameter_counts"]
        nll = float(final["validation_nll"])
        rows.append(
            {
                "variant": variant,
                "label": LABELS[variant],
                "profile": run["config"]["profile_name"],
                "tokens_seen": int(final["tokens_seen"]),
                "target_rows_per_table": int(engram["table_rows_target"]),
                "actual_total_table_rows": int(counts["sparse_tables"])
                // int(engram["memory_dim"] // 16)
                if engram["enabled"]
                else 0,
                "sparse_table_parameters": int(counts["sparse_tables"]),
                "dense_engram_parameters": int(counts["dense_engram"]),
                "backbone_parameters": int(counts["backbone"]),
                "total_parameters": int(counts["total"]),
                "final_validation_nll": nll,
                "final_perplexity": float(final["perplexity"]),
                "nll_improvement_vs_baseline": baseline - nll,
                "engram_gate_mean": final.get("engram_gate_mean"),
                "engram_gate_std": final.get("engram_gate_std"),
            }
        )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_curves(runs, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for variant in VARIANTS:
        points = runs[variant]["validation"]
        axis.plot(
            [float(row["tokens_seen"]) / 1e6 for row in points],
            [float(row["validation_nll"]) for row in points],
            marker="o",
            markersize=3,
            label=LABELS[variant],
        )
    axis.set_xlabel("Training tokens (millions)")
    axis.set_ylabel("Validation NLL (nats/token)")
    axis.set_title("Engram Phase-0: validation loss vs. training tokens")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_capacity(rows: list[dict[str, Any]], output: Path) -> None:
    baseline = float(rows[0]["final_validation_nll"])
    memory_rows = rows[1:]
    x = [int(row["target_rows_per_table"]) for row in memory_rows]
    y = [float(row["final_validation_nll"]) for row in memory_rows]
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    axis.axhline(baseline, color="black", linestyle="--", label="baseline (no table)")
    axis.plot(x, y, marker="o", linewidth=2, label="Engram")
    axis.set_xscale("log", base=2)
    axis.set_xticks(x, ["32K", "128K", "512K"])
    axis.set_xlabel("Target rows per hash table M (log scale)")
    axis.set_ylabel("Final validation NLL (nats/token)")
    axis.set_title("Engram Phase-0: final loss vs. memory capacity")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def judge(runs, rows: list[dict[str, Any]]) -> dict[str, Any]:
    criteria = runs["baseline"]["config"]["go_criteria"]
    baseline_nll = float(rows[0]["final_validation_nll"])
    large_nll = float(rows[-1]["final_validation_nll"])
    large_improvement = baseline_nll - large_nll
    x = np.log([float(row["target_rows_per_table"]) for row in rows[1:]])
    y = np.asarray([float(row["final_validation_nll"]) for row in rows[1:]])
    capacity_slope = float(np.polyfit(x, y, 1)[0])
    overall_positive = capacity_slope < 0.0 and y[-1] < y[0]

    baseline_by_tokens = {
        int(row["tokens_seen"]): float(row["validation_nll"])
        for row in runs["baseline"]["validation"]
    }
    large_by_tokens = {
        int(row["tokens_seen"]): float(row["validation_nll"])
        for row in runs["engram_l"]["validation"]
    }
    common = sorted(set(baseline_by_tokens) & set(large_by_tokens))
    tail_points = int(criteria["late_tail_points"])
    tail = common[-tail_points:]
    tail_improvements = [
        baseline_by_tokens[tokens] - large_by_tokens[tokens] for tokens in tail
    ]
    positive_fraction = (
        sum(value > 0 for value in tail_improvements) / len(tail_improvements)
        if tail_improvements
        else 0.0
    )
    persistent = (
        len(tail) >= tail_points
        and positive_fraction >= float(criteria["late_tail_min_positive_fraction"])
    )
    threshold_met = large_improvement >= float(criteria["large_min_nll_improvement"])
    profile = runs["baseline"]["config"]["profile_name"]
    evidence_pass = threshold_met and overall_positive and persistent
    decision = "GO" if profile == "final" and evidence_pass else (
        "PILOT_ONLY" if profile != "final" else "NO_GO"
    )
    return {
        "decision": decision,
        "profile": profile,
        "evidence_passes_go_rule": evidence_pass,
        "engram_l_improvement_nats_per_token": large_improvement,
        "required_improvement_nats_per_token": float(
            criteria["large_min_nll_improvement"]
        ),
        "large_improvement_threshold_met": threshold_met,
        "engram_capacity_log_slope": capacity_slope,
        "capacity_trend_positive": overall_positive,
        "late_tail_tokens": tail,
        "late_tail_improvements": tail_improvements,
        "late_tail_positive_fraction": positive_fraction,
        "late_tail_persistent": persistent,
        "rule": (
            "GO requires final profile, Engram-L improvement >= threshold, negative OLS slope "
            "of NLL vs log(M) with L better than S, and the configured fraction of late-tail "
            "matched evaluations favoring Engram-L."
        ),
    }


def write_judgement(verdict: dict[str, Any], path: Path) -> None:
    tail = verdict["late_tail_improvements"]
    body = [
        "# Engram Phase-0 judgment",
        "",
        f"**Decision: {verdict['decision']}**",
        "",
        f"- Profile: `{verdict['profile']}`",
        f"- Engram-L improvement vs baseline: {verdict['engram_l_improvement_nats_per_token']:.6f} nats/token",
        f"- Required improvement: {verdict['required_improvement_nats_per_token']:.6f} nats/token",
        f"- Capacity log-slope: {verdict['engram_capacity_log_slope']:.6g}",
        f"- Late-tail improvements: {', '.join(f'{value:.6f}' for value in tail) if tail else 'insufficient points'}",
        "",
        verdict["rule"],
        "",
    ]
    path.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    sweep_dir = args_dir = parse_args().sweep_dir.expanduser().resolve()
    runs = load_runs(args_dir)
    rows = create_results_csv(runs, sweep_dir / "results.csv")
    plot_curves(runs, sweep_dir / "val_loss_vs_tokens.png")
    plot_capacity(rows, sweep_dir / "final_val_loss_vs_log_table_size.png")
    verdict = judge(runs, rows)
    (sweep_dir / "phase0_judgement.json").write_text(
        json.dumps(verdict, indent=2) + "\n", encoding="utf-8"
    )
    write_judgement(verdict, sweep_dir / "phase0_judgement.md")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
