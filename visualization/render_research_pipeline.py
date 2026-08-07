"""Render paper-aligned and experiment-specific figures for one research pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.learned_tokenizer import LearnedByteFallbackTokenizer


COLORS = {
    "bpe": "#4C78A8",
    "gqvae": "#F58518",
    "vqvae": "#54A24B",
    "nearest": "#4C78A8",
    "topk": "#E45756",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(
    path: Path,
    splits: str | Iterable[str],
    *,
    max_points: int | None = None,
) -> list[dict[str, Any]]:
    wanted = {splits} if isinstance(splits, str) else set(splits)

    def matching_rows():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("split") in wanted:
                    yield row

    if max_points is None:
        return list(matching_rows())
    count = sum(1 for _ in matching_rows())
    stride = max(math.ceil(count / max_points), 1)
    sampled = []
    last = None
    for index, row in enumerate(matching_rows()):
        last = row
        if index % stride == 0:
            sampled.append(row)
    if last is not None and (not sampled or sampled[-1] is not last):
        sampled.append(last)
    return sampled


def _finish(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _label_points(axis, x, y, labels) -> None:
    for x_value, y_value, label in zip(x, y, labels, strict=True):
        axis.annotate(
            label,
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )


def render_gqvae_architecture(plot_dir: Path) -> list[Path]:
    """Render current-model counterparts to the paper's Figures 1 and 2."""
    paths = []
    fig, axis = plt.subplots(figsize=(15, 5.2))
    axis.set_xlim(0, 15)
    axis.set_ylim(0, 7)
    axis.axis("off")
    nodes = [
        (0.4, 3.0, 2.0, 1.1, "UTF-8 bytes\n+ positions", "#DCEAF7"),
        (3.0, 3.0, 2.2, 1.1, "Bidirectional\nTransformer encoder", "#DCEAF7"),
        (6.0, 3.0, 1.8, 1.1, "VQ codebook\nK = 8192", "#FDE2C5"),
        (8.7, 4.6, 2.1, 1.1, "Transformer gater\nsigmoid gₜ", "#E6DDF2"),
        (11.7, 4.6, 2.1, 1.1, "threshold gₜ > 0.5\nselected codes", "#E6DDF2"),
        (8.7, 1.4, 2.1, 1.1, "decoder expand\nwidth w", "#DDEFD8"),
        (11.7, 1.4, 2.1, 1.1, "byte logits +\nlength logits", "#DDEFD8"),
    ]
    for x, y, width, height, text, color in nodes:
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.55", "facecolor": color, "edgecolor": "#44546A"},
        )

    def arrow(start, end, label=None):
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.7})
        if label:
            axis.text(
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2 + 0.18,
                label,
                ha="center",
                fontsize=8,
            )

    arrow((2.4, 3.55), (3.0, 3.55), "hidden states")
    arrow((5.2, 3.55), (6.0, 3.55), "zₜ")
    arrow((7.8, 3.75), (8.7, 4.95), "quantized z̄ₜ")
    arrow((10.8, 5.15), (11.7, 5.15))
    arrow((7.8, 3.35), (8.7, 2.0), "quantized z̄ₜ")
    arrow((10.8, 1.95), (11.7, 1.95))
    axis.text(
        7.0,
        6.35,
        "Current text GQ-VAE architecture (paper Figure 1 counterpart)",
        ha="center",
        fontsize=15,
        fontweight="bold",
    )
    axis.text(
        7.0,
        0.35,
        "Training objective: reconstruction + α·gate compression + length + codebook + β·commitment",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    path = plot_dir / "paper_fig1_gqvae_architecture.png"
    _finish(fig, path)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(12, 5))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 6)
    axis.axis("off")
    decoder_nodes = [
        (0.6, 2.4, "quantized code\n[code_dim]", "#FDE2C5"),
        (3.2, 2.4, "linear expansion + GELU\n[w × d_model]", "#DDEFD8"),
        (6.4, 3.7, "byte head\n[w × vocab]", "#DCEAF7"),
        (6.4, 1.1, "length head\n[w logits]", "#E6DDF2"),
        (9.4, 3.7, "decoded bytes\nargmax", "#DCEAF7"),
        (9.4, 1.1, "decoded length\nargmax + 1", "#E6DDF2"),
    ]
    for x, y, text, color in decoder_nodes:
        axis.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.6", "facecolor": color, "edgecolor": "#44546A"},
        )
    for start, end in (
        ((1.45, 2.4), (2.35, 2.4)),
        ((4.4, 2.65), (5.55, 3.65)),
        ((4.4, 2.15), (5.55, 1.35)),
        ((7.25, 3.7), (8.55, 3.7)),
        ((7.25, 1.1), (8.55, 1.1)),
    ):
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.7})
    axis.set_title(
        "Current decoder head (paper Figure 2 counterpart)",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )
    path = plot_dir / "paper_fig2_decoder_head.png"
    _finish(fig, path)
    paths.append(path)
    return paths


def _stage_targets(state: dict[str, Any], stage: str) -> list[Path]:
    return [Path(item["target"]) for item in state["stages"][stage]]


def _ablation_map(config_path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(config_path)
    return {item["ablation"]: item for item in payload["experiments"]}


def render_topk(state: dict[str, Any], plot_dir: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    targets = _stage_targets(state, "topk")
    runs = {target.name: target for target in targets}
    topk_target = next(target for name, target in runs.items() if "topk8-to-1" in name)
    train = _read_rows(topk_target / "metrics.jsonl", "train", max_points=2500)
    train = [row for row in train if "quantizer_topk" in row]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    steps = [row["step"] for row in train]
    axes[0].step(steps, [row["quantizer_topk"] for row in train], where="post", label="configured K")
    axes[0].plot(
        steps,
        [row.get("quantizer_effective_k", 1.0) for row in train],
        color="#E45756",
        alpha=0.8,
        label="effective K = exp(entropy)",
    )
    axes[0].set(ylabel="Codes per latent", title="Top-k sparse-mixture curriculum")
    axes[0].legend()
    axes[1].plot(steps, [row["quantizer_temperature"] for row in train], label="temperature")
    axes[1].plot(
        steps,
        [row.get("quantizer_mixture_entropy", 0.0) for row in train],
        label="mixture entropy (nats)",
    )
    axes[1].set(xlabel="Optimizer step", ylabel="Schedule / entropy")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    path_schedule = plot_dir / "topk_curriculum.png"
    _finish(fig, path_schedule)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metric_specs = (
        ("recon_nll", "Eval reconstruction NLL", "NLL (nats/token)"),
        ("token_accuracy", "Eval reconstruction accuracy", "Correct-token fraction"),
        ("codebook_utilization", "Eval codebook utilization", "Used-code fraction"),
        ("bits_per_token", "Estimated latent rate", "Bits/input token"),
    )
    records = []
    for name, target in runs.items():
        label = "Top-k 8→1" if "topk8-to-1" in name else "Nearest"
        key = "topk" if "topk8-to-1" in name else "nearest"
        rows = _read_rows(target / "metrics.jsonl", "eval")
        for axis, (metric, title, ylabel) in zip(axes.flat, metric_specs, strict=True):
            values = [(row["step"], row[metric]) for row in rows if metric in row]
            if values:
                axis.plot(
                    [item[0] for item in values],
                    [item[1] for item in values],
                    marker=".",
                    label=label,
                    color=COLORS[key],
                )
            axis.set(title=title, xlabel="Optimizer step", ylabel=ylabel)
            axis.grid(alpha=0.25)
        summary = _load_json(target / "summary.json")
        records.append({"stage": "topk", "run": name, "mode": key, **summary["final_eval"]})
    for axis in axes.flat:
        axis.legend()
    path_comparison = plot_dir / "topk_vs_nearest.png"
    _finish(fig, path_comparison)
    return [path_schedule, path_comparison], records


def render_gqvae(state: dict[str, Any], plot_dir: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    selection = _load_json(Path(state["gqvae_selection_manifest"]))
    candidates = selection["candidates"]
    selected = selection["selected"]["ablation"]
    frontier_names = set(selection.get("pareto_frontier") or [])
    fig, axis = plt.subplots(figsize=(9, 6.5))
    for candidate in candidates:
        chosen = candidate["ablation"] == selected
        axis.scatter(
            candidate["bytes_per_token"],
            candidate["reconstruction_loss"],
            s=150 if chosen else 75,
            marker="*" if chosen else "o",
            color="#E45756" if chosen else "#4C78A8",
            edgecolor="black" if candidate["ablation"] in frontier_names else "none",
            zorder=3,
        )
        axis.annotate(candidate["ablation"].replace("gqvae-k8192-", ""), (
            candidate["bytes_per_token"], candidate["reconstruction_loss"]
        ), xytext=(6, 5), textcoords="offset points", fontsize=9)
    frontier = sorted(
        (item for item in candidates if item["ablation"] in frontier_names),
        key=lambda item: item["bytes_per_token"],
    )
    if len(frontier) > 1:
        axis.plot(
            [item["bytes_per_token"] for item in frontier],
            [item["reconstruction_loss"] for item in frontier],
            linestyle="--",
            color="#666666",
            label="Pareto frontier",
        )
        axis.legend()
    axis.set(
        title="GQ-VAE rate-distortion frontier (K=8192)",
        xlabel="Validation bytes per selected token (higher compression →)",
        ylabel="Validation reconstruction loss (lower is better)",
    )
    axis.grid(alpha=0.25)
    path_frontier = plot_dir / "gqvae_rate_distortion.png"
    _finish(fig, path_frontier)

    targets = _stage_targets(state, "gqvae")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    records = []
    for target in targets:
        label = target.name.split("__", 1)[0].replace("gqvae-k8192-", "")
        train = _read_rows(target / "metrics.jsonl", "train", max_points=1500)
        evaluation = _read_rows(target / "metrics.jsonl", "eval")
        axes[0, 0].plot(
            [row["step"] for row in train],
            [row["compression_weight"] for row in train],
            label=label,
        )
        for axis, metric in (
            (axes[0, 1], "reconstruction_loss"),
            (axes[1, 0], "bytes_per_token"),
            (axes[1, 1], "byte_accuracy"),
        ):
            rows = [row for row in evaluation if metric in row]
            axis.plot([row["step"] for row in rows], [row[metric] for row in rows], marker=".", label=label)
        summary = _load_json(target / "summary.json")
        records.append({
            "stage": "gqvae",
            "run": target.name,
            "alpha": _load_json(target / "config.json")["model"]["compression_weight"],
            **summary["final_eval"],
        })
    titles = (
        (axes[0, 0], "Compression-weight curriculum", "α"),
        (axes[0, 1], "Validation reconstruction", "Loss"),
        (axes[1, 0], "Validation compression", "Bytes/token"),
        (axes[1, 1], "Validation byte accuracy", "Correct-byte fraction"),
    )
    for axis, title, ylabel in titles:
        axis.set(title=title, xlabel="Optimizer step", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    path_dynamics = plot_dir / "gqvae_training_dynamics.png"
    _finish(fig, path_dynamics)
    return [path_frontier, path_dynamics], records


def render_beta_sweep(state: dict[str, Any], plot_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    records = []
    for target in _stage_targets(state, "commitment_beta"):
        config = _load_json(target / "config.json")
        summary = _load_json(target / "summary.json")
        records.append({
            "stage": "commitment_beta",
            "run": target.name,
            "beta": float(config["model"]["commitment_beta"]),
            **summary["final_eval"],
        })
    records.sort(key=lambda item: item["beta"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    specs = (
        ("recon_nll", "Reconstruction NLL", "NLL (nats/token)"),
        ("commitment_loss", "Raw commitment loss", "MSE"),
        ("codebook_utilization", "Codebook utilization", "Used-code fraction"),
        ("bits_per_token", "Estimated latent rate", "Bits/input token"),
    )
    beta = [row["beta"] for row in records]
    for axis, (metric, title, ylabel) in zip(axes.flat, specs, strict=True):
        values = [row.get(metric, math.nan) for row in records]
        axis.plot(beta, values, marker="o")
        axis.set_xscale("log")
        axis.set(title=title, xlabel="Commitment β", ylabel=ylabel)
        axis.grid(alpha=0.25)
    path = plot_dir / "commitment_beta_sweep.png"
    _finish(fig, path)
    return path, records


def _corpus_records(state: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for target in _stage_targets(state, "lm_corpus"):
        metadata = _load_json(target / "meta.json")
        tokens = np.memmap(target / "train.bin", dtype=np.uint16, mode="r")
        counts = np.zeros(metadata["vocab_size"], dtype=np.int64)
        chunk_size = 1_000_000
        for start in range(0, len(tokens), chunk_size):
            chunk = np.asarray(tokens[start : start + chunk_size], dtype=np.int64)
            counts += np.bincount(chunk, minlength=metadata["vocab_size"])
        for token_id in {
            int(metadata["pad_token_id"]),
            int(metadata["bos_token_id"]),
            int(metadata["eos_token_id"]),
        }:
            counts[token_id] = 0
        records.append({
            "stage": "lm_corpus",
            "tokenizer": metadata["tokenizer"],
            "target": str(target),
            "vocab_size": metadata["vocab_size"],
            "used_vocab_size": int((counts > 0).sum()),
            "counts": counts,
            **metadata["train"],
        })
    return records


def render_paper_compression_and_frequency(
    state: dict[str, Any], plot_dir: Path
) -> tuple[list[Path], list[dict[str, Any]]]:
    records = _corpus_records(state)
    selection = _load_json(Path(state["gqvae_selection_manifest"]))
    gq_tokenizer = LearnedByteFallbackTokenizer.load(Path(state["gqvae_tokenizer"]))
    unique_gq = len({token for token in gq_tokenizer.learned_tokens if token})

    fig, axis = plt.subplots(figsize=(9.5, 6.5))
    for record in records:
        tokenizer = record["tokenizer"]
        axis.scatter(
            record["used_vocab_size"],
            record["bytes_per_token"],
            s=95,
            color=COLORS[tokenizer],
        )
        axis.annotate(
            f"{tokenizer.upper()} lossless",
            (record["used_vocab_size"], record["bytes_per_token"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )
    selected = selection["selected"]
    axis.scatter(
        unique_gq,
        selected["bytes_per_token"],
        s=120,
        marker="x",
        linewidth=2,
        color=COLORS["gqvae"],
    )
    axis.annotate(
        "GQVAE gate estimate\n(no fallback)",
        (unique_gq, selected["bytes_per_token"]),
        xytext=(6, 5),
        textcoords="offset points",
        fontsize=9,
    )
    axis.set(
        title="Compression and effective vocabulary (paper Figure 3 counterpart)",
        xlabel="Vocabulary entries (observed in corpora; unique decoded for gate estimate)",
        ylabel="Raw UTF-8 bytes per token (higher compression →)",
    )
    axis.grid(alpha=0.25)
    path_compression = plot_dir / "paper_fig3_compression_vocabulary.png"
    _finish(fig, path_compression)

    frequency = {record["tokenizer"]: record for record in records}
    fig, axis = plt.subplots(figsize=(11, 6.5))
    for tokenizer in ("bpe", "gqvae"):
        counts = np.sort(frequency[tokenizer]["counts"])[::-1][:50]
        fraction = counts / max(frequency[tokenizer]["counts"].sum(), 1)
        axis.plot(
            np.arange(1, len(fraction) + 1),
            fraction,
            marker=".",
            label=tokenizer.upper(),
            color=COLORS[tokenizer],
        )
    axis.set(
        title="Top-50 token frequency distribution (paper Figure 7 counterpart)",
        xlabel="Token frequency rank",
        ylabel="Fraction within top-50 assignments",
        yscale="log",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    path_frequency = plot_dir / "paper_fig7_token_frequencies.png"
    _finish(fig, path_frequency)

    for record in records:
        record.pop("counts", None)
    return [path_compression, path_frequency], records


def _nano_records(state: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for target in _stage_targets(state, "nanogpt"):
        config = _load_json(target / "config.json")
        summary = _load_json(target / "summary.json")
        validation = _read_rows(target / "metrics.jsonl", "validation")
        final = summary["final_validation"]
        if not validation or validation[-1].get("step") != summary["steps"]:
            validation.append({"step": summary["steps"], **final})
        records.append({
            "stage": "nanogpt",
            "run": target.name,
            "tokenizer": config["corpus"]["tokenizer"],
            "parameter_count": summary["parameter_count"],
            "best_validation_bpb": summary["best_validation_bpb"],
            "steps": summary["steps"],
            "elapsed_sec": summary["elapsed_sec"],
            "validation": validation,
        })
    return records


def render_nanogpt(state: dict[str, Any], plot_dir: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    records = _nano_records(state)
    fig, axis = plt.subplots(figsize=(10.5, 6.5))
    for record in records:
        rows = record["validation"]
        tokenizer = record["tokenizer"]
        axis.plot(
            [row["step"] for row in rows],
            [row["bits_per_raw_byte"] for row in rows],
            marker="o",
            label=tokenizer.upper(),
            color=COLORS[tokenizer],
        )
    axis.set(
        title="18M language modeling comparison (paper Figure 5 counterpart)",
        xlabel="Optimizer step",
        ylabel="Validation bits per raw UTF-8 byte (lower is better)",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    path_curve = plot_dir / "paper_fig5_language_modeling.png"
    _finish(fig, path_curve)

    records.sort(key=lambda item: item["tokenizer"])
    fig, axis = plt.subplots(figsize=(8.5, 5.8))
    labels = [row["tokenizer"].upper() for row in records]
    values = [row["best_validation_bpb"] for row in records]
    bars = axis.bar(
        labels,
        values,
        color=[COLORS[row["tokenizer"]] for row in records],
    )
    axis.bar_label(bars, fmt="%.4f", padding=3)
    axis.set(
        title="Matched 18M nanoGPT: best validation BPB",
        ylabel="Bits per raw UTF-8 byte (lower is better)",
    )
    axis.grid(axis="y", alpha=0.25)
    path_bar = plot_dir / "nanogpt_bpb_comparison.png"
    _finish(fig, path_bar)
    for record in records:
        record.pop("validation", None)
    return [path_curve, path_bar], records


def render_tokenizer_stats(corpus_records: list[dict[str, Any]], plot_dir: Path) -> Path:
    corpus_records = sorted(corpus_records, key=lambda item: item["tokenizer"])
    labels = [row["tokenizer"].upper() for row in corpus_records]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    specs = (
        ("bytes_per_token", "Raw bytes per token", "Higher is more compressed"),
        ("fallback_byte_fraction", "Byte fallback fraction", "Fraction [0, 1]"),
        ("used_vocab_size", "Used vocabulary", "Token IDs observed"),
    )
    for axis, (metric, title, ylabel) in zip(axes, specs, strict=True):
        values = [row[metric] for row in corpus_records]
        bars = axis.bar(labels, values, color=[COLORS[row["tokenizer"]] for row in corpus_records])
        axis.bar_label(bars, fmt="%.3g", padding=3)
        axis.set(title=title, ylabel=ylabel)
        axis.grid(axis="y", alpha=0.25)
    path = plot_dir / "tokenizer_compression_stats.png"
    _finish(fig, path)
    return path


def _write_results(records: list[dict[str, Any]], plot_dir: Path) -> tuple[Path, Path]:
    serializable = []
    for record in records:
        serializable.append({
            key: value
            for key, value in record.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        })
    json_path = plot_dir / "results_summary.json"
    json_path.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    csv_path = plot_dir / "results_table.csv"
    fields = sorted({key for record in serializable for key in record})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(serializable)
    return json_path, csv_path


def render_pipeline(pipeline_dir: Path) -> dict[str, Any]:
    state_path = pipeline_dir / "state.json"
    state = _load_json(state_path)
    plot_dir = pipeline_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    state["gqvae_selection_manifest"] = str(pipeline_dir / "gqvae_selection.json")
    state["gqvae_tokenizer"] = str(pipeline_dir / "gqvae_tokenizer.json")
    paths = render_gqvae_architecture(plot_dir)
    all_records: list[dict[str, Any]] = []

    topk_paths, topk_records = render_topk(state, plot_dir)
    paths.extend(topk_paths)
    all_records.extend(topk_records)
    gq_paths, gq_records = render_gqvae(state, plot_dir)
    paths.extend(gq_paths)
    all_records.extend(gq_records)
    beta_path, beta_records = render_beta_sweep(state, plot_dir)
    paths.append(beta_path)
    all_records.extend(beta_records)
    paper_paths, corpus_records = render_paper_compression_and_frequency(state, plot_dir)
    paths.extend(paper_paths)
    all_records.extend(corpus_records)
    nano_paths, nano_records = render_nanogpt(state, plot_dir)
    paths.extend(nano_paths)
    all_records.extend(nano_records)
    paths.append(render_tokenizer_stats(corpus_records, plot_dir))
    summary_json, summary_csv = _write_results(all_records, plot_dir)
    paths.extend((summary_json, summary_csv))
    manifest = {
        "status": "completed",
        "pipeline": state["pipeline"],
        "run_date": state["run_date"],
        "artifacts": [str(path) for path in paths],
        "paper_figures": {
            "figure_1": "paper_fig1_gqvae_architecture.png",
            "figure_2": "paper_fig2_decoder_head.png",
            "figure_3": "paper_fig3_compression_vocabulary.png",
            "figure_4": None,
            "figure_4_skip_reason": "current design has only one fixed-length VQ baseline",
            "figure_5": "paper_fig5_language_modeling.png",
            "figure_6": None,
            "figure_6_skip_reason": "current design has no compression-matched BPE tokenizer",
            "figure_7": "paper_fig7_token_frequencies.png",
        },
    }
    manifest_path = plot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = render_pipeline(args.pipeline_dir.expanduser().resolve())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
