#!/usr/bin/env python3
"""Summarize and visualize ``training.train_vlm`` run directories.

Examples:
  python scripts/summarize_vlm_training.py
  python scripts/summarize_vlm_training.py --log-root logs/vlm_training --include-confusion
  python scripts/summarize_vlm_training.py --run-dir logs/vlm_training/cotnone_f8_interleaved_p6__lora_llm_attn_qv_r2_a8_lr1e-4/Qwen3-VL-2B-Instruct/20260526_092553
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
EPOCH_METRICS_RE = re.compile(r"Epoch\s+(\d+)\s+val metrics:\s+(\{.*\})")
FINAL_TEST_RE = re.compile(r"Final test results:\s+(\{.*\})")
PARAMS_RE = re.compile(
    r"Model parameters: total=(\d+), trainable=(\d+), non_trainable=(\d+)"
)
LOADED_RE = re.compile(r"Loaded train=(\d+), val=(\d+), test=(\d+)")
BEST_CKPT_RE = re.compile(r"Saved checkpoint to\s+(.+?)\s+\(([^=]+)=([0-9.]+)\)")
SEED_DIR_RE = re.compile(r"^seed_(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize train_vlm.py logs and create comparison plots."
    )
    parser.add_argument(
        "--log-root",
        default="logs/vlm_training",
        help="Root to scan recursively for train.log files.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Specific run directory containing train.log. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <log-root>/summary.",
    )
    parser.add_argument(
        "--include-confusion",
        action="store_true",
        help="Also plot one confusion matrix per run with test_results.json.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write CSV/JSON summaries.",
    )
    return parser.parse_args()


def parse_log_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")


def literal_dict(text: str) -> dict[str, Any]:
    try:
        value = ast.literal_eval(text)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def parse_namespace(text: str) -> dict[str, Any]:
    """Parse the repr of argparse.Namespace emitted by train_vlm.py."""
    start = text.find("Namespace(")
    if start < 0:
        return {}
    snippet = text[start:]
    try:
        expr = ast.parse(snippet, mode="eval").body
    except SyntaxError:
        return {}
    if not isinstance(expr, ast.Call):
        return {}
    parsed: dict[str, Any] = {}
    for keyword in expr.keywords:
        if keyword.arg is None:
            continue
        try:
            parsed[keyword.arg] = ast.literal_eval(keyword.value)
        except Exception:
            parsed[keyword.arg] = None
    return parsed


def seed_from_path(path: Path) -> str | None:
    """Return the nearest enclosing ``seed_<value>`` directory, if present."""
    for parent in path.parents:
        match = SEED_DIR_RE.match(parent.name)
        if match:
            return match.group(1)
    return None


def run_identity(train_log: Path, parsed_args: dict[str, Any]) -> dict[str, Any]:
    run_dir = train_log.parent
    model = run_dir.parent.name if run_dir.parent != run_dir else ""
    run_tag = run_dir.parent.parent.name if len(run_dir.parents) >= 2 else ""
    log_dir = parsed_args.get("log_dir")
    if log_dir:
        parts = Path(str(log_dir)).parts
        if len(parts) >= 3:
            run_tag = parts[-3]
            model = parts[-2]
    seed = seed_from_path(run_dir)
    if seed is None and parsed_args.get("random_seed") is not None:
        seed = str(parsed_args["random_seed"])
    return {
        "run_dir": str(run_dir),
        "run_tag": run_tag,
        "model": model,
        "run_timestamp": run_dir.name,
        "seed": seed,
    }


def classification_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [str(s.get("answer")) for s in samples if "answer" in s]
    y_pred = [str(s.get("pred_answer")) for s in samples if "answer" in s]
    classes = sorted(set(y_true) | set(y_pred))
    total = len(y_true)
    correct = sum(gt == pr for gt, pr in zip(y_true, y_pred))
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    confusion = {cls: Counter() for cls in classes}
    for gt, pred in zip(y_true, y_pred):
        confusion[gt][pred] += 1
        if gt == pred:
            tp[gt] += 1
        else:
            fp[pred] += 1
            fn[gt] += 1

    per_class = {}
    for cls in classes:
        precision = tp[cls] / (tp[cls] + fp[cls]) if tp[cls] + fp[cls] else 0.0
        recall = tp[cls] / (tp[cls] + fn[cls]) if tp[cls] + fn[cls] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": y_true.count(cls),
        }

    return {
        "accuracy": correct / total if total else None,
        "macro_f1": (
            sum(m["f1"] for m in per_class.values()) / len(per_class)
            if per_class
            else None
        ),
        "correct": correct,
        "total": total,
        "classes": classes,
        "per_class": per_class,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def parse_train_log(train_log: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "train_log": str(train_log),
        "args": {},
        "val_history": [],
        "best_checkpoints": [],
        "final_test": {},
        "test_metrics_from_json": {},
    }
    first_ts = None
    last_ts = None
    training_completed = False

    for line in train_log.read_text(errors="replace").splitlines():
        ts = parse_log_timestamp(line)
        if ts:
            first_ts = first_ts or ts
            last_ts = ts
        if "Arguments: Namespace(" in line:
            info["args"] = parse_namespace(line)
        elif "Model parameters:" in line:
            match = PARAMS_RE.search(line)
            if match:
                info["total_params"] = int(match.group(1))
                info["trainable_params"] = int(match.group(2))
                info["non_trainable_params"] = int(match.group(3))
                info["trainable_pct"] = (
                    100.0 * info["trainable_params"] / info["total_params"]
                    if info["total_params"]
                    else None
                )
        elif "Loaded train=" in line:
            match = LOADED_RE.search(line)
            if match:
                info["n_train"] = int(match.group(1))
                info["n_val"] = int(match.group(2))
                info["n_test"] = int(match.group(3))
        elif "Saved checkpoint to" in line:
            match = BEST_CKPT_RE.search(line)
            if match:
                info["best_checkpoints"].append(
                    {
                        "path": match.group(1),
                        "metric": match.group(2),
                        "value": float(match.group(3)),
                    }
                )
        elif "Epoch " in line and " val metrics:" in line:
            match = EPOCH_METRICS_RE.search(line)
            if match:
                row = literal_dict(match.group(2))
                row["epoch"] = int(match.group(1))
                info["val_history"].append(row)
        elif "Training completed." in line:
            training_completed = True
        elif "Final test results:" in line:
            match = FINAL_TEST_RE.search(line)
            if match:
                info["final_test"] = literal_dict(match.group(1))

    info.update(run_identity(train_log, info["args"]))
    info["start_time"] = first_ts.isoformat(sep=" ") if first_ts else ""
    info["end_time"] = last_ts.isoformat(sep=" ") if last_ts else ""
    info["duration_min"] = (
        round((last_ts - first_ts).total_seconds() / 60.0, 2)
        if first_ts and last_ts
        else None
    )
    info["epochs_logged"] = len(info["val_history"])
    info["training_completed"] = training_completed
    info["test_completed"] = bool(info["final_test"])
    info["status"] = "complete" if info["test_completed"] else "partial"

    test_results = train_log.parent / "test_results.json"
    if test_results.exists():
        try:
            samples = json.loads(test_results.read_text())
            if isinstance(samples, list):
                info["test_metrics_from_json"] = classification_metrics(samples)
        except Exception as exc:
            info["test_metrics_error"] = str(exc)
    return info


def discover_logs(log_root: Path, run_dirs: list[str]) -> list[Path]:
    logs = []
    for run_dir in run_dirs:
        path = Path(run_dir)
        train_log = path / "train.log"
        if train_log.exists():
            logs.append(train_log)
        else:
            nested_logs = list(path.rglob("train.log")) if path.is_dir() else []
            if not nested_logs:
                raise FileNotFoundError(f"Missing train.log under --run-dir {run_dir}")
            logs.extend(nested_logs)
    if not run_dirs:
        logs.extend(log_root.rglob("train.log"))
    return sorted(set(logs))


def best_val(history: list[dict[str, Any]], metric: str) -> tuple[float | None, int | None]:
    rows = [row for row in history if isinstance(row.get(metric), (int, float))]
    if not rows:
        return None, None
    best = max(rows, key=lambda row: row[metric])
    return float(best[metric]), int(best["epoch"])


def flatten_summary(run: dict[str, Any]) -> dict[str, Any]:
    args = run.get("args", {})
    monitor = args.get("monitor") or "val_acc"
    best_monitor, best_epoch = best_val(run["val_history"], monitor)
    best_acc, best_acc_epoch = best_val(run["val_history"], "val_acc")
    best_f1, best_f1_epoch = best_val(run["val_history"], "val_macro_f1")
    json_test = run.get("test_metrics_from_json") or {}
    final_test = run.get("final_test") or {}
    test_acc = final_test.get("test_acc", json_test.get("accuracy"))
    test_macro_f1 = final_test.get("test_macro_f1", json_test.get("macro_f1"))

    row = {
        "status": run["status"],
        "run_tag": run["run_tag"],
        "model": run["model"],
        "base_model": args.get("model_name") or run["model"],
        "quantized": args.get("quantize"),
        "run_timestamp": run["run_timestamp"],
        "seed": run.get("seed"),
        "ft_type": args.get("ft_type"),
        "lora_rank": args.get("lora_rank"),
        "lora_alpha": args.get("lora_alpha"),
        "lora_scaling": (args.get("lora_alpha") / args.get("lora_rank")
                         if args.get("lora_alpha") is not None and args.get("lora_rank") else None),
        "random_seed": args.get("random_seed"),
        "optimizer": "AdamW",
        "batch_size": args.get("batch_size"),
        "lr": args.get("lr"),
        "epochs_configured": args.get("epochs"),
        "epochs_logged": run["epochs_logged"],
        "num_frames": args.get("num_frames"),
        "total_pixels": args.get("total_pixels"),
        "min_pixels": args.get("min_pixels"),
        "eval_deterministic": args.get("eval_deterministic"),
        "eval_max_new_tokens": args.get("eval_max_new_tokens"),
        "cot_type": args.get("cot_type"),
        "prompt_variant": args.get("prompt_variant"),
        "interleaved_timestamps": args.get("interleaved_timestamps"),
        "context_features": args.get("context_features"),
        "sample_n": args.get("sample_n"),
        "monitor": monitor,
        "best_monitor": best_monitor,
        "best_monitor_epoch": best_epoch,
        "best_val_acc": best_acc,
        "best_val_acc_epoch": best_acc_epoch,
        "best_val_macro_f1": best_f1,
        "best_val_macro_f1_epoch": best_f1_epoch,
        "test_acc": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_correct": json_test.get("correct"),
        "test_total": json_test.get("total"),
        "total_params": run.get("total_params"),
        "trainable_params": run.get("trainable_params"),
        "non_trainable_params": run.get("non_trainable_params"),
        "trainable_pct": run.get("trainable_pct"),
        "n_train": run.get("n_train"),
        "n_val": run.get("n_val"),
        "n_test": run.get("n_test"),
        "duration_min": run.get("duration_min"),
        "start_time": run.get("start_time"),
        "end_time": run.get("end_time"),
        "run_dir": run["run_dir"],
    }
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(out_dir: Path, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [flatten_summary(run) for run in runs]
    val_rows = []
    for run in runs:
        for row in run["val_history"]:
            val_rows.append(
                {
                    "run_tag": run["run_tag"],
                    "model": run["model"],
                    "run_timestamp": run["run_timestamp"],
                    "seed": run.get("seed"),
                    "ft_type": run["args"].get("ft_type"),
                    "prompt_variant": run["args"].get("prompt_variant"),
                    "interleaved_timestamps": run["args"].get("interleaved_timestamps"),
                    "num_frames": run["args"].get("num_frames"),
                    "context_features": run["args"].get("context_features"),
                    "epoch": row.get("epoch"),
                    "val_acc": row.get("val_acc"),
                    "val_macro_f1": row.get("val_macro_f1"),
                    "run_dir": run["run_dir"],
                }
            )

    write_csv(out_dir / "summary.csv", summary_rows)
    seed_summary_rows = summarize_across_seeds(summary_rows)
    write_csv(out_dir / "seed_summary.csv", seed_summary_rows)
    write_csv(out_dir / "val_history.csv", val_rows)
    (out_dir / "summary.json").write_text(json.dumps(runs, indent=2, sort_keys=True))
    (out_dir / "seed_summary.json").write_text(
        json.dumps(seed_summary_rows, indent=2, sort_keys=True)
    )
    return summary_rows


def summarize_across_seeds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate equivalent runs across seed directories."""
    group_fields = (
        "run_tag", "model", "base_model", "quantized", "ft_type", "lora_rank",
        "lora_alpha", "batch_size", "lr", "epochs_configured", "num_frames",
        "total_pixels", "min_pixels", "eval_deterministic", "eval_max_new_tokens",
        "cot_type", "prompt_variant", "interleaved_timestamps", "context_features",
        "sample_n", "monitor",
    )
    metric_fields = (
        "best_monitor", "best_val_acc", "best_val_macro_f1", "test_acc",
        "test_macro_f1", "duration_min",
    )
    def group_value(value: Any) -> Any:
        return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, set, tuple)) else value

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(group_value(row.get(field)) for field in group_fields), []).append(row)

    aggregates = []
    for key, members in sorted(groups.items(), key=lambda item: str(item[0])):
        aggregate = dict(zip(group_fields, key))
        seeds = sorted({str(row["seed"]) for row in members if row.get("seed") is not None})
        if not seeds:
            seeds = sorted({str(row["random_seed"]) for row in members if row.get("random_seed") is not None})
        aggregate.update({
            "n_runs": len(members),
            "n_seeds": len(seeds) if seeds else 1,
            "seeds": ",".join(seeds) if seeds else "single-layout",
        })
        for metric in metric_fields:
            values = [float(row[metric]) for row in members if isinstance(row.get(metric), (int, float))]
            aggregate[f"{metric}_mean"] = sum(values) / len(values) if values else None
            aggregate[f"{metric}_std"] = (
                math.sqrt(sum((value - aggregate[f"{metric}_mean"]) ** 2 for value in values) / (len(values) - 1))
                if len(values) > 1 else 0.0 if values else None
            )
        aggregates.append(aggregate)
    return aggregates


def import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def cleanup_legacy_plot_files(out_dir: Path) -> None:
    legacy_patterns = [
        "val_acc_curves*.png",
        "val_macro_f1_curves*.png",
        "best_val_acc_bars*.png",
        "best_val_macro_f1_bars*.png",
        "test_acc_bars*.png",
        "test_macro_f1_bars*.png",
        "best_val_metrics_bars_*.png",
        "test_metrics_bars_*.png",
    ]
    for pattern in legacy_patterns:
        for path in out_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass


def input_mode_label(row: dict[str, Any]) -> str:
    return "interleaved" if row.get("interleaved_timestamps") else "nointerleave"


def context_feature_label(row: dict[str, Any]) -> str | None:
    value = context_feature_value(row)
    return f"ctx{value}" if value else None


def context_feature_value(row: dict[str, Any]) -> str | None:
    value = row.get("context_features")
    if value is None or value == "":
        run_tag = str(row.get("run_tag") or row.get("run_timestamp") or "")
        match = re.search(r"(?:^|_)ctx(.+?)(?:_cfps|_preface|__|$)", run_tag)
        value = match.group(1) if match else None
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        parts = [str(part).strip() for part in value if str(part).strip()]
    else:
        text = str(value).strip()
        parts = [part.strip() for part in re.split(r"[,+]", text) if part.strip()]
    return "+".join(parts) if parts else None


def num_frames_value(row: dict[str, Any]) -> int | None:
    value = row.get("num_frames")
    if value not in (None, ""):
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    run_tag = str(row.get("run_tag") or row.get("run_timestamp") or "")
    match = re.search(r"(?:^|_)f(\d+)(?:_|$)", run_tag)
    return int(match.group(1)) if match else None


def plot_condition_label(row: dict[str, Any]) -> str:
    parts = [str(row.get("prompt_variant") or "unknown"), input_mode_label(row)]
    num_frames = num_frames_value(row)
    if num_frames is not None:
        parts.append(f"f{num_frames}")
    context = context_feature_value(row)
    if context:
        parts.append(f"ctx{context}")
    return " ".join(parts)


def prompt_input_label(row: dict[str, Any]) -> str:
    return plot_condition_label(row)


def clean_label(row: dict[str, Any]) -> str:
    label = row.get("ft_type") or row.get("run_tag") or row.get("run_timestamp")
    rank = row.get("lora_rank")
    alpha = row.get("lora_alpha")
    parts = [str(label)]
    if rank is not None and alpha is not None:
        parts.append(f"r{rank} a{alpha}")
    parts.append(prompt_input_label(row))
    return " ".join(parts)


def safe_slug(value: Any) -> str:
    text = str(value or "unknown")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def prompt_variant_groups(rows_or_runs: list[dict[str, Any]], *, runs: bool = False) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in rows_or_runs:
        if runs:
            prompt = (item.get("args") or {}).get("prompt_variant") or "unknown"
        else:
            prompt = item.get("prompt_variant") or "unknown"
        groups.setdefault(str(prompt), []).append(item)
    return dict(sorted(groups.items()))


def plot_val_curves(out_dir: Path, runs: list[dict[str, Any]]) -> None:
    plt = import_matplotlib()

    def _plot(subset: list[dict[str, Any]], suffix: str, title_suffix: str) -> None:
        if not subset:
            return
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        plotted = False
        specs = [
            ("val_acc", "Validation accuracy", axes[0]),
            ("val_macro_f1", "Validation macro F1", axes[1]),
        ]
        for metric, ylabel, ax in specs:
            for run in subset:
                history = run["val_history"]
                xs = [row["epoch"] for row in history if metric in row]
                ys = [row[metric] for row in history if metric in row]
                if not xs:
                    continue
                plotted = True
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=clean_label(flatten_summary(run)))
            ax.set_ylabel(ylabel)
            ax.set_ylim(0.4, 0.9)
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("Epoch")
        axes[0].set_title(f"Validation accuracy and macro F1{title_suffix}")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, fontsize=8, loc="upper center", ncol=2)
            fig.subplots_adjust(top=0.82)
        if plotted:
            fig.tight_layout()
            fig.savefig(out_dir / f"val_metrics_curves{suffix}.png", dpi=180)
        plt.close(fig)

    _plot(runs, "", "")
    for prompt, subset in prompt_variant_groups(runs, runs=True).items():
        _plot(subset, f"_{safe_slug(prompt)}", f" ({prompt})")


def adapter_config_label(row: dict[str, Any]) -> str:
    label = row.get("ft_type") or row.get("run_tag") or row.get("run_timestamp")
    rank = row.get("lora_rank")
    alpha = row.get("lora_alpha")
    if rank is not None and alpha is not None:
        return f"{label} r{rank} a{alpha}"
    return str(label)


def adapter_config_key(row: dict[str, Any]) -> tuple[str, Any, Any]:
    return (
        str(row.get("ft_type") or row.get("run_tag") or row.get("run_timestamp")),
        row.get("lora_rank"),
        row.get("lora_alpha"),
    )


def best_row_for_metric(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get(metric) is not None]
    if not candidates:
        return rows[0] if rows else None
    return max(candidates, key=lambda row: row.get(metric) or -math.inf)


def plot_metric_pair_bars(plt, out_dir: Path, rows: list[dict[str, Any]],
                          metrics: tuple[tuple[str, str], tuple[str, str]],
                          filename: str, title: str) -> None:
    if not rows:
        return
    if not any(row.get(metrics[0][0]) is not None or row.get(metrics[1][0]) is not None for row in rows):
        return

    config_keys = sorted({adapter_config_key(row) for row in rows})
    legend_groups = sorted({prompt_input_label(row) for row in rows})
    by_config_group: dict[tuple[tuple[str, Any, Any], str], list[dict[str, Any]]] = {}
    for row in rows:
        group = prompt_input_label(row)
        by_config_group.setdefault((adapter_config_key(row), group), []).append(row)

    config_labels = []
    for key in config_keys:
        sample = next(row for row in rows if adapter_config_key(row) == key)
        config_labels.append(adapter_config_label(sample))

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(config_keys) * 1.35), 8), sharex=True)
    xs = list(range(len(config_keys)))
    width = min(0.8 / max(len(legend_groups), 1), 0.36)
    offsets = [((i - (len(legend_groups) - 1) / 2) * width) for i in range(len(legend_groups))]

    for ax, (metric, ylabel) in zip(axes, metrics):
        for group, offset in zip(legend_groups, offsets):
            values = []
            for key in config_keys:
                chosen = best_row_for_metric(by_config_group.get((key, group), []), metric)
                values.append((chosen or {}).get(metric) or 0)
            ax.bar([x + offset for x in xs], values, width=width, label=group)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.7, 0.9)
        ax.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xticks(xs, config_labels, rotation=35, ha="right")
    axes[0].set_title(title)
    axes[0].legend(title="condition", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def unique_rows_by_adapter_config(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, Any, Any], dict[str, Any]] = {}
    for row in rows:
        key = adapter_config_key(row)
        if key not in unique:
            unique[key] = row
    return [unique[key] for key in sorted(unique)]


def context_comparison_key(row: dict[str, Any]) -> tuple[tuple[str, Any, Any], str, str, bool]:
    return (
        adapter_config_key(row),
        str(row.get("prompt_variant") or "unknown"),
        str(row.get("cot_type") or "unknown"),
        bool(row.get("interleaved_timestamps")),
    )


def context_sort_key(context: str) -> tuple[int, str]:
    """Sort key used for context feature variants.

    NOTE: For the frame-4 context comparison plot, we want a stable semantic order
    instead of relying on lexical sorting.
    """
    order = {
        "none": 0,
        "vehicle_motion": 1,
        "gaze_direction": 2,
        "gaze_on_screen_ratio": 3,
        "ego_motion": 4,
        "ego_motion+vehicle_motion": 5,
        "ego_motion+gaze_direction": 6,
        "ego_motion+gaze_on_screen_ratio": 7,
    }
    return (order.get(context, 999), context)


CONTEXT_DISPLAY_LABELS = {
    "none": "none",
    # "vehicle_motion": "vehicle motion",
    # "gaze_direction": "gaze direction",
    # "gaze_on_screen_ratio": "gaze on screen",
    "ego_motion": "ego motion",
    "ego_motion+vehicle_motion": "ego,vehicle motion",
    "ego_motion+gaze_direction": "ego,gaze direction",
    "ego_motion+gaze_on_screen_ratio": "ego,gaze on screen",
}


def context_display_label(context: str) -> str:
    if context in CONTEXT_DISPLAY_LABELS:
        return CONTEXT_DISPLAY_LABELS[context]
    # fallback: reasonable humanization for any unexpected tokens
    return context.replace("+", " + ").replace("_", " ")


def plot_frame4_context_comparison(plt, out_dir: Path, rows: list[dict[str, Any]]) -> None:
    f4_rows = [
        row for row in rows
        if num_frames_value(row) == 4
        and context_feature_value(row) in CONTEXT_DISPLAY_LABELS
        and any(row.get(metric) is not None for metric in (
            "best_val_acc", "best_val_macro_f1", "test_acc", "test_macro_f1"
        ))
    ]
    contexts = {context_feature_value(row) for row in f4_rows}
    if len(contexts) < 2:
        return

    group_counts = Counter(context_comparison_key(row) for row in f4_rows)
    common_group = max(group_counts, key=group_counts.get)
    comparable_rows = [row for row in f4_rows if context_comparison_key(row) == common_group]
    if len({context_feature_value(row) for row in comparable_rows}) < 2:
        comparable_rows = f4_rows

    by_context: dict[str, list[dict[str, Any]]] = {}
    for row in comparable_rows:
        context = context_feature_value(row)
        if context:
            by_context.setdefault(context, []).append(row)

    contexts_sorted = sorted(by_context, key=context_sort_key)
    if len(contexts_sorted) < 2:
        return

    def mean_and_std(context: str, metric: str) -> tuple[float | None, float | None]:
        values = [
            float(row[metric])
            for row in by_context[context]
            if isinstance(row.get(metric), (int, float))
        ]
        if not values:
            return None, None
        mean = sum(values) / len(values)
        # Sample standard deviation: a single seed has no estimated spread.
        std = (
            math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
            if len(values) > 1
            else 0.0
        )
        return mean, std

    labels = [context_display_label(context) for context in contexts_sorted]
    metrics = [
        ("best_val_acc", "test_acc", "Accuracy"),
        ("best_val_macro_f1", "test_macro_f1", "Macro F1"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(max(5, len(contexts_sorted) * 1.2), 6), sharex=True)
    xs = list(range(len(contexts_sorted)))
    width = 0.25
    for ax in axes:
        ax.set_axisbelow(True)

    palette = plt.get_cmap("Set2").colors
    val_color = palette[0]
    test_color = palette[1]

    # Keep bars and their standard deviations for correctly placed annotations.
    bar_pairs: list[tuple[Any, Any, list[float], list[float]]] = []

    for ax, (val_metric, test_metric, ylabel) in zip(axes, metrics):
        val_stats = [mean_and_std(context, val_metric) for context in contexts_sorted]
        test_stats = [mean_and_std(context, test_metric) for context in contexts_sorted]
        val_values = [mean or 0 for mean, _ in val_stats]
        test_values = [mean or 0 for mean, _ in test_stats]
        val_stds = [std or 0 for _, std in val_stats]
        test_stds = [std or 0 for _, std in test_stats]
        val_bars = ax.bar(
            [x - width / 2 for x in xs],
            val_values,
            width=width,
            label="Best val",
            color=val_color,
            yerr=val_stds,
            capsize=4,
            error_kw={"ecolor": "black", "elinewidth": 1, "capthick": 1},
        )
        test_bars = ax.bar(
            [x + width / 2 for x in xs],
            test_values,
            width=width,
            label="Test",
            color=test_color,
            yerr=test_stds,
            capsize=4,
            error_kw={"ecolor": "black", "elinewidth": 1, "capthick": 1},
        )
        bar_pairs.append((val_bars, test_bars, val_stds, test_stds))
        for context, bar in zip(contexts_sorted, test_bars):
            if "gaze" in context:
                bar.set_hatch("//")

        ego_val, _ = mean_and_std("ego_motion", val_metric)
        ego_test, _ = mean_and_std("ego_motion", test_metric)
        # if ego_val is not None:
        #     ax.axhline(
        #         ego_val,
        #         color=val_bars.patches[0].get_facecolor(),
        #         linestyle="--",
        #         linewidth=1.5,
        #         alpha=0.9,
        #         label="Best val (ego motion only)",
        #     )
        # if ego_test is not None:
        #     ax.axhline(
        #         ego_test,
        #         color=test_bars.patches[0].get_facecolor(),
        #         linestyle="--",
        #         linewidth=1.5,
        #         alpha=0.9,
        #         label="Test (ego motion only)",
        #     )

        ax.set_ylabel(ylabel)
        baseline_values = [value for value in (ego_val, ego_test) if value is not None]
        bar_extents = [
            value + std
            for value, std in zip(val_values + test_values, val_stds + test_stds)
        ]
        nonzero = [value for value in val_values + test_values + baseline_values if value]
        if nonzero:
            lower = max(0.0, min(nonzero) - 0.04)
            upper = min(1.0, max(bar_extents + baseline_values) + 0.04)
            ax.set_ylim(lower, upper)
        ax.grid(True, axis="y", alpha=0.25)

    axes[-1].set_xticks(xs, labels, rotation=20, ha="right")
    axes[0].set_xticks(xs, labels, rotation=20, ha="right")
    axes[0].legend(fontsize=8)
    # axes[0].set_title("Frame-4 context comparison: validation vs test")

    # adapter_key, prompt, cot_type, interleaved = common_group
    # sample = comparable_rows[0]
    # setup = f"{adapter_config_label(sample)}; {prompt}; {cot_type}; {input_mode_label(sample)}"
    # fig.text(0.01, 0.01, f"Setup: {setup}", fontsize=8, color="0.35")
    # fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.tight_layout()
    fig.savefig(out_dir / "context_val_test_comparison_xtick.png", dpi=180)
    print('Save to ', out_dir / "context_val_test_comparison_xtick.png")
    plt.close(fig)


def plot_rank_alpha_comparison(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Plot heatmaps comparing different LoRA rank and alpha values for the same FT architecture.
    
    This creates a grid of heatmaps, one per FT type, showing test accuracy across rank/alpha combinations.
    """
    plt = import_matplotlib()
    
    # Group rows by FT type
    by_ft_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ft_type = row.get("ft_type") or "unknown"
        by_ft_type.setdefault(ft_type, []).append(row)
    
    # Filter to only FT types with multiple rank/alpha combinations
    ft_types_with_variants = {
        ft_type: variants
        for ft_type, variants in by_ft_type.items()
        if len({(row.get("lora_rank"), row.get("lora_alpha")) for row in variants if row.get("lora_rank") is not None}) >= 2
    }
    
    if not ft_types_with_variants:
        return
    
    n_ft_types = len(ft_types_with_variants)
    n_cols = min(2, n_ft_types)
    n_rows = (n_ft_types + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    if n_ft_types == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for ax, (ft_type, variants) in zip(axes, sorted(ft_types_with_variants.items())):
        # Collect all unique ranks and alphas
        all_ranks = sorted({row.get("lora_rank") for row in variants if row.get("lora_rank") is not None})
        all_alphas = sorted({row.get("lora_alpha") for row in variants if row.get("lora_alpha") is not None})
        
        if not all_ranks or not all_alphas:
            continue
        
        # Build matrix: rows=rank, cols=alpha, values=test_acc
        matrix = []
        for rank in all_ranks:
            row_data = []
            for alpha in all_alphas:
                matching = [row for row in variants if row.get("lora_rank") == rank and row.get("lora_alpha") == alpha]
                if not matching:
                    row_data.append(None)
                else:
                    # Use the run with best test_acc for this rank/alpha pair
                    best = max(matching, key=lambda row: row.get("test_acc") or row.get("best_val_acc") or -math.inf)
                    row_data.append(best.get("test_acc"))
            matrix.append(row_data)
        
        # Convert to numpy array for heatmap, replacing None with NaN
        import numpy as np
        matrix_array = np.array(matrix, dtype=float)
        
        # Plot heatmap
        im = ax.imshow(matrix_array, cmap="RdYlGn", aspect="auto", vmin=0.65, vmax=0.85)
        
        # Set ticks and labels
        ax.set_xticks(range(len(all_alphas)))
        ax.set_yticks(range(len(all_ranks)))
        ax.set_xticklabels([f"α={a}" for a in all_alphas])
        ax.set_yticklabels([f"r={r}" for r in all_ranks])
        ax.set_xlabel("LoRA Alpha")
        ax.set_ylabel("LoRA Rank")
        ax.set_title(f"{ft_type}")
        
        # Add text annotations
        for i, rank in enumerate(all_ranks):
            for j, alpha in enumerate(all_alphas):
                value = matrix_array[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f"{value:.3f}", ha="center", va="center", 
                                   color="black", fontsize=10, fontweight="bold")
                                #  color="white" if value < 0.75 else "black", fontsize=10, fontweight="bold")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Test Accuracy")
    
    # Hide unused subplots
    for ax in axes[n_ft_types:]:
        ax.set_visible(False)
    
    fig.suptitle("Test Accuracy: LoRA Rank vs Alpha Heatmaps by Architecture", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "rank_alpha_comparison.png", dpi=180)
    plt.close(fig)
    print('Save to ', out_dir / "rank_alpha_comparison.png")


def plot_summary_bars(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    plt = import_matplotlib()

    def _sorted(subset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            subset,
            key=lambda row: (
                adapter_config_key(row),
                prompt_input_label(row),
                row.get("test_acc") is None,
                -(row.get("test_acc") or row.get("best_val_acc") or -math.inf),
            ),
        )

    rows = _sorted(rows)
    plot_metric_pair_bars(
        plt, out_dir, rows,
        (("best_val_acc", "Accuracy"), ("best_val_macro_f1", "Macro F1")),
        "best_val_metrics_bars.png",
        "Best validation accuracy and macro F1 by adapter config",
    )
    plot_metric_pair_bars(
        plt, out_dir, rows,
        (("test_acc", "Accuracy"), ("test_macro_f1", "Macro F1")),
        "test_metrics_bars.png",
        "Test accuracy and macro F1 by adapter config",
    )
    plot_frame4_context_comparison(plt, out_dir, rows)
    plot_rank_alpha_comparison(out_dir, rows)

    trainable_rows = unique_rows_by_adapter_config(rows)
    labels = [adapter_config_label(row) for row in trainable_rows]
    if any(row.get("trainable_params") is not None for row in trainable_rows):
        plt.figure(figsize=(max(8, len(trainable_rows) * 1.2), 5))
        xs = range(len(trainable_rows))
        plt.bar(xs, [row.get("trainable_params") or 0 for row in trainable_rows])
        plt.xticks(xs, labels, rotation=35, ha="right")
        plt.ylabel("Trainable parameters")
        plt.title("Trainable Parameters by Adapter Config")
        plt.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "trainable_params_bars.png", dpi=180)
        plt.close()

def plot_confusion_matrices(out_dir: Path, runs: list[dict[str, Any]]) -> None:
    plt = import_matplotlib()
    for run in runs:
        metrics = run.get("test_metrics_from_json") or {}
        classes = metrics.get("classes") or []
        confusion = metrics.get("confusion") or {}
        if not classes:
            continue
        matrix = [
            [confusion.get(gt, {}).get(pred, 0) for pred in classes]
            for gt in classes
        ]
        plt.figure(figsize=(max(5, len(classes) * 1.2), max(4, len(classes))))
        ax = plt.gca()
        image = ax.imshow(matrix, cmap="Blues")
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(classes)), labels=classes, rotation=35, ha="right")
        ax.set_yticks(range(len(classes)), labels=classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground truth")
        ax.set_title(clean_label(flatten_summary(run)))
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha="center", va="center", fontsize=10)
        plt.tight_layout()
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run["run_tag"])
        plt.savefig(out_dir / f"confusion_{safe_name}_{run['run_timestamp']}.png", dpi=180)
        plt.close()


def print_console_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    print(f"Wrote summary to {out_dir}")
    if not rows:
        print("No runs found.")
        return
    key = lambda row: row.get("test_acc") or row.get("best_val_acc") or -math.inf
    for row in sorted(rows, key=key, reverse=True):
        print(
            "{status:8s} {ft_type:24s} best_val_acc={best_val_acc} "
            "test_acc={test_acc} epochs={epochs_logged}/{epochs_configured} {run_timestamp}".format(
                status=str(row.get("status")),
                ft_type=str(row.get("ft_type"))[:24],
                best_val_acc=fmt(row.get("best_val_acc")),
                test_acc=fmt(row.get("test_acc")),
                epochs_logged=row.get("epochs_logged"),
                epochs_configured=row.get("epochs_configured"),
                run_timestamp=row.get("run_timestamp"),
            )
        )


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    log_root = Path(args.log_root)
    out_dir = Path(args.out_dir) if args.out_dir else log_root / "summary"
    train_logs = discover_logs(log_root, args.run_dir)
    runs = [parse_train_log(path) for path in train_logs]
    rows = write_outputs(out_dir, runs)
    if not args.no_plots:
        try:
            cleanup_legacy_plot_files(out_dir)
            plot_val_curves(out_dir, runs)
            plot_summary_bars(out_dir, rows)
            if args.include_confusion:
                plot_confusion_matrices(out_dir, runs)
        except ImportError as exc:
            print(f"Plotting skipped because matplotlib is unavailable: {exc}")
    print_console_summary(rows, out_dir)


if __name__ == "__main__":
    main()
