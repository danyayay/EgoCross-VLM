#!/usr/bin/env python3
"""Aggregate repeated DL/VLM runs and perform paired model comparisons.

Input predictions use the schema already emitted by VLM evaluation and now by
``training.train_dl``: ``video_id``, ``answer``, and ``pred_answer``.

Examples:
  python -m utils.analyze_repeated_runs \
    --discover video=logs/repeated_runs/train_dl_video \
    --discover gaze_screen=logs/repeated_runs/train_dl_gaze_screen \
    --discover gaze_orientation=logs/repeated_runs/train_dl_gaze_orientation \
    --compare video:gaze_screen --compare video:gaze_orientation

  python -m utils.analyze_repeated_runs \
    --run qwen_r2:42:/path/to/test_results.json \
    --run qwen_r4:42:/path/to/test_results.json \
    --compare qwen_r2:qwen_r4
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, t

LABELS = ("cross", "yield")
SEED_RE = re.compile(r"(?:seed[_-]?)(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], metavar="MODEL:SEED:PATH")
    parser.add_argument("--discover", action="append", default=[], metavar="MODEL=ROOT")
    parser.add_argument("--compare", action="append", default=[], metavar="MODEL_A:MODEL_B")
    parser.add_argument("--out-dir", default="logs/repeated_runs/analysis")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--allow-incomplete-seeds", action="store_true")
    return parser.parse_args()


def normalize_label(value: Any) -> str:
    label = str(value).strip().lower()
    aliases = {"0": "cross", "1": "yield", "ross": "cross"}
    return aliases.get(label, label)


def load_predictions(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list: {path}")
    rows = []
    seen: set[str] = set()
    for index, sample in enumerate(raw):
        sample_id = str(sample.get("video_id", index))
        if sample_id in seen:
            raise ValueError(f"Duplicate video_id {sample_id!r} in {path}")
        seen.add(sample_id)
        rows.append({
            "video_id": sample_id,
            "answer": normalize_label(sample.get("answer")),
            "pred_answer": normalize_label(sample.get("pred_answer")),
        })
    if not rows:
        raise ValueError(f"No predictions in {path}")
    return rows


def parse_seed(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = SEED_RE.search(part)
        if match:
            return int(match.group(1))
    metrics_path = path.parent / "test_metrics.json"
    if metrics_path.exists():
        try:
            return int(json.loads(metrics_path.read_text())["seed"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return None


def collect_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    runs = []
    for spec in args.run:
        try:
            model, seed_text, path_text = spec.split(":", 2)
        except ValueError as exc:
            raise ValueError(f"Invalid --run {spec!r}; expected MODEL:SEED:PATH") from exc
        path = Path(path_text)
        runs.append({"model": model, "seed": int(seed_text), "path": path,
                     "predictions": load_predictions(path)})
    for spec in args.discover:
        if "=" not in spec:
            raise ValueError(f"Invalid --discover {spec!r}; expected MODEL=ROOT")
        model, root_text = spec.split("=", 1)
        paths = sorted(Path(root_text).rglob("test_results.json"))
        if not paths:
            raise FileNotFoundError(f"No test_results.json beneath {root_text}")
        for path in paths:
            seed = parse_seed(path)
            if seed is None:
                raise ValueError(f"Could not infer seed for {path}; use --run")
            runs.append({"model": model, "seed": seed, "path": path,
                         "predictions": load_predictions(path)})
    keys = [(run["model"], run["seed"]) for run in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("Each model/seed pair must occur exactly once")
    if not runs:
        raise ValueError("Provide at least one --run or --discover")
    return runs


def metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    y_true = [row["answer"] for row in rows]
    y_pred = [row["pred_answer"] for row in rows]
    labels = list(LABELS)
    extras = sorted((set(y_true) | set(y_pred)) - set(labels))
    labels.extend(extras)
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    lookup = {label: i for i, label in enumerate(labels)}
    for truth, pred in zip(y_true, y_pred):
        cm[lookup[truth], lookup[pred]] += 1
    f1s = []
    for i in range(len(labels)):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return {
        "accuracy": float(np.trace(cm) / cm.sum()),
        "macro_f1": float(np.mean(f1s)),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }


def mean_ci(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) < 2 or np.all(array == array[0]):
        exact = float(array[0])
        return exact, 0.0, exact, exact
    std = float(array.std(ddof=1))
    half = float(t.ppf(0.975, len(array) - 1) * std / math.sqrt(len(array)))
    return mean, std, mean - half, mean + half


def aligned(a: dict[str, Any], b: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    a_map = {row["video_id"]: row for row in a["predictions"]}
    b_map = {row["video_id"]: row for row in b["predictions"]}
    if set(a_map) != set(b_map):
        raise ValueError(f"Prediction IDs differ for seed {a['seed']}: {a['model']} vs {b['model']}")
    ids = sorted(a_map)
    left = [a_map[key] for key in ids]
    right = [b_map[key] for key in ids]
    for x, y in zip(left, right):
        if x["answer"] != y["answer"]:
            raise ValueError(f"Ground truth differs for video_id {x['video_id']}")
    return left, right


def paired_bootstrap(a_rows: list[dict[str, str]], b_rows: list[dict[str, str]],
                     iterations: int, seed: int) -> dict[str, float]:
    # Stratification preserves the observed cross/yield class counts.
    strata: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(a_rows):
        strata[row["answer"]].append(index)
    rng = np.random.default_rng(seed)
    observed = metrics(b_rows)["macro_f1"] - metrics(a_rows)["macro_f1"]
    differences = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        indices = []
        for group in strata.values():
            indices.extend(rng.choice(group, size=len(group), replace=True).tolist())
        differences[iteration] = (
            metrics([b_rows[i] for i in indices])["macro_f1"]
            - metrics([a_rows[i] for i in indices])["macro_f1"]
        )
    p_value = 2 * min(float(np.mean(differences <= 0)), float(np.mean(differences >= 0)))
    return {
        "macro_f1_delta_b_minus_a": float(observed),
        "bootstrap_ci_low": float(np.quantile(differences, 0.025)),
        "bootstrap_ci_high": float(np.quantile(differences, 0.975)),
        "bootstrap_p": min(1.0, p_value),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion(out_dir: Path, model: str, rows: list[dict[str, str]]) -> None:
    result = metrics(rows)
    cm = np.asarray(result["confusion_matrix"], dtype=int)
    denominator = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, denominator, out=np.zeros_like(cm, dtype=float), where=denominator != 0)
    payload = {"labels": result["labels"], "rows": "true", "columns": "predicted",
               "counts": cm.tolist(), "row_normalized": normalized.tolist()}
    (out_dir / f"confusion_{model}.json").write_text(json.dumps(payload, indent=2))
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        for values, suffix, fmt in ((cm, "counts", "d"), (normalized, "normalized", ".2f")):
            fig, ax = plt.subplots(figsize=(4.5, 4))
            sns.heatmap(values, annot=True, fmt=fmt, cmap="Blues", cbar=False,
                        xticklabels=result["labels"], yticklabels=result["labels"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            fig.tight_layout()
            fig.savefig(out_dir / f"confusion_{model}_{suffix}.png", dpi=300)
            plt.close(fig)
    except ImportError:
        pass


def main() -> None:
    args = parse_args()
    runs = collect_runs(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_rows = []
    for run in runs:
        result = metrics(run["predictions"])
        run["metrics"] = result
        by_model[run["model"]].append(run)
        save_confusion(out_dir, f"{run['model']}_seed_{run['seed']}", run["predictions"])
        run_rows.append({"model": run["model"], "seed": run["seed"],
                         "accuracy": result["accuracy"], "macro_f1": result["macro_f1"],
                         "n": len(run["predictions"]), "path": str(run["path"])})
    write_csv(out_dir / "runs.csv", run_rows)

    summary_rows = []
    for model, model_runs in sorted(by_model.items()):
        model_runs.sort(key=lambda run: run["seed"])
        row: dict[str, Any] = {"model": model, "n_runs": len(model_runs),
                               "seeds": ";".join(str(run["seed"]) for run in model_runs)}
        for metric_name in ("accuracy", "macro_f1"):
            values = [run["metrics"][metric_name] for run in model_runs]
            mean, std, low, high = mean_ci(values)
            row.update({f"{metric_name}_mean": mean, f"{metric_name}_std": std,
                        f"{metric_name}_ci_low": low, f"{metric_name}_ci_high": high})
        summary_rows.append(row)
        pooled = []
        for run in model_runs:
            pooled.extend(run["predictions"])
        save_confusion(out_dir, f"{model}_pooled", pooled)
    write_csv(out_dir / "summary.csv", summary_rows)

    comparisons = []
    for spec in args.compare:
        model_a, model_b = spec.split(":", 1)
        a_by_seed = {run["seed"]: run for run in by_model[model_a]}
        b_by_seed = {run["seed"]: run for run in by_model[model_b]}
        common = sorted(set(a_by_seed) & set(b_by_seed))
        if not args.allow_incomplete_seeds and set(a_by_seed) != set(b_by_seed):
            raise ValueError(f"Seed sets differ for {model_a} and {model_b}")
        pooled_a, pooled_b = [], []
        discordant_a = discordant_b = 0
        for seed in common:
            left, right = aligned(a_by_seed[seed], b_by_seed[seed])
            pooled_a.extend(left)
            pooled_b.extend(right)
            for x, y in zip(left, right):
                a_correct = x["pred_answer"] == x["answer"]
                b_correct = y["pred_answer"] == y["answer"]
                discordant_a += int(a_correct and not b_correct)
                discordant_b += int(b_correct and not a_correct)
        discordant = discordant_a + discordant_b
        mcnemar_p = (binomtest(min(discordant_a, discordant_b), discordant, 0.5).pvalue
                     if discordant else 1.0)
        bootstrap = paired_bootstrap(pooled_a, pooled_b, args.bootstrap, args.bootstrap_seed)
        comparisons.append({"model_a": model_a, "model_b": model_b,
                            "seeds": ";".join(map(str, common)),
                            "n_paired": len(pooled_a),
                            "accuracy_delta_b_minus_a": metrics(pooled_b)["accuracy"] - metrics(pooled_a)["accuracy"],
                            "mcnemar_a_only_correct": discordant_a,
                            "mcnemar_b_only_correct": discordant_b,
                            "mcnemar_p": float(mcnemar_p), **bootstrap})
    if comparisons:
        for source, target in (("mcnemar_p", "mcnemar_p_holm"),
                               ("bootstrap_p", "bootstrap_p_holm")):
            adjusted = holm_adjust([row[source] for row in comparisons])
            for row, value in zip(comparisons, adjusted):
                row[target] = value
        write_csv(out_dir / "significance.csv", comparisons)

    payload = {"runs": run_rows, "summary": summary_rows, "comparisons": comparisons,
               "notes": {"confidence_interval": "two-sided 95% Student-t interval across seeds",
                         "macro_f1_test": "paired label-stratified bootstrap",
                         "accuracy_test": "exact McNemar test; pooled matched seed/sample predictions",
                         "multiple_testing": "Holm correction across requested comparisons"}}
    (out_dir / "analysis.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote repeated-run analysis to {out_dir}")


if __name__ == "__main__":
    main()
