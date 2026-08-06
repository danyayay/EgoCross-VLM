#!/usr/bin/env python3
"""Compare MMBench reports from a base model vs. a fine-tuned (LoRA) model.

Single-seed usage:
    python -m utils.analyze_mmbench_results \
        --base_report logs/mmbench_eval/Qwen3-VL-2B-Instruct/.../base_..._report.json \
        --finetuned_report logs/mmbench_eval/Qwen3-VL-2B-Instruct/.../<adapter>_report.json \
        --out_dir results/mmbench_compare

Multi-seed usage (aggregates mean +/- std of the delta across seeds):
    python -m utils.analyze_mmbench_results \
        --base_reports seed42/base_..._report.json seed43/base_..._report.json seed44/base_..._report.json \
        --finetuned_reports seed42/<adapter>_report.json seed43/<adapter>_report.json seed44/<adapter>_report.json \
        --out_dir results/mmbench_compare

Generalization-robustness usage (base vs. 2 fine-tuned models, each trained with
3 model seeds and evaluated on 3 randomly-sampled MMBench subsets):
    python -m utils.analyze_mmbench_results generalization \
        --log_dir logs/mmbench_eval \
        --category_level l2-category --samples_per_category 100 \
        --subset_seeds 42 43 44 \
        --model1_name "Ego-only" \
        --model1_adapters logs/vlm_training/seed_42/.../20260528_034756 \
                           logs/vlm_training/seed_43/.../20260723_234732 \
                           logs/vlm_training/seed_44/.../20260724_162218 \
        --model2_name "Ego+Gaze" \
        --model2_adapters logs/vlm_training_dot/seed_42/.../20260529_000601_ctx... \
                           logs/vlm_training_dot/seed_43/.../20260717_223227_ctx... \
                           logs/vlm_training_dot/seed_44/.../20260719_000202_ctx... \
        --out_dir results/mmbench_generalization
"""

import argparse
import glob
import json
import os
import re
import statistics

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False

# Fixed assignment by model identity -- never re-cycled if a model drops out
# of a given plot.
_MODEL_COLORS = sns.color_palette("Set2") if PLOTTING_AVAILABLE else []

_TIMESTAMP_RE = re.compile(r"\d{8}_\d{6}")
_SEED_RE = re.compile(r"seed_(\d+)")


def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compare(base: dict, finetuned: dict, drop_threshold: float = 0.05) -> dict:
    groups = sorted(set(base["per_category"]) | set(finetuned["per_category"]))
    per_category = {}
    flagged = []
    for group in groups:
        base_stats = base["per_category"].get(group, {"accuracy": 0.0, "correct": 0, "total": 0})
        ft_stats = finetuned["per_category"].get(group, {"accuracy": 0.0, "correct": 0, "total": 0})
        delta = ft_stats["accuracy"] - base_stats["accuracy"]
        per_category[group] = {
            "base_accuracy": base_stats["accuracy"],
            "finetuned_accuracy": ft_stats["accuracy"],
            "delta": delta,
            "base_support": base_stats["total"],
            "finetuned_support": ft_stats["total"],
        }
        if delta <= -drop_threshold:
            flagged.append(group)

    overall_delta = finetuned["overall"]["accuracy"] - base["overall"]["accuracy"]
    return {
        "category_level": base.get("category_level", finetuned.get("category_level")),
        "overall": {
            "base_accuracy": base["overall"]["accuracy"],
            "finetuned_accuracy": finetuned["overall"]["accuracy"],
            "delta": overall_delta,
        },
        "per_category": per_category,
        "regressed_categories": flagged,
        "drop_threshold": drop_threshold,
    }


def _mean_std(values: list) -> tuple:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def aggregate_across_seeds(base_reports: list, finetuned_reports: list, drop_threshold: float = 0.05) -> dict:
    """Aggregate per-seed base-vs-finetuned comparisons into mean +/- std deltas.

    ``base_reports`` and ``finetuned_reports`` must be same-length, seed-aligned lists.
    """
    assert len(base_reports) == len(finetuned_reports), "Need one finetuned report per base report (per seed)"
    per_seed = [compare(b, f, drop_threshold=drop_threshold) for b, f in zip(base_reports, finetuned_reports)]

    overall_base = [r["overall"]["base_accuracy"] for r in per_seed]
    overall_ft = [r["overall"]["finetuned_accuracy"] for r in per_seed]
    overall_delta = [r["overall"]["delta"] for r in per_seed]
    base_mean, base_std = _mean_std(overall_base)
    ft_mean, ft_std = _mean_std(overall_ft)
    delta_mean, delta_std = _mean_std(overall_delta)

    groups = sorted(set().union(*(r["per_category"] for r in per_seed)))
    per_category = {}
    flagged = []
    for group in groups:
        base_accs = [r["per_category"][group]["base_accuracy"] for r in per_seed if group in r["per_category"]]
        ft_accs = [r["per_category"][group]["finetuned_accuracy"] for r in per_seed if group in r["per_category"]]
        deltas = [r["per_category"][group]["delta"] for r in per_seed if group in r["per_category"]]
        b_mean, b_std = _mean_std(base_accs)
        f_mean, f_std = _mean_std(ft_accs)
        d_mean, d_std = _mean_std(deltas)
        per_category[group] = {
            "base_accuracy_mean": b_mean, "base_accuracy_std": b_std,
            "finetuned_accuracy_mean": f_mean, "finetuned_accuracy_std": f_std,
            "delta_mean": d_mean, "delta_std": d_std,
        }
        if d_mean <= -drop_threshold:
            flagged.append(group)

    return {
        "category_level": per_seed[0]["category_level"],
        "num_seeds": len(per_seed),
        "overall": {
            "base_accuracy_mean": base_mean, "base_accuracy_std": base_std,
            "finetuned_accuracy_mean": ft_mean, "finetuned_accuracy_std": ft_std,
            "delta_mean": delta_mean, "delta_std": delta_std,
        },
        "per_category": per_category,
        "regressed_categories": flagged,
        "drop_threshold": drop_threshold,
        "per_seed": per_seed,
    }


def pretty_print_aggregate(report: dict):
    o = report["overall"]
    n = report["num_seeds"]
    print(f"Aggregated over {n} seed(s).")
    print(f"Overall accuracy: base={o['base_accuracy_mean']:.4f}+/-{o['base_accuracy_std']:.4f}  "
          f"finetuned={o['finetuned_accuracy_mean']:.4f}+/-{o['finetuned_accuracy_std']:.4f}  "
          f"delta={o['delta_mean']:+.4f}+/-{o['delta_std']:.4f}")
    print(f"\nPer-category ({report['category_level']}):")
    print(f"{'category':30} {'base':>16} {'finetuned':>16} {'delta':>16}")
    for group, stats in sorted(report["per_category"].items(), key=lambda kv: kv[1]["delta_mean"]):
        print(f"{group[:30]:30} "
              f"{stats['base_accuracy_mean']:.4f}+/-{stats['base_accuracy_std']:.4f}  "
              f"{stats['finetuned_accuracy_mean']:.4f}+/-{stats['finetuned_accuracy_std']:.4f}  "
              f"{stats['delta_mean']:+.4f}+/-{stats['delta_std']:.4f}")
    if report["regressed_categories"]:
        print(f"\nCategories that dropped by more than {report['drop_threshold']:.2f} (mean delta):")
        for group in report["regressed_categories"]:
            stats = report["per_category"][group]
            print(f"  - {group} (delta={stats['delta_mean']:+.4f}+/-{stats['delta_std']:.4f})")
    else:
        print(f"\nNo category regressed by more than {report['drop_threshold']:.2f} (mean delta).")


def plot_deltas_aggregate(report: dict, out_path: str):
    if not PLOTTING_AVAILABLE:
        print("matplotlib not available; skipping delta plot.")
        return
    groups = sorted(report["per_category"], key=lambda g: report["per_category"][g]["delta_mean"])
    means = [report["per_category"][g]["delta_mean"] for g in groups]
    stds = [report["per_category"][g]["delta_std"] for g in groups]
    colors = ["#d62728" if d < 0 else "#2ca02c" for d in means]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(max(8, len(groups) * 0.5), 5))
    plt.bar(range(len(groups)), means, yerr=stds, capsize=3, color=colors)
    plt.xticks(range(len(groups)), groups, rotation=45, ha="right")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Accuracy delta (finetuned - base), mean +/- std across seeds")
    plt.title(f"MMBench accuracy delta by {report['category_level']} (n={report['num_seeds']} seeds)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved delta plot to {out_path}")


def pretty_print(report: dict):
    o = report["overall"]
    print(f"Overall accuracy: base={o['base_accuracy']:.4f}  finetuned={o['finetuned_accuracy']:.4f}  "
          f"delta={o['delta']:+.4f}")
    print(f"\nPer-category ({report['category_level']}):")
    print(f"{'category':30} {'base':>8} {'finetuned':>10} {'delta':>8} {'n(base/ft)':>12}")
    for group, stats in sorted(report["per_category"].items(), key=lambda kv: kv[1]["delta"]):
        print(f"{group[:30]:30} {stats['base_accuracy']:.4f}  {stats['finetuned_accuracy']:9.4f}  "
              f"{stats['delta']:+.4f}  {stats['base_support']:>5}/{stats['finetuned_support']:<5}")
    if report["regressed_categories"]:
        print(f"\nCategories that dropped by more than {report['drop_threshold']:.2f}:")
        for group in report["regressed_categories"]:
            print(f"  - {group} (delta={report['per_category'][group]['delta']:+.4f})")
    else:
        print(f"\nNo category regressed by more than {report['drop_threshold']:.2f}.")


def plot_deltas(report: dict, out_path: str):
    if not PLOTTING_AVAILABLE:
        print("matplotlib not available; skipping delta plot.")
        return
    groups = sorted(report["per_category"], key=lambda g: report["per_category"][g]["delta"])
    deltas = [report["per_category"][g]["delta"] for g in groups]
    colors = ["#d62728" if d < 0 else "#2ca02c" for d in deltas]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(max(8, len(groups) * 0.5), 5))
    plt.bar(range(len(groups)), deltas, color=colors)
    plt.xticks(range(len(groups)), groups, rotation=45, ha="right")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Accuracy delta (finetuned - base)")
    plt.title(f"MMBench accuracy delta by {report['category_level']}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved delta plot to {out_path}")


# --------------------------------------------------------------------------- #
# Generalization-robustness analysis: base model (no model seed) vs. N=2
# fine-tuned models, each trained with 3 model seeds and evaluated on 3
# randomly-sampled MMBench subsets (subset seeds) -- 9 runs per fine-tuned
# model, 3 runs for the base model, per task (l2-category group).
# --------------------------------------------------------------------------- #

def _model_seed_dir(model_root: str, category_level: str, samples_per_category: int,
                     subset_seed: int, model_seed: str) -> str:
    return os.path.join(
        model_root,
        f"{category_level}_{samples_per_category}_subset-seed-{subset_seed}",
        model_seed,
    )


def find_base_report(log_dir: str, model_name: str, category_level: str,
                      samples_per_category: int, subset_seed: int) -> str:
    """Locate the base-model report for a given subset seed.

    eval_mmbench.py files base runs (no --lora_adapter) under
    model-seed-unknown/, but reports may also sit directly under the
    subset-seed directory, or under an adapter's model-seed-<N> directory if
    the base run was organized alongside a specific finetuned comparison.
    Search all three.
    """
    model_root = os.path.join(log_dir, model_name.split("/")[-1])
    subset_dir = os.path.join(model_root, f"{category_level}_{samples_per_category}_subset-seed-{subset_seed}")
    matches = sorted(
        glob.glob(os.path.join(subset_dir, "base_*_report.json"))
        + glob.glob(os.path.join(subset_dir, "model-seed-*", "base_*_report.json"))
    )
    if not matches:
        raise FileNotFoundError(f"No base MMBench report found under {subset_dir}/ (or its model-seed-* subdirs)")
    if len(matches) > 1 and len({os.path.dirname(m) for m in matches}) > 1:
        print(f"Warning: multiple base reports found for subset_seed={subset_seed}, "
              f"across different locations; using the most recent: {matches[-1]}")
    return matches[-1]


def find_adapter_report(log_dir: str, model_name: str, category_level: str,
                         samples_per_category: int, subset_seed: int, adapter_path: str) -> str:
    """Locate the report produced by evaluating ``adapter_path`` on a given subset seed.

    eval_mmbench.py names both the ``model-seed-<N>`` directory and the report
    file itself from substrings found in the adapter path: the directory from
    ``seed_<N>`` and the report filename from the first YYYYMMDD_HHMMSS-style
    timestamp. We use the same two substrings to locate the file unambiguously,
    even when multiple adapters' reports share one model-seed directory.
    """
    seed_match = _SEED_RE.search(adapter_path)
    model_seed = f"model-seed-{seed_match.group(1)}" if seed_match else "model-seed-unknown"
    ts_match = _TIMESTAMP_RE.search(adapter_path)
    if not ts_match:
        raise ValueError(f"Could not find a YYYYMMDD_HHMMSS timestamp in adapter path: {adapter_path}")
    timestamp = ts_match.group(0)

    model_root = os.path.join(log_dir, model_name.split("/")[-1])
    run_dir = _model_seed_dir(model_root, category_level, samples_per_category, subset_seed, model_seed)
    matches = sorted(glob.glob(os.path.join(run_dir, f"{timestamp}*_report.json")))
    if not matches:
        raise FileNotFoundError(
            f"No MMBench report matching timestamp {timestamp} found under {run_dir}")
    return matches[-1]


def collect_model_runs(log_dir: str, model_name: str, category_level: str,
                        samples_per_category: int, subset_seeds: list,
                        adapter_paths: list = None) -> dict:
    """Load reports for one model across all (model_seed x subset_seed) cells.

    If ``adapter_paths`` is None, this is the base model: one run per subset
    seed (no model-seed axis). Otherwise, one run per (adapter, subset_seed)
    pair -- adapter order defines the model-seed order.

    Returns {"model_seeds": [...] or None, "subset_seeds": [...],
             "reports": {(model_seed_label_or_None, subset_seed): report_dict}}
    """
    reports = {}
    if adapter_paths is None:
        for subset_seed in subset_seeds:
            path = find_base_report(log_dir, model_name, category_level, samples_per_category, subset_seed)
            reports[(None, subset_seed)] = load_report(path)
        return {"model_seeds": None, "subset_seeds": list(subset_seeds), "reports": reports}

    model_seed_labels = []
    for adapter_path in adapter_paths:
        seed_match = _SEED_RE.search(adapter_path)
        label = f"seed_{seed_match.group(1)}" if seed_match else adapter_path
        model_seed_labels.append(label)
        for subset_seed in subset_seeds:
            path = find_adapter_report(log_dir, model_name, category_level,
                                        samples_per_category, subset_seed, adapter_path)
            reports[(label, subset_seed)] = load_report(path)
    return {"model_seeds": model_seed_labels, "subset_seeds": list(subset_seeds), "reports": reports}


def _task_accuracy(report: dict, task: str) -> float:
    if task == "overall":
        return report["overall"]["accuracy"]
    stats = report["per_category"].get(task)
    return stats["accuracy"] if stats else float("nan")


def task_list(model_runs: dict) -> list:
    any_report = next(iter(model_runs["reports"].values()))
    return sorted(any_report["per_category"])


def model_task_summary(model_runs: dict, task: str) -> dict:
    """Per-task summary for one model: overall mean/std across all runs, plus
    mean/std across model seeds (each model seed's value is its mean over
    subset seeds first, so subset-seed sampling noise doesn't inflate the
    seed-to-seed spread used for the error bar).
    """
    reports = model_runs["reports"]
    all_values = [_task_accuracy(r, task) for r in reports.values()]
    overall_mean, overall_std_over_runs = _mean_std(all_values)

    if model_runs["model_seeds"] is None:
        # Base model: no model-seed axis: error bar is std across subset seeds.
        return {
            "mean": overall_mean,
            "std": overall_std_over_runs,
            "n_runs": len(all_values),
            "per_model_seed_mean": None,
        }

    per_seed_means = []
    for model_seed in model_runs["model_seeds"]:
        seed_values = [_task_accuracy(reports[(model_seed, s)], task) for s in model_runs["subset_seeds"]]
        per_seed_means.append(statistics.fmean(seed_values))
    seed_mean, seed_std = _mean_std(per_seed_means)
    return {
        "mean": overall_mean,
        "std": seed_std,
        "n_runs": len(all_values),
        "per_model_seed_mean": dict(zip(model_runs["model_seeds"], per_seed_means)),
    }


def build_generalization_report(base_runs: dict, model_runs_by_name: dict) -> dict:
    """``model_runs_by_name``: e.g. {"Ego-only": model1_runs, "Ego+Gaze": model2_runs}."""
    tasks = task_list(base_runs)
    groups = tasks + ["overall"]
    summary = {"tasks": tasks, "models": {}}

    summary["models"]["Base"] = {
        group: model_task_summary(base_runs, group) for group in groups
    }
    for name, runs in model_runs_by_name.items():
        summary["models"][name] = {group: model_task_summary(runs, group) for group in groups}

    summary["breakdown"] = {"Base": base_runs}
    summary["breakdown"].update(model_runs_by_name)
    return summary


def pretty_print_generalization(summary: dict):
    tasks = summary["tasks"]
    model_names = list(summary["models"])
    print(f"Generalization summary across {len(tasks)} task(s), models: {', '.join(model_names)}\n")
    header = f"{'task':40}" + "".join(f"{name:>18}" for name in model_names)
    print(header)
    for task in tasks:
        row = f"{task[:40]:40}"
        for name in model_names:
            stats = summary["models"][name][task]
            row += f"{stats['mean']:.4f}+/-{stats['std']:.4f}".rjust(18)
        print(row)


def plot_generalization_bars(summary: dict, out_path: str, metric_label: str = "Accuracy"):
    if not PLOTTING_AVAILABLE:
        print("matplotlib not available; skipping generalization bar chart.")
        return
    tasks = summary["tasks"]
    model_names = list(summary["models"])
    n_models = len(model_names)

    x = range(len(tasks))
    width = 0.8 / n_models

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(max(9, len(tasks) * 1.6), 5.5))
    for i, name in enumerate(model_names):
        means = [summary["models"][name][task]["mean"] for task in tasks]
        stds = [summary["models"][name][task]["std"] for task in tasks]
        offsets = [xi + (i - (n_models - 1) / 2) * width for xi in x]
        color = _MODEL_COLORS[i % len(_MODEL_COLORS)]
        plt.bar(offsets, means, width=width * 0.9, yerr=stds, capsize=3,
                label=name, color=color)

    plt.xticks(list(x), tasks, rotation=30, ha="right")
    plt.ylabel(metric_label)
    plt.ylim(0, 1.0)
    plt.title("MMBench generalization by task (mean +/- std across model seeds)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved generalization bar chart to {out_path}")


_TASK_LABELS = {
    "attribute_reasoning": "Attribute reasoning",
    "coarse_perception": "Coarse perception",
    "finegrained_perception (cross-instance)": "Finegrained perception\n(cross-instance)",
    "finegrained_perception (instance-level)": "Finegrained perception\n(instance-level)",
    "logic_reasoning": "Logic reasoning",
    "relation_reasoning": "Relation reasoning",
    "overall": "Overall",
}


def _format_task_label(task: str) -> str:
    return _TASK_LABELS.get(task, task.replace("_", " ").capitalize())


def plot_generalization_bars_ranked(summary: dict, out_path: str, metric_label: str = "Accuracy"):
    """Grouped bar chart with an 'Overall' group (leftmost) plus the 6 per-task
    groups, sorted descending by the base model's accuracy. Same mean +/- std
    (across model seeds) convention as plot_generalization_bars.
    """
    if not PLOTTING_AVAILABLE:
        print("matplotlib not available; skipping ranked generalization bar chart.")
        return
    model_names = list(summary["models"])
    if "Base" not in model_names:
        raise ValueError("plot_generalization_bars_ranked requires a 'Base' model in the summary")

    ranked_tasks = sorted(summary["tasks"], key=lambda t: summary["models"]["Base"][t]["mean"], reverse=True)
    groups = ["overall"] + ranked_tasks
    group_labels = [_format_task_label(g) for g in groups]

    n_models = len(model_names)
    x = range(len(groups))
    width = 0.8 / n_models

    # Set2: grey for Base, purple for the 1st fine-tuned model, pink for the 2nd.
    non_base_names = [n for n in model_names if n != "Base"]
    color_by_name = {"Base": _MODEL_COLORS[7]}
    if len(non_base_names) > 0:
        color_by_name[non_base_names[0]] = _MODEL_COLORS[2]
    if len(non_base_names) > 1:
        color_by_name[non_base_names[1]] = _MODEL_COLORS[3]

    for name in model_names:
        overall = summary["models"][name]["overall"]
        print(f"{name}: overall {metric_label} = {overall['mean']:.3f} +/- {overall['std']:.3f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(max(9, len(groups) * 1.6), 4))
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="grey", linewidth=0.6, linestyle="-", alpha=0.4, zorder=0)
    for i, name in enumerate(model_names):
        means = [summary["models"][name][g]["mean"] for g in groups]
        stds = [summary["models"][name][g]["std"] for g in groups]
        offsets = [xi + (i - (n_models - 1) / 2) * width for xi in x]
        color = color_by_name.get(name, _MODEL_COLORS[i % len(_MODEL_COLORS)])
        plt.bar(offsets, means, width=width * 0.9, yerr=stds, capsize=3,
                label=name, color=color, zorder=3)

    plt.xticks(list(x), group_labels, rotation=30, ha="right")
    plt.axvline(0.5, color="black", linewidth=0.8, linestyle="--")
    plt.ylabel(metric_label)
    plt.ylim(0, 1.0)
    plt.title(f"MMBench {metric_label}")
    # plt.title("MMBench accuracy: overall + per-ability (ranked by base model, mean +/- std across model seeds)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved ranked generalization bar chart to {out_path}")


def _format_cell(value: float) -> str:
    return f"{value:.3f}" if value == value else "--"  # NaN check


def _format_cell_std(mean: float, std: float, bold: bool = False) -> str:
    if mean != mean:  # NaN check
        return "--"
    cell = f"{mean:.3f}\\std{{{std:.2f}}}"
    return f"\\textbf{{{cell}}}" if bold else cell


def latex_task_table(summary: dict, task: str) -> str:
    """Appendix table for one task: a 'Base model' column (mean +/- std across
    its subset seeds) plus one block per fine-tuned model (rows = subset
    seeds, columns = model seeds + an 'overall' column), matching the
    requested layout:

        \\begin{tabular}{c|c|c|c|c|c|c|c|c}
                     & Base model & \\multicolumn{4}{c}{Ego-only} & \\multicolumn{4}{|c}{Ego+Gaze} \\
                     &            & MS=42 & MS=43 & MS=44 & Overall & MS=42 & MS=43 & MS=44 & Overall \\
            SS=42    &            & ...
            SS=43    &            & ...
            SS=44    &            & ...
            Average  & mean+/-std & mean+/-std ... & mean+/-std & ...
        \\end{tabular}
    """
    base_runs = summary["breakdown"]["Base"]
    subset_seeds = base_runs["subset_seeds"]
    base_reports = base_runs["reports"]

    model_names = [n for n in summary["models"] if n != "Base" and summary["breakdown"][n]["model_seeds"] is not None]
    model_runs = {name: summary["breakdown"][name] for name in model_names}

    n_cols = 2 + sum(len(model_runs[name]["model_seeds"]) + 1 for name in model_names)
    lines = []
    lines.append(r"\begin{tabular}{c|c|" + "|".join("c" * (len(model_runs[name]["model_seeds"]) + 1) for name in model_names) + "}")
    lines.append(r"\toprule")
    header1 = " & \multirow{2}{*}{Base model} & " + " & ".join(
        f"\\multicolumn{{{len(model_runs[name]['model_seeds']) + 1}}}{{|c}}{{{name}}}" for name in model_names
    ) + r" \\"
    lines.append(header1)
    col = 3
    cline_parts = []
    for name in model_names:
        span = len(model_runs[name]["model_seeds"]) + 1
        cline_parts.append(f"\\cmidrule(l){{{col}-{col + span - 1}}}")
        col += span
    lines.append("".join(cline_parts))
    header2 = " & & " + " & ".join(
        " & ".join(f"MS={ms.split('_')[-1]}" for ms in model_runs[name]["model_seeds"]) + " & Overall"
        for name in model_names
    ) + r" \\"
    lines.append(header2)
    lines.append(r"\midrule")

    col_values = {name: {ms: [] for ms in model_runs[name]["model_seeds"]} for name in model_names}
    for subset_seed in subset_seeds:
        row_label = f"SS={subset_seed}"
        base_cell = _format_cell(_task_accuracy(base_reports[(None, subset_seed)], task))
        row_cells = [base_cell]
        for name in model_names:
            runs = model_runs[name]
            reports = runs["reports"]
            row_values = []
            for ms in runs["model_seeds"]:
                acc = _task_accuracy(reports[(ms, subset_seed)], task)
                row_values.append(acc)
                col_values[name][ms].append(acc)
            row_mean, row_std = _mean_std(row_values)
            row_cells.append(" & ".join(_format_cell(v) for v in row_values) + f" & {_format_cell_std(row_mean, row_std)}")
        lines.append(f"{row_label} & " + " & ".join(row_cells) + r" \\")

    lines.append(r"\midrule")
    base_values = [_task_accuracy(r, task) for r in base_reports.values()]
    base_mean, base_std = _mean_std(base_values)
    avg_cells = [_format_cell_std(base_mean, base_std, bold=True)]
    for name in model_names:
        runs = model_runs[name]
        col_stats = [_mean_std(col_values[name][ms]) for ms in runs["model_seeds"]]
        grand_mean, grand_std = _mean_std([v for vs in col_values[name].values() for v in vs])
        avg_cells.append(" & ".join(_format_cell_std(m, s) for m, s in col_stats) + f" & {_format_cell_std(grand_mean, grand_std, bold=True)}")
    lines.append("Average & " + " & ".join(avg_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def write_generalization_tables(summary: dict, out_dir: str):
    tasks = summary["tasks"]
    tables_dir = os.path.join(out_dir, "appendix_tables")
    os.makedirs(tables_dir, exist_ok=True)
    for task in tasks:
        task_slug = re.sub(r"[^a-zA-Z0-9]+", "_", task).strip("_")
        table = latex_task_table(summary, task)
        out_path = os.path.join(tables_dir, f"{task_slug}.tex")
        with open(out_path, "w") as f:
            f.write(f"% Appendix table for task: {task}\n\n")
            f.write(table + "\n")
        print(f"Wrote appendix table for '{task}' to {out_path}")


def run_generalization(args):
    subset_seeds = args.subset_seeds
    base_runs = collect_model_runs(args.log_dir, args.model_name, args.category_level,
                                    args.samples_per_category, subset_seeds, adapter_paths=None)

    model_runs_by_name = {}
    if args.model1_adapters:
        model_runs_by_name[args.model1_name] = collect_model_runs(
            args.log_dir, args.model_name, args.category_level,
            args.samples_per_category, subset_seeds, adapter_paths=args.model1_adapters)
    if args.model2_adapters:
        model_runs_by_name[args.model2_name] = collect_model_runs(
            args.log_dir, args.model_name, args.category_level,
            args.samples_per_category, subset_seeds, adapter_paths=args.model2_adapters)

    if not model_runs_by_name:
        raise SystemExit("Provide at least --model1_adapters (and optionally --model2_adapters)")

    summary = build_generalization_report(base_runs, model_runs_by_name)
    pretty_print_generalization(summary)

    os.makedirs(args.out_dir, exist_ok=True)
    json_out = {
        "tasks": summary["tasks"],
        "models": summary["models"],
    }
    out_path = os.path.join(args.out_dir, "mmbench_generalization_summary.json")
    with open(out_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nWrote generalization summary to {out_path}")

    if args.plot:
        plot_generalization_bars(summary, os.path.join(args.out_dir, "mmbench_generalization_bars.png"))
        plot_generalization_bars_ranked(summary, os.path.join(args.out_dir, "mmbench_generalization_bars_ranked.png"))

    write_generalization_tables(summary, args.out_dir)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    # --- generalization-robustness subcommand ---
    gen = subparsers.add_parser(
        "generalization",
        help="Base vs. N fine-tuned models (each with 3 model seeds) across 3 MMBench subset seeds")
    gen.add_argument("--log_dir", default="logs/mmbench_eval",
                      help="Root log dir passed to eval_mmbench.py (--log_dir)")
    gen.add_argument("--model_name", default="Qwen/Qwen3-VL-2B-Instruct")
    gen.add_argument("--category_level", default="l2-category", choices=["category", "l2-category"])
    gen.add_argument("--samples_per_category", type=int, default=100)
    gen.add_argument("--subset_seeds", type=int, nargs="+", default=[42, 43, 44])
    gen.add_argument("--model1_name", default="Model 1")
    gen.add_argument("--model1_adapters", nargs="+", default=None,
                      help="3 adapter paths for model 1, one per model seed")
    gen.add_argument("--model2_name", default="Model 2")
    gen.add_argument("--model2_adapters", nargs="+", default=None,
                      help="3 adapter paths for model 2, one per model seed")
    gen.add_argument("--out_dir", default="results/mmbench_generalization")
    gen.add_argument("--plot", action="store_true", help="Also save the grouped bar chart")

    # --- legacy base-vs-finetuned delta subcommand (default when no subcommand given) ---
    parser.add_argument("--base_report", help="Path to base model's *_report.json (single-seed mode)")
    parser.add_argument("--finetuned_report", help="Path to fine-tuned model's *_report.json (single-seed mode)")
    parser.add_argument("--base_reports", nargs="+",
                        help="Paths to base model's *_report.json, one per seed (multi-seed mode)")
    parser.add_argument("--finetuned_reports", nargs="+",
                        help="Paths to fine-tuned model's *_report.json, one per seed, "
                             "seed-aligned with --base_reports (multi-seed mode)")
    parser.add_argument("--out_dir", default="results/mmbench_compare")
    parser.add_argument("--drop_threshold", type=float, default=0.05,
                        help="Flag categories whose accuracy drops by more than this (finetuned vs base)")
    parser.add_argument("--plot", action="store_true", help="Also save a per-category delta bar chart")
    args = parser.parse_args()

    if args.mode == "generalization":
        run_generalization(args)
        return

    os.makedirs(args.out_dir, exist_ok=True)

    if args.base_reports or args.finetuned_reports:
        if not (args.base_reports and args.finetuned_reports):
            parser.error("--base_reports and --finetuned_reports must be given together")
        base_reports = [load_report(p) for p in args.base_reports]
        finetuned_reports = [load_report(p) for p in args.finetuned_reports]
        report = aggregate_across_seeds(base_reports, finetuned_reports, drop_threshold=args.drop_threshold)
        pretty_print_aggregate(report)

        out_path = os.path.join(args.out_dir, "mmbench_comparison_aggregate.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote aggregated comparison report to {out_path}")

        if args.plot:
            plot_deltas_aggregate(report, os.path.join(args.out_dir, "mmbench_deltas_aggregate.png"))
        return

    if not (args.base_report and args.finetuned_report):
        parser.error("Provide either --base_report/--finetuned_report, --base_reports/--finetuned_reports, "
                     "or the 'generalization' subcommand")

    base = load_report(args.base_report)
    finetuned = load_report(args.finetuned_report)
    report = compare(base, finetuned, drop_threshold=args.drop_threshold)
    pretty_print(report)

    out_path = os.path.join(args.out_dir, "mmbench_comparison.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote comparison report to {out_path}")

    if args.plot:
        plot_deltas(report, os.path.join(args.out_dir, "mmbench_deltas.png"))


if __name__ == "__main__":
    main()
