import glob
import os
import re

import pandas as pd
import matplotlib.pyplot as plt


_TEST_START_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ INFO Starting test evaluation")
_TEST_SAVED_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ INFO Saved (\d+) results")

_VLP_LOG_ROOT = "../GroundVQA/lightning_logs/ft_methods_seeds/qn0"
_VLP_METRIC_RE = re.compile(
    r"test_close_acc\s+([0-9.]+)\s*\n\s*test_cross_f1\s+([0-9.]+)\s*\n\s*test_macro_f1\s+([0-9.]+)"
)
_VLP_SEED_DIR_RE = re.compile(r"^seed_(.+)$")
_VLP_VERSION_DIR_RE = re.compile(r"^version_\d+__(.+)$")
_VLP_LABEL_MAP = {
    "lora_attn_qv": "lan.",
    "connector": "cm.",
    "lora_attn_ff+connector": "lan.+cm.",
}

_LABEL_ORDER = ["lan.", "cm.", "lan.+cm."]


def _test_inference_time_per_sample(run_dir):
    """Seconds/sample for the final test-set eval, parsed from train.log timestamps."""
    train_log = os.path.join(run_dir, "train.log")
    if not os.path.isfile(train_log):
        return None
    start_ts = saved_ts = n_saved = None
    with open(train_log) as f:
        for line in f:
            m = _TEST_START_RE.search(line)
            if m:
                start_ts = pd.Timestamp(m.group(1))
                continue
            m = _TEST_SAVED_RE.search(line)
            if m:
                saved_ts = pd.Timestamp(m.group(1))
                n_saved = int(m.group(2))
    if start_ts is None or saved_ts is None or not n_saved:
        return None
    return (saved_ts - start_ts).total_seconds() / n_saved


def _parse_vlp_eval_log(run_dir):
    eval_log = os.path.join(run_dir, "eval.log")
    if not os.path.isfile(eval_log):
        return None
    with open(eval_log) as f:
        text = f.read()
    m = _VLP_METRIC_RE.search(text)
    if not m:
        return None
    return {
        "Accuracy": float(m.group(1)),
        "Macro F1": float(m.group(3)),
    }


def _collect_vlp_runs(log_root):
    rows = []
    for seed_dir in sorted(glob.glob(os.path.join(log_root, "seed_*"))):
        seed_m = _VLP_SEED_DIR_RE.match(os.path.basename(seed_dir))
        if not seed_m:
            continue
        seed = seed_m.group(1)
        for version_dir in sorted(glob.glob(os.path.join(seed_dir, "version_*"))):
            version_m = _VLP_VERSION_DIR_RE.match(os.path.basename(version_dir))
            if not version_m:
                continue
            ft_method = version_m.group(1)
            metrics = _parse_vlp_eval_log(version_dir)
            if metrics is None:
                continue
            rows.append({
                "label": _VLP_LABEL_MAP.get(ft_method, ft_method),
                "random_seed": seed,
                **metrics,
            })
    return pd.DataFrame(rows)


def _summarize_by_label(df, metrics, label_order=_LABEL_ORDER):
    """Collapse repeated runs within each seed, then compute mean/std across seeds."""
    df = df.copy()
    df["label"] = pd.Categorical(df["label"], categories=label_order, ordered=True)
    df = df.dropna(subset=["label"]).sort_values("label").reset_index(drop=True)

    per_seed = (
        df
        .groupby(["label", "random_seed"], observed=True)[metrics]
        .mean()
        .reset_index()
    )
    summary = (
        per_seed
        .groupby("label", observed=True)[metrics]
        .agg(["mean", "std"])
        .reindex(label_order)
    )
    means = summary.xs("mean", axis=1, level=1)
    # A category represented by one seed has no sample std; plot it as zero.
    stds = summary.xs("std", axis=1, level=1).fillna(0)
    return means, stds


def _print_summary_table(means, stds, metrics, label_order=_LABEL_ORDER):
    label_width = max(len(label) for label in label_order)
    col_widths = {metric: max(len(metric), 15) for metric in metrics}

    print("\nMean +/- std across seeds:")
    header = " " * label_width + "  " + "  ".join(metric.ljust(col_widths[metric]) for metric in metrics)
    print(header)
    print("-" * len(header))
    for label in label_order:
        if label not in means.index or means.loc[label].isna().all():
            continue
        cells = [
            f"{means.loc[label, metric]:.3f}+/-{stds.loc[label, metric]:.3f}".ljust(col_widths[metric])
            for metric in metrics
        ]
        print(f"{label.ljust(label_width)}  " + "  ".join(cells))
    print()


def _plot_bar_comparison(means, stds, plot_metrics, title, out_path, color=("#51aba0", "#a9d5cf")):
    ax = means[plot_metrics].plot.bar(
        yerr=stds[plot_metrics],
        figsize=(5, 4),
        label=plot_metrics,
        color=list(color),
        width=0.8,
        capsize=4,
    )

    # Show aggregated mean values on top of the bars.
    bar_containers = [c for c in ax.containers if hasattr(c, "datavalues")]
    for container, metric in zip(bar_containers, plot_metrics):
        ax.bar_label(
            container,
            labels=[f"{value:.3f}" if pd.notna(value) else "" for value in means[metric]],
            padding=7,
            fontsize=12,
        )

    ax.set_axisbelow(True)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0.7, 0.8)
    plt.xlabel("Fine-tuning strategy", fontsize=13)
    plt.ylabel("Metrics", fontsize=13)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved figure to {out_path}")


def plot_qwen_finetuning_comparison():
    df_qwen = pd.read_csv("logs/vlm_training/summary/summary.csv")
    df_qwen = df_qwen[(df_qwen.prompt_variant=='p6') & (df_qwen.interleaved_timestamps==True) & (df_qwen.lora_rank==2) & (df_qwen.lora_alpha==8)]
    df_qwen = df_qwen.copy()
    df_qwen['Inference Time (s/sample)'] = df_qwen['run_dir'].map(_test_inference_time_per_sample)

    df_merge_ = df_qwen[['test_acc', 'test_macro_f1', 'Inference Time (s/sample)', 'ft_type', 'random_seed']].rename(columns={
        'test_acc': 'Accuracy',
        'test_macro_f1': 'Macro F1',
        'ft_type': 'label'})
    df_merge_['label'] = df_merge_['label'].map({
        'lora_llm_vlm_bridger': 'lan.+cm.',
        'lora_vlm_bridger': 'cm.',
        'lora_llm_attn_mlp': 'lan.'
    })
    df_merge_ = df_merge_.dropna(subset=['label', 'random_seed']).reset_index(drop=True)

    metrics = ["Accuracy", "Macro F1", "Inference Time (s/sample)"]
    means, stds = _summarize_by_label(df_merge_, metrics)
    _print_summary_table(means, stds, metrics)
    _plot_bar_comparison(
        means, stds, ["Accuracy", "Macro F1"],
        title="Qwen3-VL-2B fine-tuning",
        out_path="results/qwen_finetuning_multiruns.png",
    )


def plot_vlp_finetuning_comparison(log_root=_VLP_LOG_ROOT):
    df = _collect_vlp_runs(log_root)
    if df.empty:
        print(f"No runs found under {log_root}")
        return

    metrics = ["Accuracy", "Macro F1"]
    means, stds = _summarize_by_label(df, metrics)
    _print_summary_table(means, stds, metrics)
    _plot_bar_comparison(
        means, stds, metrics,
        title="VLP fine-tuning",
        out_path="results/vlp_finetuning_multiruns.png",
        color=["#7b5ea6", "#a5add3"],
    )


if __name__ == "__main__":
    plot_qwen_finetuning_comparison()
    plot_vlp_finetuning_comparison()
