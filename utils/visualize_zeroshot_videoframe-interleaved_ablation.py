"""Video frame-count and interleaved-timestamps ablations from logs/vlm_eval/.

Aggregates per-seed `*_responses_report.json` files (multiple seeds per model)
into mean lines/bars with an error bar of +/- 1 std across seeds.

Set 1 (frame_num sweep): all three Qwen models (interleaved runs only, since
that is the only variant with multi-frame_num coverage for the 7B/8B models).

Set 2 (interleaved vs not): grouped bars per Qwen model at a fixed frame_num
(INTERLEAVED_COMPARISON_FRAME_NUM); only Qwen3-VL-2B-Instruct has both
interleaved and non-interleaved runs, so the other models only show the
interleaved bar.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS = ["Qwen3-VL-2B-Instruct", "Qwen3-VL-8B-Instruct", "Qwen2.5-VL-7B-Instruct"]
MODEL_DISPLAY = {
    "Qwen3-VL-2B-Instruct": "Qwen3-VL-2B",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
}
INTERLEAVED_COMPARISON_FRAME_NUM = 8


def _parse_run_dir_name(run_dir: str) -> dict:
    name = os.path.basename(run_dir)
    frame_match = re.search(r"_f(\d+)_", name)
    frame_num = int(frame_match.group(1)) if frame_match else None
    interleaved = "interleaved" in name and "nointerleave" not in name
    quantized = "int8" in name
    prompt_match = re.search(r"_p(\d+)", name)
    question = f"p{prompt_match.group(1)}" if prompt_match else None
    return {
        "frame_num": frame_num,
        "interleaved": interleaved,
        "quantized": "8-bit" if quantized else None,
        "question": question,
    }


def load_reports(logdir: str, models: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        model_dir = os.path.join(logdir, model)
        report_paths = sorted(glob.glob(os.path.join(model_dir, "seed_*", "*", "*_responses_report.json")))
        for report_path in report_paths:
            seed_dir = report_path.split(os.sep)[-3]
            run_dir = os.path.dirname(report_path)
            with open(report_path, "r") as f:
                report = json.load(f)
            classification = report.get("classification", {})
            accuracy = classification.get("accuracy")
            macro_f1 = classification.get("macro_f1")
            if accuracy is None or macro_f1 is None:
                continue
            meta = _parse_run_dir_name(run_dir)
            rows.append({
                "Model": MODEL_DISPLAY.get(model, model),
                "seed_dir": seed_dir,
                "Accuracy": accuracy,
                "Macro_F1": macro_f1,
                **meta,
            })
    if not rows:
        raise FileNotFoundError(f"No responses_report.json files found under {logdir} for models {models}.")
    return pd.DataFrame(rows)


def select_p6_records(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()
    if "question" in d.columns:
        d = d[d["question"].astype(str).str.strip().eq("p6")]
    return d


def filter_interleaved_multiframe_series(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (filtered_df, series_cols) only for interleaved series with >1 frame_num."""
    d = d.dropna(subset=["frame_num", "Accuracy", "Macro_F1"]).copy()
    d = d[d["interleaved"].eq(True)]
    series_cols = ["Model", "quantized"] if "quantized" in d.columns else ["Model"]
    if "quantized" in d.columns:
        d["quantized"] = d["quantized"].fillna("none")
    counts = d.groupby(series_cols)["frame_num"].nunique(dropna=True)
    keep = counts[counts > 1].index
    if len(keep) == 0:
        raise ValueError("No interleaved models found with >1 distinct frame_num to plot.")
    if len(series_cols) == 1:
        d = d[d[series_cols[0]].isin(list(keep))]
    else:
        d = d.merge(keep.to_frame(index=False), on=series_cols, how="inner")
    return d, series_cols


def format_legend_label(model_name: str, quantized_value: str | None) -> str:
    m = str(model_name)
    q = "" if quantized_value is None else str(quantized_value).strip().lower()
    if q == "8-bit":
        return f"{m} (8-bit)"
    return m


def aggregate_across_seeds(d: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Mean/std across seeds per group_cols, keeping std as NaN->0 for single-seed groups."""
    d_agg = (
        d.groupby(group_cols, as_index=False)
        .agg(
            Accuracy_mean=("Accuracy", "mean"),
            Accuracy_std=("Accuracy", "std"),
            Macro_F1_mean=("Macro_F1", "mean"),
            Macro_F1_std=("Macro_F1", "std"),
            n_seeds=("seed_dir", "nunique"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    d_agg["Accuracy_std"] = d_agg["Accuracy_std"].fillna(0.0)
    d_agg["Macro_F1_std"] = d_agg["Macro_F1_std"].fillna(0.0)
    return d_agg


def plot_frame_num_ablation(df_in: pd.DataFrame) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    d, series_cols = filter_interleaved_multiframe_series(df_in)
    d_agg = aggregate_across_seeds(d, series_cols + ["frame_num"])
    xticks = sorted(d_agg["frame_num"].dropna().astype(int).unique())

    # Requested palette (RGB): 186,228,188 / 123,204,196 / 67,162,202
    palette = [
        (186/255, 228/255, 188/255),
        (123/255, 204/255, 196/255),
        (67/255, 162/255, 202/255),
    ]
    model_to_color = {
        "Qwen3-VL-2B": palette[0],
        "Qwen3-VL-8B": palette[1],
        "Qwen2.5-VL-7B": palette[2],
    }

    model_order = ["Qwen3-VL-2B", "Qwen3-VL-8B", "Qwen2.5-VL-7B"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), sharex=True, sharey=True)
    for key, dd in d_agg.groupby(series_cols):
        if not isinstance(key, tuple):
            key = (key,)
        model_name = str(key[0])
        quantized_value = str(key[1]) if len(key) > 1 else None
        label = format_legend_label(model_name, quantized_value)
        color = model_to_color.get(model_name, palette[0])
        x = dd["frame_num"].astype(int)
        acc_mean = dd["Accuracy_mean"].astype(float)
        acc_std = dd["Accuracy_std"].astype(float)
        f1_mean = dd["Macro_F1_mean"].astype(float)
        f1_std = dd["Macro_F1_std"].astype(float)

        ax1.plot(x, acc_mean, marker="o", linewidth=2, label=label, color=color, zorder=3)
        ax1.fill_between(x, acc_mean - acc_std, acc_mean + acc_std, color=color, alpha=0.2, linewidth=0, zorder=2)

        ax2.plot(x, f1_mean, marker="o", linewidth=2, label=label, color=color, zorder=3)
        ax2.fill_between(x, f1_mean - f1_std, f1_mean + f1_std, color=color, alpha=0.2, linewidth=0, zorder=2)

    for ax, metric in [(ax1, "Accuracy"), (ax2, "Macro F1")]:
        ax.set_xscale("log", base=2)
        ax.set_xticks(xticks)
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, pos: f"{int(v)}" if v in xticks else ""))
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.25, zorder=0)
        ax.set_xlabel("Video frame number")
        ax.set_title(metric)

    ax1.set_ylabel("Metric score")
    ax1.set_ylim(0.5, 0.65)
    ax2.set_ylim(0.5, 0.65)

    def _ordered_handles_labels(ax):
        handles, labels = ax.get_legend_handles_labels()
        order = sorted(
            range(len(labels)),
            key=lambda i: next((j for j, m in enumerate(model_order) if labels[i].startswith(m)), len(model_order)),
        )
        return [handles[i] for i in order], [labels[i] for i in order]

    h1, l1 = _ordered_handles_labels(ax1)
    h2, l2 = _ordered_handles_labels(ax2)
    ax1.legend(h1, l1, fontsize=8, loc="lower right")
    ax2.legend(h2, l2, fontsize=8, loc="upper right")
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_interleaved_ablation(df_in: pd.DataFrame) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Grouped bar plot: one bar per Qwen model, grouped by interleaved (no/yes), at a fixed frame_num."""
    d = df_in[df_in["frame_num"].eq(INTERLEAVED_COMPARISON_FRAME_NUM)].copy()
    d = d.dropna(subset=["Accuracy", "Macro_F1"])
    d_agg = aggregate_across_seeds(d, ["Model", "interleaved"])

    model_order = ["Qwen3-VL-2B", "Qwen3-VL-8B", "Qwen2.5-VL-7B"]
    palette = [
        (186/255, 228/255, 188/255),
        (123/255, 204/255, 196/255),
        (67/255, 162/255, 202/255),
    ]
    model_to_color = dict(zip(model_order, palette))

    print(
        f"Averaged rows for interleaved vs not (frame_num={INTERLEAVED_COMPARISON_FRAME_NUM}):"
    )
    print(d_agg.sort_values(["interleaved", "Model"]).to_string(index=False))

    x_labels = ["no", "yes"]
    x_pos = np.arange(len(x_labels))
    n_models = len(model_order)
    bar_width = 0.8 / n_models

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)
    for ax, metric in [(ax1, "Accuracy"), (ax2, "Macro_F1")]:
        for i, model_name in enumerate(model_order):
            dd = d_agg[d_agg["Model"].eq(model_name)].set_index("interleaved")
            offset = (i - (n_models - 1) / 2) * bar_width
            xi = x_pos + offset
            means = [dd[f"{metric}_mean"].get(v, np.nan) for v in [False, True]]
            stds = [dd[f"{metric}_std"].get(v, np.nan) for v in [False, True]]
            valid = ~np.isnan(means)
            color = model_to_color[model_name]
            # Opaque backing bars so gridlines don't show through the semi-transparent fill
            ax.bar(xi[valid], np.array(means)[valid], width=bar_width, color="white", zorder=2)
            ax.bar(
                xi[valid], np.array(means)[valid], width=bar_width,
                yerr=np.array(stds)[valid], capsize=3, alpha=0.85, color=color,
                label=model_name, zorder=3,
            )
        ax.set_xticks(x_pos, x_labels)
        ax.set_xlabel("Interleaved inputs?")
        ax.set_title(metric.replace("_", " "))
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", alpha=0.25, zorder=0)

    ax1.set_ylabel("Metric score")
    ax1.set_ylim(0.3, 0.7)
    ax1.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    return fig, (ax1, ax2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logdir",
        default="logs/vlm_eval",
        help="Directory containing per-model/per-seed subdirs (Model/seed_*/run/*_responses_report.json).",
    )
    parser.add_argument(
        "--outdir",
        default="logs/vlm_eval",
        help="Directory to save the output figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_reports(args.logdir, MODELS)
    df_plot = select_p6_records(df)

    os.makedirs(args.outdir, exist_ok=True)

    fig1, _ = plot_frame_num_ablation(df_plot)
    fig1_path = os.path.join(args.outdir, "zeroshot_video_frame_num_ablation_std.png")
    fig1.savefig(fig1_path)
    print(f"Saved figure to {fig1_path}")

    fig2, _ = plot_interleaved_ablation(df_plot)
    fig2_path = os.path.join(args.outdir, "zeroshot_interleaved_ablation_std.png")
    fig2.savefig(fig2_path)
    print(f"Saved figure to {fig2_path}")

    plt.show()


if __name__ == "__main__":
    main()
