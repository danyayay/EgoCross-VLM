"""Mimics plot_set1_frame_num_side_by_side: context feature fps influence.

Aggregates per-seed summary.csv files (multiple model seeds) into mean lines
with a shaded uncertainty band (min-max range or +/- 1 std) per context feature.
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Config ---
LEGEND_ORDER = [
    "ego, gaze direction",
    "ego, gaze on screen",
    "ego, vehicle motion",
    "ego motion",  # <- ego motion last in row 1
    "gaze direction",  # row 2 starts here
    "gaze on screen",
    "vehicle motion",
]
LEGEND_NCOL = 4
LEGEND_LABELS = {
    "ego_motion": "ego motion",
    "ego_motion,gaze_direction": "ego, gaze direction",
    "ego_motion,gaze_direction_change": "ego, gaze direction change",
    "ego_motion,gaze_on_screen_ratio": "ego, gaze on screen",
    "ego_motion,vehicle_motion": "ego, vehicle motion",
    "gaze_direction": "gaze direction",
    "gaze_direction_change": "gaze direction change",
    "gaze_on_screen_ratio": "gaze on screen",
    "vehicle_motion": "vehicle motion",
}


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    # fuzzy contains
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    raise KeyError(f"None of the candidate columns found: {candidates}. Available: {list(df.columns)}")


def _base_feature_name(name: str) -> str:
    # Maps 'ego_motion,gaze_direction' -> 'gaze_direction'
    s = str(name).strip()
    if "," in s:
        return s.split(",")[-1].strip()
    return s


def _legend_label(name: str) -> str:
    return LEGEND_LABELS.get(name, "unknown")


def _transpose_legend_grid(labels, handles, ncol: int, pad_label: str = ""):
    """Return (labels_T, handles_T) to transpose a legend laid out as rows x ncol."""
    n = len(labels)
    nrow = int(np.ceil(n / ncol))
    # pad to full grid
    total = nrow * ncol
    labels2 = list(labels) + [pad_label] * (total - n)
    handles2 = list(handles) + [plt.Line2D([], [], linestyle="none")] * (total - n)

    # row-major grid
    grid_lab = [labels2[r * ncol:(r + 1) * ncol] for r in range(nrow)]
    grid_h = [handles2[r * ncol:(r + 1) * ncol] for r in range(nrow)]

    # column-major flatten => transposed appearance
    out_lab = []
    out_h = []
    for c in range(ncol):
        for r in range(nrow):
            out_lab.append(grid_lab[r][c])
            out_h.append(grid_h[r][c])
    return out_lab, out_h


def load_seed_summaries(summary_glob: str) -> pd.DataFrame:
    summary_paths = sorted(glob.glob(summary_glob))
    if not summary_paths:
        raise FileNotFoundError(f"No per-seed summary.csv files found matching: {summary_glob}")

    seed_frames = []
    for p in summary_paths:
        d = pd.read_csv(p)
        d["seed_dir"] = p.split("/")[-3]  # e.g. "seed_42"
        seed_frames.append(d)
        print(f"Loaded {len(d)} rows from {p}")

    df = pd.concat(seed_frames, ignore_index=True)
    print(f"Aggregating over {len(summary_paths)} seeds: {[p.split('/')[-3] for p in summary_paths]}")
    return df


def select_and_clean(df_context: pd.DataFrame) -> pd.DataFrame:
    # only preface-format features, to keep it clean and focused on the main trend
    dctx = df_context[
        (df_context.context_prompt_mode == "preface") & (df_context.context_feature_format == "legacy")
    ].copy()

    col_feature = _find_col(dctx, ["context_features", "context_feature", "context_feature_name"])
    col_fps = _find_col(dctx, ["context_feature_fps", "fps"])
    col_acc = _find_col(dctx, ["accuracy", "Accuracy"])
    col_f1 = _find_col(dctx, ["macro_f1", "Macro_F1", "Macro F1"])

    dctx = dctx.rename(
        columns={col_feature: "context_feature", col_fps: "fps", col_acc: "Accuracy", col_f1: "Macro_F1"},
    ).copy()

    # Remove gaze_direction_change pairs (both single feature and ego_motion,feature)
    dctx["context_feature"] = dctx["context_feature"].astype(str).str.strip()
    dctx = dctx[dctx["context_feature"].map(_base_feature_name) != "gaze_direction_change"].copy()

    print(f"Selected {len(dctx)} records for plotting context feature FPS influence.")

    dctx["fps"] = pd.to_numeric(dctx["fps"], errors="coerce")
    # Map NaN fps option to 4 (treat missing context_feature_fps as 4 FPS)
    dctx["fps"] = dctx["fps"].fillna(4)
    dctx["Accuracy"] = pd.to_numeric(dctx["Accuracy"], errors="coerce")
    dctx["Macro_F1"] = pd.to_numeric(dctx["Macro_F1"], errors="coerce")
    return dctx


def aggregate_across_seeds(dctx: pd.DataFrame, band_mode: str) -> pd.DataFrame:
    """Average metrics across seeds per (context_feature, fps), with a lo/hi band.

    Points where fewer seeds have finished are averaged over whatever is
    currently available; the band reflects only those seeds.
    """
    # Within a single seed there should be at most 1 record per (context_feature, fps).
    per_seed_counts = dctx.groupby(["seed_dir", "context_feature", "fps"]).size().reset_index(name="n")
    dup_within_seed = per_seed_counts.query("n != 1")
    if len(dup_within_seed):
        raise ValueError(f"Multiple records for the same (seed, context_feature, fps): {dup_within_seed}")

    dsel = (
        dctx.groupby(["context_feature", "fps"], as_index=False)
        .agg(
            Accuracy_mean=("Accuracy", "mean"),
            Accuracy_min=("Accuracy", "min"),
            Accuracy_max=("Accuracy", "max"),
            Accuracy_std=("Accuracy", "std"),
            Macro_F1_mean=("Macro_F1", "mean"),
            Macro_F1_min=("Macro_F1", "min"),
            Macro_F1_max=("Macro_F1", "max"),
            Macro_F1_std=("Macro_F1", "std"),
            n_seeds=("seed_dir", "nunique"),
        )
        .sort_values(["context_feature", "fps"])
        .reset_index(drop=True)
    )
    # std is NaN when only 1 seed contributed; treat as zero band width
    dsel["Accuracy_std"] = dsel["Accuracy_std"].fillna(0.0)
    dsel["Macro_F1_std"] = dsel["Macro_F1_std"].fillna(0.0)

    if band_mode == "minmax":
        dsel["Accuracy_lo"], dsel["Accuracy_hi"] = dsel["Accuracy_min"], dsel["Accuracy_max"]
        dsel["Macro_F1_lo"], dsel["Macro_F1_hi"] = dsel["Macro_F1_min"], dsel["Macro_F1_max"]
    elif band_mode == "std":
        dsel["Accuracy_lo"] = dsel["Accuracy_mean"] - dsel["Accuracy_std"]
        dsel["Accuracy_hi"] = dsel["Accuracy_mean"] + dsel["Accuracy_std"]
        dsel["Macro_F1_lo"] = dsel["Macro_F1_mean"] - dsel["Macro_F1_std"]
        dsel["Macro_F1_hi"] = dsel["Macro_F1_mean"] + dsel["Macro_F1_std"]
    else:
        raise ValueError(f"Unknown band_mode: {band_mode!r}. Use 'minmax' or 'std'.")

    print(
        "Seed coverage per (context_feature, fps):\n",
        dsel[["context_feature", "fps", "n_seeds"]].to_string(),
    )
    return dsel


def build_feature_colors(features: list[str]) -> dict[str, tuple]:
    """sns.color_palette('Paired') with paired setups (single feature vs. + ego_motion)."""
    unique_bases = sorted({_base_feature_name(f) for f in features})
    paired = sns.color_palette("Paired", n_colors=max(2 * len(unique_bases), 2))
    base_to_pair = {b: (paired[2 * i], paired[2 * i + 1]) for i, b in enumerate(unique_bases)}

    feat_to_color: dict[str, tuple] = {}
    for f in features:
        base = _base_feature_name(f)
        c_single, c_ego = base_to_pair[base]
        feat_to_color[f] = c_ego if "ego_motion" in f else c_single
    return feat_to_color


def plot_metrics(dsel: pd.DataFrame, feat_to_color: dict[str, tuple], xticks: list[float]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.8), sharex=True, sharey=False)
    for feat, dd in dsel.groupby("context_feature"):
        x = dd["fps"].to_numpy()
        color = feat_to_color.get(feat, "C0")
        label = _legend_label(feat)

        ax1.plot(x, dd["Accuracy_mean"].to_numpy(), marker="o", linewidth=2, label=label, color=color)
        ax1.fill_between(x, dd["Accuracy_lo"].to_numpy(), dd["Accuracy_hi"].to_numpy(), color=color, alpha=0.15, linewidth=0)

        ax2.plot(x, dd["Macro_F1_mean"].to_numpy(), marker="o", linewidth=2, label=label, color=color)
        ax2.fill_between(x, dd["Macro_F1_lo"].to_numpy(), dd["Macro_F1_hi"].to_numpy(), color=color, alpha=0.15, linewidth=0)

    for ax, metric in [(ax1, "Accuracy"), (ax2, "Macro F1")]:
        ax.set_xscale("log", base=2)
        ax.set_xticks(xticks)
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, pos: f"{int(v)}" if v in xticks else ""))
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Context feature sampling rate")
        ax.set_title(metric)

    return fig, ax1, ax2


def add_shared_legend(fig, ax1, requested: list[str], ncol: int):
    handles, labels = ax1.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))

    # Put 'ego motion' as the LAST item in the FIRST row.
    # With ncol=4, the first row is indices [0,1,2,3] in `requested`.
    pad = " "
    ordered_labels = []
    for l in requested:
        if l in label_to_handle:
            ordered_labels.append(l)
        else:
            ordered_labels.append(pad)  # keep slot if that label isn't present in this slice

    # pad to complete 2 rows x ncol
    while len(ordered_labels) < 2 * ncol:
        ordered_labels.append(pad)

    # Append any remaining labels not explicitly requested (optional, keeps info if more features exist)
    extras = [l for l in labels if l not in requested]
    for l in sorted(extras):
        ordered_labels.append(l)

    ordered_handles = [label_to_handle.get(l, plt.Line2D([], [], linestyle="none")) for l in ordered_labels]
    # If you're transposing the legend grid, moving an item within a "row" is easier to do BEFORE transpose.
    # So we do NOT apply any extra post-transpose shuffling here -- `requested` already encodes the desired order.
    ordered_labels_T, ordered_handles_T = _transpose_legend_grid(ordered_labels, ordered_handles, ncol=ncol)

    legend_rows = int(np.ceil(len(ordered_labels) / ncol))
    bottom = 0.14 + 0.045 * max(0, legend_rows - 1)
    bottom = min(0.12, bottom)
    fig.subplots_adjust(bottom=bottom)

    fig.legend(
        ordered_handles_T, ordered_labels_T,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=ncol,
        fontsize=8,
        title_fontsize=8,
        frameon=False,
        handlelength=2.0,
    )
    return bottom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logdir",
        default="logs/vlm_eval_context_guided_dot/Qwen2.5-VL-7B-Instruct",
        help="Directory containing per-seed subdirs (seed_*/summary/summary.csv). "
        "The output figure is also saved here.",
    )
    parser.add_argument(
        "--band_mode",
        default='std',
        choices=["minmax", "std"],
        help="Shaded band across seeds: 'minmax' (min-max range) or 'std' (mean +/- 1 std).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary_glob = os.path.join(args.logdir, "seed_*", "summary", "summary.csv")

    df_context = load_seed_summaries(summary_glob)
    dctx = select_and_clean(df_context)
    dsel = aggregate_across_seeds(dctx, band_mode=args.band_mode)

    xticks = sorted(dsel["fps"].unique().tolist())
    features = sorted(dsel["context_feature"].unique().tolist())
    feat_to_color = build_feature_colors(features)

    fig, ax1, ax2 = plot_metrics(dsel, feat_to_color, xticks)
    bottom = add_shared_legend(fig, ax1, LEGEND_ORDER, LEGEND_NCOL)

    fig.tight_layout(rect=(0, bottom, 1, 1))
    logdir_abs = os.path.abspath(args.logdir)
    eval_name = os.path.basename(os.path.dirname(logdir_abs)) or os.path.basename(logdir_abs)
    fig_filename = os.path.join(args.logdir, f"{eval_name}_context_fps_{args.band_mode}.png")
    plt.savefig(fig_filename)
    print(f"Saved figure to {fig_filename}")
    plt.show()


if __name__ == "__main__":
    main()
