import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

colors = sns.color_palette("Set2")
# Palette in the style of the frame-number ablation figure: light -> dark
# sequential, one shade per series. Reused across both plots below.
_RANK_PALETTE = {
    2: colors[0],
    4: colors[1],
    8: colors[2],
}
_ALPHA_PALETTE = {
    4: colors[0],
    8: colors[1],
    16: colors[2],
}

# The configuration reported in the paper.
_REPORTED_RANK = 2
_REPORTED_ALPHA = 8

_METRICS = ["test_acc", "test_macro_f1"]
_METRIC_LABELS = {"test_acc": "Accuracy", "test_macro_f1": "Macro F1"}



def _load_summary(
    summary_csv: str, ft_type: str, ranks: list[int], alphas: list[int]
) -> pd.DataFrame:
    """Load and filter to the completed r x alpha sweep, aggregated across seeds."""
    df = pd.read_csv(summary_csv)
    df = df[
        (df.ft_type == ft_type)
        & (df.prompt_variant == "p6")
        & (df.interleaved_timestamps == True)  # noqa: E712
        & (df.context_features.isna() | (df.context_features == "none"))
    ].copy()
    df = df.dropna(subset=["lora_rank", "lora_alpha", "test_acc", "test_macro_f1", "random_seed"])
    df = df[df.lora_rank.isin(ranks) & df.lora_alpha.isin(alphas)]

    if df.empty:
        raise ValueError(
            f"No matching rows in {summary_csv} for ft_type={ft_type!r}, "
            f"ranks={ranks}, alphas={alphas}. Run scripts/run_vlm_training_egoonly_rank.sh first."
        )

    per_seed = df.groupby(["lora_rank", "lora_alpha", "random_seed"])[_METRICS].mean().reset_index()
    summary = per_seed.groupby(["lora_rank", "lora_alpha"])[_METRICS].agg(["mean", "std", "count"])
    return summary


def _style_axes(axes, xlabel: str, xticks: list[int]) -> None:
    for ax, metric in zip(axes, _METRICS):
        ax.set_xlabel(xlabel)
        ax.set_xticks(xticks)
        ax.set_title(_METRIC_LABELS[metric])
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", alpha=0.25, zorder=0)
        # ax.spines["top"].set_visible(False)
        # ax.spines["right"].set_visible(False)
        ax.set_ylim(0.7, 0.8)
    # axes[0].set_ylabel("Metric score")


def plot_alpha_ablation(
    summary_csv: str = "logs/vlm_training/summary/summary.csv",
    ft_type: str = "lora_vlm_bridger",
    out_path: str = "results/lora_alpha_rank_ablation.png",
) -> None:
    """LoRA scaling-factor ablation (primary figure): x-axis = alpha, one line per rank.

    Restricted to the fully completed sweep (r in {2,4,8}, alpha in {4,8,16},
    3 seeds per cell) so every point/band reflects the same amount of evidence.
    """
    ranks = [2, 4, 8]
    alphas = [4, 8, 16]
    summary = _load_summary(summary_csv, ft_type, ranks, alphas)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)

    for rank in ranks:
        sub = summary.xs(rank, level="lora_rank").sort_index()
        x = sub.index.to_numpy()
        color = _RANK_PALETTE[rank]
        for ax, metric in zip(axes, _METRICS):
            mean = sub[(metric, "mean")].to_numpy()
            std = sub[(metric, "std")].fillna(0).to_numpy()
            ax.plot(x, mean, marker="o", linewidth=2, color=color, label=f"$r$={rank}", zorder=3)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, linewidth=0, zorder=2)

    # if (_REPORTED_RANK, _REPORTED_ALPHA) in summary.index:
    #     reported = summary.loc[(_REPORTED_RANK, _REPORTED_ALPHA)]
    #     for ax, metric in zip(axes, _METRICS):
    #         value = reported[(metric, "mean")]
    #         ax.scatter(
    #             [_REPORTED_ALPHA], [value], s=140, facecolors="none",
    #             edgecolors="black", linewidths=1.6, zorder=5,
    #         )
    #         ax.annotate(
    #             "reported config", xy=(_REPORTED_ALPHA, value),
    #             xytext=(_REPORTED_ALPHA + 0.6, value + 0.002), fontsize=8, color="#333",
    #         )

    _style_axes(axes, r"Scaling factor $\alpha$", alphas)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    axes[1].legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


def plot_ratio_ablation(
    summary_csv: str = "logs/vlm_training/summary/summary.csv",
    ft_type: str = "lora_vlm_bridger",
    out_path: str = "results/lora_ratio_ablation.png",
) -> None:
    """LoRA alpha/rank-ratio ablation: grouped bars per ratio, one bar per rank.

    Several (rank, alpha) cells share the same ratio (e.g. ratio=2.0 comes from
    r=2/a=4, r=4/a=8, r=8/a=16). Grouping by ratio with rank as the bar series
    shows directly whether different (rank, alpha) pairs at the same ratio
    agree (bars at the same x-group land at similar heights) or scatter.
    """
    ranks = [2, 4, 8]
    alphas = [4, 8, 16]
    df = pd.read_csv(summary_csv)
    df = df[
        (df.ft_type == ft_type)
        & (df.prompt_variant == "p6")
        & (df.interleaved_timestamps == True)  # noqa: E712
        & (df.context_features.isna() | (df.context_features == "none"))
    ].copy()
    df = df.dropna(subset=["lora_rank", "lora_alpha", "test_acc", "test_macro_f1", "random_seed"])
    df = df[df.lora_rank.isin(ranks) & df.lora_alpha.isin(alphas)]
    df["ratio"] = df["lora_alpha"] / df["lora_rank"]

    per_seed = df.groupby(["lora_rank", "ratio", "random_seed"])[_METRICS].mean().reset_index()
    per_config = per_seed.groupby(["lora_rank", "ratio"])[_METRICS].agg(["mean", "std", "count"])

    ratios = sorted(df["ratio"].unique())
    ratio_labels = [f"{r:g}" for r in ratios]
    x_pos = np.arange(len(ratios))
    n_ranks = len(ranks)
    bar_width = 0.8 / n_ranks

    reported_ratio = _REPORTED_ALPHA / _REPORTED_RANK

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)
    for ax, metric in zip(axes, _METRICS):
        for i, rank in enumerate(ranks):
            offset = (i - (n_ranks - 1) / 2) * bar_width
            xi = x_pos + offset
            means = np.array([
                per_config.loc[(rank, r), (metric, "mean")] if (rank, r) in per_config.index else np.nan
                for r in ratios
            ])
            stds = np.array([
                per_config.loc[(rank, r), (metric, "std")] if (rank, r) in per_config.index else np.nan
                for r in ratios
            ])
            valid = ~np.isnan(means)
            color = _RANK_PALETTE[rank]
            # Opaque backing bars so gridlines don't show through the semi-transparent fill.
            ax.bar(xi[valid], means[valid], width=bar_width, color="white", zorder=2)
            ax.bar(
                xi[valid], means[valid], width=bar_width, yerr=np.nan_to_num(stds[valid]),
                capsize=3, alpha=0.85, color=color, label=f"$r$={rank}", zorder=3,
            )
            # if (rank, reported_ratio) in per_config.index and rank == _REPORTED_RANK:
            #     j = ratios.index(reported_ratio)
            #     std = per_config.loc[(rank, reported_ratio), (metric, "std")]
            #     value = per_config.loc[(rank, reported_ratio), (metric, "mean")]
            #     ax.annotate(
            #         "reported config", xy=(xi[j], value + std), xytext=(xi[j], value + std + 0.006),
            #         fontsize=7, color="#333", ha="center",
            #     )

        ax.set_xticks(x_pos, ratio_labels)
        ax.set_xlabel(r"Scaling ratio $\alpha / r$")
        ax.set_title(_METRIC_LABELS[metric])
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", alpha=0.25, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0.7, 0.8)
    axes[0].set_ylabel("Metric score")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")

    print("\nSeeds per (rank, ratio) bar:")
    print(per_config[(_METRICS[0], "count")].unstack("ratio"))


def plot_rank_alpha_ablation(
    summary_csv: str = "logs/vlm_training/summary/summary.csv",
    ft_type: str = "lora_vlm_bridger",
    out_path: str = "results/lora_rank_alpha_ablation.png",
) -> None:
    """LoRA rank ablation (secondary figure): x-axis = rank, one line per alpha.

    Restricted to the fully completed sweep (r in {2,4,8}, alpha in {4,8,16},
    3 seeds per cell); the incomplete alpha=2 cell (only r=1 was run) is
    excluded since it has no comparable ranks to form a trend.
    """
    ranks = [2, 4, 8]
    alphas = [4, 8, 16]
    summary = _load_summary(summary_csv, ft_type, ranks, alphas)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), sharex=True, sharey=True)

    for alpha in alphas:
        sub = summary.xs(alpha, level="lora_alpha").sort_index()
        x = sub.index.to_numpy()
        color = _ALPHA_PALETTE[alpha]
        for ax, metric in zip(axes, _METRICS):
            mean = sub[(metric, "mean")].to_numpy()
            std = sub[(metric, "std")].fillna(0).to_numpy()
            ax.plot(x, mean, marker="o", linewidth=2, color=color, label=f"$\\alpha$={alpha}", zorder=3)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, linewidth=0, zorder=2)

    # if (_REPORTED_RANK, _REPORTED_ALPHA) in summary.index:
    #     reported = summary.loc[(_REPORTED_RANK, _REPORTED_ALPHA)]
    #     for ax, metric in zip(axes, _METRICS):
    #         value = reported[(metric, "mean")]
    #         ax.scatter(
    #             [_REPORTED_RANK], [value], s=140, facecolors="none",
    #             edgecolors="black", linewidths=1.6, zorder=5,
    #         )
    #         ax.annotate(
    #             "reported config", xy=(_REPORTED_RANK, value),
    #             xytext=(_REPORTED_RANK + 0.15, value + 0.002), fontsize=8, color="#333",
    #         )

    _style_axes(axes, "LoRA rank $r$", ranks)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    axes[1].legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")

    n_seeds = (
        pd.read_csv(summary_csv)
        .pipe(lambda d: d[
            (d.ft_type == ft_type) & (d.prompt_variant == "p6")
            & (d.interleaved_timestamps == True)  # noqa: E712
            & (d.context_features.isna() | (d.context_features == "none"))
        ])
        .dropna(subset=["lora_rank", "lora_alpha", "test_acc", "test_macro_f1", "random_seed"])
        .pipe(lambda d: d[d.lora_rank.isin(ranks) & d.lora_alpha.isin(alphas)])
        .groupby(["lora_rank", "lora_alpha"])["random_seed"].nunique()
    )
    print("\nSeeds per (rank, alpha) cell:")
    print(n_seeds.unstack("lora_alpha"))


if __name__ == "__main__":
    plot_alpha_ablation()
    plot_ratio_ablation()
    plot_rank_alpha_ablation()
