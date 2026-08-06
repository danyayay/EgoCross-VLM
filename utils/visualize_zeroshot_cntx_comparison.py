#!/usr/bin/env python3
"""Visualize context-guided VLM experiment comparisons.

Reads the workbook produced by utils/summarize_context_guided.py and writes plots
that compare context feature, prompt placement, serialization format, and metric
tradeoffs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


_METRIC_COLS = [
    'accuracy', 'macro_f1', 'cross_recall', 'yield_recall',
    'cross_f1', 'yield_f1', 'cross_precision', 'yield_precision',
]

_CONTEXT_FEATURE_ORDER = [
    'gaze_direction',
    'gaze_on_screen_ratio',
    'vehicle_motion',
    'ego_motion',
    'ego_motion,gaze_direction',
    'ego_motion,gaze_on_screen_ratio',
    'ego_motion,vehicle_motion',
]


def _reindex_context_features(pivot: pd.DataFrame) -> pd.DataFrame:
    order = [f for f in _CONTEXT_FEATURE_ORDER if f in pivot.index] + [f for f in pivot.index if f not in _CONTEXT_FEATURE_ORDER]
    return pivot.loc[order]


def _normalize_aggregated_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map summarize_context_guided's cross-seed aggregate (<metric>_mean/_std)
    onto the plain per-run column names the rest of this module expects, so
    the same plots work on both a per-seed summary.csv and a summary_aggregated.csv.
    """
    mean_cols = {c: c[: -len('_mean')] for c in df.columns if c.endswith('_mean') and c[: -len('_mean')] in _METRIC_COLS}
    if not mean_cols:
        return df
    df = df.rename(columns=mean_cols)
    if 'run_tag' not in df.columns:
        if 'seeds' in df.columns:
            df['run_tag'] = 'seeds_' + df['seeds'].astype(str)
        else:
            df['run_tag'] = df.index.astype(str)
    return df


def _load_runs(path: Path, sheet: str) -> pd.DataFrame:
    if path.suffix in ['.csv', '.tsv']:
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=sheet)
    df = _normalize_aggregated_columns(df)
    for col in _METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _save(fig, out_base: Path, formats: list[str]) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f'.{fmt}'), bbox_inches='tight', dpi=220)


def _metric_label(metric: str) -> str:
    return metric.replace('_', ' ').title()


def _format_fps_number(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip('0').rstrip('.')


def _effective_video_frame_fps(row: pd.Series) -> float | None:
    if 'video_sample_fps' in row and pd.notna(row.get('video_sample_fps')):
        try:
            return float(row.get('video_sample_fps'))
        except (TypeError, ValueError):
            pass
    if 'num_frames' in row and pd.notna(row.get('num_frames')):
        try:
            num_frames = float(row.get('num_frames'))
            duration = float(row.get('video_duration', 2.0)) if pd.notna(row.get('video_duration', 2.0)) else 2.0
            if duration > 0:
                return num_frames / duration
        except (TypeError, ValueError):
            return None
    return None


def _context_fps_plot_value(row: pd.Series) -> str:
    raw = row.get('context_feature_fps')
    if pd.isna(raw):
        return 'NA'
    text = str(raw)
    if text == 'auto':
        fps = _effective_video_frame_fps(row)
        return _format_fps_number(fps) if fps is not None else 'auto'
    return text


def _with_context_fps_plot_value(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if 'context_feature_fps' in work.columns:
        work['context_feature_fps_plot'] = work.apply(_context_fps_plot_value, axis=1)
    return work


def _setup_matplotlib():
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style='whitegrid', context='talk')
    plt.rcParams.update({
        'figure.dpi': 120,
        'savefig.dpi': 220,
        'axes.titlesize': 15,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.family': 'DejaVu Sans',
    })
    return plt, sns


def plot_top_runs(df: pd.DataFrame, out_dir: Path, metric: str, top_k: int, formats: list[str]) -> None:
    plt, sns = _setup_matplotlib()
    _require_columns(df, [metric, 'context_features', 'context_prompt_mode', 'context_feature_format'])

    top = df.sort_values([metric, 'accuracy'], ascending=[False, False]).head(top_k).copy()
    top['setting'] = (
        top['context_features'].astype(str) + '\n'
        + top['context_prompt_mode'].astype(str) + ' / '
        + top['context_feature_format'].astype(str) + ' / fps='
        + _with_context_fps_plot_value(top).get('context_feature_fps_plot', pd.Series(['NA'] * len(top), index=top.index)).astype(str)
    )

    fig, ax = plt.subplots(figsize=(12, max(5, 0.5 * len(top))))
    sns.barplot(data=top, y='setting', x=metric, hue='context_features', dodge=False, ax=ax)
    ax.set_title(f'Top {top_k} Runs by {_metric_label(metric)}')
    ax.set_xlabel(_metric_label(metric))
    ax.set_ylabel('')
    ax.set_xlim(0, max(0.75, float(top[metric].max()) + 0.04))
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
    ax.legend(title='Feature', loc='lower right')
    _save(fig, out_dir / f'top_{top_k}_{metric}', formats)
    plt.close(fig)


def plot_heatmaps(df: pd.DataFrame, out_dir: Path, metrics: list[str], formats: list[str], auto_only=True) -> None:
    plt, sns = _setup_matplotlib()
    _require_columns(df, ['context_features', 'context_prompt_mode', 'context_feature_format'])

    work = df.copy()
    if auto_only and 'context_feature_fps' in work.columns:
        if 'context_feature_fps' in work.columns:
            work = work[work['context_feature_fps'].astype(str) == 'auto'].copy()
    work = work[work['context_features'] != 'none'].copy()
    if work.empty:
        return
    for metric in metrics:
        if metric not in work.columns:
            continue
        modes = [m for m in ['preface', 'interleaved'] if m in set(work['context_prompt_mode'].dropna())]
        if not modes:
            modes = sorted(work['context_prompt_mode'].dropna().unique())
        # create subplots without shared y-axis so we can hide y-tick labels on
        # non-left subplots without affecting the left-most axis
        fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 5), squeeze=False)
        vmin = float(work[metric].min())
        vmax = float(work[metric].max())
        for ax, mode in zip(axes[0], modes):
            sub = work[work['context_prompt_mode'] == mode]
            pivot = sub.pivot_table(
                index='context_features',
                columns='context_feature_format',
                values=metric,
                aggfunc='max',
            )
            # preferred_cols = [c for c in ['detailed', 'legacy', 'compact', 'summary', 'schema'] if c in pivot.columns]
            preferred_cols = [c for c in ['legacy', 'schema'] if c in pivot.columns]
            pivot = pivot[preferred_cols]
            pivot = _reindex_context_features(pivot)
            sns.heatmap(
                pivot, annot=True, fmt='.3f', cmap='viridis',
                vmin=vmin, vmax=vmax, linewidths=0.5, ax=ax,
                cbar=ax is axes[0][-1],
            )
            ax.set_title(f'{mode}: {_metric_label(metric)}')
            ax.set_xlabel('Context Format')
            # only set ylabel on the left-most subplot
            if ax is axes[0][0]:
                ax.set_ylabel('Context Feature')
            else:
                # hide y tick labels on non-left subplots to avoid duplication
                ax.set_yticklabels([])
        fig.suptitle(f'{_metric_label(metric)} by Feature, Format, and Prompt Mode (Context FPS=Video Frame FPS)', y=1.03)
        if auto_only:
            _save(fig, out_dir / f'heatmap_{metric}_by_feature_format_mode', formats)
        else:
            _save(fig, out_dir / f'heatmap_{metric}_by_feature_format_mode_max', formats)
        plt.close(fig)


def plot_mode_delta(df: pd.DataFrame, out_dir: Path, metric: str, formats: list[str]) -> None:
    plt, sns = _setup_matplotlib()
    _require_columns(df, ['context_features', 'context_prompt_mode', 'context_feature_format', metric])

    pivot = df.pivot_table(
        index=['context_features', 'context_feature_format'],
        columns='context_prompt_mode',
        values=metric,
        aggfunc='max',
    ).reset_index()
    if not {'preface', 'interleaved'}.issubset(pivot.columns):
        return
    pivot['interleaved_minus_preface'] = pivot['interleaved'] - pivot['preface']
    pivot['setting'] = pivot['context_features'].astype(str) + ' / ' + pivot['context_feature_format'].astype(str)
    pivot = pivot.sort_values('interleaved_minus_preface')

    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(pivot))))
    colors = ['#b23a48' if x < 0 else '#287c71' for x in pivot['interleaved_minus_preface']]
    ax.barh(pivot['setting'], pivot['interleaved_minus_preface'], color=colors)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_title(f'Effect of Interleaving on {_metric_label(metric)}')
    ax.set_xlabel(f'Interleaved minus Preface {_metric_label(metric)}')
    ax.set_ylabel('')
    for y, val in enumerate(pivot['interleaved_minus_preface']):
        ax.text(val + (0.003 if val >= 0 else -0.003), y, f'{val:+.3f}',
                va='center', ha='left' if val >= 0 else 'right', fontsize=9)
    _save(fig, out_dir / f'delta_interleaved_minus_preface_{metric}', formats)
    plt.close(fig)


def plot_recall_tradeoff(df: pd.DataFrame, out_dir: Path, formats: list[str], interactive: bool) -> None:
    plt, sns = _setup_matplotlib()
    _require_columns(df, ['cross_recall', 'yield_recall', 'macro_f1', 'context_features', 'context_prompt_mode', 'context_feature_format'])

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=df,
        x='cross_recall', y='yield_recall',
        hue='context_features', style='context_prompt_mode',
        size='macro_f1', sizes=(70, 280), ax=ax,
    )
    best = df.sort_values(['macro_f1', 'accuracy'], ascending=[False, False]).head(5)
    for _, row in best.iterrows():
        ax.annotate(
            f"{row['context_features']}\n{row['context_prompt_mode']}/{row['context_feature_format']}",
            (row['cross_recall'], row['yield_recall']),
            xytext=(6, 6), textcoords='offset points', fontsize=8,
        )
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Cross vs Yield Recall Tradeoff')
    ax.set_xlabel('Cross Recall')
    ax.set_ylabel('Yield Recall')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    _save(fig, out_dir / 'recall_tradeoff_cross_vs_yield', formats)
    plt.close(fig)

    if interactive:
        import plotly.express as px

        hover_cols = [
            'run_tag', 'accuracy', 'macro_f1', 'cross_f1', 'yield_f1',
            'context_features', 'context_prompt_mode', 'context_feature_format', 'context_feature_fps',
        ]
        fig_px = px.scatter(
            df,
            x='cross_recall', y='yield_recall',
            color='context_features', symbol='context_prompt_mode',
            size='macro_f1', hover_data=[c for c in hover_cols if c in df.columns],
            title='Cross vs Yield Recall Tradeoff',
        )
        fig_px.update_xaxes(range=[0, 1])
        fig_px.update_yaxes(range=[0, 1])
        fig_px.write_html(out_dir / 'recall_tradeoff_cross_vs_yield.html')


def plot_metric_panels(df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    plt, sns = _setup_matplotlib()
    metric_cols = ['accuracy', 'macro_f1', 'cross_recall', 'yield_recall']
    available = [c for c in metric_cols if c in df.columns]
    long = df.melt(
        id_vars=['context_features', 'context_prompt_mode', 'context_feature_format', 'run_tag'],
        value_vars=available,
        var_name='metric', value_name='value',
    )
    long['setting'] = long['context_prompt_mode'].astype(str) + ' / ' + long['context_feature_format'].astype(str)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=False)
    axes = axes.ravel()
    for ax, metric in zip(axes, available):
        sub = long[long['metric'] == metric]
        sns.barplot(data=sub, x='context_features', y='value', hue='setting', ax=ax)
        ax.set_title(_metric_label(metric))
        ax.set_xlabel('')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=20)
        ax.set_ylim(0, 1)
        if ax is not axes[0]:
            ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend_.remove()
    fig.legend(handles, labels, title='Mode / Format', loc='lower center', ncol=4)
    fig.suptitle('Metric Comparison Across Context Dimensions', y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    _save(fig, out_dir / 'metrics_by_context_dimensions', formats)
    plt.close(fig)



def _fps_sort_key(value: object) -> tuple[int, float | str]:
    text = str(value)
    if text == 'auto':
        return (0, -1.0)
    try:
        return (1, float(text))
    except ValueError:
        return (2, text)


def _ordered_fps_values(df: pd.DataFrame) -> list[str]:
    col = 'context_feature_fps_plot' if 'context_feature_fps_plot' in df.columns else 'context_feature_fps'
    values = [str(v) for v in df[col].dropna().unique()]
    return sorted(values, key=_fps_sort_key)


def _split_feature_set(value: object) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(',') if part.strip())


def _lighten_color(color: str, amount: float = 0.55) -> str:
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(color)
    r = r + (1.0 - r) * amount
    g = g + (1.0 - g) * amount
    b = b + (1.0 - b) * amount
    return mcolors.to_hex((r, g, b))


def _feature_family_palette(features: Iterable[object]) -> dict[str, str]:
    base_colors = [
        '#1f77b4',  # blue
        '#d62728',  # red
        '#2ca02c',  # green
        '#9467bd',  # purple
        '#ff7f0e',  # orange
        '#17becf',  # cyan
        '#8c564b',  # brown
        '#7f7f7f',  # gray
    ]
    names = list(dict.fromkeys(str(f) for f in features))
    combos = [name for name in names if len(_split_feature_set(name)) > 1]
    singles = [name for name in names if len(_split_feature_set(name)) == 1]
    palette: dict[str, str] = {}

    for i, combo in enumerate(combos):
        color = base_colors[i % len(base_colors)]
        palette[combo] = color
        for part in _split_feature_set(combo):
            if part in singles and part not in palette:
                palette[part] = _lighten_color(color)

    used = len(combos)
    for single in singles:
        if single not in palette:
            palette[single] = base_colors[used % len(base_colors)]
            used += 1
    return palette


def plot_fps_heatmaps(df: pd.DataFrame, out_dir: Path, metrics: list[str], formats: list[str]) -> None:
    if 'context_feature_fps' not in df.columns:
        return
    plt, sns = _setup_matplotlib()
    _require_columns(df, ['context_features', 'context_prompt_mode', 'context_feature_fps'])
    work = _with_context_fps_plot_value(df.dropna(subset=['context_feature_fps']).copy())
    if work.empty:
        return
    fps_order = _ordered_fps_values(work)
    modes = [m for m in ['preface', 'interleaved'] if m in set(work['context_prompt_mode'].dropna())]
    if not modes:
        modes = sorted(work['context_prompt_mode'].dropna().unique())
    if 'context_feature_format' in work.columns:
        format_values = [f for f in ['legacy', 'schema'] if f in set(work['context_feature_format'].dropna().astype(str))]
        if not format_values:
            format_values = sorted(work['context_feature_format'].dropna().astype(str).unique())
    else:
        format_values = ['all']
    for metric in metrics:
        if metric not in work.columns:
            continue
        for feature_format in format_values:
            sub_format = work if feature_format == 'all' else work[work['context_feature_format'].astype(str) == feature_format]
            if sub_format.empty:
                continue
            fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 5), squeeze=False)
            vmin = float(sub_format[metric].min())
            vmax = float(sub_format[metric].max())
            for ax, mode in zip(axes[0], modes):
                sub = sub_format[sub_format['context_prompt_mode'] == mode]
                pivot = sub.pivot_table(
                    index='context_features',
                    columns='context_feature_fps_plot',
                    values=metric,
                    aggfunc='max',
                )
                pivot = pivot[[c for c in fps_order if c in pivot.columns]]
                pivot = _reindex_context_features(pivot)
                sns.heatmap(
                    pivot, annot=True, fmt='.3f', cmap='mako',
                    vmin=vmin, vmax=vmax, linewidths=0.5, ax=ax,
                    cbar=ax is axes[0][-1],
                )
                ax.set_title(f'{mode}: {_metric_label(metric)}')
                ax.set_xlabel('Context Feature FPS')
                if ax is axes[0][0]:
                    ax.set_ylabel('Context Feature')
                else:
                    ax.set_yticklabels([])
            suffix = '' if feature_format == 'all' else f' ({feature_format})'
            file_suffix = '' if feature_format == 'all' else f'_{feature_format}'
            fig.suptitle(f'{_metric_label(metric)} by Feature FPS and Prompt Mode{suffix}', y=1.03)
            _save(fig, out_dir / f'heatmap_{metric}_by_feature_fps_mode{file_suffix}', formats)
            plt.close(fig)


def plot_fps_metric_panels(df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if 'context_feature_fps' not in df.columns:
        return
    plt, sns = _setup_matplotlib()
    metric_cols = ['accuracy', 'macro_f1']
    available = [c for c in metric_cols if c in df.columns]
    required = ['context_features', 'context_prompt_mode', 'context_feature_format', 'context_feature_fps']
    if not set(required).issubset(df.columns) or not available:
        return
    work = _with_context_fps_plot_value(df.dropna(subset=['context_feature_fps']).copy())
    work = work[work['context_prompt_mode'] == 'preface'].copy()
    if work.empty:
        return
    work['context_feature_fps_plot'] = pd.Categorical(
        work['context_feature_fps_plot'].astype(str),
        categories=_ordered_fps_values(work),
        ordered=True,
    )
    work['setting'] = work['context_prompt_mode'].astype(str) + ' / ' + work['context_feature_format'].astype(str)
    long = work.melt(
        id_vars=['context_features', 'context_prompt_mode', 'context_feature_format', 'context_feature_fps_plot', 'setting', 'run_tag'],
        value_vars=available,
        var_name='metric', value_name='value',
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    axes = axes.ravel()
    for ax, metric in zip(axes, available):
        sub = long[long['metric'] == metric]
        sns.lineplot(
            data=sub,
            x='context_feature_fps_plot', y='value',
            hue='context_features', style='setting',
            palette=_feature_family_palette(long['context_features'].dropna().unique()),
            markers=True, dashes=False, ax=ax,
        )
        ax.set_title(_metric_label(metric))
        ax.set_xlabel('Context Feature FPS')
        ax.set_ylabel('Score')
        values = pd.to_numeric(sub['value'], errors='coerce').dropna()
        if not values.empty:
            ymin = float(values.min())
            ymax = float(values.max())
            pad = max(0.02, (ymax - ymin) * 0.15)
            ax.set_ylim(max(0, ymin - pad), min(1, ymax + pad))
        else:
            ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=20)
        if ax is not axes[0] and ax.get_legend() is not None:
            ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()
    fig.legend(handles, labels, title='Feature / Setting', loc='lower center', ncol=3)
    fig.suptitle('Metric Comparison Across Context Feature FPS (Preface Only)', y=0.98)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    _save(fig, out_dir / 'metrics_by_context_feature_fps', formats)
    plt.close(fig)


def plot_fps_accuracy_f1_by_format(df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if 'context_feature_fps' not in df.columns:
        return
    plt, sns = _setup_matplotlib()
    required = ['context_features', 'context_prompt_mode', 'context_feature_format', 'context_feature_fps']
    metrics = ['accuracy', 'macro_f1']
    if not set(required).issubset(df.columns) or not all(m in df.columns for m in metrics):
        return
    work = _with_context_fps_plot_value(df.dropna(subset=['context_feature_fps']).copy())
    work = work[work['context_prompt_mode'] == 'preface'].copy()
    work = work[work['context_feature_format'].astype(str).isin(['legacy', 'schema'])].copy()
    if work.empty:
        return
    work['context_feature_fps_plot'] = pd.Categorical(
        work['context_feature_fps_plot'].astype(str),
        categories=_ordered_fps_values(work),
        ordered=True,
    )
    palette = _feature_family_palette(work['context_features'].dropna().unique())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=False)
    row_defs = [('accuracy', 'Accuracy'), ('macro_f1', 'Macro F1')]
    col_defs = [('legacy', 'Legacy'), ('schema', 'Schema')]
    for r, (metric, metric_title) in enumerate(row_defs):
        metric_values = pd.to_numeric(work[metric], errors='coerce').dropna()
        if not metric_values.empty:
            ymin = float(metric_values.min())
            ymax = float(metric_values.max())
            pad = max(0.02, (ymax - ymin) * 0.15)
            ylim = (max(0, ymin - pad), min(1, ymax + pad))
        else:
            ylim = (0, 1)
        for c, (feature_format, format_title) in enumerate(col_defs):
            ax = axes[r][c]
            sub = work[work['context_feature_format'].astype(str) == feature_format]
            if sub.empty:
                ax.set_visible(False)
                continue
            sns.lineplot(
                data=sub,
                x='context_feature_fps_plot', y=metric,
                hue='context_features',
                palette=palette,
                markers=True, dashes=False, marker='o', ax=ax,
            )
            ax.set_title(f'{metric_title} / {format_title}')
            ax.set_xlabel('Context Feature FPS' if r == 1 else '')
            ax.set_ylabel(metric_title if c == 0 else '')
            ax.set_ylim(*ylim)
            ax.tick_params(axis='x', rotation=20)
            if not (r == 0 and c == 0) and ax.get_legend() is not None:
                ax.get_legend().remove()
    handles, labels = axes[0][0].get_legend_handles_labels() if axes[0][0].get_legend() is not None else ([], [])
    if axes[0][0].get_legend() is not None:
        axes[0][0].get_legend().remove()
    if handles:
        fig.legend(handles, labels, title='Context Feature', loc='lower center', ncol=3)
    fig.suptitle('Accuracy and Macro F1 Across Context Feature FPS (Preface Only)', y=0.98)
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    _save(fig, out_dir / 'accuracy_f1_by_context_feature_fps_format', formats)
    plt.close(fig)


def plot_aggregated_metric(
    df: pd.DataFrame, out_dir: Path, metrics: list[str], formats: list[str],
    context_prompt_mode: str = 'preface', context_feature_format: str = 'legacy',
    select_by: str = 'macro_f1',
) -> None:
    """Heatmap-style table across seeds: one row per context_features, one
    column per metric, cells annotated with 'mean±std' and colored by mean.
    Restricted to a single (context_prompt_mode, context_feature_format)
    setup (default preface/legacy) so rows are directly comparable. Each
    context config typically has multiple context_feature_fps runs per seed;
    for each seed we first pick the single fps run that maximizes `select_by`
    (e.g. macro_f1 or accuracy), then average that best-per-seed value across
    seeds — averaging across fps values directly would understate the best
    achievable performance for that context config.
    """
    plt, sns = _setup_matplotlib()
    _require_columns(df, ['context_features', 'context_prompt_mode', 'context_feature_format', 'seed_dir'])
    metric_order = [m for m in ['accuracy', 'macro_f1'] if m in metrics] + [m for m in metrics if m not in ('accuracy', 'macro_f1')]
    available = [m for m in metric_order if m in df.columns]
    if not available or select_by not in df.columns:
        return

    work = df[
        (df['context_prompt_mode'] == context_prompt_mode)
        & (df['context_feature_format'] == context_feature_format)
    ].copy()
    if work.empty:
        return

    best_per_seed = (
        work.sort_values(select_by, ascending=False)
        .groupby(['context_features', 'seed_dir'], as_index=False)
        .head(1)
    )

    stats = best_per_seed.groupby('context_features')[available].agg(['mean', 'std'])
    stats[[(m, 'std') for m in available]] = stats[[(m, 'std') for m in available]].fillna(0.0)
    means = _reindex_context_features(stats.xs('mean', axis=1, level=1))
    stds = stats.xs('std', axis=1, level=1).loc[means.index]

    labels = means.copy().astype(object)
    for m in available:
        labels[m] = [f'{mean_val:.3f} ± {std_val:.3f}' for mean_val, std_val in zip(means[m], stds[m])]

    fig, ax = plt.subplots(figsize=(3.2 * len(available) + 2, max(5, 0.7 * len(means))))
    sns.heatmap(
        means, annot=labels, fmt='', cmap='viridis', annot_kws={'fontsize': 11},
        vmin=float(means.values.min()), vmax=float(means.values.max()),
        linewidths=0.5, ax=ax, cbar_kws={'label': 'Mean'},
    )
    ax.set_xticklabels([_metric_label(m) for m in available], rotation=0)
    ax.set_title(f'Mean ± Std Across Seeds, Best {_metric_label(select_by)} per Seed ({context_prompt_mode} / {context_feature_format})')
    ax.set_xlabel('')
    ax.set_ylabel('Context Feature')
    metric_suffix = '_'.join(available)
    _save(fig, out_dir / f'aggregated_table_{metric_suffix}_best_{select_by}_{context_prompt_mode}_{context_feature_format}', formats)
    plt.close(fig)


def plot_fps_best_runs(df: pd.DataFrame, out_dir: Path, metric: str, formats: list[str]) -> None:
    if 'context_feature_fps' not in df.columns or metric not in df.columns:
        return
    plt, sns = _setup_matplotlib()
    required = ['context_features', 'context_feature_fps', 'context_prompt_mode', 'context_feature_format']
    if not set(required).issubset(df.columns):
        return
    work = _with_context_fps_plot_value(df.dropna(subset=['context_feature_fps']).copy())
    work = work[work['context_prompt_mode'] == 'preface'].copy()
    if work.empty:
        return
    work['context_feature_fps_plot'] = pd.Categorical(
        work['context_feature_fps_plot'].astype(str),
        categories=_ordered_fps_values(work),
        ordered=True,
    )
    best = work.sort_values([metric, 'accuracy'], ascending=[False, False]).groupby(
        ['context_features', 'context_feature_fps_plot'], as_index=False
    ).head(1)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(best))))
    best['setting'] = best['context_features'].astype(str) + ' / fps=' + best['context_feature_fps_plot'].astype(str)
    best = best.sort_values(metric)
    sns.barplot(data=best, y='setting', x=metric, hue='context_prompt_mode', ax=ax)
    ax.set_title(f'Best {_metric_label(metric)} by Feature and Context FPS (Preface Only)')
    ax.set_xlabel(_metric_label(metric))
    ax.set_ylabel('')
    ax.set_xlim(0, max(0.75, float(best[metric].max()) + 0.04))
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)
    ax.legend(title='Prompt Mode')
    _save(fig, out_dir / f'best_{metric}_by_feature_fps', formats)
    plt.close(fig)


_EXCLUDED_CONTEXT_FEATURES = {'gaze_direction_change'}


def _clean_runs(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ['context_features', 'context_prompt_mode', 'context_feature_format'])
    df = df.dropna(subset=['context_features', 'context_prompt_mode', 'context_feature_format'])
    keep = df['context_features'].apply(
        lambda v: not (_EXCLUDED_CONTEXT_FEATURES & set(_split_feature_set(v)))
    )
    return df[keep]


def make_plots(df: pd.DataFrame, out_dir: Path, metrics: list[str], top_k: int, formats: list[str], interactive: bool) -> None:
    plot_top_runs(df, out_dir, 'macro_f1', top_k, formats)
    plot_heatmaps(df, out_dir, metrics, formats, auto_only=True)
    plot_heatmaps(df, out_dir, metrics, formats, auto_only=False)
    plot_mode_delta(df, out_dir, 'macro_f1', formats)
    plot_recall_tradeoff(df, out_dir, formats, interactive)
    plot_metric_panels(df, out_dir, formats)
    plot_fps_heatmaps(df, out_dir, metrics, formats)
    plot_fps_metric_panels(df, out_dir, formats)
    plot_fps_accuracy_f1_by_format(df, out_dir, formats)
    plot_fps_best_runs(df, out_dir, 'macro_f1', formats)


def _find_seed_summaries(log_root: Path, summary_name: str) -> list[Path]:
    return sorted(log_root.glob(f'seed_*/summary/{summary_name}'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--log_root', default='logs/vlm_eval_context_guided_dot/Qwen2.5-VL-7B-Instruct',
        help='Directory containing seed_*/summary/<summary_name> per-seed CSVs, '
        'e.g. logs/vlm_eval_context_guided_dot/Qwen2.5-VL-7B-Instruct. Plots are generated per seed under '
        '<out_dir>/seed_<N>/, plus an aggregated plot across all seeds under <out_dir>/summary/.',
    )
    parser.add_argument('--summary_name', default='summary.csv', help='Filename of each per-seed summary CSV.')
    parser.add_argument('--sheet', default='runs')
    parser.add_argument('--out_dir', default='logs/vlm_eval_context_guided_dot/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--metrics', nargs='+', default=['macro_f1', 'accuracy'])
    parser.add_argument('--top_k', type=int, default=12)
    parser.add_argument('--formats', nargs='+', default=['png'])
    parser.add_argument('--interactive', action='store_true', help='Also write an interactive Plotly HTML recall scatter.')
    args = parser.parse_args()

    log_root = Path(args.log_root)
    out_dir = Path(args.out_dir)
    seed_paths = _find_seed_summaries(log_root, args.summary_name)

    if not seed_paths:
        # Fall back to legacy behavior: log_root itself points at a single summary CSV.
        df = _clean_runs(_load_runs(log_root, args.sheet))
        make_plots(df, out_dir, args.metrics, args.top_k, args.formats, args.interactive)
        print(f'Wrote plots to {out_dir}')
        return

    all_dfs = []
    for seed_path in seed_paths:
        seed_name = seed_path.parent.parent.name  # e.g. "seed_42"
        df_seed = _clean_runs(_load_runs(seed_path, args.sheet))
        df_seed['seed_dir'] = df_seed.get('seed_dir', seed_name)
        seed_out_dir = log_root / seed_name / 'summary'
        make_plots(df_seed, seed_out_dir, args.metrics, args.top_k, args.formats, args.interactive)
        print(f'Wrote plots for {seed_name} to {seed_out_dir}')
        all_dfs.append(df_seed)

    df_all = pd.concat(all_dfs, ignore_index=True)
    agg_out_dir = out_dir / 'summary'
    for select_metric in ('macro_f1', 'accuracy'):
        plot_aggregated_metric(df_all, agg_out_dir, args.metrics, args.formats, select_by=select_metric)
    print(f'Wrote aggregated plots (across {len(seed_paths)} seeds) to {agg_out_dir}')


if __name__ == '__main__':
    main()
