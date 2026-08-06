#!/usr/bin/env python3
"""Summarize context-guided VLM evaluation logs into Excel."""

from __future__ import annotations

import argparse
import ast
import glob
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _find_log_for_report(report_path: Path) -> Path | None:
    match = re.match(r'^(\d{8}_\d{6})', report_path.name)
    if match:
        candidate = report_path.with_name(f'{match.group(1)}.log')
        if candidate.exists():
            return candidate
    logs = sorted(report_path.parent.glob('*.log'))
    return logs[-1] if logs else None


def _parse_log_args(log_path: Path | None) -> dict[str, Any]:
    if not log_path or not log_path.exists():
        return {}
    for line in log_path.read_text(errors='ignore').splitlines()[:30]:
        if 'eval_vlm arguments:' not in line:
            continue
        text = line.split('eval_vlm arguments:', 1)[1].strip()
        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _parse_run_tag(run_tag: str) -> dict[str, Any]:
    out: dict[str, Any] = {'run_tag': run_tag}
    parts = run_tag.split('_')
    for part in parts:
        if part.startswith('tcot'):
            out['tcot_type_from_tag'] = part.removeprefix('tcot') or 'none'
        elif re.fullmatch(r'f\d+', part):
            out['num_frames_from_tag'] = int(part[1:])
        elif part in {'interleaved', 'nointerleave'}:
            out['interleaved_from_tag'] = part == 'interleaved'
        elif re.fullmatch(r'p\d+', part):
            out['prompt_variant_from_tag'] = part
        elif part.startswith('ctx'):
            out['context_features_from_tag'] = part.removeprefix('ctx').replace('+', ',') or 'none'
        elif part.startswith('cfps'):
            out['context_feature_fps_from_tag'] = part.removeprefix('cfps')
        elif part in {'preface', 'interleaved'} and 'context_prompt_mode_from_tag' not in out:
            out['context_prompt_mode_from_tag'] = part
        elif part in {'detailed', 'legacy', 'compact', 'summary'}:
            out['context_feature_format_from_tag'] = part
        elif part.startswith('interp'):
            out['context_feature_interpretation_from_tag'] = part.removeprefix('interp')
        elif part == 'int8':
            out['quantize_from_tag'] = True
        elif part == 'novideo':
            out['no_video_from_tag'] = True
    return out


def _first_response_metadata(response_path: Path | None) -> dict[str, Any]:
    if not response_path or not response_path.exists():
        return {}
    try:
        rows = _load_json(response_path)
    except Exception:
        return {}
    if not rows:
        return {}
    for row in rows:
        if isinstance(row, dict) and 'error' not in row:
            keys = [
                'context_features', 'context_feature_fps', 'context_prompt_mode',
                'context_feature_format', 'context_feature_interpretation', 'tcot_type',
                'task_name', 'vcot_visual', 'vcot_text', 'temperature', 'max_new_tokens',
            ]
            return {k: row.get(k) for k in keys if k in row}
    return {}


def _metric(per_class: dict[str, Any], label: str, metric: str) -> Any:
    return (per_class.get(label) or {}).get(metric)


def _confusion_value(confusion: dict[str, Any], gt: str, pred: str) -> Any:
    return (confusion.get(gt) or {}).get(pred, 0)


def collect_rows(log_root: Path) -> list[dict[str, Any]]:
    report_paths = sorted(log_root.glob('**/*_responses_report.json'))
    rows: list[dict[str, Any]] = []
    for report_path in report_paths:
        try:
            report = _load_json(report_path)
        except Exception as exc:
            rows.append({'report_path': str(report_path), 'error': f'failed_to_read_report: {exc}'})
            continue

        response_path = Path(report.get('input', '')) if report.get('input') else report_path.with_name(report_path.name.replace('_report.json', '.json'))
        if not response_path.is_absolute():
            response_path = Path.cwd() / response_path
        if not response_path.exists():
            fallback = report_path.with_name(report_path.name.replace('_responses_report.json', '_responses.json'))
            response_path = fallback if fallback.exists() else None

        log_path = _find_log_for_report(report_path)
        args = _parse_log_args(log_path)
        meta = _first_response_metadata(response_path)

        rel_parts = report_path.relative_to(log_root).parts
        model_dir = rel_parts[0] if len(rel_parts) >= 2 else ''
        run_tag = rel_parts[1] if len(rel_parts) >= 2 else report_path.parent.name
        tag = _parse_run_tag(run_tag)

        cls = report.get('classification') or {}
        per_class = cls.get('per_class') or {}
        confusion = cls.get('confusion') or {}

        model_name = args.get('model_name') or model_dir
        row = {
            'model': model_name,
            'model_short': str(model_name).split('/')[-1],
            'run_tag': run_tag,
            'timestamp': report_path.name.split('_responses_report.json')[0],
            'task_name': meta.get('task_name') or args.get('task_name') or 'crossing_intention',
            'tcot_type': meta.get('tcot_type') or args.get('tcot_type') or tag.get('tcot_type_from_tag') or 'none',
            'prompt_variant': args.get('prompt_variant') or tag.get('prompt_variant_from_tag'),
            'num_frames': args.get('num_frames') or tag.get('num_frames_from_tag'),
            'interleaved_timestamps': args.get('interleaved_timestamps', tag.get('interleaved_from_tag')),
            'quantize': args.get('quantize', tag.get('quantize_from_tag', False)),
            'no_video': args.get('no_video', tag.get('no_video_from_tag', False)),
            'context_features': meta.get('context_features') or args.get('context_features') or tag.get('context_features_from_tag') or 'none',
            'context_feature_fps': meta.get('context_feature_fps') or args.get('context_feature_fps') or tag.get('context_feature_fps_from_tag'),
            'context_prompt_mode': meta.get('context_prompt_mode') or args.get('context_prompt_mode') or tag.get('context_prompt_mode_from_tag'),
            'context_feature_format': meta.get('context_feature_format') or args.get('context_feature_format') or tag.get('context_feature_format_from_tag'),
            'context_feature_interpretation': meta.get('context_feature_interpretation') or args.get('context_feature_interpretation') or tag.get('context_feature_interpretation_from_tag') or 'none',
            'vcot_visual': meta.get('vcot_visual') or args.get('vcot_visual'),
            'vcot_text': meta.get('vcot_text') or args.get('vcot_text'),
            'temperature': meta.get('temperature') if meta.get('temperature') is not None else args.get('temperature'),
            'max_new_tokens': meta.get('max_new_tokens') or args.get('max_new_tokens'),
            'ann_file': args.get('ann_file'),
            'video_root': args.get('video_root'),
            'sample_n': args.get('sample_n'),
            'sample_seed': args.get('sample_seed'),
            'seed': args.get('seed'),
            'num_samples': report.get('num_samples') or cls.get('total'),
            'correct': cls.get('correct'),
            'accuracy': cls.get('accuracy'),
            'macro_f1': cls.get('macro_f1'),
            'cross_precision': _metric(per_class, 'cross', 'precision'),
            'cross_recall': _metric(per_class, 'cross', 'recall'),
            'cross_f1': _metric(per_class, 'cross', 'f1'),
            'cross_support': _metric(per_class, 'cross', 'support'),
            'yield_precision': _metric(per_class, 'yield', 'precision'),
            'yield_recall': _metric(per_class, 'yield', 'recall'),
            'yield_f1': _metric(per_class, 'yield', 'f1'),
            'yield_support': _metric(per_class, 'yield', 'support'),
            'cm_cross_cross': _confusion_value(confusion, 'cross', 'cross'),
            'cm_cross_yield': _confusion_value(confusion, 'cross', 'yield'),
            'cm_yield_cross': _confusion_value(confusion, 'yield', 'cross'),
            'cm_yield_yield': _confusion_value(confusion, 'yield', 'yield'),
            'mean_iou': (report.get('temporal_iou') or {}).get('mean_iou'),
            'pct_iou_ge_0.5': (report.get('temporal_iou') or {}).get('pct_iou_ge_0.5'),
            'report_path': str(report_path),
            'responses_path': str(response_path) if response_path else '',
            'log_path': str(log_path) if log_path else '',
        }
        rows.append(row)
    return rows


def _write_summary(df: pd.DataFrame, out_path: Path) -> None:
    sort_cols = [
        'model_short', 'num_frames', 'interleaved_timestamps', 'quantize',
        'context_features', 'context_prompt_mode', 'context_feature_format',
        'context_feature_interpretation', 'timestamp',
    ]
    existing_sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(existing_sort_cols).reset_index(drop=True)

    metric_cols = [
        'accuracy', 'macro_f1', 'cross_precision', 'cross_recall', 'cross_f1',
        'yield_precision', 'yield_recall', 'yield_f1', 'mean_iou', 'pct_iou_ge_0.5',
    ]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(4)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_cols = [
        'context_features', 'context_prompt_mode', 'context_feature_format',
        'accuracy', 'macro_f1', 'cross_recall', 'yield_recall', 'run_tag', 'timestamp',
    ]
    best = df.sort_values(['macro_f1', 'accuracy'], ascending=[False, False])[[c for c in best_cols if c in df.columns]]

    pivot = pd.DataFrame()
    if {'context_features', 'context_prompt_mode', 'context_feature_format', 'macro_f1'}.issubset(df.columns):
        pivot = df.pivot_table(
            index=['context_features', 'context_feature_format'],
            columns='context_prompt_mode',
            values='macro_f1',
            aggfunc='max',
        ).reset_index()

    if out_path.suffix in {'.xlsx', '.xls'}:
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='runs', index=False)
            best.to_excel(writer, sheet_name='sorted_by_macro_f1', index=False)
            if not pivot.empty:
                pivot.to_excel(writer, sheet_name='macro_f1_pivot', index=False)
    elif out_path.suffix == '.csv':
        df.to_csv(out_path, index=False)

    print(f'Wrote {len(df)} runs to {out_path}')
    if not best.empty:
        top = best.iloc[0]
        print('Best macro_f1 run:', top.to_dict())


# Columns that identify a run *configuration*, shared across seeds. Note
# `run_tag` is NOT included here: it is derived from the per-run timestamp
# (see collect_rows), so it is unique per run and would defeat aggregation.
AGG_KEY_COLS = [
    'model_short', 'task_name', 'tcot_type', 'prompt_variant', 'num_frames',
    'interleaved_timestamps', 'quantize', 'no_video', 'context_features',
    'context_feature_fps', 'context_prompt_mode', 'context_feature_format',
    'context_feature_interpretation',
]
AGG_METRIC_COLS = [
    'accuracy', 'macro_f1', 'cross_precision', 'cross_recall', 'cross_f1',
    'yield_precision', 'yield_recall', 'yield_f1', 'mean_iou', 'pct_iou_ge_0.5',
]


def aggregate_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean +/- std per run configuration across model seeds.

    Only run configs (`run_tag`) present are aggregated over whichever seeds
    have a matching row; `n_seeds` reports how many contributed.
    """
    key_cols = [c for c in AGG_KEY_COLS if c in df.columns]
    metric_cols = [c for c in AGG_METRIC_COLS if c in df.columns]
    if not key_cols or not metric_cols or 'seed' not in df.columns:
        return pd.DataFrame()

    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    agg_spec = {}
    for col in metric_cols:
        agg_spec[f'{col}_mean'] = (col, 'mean')
        agg_spec[f'{col}_std'] = (col, 'std')
    agg_spec['n_seeds'] = ('seed', 'nunique')
    agg_spec['seeds'] = ('seed', lambda s: ','.join(str(v) for v in sorted(s.unique())))

    agg = df.groupby(key_cols, as_index=False, dropna=False).agg(**agg_spec)
    for col in metric_cols:
        agg[f'{col}_std'] = agg[f'{col}_std'].fillna(0.0)
        agg[f'{col}_mean'] = agg[f'{col}_mean'].round(4)
        agg[f'{col}_std'] = agg[f'{col}_std'].round(4)

    sort_cols = [c for c in ['context_features', 'context_prompt_mode', 'context_feature_format'] if c in agg.columns]
    if sort_cols:
        agg = agg.sort_values(sort_cols).reset_index(drop=True)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--log_root', default='logs/vlm_eval_context_guided_dot/Qwen2.5-VL-7B-Instruct',
        help='Parent directory containing per-seed subdirs (seed_*/). Each seed is '
        'summarized independently; results are also aggregated (mean +/- std) across seeds.',
    )
    parser.add_argument(
        '--out', default='summary.csv',
        help='Basename for the per-seed summary, written to <log_root>/seed_<N>/summary/<out>. '
        'The cross-seed aggregate is written to <log_root>/summary/<stem>_aggregated<suffix>.',
    )
    args = parser.parse_args()

    log_root = Path(args.log_root)
    out_path = Path(args.out)
    seed_dirs = sorted(log_root.glob('seed_*'))

    if not seed_dirs:
        # No seed_* subdirs: treat log_root itself as a single run directory (legacy behavior).
        rows = collect_rows(log_root)
        if not rows:
            raise SystemExit(f'No *_responses_report.json files found under {log_root}')
        df = pd.DataFrame(rows)
        _write_summary(df, out_path)
        return

    all_rows: list[dict[str, Any]] = []
    for seed_dir in seed_dirs:
        rows = collect_rows(seed_dir)
        if not rows:
            print(f'Warning: no *_responses_report.json files found under {seed_dir}, skipping.')
            continue
        seed_name = seed_dir.name  # e.g. "seed_42"
        seed_num = seed_name.removeprefix('seed_')
        for row in rows:
            row.setdefault('seed', row.get('seed') or seed_num)
            row['seed_dir'] = seed_name
        df_seed = pd.DataFrame(rows)
        seed_out_path = seed_dir / 'summary' / out_path.name
        _write_summary(df_seed, seed_out_path)
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit(f'No *_responses_report.json files found under any seed_* dir in {log_root}')

    df_all = pd.DataFrame(all_rows)
    agg = aggregate_across_seeds(df_all)
    if agg.empty:
        print('Skipping cross-seed aggregation: missing key/seed/metric columns.')
        return

    agg_out_path = log_root / 'summary' / f'{out_path.stem}_aggregated{out_path.suffix}'
    agg_out_path.parent.mkdir(parents=True, exist_ok=True)
    if agg_out_path.suffix in {'.xlsx', '.xls'}:
        agg.to_excel(agg_out_path, index=False)
    else:
        agg.to_csv(agg_out_path, index=False)
    print(f'Wrote {len(agg)} aggregated run configs (across {len(seed_dirs)} seeds) to {agg_out_path}')


if __name__ == '__main__':
    main()
