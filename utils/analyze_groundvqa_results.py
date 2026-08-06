#!/usr/bin/env python3
"""
Simple analyzer for prediction JSON outputs.

Usage examples:
  python3 analyze_results.py --input results/outputs_300_binary.json --out report_binary.json
  python3 analyze_results.py --input results/outputs_300_multi.json --out report_multi.json

The script computes:
 - overall accuracy
 - per-class precision/recall/F1
 - confusion matrix (printed)
 - temporal IoU between ground-truth moment [moment_start_frame,moment_end_frame]
   and predicted [moment_start_frame_pred,moment_end_frame_pred]. Reports mean IoU and % IoU>=0.5

The script writes a small JSON report when --out is provided.
"""
import argparse
import json
from collections import defaultdict, Counter
from typing import List, Tuple
import os

# plotting imports will be optional/guarded so the script still runs without them
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False


def clean_pred_answer(pred_raw: str) -> str:
    # pred_answer tends to be like "(A) yes" or "(C) maintaining speed"
    if pred_raw is None:
        return ""
    # remove leading parenthetical label if present
    parts = pred_raw.split(')', 1)
    if len(parts) == 2 and parts[0].startswith('('):
        return parts[1].strip()
    return pred_raw.strip()


def load_samples(path: str) -> List[dict]:
    with open(path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('expected a list of samples in JSON file')
    return data


def classification_metrics(gts: List[str], preds: List[str]):
    # compute accuracy
    total = len(gts)
    correct = sum(1 for a, b in zip(gts, preds) if a == b)
    acc = correct / total if total else 0.0

    classes = sorted(set(gts) | set(preds))
    tp = Counter()
    fp = Counter()
    fn = Counter()
    cm = {c: Counter() for c in classes}

    for gt, pr in zip(gts, preds):
        cm[gt][pr] += 1
        if gt == pr:
            tp[gt] += 1
        else:
            fp[pr] += 1
            fn[gt] += 1

    per_class = {}
    for c in classes:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        support = sum(1 for x in gts if x == c)
        per_class[c] = {
            'precision': p,
            'recall': r,
            'f1': f1,
            'support': support,
        }

    macro_f1 = sum(m['f1'] for m in per_class.values())/len(per_class) if per_class else 0.0
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'total': total,
        'correct': int(correct),
        'classes': classes,
        'per_class': per_class,
        'confusion': {k: dict(v) for k, v in cm.items()},
    }


def interval_iou(gt_s: float, gt_e: float, pred_s: float, pred_e: float) -> float:
    # normalize: ensure start <= end
    if pred_e < pred_s:
        pred_s, pred_e = pred_e, pred_s
    inter_s = max(gt_s, pred_s)
    inter_e = min(gt_e, pred_e)
    intersection = max(0.0, inter_e - inter_s)
    union = (gt_e - gt_s) + max(0.0, pred_e - pred_s) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def analyze(path: str, top_k_wrong: int = 10):
    samples = load_samples(path)

    gts = []
    preds = []
    ious = []
    wrong_examples = []

    for s in samples:
        if 'error' in s:
            continue
        gt = s.get('answer')
        pred_raw = s.get('pred_answer')
        pred = clean_pred_answer(pred_raw) if pred_raw is not None else 'unknown'
        gts.append(gt)
        preds.append(pred)

        gt_s = float(s.get('moment_start_frame', 0.0) or 0.0)
        gt_e = float(s.get('moment_end_frame', 0.0) or 0.0)
        pred_s = float(s.get('moment_start_frame_pred', 0.0) or 0.0)
        pred_e = float(s.get('moment_end_frame_pred', 0.0) or 0.0)
        iou = interval_iou(gt_s, gt_e, pred_s, pred_e)
        ious.append(iou)

        if gt != pred:
            wrong_examples.append({
                'sample_id': s.get('sample_id'),
                'video_id': s.get('video_id'),
                'question': s.get('question'),
                'gt': gt,
                'pred': pred,
                'pred_raw': pred_raw,
                'moment_start_frame_pred': pred_s,
                'moment_end_frame_pred': pred_e,
                'iou': iou,
            })

    cls_metrics = classification_metrics(gts, preds)

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    pct_iou_ge_05 = sum(1 for v in ious if v >= 0.5) / len(ious) if ious else 0.0

    report = {
        'input': path,
        'num_samples': len(samples),
        'classification': cls_metrics,
        'temporal_iou': {
            'mean_iou': mean_iou,
            'pct_iou_ge_0.5': pct_iou_ge_05,
        },
        # 'top_wrong_examples': wrong_examples[:top_k_wrong],
    }

    return report


def pretty_print_report(r: dict, logger=None):
    """Print the report to stdout and, if logger is provided, also to the log file."""
    import logging as _logging

    def _out(line: str):
        print(line)
        if logger is not None:
            logger.info(line)

    c = r['classification']
    _out(f"Input: {r['input']}")
    _out(f"Samples: {r['num_samples']}, Correct: {c['correct']}/{c['total']}, Accuracy: {c['accuracy']:.4f}, Macro F1: {c['macro_f1']:.4f}")
    _out('\nPer-class metrics:')
    _out(f"{'class':30} {'precision':9} {'recall':9} {'f1':9} {'support':7}")
    for cls in c['classes']:
        m = c['per_class'][cls]
        _out(f"{cls:30} {m['precision']:.4f}    {m['recall']:.4f}    {m['f1']:.4f}    {m['support']:7}")

    _out('\nConfusion matrix (rows=gt, cols=pred):')
    classes = c['classes']
    _out(''.ljust(32) + ' '.join(f"{x[:10]:>10}" for x in classes))
    for gt in classes:
        row = c['confusion'].get(gt, {})
        line = f"{gt[:30]:30} "
        for pred in classes:
            line += f"{row.get(pred, 0):10}"
        _out(line)

    ti = r['temporal_iou']
    _out('\nTemporal IoU:')
    _out(f"Mean IoU: {ti['mean_iou']:.4f}")
    _out(f"% IoU >= 0.5: {ti['pct_iou_ge_0.5']*100:.2f}%")

    if r.get('top_wrong_examples'):
        _out('\nTop wrong examples:')
        for e in r['top_wrong_examples']:
            _out(f"{e['sample_id']:12} gt={e['gt']!r:12} pred={e['pred']!r:12} iou={e['iou']:.3f} pred_end={e['moment_end_frame_pred']}")


def plot_confusion_heatmap(confusion: dict, classes: List[str], out_path: str):
    """Plot and save a confusion matrix heatmap.

    confusion: mapping gt -> {pred: count}
    classes: list of class labels (ordered)
    out_path: output filename (PNG)
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available; skipping confusion heatmap.")
        return

    # build matrix
    n = len(classes)
    mat = np.zeros((n, n), dtype=int)
    idx = {c: i for i, c in enumerate(classes)}
    for gt, row in confusion.items():
        for pr, cnt in row.items():
            if gt in idx and pr in idx:
                mat[idx[gt], idx[pr]] = int(cnt)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.figure(figsize=(max(6, n * 0.5), max(4, n * 0.4)))
    sns.set(font_scale=0.9)
    ax = sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground truth')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_per_class_bars(per_class: dict, classes: List[str], out_path: str):
    """Plot per-class precision/recall/F1 as grouped bars and save PNG.

    per_class: mapping class -> {precision, recall, f1, support}
    classes: ordered list of classes
    out_path: output filename
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available; skipping per-class metric bars.")
        return

    precisions = [per_class[c]['precision'] for c in classes]
    recalls = [per_class[c]['recall'] for c in classes]
    f1s = [per_class[c]['f1'] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.figure(figsize=(max(8, len(classes) * 0.5), 5))
    plt.bar(x - width, precisions, width, label='precision')
    plt.bar(x, recalls, width, label='recall')
    plt.bar(x + width, f1s, width, label='f1')
    plt.xticks(x, classes, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.ylabel('Score')
    plt.title('Per-class precision / recall / F1')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filename', '-i', default='Qwen3-VL-2B-Instruct_responses.json', help='path to JSON results file')
    parser.add_argument('--input_dir', '-d', default='results/qwen/', help='directory containing JSON results files')
    parser.add_argument('--out_dir', '-o', default='results/qwen/', help='optional path to write JSON report')
    parser.add_argument('--topk', type=int, default=3, help='how many wrong examples to include')
    parser.add_argument('--plot', type=bool, default=False, help='save confusion heatmap and per-class metric bars')
    parser.add_argument('--plots_dir', default='results/plots', help='directory to save plots when --plot is set')
    args = parser.parse_args()

    input = os.path.join(args.input_dir, args.input_filename)
    report = analyze(input, top_k_wrong=args.topk)
    pretty_print_report(report)

    if args.out_dir:
        out = os.path.join(args.out_dir, args.input_filename.split('.')[0] + '_report.json')
        os.makedirs(args.out_dir, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote report to {out}")

    if args.plot:
        # derive file-stem to name plots
        stem = args.input_filename.split('.')[0]
        os.makedirs(args.plots_dir, exist_ok=True)
        conf_out = os.path.join(args.plots_dir, f"{stem}_confusion.png")
        perclass_out = os.path.join(args.plots_dir, f"{stem}_per_class.png")

        cls = report.get('classification', {})
        classes = cls.get('classes', [])
        confusion = cls.get('confusion', {})
        per_class = cls.get('per_class', {})

        try:
            plot_confusion_heatmap(confusion, classes, conf_out)
            print(f"Saved confusion heatmap to {conf_out}")
        except Exception as e:
            print(f"Failed to plot confusion heatmap: {e}")

        try:
            plot_per_class_bars(per_class, classes, perclass_out)
            print(f"Saved per-class metric bars to {perclass_out}")
        except Exception as e:
            print(f"Failed to plot per-class bars: {e}")


if __name__ == '__main__':
    main()
