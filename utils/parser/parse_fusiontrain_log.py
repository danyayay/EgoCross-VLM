#!/usr/bin/env python3
"""Parse a training log produced by the training script and write a CSV summary.

Usage:
  python3 scripts/parse_training_log.py --log /abs/path/to/training_log.txt --out /abs/path/to/summary.csv

The script extracts:
 - config JSON printed near the top
 - all epoch metrics and picks the best epoch according to config['eval_metric'] (default: 'acc')
 - test metrics and per-class/average metrics
 - writes a single-row CSV (appends if the CSV exists) with flattened info
"""

import argparse
import csv
import json
import os
import re
from collections import OrderedDict


def parse_config(text):
    m = re.search(r"Starting training with config:\s*(\{.+?\})", text, re.S)
    if not m:
        return None
    cfg_text = m.group(1)
    try:
        cfg = json.loads(cfg_text)
    except Exception:
        # try to fix single quotes
        cfg = json.loads(cfg_text.replace("'", '"'))
    return cfg


def parse_epochs(text):
    epochs = []
    for m in re.finditer(r"Epoch\s+(\d+)/(\d+):\s*(.*)", text):
        epoch_no = int(m.group(1))
        rest = m.group(3)
        # find key=value pairs like train_loss=0.123 or lr=4e-05
        kv = {k: v for k, v in re.findall(r"([a-zA-Z0-9_]+)=([0-9.eE+-]+)", rest)}
        # convert numeric
        for k, v in kv.items():
            try:
                kv[k] = float(v)
            except Exception:
                kv[k] = v
        kv['epoch'] = epoch_no
        epochs.append(kv)
    return epochs


def parse_test_metrics(text):
    m = re.search(r"Test:\s*(.*)", text)
    if not m:
        return {}
    rest = m.group(1)
    # comma-separated key=val
    kv = {}
    for part in rest.split(','):
        part = part.strip()
        if '=' in part:
            k, v = part.split('=', 1)
            k = k.strip()
            v = v.strip()
            try:
                kv[k] = float(v)
            except Exception:
                kv[k] = v
    return kv


def parse_per_class_and_averages(text):
    res = {'per_class': [], 'averages': {}}
    m = re.search(r"Per-class metrics:(.*)Averages:\s*", text, re.S)
    per_class_text = None
    if m:
        per_class_text = m.group(1)
    else:
        # maybe 'Per-class metrics:' until end
        m2 = re.search(r"Per-class metrics:(.*)Averages:.*", text, re.S)
        if m2:
            per_class_text = m2.group(1)

    if per_class_text:
        lines = [l.strip() for l in per_class_text.splitlines() if l.strip()]
        # skip header lines until we reach actual rows like: 0 0.7242 0.5923 0.6517 758
        for line in lines:
            if re.match(r"^-+", line):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) >= 5 and re.match(r"^\d+|^\w+", parts[0]):
                # class, precision, recall, f1, support
                cls = parts[0]
                try:
                    prec = float(parts[1]); rec = float(parts[2]); f1 = float(parts[3]); sup = int(parts[4])
                except Exception:
                    continue
                res['per_class'].append({'class': cls, 'precision': prec, 'recall': rec, 'f1': f1, 'support': sup})

    # parse averages lines like: macro - precision: 0.7256, recall: 0.7102, f1: 0.7129
    m_avg = re.search(r"Averages:\s*(.*)", text, re.S)
    if m_avg:
        avg_text = m_avg.group(1)
        # look for macro and micro
        m_macro = re.search(r"macro\s*-\s*precision:\s*([0-9.]+),\s*recall:\s*([0-9.]+),\s*f1:\s*([0-9.]+)", avg_text)
        if m_macro:
            res['averages']['macro_precision'] = float(m_macro.group(1))
            res['averages']['macro_recall'] = float(m_macro.group(2))
            res['averages']['macro_f1'] = float(m_macro.group(3))
        m_micro = re.search(r"micro\s*-\s*precision:\s*([0-9.]+),\s*recall:\s*([0-9.]+),\s*f1:\s*([0-9.]+)", avg_text)
        if m_micro:
            res['averages']['micro_precision'] = float(m_micro.group(1))
            res['averages']['micro_recall'] = float(m_micro.group(2))
            res['averages']['micro_f1'] = float(m_micro.group(3))

    return res


def pick_best_epoch(epochs, eval_metric):
    if not epochs:
        return None
    # map eval_metric to a key in epoch dict
    metric_key = None
    if eval_metric is None:
        eval_metric = 'acc'
    if eval_metric.lower() in ('acc', 'accuracy'):
        metric_key = 'val_acc'
        maximize = True
    elif eval_metric.lower() in ('macrof1', 'f1', 'macro_f1'):
        metric_key = 'val_macrof1'
        maximize = True
    elif eval_metric.lower() in ('loss', 'val_loss'):
        metric_key = 'val_loss'
        maximize = False
    else:
        # fallback to val_acc if present, else val_macrof1, else val_loss
        if 'val_acc' in epochs[0]:
            metric_key = 'val_acc'; maximize = True
        elif 'val_macrof1' in epochs[0]:
            metric_key = 'val_macrof1'; maximize = True
        else:
            metric_key = 'val_loss'; maximize = False

    best = None
    for e in epochs:
        if metric_key not in e:
            continue
        if best is None:
            best = e
            continue
        if maximize:
            if e[metric_key] > best[metric_key]:
                best = e
        else:
            if e[metric_key] < best[metric_key]:
                best = e
    return best, metric_key


def flatten_config(cfg):
    flat = {}
    if not isinstance(cfg, dict):
        return flat
    for k, v in cfg.items():
        if isinstance(v, (str, int, float, bool, list)):
            try:
                if 'anno' in k and 'path' in k:
                    # if key looks like an annotation path, just store the filename
                    flat[f'cfg_{k}'] = v.split('/')[-1] if isinstance(v, str) else v
                else:
                    flat[f'cfg_{k}'] = json.dumps(v) if isinstance(v, (list, dict)) else v
            except Exception:
                flat[f'cfg_{k}'] = str(v)
        else:
            flat[f'cfg_{k}'] = str(v)
    return flat


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--logdir', default='logs/train_fusion', help='path to training_log.txt')
    p.add_argument('--filepath', default='20260308_132144', help='filename of training log within each subdir')
    p.add_argument('--outdir', default='summary.csv', help='path to output CSV (appended)')
    args = p.parse_args()

    out_path = args.outdir

    paths = os.listdir(args.logdir)
    paths.sort()
    for path in paths:
        log_path = os.path.join(args.logdir, path, 'training_log.txt')
        if not os.path.isfile(log_path):
            print(f'Warning: {log_path} does not exist, skipping')
            continue

        with open(log_path, 'r') as f:
            text = f.read()

        cfg = parse_config(text) or {}
        epochs = parse_epochs(text)
        best_info = None
        metric_key = None
        if cfg:
            eval_metric = cfg.get('eval_metric')
        else:
            eval_metric = None
        best = None
        try:
            best, metric_key = pick_best_epoch(epochs, eval_metric)
        except Exception:
            best = None

        test_metrics = parse_test_metrics(text)
        pc = parse_per_class_and_averages(text)

        # prepare row as OrderedDict to keep stable header ordering
        row = OrderedDict()
        row['log_path'] = re.search(r'train_fusion/(\d{8}_\d+)/training_log\.txt', log_path).group(1)
        row['best_epoch'] = best.get('epoch') if best else ''
        row['best_metric_key'] = metric_key or ''
        row['best_metric_value'] = best.get(metric_key) if best and metric_key in best else ''
        row['best_train_loss'] = best.get('train_loss', '') if best else ''
        row['best_val_loss'] = best.get('val_loss', '') if best else ''
        row['best_val_acc'] = best.get('val_acc', '') if best else ''
        row['best_val_macrof1'] = best.get('val_macrof1', '') if best else ''
        row['best_lr'] = best.get('lr', '') if best else ''

        # test metrics
        row['test_loss'] = test_metrics.get('test_loss', '')
        row['test_acc'] = test_metrics.get('test_acc', '')
        row['test_macrof1'] = test_metrics.get('test_macrof1', '')

        # per-class and averages as JSON strings
        # row['per_class'] = json.dumps(pc.get('per_class', []))
        row['avg_macro_precision'] = pc.get('averages', {}).get('macro_precision', '')
        row['avg_macro_recall'] = pc.get('averages', {}).get('macro_recall', '')
        row['avg_macro_f1'] = pc.get('averages', {}).get('macro_f1', '')
        row['avg_micro_precision'] = pc.get('averages', {}).get('micro_precision', '')
        row['avg_micro_recall'] = pc.get('averages', {}).get('micro_recall', '')
        row['avg_micro_f1'] = pc.get('averages', {}).get('micro_f1', '')

        # include full config json and flattened config keys
        # row['config_json'] = json.dumps(cfg)
        flat = flatten_config(cfg)
        for k, v in flat.items():
            # avoid collisions with existing keys
            if k not in row:
                row[k] = v

        # write header if needed and append row
        write_header = not os.path.exists(out_path)
        # os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'a', newline='') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    print(f'Wrote summary to {out_path}')


if __name__ == '__main__':
    main()
