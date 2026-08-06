#!/usr/bin/env python3
import json
import re
import sys
import argparse
from collections import defaultdict, Counter

ID_CANDIDATES = ['video_uid', 'video_id', 'video', 'id']
PRED_CANDIDATES = ['pred', 'prediction', 'pred_label', 'predicted', 'y_pred', 'output', 'predictions', 'pred_answer']
GT_CANDIDATES = ['gt', 'label', 'labels', 'target', 'y_true', 'true_label', 'answer']


def find_field(d, candidates):
    for k in candidates:
        if k in d:
            return d[k]
    return None


def normalize_label(v):
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return None
        if len(v) == 1:
            return v[0]
        return tuple(v)
    return v


def extract_label_from_pred_string(s):
    if not isinstance(s, str):
        return s
    m = re.search(r"\)\s*(.*)$", s)
    if m:
        return m.group(1).strip()
    return s.strip()


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    # unwrap if needed
    if isinstance(data, dict):
        for k in ['results', 'data', 'items', 'predictions']:
            if k in data and isinstance(data[k], (list, dict)):
                data = data[k]
                break
    if isinstance(data, dict):
        vals = list(data.values())
        if all(isinstance(x, dict) for x in vals):
            data = vals
    if not isinstance(data, list):
        raise RuntimeError('Unexpected JSON format')
    return data


def compute_metrics(entries, persons):
    # aggregate counters
    total = 0
    correct = 0
    classes = set()
    tp = Counter(); fp = Counter(); fn = Counter()

    for e in entries:
        vid = find_field(e, ID_CANDIDATES)
        if vid is None:
            continue
        vid = str(vid)
        # check if vid starts with any person in list
        if not any(vid.startswith(p) for p in persons):
            continue
        pred = find_field(e, PRED_CANDIDATES)
        gt = find_field(e, GT_CANDIDATES)
        pred = normalize_label(pred); gt = normalize_label(gt)
        if isinstance(pred, str):
            pred = extract_label_from_pred_string(pred)
        if isinstance(gt, str):
            gt = extract_label_from_pred_string(gt)
        if pred is None or gt is None:
            continue
        pred_key = str(pred)
        gt_key = str(gt)
        total += 1
        classes.add(pred_key); classes.add(gt_key)
        if pred_key == gt_key:
            correct += 1
            tp[gt_key] += 1
        else:
            fp[pred_key] += 1
            fn[gt_key] += 1

    # compute accuracy and per-class metrics
    accuracy = (correct / total) if total>0 else 0.0
    per_class = {}
    f1_sum = 0.0; n_class = 0
    for c in sorted(classes):
        _tp = tp.get(c,0); _fp = fp.get(c,0); _fn = fn.get(c,0)
        prec = _tp/(_tp+_fp) if (_tp+_fp)>0 else 0.0
        rec = _tp/(_tp+_fn) if (_tp+_fn)>0 else 0.0
        f1 = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
        per_class[c] = {'precision': prec, 'recall': rec, 'f1': f1, 'tp': _tp, 'fp': _fp, 'fn': _fn}
        f1_sum += f1; n_class += 1
    macro_f1 = f1_sum / n_class if n_class>0 else 0.0
    return {'total': total, 'accuracy': accuracy, 'macro_f1': macro_f1, 'per_class': per_class}


def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--path', type=str,
        help="Path to JSON results file", required=True)
    parser.add_argument('--persons', nargs='+', default=['P12', 'P13', 'P2'], 
        help="List of person prefixes like P13 P12 P2", required=False)
    args = parser.parse_args()

    print(f'Loading data from {args.path}')
    persons = args.persons or []
    if not persons:
        print('No persons provided. Use --persons P13 P12 P2')
        sys.exit(1)
    data = load_json(args.path)
    res = compute_metrics(data, persons)
    print(json.dumps({'persons': persons, 'aggregated': res}, indent=2))

if __name__ == '__main__':
    main()