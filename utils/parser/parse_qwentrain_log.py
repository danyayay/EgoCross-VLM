#!/usr/bin/env python3
"""Parse training log(s) and extract config, model params, best validation metrics, and test results.

Usage:
  python scripts/parse_train_log.py /path/to/train.log --out outputs/qwen_train_parsed.csv

The script writes a CSV (overwrites by default) with one row per log file.
"""
import re
import csv
import argparse
import ast
from pathlib import Path


def parse_arguments_line(text):
    m = re.search(r"Arguments:\s*Namespace\((.*)\)", text)
    if not m:
        return {}
    args_blob = m.group(1)
    # find key=value pairs where value may be quoted string or token without comma
    pairs = re.findall(r"(\w+)=('.*?'|\".*?\"|[^,\)]+)", args_blob)
    result = {}
    for k, v in pairs:
        v = v.strip()
        if (v.startswith("'") and v.endswith("'") ) or (v.startswith('"') and v.endswith('"')):
            result[k] = v[1:-1]
            continue
        if v in ("True", "False"):
            result[k] = v == "True"
            continue
        # try int/float
        try:
            if '.' in v:
                result[k] = float(v)
            else:
                result[k] = int(v)
            continue
        except Exception:
            result[k] = v
    return result


def parse_model_params(text):
    m = re.search(r"Model parameters:\s*total=([0-9,]+),\s*trainable=([0-9,]+),\s*non_trainable=([0-9,]+)", text)
    if not m:
        return {}
    total = int(m.group(1).replace(',', ''))
    trainable = int(m.group(2).replace(',', ''))
    non_trainable = int(m.group(3).replace(',', ''))
    return {
        'model_total_params': total,
        'model_trainable_params': trainable,
        'model_non_trainable_params': non_trainable,
    }


def parse_epoch_val_metrics(text):
    # find all lines like: Epoch 1 val metrics: {'val_acc': 0.6456, 'val_macro_f1': 0.4964}
    epochs = []
    for m in re.finditer(r"Epoch\s+(\d+)\s+val metrics:\s*(\{.*?\})", text):
        epoch = int(m.group(1))
        blob = m.group(2)
        try:
            metrics = ast.literal_eval(blob)
        except Exception:
            metrics = {}
        epochs.append((epoch, metrics))
    return epochs


def parse_final_test_results(text):
    result = {}
    # Deterministic block: capture optional inference time and final results
    m_det = re.search(r"### Deterministic generation:.*?Inference time:\s*([0-9.]+)\s*seconds.*?Final test results:\s*(\{.*?\})", text, re.S)
    result['deterministic_test_acc'] = None
    result['deterministic_test_macro_f1'] = None
    result['deterministic_inference_time'] = None
    if m_det:
        try:
            infer_t = float(m_det.group(1))
            d = ast.literal_eval(m_det.group(2))
            result['deterministic_inference_time'] = infer_t
            result['deterministic_test_acc'] = d.get('test_acc')
            result['deterministic_test_macro_f1'] = d.get('test_macro_f1')
        except Exception:
            pass
    else:
        # fallback: maybe inference time appears elsewhere in the block
        m_det2 = re.search(r"### Deterministic generation:.*?Final test results:\s*(\{.*?\})", text, re.S)
        if m_det2:
            try:
                d = ast.literal_eval(m_det2.group(1))
                result['deterministic_test_acc'] = d.get('test_acc')
                result['deterministic_test_macro_f1'] = d.get('test_macro_f1')
            except Exception:
                pass
        m_det_time = re.search(r"### Deterministic generation:.*?Inference time:\s*([0-9.]+)\s*seconds", text, re.S)
        if m_det_time:
            try:
                result['deterministic_inference_time'] = float(m_det_time.group(1))
            except Exception:
                pass

    # Default block: capture optional inference time and final results
    m_def = re.search(r"### Default generation:.*?Inference time:\s*([0-9.]+)\s*seconds.*?Final test results:\s*(\{.*?\})", text, re.S)
    result['default_test_acc'] = None
    result['default_test_macro_f1'] = None
    result['default_inference_time'] = None
    if m_def:
        try:
            infer_t = float(m_def.group(1))
            d = ast.literal_eval(m_def.group(2))
            result['default_inference_time'] = infer_t
            result['default_test_acc'] = d.get('test_acc')
            result['default_test_macro_f1'] = d.get('test_macro_f1')
        except Exception:
            pass
    else:
        m_def2 = re.search(r"### Default generation:.*?Final test results:\s*(\{.*?\})", text, re.S)
        if m_def2:
            try:
                d = ast.literal_eval(m_def2.group(1))
                result['default_test_acc'] = d.get('test_acc')
                result['default_test_macro_f1'] = d.get('test_macro_f1')
            except Exception:
                pass
        m_def_time = re.search(r"### Default generation:.*?Inference time:\s*([0-9.]+)\s*seconds", text, re.S)
        if m_def_time:
            try:
                result['default_inference_time'] = float(m_def_time.group(1))
            except Exception:
                pass
    return result


def parse_saved_checkpoint_lines(text):
    # Example: Saved candidate checkpoint to ... best-val_macro_f1-0.4964-step-29 with val_macro_f1=0.4964
    saved = []
    for m in re.finditer(r"Saved candidate checkpoint to\s+(\S+)\s+with val_macro_f1=([0-9.]+)", text):
        path = m.group(1)
        val_macro = float(m.group(2))
        saved.append((path, val_macro))
    return saved


def process_log(path: Path):
    text = path.read_text(encoding='utf-8', errors='ignore')
    out = {}
    out['log_path'] = str(path).split('/')[-2]
    out['gaze_overlay_video'] = True if 'clips_overlay' in text else False
    out['explain_overlay_prompt'] = True if 'groundvqa_overlay' in text else False

    args = parse_arguments_line(text)
    # include some common args as top-level columns if present
    for k in ['model_name','epochs','batch_size','lr','ft_type','lora_rank','lora_alpha','random_seed','monitor','top_k','eval_deterministic','eval_max_new_tokens','eval_batch_size','ann_file_train','ann_file_val','ann_file_test']:
        if k in args:
            out[k] = args[k]
    out['data_setup'] = re.search(r'_(\d+)_', out['ann_file_train']).group(1) if out['ann_file_train'] else None
    out.update(parse_model_params(text))

    epochs = parse_epoch_val_metrics(text)
    # determine best by val_macro_f1
    best_epoch = None
    best_macro = None
    best_acc_at_best = None
    for e, metrics in epochs:
        mv = metrics.get('val_macro_f1')
        if mv is None:
            continue
        if best_macro is None or mv > best_macro:
            best_macro = mv
            best_epoch = e
            best_acc_at_best = metrics.get('val_acc')
    if best_epoch is not None:
        out['best_val_epoch'] = best_epoch
        out['best_val_macro_f1'] = best_macro
        out['best_val_acc_at_best_epoch'] = best_acc_at_best
    else:
        out['best_val_epoch'] = None
        out['best_val_macro_f1'] = None
        out['best_val_acc_at_best_epoch'] = None

    out.update(parse_final_test_results(text))

    # Also include any saved checkpoint summaries (for reference)
    saved = parse_saved_checkpoint_lines(text)
    if saved:
        # flatten into strings
        out['saved_checkpoints'] = ';'.join([f"{p}|{v}" for p, v in saved])

    return out


def write_csv(rows, outpath: Path, overwrite=False):
    if not rows:
        print('No rows to write')
        return
    # collect all keys
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    # ensure some preferred ordering
    preferred = ['log_path','model_name','epochs','batch_size','lr','ft_type','lora_rank','lora_alpha','data_setup', 'overlay', 
                 'random_seed','monitor','top_k','eval_deterministic','eval_max_new_tokens','eval_batch_size',
                 'model_total_params','model_trainable_params','model_non_trainable_params','best_val_epoch',
                 'best_val_macro_f1','best_val_acc_at_best_epoch','deterministic_test_acc','deterministic_test_macro_f1',
                 'deterministic_inference_time','default_test_acc','default_test_macro_f1','default_inference_time',
                 'saved_checkpoints','ann_file_train','ann_file_val','ann_file_test']
    final_keys = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]

    outpath.parent.mkdir(parents=True, exist_ok=True)
    mode = 'w' if overwrite else 'a'
    write_header = True
    if not overwrite and outpath.exists() and outpath.stat().st_size > 0:
        write_header = False
    with outpath.open(mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=final_keys)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in final_keys})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log_file', default='logs/qwen_train/Qwen3-VL-2B-Instruct/20260226_155829/train.log', 
                   help='Path(s) to train.log files to parse')
    p.add_argument('--out', default='qwenft_summary.csv', help='Output CSV path')
    args = p.parse_args()

    rows = []
    path = Path(args.log_file)
    if not path.exists():
        print(f"Skipping missing file: {path}")
        return 
    try:
        rows.append(process_log(path))
    except Exception as e:
        print(f"Error parsing {path}: {e}")

    write_csv(rows, Path(args.out), overwrite=False)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == '__main__':
    main()
