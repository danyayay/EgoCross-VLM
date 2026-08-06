#!/usr/bin/env python3
"""Parse eval_qwen log and append summary row to CSV.

Usage:
  python scripts/parse_eval_log.py --log-file /path/to/log --csv-file /path/to/output.csv

The script extracts the evaluation arguments and results (saved count, samples/correct/accuracy/macro F1, report path)
and writes/appends a single row to the CSV file.
"""
import re
import ast
import csv
import argparse
import os
from datetime import datetime


def parse_log_file(path):
    eval_args = None
    saved_count = None
    saved_path = None
    samples = None
    correct_count = None
    accuracy = None
    macro_f1 = None
    inference_time = None
    report_path = None

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            # Evaluation arguments line
            m = re.search(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).*Evaluation arguments:\s*(?P<args>.*)$", line)
            if m and not eval_args:
                args_str = m.group('args')
                try:
                    eval_args = ast.literal_eval(args_str)
                except Exception:
                    # fallback: try json-like replace of single quotes
                    try:
                        eval_args = eval(args_str)
                    except Exception:
                        eval_args = None
                continue

            # Saved results line
            m = re.search(r"Saved\s+(?P<count>\d+)\s+results\s+to\s+(?P<path>.+)$", line)
            if m:
                saved_count = int(m.group('count'))
                saved_path = m.group('path').strip()
                continue

            # Samples / Correct / Accuracy / Macro F1
            m = re.search(r"Samples:\s*(?P<samples>\d+),\s*Correct:\s*(?P<correct>\d+)/(?:\d+),\s*Accuracy:\s*(?P<acc>[0-9.]+),\s*Macro F1:\s*(?P<f1>[0-9.]+)", line)
            if m:
                samples = int(m.group('samples'))
                correct_count = int(m.group('correct'))
                accuracy = float(m.group('acc'))
                macro_f1 = float(m.group('f1'))
                continue

            # Inference time
            m = re.search(r"Inference time:\s*(?P<time>[0-9.]+)\s*seconds", line)
            if m:
                try:
                    inference_time = float(m.group('time'))
                except Exception:
                    inference_time = None
                continue

            # Wrote report
            m = re.search(r"Wrote report to\s+(?P<path>.+)$", line)
            if m:
                report_path = m.group('path').strip()
                continue

    return {
        'eval_ts': path.split('/')[-1].split('.log')[0], 
        'eval_args': eval_args,
        'saved_count': saved_count,
        'saved_path': saved_path,
        'samples': samples,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'inference_time': inference_time,
        'report_path': report_path,
    }


def row_from_parsed(parsed):
    args = parsed.get('eval_args') or {}
    row = {
        'timestamp': parsed.get('eval_ts', None),
        'ann_file': args.get('ann_file', '').split('.')[-2],  # just the filename
        'model_name': args.get('model_name', ''),
        'video_root': args.get('video_root', ''),
        'lora_adapter': args.get('lora_adapter', ''),
        'num_frames': args.get('num_frames', ''),
        'max_new_tokens': args.get('max_new_tokens', ''),
        'total_pixels': args.get('total_pixels', ''),
        'min_pixels': args.get('min_pixels', ''),
        'max_frames': args.get('max_frames', ''),
        'sample_fps': args.get('sample_fps', ''),
        'temperature': args.get('temperature', ''),
        'top_p': args.get('top_p', ''),
        'repetition_penalty': args.get('repetition_penalty', ''),
        'seed': args.get('seed', ''),
        'samples': parsed.get('samples', ''),
        'correct_count': parsed.get('correct_count', ''),
        'accuracy': parsed.get('accuracy', ''),
        'macro_f1': parsed.get('macro_f1', ''),
        'inference_time': parsed.get('inference_time', ''),
    }
    return row


def append_row_to_csv(csv_path, row):
    header = [
        'timestamp', 'model_name', 'lora_adapter',
        'samples', 'correct_count', 'accuracy', 'macro_f1', 
        'temperature', 'top_p', 'repetition_penalty', 'seed',
        'num_frames', 'max_new_tokens', 'total_pixels', 'min_pixels', 'max_frames', 'sample_fps',
        'inference_time', 'ann_file', 'video_root', 
    ]

    write_header = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file', default='logs/qwen_eval/20260226_111756.log', help='Path to log file to parse')
    parser.add_argument('--csv-file', required=False, default='eval_qwen_summary.csv', help='CSV file to append to')
    args = parser.parse_args()

    parsed = parse_log_file(args.log_file)
    row = row_from_parsed(parsed)
    append_row_to_csv(args.csv_file, row)
    print(f"Appended summary to {args.csv_file}")


if __name__ == '__main__':
    main()
