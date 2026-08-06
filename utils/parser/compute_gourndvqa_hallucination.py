#!/usr/bin/env python3
"""
Compute how many records in a JSON file have a predicted answer not in a target set.

Usage:
  python scripts/compute_pred_mismatch.py /path/to/file.json

The script expects the JSON file to contain either:
 - a list of records (dicts), or
 - an object with a top-level key (like 'records' or 'data') containing the list.

It will look for a prediction field in each record called one of:
  'pred_answer', 'pred', 'prediction', 'pred_answer_str'
You can modify the TARGETS list below if needed.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Adjust these targets if the canonical forms differ in your dataset.
TARGETS = {"(A) yield", "(B) yield", "(A) cross", "(B) cross"}

PRED_KEYS = ["pred_answer", "pred", "prediction", "pred_answer_str", "pred_ans"]


def load_records(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    # If top-level is a dict and contains a list, try common keys
    if isinstance(data, dict):
        for key in ("records", "data", "items", "results"):
            if key in data and isinstance(data[key], list):
                return data[key]
        # If dict looks like a single record, wrap it
        # But we prefer to search for nested lists
        # Fall back: try to find any list at top-level
        for v in data.values():
            if isinstance(v, list):
                return v
        # No list found; treat the dict itself as single record
        return [data]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Unsupported JSON top-level structure: expected list or dict")


def get_pred_value(record: Dict[str, Any]) -> str:
    # Try common keys first
    for k in PRED_KEYS:
        if k in record:
            v = record[k]
            if v is None:
                return ""
            return str(v).strip()
    # Try to find any value that looks like a short string answer
    for v in record.values():
        if isinstance(v, str) and len(v) <= 100:
            # heuristic: return the first string-like field
            return v.strip()
    return ""


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python scripts/compute_pred_mismatch.py /path/to/file.json")
        return 2

    path = Path(argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        return 3

    records = load_records(path)
    total = len(records)
    mismatch_count = 0

    for rec in records:
        pred = get_pred_value(rec)
        if pred not in TARGETS:
            mismatch_count += 1

    pct = (mismatch_count / total * 100) if total else 0.0

    print(f"File: {path}")
    print(f"Total records: {total}")
    print(f"Predicted answers NOT in target set: {mismatch_count}")
    print(f"Percentage not in target set: {pct:.2f}%")

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))


# python3 scripts/compute_gourndvqa_hallucination.py logs/groundvqa_eval/generalize__00010_every10_qn0.json
# python3 scripts/compute_gourndvqa_hallucination.py logs/groundvqa_eval/generalize__00100_every10_qn0.json