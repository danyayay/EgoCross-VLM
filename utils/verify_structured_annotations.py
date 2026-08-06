"""Verify new structured annotations against legacy generated files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils.eval_utils import build_prompt_from_annotation, infer_task_from_path

_COMPARE_FIELDS = [
    "video_uid", "video_start_sec", "video_end_sec", "video_start_frame",
    "video_end_frame", "video_id", "sample_id", "answer",
    "moment_start_frame", "moment_end_frame", "wrong_answers",
]


def load_by_video_id(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {path}")
    return {str(item["video_id"]): item for item in data}


def norm(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [norm(v) for v in value]
    if isinstance(value, dict):
        return {k: norm(v) for k, v in sorted(value.items())}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare structured annotations with a legacy file.")
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--check_prompts", action="store_true")
    parser.add_argument("--prompt_variant", default="p6")
    parser.add_argument("--context_features", default="none")
    parser.add_argument("--context_feature_fps", default="auto")
    parser.add_argument("--max_prompt_mismatches", type=int, default=10)
    args = parser.parse_args()

    old = load_by_video_id(args.old)
    new = load_by_video_id(args.new)
    old_ids = set(old)
    new_ids = set(new)
    missing = sorted(old_ids - new_ids)
    extra = sorted(new_ids - old_ids)
    field_mismatches = []
    for vid in sorted(old_ids & new_ids):
        for field in _COMPARE_FIELDS:
            if norm(old[vid].get(field)) != norm(new[vid].get(field)):
                field_mismatches.append({
                    "video_id": vid,
                    "field": field,
                    "old": old[vid].get(field),
                    "new": new[vid].get(field),
                })

    prompt_mismatches = []
    if args.check_prompts:
        task_name = infer_task_from_path(args.new)
        for vid in sorted(old_ids & new_ids):
            old_prompt = build_prompt_from_annotation(old[vid], prompt_variant=args.prompt_variant)[0]
            new_prompt = build_prompt_from_annotation(
                new[vid], prompt_variant=args.prompt_variant, task_name=task_name,
                context_features=args.context_features,
                context_feature_fps=args.context_feature_fps,
            )[0]
            if old_prompt != new_prompt:
                prompt_mismatches.append({
                    "video_id": vid,
                    "old_prompt": old_prompt,
                    "new_prompt": new_prompt,
                })
                if len(prompt_mismatches) >= args.max_prompt_mismatches:
                    break

    report = {
        "old": str(args.old),
        "new": str(args.new),
        "old_count": len(old),
        "new_count": len(new),
        "missing_video_ids": missing[:50],
        "extra_video_ids": extra[:50],
        "num_missing": len(missing),
        "num_extra": len(extra),
        "num_field_mismatches": len(field_mismatches),
        "field_mismatches": field_mismatches[:50],
        "num_prompt_mismatches": len(prompt_mismatches),
        "prompt_mismatches": prompt_mismatches,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if missing or extra or field_mismatches or prompt_mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
