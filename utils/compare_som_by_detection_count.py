#!/usr/bin/env python3
"""Compare SoM vs no-SoM VLM predictions by detected object count.

Splits clips into:
  - <= N detected objects across the whole clip
  - > N detected objects across the whole clip

By default, "objects" are unique track_id values in the detection cache. This
matches the SoM overlay numbering when tracking is enabled.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.analyze_groundvqa_results import classification_metrics, clean_pred_answer


DEFAULT_SOM_DIR = Path(
    "logs/vlm_eval_newformat/Qwen2.5-VL-7B-Instruct/"
    "tcotnone_f8_interleaved_vcotsom_overlay_nobg+none_p6_int8"
)
DEFAULT_NO_SOM_DIR = Path(
    "logs/vlm_eval_newformat/Qwen2.5-VL-7B-Instruct/"
    "cotnocot_f8_interleaved_p6_int8"
)
DEFAULT_DETECTION_DIR = Path(
    "~/.cache/visual_cot/detection/"
    "clips_groundingdinotiny_011fc2e60c_f8_fps4_dur2"
).expanduser()
REPO_DETECTION_DIR = Path(".cache/visual_cot/detections")


def resolve_responses_path(path: Path) -> Path:
    """Return a responses JSON path, choosing the newest one from a run dir."""
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"prediction path does not exist: {path}")
    candidates = sorted(
        (p for p in path.glob("*_responses.json") if not p.name.endswith("_report.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no *_responses.json found in {path}")
    return candidates[0]


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def load_predictions(path: Path, *, finished_only: bool) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"expected list in prediction file: {path}")

    out = {}
    duplicates = Counter()
    for sample in data:
        if "error" in sample:
            continue
        if finished_only and not sample.get("is_finished", True):
            continue
        video_id = sample.get("video_id") or sample.get("sample_id")
        if not video_id:
            continue
        if video_id in out:
            duplicates[video_id] += 1
        out[str(video_id)] = sample

    if duplicates:
        print(f"Warning: {path} has {len(duplicates)} duplicate video_ids; kept last occurrence.")
    return out


def resolve_detection_dir(path: Path) -> Path:
    if path.exists():
        return path
    if path == DEFAULT_DETECTION_DIR and REPO_DETECTION_DIR.exists():
        print(
            "Warning: default ~/.cache detection directory was not found; "
            f"using repo-local {REPO_DETECTION_DIR}."
        )
        return REPO_DETECTION_DIR
    raise FileNotFoundError(f"detection directory does not exist: {path}")


def video_id_from_detection_path(path: Path) -> str:
    return path.name.split("__", 1)[0]


def detection_files_by_video_id(detection_dir: Path) -> dict[str, Path]:
    files = sorted(detection_dir.glob("*.json"))
    out: dict[str, Path] = {}
    for path in files:
        video_id = video_id_from_detection_path(path)
        previous = out.get(video_id)
        if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
            out[video_id] = path
    return out


def frames_from_detection_cache(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict) and "frames" in data:
        frames = data["frames"]
    elif isinstance(data, list):
        frames = data
    else:
        raise ValueError(f"unsupported detection schema: {path}")
    if not isinstance(frames, list):
        raise ValueError(f"detection frames are not a list: {path}")
    return frames


def count_objects(frames: list[dict[str, Any]], mode: str) -> int:
    detections_by_frame = [
        frame.get("detections", []) for frame in frames if isinstance(frame, dict)
    ]
    if mode == "total_detections":
        return sum(len(dets) for dets in detections_by_frame)
    if mode == "max_frame":
        return max((len(dets) for dets in detections_by_frame), default=0)

    track_ids = set()
    for dets in detections_by_frame:
        for det in dets:
            if isinstance(det, dict) and det.get("track_id") is not None:
                track_ids.add(str(det["track_id"]))
    if track_ids:
        return len(track_ids)

    # Old caches might not have tracking. Fall back to max simultaneous boxes,
    # which is conservative for the ">3 objects in the clip" split.
    return max((len(dets) for dets in detections_by_frame), default=0)


def build_object_counts(detection_dir: Path, mode: str) -> dict[str, int]:
    counts = {}
    for video_id, path in detection_files_by_video_id(detection_dir).items():
        counts[video_id] = count_objects(frames_from_detection_cache(path), mode)
    return counts


def pred_label(sample: dict[str, Any]) -> str:
    pred_raw = sample.get("pred_answer")
    return clean_pred_answer(pred_raw) if pred_raw is not None else "unknown"


def sample_correct(sample: dict[str, Any]) -> bool:
    return sample.get("answer") == pred_label(sample)


def metrics_for(video_ids: list[str], preds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gts = [preds[vid].get("answer") for vid in video_ids]
    pred_labels = [pred_label(preds[vid]) for vid in video_ids]
    return classification_metrics(gts, pred_labels)


def paired_summary(video_ids: list[str], som: dict[str, dict[str, Any]],
                   no_som: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    examples = {
        "som_only_correct": [],
        "no_som_only_correct": [],
    }
    for video_id in video_ids:
        som_ok = sample_correct(som[video_id])
        no_som_ok = sample_correct(no_som[video_id])
        if som_ok and no_som_ok:
            counts["both_correct"] += 1
        elif som_ok:
            counts["som_only_correct"] += 1
            examples["som_only_correct"].append(video_id)
        elif no_som_ok:
            counts["no_som_only_correct"] += 1
            examples["no_som_only_correct"].append(video_id)
        else:
            counts["both_wrong"] += 1
    return {
        **dict(counts),
        "examples": {k: v[:20] for k, v in examples.items()},
    }


def summarize_bin(name: str, video_ids: list[str], object_counts: dict[str, int],
                  som: dict[str, dict[str, Any]],
                  no_som: dict[str, dict[str, Any]]) -> dict[str, Any]:
    som_metrics = metrics_for(video_ids, som) if video_ids else classification_metrics([], [])
    no_som_metrics = metrics_for(video_ids, no_som) if video_ids else classification_metrics([], [])
    som_acc = som_metrics["accuracy"]
    no_som_acc = no_som_metrics["accuracy"]
    som_f1 = som_metrics["macro_f1"]
    no_som_f1 = no_som_metrics["macro_f1"]
    counts = [object_counts[vid] for vid in video_ids]
    return {
        "name": name,
        "num_clips": len(video_ids),
        "object_count_min": min(counts) if counts else None,
        "object_count_max": max(counts) if counts else None,
        "object_count_mean": sum(counts) / len(counts) if counts else None,
        "som": som_metrics,
        "no_som": no_som_metrics,
        "delta_som_minus_no_som": {
            "accuracy": som_acc - no_som_acc,
            "macro_f1": som_f1 - no_som_f1,
        },
        "paired": paired_summary(video_ids, som, no_som),
    }


def print_bin_summary(summary: dict[str, Any]) -> None:
    print(f"\n{summary['name']}")
    print("-" * len(summary["name"]))
    print(
        f"clips={summary['num_clips']} "
        f"objects[min/mean/max]={summary['object_count_min']}/"
        f"{summary['object_count_mean']:.2f if summary['object_count_mean'] is not None else 'NA'}/"
        f"{summary['object_count_max']}"
    )
    for key, label in (("no_som", "without SoM"), ("som", "with SoM")):
        cls = summary[key]
        print(
            f"{label:12s}: acc={cls['accuracy']:.4f} "
            f"macro_f1={cls['macro_f1']:.4f} correct={cls['correct']}/{cls['total']}"
        )
    delta = summary["delta_som_minus_no_som"]
    paired = summary["paired"]
    print(f"delta SoM-noSoM: acc={delta['accuracy']:+.4f} macro_f1={delta['macro_f1']:+.4f}")
    print(
        "paired: "
        f"both_correct={paired.get('both_correct', 0)}, "
        f"som_only={paired.get('som_only_correct', 0)}, "
        f"no_som_only={paired.get('no_som_only_correct', 0)}, "
        f"both_wrong={paired.get('both_wrong', 0)}"
    )


def fmt_count(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SoM vs no-SoM eval results split by detected object count."
    )
    parser.add_argument("--som", type=Path, default=DEFAULT_SOM_DIR,
                        help="SoM run directory or *_responses.json file")
    parser.add_argument("--no_som", type=Path, default=DEFAULT_NO_SOM_DIR,
                        help="No-SoM run directory or *_responses.json file")
    parser.add_argument("--detection_dir", type=Path, default=DEFAULT_DETECTION_DIR,
                        help="Directory containing detection cache JSON files")
    parser.add_argument("--threshold", type=int, default=3,
                        help="Split threshold: <= threshold vs > threshold objects")
    parser.add_argument("--object_count_mode", choices=["track_ids", "max_frame", "total_detections"],
                        default="track_ids",
                        help="How to count objects in a clip")
    parser.add_argument("--include_unfinished", action="store_true",
                        help="Include samples whose is_finished is false")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional JSON report output path")
    args = parser.parse_args()

    som_path = resolve_responses_path(args.som)
    no_som_path = resolve_responses_path(args.no_som)
    detection_dir = resolve_detection_dir(args.detection_dir.expanduser())

    som = load_predictions(som_path, finished_only=not args.include_unfinished)
    no_som = load_predictions(no_som_path, finished_only=not args.include_unfinished)
    object_counts = build_object_counts(detection_dir, args.object_count_mode)

    common = sorted(set(som) & set(no_som) & set(object_counts))
    missing = {
        "som_only_predictions": len(set(som) - set(no_som)),
        "no_som_only_predictions": len(set(no_som) - set(som)),
        "missing_detection_for_common_predictions": len((set(som) & set(no_som)) - set(object_counts)),
    }

    le_ids = [vid for vid in common if object_counts[vid] <= args.threshold]
    gt_ids = [vid for vid in common if object_counts[vid] > args.threshold]

    report = {
        "inputs": {
            "som": str(som_path),
            "no_som": str(no_som_path),
            "detection_dir": str(detection_dir),
            "object_count_mode": args.object_count_mode,
            "threshold": args.threshold,
            "include_unfinished": args.include_unfinished,
        },
        "coverage": {
            "som_predictions": len(som),
            "no_som_predictions": len(no_som),
            "detection_clips": len(object_counts),
            "paired_evaluable_clips": len(common),
            **missing,
        },
        "bins": {
            f"objects_le_{args.threshold}": summarize_bin(
                f"Objects <= {args.threshold}", le_ids, object_counts, som, no_som
            ),
            f"objects_gt_{args.threshold}": summarize_bin(
                f"Objects > {args.threshold}", gt_ids, object_counts, som, no_som
            ),
        },
    }

    print("Inputs")
    print("------")
    print(f"SoM responses:    {som_path}")
    print(f"No-SoM responses: {no_som_path}")
    print(f"Detections:       {detection_dir}")
    print(f"Object count:     {args.object_count_mode}")
    print("\nCoverage")
    print("--------")
    for key, value in report["coverage"].items():
        print(f"{key}: {value}")

    for summary in report["bins"].values():
        mean = summary["object_count_mean"]
        print(f"\n{summary['name']}")
        print("-" * len(summary["name"]))
        print(
            f"clips={summary['num_clips']} "
            f"objects[min/mean/max]={summary['object_count_min']}/"
            f"{fmt_count(mean)}/{summary['object_count_max']}"
        )
        for key, label in (("no_som", "without SoM"), ("som", "with SoM")):
            cls = summary[key]
            print(
                f"{label:12s}: acc={cls['accuracy']:.4f} "
                f"macro_f1={cls['macro_f1']:.4f} correct={cls['correct']}/{cls['total']}"
            )
        delta = summary["delta_som_minus_no_som"]
        paired = summary["paired"]
        print(f"delta SoM-noSoM: acc={delta['accuracy']:+.4f} macro_f1={delta['macro_f1']:+.4f}")
        print(
            "paired: "
            f"both_correct={paired.get('both_correct', 0)}, "
            f"som_only={paired.get('som_only_correct', 0)}, "
            f"no_som_only={paired.get('no_som_only_correct', 0)}, "
            f"both_wrong={paired.get('both_wrong', 0)}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nWrote report to {args.out}")


if __name__ == "__main__":
    main()
