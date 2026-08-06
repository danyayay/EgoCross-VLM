"""Generate cached bbox detections for visual CoT evaluation.

Writes JSON files consumed by ``utils.visual_cot`` / ``training.eval_vlm``.
Example:
    conda run -n work3 python -m dataprep.generate_vcot_detections \
        --ann_file features/groundvqa_qn3/annotations.VRbinary_00000_test_close.json \
        --video_root data/videodata_256/clips \
        --num_frames 4 \
        --vcot_labels "automated vehicle,white circle on the ground" \
        --video_ids P47S3-1,P38S5-2,P48S9-2
"""

from __future__ import annotations

import argparse
import json
import inspect
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from utils.eval_utils import get_video_frames, _sample_stratified
from utils.visual_cot import (
    detection_cache_path,
    offline_overlay_video_path,
    parse_vcot_labels,
    render_detection_overlays,
    save_detection_cache,
)
from dataprep.v3overlay_gaze import encode_rgb_frames_to_video


def _mean_timestamp(timestamp: Any) -> float:
    arr = np.asarray(timestamp)
    return float(arr.mean()) if arr.size else 0.0


def _prompt_from_labels(labels: list[str]) -> str:
    # GroundingDINO expects class phrases separated by periods.
    return ". ".join(labels) + "."


def _to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if hasattr(value, "to") else value
    return out


def _norm_label(text: str) -> str:
    return " ".join(str(text).lower().replace(".", " ").split())


def _canonical_label_or_none(label_text: str, labels: list[str]) -> str | None:
    """Map a returned GroundingDINO phrase to one requested label when unambiguous."""
    norm = _norm_label(label_text)
    canonical = {_norm_label(label): label for label in labels}
    if norm in canonical:
        return canonical[norm]

    matches = [label for label in labels if _norm_label(label) in norm]
    if len(matches) == 1:
        return matches[0]
    return None


def _postprocess_grounding_dino(processor, outputs, inputs, image, labels: list[str], args, device: str):
    target_sizes = torch.tensor([image.size[::-1]], device=device)
    if hasattr(processor, "post_process_grounded_object_detection"):
        postprocess = processor.post_process_grounded_object_detection
        params = inspect.signature(postprocess).parameters
        kwargs = {
            "outputs": outputs,
            "input_ids": inputs.get("input_ids"),
            "text_threshold": args.text_threshold,
            "target_sizes": target_sizes,
        }
        if "box_threshold" in params:
            kwargs["box_threshold"] = args.box_threshold
        else:
            kwargs["threshold"] = args.box_threshold
        if "text_labels" in params:
            kwargs["text_labels"] = [labels]
        return postprocess(**kwargs)[0]

    return processor.post_process_object_detection(
        outputs,
        threshold=args.box_threshold,
        target_sizes=target_sizes,
    )[0]


def _detect_frame_prompt(
    processor,
    model,
    frame: np.ndarray,
    prompt_labels: list[str],
    all_labels: list[str],
    args,
    device: str,
    forced_label: str | None = None,
) -> list[dict[str, Any]]:
    image = Image.fromarray(frame)
    text = _prompt_from_labels(prompt_labels)
    inputs = processor(images=image, text=text, return_tensors="pt")
    inputs = _to_device(inputs, device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = _postprocess_grounding_dino(processor, outputs, inputs, image, prompt_labels, args, device)

    detections = []
    boxes = results.get("boxes", [])
    scores = results.get("scores", [])
    result_labels = results.get("labels", [])
    for box, score, label in zip(boxes, scores, result_labels):
        if isinstance(label, torch.Tensor):
            idx = int(label.item())
            raw_label = prompt_labels[idx] if idx < len(prompt_labels) else str(idx)
        else:
            raw_label = str(label).strip()

        box_values = [round(float(x), 2) for x in box.detach().cpu().tolist()]
        label_text = forced_label or _canonical_label_or_none(raw_label, all_labels)
        if label_text is None:
            if args.ambiguous_label_policy == "skip":
                continue
            if args.ambiguous_label_policy == "raw":
                label_text = raw_label
            else:
                label_text = all_labels[0] if all_labels else raw_label

        det = {
            "label": label_text,
            "box": box_values,
            "score": round(float(score.detach().cpu().item()), 4),
        }
        if _norm_label(raw_label) != _norm_label(label_text):
            det["raw_label"] = raw_label
        detections.append(det)
    return detections


def detect_frame(processor, model, frame: np.ndarray, labels: list[str], args, device: str) -> list[dict[str, Any]]:
    if args.detect_per_label:
        detections: list[dict[str, Any]] = []
        for label in labels:
            detections.extend(
                _detect_frame_prompt(
                    processor,
                    model,
                    frame,
                    [label],
                    labels,
                    args,
                    device,
                    forced_label=label,
                )
            )
        return detections

    return _detect_frame_prompt(processor, model, frame, labels, labels, args, device)


def _det_kind(label: str) -> str:
    text = str(label).lower()
    if any(term in text for term in ("vehicle", "shuttle", "pod", "car", "bus")):
        return "vehicle"
    if "circle" in text or "crossing" in text or "goal" in text:
        return "circle"
    return "other"


def _box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_intersection(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _box_iou(a: list[float], b: list[float]) -> float:
    inter = _box_intersection(a, b)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _box_min_overlap(a: list[float], b: list[float]) -> float:
    inter = _box_intersection(a, b)
    denom = min(_box_area(a), _box_area(b))
    return inter / denom if denom > 0 else 0.0


def _box_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _box_diag(box: list[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return float(np.hypot(max(0.0, x2 - x1), max(0.0, y2 - y1)))


def suppress_cross_label_overlaps(detections: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    """For near-identical regions detected under different labels, keep the higher score."""
    if not args.cross_label_nms:
        return detections

    kept: list[dict[str, Any]] = []
    for det in sorted(detections, key=lambda d: float(d.get("score", 0.0)), reverse=True):
        box = det.get("box") or det.get("bbox")
        label = str(det.get("label", "")).strip().lower()
        duplicate = False
        for other in kept:
            other_label = str(other.get("label", "")).strip().lower()
            if label == other_label:
                continue
            other_box = other.get("box") or other.get("bbox")
            iou = _box_iou(box, other_box)
            min_overlap = _box_min_overlap(box, other_box)
            if iou >= args.cross_label_nms_iou or min_overlap >= args.cross_label_nms_min_overlap:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return sorted(kept, key=lambda d: float(d.get("score", 0.0)), reverse=True)


def filter_detections(detections: list[dict[str, Any]], frame_shape, args) -> list[dict[str, Any]]:
    """Apply task-specific geometry constraints after open-vocab detection."""
    height, width = frame_shape[:2]
    filtered = []
    for det in detections:
        box = det.get("box") or det.get("bbox")
        if not box or len(box) != 4:
            continue
        kind = _det_kind(det.get("label", ""))
        x1, y1, x2, y2 = [float(v) for v in box]
        det = dict(det)
        det["kind"] = kind

        if kind == "circle" and args.circle_ground_only:
            center_y = (y1 + y2) / 2.0
            if center_y < args.circle_min_center_y_frac * height:
                continue

        filtered.append(det)

    filtered = suppress_cross_label_overlaps(filtered, args)

    if args.vehicle_select == "all":
        return filtered

    vehicles = [d for d in filtered if d.get("kind") == "vehicle"]
    if not vehicles:
        return filtered

    def vehicle_score(det: dict[str, Any]) -> float:
        x1, y1, x2, y2 = [float(v) for v in det["box"]]
        area_norm = _box_area(det["box"]) / max(1.0, width * height)
        bottom_norm = y2 / max(1.0, height)
        conf = float(det.get("score", 0.0))
        if args.vehicle_select == "largest":
            return area_norm + 0.05 * conf
        return area_norm + args.vehicle_bottom_weight * bottom_norm + 0.05 * conf

    keep_vehicle = max(vehicles, key=vehicle_score)
    return [d for d in filtered if d.get("kind") != "vehicle" or d is keep_vehicle]


def _track_match_key(det: dict[str, Any], match_by: str) -> str:
    if match_by == "label":
        return str(det.get("label", "")).strip().lower()
    return det.get("kind") or _det_kind(det.get("label", ""))


def assign_track_ids(
    frame_records: list[dict[str, Any]],
    iou_threshold: float,
    center_threshold_frac: float,
    match_by: str,
) -> list[dict[str, Any]]:
    """Assign stable track ids with greedy type-gated IoU plus center-distance matching."""
    next_id = 1
    tracks: list[dict[str, Any]] = []
    for record in frame_records:
        used = set()
        for det in sorted(record.get("detections", []), key=lambda d: float(d.get("score", 0.0)), reverse=True):
            kind = det.get("kind") or _det_kind(det.get("label", ""))
            match_key = _track_match_key(det, match_by)
            box = det.get("box") or det.get("bbox")
            best_idx, best_score = None, -1.0
            for idx, track in enumerate(tracks):
                if idx in used or track["match_key"] != match_key:
                    continue
                iou = _box_iou(box, track["box"])
                cx, cy = _box_center(box)
                tx, ty = _box_center(track["box"])
                center_dist = float(np.hypot(cx - tx, cy - ty))
                scale = max(_box_diag(box), _box_diag(track["box"]), 1.0)
                center_ratio = center_dist / scale
                center_ok = center_ratio <= center_threshold_frac
                if iou < iou_threshold and not center_ok:
                    continue
                score = iou + max(0.0, 1.0 - center_ratio)
                if score > best_score:
                    best_idx, best_score = idx, score
            if best_idx is not None:
                track_id = tracks[best_idx]["track_id"]
                tracks[best_idx] = {"track_id": track_id, "kind": kind, "match_key": match_key, "box": box}
                used.add(best_idx)
            else:
                track_id = next_id
                next_id += 1
                tracks.append({"track_id": track_id, "kind": kind, "match_key": match_key, "box": box})
                used.add(len(tracks) - 1)
            det["track_id"] = track_id
    return frame_records

def select_annotations(annos: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    if args.video_ids:
        wanted = {x.strip() for x in args.video_ids.split(",") if x.strip()}
        return [a for a in annos if a.get("video_id") in wanted]
    if args.sample_n and args.sample_n < len(annos):
        return _sample_stratified(annos, args.sample_n, args.sample_seed)
    return annos


def parse_render_visuals(value: str) -> list[str]:
    visuals = [x.strip() for x in value.split(",") if x.strip()]
    allowed = {"bbox_overlay", "som_overlay_bg", "som_overlay_nobg"}
    unknown = [x for x in visuals if x not in allowed]
    if unknown:
        raise ValueError(f"Unsupported render visuals: {unknown}; allowed={sorted(allowed)}")
    return visuals


def render_offline_overlay_videos(video_id: str, frames, frame_records: list[dict[str, Any]], labels: list[str],
                                  video_sample_fps: float, args) -> None:
    if not args.render_overlay_videos:
        return
    for visual in parse_render_visuals(args.render_visuals):
        out_path = offline_overlay_video_path(
            args.video_root, video_id, visual, labels, args.num_frames,
            video_sample_fps, args.video_duration, args.vcot_detector,
        )
        if out_path.exists() and not args.overwrite_rendered:
            continue
        rendered = render_detection_overlays(frames, frame_records, visual)
        encode_rgb_frames_to_video(rendered, str(out_path), video_sample_fps)
        logging.info("Wrote offline %s video: %s", visual, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual-CoT detection caches.")
    parser.add_argument("--ann_file", default="features/groundvqa_qn3/annotations.VRbinary_00000_test_close.json")
    parser.add_argument("--video_root", default="data/videodata_256/clips")
    parser.add_argument("--cache_dir", default=".cache")
    parser.add_argument("--vcot_cache_dir", default=".cache/visual_cot")
    parser.add_argument("--vcot_labels", default="automated vehicle,white circle")
    parser.add_argument("--vcot_detector", default="groundingdinotiny", choices=["groundingdinotiny", "groundingdinobase"])
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--video_duration", type=float, default=2.0)
    parser.add_argument("--box_threshold", type=float, default=0.40)
    parser.add_argument("--text_threshold", type=float, default=0.20)
    parser.add_argument("--circle_ground_only", action="store_true", default=True,
                        help="Filter circle detections to lower image regions to avoid lamps/signs")
    parser.add_argument("--no_circle_ground_filter", dest="circle_ground_only", action="store_false")
    parser.add_argument("--circle_min_center_y_frac", type=float, default=0.45,
                        help="Keep circle boxes whose vertical center is below this fraction of image height")
    parser.add_argument("--detect_per_label", action="store_true", default=True,
                        help="Run one detector prompt per label to avoid ambiguous multi-label text spans")
    parser.add_argument("--detect_joint_prompt", dest="detect_per_label", action="store_false",
                        help="Run all labels in one GroundingDINO prompt; faster but can produce ambiguous spans")
    parser.add_argument("--ambiguous_label_policy", choices=["skip", "raw", "first"], default="skip",
                        help="Policy for ambiguous labels when using --detect_joint_prompt")
    parser.add_argument("--cross_label_nms", action="store_true", default=True,
                        help="Suppress overlapping detections with different labels, keeping the higher score")
    parser.add_argument("--no_cross_label_nms", dest="cross_label_nms", action="store_false")
    parser.add_argument("--cross_label_nms_iou", type=float, default=0.50,
                        help="IoU threshold for cross-label duplicate suppression")
    parser.add_argument("--cross_label_nms_min_overlap", type=float, default=0.90,
                        help="Intersection/min-area threshold for nested cross-label duplicate suppression")
    parser.add_argument("--vehicle_select", choices=["all", "nearest", "largest"], default="all",
                        help="For vehicle detections, keep all or one likely closest vehicle per frame")
    parser.add_argument("--vehicle_bottom_weight", type=float, default=0.35,
                        help="Extra weight for lower image position when selecting nearest vehicle")
    parser.add_argument("--track_objects", action="store_true", default=True,
                        help="Assign stable track_id values across sampled frames")
    parser.add_argument("--no_tracking", dest="track_objects", action="store_false")
    parser.add_argument("--track_iou_threshold", type=float, default=0.10)
    parser.add_argument("--track_center_threshold_frac", type=float, default=1.25,
                        help="Also match boxes whose centers move less than this many box diagonals")
    parser.add_argument("--track_match_by", choices=["kind", "label"], default="kind",
                        help="Gate tracking matches by coarse kind or exact detected label")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--video_ids", default=None, help="Comma-separated video_id list, e.g. P38S5-2,P47S3-1")
    parser.add_argument("--sample_n", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--render_overlay_videos", action="store_true",
                        help="Also write bbox/SoM sampled-overlay MP4s under a sibling data/videodata_256 root")
    parser.add_argument("--render_visuals", default="bbox_overlay,som_overlay_bg,som_overlay_nobg",
                        help="Comma-separated visuals to render offline: bbox_overlay,som_overlay_bg,som_overlay_nobg")
    parser.add_argument("--overwrite_rendered", action="store_true",
                        help="Overwrite existing offline rendered overlay videos")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    labels = parse_vcot_labels(args.vcot_labels)
    video_sample_fps = max(1, int(args.num_frames / args.video_duration))

    annos = json.loads(Path(args.ann_file).read_text())
    annos = select_annotations(annos, args)
    logging.info("Generating detections for %d clips, labels=%s", len(annos), labels)

    if args.vcot_detector == "groundingdinotiny":
        args.model_id = "IDEA-Research/grounding-dino-tiny"
    elif args.vcot_detector == "groundingdinobase":
        args.model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(args.device)
    model.eval()

    for anno in tqdm(annos, desc="Detecting"):
        video_id = anno.get("video_id")
        if not video_id:
            continue
        out_path = detection_cache_path(
            args.vcot_cache_dir,
            video_id,
            labels,
            args.num_frames,
            video_sample_fps,
            args.video_duration,
            args.vcot_detector,
        )
        video_path = Path(args.video_root) / f"{video_id}.mp4"
        if not video_path.exists():
            logging.warning("Missing video: %s", video_path)
            continue

        _, frames, timestamps = get_video_frames(str(video_path), num_frames=args.num_frames, cache_dir=args.cache_dir)
        if out_path.exists() and not args.overwrite:
            cached = json.loads(out_path.read_text())
            frame_records = cached["frames"] if isinstance(cached, dict) and "frames" in cached else cached
            render_offline_overlay_videos(video_id, frames, frame_records, labels, video_sample_fps, args)
            continue

        frame_records = []
        for frame, ts in zip(frames, timestamps):
            frame_records.append({
                "timestamp": _mean_timestamp(ts),
                "detections": filter_detections(
                    detect_frame(processor, model, frame, labels, args, args.device),
                    frame.shape,
                    args,
                ),
            })
        if args.track_objects:
            frame_records = assign_track_ids(
                frame_records,
                args.track_iou_threshold,
                args.track_center_threshold_frac,
                args.track_match_by,
            )

        save_detection_cache(out_path, frame_records, {
            "video_id": video_id,
            "labels": labels,
            "detector": args.vcot_detector,
            "model_id": args.model_id,
            "num_frames": args.num_frames,
            "video_sample_fps": video_sample_fps,
            "video_duration": args.video_duration,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "circle_ground_only": args.circle_ground_only,
            "circle_min_center_y_frac": args.circle_min_center_y_frac,
            "detect_per_label": args.detect_per_label,
            "ambiguous_label_policy": args.ambiguous_label_policy,
            "cross_label_nms": args.cross_label_nms,
            "cross_label_nms_iou": args.cross_label_nms_iou,
            "cross_label_nms_min_overlap": args.cross_label_nms_min_overlap,
            "vehicle_select": args.vehicle_select,
            "track_objects": args.track_objects,
            "track_iou_threshold": args.track_iou_threshold,
            "track_center_threshold_frac": args.track_center_threshold_frac,
            "track_match_by": args.track_match_by,
        })
        logging.info("Wrote %s", out_path)
        render_offline_overlay_videos(video_id, frames, frame_records, labels, video_sample_fps, args)


if __name__ == "__main__":
    main()
