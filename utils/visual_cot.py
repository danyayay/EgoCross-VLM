"""Visual chain-of-thought helpers for VLM evaluation.

The module is intentionally cache-first: detector-produced boxes are read from
JSON artifacts when present. Online detection is opt-in because detector model
versions and latency would otherwise make eval sweeps hard to reproduce.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dataprep.v3overlay_gaze import (
    encode_rgb_frames_to_video,
    load_gaze,
    overlay_frames as overlay_gaze_frames,
    render_sampled_gaze_video,
)

_GAZE_STYLES = {
    "gaze_dot": "dot",
    "gaze_rainbow": "rainbow",
    "gaze_bone": "bone",
}
_BBOX_VISUALS = {"bbox_overlay", "som_overlay_nobg", 'som_overlay_bg'}


def parse_vcot_labels(labels: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(labels, str):
        return [x.strip() for x in labels.split(",") if x.strip()]
    return [str(x).strip() for x in labels if str(x).strip()]


def uses_frame_vcot(vcot_visual: str, vcot_text: str) -> bool:
    return vcot_visual in _GAZE_STYLES or vcot_visual in _BBOX_VISUALS or vcot_text == "bbox_coords"


def forces_interleaved_frames(vcot_visual: str) -> bool:
    return vcot_visual in _BBOX_VISUALS


def _model_supports_video_path(args: Any) -> bool:
    name = str(getattr(args, "model_name", "")).lower()
    return "qwen" in name or "gemini" in name


def sampling_tag(num_frames: int, video_sample_fps: float, video_duration: float) -> str:
    fps = f"{float(video_sample_fps):.4g}".replace(".", "p")
    dur = f"{float(video_duration):.4g}".replace(".", "p")
    return f"f{int(num_frames)}_fps{fps}_dur{dur}"


def labels_hash(labels: list[str]) -> str:
    joined = "\n".join(labels).lower().encode("utf-8")
    return hashlib.md5(joined).hexdigest()[:10]


def detection_cache_path(cache_dir: str, video_id: str, labels: list[str], num_frames: int,
                         video_sample_fps: float, video_duration: float, detector: str) -> Path:
    stem = f"{video_id}__{sampling_tag(num_frames, video_sample_fps, video_duration)}__{detector}_{labels_hash(labels)}.json"
    return Path(cache_dir) / "detections" / stem


def rendered_sampled_video_path(cache_dir: str, video_id: str, visual_name: str, num_frames: int,
                                video_sample_fps: float, video_duration: float,
                                labels: list[str] | None = None,
                                detector: str | None = None) -> Path:
    detector_part = f"_{detector}" if detector else ""
    label_part = f"_{labels_hash(labels)}" if labels else ""
    stem = f"{video_id}__{sampling_tag(num_frames, video_sample_fps, video_duration)}__{visual_name}{detector_part}{label_part}.mp4"
    return Path(cache_dir) / "rendered_videos" / stem


def offline_overlay_root(video_root: str, visual_name: str, labels: list[str], num_frames: int,
                         video_sample_fps: float, video_duration: float, detector: str) -> Path:
    root = Path(video_root)
    key = f"{visual_name}_{detector}_{labels_hash(labels)}_{sampling_tag(num_frames, video_sample_fps, video_duration)}"
    return root if root.name.endswith(f"_{key}") else root.with_name(f"{root.name}_{key}")


def resolve_vcot_video_root(video_root: str, args: Any) -> str:
    if args.vcot_visual in _GAZE_STYLES:
        style = _GAZE_STYLES[args.vcot_visual]
        root = Path(video_root)
        return str(root if root.name.endswith(f"_{style}") else root.with_name(f"{root.name}_{style}"))
    if args.vcot_visual in _BBOX_VISUALS:
        labels = parse_vcot_labels(args.vcot_labels)
        return str(offline_overlay_root(
            video_root, args.vcot_visual, labels, args.num_frames,
            args.video_sample_fps, args.video_duration, getattr(args, "vcot_detector", None),
        ))
    return video_root


def offline_overlay_video_path(video_root: str, video_id: str, visual_name: str, labels: list[str],
                               num_frames: int, video_sample_fps: float, video_duration: float,
                               detector: str) -> Path:
    return offline_overlay_root(video_root, visual_name, labels, num_frames, video_sample_fps,
                                video_duration, detector) / f"{video_id}.mp4"


def sibling_gaze_video_path(video_root: str, video_id: str, vcot_visual: str) -> str | None:
    style = _GAZE_STYLES.get(vcot_visual)
    if style is None:
        return None
    root = Path(video_root)
    candidate_root = root if root.name.endswith(f"_{style}") else root.with_name(f"{root.name}_{style}")
    candidate = candidate_root / f"{video_id}.mp4"
    return str(candidate) if candidate.exists() else None


def resolve_vcot_video_path(video_root: str, video_id: str, args: Any) -> str | None:
    """Return an offline visual-CoT video path for vcot_visual, if one exists."""
    if args.vcot_visual in _GAZE_STYLES or args.vcot_visual in _BBOX_VISUALS:
        candidate = Path(resolve_vcot_video_root(video_root, args)) / f"{video_id}.mp4"
        return str(candidate) if candidate.exists() else None
    return None


def _mean_timestamp(timestamp: Any) -> float:
    arr = np.asarray(timestamp)
    return float(arr.mean()) if arr.size else 0.0


def _load_sampled_gaze(args: Any, anno: dict, num_frames: int) -> list[list[float | None]]:
    shim = type("Args", (), {})()
    shim.input_video = f"{anno.get('video_id')}.mp4"
    shim.input_annofile = args.ann_file
    shim.vrdata_dir = getattr(args, "vrdata_dir", "data/vrdata")
    shim.is_resized = True
    shim.original_height = 1068
    shim.original_width = 1536
    shim.resize_height = 256
    full_gaze = load_gaze(shim)
    if not full_gaze:
        return []
    indices = np.linspace(0, len(full_gaze) - 1, num=num_frames, dtype=int)
    return [full_gaze[int(i)] for i in indices]


def load_detection_cache(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "frames" in data:
        return data["frames"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported detection cache schema: {path}")


def save_detection_cache(path: Path, frames: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "frames": frames}, indent=2))


def detect_boxes_online(frames, labels: list[str], detector: str) -> list[dict[str, Any]]:
    if detector != "groundingdino":
        raise ValueError(f"Unsupported vcot detector: {detector}")
    raise ImportError(
        "Online GroundingDINO detection is not configured in this repo yet. "
        "Precompute the detection JSON cache, or install/wire a GroundingDINO backend "
        "and extend utils.visual_cot.detect_boxes_online()."
    )


def get_detections(frames, timestamps, args: Any, anno: dict) -> tuple[list[dict[str, Any]], Path, bool]:
    labels = parse_vcot_labels(args.vcot_labels)
    path = detection_cache_path(args.vcot_cache_dir, anno.get("video_id"), labels,
                                args.num_frames, args.video_sample_fps,
                                args.video_duration, args.vcot_detector)
    cached = load_detection_cache(path)
    if cached is not None:
        return cached, path, True

    if not args.vcot_online_if_missing:
        logging.warning("Missing visual-CoT detection cache: %s", path)
        empty = [
            {"timestamp": _mean_timestamp(ts), "detections": []}
            for ts in (timestamps if timestamps is not None else range(len(frames)))
        ]
        return empty, path, False

    detected = detect_boxes_online(frames, labels, args.vcot_detector)
    save_detection_cache(path, detected, {
        "video_id": anno.get("video_id"),
        "labels": labels,
        "detector": args.vcot_detector,
        "num_frames": args.num_frames,
        "video_sample_fps": args.video_sample_fps,
        "video_duration": args.video_duration,
    })
    return detected, path, False


def _color_for_index(idx: int) -> tuple[int, int, int]:
    palette = [(0, 255, 255), (255, 0, 255), (0, 200, 0), (255, 128, 0), (0, 128, 255), (255, 255, 0)]
    return palette[idx % len(palette)]


def _draw_text_with_background(
    image,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    padding_x: int = 1,
    padding_y: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, baseline_y = origin
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x1 = max(0, x - padding_x)
    y1 = max(0, baseline_y - text_h - padding_y)
    x2 = min(image.shape[1] - 1, x + text_w + padding_x)
    y2 = min(image.shape[0] - 1, baseline_y + max(1, baseline // 2) + padding_y)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(image, text, (x, baseline_y), font, font_scale, color, thickness, cv2.LINE_AA)


def render_detection_overlays(frames, detections: list[dict[str, Any]], mode: str):
    rendered = []
    use_som = mode in {"som_overlay_nobg", "som_overlay_bg"}
    use_text_bg = mode in {"bbox_overlay", "som_overlay_bg"}
    for i, frame in enumerate(frames):
        out = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
        dets = detections[i].get("detections", []) if i < len(detections) else []
        for j, det in enumerate(dets):
            box = det.get("box") or det.get("bbox")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            track_id = det.get("track_id")
            color_idx = int(track_id) - 1 if track_id is not None else j
            color = _color_for_index(color_idx)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = str(track_id if track_id is not None else j + 1) if use_som else str(det.get("label", "object"))
            if det.get("score") is not None and not use_som:
                label = f"{label} {float(det['score']):.2f}"
            text_y = y1 - 5 if y1 >= 18 else min(out.shape[0] - 4, y1 + 16)
            font_scale = 0.48 if use_som else 0.42
            if use_text_bg:
                _draw_text_with_background(out, label, (x1, text_y), font_scale, color, 1)
            else:
                cv2.putText(out, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
        rendered.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    return np.asarray(rendered, dtype=np.uint8)


def format_bbox_coords_text(detections: list[dict[str, Any]]) -> str:
    lines = []
    for frame in detections:
        parts = []
        for det in frame.get("detections", []):
            box = det.get("box") or det.get("bbox")
            if not box or len(box) != 4:
                continue
            coords = ",".join(str(int(round(float(v)))) for v in box)
            prefix = f"#{det['track_id']} " if det.get("track_id") is not None else ""
            parts.append(f"{prefix}{det.get('label', 'object')} box=({coords})")
        if parts:
            lines.append(f"[t={float(frame.get('timestamp', 0.0)):.2f}s] " + "; ".join(parts))
    if not lines:
        return "Detected visual boxes: none available in cache for this sampled clip."
    return "Detected visual boxes:\n" + "\n".join(lines)


def prepare_visual_cot(anno: dict, video_path: str, frames, timestamps, args: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "vcot_visual": args.vcot_visual,
        "vcot_text": args.vcot_text,
        "vcot_labels": parse_vcot_labels(args.vcot_labels),
        "vcot_detection_cache": None,
        "vcot_detection_cache_hit": None,
        "vcot_rendered_video": None,
        "vcot_rendered_video_cache_hit": None,
        "vcot_input_source": "raw_video",
    }
    context_addition = ""
    out_frames = frames
    out_video_path = video_path
    force_interleaved = False

    if args.vcot_visual in _GAZE_STYLES:
        gaze_video = sibling_gaze_video_path(args.video_root, anno.get("video_id"), args.vcot_visual)
        if gaze_video and str(Path(video_path)) == str(Path(gaze_video)):
            metadata["vcot_rendered_video"] = gaze_video
            metadata["vcot_rendered_video_cache_hit"] = True
            metadata["vcot_input_source"] = "offline_overlay_video"
        elif gaze_video:
            out_video_path = gaze_video
            metadata["vcot_rendered_video"] = gaze_video
            metadata["vcot_rendered_video_cache_hit"] = True
            metadata["vcot_input_source"] = "offline_overlay_video"
        elif frames is not None:
            gaze = _load_sampled_gaze(args, anno, args.num_frames)
            out_frames = overlay_gaze_frames(frames, gaze, overlay_style=_GAZE_STYLES[args.vcot_visual])
            force_interleaved = True
            metadata["vcot_input_source"] = "rendered_frames"
            rendered_path = rendered_sampled_video_path(args.vcot_cache_dir, anno.get("video_id"), args.vcot_visual,
                                                        args.num_frames, args.video_sample_fps, args.video_duration)
            metadata["vcot_rendered_video"] = str(rendered_path)
            if getattr(args, "vcot_render_sampled_video", False) and not rendered_path.exists():
                render_sampled_gaze_video(out_frames, gaze, str(rendered_path), args.video_sample_fps,
                                          overlay_style=_GAZE_STYLES[args.vcot_visual])
        else:
            logging.warning("Requested %s but no frames were available for %s", args.vcot_visual, anno.get("video_id"))

    detections = None
    rendered_path = None
    can_use_rendered_video = _model_supports_video_path(args) and not getattr(args, "interleaved_timestamps", False)
    labels = parse_vcot_labels(args.vcot_labels)
    if args.vcot_visual in _BBOX_VISUALS:
        offline_path = offline_overlay_video_path(args.video_root, anno.get("video_id"), args.vcot_visual,
                                                  labels, args.num_frames, args.video_sample_fps,
                                                  args.video_duration, getattr(args, "vcot_detector", None))
        rendered_path = rendered_sampled_video_path(args.vcot_cache_dir, anno.get("video_id"), args.vcot_visual,
                                                    args.num_frames, args.video_sample_fps, args.video_duration,
                                                    labels=labels,
                                                    detector=getattr(args, "vcot_detector", None))
        metadata["vcot_rendered_video"] = str(offline_path if offline_path.exists() else rendered_path)
        metadata["vcot_rendered_video_cache_hit"] = offline_path.exists() or rendered_path.exists()
        if offline_path.exists() and str(Path(video_path)) == str(Path(offline_path)):
            metadata["vcot_input_source"] = "offline_overlay_video"
        elif can_use_rendered_video and offline_path.exists():
            out_video_path = str(offline_path)
            out_frames = None
            timestamps = None
            force_interleaved = False
            metadata["vcot_input_source"] = "offline_overlay_video"
        elif can_use_rendered_video and rendered_path.exists():
            out_video_path = str(rendered_path)
            out_frames = None
            timestamps = None
            force_interleaved = False
            metadata["vcot_input_source"] = "rendered_video_cache"

    using_overlay_video = metadata.get("vcot_input_source") in {"offline_overlay_video", "rendered_video_cache"}
    needs_detection_records = (
        args.vcot_text == "bbox_coords"
        or (args.vcot_visual in _BBOX_VISUALS and not using_overlay_video)
    )
    if needs_detection_records:
        if frames is None:
            raise ValueError("BBox/SoM visual CoT requires pre-extracted frames when rendered video cache is unavailable")
        detections, det_path, cache_hit = get_detections(frames, timestamps, args, anno)
        metadata["vcot_detection_cache"] = str(det_path)
        metadata["vcot_detection_cache_hit"] = cache_hit

    if args.vcot_visual in _BBOX_VISUALS and not using_overlay_video:
        out_frames = render_detection_overlays(frames, detections or [], args.vcot_visual)
        force_interleaved = True
        metadata["vcot_input_source"] = "rendered_frames"
        if rendered_path is not None:
            metadata["vcot_rendered_video"] = str(rendered_path)
            if getattr(args, "vcot_render_sampled_video", False) and not rendered_path.exists():
                encode_rgb_frames_to_video(out_frames, str(rendered_path), args.video_sample_fps)
                metadata["vcot_rendered_video_cache_hit"] = True
            if rendered_path.exists() and can_use_rendered_video:
                out_video_path = str(rendered_path)
                out_frames = None
                timestamps = None
                force_interleaved = False
                metadata["vcot_input_source"] = "rendered_video_cache"

    if args.vcot_text == "bbox_coords":
        context_addition = format_bbox_coords_text(detections or [])

    return {
        "video_path": out_video_path,
        "frames": out_frames,
        "timestamps": timestamps,
        "context_addition": context_addition,
        "force_interleaved": force_interleaved,
        "metadata": metadata,
    }
