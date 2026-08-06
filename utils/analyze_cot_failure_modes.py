#!/usr/bin/env python3
"""Analyze VLM CoT failure modes without requiring gold reasoning traces.

This script joins VLM response JSON files with GroundVQA annotations and VR
top-down coordinates, then assigns conservative deterministic failure labels.
An optional local VLM judge can inspect the sampled frames using the same frame
sampling helper as ``training.eval_vlm``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.eval_utils import get_video_frames

# Suppress warnings from transformers and other libraries
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
warnings.filterwarnings("ignore", message=".*CUDA.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


UNCERTAINTY_PATTERNS = [
    r"\bseems?\b",
    r"\bappears?\b",
    r"\blikely\b",
    r"\bprobably\b",
    r"\bpossibly\b",
    r"\bassuming\b",
    r"\bmight\b",
    r"\bmay\b",
    r"\bcould\b",
    r"\bunclear\b",
    r"\bnot clear\b",
    r"\bhard to tell\b",
    r"\bdifficult to tell\b",
    r"\bI might be missing\b",
    r"\bI may be missing\b",
    r"\bmaybe\b",
    r"\bperhaps\b",
]

CROSS_INTENT_TERMS = [
    r"\b(?:I|we|pedestrian|camera wearer|person)\s+(?:should|will|can|would|is going to|am going to)\s+(?:cross|go|continue|proceed|move forward|walk forward)\b",
    r"\b(?:safe|clear|okay|ok)\s+(?:for me\s+)?to\s+(?:cross|continue|proceed|move forward|walk forward)\b",
    r"\b(?:continue|proceed|move|walk)\s+forward\b",
    r"\bcontinue\s+walking\b",
    r"\bcross\s+(?:now|the street|the road|the path|the crossing)\b",
]

YIELD_INTENT_TERMS = [
    r"\b(?:I|we|pedestrian|camera wearer|person)\s+(?:should|will|need to|has to|must)\s+(?:yield|wait|stop|slow down)\b",
    r"\b(?:yield|wait|stop|slow down)\s+(?:to|for|before|until)\b",
    r"\blet(?:s|ting)?\s+(?:the\s+)?(?:vehicle|shuttle|pod|car)\s+pass\b",
    r"\b(?:before|until)\s+(?:I|we|the pedestrian|the person)?\s*(?:can\s+|should\s+)?cross(?:ing)?\b",
    r"\b(?:before|until)\s+crossing\b",
    r"\bensure\s+(?:that\s+)?(?:the\s+)?(?:vehicle|shuttle|pod|car).*\bpass\b",
    r"\bcautious\b.*\b(?:yield|wait|pass before crossing|before crossing)\b",
]

EGO_FORWARD_TERMS = [
    r"\bpedestrian .* (?:moves?|moving|walks?|walking|continues?|proceeds?)\b",
    r"\bcamera wearer .* (?:moves?|moving|walks?|walking|continues?|proceeds?)\b",
    r"\bmove(?:s|ing)? forward\b",
    r"\bwalk(?:s|ing)? forward\b",
    r"\btoward(?:s)? the (?:circle|goal|crossing)\b",
]
EGO_BACKWARD_TERMS = [
    r"\bstep(?:s|ping)? back\b",
    r"\bmove(?:s|ing)? back(?:ward)?\b",
    r"\baway from the (?:circle|goal|crossing)\b",
]
EGO_STILL_TERMS = [
    r"\bstanding still\b",
    r"\bstationary\b",
    r"\bstopped\b",
    r"\bnot moving\b",
]

VEHICLE_MOVING_TERMS = [
    r"\bvehicle .* (?:moves?|moving|drives?|driving)\b",
    r"\bshuttle .* (?:moves?|moving|drives?|driving)\b",
    r"\bpod .* (?:moves?|moving|drives?|driving)\b",
]
VEHICLE_APPROACH_TERMS = [
    r"\bvehicle .* (?:approaches?|approaching|coming closer|moving toward(?:s)?|moving towards me)\b",
    r"\bshuttle .* (?:approaches?|approaching|coming closer|moving toward(?:s)?|moving towards me)\b",
    r"\bpod .* (?:approaches?|approaching|coming closer|moving toward(?:s)?|moving towards me)\b",
    r"\bmoving toward(?:s)? me\b",
    r"\bcoming toward(?:s)? me\b",
    r"\bapproaching me\b",
]
VEHICLE_RECEDING_TERMS = [
    r"\bvehicle .* (?:moving away|go(?:es|ing)? away|reced(?:es|ing)|passed|passing away)\b",
    r"\bshuttle .* (?:moving away|go(?:es|ing)? away|reced(?:es|ing)|passed|passing away)\b",
    r"\bpod .* (?:moving away|go(?:es|ing)? away|reced(?:es|ing)|passed|passing away)\b",
    r"\bmoving away from me\b",
    r"\bpassed me\b",
]
VEHICLE_STOPPED_TERMS = [
    r"\bvehicle .* (?:stops?|stopped|stationary|not moving)\b",
    r"\bshuttle .* (?:stops?|stopped|stationary|not moving)\b",
    r"\bpod .* (?:stops?|stopped|stationary|not moving)\b",
]

VEHICLE_WORDS = re.compile(r"\b(?:vehicle|vehicles|shuttle|shuttles|pod|pods|car|cars)\b", re.I)
MULTI_VEHICLE_WORDS = re.compile(
    r"\b(?:two|both|multiple|several|another|different|second)\s+"
    r"(?:vehicle|vehicles|shuttle|shuttles|pod|pods|car|cars)\b",
    re.I,
)
NO_VEHICLE_WORDS = re.compile(
    r"\b(?:no|without|not any|no visible)\s+"
    r"(?:vehicle|vehicles|shuttle|shuttles|pod|pods|car|cars)\b",
    re.I,
)
GAZE_WORDS = re.compile(r"\b(?:gaze|look(?:s|ing)?|attention|fixat(?:e|es|ing|ion))\b", re.I)


@dataclass
class RunMetadata:
    response_path: str
    run_name: str
    model: str | None
    cot: str | None
    num_frames: int | None
    interleaved: bool | None
    prompt_variant: str | None
    quantized: bool


def clean_label(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    text = re.sub(r"^\(?[a-d]\)?[.)]?\s+", "", text)
    if "cross" in text:
        return "cross"
    if "yield" in text or "wait" in text or "stop" in text:
        return "yield"
    return text or "unknown"


def safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def finite_mean(values: pd.Series | np.ndarray | list[Any]) -> float | None:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if arr.empty:
        return None
    return float(arr.mean())


def finite_first(values: pd.Series) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return None if vals.empty else float(vals.iloc[0])


def finite_last(values: pd.Series) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return None if vals.empty else float(vals.iloc[-1])


def text_has(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.I) for p in patterns)


def count_pattern_hits(patterns: list[str], text: str) -> int:
    return sum(len(re.findall(p, text, flags=re.I | re.S)) for p in patterns)


def reasoning_action_support(text: str) -> dict[str, int]:
    """Estimate which final action the reasoning actually supports.

    This intentionally avoids counting bare words like "crossing" as cross
    support. Phrases such as "before crossing" usually indicate waiting or
    yielding, so they are counted as yield support instead.
    """
    compact = re.sub(r"\s+", " ", text or " ").strip()
    cross_support = count_pattern_hits(CROSS_INTENT_TERMS, compact)
    yield_support = count_pattern_hits(YIELD_INTENT_TERMS, compact)

    # Common narrative shape: vehicle/risk is present and crossing is delayed
    # until after the vehicle passes. Treat this as yield support even when the
    # word "yield" is absent.
    if re.search(r"\b(?:vehicle|shuttle|pod|car)\b", compact, re.I):
        if re.search(r"\b(?:before|until|after)\b.{0,80}\bcross(?:ing)?\b", compact, re.I | re.S):
            yield_support += 1
        if re.search(r"\b(?:cautious|careful|ensure safety|avoid accident|enough space to pass)\b", compact, re.I):
            yield_support += 1

    return {"cross": cross_support, "yield": yield_support}


def short_text(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def parse_run_metadata(path: Path) -> RunMetadata:
    run_tag = path.parent.name
    model = path.parent.parent.name if path.parent.parent != path.parent else None
    cot = None
    num_frames = None
    interleaved = None
    prompt_variant = None

    cot_match = re.search(r"(?:^|_)cot([^_]+)", run_tag)
    if cot_match:
        cot = cot_match.group(1)
        if cot == "nocot":
            cot = "none"
    frame_match = re.search(r"(?:^|_)f(\d+)(?:_|$)", run_tag)
    if frame_match:
        num_frames = int(frame_match.group(1))
    if "_interleaved" in run_tag or run_tag.endswith("interleaved"):
        interleaved = True
    if "_nointerleave" in run_tag or run_tag.endswith("nointerleave"):
        interleaved = False
    prompt_match = re.search(r"(?:^|_)(p\d+)(?:_|$)", run_tag)
    if prompt_match:
        prompt_variant = prompt_match.group(1)

    return RunMetadata(
        response_path=str(path),
        run_name=f"{model or 'unknown_model'}__{run_tag}__{path.stem}",
        model=model,
        cot=cot,
        num_frames=num_frames,
        interleaved=interleaved,
        prompt_variant=prompt_variant,
        quantized=run_tag.endswith("_int8") or "_int8" in run_tag,
    )


def load_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    return {item["video_id"]: item for item in load_json_list(path)}


def clip_uid_from_video_id(video_id: str) -> str:
    return video_id.rsplit("-", 1)[0]


def question_hash(text: str | None) -> str:
    if not text:
        return ""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def summarize_signed_trajectory(
    values: pd.Series,
    positive_label: str,
    negative_label: str,
    flat_label: str = "stationary",
    mixed_positive_label: str | None = None,
    mixed_negative_label: str | None = None,
    mixed_label: str = "mixed",
    eps_step: float = 1.0,
    eps_total: float = 5.0,
) -> dict[str, Any]:
    """Summarize a whole 1D trajectory instead of only first/last points.

    Positive and negative labels are caller-defined. For pedestrian y, positive
    means backward; for distance-to-goal, positive means away from goal.
    """
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if len(vals) < 2:
        return {
            "label": "unknown",
            "net_delta": None,
            "positive_total": 0.0,
            "negative_total": 0.0,
            "has_positive_segment": False,
            "has_negative_segment": False,
            "reversal_count": 0,
        }

    diffs = vals.diff().dropna()
    pos = diffs[diffs > eps_step]
    neg = diffs[diffs < -eps_step]
    positive_total = float(pos.sum()) if not pos.empty else 0.0
    negative_total = float((-neg).sum()) if not neg.empty else 0.0
    net_delta = float(vals.iloc[-1] - vals.iloc[0])
    has_pos = positive_total > eps_total
    has_neg = negative_total > eps_total

    signs = []
    for diff in diffs:
        if diff > eps_step:
            signs.append(1)
        elif diff < -eps_step:
            signs.append(-1)
    reversal_count = sum(1 for a, b in zip(signs, signs[1:]) if a != b)

    mixed_positive_label = mixed_positive_label or f"mixed_{positive_label}"
    mixed_negative_label = mixed_negative_label or f"mixed_{negative_label}"
    if has_pos and has_neg:
        if net_delta > eps_total:
            label = mixed_positive_label
        elif net_delta < -eps_total:
            label = mixed_negative_label
        else:
            label = mixed_label
    elif has_pos:
        label = positive_label
    elif has_neg:
        label = negative_label
    else:
        label = flat_label

    return {
        "label": label,
        "net_delta": net_delta,
        "positive_total": positive_total,
        "negative_total": negative_total,
        "has_positive_segment": has_pos,
        "has_negative_segment": has_neg,
        "reversal_count": reversal_count,
    }


def classify_raw_vehicle_motion(vel: float | None, moving_thr: float = 5.0) -> str:
    if vel is None:
        return "unknown"
    if abs(vel) < moving_thr:
        return "stationary"
    return "moving_positive_x" if vel > 0 else "moving_negative_x"


def summarize_vehicle_relative_motion(
    vehicle_x: pd.Series,
    ped_x: pd.Series | None,
    eps_step: float = 1.0,
    eps_total: float = 25.0,
) -> dict[str, Any]:
    if ped_x is None:
        return {"label": "unknown", "distance_delta": None, "has_approach_segment": False, "has_recede_segment": False}
    rel_abs = (pd.to_numeric(vehicle_x, errors="coerce") - pd.to_numeric(ped_x, errors="coerce")).abs()
    summary = summarize_signed_trajectory(
        rel_abs,
        positive_label="receding",
        negative_label="approaching",
        flat_label="relative_stationary",
        mixed_positive_label="mixed_receding",
        mixed_negative_label="mixed_approaching",
        mixed_label="mixed_relative",
        eps_step=eps_step,
        eps_total=eps_total,
    )
    return {
        "label": summary["label"],
        "distance_delta": summary["net_delta"],
        "approach_total": summary["negative_total"],
        "recede_total": summary["positive_total"],
        "has_approach_segment": summary["has_negative_segment"],
        "has_recede_segment": summary["has_positive_segment"],
        "reversal_count": summary["reversal_count"],
    }


def extract_vr_features(
    df: pd.DataFrame,
    anno: dict[str, Any],
    fut_sec: float = 1.0,
) -> dict[str, Any]:
    start = int(anno.get("video_start_frame", 0) or 0)
    end = int(anno.get("video_end_frame", start) or start)
    duration = float(anno.get("video_end_sec", 0) or 0) - float(anno.get("video_start_sec", 0) or 0)
    fps = round((end - start) / duration) if duration > 0 and end > start else 30
    fut_end = min(len(df), end + int(round(fut_sec * fps)))

    obs = df.iloc[max(0, start) : min(len(df), end)]
    fut = df.iloc[min(len(df), end) : fut_end]
    if obs.empty:
        return {"vr_missing_window": True, "fps": fps}

    ego_summary = summarize_signed_trajectory(
        obs["ped_loc_y"] if "ped_loc_y" in obs else pd.Series(dtype=float),
        positive_label="backward",
        negative_label="forward",
        flat_label="stationary",
        mixed_positive_label="mixed_backward",
        mixed_negative_label="mixed_forward",
        mixed_label="mixed_ego",
        eps_step=1.0,
        eps_total=5.0,
    )
    ped_y_delta = ego_summary["net_delta"]
    goal_summary = summarize_signed_trajectory(
        obs["dist_ped2center"] if "dist_ped2center" in obs else pd.Series(dtype=float),
        positive_label="away_from_goal",
        negative_label="toward_goal",
        flat_label="flat",
        mixed_positive_label="mixed_away_from_goal",
        mixed_negative_label="mixed_toward_goal",
        mixed_label="mixed_goal_progress",
        eps_step=1.0,
        eps_total=5.0,
    )
    dist_delta = goal_summary["net_delta"]

    fut_vel = finite_mean(fut["ped_vel"]) if (not fut.empty and "ped_vel" in fut) else None
    obs_vel = finite_mean(obs["ped_vel"]) if "ped_vel" in obs else None
    obs_acc = None
    if "ped_vel" in obs and "time" in obs and len(obs) > 2:
        vel = pd.to_numeric(obs["ped_vel"], errors="coerce")
        time = pd.to_numeric(obs["time"], errors="coerce")
        acc = vel.diff() / time.diff()
        obs_acc = finite_mean(acc.replace([np.inf, -np.inf], np.nan))

    vehicle_stats: dict[str, dict[str, Any]] = {}
    ped_x = pd.to_numeric(obs.get("ped_loc_x"), errors="coerce") if "ped_loc_x" in obs else None
    for agent in ("leader", "follower"):
        loc_col = f"{agent}_loc_x"
        vel_col = f"{agent}_vel_x"
        ehmi_col = f"{agent}_ehmi"
        if loc_col not in obs:
            continue
        loc = pd.to_numeric(obs[loc_col], errors="coerce")
        active = bool(loc.notna().any())
        if not active:
            vehicle_stats[agent] = {"active": False}
            continue
        rel_dist = None
        rel_motion = summarize_vehicle_relative_motion(loc, ped_x)
        if ped_x is not None:
            rel_dist = finite_mean((loc - ped_x).abs())
        vel_mean = finite_mean(obs[vel_col]) if vel_col in obs else None
        vehicle_stats[agent] = {
            "active": active,
            "mean_abs_x_distance": rel_dist,
            "mean_vel_x": vel_mean,
            "raw_motion": classify_raw_vehicle_motion(vel_mean),
            "motion": rel_motion["label"],
            "relative_distance_delta": rel_motion["distance_delta"],
            "has_approach_segment": rel_motion["has_approach_segment"],
            "has_recede_segment": rel_motion["has_recede_segment"],
            "relative_reversal_count": rel_motion["reversal_count"],
            "ehmi_values": sorted(
                {str(x) for x in obs[ehmi_col].dropna().unique()}
            )
            if ehmi_col in obs
            else [],
        }

    active_agents = [k for k, v in vehicle_stats.items() if v.get("active")]
    nearest_agent = None
    nearest_dist = None
    for agent in active_agents:
        dist = vehicle_stats[agent].get("mean_abs_x_distance")
        if dist is not None and (nearest_dist is None or dist < nearest_dist):
            nearest_agent = agent
            nearest_dist = dist

    hit_counts: dict[str, int] = {}
    visible_vehicle_hit_count = 0
    visible_vehicle_targets: list[str] = []
    if "hit_object" in obs:
        hit_counts = {
            str(k): int(v)
            for k, v in obs["hit_object"].dropna().astype(str).value_counts().to_dict().items()
        }
        visible_vehicle_targets = sorted([k for k in hit_counts if "pod" in k.lower() or "vehicle" in k.lower()])
        visible_vehicle_hit_count = sum(hit_counts[k] for k in visible_vehicle_targets)
    mask_attention_rate = None
    if "mask_attention" in obs:
        mask_attention_rate = float(obs["mask_attention"].astype(bool).mean())

    return {
        "vr_missing_window": False,
        "fps": fps,
        "observed_speed_mean": obs_vel,
        "observed_acc_mean": obs_acc,
        "future_speed_mean": fut_vel,
        "ped_y_delta": ped_y_delta,
        "ego_direction": ego_summary["label"],
        "ego_forward_total": ego_summary["negative_total"],
        "ego_backward_total": ego_summary["positive_total"],
        "ego_has_forward_segment": ego_summary["has_negative_segment"],
        "ego_has_backward_segment": ego_summary["has_positive_segment"],
        "ego_reversal_count": ego_summary["reversal_count"],
        "dist_ped2center_delta": dist_delta,
        "goal_progress": goal_summary["label"],
        "goal_toward_total": goal_summary["negative_total"],
        "goal_away_total": goal_summary["positive_total"],
        "goal_has_toward_segment": goal_summary["has_negative_segment"],
        "goal_has_away_segment": goal_summary["has_positive_segment"],
        "goal_reversal_count": goal_summary["reversal_count"],
        "active_vehicle_count": len(active_agents),
        "visible_vehicle_hit_count": visible_vehicle_hit_count,
        "visible_vehicle_targets": visible_vehicle_targets,
        "has_visible_vehicle_evidence": visible_vehicle_hit_count > 0,
        "active_agents": active_agents,
        "nearest_agent": nearest_agent,
        "nearest_vehicle_x_distance": nearest_dist,
        "leader_motion": vehicle_stats.get("leader", {}).get("motion", "unknown"),
        "follower_motion": vehicle_stats.get("follower", {}).get("motion", "unknown"),
        "leader_raw_motion": vehicle_stats.get("leader", {}).get("raw_motion", "unknown"),
        "follower_raw_motion": vehicle_stats.get("follower", {}).get("raw_motion", "unknown"),
        "leader_relative_distance_delta": vehicle_stats.get("leader", {}).get("relative_distance_delta"),
        "follower_relative_distance_delta": vehicle_stats.get("follower", {}).get("relative_distance_delta"),
        "leader_has_approach_segment": vehicle_stats.get("leader", {}).get("has_approach_segment", False),
        "follower_has_approach_segment": vehicle_stats.get("follower", {}).get("has_approach_segment", False),
        "leader_has_recede_segment": vehicle_stats.get("leader", {}).get("has_recede_segment", False),
        "follower_has_recede_segment": vehicle_stats.get("follower", {}).get("has_recede_segment", False),
        "leader_mean_vel_x": vehicle_stats.get("leader", {}).get("mean_vel_x"),
        "follower_mean_vel_x": vehicle_stats.get("follower", {}).get("mean_vel_x"),
        "leader_ehmi_values": vehicle_stats.get("leader", {}).get("ehmi_values", []),
        "follower_ehmi_values": vehicle_stats.get("follower", {}).get("ehmi_values", []),
        "mask_attention_rate": mask_attention_rate,
        "hit_object_counts": hit_counts,
    }


def analyze_uncertainty(text: str) -> dict[str, Any]:
    found: list[str] = []
    for pattern in UNCERTAINTY_PATTERNS:
        found.extend(re.findall(pattern, text, flags=re.I))
    words = re.findall(r"\b\w+\b", text)
    count = len(found)
    rate = count / max(1, len(words))
    return {
        "uncertainty_marker_count": count,
        "uncertainty_marker_rate": rate,
        "uncertainty_markers_found": sorted({x.lower() for x in found}),
        "has_high_uncertainty": count >= 2 or rate >= 0.025,
    }


def deterministic_failure_modes(
    sample: dict[str, Any],
    features: dict[str, Any],
    uncertainty: dict[str, Any],
) -> list[str]:
    modes: list[str] = []
    gt = clean_label(sample.get("answer"))
    pred = clean_label(sample.get("pred_answer"))
    is_finished = sample.get("is_finished", True)
    text = " ".join(
        str(sample.get(k) or "") for k in ("reasoning", "output_text")
    ).strip()

    if not is_finished or pred not in {"cross", "yield"}:
        modes.append("parse_or_truncation_failure")
    if gt == "yield" and pred == "cross":
        modes.append("cross_instead_of_yield")
    if gt == "cross" and pred == "yield":
        modes.append("yield_instead_of_cross")

    action_support = reasoning_action_support(text)
    cross_terms = action_support["cross"]
    yield_terms = action_support["yield"]
    if pred == "cross" and yield_terms > 0 and cross_terms == 0:
        modes.append("answer_reasoning_inconsistent")
    if pred == "yield" and cross_terms > 0 and yield_terms == 0:
        modes.append("answer_reasoning_inconsistent")

    obs_speed = features.get("observed_speed_mean")
    goal_progress = features.get("goal_progress")
    has_forward_segment = bool(features.get("ego_has_forward_segment"))
    has_backward_segment = bool(features.get("ego_has_backward_segment"))
    if text_has(EGO_BACKWARD_TERMS, text) and not has_backward_segment:
        modes.append("egomotion_direction_misread")
    if text_has(EGO_FORWARD_TERMS, text) and not has_forward_segment:
        modes.append("egomotion_direction_misread")
    if obs_speed is not None and abs(obs_speed) > 50 and text_has(EGO_STILL_TERMS, text):
        modes.append("egomotion_direction_misread")
    if obs_speed is not None and abs(obs_speed) < 15 and text_has(EGO_FORWARD_TERMS, text):
        modes.append("egomotion_direction_misread")
    has_goal_toward = bool(features.get("goal_has_toward_segment"))
    has_goal_away = bool(features.get("goal_has_away_segment"))
    if re.search(r"\baway from the (?:circle|goal)\b", text, re.I) and not has_goal_away:
        modes.append("goal_progress_misread")
    if re.search(r"\btoward(?:s)? the (?:circle|goal)\b", text, re.I) and not has_goal_toward:
        modes.append("goal_progress_misread")

    active_vehicle_count = int(features.get("active_vehicle_count") or 0)
    has_visible_vehicle_evidence = bool(features.get("has_visible_vehicle_evidence"))
    relative_motions = [
        m for m in (features.get("leader_motion"), features.get("follower_motion"))
        if m and m != "unknown"
    ]
    raw_motions = [
        m for m in (features.get("leader_raw_motion"), features.get("follower_raw_motion"))
        if m and m != "unknown"
    ]
    has_vehicle_approach = bool(features.get("leader_has_approach_segment")) or bool(features.get("follower_has_approach_segment"))
    has_vehicle_recede = bool(features.get("leader_has_recede_segment")) or bool(features.get("follower_has_recede_segment"))
    any_vehicle_raw_moving = any(m != "stationary" for m in raw_motions)
    all_relative_stationary = bool(relative_motions) and all(m == "relative_stationary" for m in relative_motions)
    all_relative_receding = bool(relative_motions) and all("receding" in m for m in relative_motions)
    if any_vehicle_raw_moving and text_has(VEHICLE_STOPPED_TERMS, text):
        modes.append("vehicle_motion_misread")
    negated_approach_claim = bool(re.search(
        r"\b(?:no|not|without)\s+(?:visible\s+)?(?:vehicle|vehicles|shuttle|shuttles|pod|pods|car|cars)\s+(?:is\s+|are\s+)?(?:approaching|coming|moving toward)",
        text,
        re.I,
    ))
    has_positive_approach_claim = text_has(VEHICLE_APPROACH_TERMS, text) and not negated_approach_claim
    if all_relative_stationary and (has_positive_approach_claim or text_has(VEHICLE_RECEDING_TERMS, text)):
        modes.append("vehicle_motion_misread")
    # Do not mark approach/recede conflicts as deterministic failures when a
    # vehicle is visible; egocentric apparent motion needs visual inspection.
    if not has_visible_vehicle_evidence:
        if has_positive_approach_claim and not has_vehicle_approach and all_relative_receding:
            modes.append("vehicle_motion_misread")
        if text_has(VEHICLE_RECEDING_TERMS, text) and not has_vehicle_recede and has_vehicle_approach:
            modes.append("vehicle_motion_misread")
    if active_vehicle_count <= 1 and MULTI_VEHICLE_WORDS.search(text):
        modes.append("vehicle_count_confusion")
        modes.append("vehicle_identity_switch")
    if active_vehicle_count > 0 and NO_VEHICLE_WORDS.search(text) and has_visible_vehicle_evidence:
        modes.append("vehicle_count_confusion")
    if active_vehicle_count == 0 and VEHICLE_WORDS.search(text):
        modes.append("unsupported_visual_claim")

    nearest_dist = features.get("nearest_vehicle_x_distance")
    if nearest_dist is not None:
        says_close = re.search(r"\b(?:close|near|nearby|immediate|right in front)\b", text, re.I)
        says_far = re.search(r"\b(?:far|distant|far away|not close)\b", text, re.I)
        if nearest_dist < 250 and says_far:
            modes.append("proximity_or_risk_misread")
        if nearest_dist > 900 and says_close:
            modes.append("proximity_or_risk_misread")

    if GAZE_WORDS.search(text):
        modes.append("attention_or_gaze_claim_unverified")

    if uncertainty["has_high_uncertainty"]:
        modes.append("high_uncertainty_reasoning")
        if gt != pred:
            modes.append("hedged_answer_with_wrong_prediction")

    return sorted(set(modes))


def _status_from_claims(has_correct: bool, has_wrong: bool, has_claim: bool) -> str:
    if not has_claim:
        return "unspecified"
    if has_wrong and not has_correct:
        return "wrong"
    if has_correct and not has_wrong:
        return "correct"
    return "mixed"


def _uncertainty_level(uncertainty: dict[str, Any]) -> str:
    count = int(uncertainty.get("uncertainty_marker_count") or 0)
    rate = float(uncertainty.get("uncertainty_marker_rate") or 0.0)
    if count >= 3 or rate >= 0.04:
        return "high"
    if count >= 1 or rate >= 0.015:
        return "medium"
    return "low"


def verbalize_vr_groundtruth(features: dict[str, Any]) -> str:
    """Convert compact VR features into readable facts for text judging."""
    ego_direction = features.get("ego_direction", "unknown")
    speed = features.get("observed_speed_mean")
    speed_abs = abs(speed) if isinstance(speed, (int, float)) else None
    if speed_abs is None:
        ego_motion = "unknown movement speed"
    elif speed_abs < 15:
        ego_motion = "mostly stationary or barely moving"
    elif speed_abs < 50:
        ego_motion = "moving slowly"
    else:
        ego_motion = "clearly moving"

    vehicle_count = int(features.get("active_vehicle_count") or 0)
    visible_count = int(features.get("visible_vehicle_hit_count") or 0)
    visible_targets = features.get("visible_vehicle_targets") or []
    if visible_count > 0:
        visible_text = f"vehicle was visually attended/detected in {visible_count} VR gaze-hit frames ({', '.join(visible_targets)})"
    else:
        visible_text = "no vehicle was detected in the VR gaze-hit labels for this clip"
    vehicle_parts = []
    for agent in ("leader", "follower"):
        motion = features.get(f"{agent}_motion", "unknown")
        if motion != "unknown":
            vehicle_parts.append(f"{agent} tracked vehicle is {motion.replace('_', ' ')} in top-down distance to the pedestrian")
    vehicle_text = "; ".join(vehicle_parts) if vehicle_parts else "no tracked vehicle motion available"

    return (
        f"Ego motion: {ego_motion}; ego direction/progress: {ego_direction.replace('_', ' ')}. "
        f"Tracked vehicle count from top-down data: {vehicle_count}. Visual vehicle evidence: {visible_text}. "
        f"Top-down vehicle motion: {vehicle_text}."
    )


def vr_text_reasoning_judge(
    reasoning_text: str,
    features: dict[str, Any],
    uncertainty: dict[str, Any],
) -> dict[str, Any]:
    """Judge reasoning against verbalized VR facts on coarse recognition dimensions.

    Labels are intentionally ternary: correct, wrong, unspecified. ``mixed`` is
    used only when the reasoning contains both correct and wrong claims.
    """
    text = re.sub(r"\s+", " ", reasoning_text or " ").strip()
    speed = features.get("observed_speed_mean")
    speed_abs = abs(speed) if isinstance(speed, (int, float)) else None
    ego_gt_stationary = speed_abs is not None and speed_abs < 15
    ego_gt_moving = speed_abs is not None and speed_abs >= 15

    ego_still_claim = text_has(EGO_STILL_TERMS, text)
    ego_move_claim = text_has(EGO_FORWARD_TERMS + EGO_BACKWARD_TERMS, text) or bool(
        re.search(r"\b(?:I|pedestrian|camera wearer|person)\s+(?:am\s+)?(?:walking|moving)\b", text, re.I)
    )
    ego_motion_claim = ego_still_claim or ego_move_claim
    ego_motion_correct = (ego_still_claim and ego_gt_stationary) or (ego_move_claim and ego_gt_moving)
    ego_motion_wrong = (ego_still_claim and ego_gt_moving) or (ego_move_claim and ego_gt_stationary)

    has_forward_segment = bool(features.get("ego_has_forward_segment"))
    has_backward_segment = bool(features.get("ego_has_backward_segment"))
    forward_claim = text_has(EGO_FORWARD_TERMS, text)
    backward_claim = text_has(EGO_BACKWARD_TERMS, text)
    ego_direction_claim = forward_claim or backward_claim
    ego_direction_correct = (forward_claim and has_forward_segment) or (backward_claim and has_backward_segment)
    ego_direction_wrong = (forward_claim and not has_forward_segment) or (backward_claim and not has_backward_segment)

    raw_motions = [
        m for m in (features.get("leader_raw_motion"), features.get("follower_raw_motion"))
        if m and m != "unknown"
    ]
    relative_motions = [
        m for m in (features.get("leader_motion"), features.get("follower_motion"))
        if m and m != "unknown"
    ]
    any_raw_moving = any(m != "stationary" for m in raw_motions)
    all_raw_stationary = bool(raw_motions) and all(m == "stationary" for m in raw_motions)
    has_approach = bool(features.get("leader_has_approach_segment")) or bool(features.get("follower_has_approach_segment"))
    has_recede = bool(features.get("leader_has_recede_segment")) or bool(features.get("follower_has_recede_segment"))
    all_relative_stationary = bool(relative_motions) and all(m == "relative_stationary" for m in relative_motions)

    negated_approach_claim = bool(re.search(
        r"\b(?:no|not|without)\s+(?:visible\s+)?(?:vehicle|vehicles|shuttle|shuttles|pod|pods|car|cars)\s+(?:is\s+|are\s+)?(?:approaching|coming|moving toward)",
        text,
        re.I,
    ))
    vehicle_approach_claim = text_has(VEHICLE_APPROACH_TERMS, text) and not negated_approach_claim
    vehicle_recede_claim = text_has(VEHICLE_RECEDING_TERMS, text)
    vehicle_stop_claim = text_has(VEHICLE_STOPPED_TERMS, text)
    vehicle_move_claim = text_has(VEHICLE_MOVING_TERMS, text)
    vehicle_motion_claim = vehicle_approach_claim or vehicle_recede_claim or vehicle_stop_claim or vehicle_move_claim
    vehicle_motion_correct = (
        (vehicle_approach_claim and has_approach)
        or (vehicle_recede_claim and has_recede)
        or (vehicle_stop_claim and (all_raw_stationary or all_relative_stationary))
        or (vehicle_move_claim and any_raw_moving)
    )
    has_visible_vehicle_evidence = bool(features.get("has_visible_vehicle_evidence"))
    topdown_direction_conflict = (
        (vehicle_approach_claim and not has_approach and has_recede)
        or (vehicle_recede_claim and not has_recede and has_approach)
    )
    vehicle_motion_wrong = (
        # If the vehicle is visible, apparent egocentric approach/recede needs
        # visual inspection; top-down distance alone is not enough to mark it
        # wrong. If it is not visible, top-down conflict is acceptable evidence.
        (topdown_direction_conflict and not has_visible_vehicle_evidence)
        or (vehicle_stop_claim and any_raw_moving and not all_relative_stationary)
        or (vehicle_move_claim and all_raw_stationary)
    )

    vehicle_count = int(features.get("active_vehicle_count") or 0)
    visible_vehicle_count = int(features.get("visible_vehicle_hit_count") or 0)
    no_vehicle_claim = bool(NO_VEHICLE_WORDS.search(text))
    multi_vehicle_claim = bool(MULTI_VEHICLE_WORDS.search(text))
    singular_vehicle_claim = bool(re.search(r"\b(?:a|one|the)\s+(?:vehicle|shuttle|pod|car)\b", text, re.I)) and not multi_vehicle_claim
    vehicle_count_claim = no_vehicle_claim or multi_vehicle_claim or singular_vehicle_claim
    # Count recognition is visual-first. If top-down tracks vehicles but the clip
    # has no vehicle hit-object evidence, "no visible vehicle" should not be
    # treated as wrong without an actual visual judge.
    no_visible_vehicle = visible_vehicle_count == 0
    vehicle_count_correct = (
        (no_vehicle_claim and (vehicle_count == 0 or no_visible_vehicle))
        or (singular_vehicle_claim and vehicle_count == 1)
        or (multi_vehicle_claim and vehicle_count >= 2 and not no_visible_vehicle)
    )
    vehicle_count_wrong = (
        (no_vehicle_claim and vehicle_count > 0 and not no_visible_vehicle)
        or (singular_vehicle_claim and vehicle_count != 1 and not no_visible_vehicle)
        or (multi_vehicle_claim and (vehicle_count < 2 or no_visible_vehicle))
    )

    vehicle_motion_status = _status_from_claims(
        vehicle_motion_correct, vehicle_motion_wrong, vehicle_motion_claim
    )
    vehicle_count_status = _status_from_claims(
        vehicle_count_correct, vehicle_count_wrong, vehicle_count_claim
    )
    if no_visible_vehicle and negated_approach_claim:
        vehicle_motion_status = "invisible"
    if no_visible_vehicle and no_vehicle_claim:
        vehicle_count_status = "invisible"

    return {
        "vr_groundtruth_text": verbalize_vr_groundtruth(features),
        "ego_motion_recognition": _status_from_claims(ego_motion_correct, ego_motion_wrong, ego_motion_claim),
        "ego_direction_recognition": _status_from_claims(ego_direction_correct, ego_direction_wrong, ego_direction_claim),
        "vehicle_motion_recognition": vehicle_motion_status,
        "vehicle_count_recognition": vehicle_count_status,
        "reasoning_uncertainty_level": _uncertainty_level(uncertainty),
    }


def compact_vr_facts(features: dict[str, Any]) -> str:
    facts = {
        "ego_direction": features.get("ego_direction"),
        "ego_has_forward_segment": features.get("ego_has_forward_segment"),
        "ego_has_backward_segment": features.get("ego_has_backward_segment"),
        "ego_reversal_count": features.get("ego_reversal_count"),
        "observed_speed_mean": features.get("observed_speed_mean"),
        "goal_progress": features.get("goal_progress"),
        "goal_has_toward_segment": features.get("goal_has_toward_segment"),
        "goal_has_away_segment": features.get("goal_has_away_segment"),
        "active_vehicle_count": features.get("active_vehicle_count"),
        "visible_vehicle_hit_count": features.get("visible_vehicle_hit_count"),
        "visible_vehicle_targets": features.get("visible_vehicle_targets"),
        "has_visible_vehicle_evidence": features.get("has_visible_vehicle_evidence"),
        "nearest_agent": features.get("nearest_agent"),
        "nearest_vehicle_x_distance": features.get("nearest_vehicle_x_distance"),
        "leader_motion": features.get("leader_motion"),
        "follower_motion": features.get("follower_motion"),
        "leader_raw_motion": features.get("leader_raw_motion"),
        "follower_raw_motion": features.get("follower_raw_motion"),
        "leader_relative_distance_delta": features.get("leader_relative_distance_delta"),
        "follower_relative_distance_delta": features.get("follower_relative_distance_delta"),
        "mask_attention_rate": features.get("mask_attention_rate"),
        "hit_object_counts": features.get("hit_object_counts"),
    }
    return json.dumps(facts, ensure_ascii=False, sort_keys=True)


def load_judge_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            cache[item["cache_key"]] = item
    return cache


def append_judge_cache(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix common issues in JSON
            # Remove trailing commas before closing braces/brackets
            json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try to find and fix unterminated strings
                # This is a best-effort approach
                json_str = re.sub(r'"\s*:\s*"([^"]*?)$', r'": "\1"', json_str)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    # If all else fails, return a partial result from what we could parse
                    # This helps prevent complete failure
                    raise


VALID_RECOGNITION_LABELS = {"correct", "wrong", "unspecified", "mixed", "invisible"}
VALID_UNCERTAINTY_LABELS = {"low", "medium", "high"}


def normalize_recognition_label(value: Any) -> str:
    text = str(value or "unspecified").strip().lower()
    if text in VALID_RECOGNITION_LABELS:
        return text
    if text in {"not mentioned", "not specified", "unknown", "n/a", "none"}:
        return "unspecified"
    if text in {"partially correct", "partial", "ambiguous"}:
        return "mixed"
    return "unspecified"


def normalize_uncertainty_label(value: Any) -> str:
    text = str(value or "medium").strip().lower()
    if text in VALID_UNCERTAINTY_LABELS:
        return text
    return "medium"


def make_text_judge_prompt(sample: dict[str, Any], features: dict[str, Any], uncertainty: dict[str, Any]) -> str:
    reasoning = sample.get("reasoning") or sample.get("output_text") or ""
    return (
        "You are evaluating a model's chain-of-thought for an egocentric pedestrian-video task. "
        "You do NOT need to decide the final answer. Evaluate only whether the reasoning recognizes facts "
        "that are stated in the provided VR ground-truth summary. The VR summary is derived from top-down coordinates; "
        "when it says a tracked vehicle is not visually detected, a reasoning claim like 'no visible vehicle' can be acceptable. "
        "For time-dependent reasoning, consider the full chronological statement, not isolated keywords.\n\n"
        f"Ground-truth answer: {clean_label(sample.get('answer'))}\n"
        f"Model final answer: {clean_label(sample.get('pred_answer'))}\n"
        f"VR ground-truth summary: {verbalize_vr_groundtruth(features)}\n"
        f"Deterministic uncertainty markers: {uncertainty.get('uncertainty_markers_found', [])}\n"
        f"Model reasoning/output:\n{reasoning}\n\n"
        "Return ONLY valid JSON with exactly these keys:\n"
        "{\n"
        "  \"ego_motion_recognition\": \"correct|wrong|unspecified|mixed\",\n"
        "  \"ego_direction_recognition\": \"correct|wrong|unspecified|mixed\",\n"
        "  \"vehicle_motion_recognition\": \"correct|wrong|unspecified|mixed|invisible\",\n"
        "  \"vehicle_count_recognition\": \"correct|wrong|unspecified|mixed|invisible\",\n"
        "  \"reasoning_uncertainty_level\": \"low|medium|high\",\n"
        "  \"short_rationale\": \"one concise sentence\"\n"
        "}\n"
        "Use 'unspecified' when the reasoning does not discuss that dimension. "
        "Use 'invisible' for vehicle motion/count when the model says there is no visible vehicle and the VR summary says no vehicle was visually detected. "
        "Use 'mixed' only when the reasoning contains both correct and wrong claims for the same dimension."
    )


def normalize_text_judge_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ego_motion_recognition": normalize_recognition_label(result.get("ego_motion_recognition")),
        "ego_direction_recognition": normalize_recognition_label(result.get("ego_direction_recognition")),
        "vehicle_motion_recognition": normalize_recognition_label(result.get("vehicle_motion_recognition")),
        "vehicle_count_recognition": normalize_recognition_label(result.get("vehicle_count_recognition")),
        "reasoning_uncertainty_level": normalize_uncertainty_label(result.get("reasoning_uncertainty_level")),
        "short_rationale": short_text(str(result.get("short_rationale") or ""), limit=300),
    }


class TextLLMJudge:
    def __init__(self, model_name: str, quantize: bool) -> None:
        from models.vlm_adapters import get_adapter

        self.adapter = get_adapter(model_name)
        self.adapter.load(model_name, quantize=quantize)

    def close(self) -> None:
        self.adapter.unload()

    def judge(self, sample: dict[str, Any], features: dict[str, Any], uncertainty: dict[str, Any]) -> dict[str, Any]:
        prompt = make_text_judge_prompt(sample, features, uncertainty)
        messages = self.adapter.build_messages(
            video_path="",
            prompt=prompt,
            no_video=True,
        )
        output, meta = self.adapter.run_inference(messages, max_new_tokens=256, temperature=0.0)
        try:
            parsed = normalize_text_judge_result(parse_json_object(output))
        except (json.JSONDecodeError, ValueError) as e:
            import logging
            logging.warning(f"Failed to parse judge output: {str(e)[:200]}. Output: {output[:200]}")
            # Return a default result with unspecified values
            parsed = {
                "ego_motion_recognition": "unspecified",
                "ego_direction_recognition": "unspecified",
                "vehicle_motion_recognition": "unspecified",
                "vehicle_count_recognition": "unspecified",
                "reasoning_uncertainty_level": "medium",
                "short_rationale": "Failed to parse judge output",
            }
        parsed["_judge_output_text"] = output
        parsed["_judge_meta"] = meta
        return parsed



def make_judge_prompt(sample: dict[str, Any], features: dict[str, Any]) -> str:
    reasoning = sample.get("reasoning") or sample.get("output_text") or ""
    return (
        "You are auditing a vision-language model's reasoning about an "
        "egocentric pedestrian-vehicle interaction clip. Use the frames and "
        "the compact top-down facts. Do not assume the model needs to match a "
        "gold reasoning trace; judge whether its claims are visually and "
        "spatially supported.\n\n"
        f"Ground-truth answer: {clean_label(sample.get('answer'))}\n"
        f"Model final answer: {clean_label(sample.get('pred_answer'))}\n"
        f"Model reasoning/output: {reasoning}\n"
        f"Top-down facts: {compact_vr_facts(features)}\n\n"
        "Return only valid JSON with keys: egomotion_consistent, "
        "vehicle_motion_consistent, vehicle_identity_consistent, "
        "vehicle_count_consistent, spatial_risk_consistent, "
        "attention_claim_supported, final_answer_consistent, failure_modes, "
        "short_rationale. The consistency fields must be booleans, "
        "failure_modes must be a list of strings."
    )


class LocalJudge:
    def __init__(self, model_name: str, quantize: bool) -> None:
        from models.vlm_adapters import get_adapter

        self.adapter = get_adapter(model_name)
        self.adapter.load(model_name, quantize=quantize)

    def close(self) -> None:
        self.adapter.unload()

    def judge(
        self,
        video_path: str,
        num_frames: int,
        sample: dict[str, Any],
        features: dict[str, Any],
        cache_dir: str,
    ) -> dict[str, Any]:
        _, frames, timestamps = get_video_frames(video_path, num_frames=num_frames, cache_dir=cache_dir)
        prompt = make_judge_prompt(sample, features)
        messages = self.adapter.build_messages(
            video_path,
            prompt,
            frames=frames,
            timestamps=timestamps,
            interleaved=True,
            total_pixels=20480 * 28 * 28,
            min_pixels=16 * 28 * 28,
            max_frames=2048,
            video_sample_fps=max(1, int(num_frames / 2)),
        )
        output, meta = self.adapter.run_inference(messages, max_new_tokens=512, temperature=0.0)
        parsed = parse_json_object(output)
        parsed["_judge_output_text"] = output
        parsed["_judge_meta"] = meta
        return parsed


def flatten_for_csv(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def build_report(rows: list[dict[str, Any]], analysis_methods: set[str] | None = None) -> dict[str, Any]:
    analysis_methods = analysis_methods or {"hardcoded", "llm_judge", "auxiliary"}
    total = len(rows)
    correct = sum(1 for r in rows if r.get("correct"))
    confusion: dict[str, Counter] = defaultdict(Counter)
    failure_counts: Counter = Counter()
    uncertainty_by_correct: dict[str, Counter] = defaultdict(Counter)
    prompt_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cot_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    context_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hardcoded_recognition_counts: dict[str, Counter] = defaultdict(Counter)
    text_llm_judge_recognition_counts: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        gt = row["answer"]
        pred = row["pred_answer"]
        confusion[gt][pred] += 1
        for mode in row["failure_modes"]:
            failure_counts[mode] += 1
        uncertainty_by_correct[str(bool(row["correct"]))]["n"] += 1
        if row["has_high_uncertainty"]:
            uncertainty_by_correct[str(bool(row["correct"]))]["high_uncertainty"] += 1
        prompt_groups[row.get("prompt_variant") or "unknown"].append(row)
        cot_groups[row.get("cot") or "unknown"].append(row)
        task_groups[row.get("task_name") or "unknown"].append(row)
        context_key = f"{row.get('context_features') or 'none'}|{row.get('context_feature_fps') or 'auto'}|{row.get('context_prompt_mode') or 'unknown'}"
        context_groups[context_key].append(row)
        for dim in (
            "ego_motion_recognition",
            "ego_direction_recognition",
            "vehicle_motion_recognition",
            "vehicle_count_recognition",
            "reasoning_uncertainty_level",
        ):
            if "hardcoded" in analysis_methods:
                hardcoded_recognition_counts[dim][row.get(f"hardcoded_text_judge_{dim}") or "missing"] += 1
            if "llm_judge" in analysis_methods and row.get(f"llm_text_judge_{dim}") is not None:
                text_llm_judge_recognition_counts[dim][row.get(f"llm_text_judge_{dim}") or "missing"] += 1

    def group_summary(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(group_rows)
        if n == 0:
            return {"n": 0}
        preds = Counter(r["pred_answer"] for r in group_rows)
        labels = sorted({r["answer"] for r in group_rows} | {r["pred_answer"] for r in group_rows})
        recall = {}
        for label in labels:
            denom = sum(1 for r in group_rows if r["answer"] == label)
            numer = sum(1 for r in group_rows if r["answer"] == label and r["pred_answer"] == label)
            recall[label] = numer / denom if denom else None
        return {
            "n": n,
            "accuracy": sum(1 for r in group_rows if r["correct"]) / n,
            "prediction_counts": dict(preds),
            "recall": recall,
            "high_uncertainty_rate": sum(1 for r in group_rows if r["has_high_uncertainty"]) / n,
            "failure_counts": dict(Counter(m for r in group_rows for m in r["failure_modes"])),
            "hardcoded_recognition_counts": ({
                dim: dict(Counter(r.get(f"hardcoded_text_judge_{dim}") or "missing" for r in group_rows))
                for dim in (
                    "ego_motion_recognition",
                    "ego_direction_recognition",
                    "vehicle_motion_recognition",
                    "vehicle_count_recognition",
                    "reasoning_uncertainty_level",
                )
            } if "hardcoded" in analysis_methods else {}),
        }

    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for mode in row["failure_modes"]:
            if len(examples[mode]) < 8:
                examples[mode].append(
                    {
                        "video_id": row["video_id"],
                        "answer": row["answer"],
                        "pred_answer": row["pred_answer"],
                        "prompt_variant": row.get("prompt_variant"),
                        "reasoning_excerpt": row.get("reasoning_excerpt"),
                        "uncertainty_markers_found": row.get("uncertainty_markers_found"),
                        "vr_facts": row.get("vr_facts"),
                        "judge_rationale": row.get("judge_short_rationale"),
                    }
                )

    return {
        "num_samples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "confusion": {gt: dict(preds) for gt, preds in confusion.items()},
        "failure_counts": dict(failure_counts),
        "failure_rates": {k: v / total for k, v in failure_counts.items()} if total else {},
        "uncertainty_by_correct": {
            k: {
                "n": v["n"],
                "high_uncertainty": v["high_uncertainty"],
                "high_uncertainty_rate": v["high_uncertainty"] / v["n"] if v["n"] else 0.0,
            }
            for k, v in uncertainty_by_correct.items()
        },
        "hardcoded_recognition_counts": {k: dict(v) for k, v in hardcoded_recognition_counts.items()},
        "text_llm_judge_recognition_counts": {k: dict(v) for k, v in text_llm_judge_recognition_counts.items()},
        "by_prompt_variant": {k: group_summary(v) for k, v in sorted(prompt_groups.items())},
        "by_cot": {k: group_summary(v) for k, v in sorted(cot_groups.items())},
        "by_task": {k: group_summary(v) for k, v in sorted(task_groups.items())},
        "by_context_features": {k: group_summary(v) for k, v in sorted(context_groups.items())},
        "top_examples": dict(examples),
    }


def load_auxiliary_predictions(paths: list[str] | None) -> dict[str, dict[str, Any]]:
    """Load auxiliary response files keyed by video_id and task name."""
    out: dict[str, dict[str, Any]] = defaultdict(dict)
    for raw_path in paths or []:
        path = Path(raw_path)
        if not path.exists():
            logging.warning("Auxiliary response file not found: %s", path)
            continue
        task = "unknown"
        match = re.search(r"aux[_-]([A-Za-z0-9_]+)", str(path))
        if match:
            task = match.group(1)
        for item in load_json_list(path):
            vid = str(item.get("video_id") or "")
            if not vid:
                continue
            out[vid][task] = {
                "answer": clean_label(item.get("answer")),
                "pred_answer": clean_label(item.get("pred_answer")),
                "correct": clean_label(item.get("answer")) == clean_label(item.get("pred_answer")),
                "response_path": str(path),
            }
    return out


def auxiliary_consistency_for_sample(video_id: str, aux_predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tasks = aux_predictions.get(video_id, {})
    if not tasks:
        return {
            "aux_consistency_available": False,
            "aux_consistency_tasks": "",
            "aux_consistency_all_correct": "",
            "aux_consistency_num_tasks": 0,
            "aux_consistency_num_correct": 0,
        }
    num_correct = sum(1 for v in tasks.values() if v.get("correct"))
    return {
        "aux_consistency_available": True,
        "aux_consistency_tasks": ",".join(sorted(tasks)),
        "aux_consistency_all_correct": num_correct == len(tasks),
        "aux_consistency_num_tasks": len(tasks),
        "aux_consistency_num_correct": num_correct,
        "aux_consistency_details": tasks,
    }


def process_response_file(
    path: Path,
    annotations: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    judge: LocalJudge | None,
    judge_cache: dict[str, dict[str, Any]],
    judge_cache_path: Path | None,
    text_judge: TextLLMJudge | None = None,
    text_judge_cache: dict[str, dict[str, Any]] | None = None,
    text_judge_cache_path: Path | None = None,
) -> list[dict[str, Any]]:
    metadata = parse_run_metadata(path)
    samples = load_json_list(path)
    if args.sample_n is not None:
        samples = samples[: args.sample_n]

    vr_cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(tqdm(samples, desc=f"Processing {path.name}", unit="sample")):
        video_id = str(sample.get("video_id") or "")
        if not video_id:
            continue
        anno = annotations.get(video_id, {})
        clip_uid = anno.get("video_uid") or clip_uid_from_video_id(video_id)
        csv_path = Path(args.vrdata_dir) / f"{clip_uid}.csv"
        if clip_uid not in vr_cache:
            if csv_path.exists():
                vr_cache[clip_uid] = pd.read_csv(csv_path)
            else:
                vr_cache[clip_uid] = pd.DataFrame()

        if not vr_cache[clip_uid].empty and anno:
            features = extract_vr_features(vr_cache[clip_uid], anno)
        else:
            features = {"vr_missing_window": True}

        text = " ".join(str(sample.get(k) or "") for k in ("reasoning", "output_text")).strip()
        needs_uncertainty = args.use_hardcoded_analysis or text_judge is not None
        uncertainty = analyze_uncertainty(text) if needs_uncertainty else {
            "uncertainty_marker_count": 0,
            "uncertainty_marker_rate": 0.0,
            "uncertainty_markers_found": "",
            "has_high_uncertainty": False,
        }
        vr_text_judge = vr_text_reasoning_judge(text, features, uncertainty) if args.use_hardcoded_analysis else None
        modes = deterministic_failure_modes(sample, features, uncertainty) if args.use_hardcoded_analysis else []
        gt = clean_label(sample.get("answer"))
        pred = clean_label(sample.get("pred_answer"))
        correct = gt == pred

        text_judge_result: dict[str, Any] | None = None
        if args.use_llm_judge_analysis and text_judge is not None and (not args.only_wrong or not correct):
            text_cache_key = hashlib.md5(
                f"text:{path}:{video_id}:{sample.get('output_text')}:{sample.get('reasoning')}".encode("utf-8")
            ).hexdigest()
            cached_text = (text_judge_cache or {}).get(text_cache_key)
            if cached_text is not None:
                text_judge_result = cached_text["judge_result"]
            else:
                text_judge_result = text_judge.judge(sample, features, uncertainty)
                if text_judge_cache_path is not None and text_judge_cache is not None:
                    item = {
                        "cache_key": text_cache_key,
                        "response_path": str(path),
                        "video_id": video_id,
                        "judge_result": text_judge_result,
                    }
                    append_judge_cache(text_judge_cache_path, item)
                    text_judge_cache[text_cache_key] = item

        judge_result: dict[str, Any] | None = None
        if args.use_llm_judge_analysis and judge is not None and (not args.only_wrong or not correct):
            if metadata.num_frames is None:
                raise ValueError(f"Cannot infer frame count from run tag: {path.parent.name}")
            video_path = str(Path(args.video_root) / f"{video_id}.mp4")
            cache_key = hashlib.md5(
                f"{path}:{video_id}:{metadata.num_frames}:{sample.get('output_text')}".encode("utf-8")
            ).hexdigest()
            cached = judge_cache.get(cache_key)
            if cached is not None:
                judge_result = cached["judge_result"]
            else:
                judge_result = judge.judge(
                    video_path=video_path,
                    num_frames=metadata.num_frames,
                    sample=sample,
                    features=features,
                    cache_dir=args.cache_dir,
                )
                if judge_cache_path is not None:
                    item = {
                        "cache_key": cache_key,
                        "response_path": str(path),
                        "video_id": video_id,
                        "judge_result": judge_result,
                    }
                    append_judge_cache(judge_cache_path, item)
                    judge_cache[cache_key] = item

            judge_modes = judge_result.get("failure_modes") or []
            modes = sorted(set(modes) | {str(m) for m in judge_modes})
            if (
                judge_result.get("attention_claim_supported") is True
                and "attention_or_gaze_claim_unverified" in modes
            ):
                modes.remove("attention_or_gaze_claim_unverified")

        aux_consistency = (
            auxiliary_consistency_for_sample(video_id, getattr(args, 'aux_predictions_by_video_id', {}))
            if args.use_auxiliary_analysis
            else {}
        )

        row = {
            "response_path": str(path),
            "sample_index": idx,
            "video_id": video_id,
            "video_uid": clip_uid,
            "answer": gt,
            "pred_answer": pred,
            "correct": correct,
            "failure_modes": modes,
            "failure_modes_str": "|".join(modes),
            "reasoning_text": sample.get("reasoning") or sample.get("output_text") or "",
            "reasoning_excerpt": short_text(sample.get("reasoning") or sample.get("output_text") or ""),
            "question_hash": question_hash(sample.get("question") or anno.get("question")),
            "question": sample.get("question") or anno.get("question"),
            "model": metadata.model,
            "cot": metadata.cot,
            "num_frames": metadata.num_frames,
            "interleaved": metadata.interleaved,
            "prompt_variant": metadata.prompt_variant,
            "task_name": sample.get('task_name') or (anno.get('task', {}) if isinstance(anno.get('task'), dict) else {}).get('name'),
            "context_features": sample.get('context_features'),
            "context_feature_fps": sample.get('context_feature_fps'),
            "context_prompt_mode": sample.get('context_prompt_mode'),
            "quantized": metadata.quantized,
            "vr_facts": compact_vr_facts(features),
            **features,
            **uncertainty,
            **aux_consistency,
        }
        if vr_text_judge is not None:
            row.update({
                "vr_groundtruth_text": vr_text_judge["vr_groundtruth_text"],
                "hardcoded_text_judge_ego_motion_recognition": vr_text_judge["ego_motion_recognition"],
                "hardcoded_text_judge_ego_direction_recognition": vr_text_judge["ego_direction_recognition"],
                "hardcoded_text_judge_vehicle_motion_recognition": vr_text_judge["vehicle_motion_recognition"],
                "hardcoded_text_judge_vehicle_count_recognition": vr_text_judge["vehicle_count_recognition"],
                "hardcoded_text_judge_reasoning_uncertainty_level": vr_text_judge["reasoning_uncertainty_level"],
                # Backward-compatible aliases for older CSV consumers.
                "vr_text_judge_ego_motion_recognition": vr_text_judge["ego_motion_recognition"],
                "vr_text_judge_ego_direction_recognition": vr_text_judge["ego_direction_recognition"],
                "vr_text_judge_vehicle_motion_recognition": vr_text_judge["vehicle_motion_recognition"],
                "vr_text_judge_vehicle_count_recognition": vr_text_judge["vehicle_count_recognition"],
                "vr_text_judge_reasoning_uncertainty_level": vr_text_judge["reasoning_uncertainty_level"],
            })
        if text_judge_result is not None:
            row.update({
                "llm_text_judge_ego_motion_recognition": text_judge_result.get("ego_motion_recognition"),
                "llm_text_judge_ego_direction_recognition": text_judge_result.get("ego_direction_recognition"),
                "llm_text_judge_vehicle_motion_recognition": text_judge_result.get("vehicle_motion_recognition"),
                "llm_text_judge_vehicle_count_recognition": text_judge_result.get("vehicle_count_recognition"),
                "llm_text_judge_reasoning_uncertainty_level": text_judge_result.get("reasoning_uncertainty_level"),
                "llm_text_judge_short_rationale": text_judge_result.get("short_rationale"),
            })
        if judge_result is not None:
            row.update(
                {
                    "judge_egomotion_consistent": judge_result.get("egomotion_consistent"),
                    "judge_vehicle_motion_consistent": judge_result.get("vehicle_motion_consistent"),
                    "judge_vehicle_identity_consistent": judge_result.get("vehicle_identity_consistent"),
                    "judge_vehicle_count_consistent": judge_result.get("vehicle_count_consistent"),
                    "judge_spatial_risk_consistent": judge_result.get("spatial_risk_consistent"),
                    "judge_attention_claim_supported": judge_result.get("attention_claim_supported"),
                    "judge_final_answer_consistent": judge_result.get("final_answer_consistent"),
                    "judge_short_rationale": judge_result.get("short_rationale"),
                }
            )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: flatten_for_csv(row.get(k, "")) for k in fields})


def write_method_recognition_per_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "response_path",
        "sample_index",
        "video_id",
        "video_uid",
        "answer",
        "pred_answer",
        "correct",
        "model",
        "cot",
        "num_frames",
        "interleaved",
        "prompt_variant",
        "hardcoded_text_judge_ego_motion_recognition",
        "hardcoded_text_judge_ego_direction_recognition",
        "hardcoded_text_judge_vehicle_motion_recognition",
        "hardcoded_text_judge_vehicle_count_recognition",
        "hardcoded_text_judge_reasoning_uncertainty_level",
        "llm_text_judge_ego_motion_recognition",
        "llm_text_judge_ego_direction_recognition",
        "llm_text_judge_vehicle_motion_recognition",
        "llm_text_judge_vehicle_count_recognition",
        "llm_text_judge_reasoning_uncertainty_level",
        "llm_text_judge_short_rationale",
        "uncertainty_marker_count",
        "uncertainty_marker_rate",
        "uncertainty_markers_found",
        "vr_groundtruth_text",
        "vr_text_judge_ego_motion_recognition",
        "vr_text_judge_ego_direction_recognition",
        "vr_text_judge_vehicle_motion_recognition",
        "vr_text_judge_vehicle_count_recognition",
        "vr_text_judge_reasoning_uncertainty_level",
        "reasoning_text",
        "reasoning_excerpt",
        "failure_modes_str",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: flatten_for_csv(row.get(k, "")) for k in fields})


def write_count_csv(path: Path, counts_by_group: dict[str, dict[str, int]], total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "label", "count", "rate"])
        writer.writeheader()
        for group, counts in sorted(counts_by_group.items()):
            group_total = sum(int(v) for v in counts.values()) or total
            for label, count in sorted(counts.items()):
                count = int(count)
                writer.writerow({
                    "group": group,
                    "label": label,
                    "count": count,
                    "rate": count / group_total if group_total else 0.0,
                })


def write_failure_counts_csv(path: Path, failure_counts: dict[str, int], total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["failure_mode", "count", "rate"])
        writer.writeheader()
        for mode, count in sorted(failure_counts.items(), key=lambda kv: (-int(kv[1]), kv[0])):
            count = int(count)
            writer.writerow({
                "failure_mode": mode,
                "count": count,
                "rate": count / total if total else 0.0,
            })


def write_report_csvs(outdir: Path, stem: str, report: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, str]:
    method_recognition_per_sample_path = outdir / f"{stem}_method_recognition_per_sample.csv"
    hardcoded_recognition_path = outdir / f"{stem}_hardcoded_recognition_counts.csv"
    text_llm_judge_recognition_path = outdir / f"{stem}_text_llm_judge_recognition_counts.csv"
    hardcoded_failure_path = outdir / f"{stem}_hardcoded_failure_counts.csv"
    write_method_recognition_per_sample_csv(method_recognition_per_sample_path, rows)
    write_count_csv(hardcoded_recognition_path, report.get("hardcoded_recognition_counts", {}), int(report.get("num_samples") or 0))
    write_count_csv(text_llm_judge_recognition_path, report.get("text_llm_judge_recognition_counts", {}), int(report.get("num_samples") or 0))
    write_failure_counts_csv(hardcoded_failure_path, report.get("failure_counts", {}), int(report.get("num_samples") or 0))
    return {
        "method_recognition_per_sample_csv": str(method_recognition_per_sample_path),
        "hardcoded_recognition_counts_csv": str(hardcoded_recognition_path),
        "text_llm_judge_recognition_counts_csv": str(text_llm_judge_recognition_path),
        "hardcoded_failure_counts_csv": str(hardcoded_failure_path),
        # Backward-compatible aliases for older downstream scripts.
        "recognition_per_sample_csv": str(method_recognition_per_sample_path),
        "recognition_counts_csv": str(hardcoded_recognition_path),
        "llm_text_recognition_counts_csv": str(text_llm_judge_recognition_path),
        "failure_counts_csv": str(hardcoded_failure_path),
    }


def make_output_stem(response_paths: list[Path]) -> str:
    if len(response_paths) == 1:
        meta = parse_run_metadata(response_paths[0])
        stem = meta.run_name
    else:
        digest = hashlib.md5("|".join(str(p) for p in response_paths).encode("utf-8")).hexdigest()[:10]
        stem = f"multi_response_{len(response_paths)}_{digest}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def normalize_analysis_methods(raw_methods: list[str] | None) -> set[str]:
    """Normalize requested analysis method families."""
    aliases = {
        "hardcoded": "hardcoded",
        "hard-coded": "hardcoded",
        "rule": "hardcoded",
        "rules": "hardcoded",
        "deterministic": "hardcoded",
        "llm": "llm_judge",
        "llm_judge": "llm_judge",
        "llm-as-a-judge": "llm_judge",
        "judge": "llm_judge",
        "aux": "auxiliary",
        "auxiliary": "auxiliary",
        "auxiliary_task": "auxiliary",
        "auxiliary_tasks": "auxiliary",
    }
    requested: list[str] = []
    for item in raw_methods or ["all"]:
        requested.extend(part.strip() for part in str(item).split(",") if part.strip())
    if not requested or any(item.lower() == "all" for item in requested):
        return {"hardcoded", "llm_judge", "auxiliary"}
    if any(item.lower() in {"none", "baseline"} for item in requested):
        return set()
    methods: set[str] = set()
    invalid: list[str] = []
    for item in requested:
        key = item.lower().replace(" ", "_")
        method = aliases.get(key)
        if method is None:
            invalid.append(item)
        else:
            methods.add(method)
    if invalid:
        valid = "all, hardcoded, llm_judge, auxiliary"
        raise ValueError(f"Unknown --analysis_methods value(s): {invalid}. Valid values: {valid}")
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CoT failure modes for VLM response files.")
    parser.add_argument("--responses", nargs="+", default=[
        "logs/vlm_eval_newformat/Qwen2.5-VL-7B-Instruct/cotcot4_f8_interleaved_p6_int8/20260518_154823_responses.json"], help="One or more *_responses.json files")
    parser.add_argument(
        "--ann_file",
        default="features/groundvqa/annotations.VRbinary__crossing_intention__test_close.json",
    )
    parser.add_argument("--video_root", default="data/videodata_256/clips")
    parser.add_argument("--vrdata_dir", default="data/vrdata")
    parser.add_argument("--outdir", default="logs/cot_failure_mode_analysis")
    parser.add_argument("--cache_dir", default=".cache")
    parser.add_argument("--sample_n", type=int, default=2)
    parser.add_argument("--only_wrong", action="store_true")
    parser.add_argument(
        "--analysis_methods",
        nargs="+",
        default=["hardcoded", "llm_judge"],
        help=(
            "Analysis method families to run: all, hardcoded, llm_judge, auxiliary. "
            "Aliases such as rule, llm, aux, or comma-separated values are accepted."
        ),
    )
    parser.add_argument("--llm_judge", action="store_true",
                        help="Within --analysis_methods llm_judge, also run the visual local VLM judge.")
    parser.add_argument("--judge_model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--text_llm_judge", action=argparse.BooleanOptionalAction, default=True,
                        help="Within --analysis_methods llm_judge, run a text-only local LLM judge over reasoning + verbalized VR facts.")
    parser.add_argument("--text_judge_model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--aux_responses", nargs="*", default=[],
                        help="Optional auxiliary-task *_responses.json files joined by video_id.")
    parser.add_argument(
        "--output_suffix",
        default="",
        help="Optional suffix appended to output filenames so repeated analyses do not overwrite earlier CSV/report files.",
    )
    return parser.parse_args()


def main() -> None:
    # Configure logging to reduce verbosity
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    
    args = parse_args()
    args.enabled_analysis_methods = normalize_analysis_methods(args.analysis_methods)
    args.use_hardcoded_analysis = "hardcoded" in args.enabled_analysis_methods
    args.use_llm_judge_analysis = "llm_judge" in args.enabled_analysis_methods
    args.use_auxiliary_analysis = "auxiliary" in args.enabled_analysis_methods
    if not args.use_llm_judge_analysis:
        args.llm_judge = False
        args.text_llm_judge = False
    if not args.use_auxiliary_analysis:
        args.aux_responses = []

    response_paths = [Path(p) for p in args.responses]
    annotations = load_annotations(Path(args.ann_file))
    args.aux_predictions_by_video_id = load_auxiliary_predictions(args.aux_responses) if args.use_auxiliary_analysis else {}
    if args.sample_n is not None: # Default to 100 samples for LLM judge analysis to control runtime
        args.outdir = args.outdir + '_debugging'
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stem = make_output_stem(response_paths)
    if args.output_suffix:
        suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.output_suffix.strip())
        if suffix:
            stem = f"{stem}_{suffix}"
    judge_cache_path = outdir / f"{stem}_judge_cache.jsonl" if args.llm_judge else None
    judge_cache = load_judge_cache(judge_cache_path) if judge_cache_path else {}
    text_judge_cache_path = outdir / f"{stem}_text_judge_cache.jsonl" if args.text_llm_judge else None
    text_judge_cache = load_judge_cache(text_judge_cache_path) if text_judge_cache_path else {}

    judge = LocalJudge(args.judge_model, args.quantize) if args.llm_judge else None
    text_judge = TextLLMJudge(args.text_judge_model, args.quantize) if args.text_llm_judge else None
    try:
        all_rows: list[dict[str, Any]] = []
        for path in tqdm(response_paths, desc="Processing response files", unit="file"):
            all_rows.extend(
                process_response_file(
                    path=path,
                    annotations=annotations,
                    args=args,
                    judge=judge,
                    judge_cache=judge_cache,
                    judge_cache_path=judge_cache_path,
                    text_judge=text_judge,
                    text_judge_cache=text_judge_cache,
                    text_judge_cache_path=text_judge_cache_path,
                )
            )
    finally:
        if judge is not None:
            judge.close()
        if text_judge is not None:
            text_judge.close()

    csv_path = outdir / f"{stem}_per_sample.csv"
    report_path = outdir / f"{stem}_report.json"
    write_csv(csv_path, all_rows)
    report = {
        "responses": [str(p) for p in response_paths],
        "ann_file": args.ann_file,
        "video_root": args.video_root,
        "vrdata_dir": args.vrdata_dir,
        "analysis_methods": sorted(args.enabled_analysis_methods),
        "llm_judge": bool(args.llm_judge),
        "judge_model": args.judge_model if args.llm_judge else None,
        "text_llm_judge": bool(args.text_llm_judge),
        "text_judge_model": args.text_judge_model if args.text_llm_judge else None,
        "aux_responses": args.aux_responses,
        "per_sample_csv": str(csv_path),
        **build_report(all_rows, args.enabled_analysis_methods),
    }
    # Backward-compatible JSON aliases; new names below make method ownership explicit.
    report["recognition_counts"] = report.get("hardcoded_recognition_counts", {})
    report["llm_text_recognition_counts"] = report.get("text_llm_judge_recognition_counts", {})
    report.update(write_report_csvs(outdir, stem, report, all_rows))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(all_rows)} per-sample rows to {csv_path}")
    print(f"Wrote method recognition per-sample rows to {report['method_recognition_per_sample_csv']}")
    print(f"Wrote hardcoded recognition counts to {report['hardcoded_recognition_counts_csv']}")
    print(f"Wrote text-LLM judge recognition counts to {report['text_llm_judge_recognition_counts_csv']}")
    print(f"Wrote hardcoded failure counts to {report['hardcoded_failure_counts_csv']}")
    print(f"Wrote report to {report_path}")
    print(f"Accuracy: {report['accuracy']:.4f} ({report['correct']}/{report['num_samples']})")


if __name__ == "__main__":
    main()
