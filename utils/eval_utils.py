"""Shared evaluation utilities for VLM evaluation scripts.

These helpers are model-agnostic and are used by both eval_qwen.py and
eval_vlm.py (and any future evaluation scripts).

Provides:
  - _TCOT_PROMPTS           : text-CoT prompt suffix templates
  - _QUESTION_STARTERS      : Sentence patterns that begin the question part
  - _split_context_question : Heuristic context/question splitter
  - build_prompt_from_annotation : Build MC prompt from annotation dict
  - get_video_frames        : Extract & cache frames from a video file
  - _parse_model_output     : Parse model output into (answer, reasoning)
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import requests


# Models that always require pre-extracted frames (no video-file-path mode)
_FRAMES_REQUIRED_MODELS = {"OpenGVLab/InternVL3-2B", "OpenGVLab/InternVL3-8B",
                            "gemini-2.0-flash", "gpt-4o"}

# Per-frame pixel budget: allocate total_pixels across num_frames
# so that per-frame resolution decreases gracefully as frame count increases
_DEFAULT_TOTAL_PIXELS = 20480 * 28 * 28
_DEFAULT_MIN_PIXELS = 16 * 28 * 28


# ---------------------------------------------------------------------------
# Text-CoT prompt suffixes keyed by tcot_type
# ---------------------------------------------------------------------------

_TCOT_PROMPTS = {
    'tcot1': (
        " Let's think step by step. Output format:\n"
        "Reasoning: [maximum 5 sentences about your reasoning]. \n"
        "Answer: [just the letter and action]."
    ),
    'tcot2': (
        " Let's think step by step. You can reason based on 1) dynamic and spatial inference,"
        " 2) interaction and attention, safety & communication, intent analysis. Output format:\n"
        "Reasoning: [maximum 5 sentences about your reasoning]. \n"
        "Answer: [just the letter and action]."
    ),
    'tcot3': (
        " Let's think step by step. Analyze what is seen in the egocentric video."
        " Evaluate the attention presence, perceived proximity, and perceived risk. Output format:\n"
        "Reasoning: [maximum 5 sentences about your reasoning]. \n"
        "Answer: [just the letter and action]."
    ),
    'tcot4': (
        " Let's think step by step. Analyze the egocentric video. First, describe the visual"
        " elements related to the crossing task. Second, Evaluate the attention presence, perceived"
        " proximity, and perceived risk. Thirdly, explain the logic connecting these elements."
        " Finally, provide the final answer. Output format:\n"
        "Reasoning: [maximum 5 sentences about your reasoning]. \n"
        "Answer: [just the letter and option]."
    ),
    'tcot5': (
        " Analyze the situation using exactly three short sentences focusing on spatial dynamic"
        " inference, attention, intent, and perceived risk. Output format:\n"
        "Reasoning: [Exactly 3 sentences of analysis]. \n"
        "Answer: [Just the letter and action]."
    ),
    'tcot6': (
        " You must format your entire response as a valid JSON object matching the exact schema"
        " below. Do not include markdown formatting or any other text."
        " {{'spatial analysis': 'In one sentence, analyze the gaze trajectory.',"
        " 'attention analysis': 'In one sentence, analyze the attention distribution.',"
        " 'risk analysis': 'In one sentence, state the safety risks.',"
        " 'logic connection': 'In one sentence, state the logic connecting the elements.',"
        " 'final answer': 'the letter and option'}}"
    ),
    'tcot7': (
        " Let's analyze the scene step by step:\n"
        "Step 1: Is the camera wearer currently walking or standing still?\n"
        "Step 2: Is the camera wearer moving toward or away from the white crossing circle?\n"
        "Step 3: Is the automated shuttle close to or far from the camera wearer?\n"
        "Step 4: Is the automated shuttle moving toward or away from the camera wearer?\n"
        "Step 5: Based on the above, what is the pedestrian's most likely action?\n"
        "Output format:\n"
        "Step 1: [answer]. Step 2: [answer]. Step 3: [answer]. Step 4: [answer].\n"
        "Reasoning: [one sentence]. Answer: [just the letter and action]."
    ),
    'tcot8': (
        " Analyze the following egocentric video frame sequence. Determine how the wearer perceives the situation and what they are likely to do next. "
        "Let's reason through this step-by-step by mapping your observations directly to the timestamps provided above. Format your response exactly like this:\n-" 
        "**Thinking Process:** [Your chronological breakdown referencing specific [t=X.XXs] tags]\n- **Final Answer:** [Your definitive conclusion]"
    ),
    'tcot9': (
        " Before answering, list the exact coordinate / locations of the objects relevant to the crossing task that you can visually identify in the video (e.g. 'shuttle at (x1,y1), white circle at (x2,y2)'). Then, based on these visual cues, determine your most likely action. Output format:\n"
        "Reasoning: [Maximum five sentences of analysis]. \n"
        "Answer: [Just the letter and action]."
    ),
    'tcot10': (
        " Analyze the situation using the labeled objects in the video. "
        "First, list the objects relevant to the crossing task that you can visually identify in the video (e.g. 'shuttle, white circle'). "
        "Then, analyze the spatial and dynamic relationships between these objects. "
        "Finally, based on this analysis, determine your most likely action. Output format:\n"
        "Reasoning: [Maximum five sentences of analysis]. \n"
        "Answer: [Just the letter and action]."
    )
}


# Backwards-compatible prompt aliases for older callers/scripts.
_COT_PROMPTS = {**_TCOT_PROMPTS, **{k.replace('tcot', 'cot'): v for k, v in _TCOT_PROMPTS.items()}}


def normalize_tcot_type(tcot_type: str | None) -> str | None:
    """Normalize text-CoT names while accepting legacy cotN aliases."""
    if tcot_type in (None, 'none'):
        return None
    if isinstance(tcot_type, str) and tcot_type.startswith('cot'):
        return 't' + tcot_type
    return tcot_type


# ---------------------------------------------------------------------------
# Context / question splitting
# ---------------------------------------------------------------------------

# Sentence-level prefixes that signal the start of the actual question.
# The splitter looks for the first sentence that *starts with* one of these
# patterns (case-insensitive). Everything before it becomes context_text;
# from it onwards becomes question_text.
_QUESTION_STARTERS = (
    r"based on",
    r"which action",
    r"what is your most likely action",
    r"what action",
    r"are you going to",
    r"which option",
    r"what do you",
)


def _split_context_question(merged: str) -> tuple[str, str]:
    """Heuristically split a merged question string into (context, question).

    Sentences are split on '. ' / '? ' / '! '.  The first sentence whose
    lowercased content starts with one of ``_QUESTION_STARTERS`` is treated
    as the start of the question part; everything before it is the context.
    If no split point is found the entire string is returned as the question
    with an empty context.

    Args:
        merged: The full, unsplit question string from the annotation.

    Returns:
        (context_text, question_text) — either part may be empty.
    """
    parts = re.split(r'(?<=[\.?\!])\s+', merged.strip())
    pattern = re.compile(
        r'^(?:' + '|'.join(_QUESTION_STARTERS) + r')', re.IGNORECASE)
    for i, part in enumerate(parts):
        if pattern.match(part.strip()):
            context = ' '.join(parts[:i]).strip()
            question = ' '.join(parts[i:]).strip()
            return context, question
    # No split point found — treat the whole string as the question
    return '', merged.strip()


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

# Structured prompt variants. ``question`` intentionally contains no options;
# build_prompt_from_annotation appends MC options and CoT instructions.
_PROMPT_SPECS = {
    'p0': {
        'context': (
            "You are a pedestrian in a shared space environment where pedestrians and vehicles share "
            "the same area without strict traffic rules. You need to get to the white circle safely."
        ),
        'question': (
            "Based on what you saw in the egocentric video for the past {past:g} seconds, "
            "what is your most likely action in the next {future:g} second?"
        ),
    },
    'p1': {
        'context': "The video is an egocentric clip from a shared pedestrian-vehicle space.",
        'question': "Based on the past {past:g} seconds, which action does the pedestrian most likely take in the next {future:g} second?",
    },
    'p2': {
        'context': (
            "You are a pedestrian in a shared space environment where pedestrians and vehicles share "
            "the same area without strict traffic rules. You need to get to the white circle safely."
        ),
        'question': (
            "Based ONLY on what is visually observable in the egocentric video for the past {past:g} seconds "
            "(do not rely on general assumptions), what is your most likely action in the next {future:g} second?"
        ),
    },
    'p3': {
        'context': "Watch the egocentric video carefully.",
        'question': (
            "Based ONLY on the visual evidence in this specific video clip (do not rely on general assumptions), "
            "which action does the pedestrian most likely take in the next {future:g} second?"
        ),
    },
    'p4': {
        'context': (
            "You're a pedestrian walking in a shared space where pedestrians and vehicles share the same area "
            "without strict traffic rules. The videos show what you saw in the last {past:g} seconds."
        ),
        'question': "Based on the visual cues, are you going to cross or yield in the next {future:g} second?",
    },
    'p5': {
        'context': (
            "You're a pedestrian walking in a shared space where pedestrians and vehicles share the same area "
            "without strict traffic rules. The videos show what you saw in the last {past:g} seconds."
        ),
        'question': "Based on the visual cues, are you going to yield or cross in the next {future:g} second?",
    },
    'p6': {
        'context': (
            "You're a pedestrian walking in a shared space where pedestrians and vehicles share the same area "
            "without strict traffic rules. The videos show what you saw in the last {past:g} seconds."
        ),
        'question': "Which option best describes your behavior in the next second?",
    },
    'p7': {
        'context': (
            "You are watching an egocentric clip of a pedestrian navigating around an automated shuttle "
            "and a white crossing circle."
        ),
        'question': "Using the video evidence, choose the most likely future behavior in the next {future:g} second.",
    },
}

_PROMPT_VARIANTS = {
    key: (f"{spec['context']} {spec['question']}").strip()
    for key, spec in _PROMPT_SPECS.items()
}

_TASK_QUESTION_TEMPLATES = {
    'crossing_intention': "Which option best describes your behavior in the next second?",
    'aux_speed': "Is the closest vehicle (automated shuttle) moving or stopped?",
    'speed': "Is the closest vehicle (automated shuttle) moving or stopped?",
    'aux_approach': "Is the closest vehicle (automated shuttle) moving toward you?",
    'approach': "Is the closest vehicle (automated shuttle) moving toward you?",
    'aux_ped_moving': "Is the camera wearer walking or standing still?",
    'ped_moving': "Is the camera wearer walking or standing still?",
    'aux_ped_direction': "Is the camera wearer moving toward or away from the target crossing area (the white circle on the ground)?",
    'ped_direction': "Is the camera wearer moving toward or away from the target crossing area (the white circle on the ground)?",
    'aux_gaze_vehicle': "Was the camera wearer looking at the approaching automated shuttle?",
    'gaze_vehicle': "Was the camera wearer looking at the approaching automated shuttle?",
    'aux_ehmi': "Based on its external display signal, will the automated shuttle yield to you?",
    'ehmi': "Based on its external display signal, will the automated shuttle yield to you?",
    'aux_head_turning': "Which way is the camera wearer turning their head?",
    'head_turning': "Which way is the camera wearer turning their head?",
    'aux_vehicle_proximity': "Is the automated shuttle close to or far from the camera wearer?",
    'vehicle_proximity': "Is the automated shuttle close to or far from the camera wearer?",
    'aux_crossing_proximity': "Is the camera wearer close to or far from the crossing area (the white circle on the ground)?",
    'crossing_proximity': "Is the camera wearer close to or far from the crossing area (the white circle on the ground)?",
}

_CONTEXT_FEATURES = {'ego_motion', 'vehicle_motion', 'gaze_direction', 'gaze_direction_change', 'gaze_on_screen', 'gaze_on_screen_ratio', 'gaze_target', 'pose', 'demographics'}

def infer_task_from_path(path: str | os.PathLike[str] | None) -> str | None:
    if not path:
        return None
    name = Path(path).name
    match = re.match(r"annotations\.VR[^_]*__(.+)__[^_]+_close\.json$", name)
    if match:
        return match.group(1)
    legacy = re.match(r"annotations\.VR[^_]*_aux_(.+?)_(?:full|train|val|test|testing)_close\.json$", name)
    if legacy:
        return f"aux_{legacy.group(1)}"
    return "crossing_intention"

def _clip_duration(anno: dict[str, Any], default: float = 2.0) -> float:
    try:
        return float(anno.get('video_end_sec')) - float(anno.get('video_start_sec'))
    except (TypeError, ValueError):
        return default

def _future_duration(anno: dict[str, Any], default: float = 1.0) -> float:
    meta = anno.get('task') if isinstance(anno.get('task'), dict) else {}
    try:
        return float(meta.get('future_sec', default))
    except (TypeError, ValueError):
        return default

def resolve_prompt_parts(anno: dict[str, Any], prompt_variant: str | None = None, task_name: str | None = None) -> tuple[str, str]:
    raw_question = anno.get('question')
    if raw_question:
        if 'context' in anno:
            return (anno.get('context') or '').strip(), str(raw_question).strip()
        return _split_context_question(str(raw_question))
    past = _clip_duration(anno)
    future = _future_duration(anno)
    task_name = task_name or anno.get('task_name') or anno.get('task')
    if isinstance(task_name, dict):
        task_name = task_name.get('name')
    task_name = str(task_name or 'crossing_intention')
    spec = _PROMPT_SPECS.get(prompt_variant or 'p0', _PROMPT_SPECS['p0'])
    context = spec['context'].format(past=past, future=future)
    if task_name != 'crossing_intention':
        base_task = task_name.replace('aux_', '')
        question_template = _TASK_QUESTION_TEMPLATES.get(task_name, _TASK_QUESTION_TEMPLATES.get(base_task, spec['question']))
    else:
        question_template = spec['question']
    question = question_template.format(past=past, future=future)
    return context.strip(), question.strip()

def parse_context_features(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value in (None, '', 'none', []):
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(',') if p.strip()]
    else:
        parts = [str(p).strip() for p in value if str(p).strip()]
    if 'none' in parts:
        return []
    if 'all' in parts:
        return sorted(_CONTEXT_FEATURES)
    unknown = [p for p in parts if p not in _CONTEXT_FEATURES]
    if unknown:
        raise ValueError(f"Unknown context feature(s): {', '.join(unknown)}")
    return parts

_FEATURE_DISPLAY_NAMES = {
    'ego_motion': 'Ego motion cues',
    'vehicle_motion': 'Vehicle motion cues',
    'gaze_direction': 'Gaze direction',
    'gaze_direction_change': 'Gaze direction relative to previous frame',
    'gaze_on_screen': 'Gaze position on screen',
    'gaze_on_screen_ratio': 'Gaze position ratio on screen',
    'gaze_target': 'Gaze target',
    'pose': 'Head pose',
    'demographics': 'Your demographics',
}

_FIELD_DISPLAY_NAMES = {
    'ped_x': 'pedestrian x position',
    'ped_y': 'pedestrian y position',
    'ped_speed': 'pedestrian speed',
    'ped_vx': 'pedestrian x velocity',
    'ped_vy': 'pedestrian y velocity',
    'leader_x': 'vehicle x position',
    'leader_vx': 'vehicle x velocity',
    'follower_x': 'following vehicle x position',
    'dist_to_crossing': 'distance to crossing',
    'leader_rel_x': 'vehicle relative x distance',
    'yaw': 'horizontal angle',
    'x': 'screen x',
    'y': 'screen y',
    'valid': 'valid measurement',
    'hit_object': 'looked-at object',
    'head_yaw': 'head horizontal angle',
    'head_vel': 'head turning speed',
    'x_ratio': 'screen x ratio',
    'y_ratio': 'screen y ratio',
    'yaw_delta_previous': 'yaw difference from previous frame',
    'relative_direction': 'direction relative to previous frame',
}

_FIELD_UNITS = {
    'yaw': 'deg',
    'head_yaw': 'deg',
    'head_vel': 'deg/s',
    'ped_speed': 'cm/s',
    'ped_vx': 'cm/s',
    'ped_vy': 'cm/s',
    'leader_vx': 'cm/s',
    'ped_x': 'cm',
    'ped_y': 'cm',
    'leader_x': 'cm',
    'follower_x': 'cm',
    'dist_to_crossing': 'cm',
    'leader_rel_x': 'cm',
    'yaw_delta_previous': 'deg',
}

_FEATURE_INTERPRETATIONS = {
    'ego_motion': 'Ego motion reports your position and velocity over the observation window.',
    'vehicle_motion': 'Vehicle motion reports vehicle position, velocity, and relative distance to the pedestrian over the observation window.',
    'gaze_direction': 'Gaze direction reports the top-down view horizontal gaze angle in degrees; more negative values indicate looking more to the left, and more positive values indicate looking more to the right.',
    'gaze_direction_change': 'Gaze direction compared to the previous sampled frame shows whether gaze shifted left or right since the prior frame.',
    'gaze_on_screen': 'Gaze position on screen reports the fixation coordinates in the video image, where x increases left-to-right and y increases top-to-bottom.',
    'gaze_on_screen_ratio': 'Gaze position ratio reports normalized fixation coordinates, where x_ratio and y_ratio are approximately in [0, 1].',
    'gaze_target': 'Gaze target reports which scene object the gaze ray/fixation is associated with when available.',
    'pose': 'Head pose reports head orientation and turning speed over time.',
    'demographics': 'Demographics are static person-level attributes and behavioral scores.',
}


_CONTEXT_FEATURE_FORMATS = {'detailed', 'legacy', 'compact', 'schema', 'summary'}

_FEATURE_LEGACY_DESCRIPTIONS = {
    'ego_motion': 'your position and velocity',
    'vehicle_motion': 'vehicle position, velocity, and relative distance to you',
    'gaze_direction': 'your gaze direction in degrees',
    'gaze_direction_change': 'your gaze direction relative to previous frame',
    'gaze_on_screen': 'your gaze on screen in {x, y}',
    'gaze_on_screen_ratio': 'your gaze on screen ratio in {x_ratio, y_ratio}',
    'gaze_target': 'your gaze target',
    'pose': 'your head pose',
}

_FEATURE_LEGACY_SCHEMA_DESCRIPTIONS = {
    'ego_motion': 'your motion',
    'vehicle_motion': 'vehicle motion',
    'gaze_direction': 'your gaze direction in degrees',
    'gaze_direction_change': 'your gaze direction relative to previous frame',
    'gaze_on_screen': 'your gaze on screen',
    'gaze_on_screen_ratio': 'your gaze on screen',
    'gaze_target': 'your gaze target',
    'pose': 'your head pose',
}

_COMPACT_FIELD_NAMES = {
    'ped_x': 'ped_x',
    'ped_y': 'ped_y',
    'ped_speed': 'ped_speed',
    'ped_vx': 'ped_vx',
    'ped_vy': 'ped_vy',
    'leader_x': 'veh_x',
    'leader_vx': 'veh_vx',
    'follower_x': 'follow_x',
    'dist_to_crossing': 'cross_dist',
    'leader_rel_x': 'veh_rel_x',
    'yaw': 'yaw',
    'yaw_delta_previous': 'yaw_delta_previous',
    'relative_direction': 'relative_direction',
    'x': 'x',
    'y': 'y',
    'x_ratio': 'x_ratio',
    'y_ratio': 'y_ratio',
    'valid': 'valid',
    'hit_object': 'target',
    'head_yaw': 'head_yaw',
    'head_vel': 'head_vel',
}


def _normalize_context_feature_format(context_feature_format: str | None) -> str:
    fmt = (context_feature_format or 'detailed').strip().lower()
    if fmt not in _CONTEXT_FEATURE_FORMATS:
        valid = ', '.join(sorted(_CONTEXT_FEATURE_FORMATS))
        raise ValueError(f"--context_feature_format must be one of: {valid}")
    return fmt


def _display_feature_name(feature: str) -> str:
    return _FEATURE_DISPLAY_NAMES.get(feature, feature.replace('_', ' ').title())


def _display_field_name(field: str) -> str:
    return _FIELD_DISPLAY_NAMES.get(field, field.replace('_', ' '))


def _compact_field_name(field: str) -> str:
    return _COMPACT_FIELD_NAMES.get(field, field)


def _fmt_field_value(field: str, value: Any) -> str:
    if field in {'x_ratio', 'y_ratio'} and isinstance(value, (float, np.floating, int, np.integer)):
        value = float(value)
        return "NA" if np.isnan(value) else f"{value:.2f}"
    return _fmt_value(value)


def _render_field_value(field: str, value: Any) -> str:
    text = _fmt_field_value(field, value)
    unit = _FIELD_UNITS.get(field)
    if unit and text != 'NA':
        return f"{text} {unit}"
    return text


def build_context_feature_interpretation(context_features: str | list[str] | tuple[str, ...] | None, mode: str = 'none', context_feature_format: str | None = None) -> str:
    selected = parse_context_features(context_features)
    if mode in (None, '', 'none') or not selected:
        return ''
    if mode not in {'brief', 'detailed'}:
        raise ValueError("--context_feature_interpretation must be 'none', 'brief', or 'detailed'")
    context_feature_format = _normalize_context_feature_format(context_feature_format)
    if context_feature_format == 'schema':
        lines = ['- Schema rows: each tuple follows the variable order shown in the time-series header.']
        if mode == 'detailed':
            lines.extend([
                '- Timestamps use [t=X.XXs], measured from the beginning of the 2-second clip.',
                '- For ratio features, 0 is the left/top edge of the screen and 1 is the right/bottom edge.',
                '- For gaze_direction_change, left/right is relative to the previous sampled frame, not absolute world direction.',
            ])
        return "\n".join(lines)
    lines = []
    for feature in selected:
        text = _FEATURE_INTERPRETATIONS.get(feature)
        if text:
            lines.append(f"- {_display_feature_name(feature)}: {text}")
    if mode == 'detailed':
        lines.extend([
            '- Timestamps use [t=X.XXs], measured from the beginning of the 2-second clip.',
            '- For ratio features, 0 is the left/top edge of the screen and 1 is the right/bottom edge.',
            '- For gaze_direction_change, left/right is relative to the previous sampled frame, not absolute world direction.',
        ])
    return "\n".join(lines)

def _mean_ts(ts: Any) -> float:
    arr = np.asarray(ts)
    return float(arr.mean()) if arr.size else 0.0

def _target_times_for_feature_sampling(anno: dict[str, Any], context_feature_fps: str | float = 'auto', timestamps: Any = None, num_frames: int | None = None, video_duration: float | None = None) -> list[float]:
    duration = float(video_duration or _clip_duration(anno))
    if str(context_feature_fps) == 'auto':
        if timestamps is not None:
            return [_mean_ts(ts) for ts in timestamps]
        n = min(int(num_frames or 8), 8)
        return np.linspace(0.0, max(0.0, duration), num=max(1, n)).round(4).tolist()
    fps = float(context_feature_fps)
    if fps <= 0:
        raise ValueError("--context_feature_fps must be 'auto' or a positive number")
    n = max(1, int(round(duration * fps)) + 1)
    return np.linspace(0.0, max(0.0, duration), num=n).round(4).tolist()

def _nearest_samples(stream: dict[str, Any], target_times: list[float]) -> list[dict[str, Any]]:
    samples = stream.get('samples') or []
    if not samples:
        return []
    out = []
    for target in target_times:
        best = min(samples, key=lambda row: abs(float(row.get('t', row.get('time', 0.0))) - target))
        out.append(best)
    return out

def _fmt_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isnan(value):
            return "NA"
        text = f"{value:.1f}"
        return text[:-2] if text.endswith('.0') else text
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}:{_fmt_value(v)}" for k, v in value.items()) + "}"
    return str(value)

def _relative_time(anno: dict[str, Any], row: dict[str, Any], window_end_sec: float | None = None) -> float:
    end_sec = _clip_duration(anno) if window_end_sec is None else float(window_end_sec)
    return float(row.get('t', row.get('time', 0.0))) - end_sec


_FEATURE_STREAM_SOURCES = {
    'ego_motion': 'motion',
    'vehicle_motion': 'motion',
    'gaze_direction_change': 'gaze_direction',
    'gaze_on_screen_ratio': 'gaze_on_screen',
}

_FEATURE_VECTOR_SCHEMAS = {
    'ego_motion': [
        ('ped_x', 'pedestrian position in x'),
        ('ped_y', 'pedestrian position in y'),
        ('ped_speed', 'pedestrian speed'),
        ('ped_vx', 'pedestrian speed in x'),
        ('ped_vy', 'pedestrian speed in y'),
    ],
    'vehicle_motion': [
        ('leader_x', 'vehicle position in x'),
        ('leader_vx', 'vehicle speed in x'),
        ('follower_x', 'following vehicle position in x'),
        ('leader_rel_x', 'vehicle relative position in x'),
    ],
    'gaze_direction': [
        ('yaw', 'gaze yaw angle'),
    ],
    'gaze_direction_change': [
        ('relative_direction', 'relative gaze direction'),
    ],
    'gaze_on_screen': [
        ('x', 'screen position in x'),
        ('y', 'screen position in y'),
    ],
    'gaze_on_screen_ratio': [
        ('x_ratio', 'screen position ratio in x'),
        ('y_ratio', 'screen position ratio in y'),
        ('valid', 'whether gaze point is valid'),
    ],
    'gaze_target': [
        ('hit_object', 'gaze target'),
    ],
    'pose': [
        ('head_yaw', 'head yaw angle'),
        ('head_vel', 'head turning speed'),
    ],
}

_FEATURE_FIELDS = {feature: {field for field, _ in schema} for feature, schema in _FEATURE_VECTOR_SCHEMAS.items()}


def _source_feature(feature: str) -> str:
    return _FEATURE_STREAM_SOURCES.get(feature, feature)


def _stream_for_feature(clip_features: dict[str, Any], feature: str) -> Any:
    return clip_features.get(_source_feature(feature))


def _gaze_direction_word(delta: float, eps: float = 1.0) -> str:
    if delta < -eps:
        return 'left'
    if delta > eps:
        return 'right'
    return 'same'


def _screen_size(stream: dict[str, Any]) -> tuple[float, float]:
    width = stream.get('screen_width') or stream.get('width') or stream.get('image_width')
    height = stream.get('screen_height') or stream.get('height') or stream.get('image_height')
    # The gaze files do not currently store screen dimensions. Use the original
    # eye-tracker/video coordinate frame fallback instead of per-clip maxima so
    # ratios are comparable across clips.
    if width is None:
        width = 1600.0
    if height is None:
        height = 1200.0
    return max(float(width), 1.0), max(float(height), 1.0)


def _is_false_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {'false', '0', 'no', 'n'}
    return value is False or value == 0


def _is_invalid_gaze_screen_row(feature: str, row: dict[str, Any]) -> bool:
    return feature in {'gaze_on_screen', 'gaze_on_screen_ratio'} and 'valid' in row and _is_false_value(row.get('valid'))


def _mark_invalid_gaze_row(base: dict[str, Any]) -> dict[str, Any]:
    base['_invalid'] = True
    return base


def _is_invalid_render_row(feature: str, row: dict[str, Any]) -> bool:
    return feature in {'gaze_on_screen', 'gaze_on_screen_ratio'} and bool(row.get('_invalid'))


def _row_for_feature(feature: str, row: dict[str, Any], stream: dict[str, Any] | None = None, previous_row: dict[str, Any] | None = None) -> dict[str, Any]:
    base = {k: v for k, v in row.items() if k in {'t', 'time'}}
    if feature == 'gaze_on_screen':
        if _is_invalid_gaze_screen_row(feature, row):
            return _mark_invalid_gaze_row(base)
        for key in ('x', 'y'):
            if key in row:
                base[key] = row.get(key)
        return base
    if feature == 'gaze_on_screen_ratio':
        if _is_invalid_gaze_screen_row(feature, row):
            return _mark_invalid_gaze_row(base)
        width, height = _screen_size(stream or {})
        x = row.get('x')
        y = row.get('y')
        if x is not None:
            base['x_ratio'] = max(0.0, min(1.0, float(x) / width))
        if y is not None:
            base['y_ratio'] = max(0.0, min(1.0, float(y) / height))
        return base
    if feature == 'gaze_direction_change':
        yaw = row.get('yaw')
        previous_yaw = (previous_row or {}).get('yaw')
        if yaw is None or previous_yaw is None:
            # base['yaw_delta_previous'] = None
            base['relative_direction'] = None
        else:
            delta = float(yaw) - float(previous_yaw)
            # base['yaw_delta_previous'] = delta
            base['relative_direction'] = _gaze_direction_word(delta)
        return base
    if feature in _FEATURE_FIELDS:
        base.update({k: v for k, v in row.items() if k in _FEATURE_FIELDS[feature]})
        return base
    return dict(row)


def _rows_for_feature(feature: str, rows: list[dict[str, Any]], stream: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    out = []
    previous_row = None
    for row in rows:
        out.append(_row_for_feature(feature, row, stream=stream, previous_row=previous_row))
        previous_row = row
    return out

def _legacy_value(feature: str, row: dict[str, Any]) -> Any:
    if _is_invalid_render_row(feature, row):
        return None
    if feature in {'ego_motion', 'vehicle_motion', 'gaze_on_screen_ratio', 'gaze_direction_change'}:
        return {k: v for k, v in row.items() if k not in {'t', 'time', '_invalid'}}
    if feature == 'gaze_on_screen':
        return {'x': row.get('x'), 'y': row.get('y')}
    if feature == 'gaze_direction':
        return row.get('yaw')
    if feature == 'gaze_target':
        return row.get('hit_object')
    if feature == 'pose':
        return {'head_yaw': row.get('head_yaw'), 'head_vel': row.get('head_vel')}
    return {k: v for k, v in row.items() if k not in {'t', 'time', '_invalid'}}


def _render_sample(feature: str, row: dict[str, Any], include_timestamp: bool = True) -> str:
    t = float(row.get('t', row.get('time', 0.0)))
    if _is_invalid_render_row(feature, row):
        body = f"{_display_feature_name(feature)}: NA"
        return f"[t={t:.2f}s] {body}" if include_timestamp else body
    parts = []
    for key, value in row.items():
        if key in {'t', 'time', '_invalid'}:
            continue
        parts.append(f"{_display_field_name(key)}={_render_field_value(key, value)}")
    body = f"{_display_feature_name(feature)}: " + ", ".join(parts)
    return f"[t={t:.2f}s] {body}" if include_timestamp else body


def _render_compact_sample(anno: dict[str, Any], feature: str, row: dict[str, Any], include_timestamp: bool = True, window_end_sec: float | None = None) -> str:
    if _is_invalid_render_row(feature, row):
        body = f"{feature}: NA"
        t = float(row.get('t', row.get('time', 0.0)))
        return f"[t={t:.2f}s] {body}" if include_timestamp else body
    parts = []
    for key, value in row.items():
        if key in {'t', 'time', '_invalid'}:
            continue
        parts.append(f"{_compact_field_name(key)}={_fmt_field_value(key, value)}")
    body = f"{feature}: " + ", ".join(parts)
    t = float(row.get('t', row.get('time', 0.0)))
    return f"[t={t:.2f}s] {body}" if include_timestamp else body


def _render_legacy_value(value: Any) -> str:
    if isinstance(value, dict):
        parts = [f"{k}={_fmt_field_value(k, v)}" for k, v in value.items()]
        return "{" + ", ".join(parts) + "}"
    return _fmt_value(value)


def _schema_items_for_feature(feature: str, row: dict[str, Any]) -> list[tuple[str, str]]:
    schema = _FEATURE_VECTOR_SCHEMAS.get(feature)
    if schema:
        if _is_invalid_render_row(feature, row):
            return schema
        return [(field, desc) for field, desc in schema if field in row]
    return [(field, _display_field_name(field)) for field in row if field not in {'t', 'time', '_invalid'}]


def _render_schema_values(feature: str, row: dict[str, Any]) -> str:
    if _is_invalid_render_row(feature, row):
        return "NA"
    values = [_fmt_field_value(field, row.get(field)) for field, _ in _schema_items_for_feature(feature, row)]
    return "{" + ", ".join(values) + "}"


def _render_schema_header(feature: str, row: dict[str, Any]) -> str:
    labels = [desc for _, desc in _schema_items_for_feature(feature, row)]
    return "{" + ", ".join(labels) + "}"


def _render_feature_payload(feature: str, row: dict[str, Any], context_feature_format: str) -> str:
    if context_feature_format == 'schema':
        return _render_schema_values(feature, row)
    if context_feature_format == 'legacy':
        return _render_legacy_value(_legacy_value(feature, row))
    if context_feature_format == 'compact':
        if _is_invalid_render_row(feature, row):
            return 'NA'
        parts = []
        for key, value in row.items():
            if key in {'t', 'time', '_invalid'}:
                continue
            parts.append(f"{_compact_field_name(key)}={_fmt_field_value(key, value)}")
        return ", ".join(parts) if parts else 'NA'
    if context_feature_format == 'detailed':
        if _is_invalid_render_row(feature, row):
            return 'NA'
        parts = []
        for key, value in row.items():
            if key in {'t', 'time', '_invalid'}:
                continue
            parts.append(f"{_display_field_name(key)}={_render_field_value(key, value)}")
        return ", ".join(parts) if parts else 'NA'
    return _render_legacy_value(_legacy_value(feature, row))


def _merged_schema_header(feature_rows: dict[str, list[dict[str, Any]]], context_feature_format: str) -> str:
    pieces = []
    for feature, rows in feature_rows.items():
        if not rows:
            continue
        if context_feature_format == 'schema':
            pieces.append(f"{feature}: {_render_schema_header(feature, rows[0])}")
        elif context_feature_format == 'legacy':
            desc = _FEATURE_LEGACY_DESCRIPTIONS.get(feature, feature.replace('_', ' '))
            pieces.append(f"{feature}: {desc}")
        elif context_feature_format == 'compact':
            pieces.append(feature)
        else:
            pieces.append(_display_feature_name(feature))
    return "{" + "; ".join(pieces) + "}"


def _format_merged_feature_stream(anno: dict[str, Any], target_times: list[float], feature_rows: dict[str, list[dict[str, Any]]], context_feature_format: str) -> str:
    header = _merged_schema_header(feature_rows, context_feature_format)
    pairs = []
    for i, target in enumerate(target_times):
        parts = []
        for feature, rows in feature_rows.items():
            if i >= len(rows):
                continue
            parts.append(f"{feature}={_render_feature_payload(feature, rows[i], context_feature_format)}")
        if parts:
            pairs.append(f"[[t={float(target):.2f}s], " + "{" + "; ".join(parts) + "}]")
    if context_feature_format == 'schema':
        return "The following time series shows the selected context cues, in [[t=X.XXs], " + header + "]: " + ", ".join(pairs) + "."
    if context_feature_format == 'legacy':
        return "The following time series are observed pairs in the past " + f"{_clip_duration(anno):g}" + " seconds of [[t=X.XXs], " + header + "]: " + ", ".join(pairs) + "."
    return "\n".join(pairs)


def _format_schema_feature_block(anno: dict[str, Any], feature: str, sampled: list[dict[str, Any]], window_end_sec: float | None = None) -> str:
    desc = _FEATURE_LEGACY_SCHEMA_DESCRIPTIONS.get(feature, feature.replace('_', ' '))
    schema = _render_schema_header(feature, sampled[0]) if sampled else '{}'
    pairs = []
    for row in sampled:
        t = float(row.get('t', row.get('time', 0.0)))
        pairs.append(f"[[t={t:.2f}s], {_render_schema_values(feature, row)}]")
    return (
        f"The following time series shows {desc}, in [[t=X.XXs], {schema}]: "
        + ", ".join(pairs)
        + "."
    )


def _format_interleaved_feature_preface(feature: str, sampled: list[dict[str, Any]], context_feature_format: str) -> str:
    if not sampled:
        return ''
    if context_feature_format == 'schema':
        desc = _FEATURE_LEGACY_SCHEMA_DESCRIPTIONS.get(feature, feature.replace('_', ' '))
        schema = _render_schema_header(feature, sampled[0])
        return f"The interleaved time series shows {desc}, {feature}: {schema}."
    if context_feature_format == 'legacy':
        desc = _FEATURE_LEGACY_DESCRIPTIONS.get(feature, feature.replace('_', ' '))
        return f"The interleaved time series {feature} shows {desc}."
    if context_feature_format == 'compact':
        return f"Compact interleaved clip cues are shown beside the matching frame timestamps for {feature}."
    if context_feature_format == 'summary':
        return ''
    return f"Structured interleaved clip features are shown beside the matching frame timestamps for {_display_feature_name(feature)}."


def _format_legacy_feature_block(anno: dict[str, Any], feature: str, sampled: list[dict[str, Any]], window_end_sec: float | None = None) -> str:
    desc = _FEATURE_LEGACY_DESCRIPTIONS.get(feature, feature.replace('_', ' '))
    pairs = []
    for row in sampled:
        t = float(row.get('t', row.get('time', 0.0)))
        pairs.append(f"[[t={t:.2f}s], {_render_legacy_value(_legacy_value(feature, row))}]")
    return (
        "The following time series are observed pairs in the past "
        f"{_clip_duration(anno):g} seconds of [[t=X.XXs], {desc}]: "
        + ", ".join(pairs)
        + "."
    )


def _trend_word(delta: float, eps: float = 1e-3) -> str:
    if delta > eps:
        return "increases"
    if delta < -eps:
        return "decreases"
    return "stays about the same"


def _render_summary_feature(anno: dict[str, Any], feature: str, sampled: list[dict[str, Any]], window_end_sec: float | None = None) -> str:
    first = sampled[0]
    last = sampled[-1]
    t0 = _relative_time(anno, first, window_end_sec)
    t1 = _relative_time(anno, last, window_end_sec)
    if feature in {'gaze_on_screen', 'gaze_on_screen_ratio'}:
        x_key = 'x_ratio' if feature == 'gaze_on_screen_ratio' else 'x'
        y_key = 'y_ratio' if feature == 'gaze_on_screen_ratio' else 'y'
        dx = float(last.get(x_key, 0.0) or 0.0) - float(first.get(x_key, 0.0) or 0.0)
        dy = float(last.get(y_key, 0.0) or 0.0) - float(first.get(y_key, 0.0) or 0.0)
        return f"{feature} from [t={float(first.get('t', first.get('time', 0.0))):.2f}s] to [t={float(last.get('t', last.get('time', 0.0))):.2f}s]: {x_key} {_trend_word(dx)}, {y_key} {_trend_word(dy)}; last=({x_key}={_fmt_value(last.get(x_key))}, {y_key}={_fmt_value(last.get(y_key))})."
    if feature == 'gaze_direction_change':
        return f"gaze_direction_change from [t={float(first.get('t', first.get('time', 0.0))):.2f}s] to [t={float(last.get('t', last.get('time', 0.0))):.2f}s]: each value compares gaze with the previous sampled frame; final relative_direction={last.get('relative_direction', 'NA')}, final yaw_delta_previous={_fmt_value(last.get('yaw_delta_previous'))}."
    if feature == 'gaze_direction':
        dyaw = float(last.get('yaw', 0.0) or 0.0) - float(first.get('yaw', 0.0) or 0.0)
        return f"gaze_direction from [t={float(first.get('t', first.get('time', 0.0))):.2f}s] to [t={float(last.get('t', last.get('time', 0.0))):.2f}s]: yaw {_trend_word(dyaw)}; last yaw={_fmt_value(last.get('yaw'))} deg."
    if feature == 'gaze_target':
        values = [str(row.get('hit_object')) for row in sampled if row.get('hit_object') not in (None, '')]
        if values:
            return f"gaze_target over the clip: {' -> '.join(values)}."
        return "gaze_target over the clip: no valid target."
    if feature == 'pose':
        dyaw = float(last.get('head_yaw', 0.0) or 0.0) - float(first.get('head_yaw', 0.0) or 0.0)
        return f"head_pose from [t={float(first.get('t', first.get('time', 0.0))):.2f}s] to [t={float(last.get('t', last.get('time', 0.0))):.2f}s]: head_yaw {_trend_word(dyaw)}; last head_vel={_fmt_value(last.get('head_vel'))} deg/s."
    if feature in {'ego_motion', 'vehicle_motion'}:
        pieces = []
        fields = [('ped_speed', 'ped_speed'), ('ped_x', 'ped_x'), ('ped_y', 'ped_y')] if feature == 'ego_motion' else [('leader_x', 'veh_x'), ('leader_vx', 'veh_vx'), ('leader_rel_x', 'veh_rel_x')]
        for field, label in fields:
            if field in first and field in last:
                delta = float(last.get(field, 0.0) or 0.0) - float(first.get(field, 0.0) or 0.0)
                pieces.append(f"{label} {_trend_word(delta)} to {_fmt_value(last.get(field))}")
        return f"{feature} from [t={float(first.get('t', first.get('time', 0.0))):.2f}s] to [t={float(last.get('t', last.get('time', 0.0))):.2f}s]: " + "; ".join(pieces) + "."
    return f"{feature} summary: first={_legacy_value(feature, first)}, last={_legacy_value(feature, last)}."


def build_context_feature_text(anno: dict[str, Any], context_features: str | list[str] | tuple[str, ...] | None = None, context_feature_fps: str | float = 'auto', timestamps: Any = None, num_frames: int | None = None, video_duration: float | None = None, context_feature_interpretation: str = 'none', context_feature_format: str = 'detailed') -> tuple[str, list[str]]:
    selected = parse_context_features(context_features)
    if not selected:
        return "", []
    context_feature_format = _normalize_context_feature_format(context_feature_format)
    clip_features = anno.get('clip_features') or {}
    target_times = _target_times_for_feature_sampling(anno, context_feature_fps, timestamps=timestamps, num_frames=num_frames, video_duration=video_duration)
    lines: list[str] = []
    interpretation = build_context_feature_interpretation(selected, context_feature_interpretation, context_feature_format=context_feature_format)
    if interpretation:
        lines.append(interpretation)
    used: list[str] = []

    dynamic_features: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    for feature in selected:
        data = _stream_for_feature(clip_features, feature)
        if not data:
            continue
        if feature == 'demographics':
            value = data.get('value', data) if isinstance(data, dict) else data
            if value:
                prefix = "Your demographics"
                if context_feature_format in {'compact', 'summary'}:
                    prefix = "demographics"
                lines.append(prefix + ": " + ", ".join(f"{k}={_fmt_value(v)}" for k, v in value.items() if v is not None))
                used.append(feature)
            continue
        if not isinstance(data, dict):
            continue
        sampled = _nearest_samples(data, target_times)
        sampled = _rows_for_feature(feature, sampled, stream=data)
        if sampled:
            dynamic_features.append((feature, data, sampled))
            used.append(feature)

    if len(dynamic_features) > 1 and context_feature_format in {'legacy', 'schema', 'compact', 'detailed'}:
        feature_rows = {feature: sampled for feature, _, sampled in dynamic_features}
        lines.append(_format_merged_feature_stream(anno, target_times, feature_rows, context_feature_format))
    else:
        for feature, data, sampled in dynamic_features:
            window_end_sec = data.get('window_end_sec')
            if context_feature_format == 'legacy':
                lines.append(_format_legacy_feature_block(anno, feature, sampled, window_end_sec=window_end_sec))
            elif context_feature_format == 'schema':
                lines.append(_format_schema_feature_block(anno, feature, sampled, window_end_sec=window_end_sec))
            elif context_feature_format == 'compact':
                lines.extend(_render_compact_sample(anno, feature, row, window_end_sec=window_end_sec) for row in sampled)
            elif context_feature_format == 'summary':
                lines.append(_render_summary_feature(anno, feature, sampled, window_end_sec=window_end_sec))
            else:
                lines.extend(_render_sample(feature, row) for row in sampled)

    if not lines:
        return "", []
    if context_feature_format in {'legacy', 'schema'}:
        return "\n".join(lines), used
    if context_feature_format == 'compact':
        return "Compact clip cues:\n" + "\n".join(lines), used
    if context_feature_format == 'summary':
        return "Summary clip cues:\n" + "\n".join(lines), used
    return "Structured clip features:\n" + "\n".join(lines), used

def build_interleaved_context_texts(anno: dict[str, Any], timestamps: Any, context_features: str | list[str] | tuple[str, ...] | None = None, context_feature_format: str = 'detailed') -> list[str]:
    selected = parse_context_features(context_features)
    if not selected or timestamps is None:
        return []
    context_feature_format = _normalize_context_feature_format(context_feature_format)
    target_times = [_mean_ts(ts) for ts in timestamps]
    clip_features = anno.get('clip_features') or {}
    snippets = ["" for _ in target_times]
    feature_rows: dict[str, list[dict[str, Any]]] = {}
    feature_streams: dict[str, dict[str, Any]] = {}
    for feature in selected:
        if feature == 'demographics':
            continue
        stream = _stream_for_feature(clip_features, feature)
        if not isinstance(stream, dict):
            continue
        sampled = _nearest_samples(stream, target_times)
        sampled = _rows_for_feature(feature, sampled, stream=stream)
        if sampled:
            feature_rows[feature] = sampled
            feature_streams[feature] = stream

    if len(feature_rows) > 1:
        for i, _target in enumerate(target_times):
            parts = []
            for feature, rows in feature_rows.items():
                if i >= len(rows):
                    continue
                parts.append(f"{feature}={_render_feature_payload(feature, rows[i], context_feature_format)}")
            snippets[i] = "context: {" + "; ".join(parts) + "}" if parts else ""
        return snippets

    for feature, sampled in feature_rows.items():
        stream = feature_streams[feature]
        for i, row in enumerate(sampled[:len(target_times)]):
            if context_feature_format in {'compact', 'summary'}:
                text = _render_compact_sample(anno, feature, row, include_timestamp=False, window_end_sec=stream.get('window_end_sec'))
            elif context_feature_format == 'legacy':
                text = f"{feature}: {_render_legacy_value(_legacy_value(feature, row))}"
            elif context_feature_format == 'schema':
                text = f"{feature}: {_render_schema_values(feature, row)}"
            else:
                text = _render_sample(feature, row, include_timestamp=False)
            snippets[i] = text
    return snippets

def build_interleaved_context_preface(anno: dict[str, Any], timestamps: Any, context_features: str | list[str] | tuple[str, ...] | None = None, context_feature_format: str = 'detailed') -> str:
    selected = parse_context_features(context_features)
    if not selected or timestamps is None:
        return ''
    context_feature_format = _normalize_context_feature_format(context_feature_format)
    target_times = [_mean_ts(ts) for ts in timestamps]
    clip_features = anno.get('clip_features') or {}
    feature_rows: dict[str, list[dict[str, Any]]] = {}
    lines: list[str] = []
    for feature in selected:
        if feature == 'demographics':
            continue
        stream = _stream_for_feature(clip_features, feature)
        if not isinstance(stream, dict):
            continue
        sampled = _nearest_samples(stream, target_times[:1])
        sampled = _rows_for_feature(feature, sampled, stream=stream)
        if sampled:
            feature_rows[feature] = sampled

    if len(feature_rows) > 1:
        header = _merged_schema_header(feature_rows, context_feature_format)
        if context_feature_format == 'schema':
            return f"The interleaved time series shows the selected context cues, context: {header}."
        if context_feature_format == 'legacy':
            return f"The interleaved time series shows the selected context cues, context: {header}."
        if context_feature_format == 'compact':
            return f"Compact interleaved clip cues are merged beside each matching frame timestamp, context: {header}."
        if context_feature_format == 'summary':
            return ''
        return f"Structured interleaved clip features are merged beside each matching frame timestamp, context: {header}."

    for feature, sampled in feature_rows.items():
        line = _format_interleaved_feature_preface(feature, sampled, context_feature_format)
        if line:
            lines.append(line)
    return "\n".join(lines)

def build_prompt_from_annotation(anno: dict, cot_type: str | None = None, tcot_type: str | None = None, prompt_variant: str | None = None, task_name: str | None = None, context_features: str | list[str] | tuple[str, ...] | None = None, context_feature_fps: str | float = 'auto', timestamps: Any = None, num_frames: int | None = None, video_duration: float | None = None, context_feature_interpretation: str = 'none', context_feature_format: str = 'detailed'):
    """Build a multiple-choice prompt from old or structured annotation dicts."""
    correct = anno.get('answer')
    wrongs = anno.get('wrong_answers')
    context_text, question_stem = resolve_prompt_parts(anno, prompt_variant=prompt_variant, task_name=task_name)
    feature_text, _ = build_context_feature_text(anno, context_features=context_features, context_feature_fps=context_feature_fps, timestamps=timestamps, num_frames=num_frames, video_duration=video_duration, context_feature_interpretation=context_feature_interpretation, context_feature_format=context_feature_format)
    if feature_text:
        context_text = (context_text + "\n" + feature_text).strip() if context_text else feature_text
    if isinstance(wrongs, list):
        wrongs = [w.strip() for w in wrongs if w.strip()]
    options = [str(correct)] + [str(w) for w in wrongs]
    assert len(options) > 1, "Annotation must have at least one correct and one wrong answer"
    random.shuffle(options)
    letters = [chr(ord('A') + i) for i in range(len(options))]
    options_map = {letters[i]: options[i] for i in range(len(options))}
    options_str = ' '.join([f"({k}) {v}" for k, v in options_map.items()])
    resolved_tcot = normalize_tcot_type(tcot_type if tcot_type is not None else cot_type)
    if resolved_tcot in _TCOT_PROMPTS:
        question_text = f"{question_stem} Options are {options_str}." + _TCOT_PROMPTS[resolved_tcot]
    else:
        question_text = f"{question_stem} Choose one option: {options_str}."
    if resolved_tcot is None:
        question_text += " Answer with the correct option only."
    prompt_text = (context_text + ' ' + question_text).strip() if context_text else question_text
    return prompt_text, context_text, question_text, options_map, correct


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------

def _download_video(url: str, dest_path: str) -> None:
    """Download a video from a URL to a local file."""
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    logging.info("Video downloaded to %s", dest_path)


def get_video_frames(video_path: str, num_frames: int = 128,
                     cache_dir: str = '.cache') -> tuple:
    """Extract uniformly-sampled frames (and timestamps) from a video file.

    Results are cached on disk so repeated calls with the same arguments are
    instant.

    Args:
        video_path: Local path or HTTP(S) URL to an MP4 video file.
        num_frames: Number of frames to sample uniformly across the clip.
        cache_dir: Directory used to cache downloaded videos and frame arrays.

    Returns:
        (video_file_path, frames, timestamps)

        - ``video_file_path``: Local path to the (possibly downloaded) video.
        - ``frames``: uint8 RGB array of shape [N, H, W, 3].
        - ``timestamps``: float array of shape [N, 2] with (start, end) times
          for each sampled frame as returned by decord.
    """
    from decord import VideoReader, cpu  # import here to keep module lightweight

    os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
    if video_path.startswith('http://') or video_path.startswith('https://'):
        video_file_path = os.path.join(cache_dir, f'{video_hash}.mp4')
        if not os.path.exists(video_file_path):
            _download_video(video_path, video_file_path)
    else:
        video_file_path = video_path

    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_frames.npz')
    timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_timestamps.npz')

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        try:
            # Older cache files were written with an unnamed positional array
            # (``arr_0``); newer files use the explicit ``data`` key.
            with np.load(frames_cache_file, allow_pickle=False) as archive:
                frames = archive['data'] if 'data' in archive.files else archive['arr_0']
            with np.load(timestamps_cache_file, allow_pickle=False) as archive:
                timestamps = archive['data'] if 'data' in archive.files else archive['arr_0']
            if frames.shape[0] != num_frames or timestamps.shape[0] != num_frames:
                raise ValueError(
                    f"cache length mismatch: frames={frames.shape}, timestamps={timestamps.shape}")
            return video_file_path, frames, timestamps
        except (OSError, ValueError, KeyError, EOFError) as exc:
            logging.warning("Ignoring incompatible video cache for %s: %s", video_path, exc)

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    # Write temporary archives and atomically replace the targets so an
    # interrupted job cannot leave a partially written cache.
    frames_tmp = f'{frames_cache_file}.tmp-{os.getpid()}.npz'
    timestamps_tmp = f'{timestamps_cache_file}.tmp-{os.getpid()}.npz'
    try:
        np.savez_compressed(frames_tmp, data=frames)
        np.savez_compressed(timestamps_tmp, data=timestamps)
        os.replace(frames_tmp, frames_cache_file)
        os.replace(timestamps_tmp, timestamps_cache_file)
    finally:
        for tmp_path in (frames_tmp, timestamps_tmp):
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return video_file_path, frames, timestamps


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _normalize_answer_text(text: str | None) -> str:
    """Normalize answer text for robust label/option matching."""
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"^\s*(?:[(\[][a-z][)\]]|[a-z][)\].:-])\s*", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _match_option_letter(output_text: str, options_map: dict) -> str | None:
    """Return the option label for outputs such as C, (C), C., or Answer: C."""
    if not output_text or not options_map:
        return None
    option_letters = "".join(re.escape(str(k)) for k in options_map.keys())
    patterns = [
        rf"(?i)\b(?:answer|final answer|option)\s*[:\-]?\s*[(\[]?([{option_letters}])[)\].]?\b",
        rf"^\s*[(\[]?([{option_letters}])[)\].]?\s*$",
        rf"^\s*[(\[]?([{option_letters}])[)\].:]\s+",
        rf"(?<![A-Za-z0-9])[(\[]([{option_letters}])[)\]]",
    ]
    for pattern in patterns:
        match = re.search(pattern, output_text)
        if match:
            label = options_map.get(match.group(1).upper())
            if label:
                return str(label).lower()
    return None


def _match_option_text(output_text: str, options_map: dict) -> str | None:
    """Return the option label when the model writes the option content itself."""
    normalized_output = _normalize_answer_text(output_text)
    if not normalized_output:
        return None
    candidates = []
    for label in options_map.values():
        normalized_label = _normalize_answer_text(label)
        if normalized_label:
            candidates.append((normalized_label, str(label).lower()))
    for normalized_label, label in sorted(candidates, key=lambda item: len(item[0]), reverse=True):
        if normalized_output == normalized_label:
            return label
        if re.search(rf"(?<!\w){re.escape(normalized_label)}(?!\w)", normalized_output):
            return label
    return None


def _parse_model_output(output_text: str, options_map: dict,
                        cot_type: str | None = None,
                        tcot_type: str | None = None) -> tuple[str | None, str | None]:
    """Extract (pred_answer, reasoning) from raw model output.

    For CoT outputs, parse Reasoning/Answer blocks. For plain outputs, map
    letter choices back to option text.

    Args:
        output_text: Raw string produced by the model.
        options_map: Mapping of letter → option text (e.g. {'A': 'cross', …}).
        cot_type: The CoT variant used, or None for plain MC output.

    Returns:
        (pred_answer, reasoning) — either may be None if not found.
    """
    resolved_tcot = normalize_tcot_type(tcot_type if tcot_type is not None else cot_type)
    pred_answer = _match_option_letter(output_text, options_map) or _match_option_text(output_text, options_map)
    reasoning = None

    if resolved_tcot is not None:
        match = re.search(r"(?i)Reasoning:\s*(.*?)(?=\s*(?:Answer:|Final Answer|$))",
                          output_text, re.MULTILINE)
        if match:
            reasoning = match.group(1).strip()

    # Fallback for legacy crossing-intention outputs where the full text is a label
    # but not necessarily one of the current multiple-choice options.
    if pred_answer is None and ('cross' in output_text.lower() or 'yield' in output_text.lower()):
        pred_answer = (output_text
                       .lstrip('(A)').lstrip('(B)').lstrip('(C)').lstrip('(D)')
                       .strip()
                       .lower())

    return pred_answer, reasoning


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _sample_stratified(annos: list, n: int, seed: int) -> list:
    """Return a stratified random subset of n annotations preserving label balance."""
    import random
    rng = random.Random(seed)
    by_label = {}
    for a in annos:
        label = a.get('answer', 'unknown')
        by_label.setdefault(label, []).append(a)
    labels = sorted(by_label)
    per_label = max(1, n // len(labels))
    sampled = []
    for label in labels:
        pool = by_label[label]
        rng.shuffle(pool)
        sampled.extend(pool[:per_label])
    # fill remainder if n is not evenly divisible
    remaining = n - len(sampled)
    if remaining > 0:
        all_remaining = [a for a in annos if a not in sampled]
        rng.shuffle(all_remaining)
        sampled.extend(all_remaining[:remaining])
    return sampled[:n]
