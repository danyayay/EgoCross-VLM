"""Unified zero-shot VLM evaluation runner.

Supports multiple model families (Qwen3-VL, Qwen2.5-VL, InternVL3, Gemini, GPT-4o)
via a common adapter interface. Outputs the same JSON schema as eval_qwen.py.

Usage example (frame ablation):
    python -m training.eval_vlm \
        --model_name Qwen/Qwen3-VL-2B-Instruct \
        --ann_file features/groundvqa_qn3/annotations.VRbinary_00000_test_close.json \
        --num_frames 16 \
        --tcot_type none

Usage example (interleaved timestamps):
    python -m training.eval_vlm \
        --model_name Qwen/Qwen3-VL-2B-Instruct \
        --ann_file features/groundvqa_qn3/annotations.VRbinary_00000_test_close.json \
        --num_frames 16 --interleaved_timestamps

Usage example (API model subset):
    OPENAI_API_KEY=... python -m training.eval_vlm \
        --model_name gpt-4o \
        --ann_file features/groundvqa_qn3/annotations.VRbinary_00000_test_close.json \
        --num_frames 16 --sample_n 10
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import numpy as np

from models.vlm_adapters import MODEL_NAMES, get_adapter
from utils.eval_utils import (
    _TCOT_PROMPTS, get_video_frames, build_prompt_from_annotation, _parse_model_output,
    _PROMPT_VARIANTS, _FRAMES_REQUIRED_MODELS,
    _DEFAULT_TOTAL_PIXELS, _DEFAULT_MIN_PIXELS,
    _sample_stratified, normalize_tcot_type, infer_task_from_path,
    build_interleaved_context_texts, build_interleaved_context_preface, parse_context_features, build_context_feature_interpretation,
)
from utils.visual_cot import prepare_visual_cot, resolve_vcot_video_path, resolve_vcot_video_root, uses_frame_vcot
from utils.analyze_groundvqa_results import analyze, pretty_print_report
from utils.util import enable_strict_determinism, setup_logging
from utils.parser.extract_answers import extract_answer_binary


_VISUAL_ABLATIONS = ('original', 'black', 'noise', 'mismatched', 'shuffle', 'reverse')


def _stable_sample_seed(base_seed: int, video_id: str) -> int:
    """Derive a process-independent uint32 seed for one evaluation sample."""
    import hashlib
    digest = hashlib.sha256(f'{base_seed}:{video_id}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], byteorder='little', signed=False)


def apply_visual_ablation(frames, *, condition: str, seed: int, video_id: str,
                          donor_frames=None):
    """Return perturbed frames without modifying the cached source array."""
    if condition == 'original':
        return frames, None
    if frames is None:
        raise ValueError(f'visual ablation {condition!r} requires sampled frames')

    rng = np.random.default_rng(_stable_sample_seed(seed, video_id))
    if condition == 'black':
        return np.zeros_like(frames), None
    if condition == 'noise':
        return rng.integers(0, 256, size=frames.shape, dtype=np.uint8), None
    if condition == 'reverse':
        return frames[::-1].copy(), list(range(len(frames) - 1, -1, -1))
    if condition == 'shuffle':
        order = rng.permutation(len(frames))
        if len(order) > 1 and np.array_equal(order, np.arange(len(order))):
            order = np.roll(order, 1)
        return frames[order].copy(), order.tolist()
    if condition == 'mismatched':
        if donor_frames is None:
            raise ValueError('mismatched ablation requires donor_frames')
        if donor_frames.shape != frames.shape:
            raise ValueError(
                f'donor/source frame shapes differ: {donor_frames.shape} vs {frames.shape}')
        return donor_frames.copy(), None
    raise ValueError(f'unknown visual ablation: {condition}')



def _timestamp_value(ts) -> float:
    arr = np.asarray(ts)
    return float(arr.mean()) if arr.size else 0.0


def describe_model_input(
    *,
    video_path: str,
    context_text: str,
    question_text: str,
    prompt_text: str,
    args,
    frames=None,
    timestamps=None,
    interleaved: bool = False,
    no_video: bool = False,
    frame_contexts=None,
    visual_meta: dict | None = None,
) -> dict:
    """Create a JSON-safe description of the multimodal input without storing images."""
    visual_meta = visual_meta or {}
    frame_contexts = frame_contexts or []
    if no_video:
        sequence = [{"type": "text", "role": "prompt", "text": prompt_text}]
        visual_input = {
            "mode": "text_only",
            "video_path": None,
            "num_frames_requested": 0,
            "video_sample_fps": None,
            "interleaved": False,
        }
    elif interleaved and frames is not None and timestamps is not None:
        frame_items = []
        for i, ts in enumerate(timestamps):
            frame_items.append({
                "index": i,
                "timestamp_sec": round(_timestamp_value(ts), 4),
                "placeholder": f"<image frame {i}>",
            })
        sequence = []
        if context_text:
            sequence.append({"type": "text", "role": "context", "text": context_text})
        for item in frame_items:
            ts_text = f"[t={item['timestamp_sec']:.2f}s]"
            frame_context = frame_contexts[item["index"]] if item["index"] < len(frame_contexts) else ""
            if frame_context:
                ts_text = ts_text + "\n" + frame_context
            sequence.append({"type": "text", "role": "timestamp", "text": ts_text})
            sequence.append({"type": "image", "role": "frame", **item})
        sequence.append({"type": "text", "role": "question", "text": question_text or prompt_text})
        visual_input = {
            "mode": "interleaved_frames",
            "video_path": video_path,
            "input_source": visual_meta.get("vcot_input_source", "sampled_frames"),
            "num_frames_requested": args.num_frames,
            "num_frames_actual": len(frame_items),
            "video_duration_sec": args.video_duration,
            "video_sample_fps": args.video_sample_fps,
            "interleaved": True,
            "frame_placeholders": frame_items,
        }
    else:
        text_content = (context_text + " " + (question_text or prompt_text)).strip() if context_text else (question_text or prompt_text)
        sequence = [
            {
                "type": "video",
                "role": "video",
                "placeholder": "<video clip>",
                "video_path": video_path,
                "video_sample_fps": args.video_sample_fps,
                "max_frames": args.max_frames,
            },
            {"type": "text", "role": "prompt", "text": text_content},
        ]
        visual_input = {
            "mode": "video_file",
            "video_path": video_path,
            "input_source": visual_meta.get("vcot_input_source", "raw_video"),
            "num_frames_requested": args.num_frames,
            "max_frames": args.max_frames,
            "video_duration_sec": args.video_duration,
            "video_sample_fps": args.video_sample_fps,
            "interleaved": False,
            "placeholder": "<video clip>",
        }

    return {
        "visual_input": visual_input,
        "text_input": {
            "context": context_text,
            "question": question_text,
            "prompt_text_flat": prompt_text,
        },
        "sequence": sequence,
    }


def run_evaluation(adapter, ann_file: str, args, ts: str) -> None:
    """Main evaluation loop. Mirrors eval_qwen.py's evaluate_on_annotations()."""
    annos = json.loads(Path(ann_file).read_text())
    task_name = infer_task_from_path(ann_file)

    if args.sample_n and args.sample_n < len(annos):
        annos = _sample_stratified(annos, args.sample_n, args.sample_seed)
        logging.info("Using stratified subset of %d annotations (seed=%d)",
                     len(annos), args.sample_seed)

    results = []
    inference_time = 0.0
    ablation = args.visual_ablation
    needs_frames = (not args.no_video
                    and (ablation != 'original'
                         or args.model_name in _FRAMES_REQUIRED_MODELS
                         or args.interleaved_timestamps
                         or (args.context_prompt_mode == 'interleaved' and bool(parse_context_features(args.context_features)))
                         or uses_frame_vcot(args.vcot_visual, args.vcot_text)))

    logging.info("Starting evaluation: %d samples, needs_frames=%s, no_video=%s",
                 len(annos), needs_frames, getattr(args, 'no_video', False))

    donor_by_video_id = {}
    if ablation == 'mismatched':
        if len(annos) < 2:
            raise ValueError('mismatched ablation requires at least two annotations')
        donor_ids = list(dict.fromkeys(a.get('video_id') for a in annos))
        if len(donor_ids) < 2:
            raise ValueError('mismatched ablation requires at least two distinct videos')
        donor_rng = np.random.default_rng(args.ablation_seed)
        donor_rng.shuffle(donor_ids)
        offset = int(donor_rng.integers(1, len(donor_ids)))
        donor_by_video_id = {
            donor_ids[i]: donor_ids[(i + offset) % len(donor_ids)]
            for i in range(len(donor_ids))
        }

    for anno in tqdm(annos, desc='Evaluating'):
        video_id = anno.get('video_id')
        video_path = os.path.join(args.video_root, video_id + '.mp4')
        if not os.path.exists(video_path):
            results.append({'video_id': video_id, 'error': 'video_not_found', 'video_path': video_path})
            continue

        # Pre-extract frames when needed. Structured text features can align to these timestamps.
        frames, timestamps = None, None
        if needs_frames:
            _, frames, timestamps = get_video_frames(
                video_path, num_frames=args.num_frames, cache_dir=args.cache_dir)
            donor_video_id = donor_by_video_id.get(video_id)
            donor_frames = None
            if donor_video_id is not None:
                donor_path = os.path.join(args.video_root, donor_video_id + '.mp4')
                if not os.path.exists(donor_path):
                    results.append({
                        'video_id': video_id, 'error': 'donor_video_not_found',
                        'donor_video_id': donor_video_id, 'video_path': donor_path})
                    continue
                _, donor_frames, _ = get_video_frames(
                    donor_path, num_frames=args.num_frames, cache_dir=args.cache_dir)
            frames, frame_order = apply_visual_ablation(
                frames, condition=ablation, seed=args.ablation_seed,
                video_id=video_id, donor_frames=donor_frames)
        else:
            donor_video_id, frame_order = None, None

        context_features_for_preface = args.context_features
        frame_contexts = None
        if args.context_prompt_mode == 'interleaved' and timestamps is not None:
            selected = parse_context_features(args.context_features)
            non_demo = [f for f in selected if f != 'demographics']
            demo = [f for f in selected if f == 'demographics']
            context_features_for_preface = ','.join(demo) if demo else 'none'
            frame_contexts = build_interleaved_context_texts(anno, timestamps, non_demo, context_feature_format=args.context_feature_format)

        prompt_interpretation = 'none' if args.context_prompt_mode == 'interleaved' else args.context_feature_interpretation
        prompt_text, context_text, question_text, options_map, correct_answer = build_prompt_from_annotation(
            anno, tcot_type=args.tcot_type, prompt_variant=args.prompt_variant,
            task_name=task_name, context_features=context_features_for_preface,
            context_feature_fps=args.context_feature_fps, timestamps=timestamps,
            num_frames=args.num_frames, video_duration=args.video_duration,
            context_feature_interpretation=prompt_interpretation,
            context_feature_format=args.context_feature_format)

        if args.context_prompt_mode == 'interleaved':
            interleaved_preface = build_interleaved_context_preface(
                anno, timestamps, args.context_features, context_feature_format=args.context_feature_format)
            if interleaved_preface and interleaved_preface not in context_text:
                context_text = (context_text + "\n" + interleaved_preface).strip() if context_text else interleaved_preface
                prompt_text = (context_text + ' ' + question_text).strip()

        if args.context_prompt_mode == 'interleaved' and args.context_feature_interpretation != 'none':
            interp_text = build_context_feature_interpretation(
                args.context_features, args.context_feature_interpretation,
                context_feature_format=args.context_feature_format)
            if interp_text and interp_text not in context_text:
                context_text = (context_text + "\n" + interp_text).strip() if context_text else interp_text
                prompt_text = (context_text + ' ' + question_text).strip()

        try:
            visual_meta = {
                'vcot_visual': args.vcot_visual,
                'vcot_text': args.vcot_text,
                'vcot_labels': [],
                'vcot_detection_cache': None,
                'vcot_detection_cache_hit': None,
                'vcot_rendered_video': None,
            }
            interleaved = args.interleaved_timestamps
            if not args.no_video and (args.vcot_visual != 'none' or args.vcot_text != 'none'):
                vcot = prepare_visual_cot(anno, video_path, frames, timestamps, args)
                video_path = vcot['video_path']
                frames = vcot['frames']
                timestamps = vcot['timestamps']
                visual_meta = vcot['metadata']
                interleaved = interleaved or vcot['force_interleaved']
                if vcot['context_addition']:
                    context_text = (context_text + "\n" + vcot['context_addition']).strip() if context_text else vcot['context_addition']
                    prompt_text = (context_text + ' ' + question_text).strip()

            input_description = describe_model_input(
                video_path=video_path,
                context_text=context_text,
                question_text=question_text,
                prompt_text=prompt_text,
                args=args,
                frames=frames,
                timestamps=timestamps,
                interleaved=interleaved,
                no_video=args.no_video,
                frame_contexts=frame_contexts,
                visual_meta=visual_meta,
            )

            messages = adapter.build_messages(
                video_path, prompt_text,
                context=context_text,
                question=question_text,
                frames=frames,
                timestamps=timestamps,
                interleaved=interleaved,
                no_video=args.no_video,
                total_pixels=args.total_pixels,
                min_pixels=args.min_pixels,
                max_frames=args.max_frames,
                video_sample_fps=args.video_sample_fps,  # clips are 2s; N frames -> N/2 fps
                frame_contexts=frame_contexts,
            )
            output_text, meta = adapter.run_inference(
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            retry_count = 0

            while not meta['is_finished'] and retry_count <= 1:
                retry_count += 1
                logging.info("Truncated output for %s, retrying (attempt %d)",
                             video_id, retry_count)
                output_text, meta = adapter.run_inference(
                    messages,
                    max_new_tokens=args.max_new_tokens * retry_count,
                    temperature=0.1 * retry_count,
                )
        except Exception:
            logging.exception("Runtime error for video_id=%s", video_id)
            results.append({'video_id': video_id, 'error': 'runtime_error'})
            continue

        # Primary parse only — no fallback here so the GPU is free for inference
        pred_answer, reasoning = _parse_model_output(output_text, options_map, tcot_type=args.tcot_type)
        if not meta['is_finished'] or pred_answer is None:
            pred_answer = 'unknown'

        results.append({
            'video_id': video_id,
            'context': context_text,
            'question': question_text,
            'prompt': input_description['sequence'],
            'options': options_map,
            'tcot_type': args.tcot_type,
            'task_name': task_name,
            'context_features': args.context_features,
            'context_feature_fps': args.context_feature_fps,
            'context_prompt_mode': args.context_prompt_mode,
            'context_feature_interpretation': args.context_feature_interpretation,
            'context_feature_format': args.context_feature_format,
            'visual_ablation': ablation,
            'ablation_seed': args.ablation_seed,
            'donor_video_id': donor_video_id,
            'frame_order': frame_order,
            **visual_meta,
            'answer': correct_answer,
            'pred_answer': pred_answer,
            'reasoning': reasoning,
            'output_text': output_text,
            'is_finished': meta['is_finished'],
            'duration': meta['duration'],
            'token_count': meta['token_count'],
            'tokens_per_sec': meta['tokens_per_sec'],
            'temperature': meta['temperature'],
            'max_new_tokens': meta['max_new_tokens'],
        })
        inference_time += meta['duration']

        n_done = len(results)
        if n_done % 100 == 0:
            avg = inference_time / n_done
            logging.info("Progress: %d/%d samples | avg %.2f s/sample | elapsed %.1f s",
                         n_done, len(annos), avg, inference_time)

    n = len(annos) if annos else 1
    logging.info("Total inference time: %.2f s (avg %.2f s/sample)",
                 inference_time, inference_time / n)

    # ── Post-processing: regex fallback after model is done with GPU ─────────
    # Unload happens in main() after this function returns, but the loop is
    # finished so no inference overlap. Extract answers for any result where
    # primary parsing didn't produce a valid label.
    fallback_count = 0
    for r in results:
        if 'error' in r or not r.get('is_finished'):
            continue
        valid_labels = [v.lower() for v in r['options'].values()]
        if r['pred_answer'] not in valid_labels:
            r['pred_answer'] = extract_answer_binary(r['output_text'], valid_labels)
            if r['reasoning'] is None:
                r['reasoning'] = r['output_text']
            fallback_count += 1
    if fallback_count:
        logging.info("Regex fallback used for %d/%d samples", fallback_count, len(results))

    out_file = os.path.join(args.outdir, f'{ts}_responses.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info("Saved %d results to %s", len(results), out_file)

    report = analyze(out_file)
    logger = logging.getLogger()
    pretty_print_report(report, logger=logger)
    report_file = out_file.replace('_responses.json', '_responses_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    logging.info("Report written to %s", report_file)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Zero-shot VLM evaluation (multi-model)'
    )

    # --- Input / Output ---
    parser.add_argument('--ann_file',
                        default='features/groundvqa/annotations.VRbinary__crossing_intention__test_close.json')
    parser.add_argument('--video_root', default='data/videodata_256/clips')
    parser.add_argument('--log_dir', default='logs/debugging')
    parser.add_argument('--cache_dir', default='.cache')

    # --- Model ---
    parser.add_argument('--model_name', default='Qwen/Qwen3-VL-2B-Instruct',
                        choices=MODEL_NAMES)
    parser.add_argument('--quantize', action='store_true',
                        help='Load local model with 8-bit quantization (bitsandbytes)')
    parser.add_argument('--lora_adapter', default=None,
                        help='Optional PEFT LoRA adapter for fine-tuned-model evaluation')

    # --- Prompt ---
    parser.add_argument('--tcot_type', default='none',
                        choices=['none', 'tcot1', 'tcot2', 'tcot3', 'tcot4', 'tcot5', 
                                 'tcot6', 'tcot7', 'tcot8', 'tcot9', 'tcot10'],
                        help='Text-only chain-of-thought prompt variant')
    parser.add_argument('--cot_type', default=None,
                        choices=['none', 'cot1', 'cot2', 'cot3', 'cot4',
                                 'cot5', 'cot6', 'cot7', 'cot8', 'cot9'],
                        help='Deprecated alias for --tcot_type')
    parser.add_argument('--prompt_variant', default='p6',
                        choices=['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'])
    parser.add_argument('--context_features', default='none',
                        help='Comma-separated context features: none,ego_motion,vehicle_motion,gaze_direction,gaze_direction_change,gaze_on_screen,gaze_on_screen_ratio,gaze_target,pose,demographics,all')
    parser.add_argument('--no_video', action='store_true',
                        help='Text-only baseline (Qwen-family only)')
    parser.add_argument('--visual_ablation', default='original', choices=_VISUAL_ABLATIONS,
                        help='Counterfactual visual input condition; altered visuals require interleaved frames')
    parser.add_argument('--ablation_seed', type=int, default=42,
                        help='Seed for noise, mismatch assignment, or frame shuffling')
    parser.add_argument('--vcot_visual', default='none',
                        choices=['none', 'gaze_dot', 'gaze_rainbow', 'gaze_bone',
                                 'bbox_overlay', 'som_overlay_nobg', 'som_overlay_bg'],
                        help='Visual CoT augmentation rendered into the visual input')
    parser.add_argument('--vcot_text', default='none', choices=['none', 'bbox_coords'],
                        help='Visual CoT text added to the prompt')

    # Keep dependent argument groups out of --help unless their parent feature is active.
    prelim = argparse.ArgumentParser(add_help=False)
    prelim.add_argument('--context_features', default='none')
    prelim.add_argument('--vcot_visual', default='none')
    prelim.add_argument('--vcot_text', default='none')
    known_args, _ = prelim.parse_known_args(argv)

    if parse_context_features(known_args.context_features):
        parser.add_argument('--context_feature_fps', default='auto',
                            help="Text feature sampling rate, e.g. auto, 1, 2, 4.")
        parser.add_argument('--context_prompt_mode', default=None, choices=['preface', 'interleaved'],
                            help='Place feature text before visual input or next to interleaved frames. Default follows --interleaved_timestamps.')
        parser.add_argument('--context_feature_interpretation', default='none', choices=['none', 'brief', 'detailed'],
                            help='Add an interpretation/legend for selected structured context features. detailed also explains how to read values.')
        parser.add_argument('--context_feature_format', default='detailed',
                            choices=['detailed', 'legacy', 'compact', 'schema', 'summary'],
                            help='Text serialization for structured context features.')

    vcot_active = known_args.vcot_visual != 'none' or known_args.vcot_text != 'none'
    bbox_vcot_active = known_args.vcot_text == 'bbox_coords' or known_args.vcot_visual in {'bbox_overlay', 'som_overlay_nobg', 'som_overlay_bg'}
    gaze_vcot_active = known_args.vcot_visual in {'gaze_dot', 'gaze_rainbow', 'gaze_bone'}
    if vcot_active:
        parser.add_argument('--vcot_labels', default='automated vehicle,white circle')
        parser.add_argument('--vcot_cache_dir', default='.cache/visual_cot')
        parser.add_argument('--vcot_render_sampled_video', action='store_true',
                            help='Also encode sampling-specific rendered videos for offline inspection')
    if bbox_vcot_active:
        parser.add_argument('--vcot_detector', default='groundingdinotiny',
                            choices=['groundingdinotiny', 'groundingdinobase'])
        parser.add_argument('--vcot_online_if_missing', action='store_true',
                            help='Run detector when a bbox/SoM cache is missing')
    if gaze_vcot_active:
        parser.add_argument('--vrdata_dir', default='data/vrdata')

    # --- Frame sampling ---
    # --num_frames is the single control for frame count.
    # For Qwen's internal decoder (non-interleaved), sample_fps is derived as num_frames/clip_duration.
    # Clips in this dataset are always 2s, so sample_fps = num_frames / 2.0.
    parser.add_argument('--num_frames', type=int, default=8,
                        help='Number of frames to sample per clip')
    parser.add_argument('--max_frames', type=int, default=2048,
                        help='Maximum number of frames to process (Qwen token limit, not a frame-count control)')
    parser.add_argument('--video_duration', type=float, default=2.0,
                        help='Duration of each video clip in seconds (for fps calculation)')
    parser.add_argument('--interleaved_timestamps', action='store_true',
                        help='Interleave frame images with [t=Xs] timestamp text')
    parser.add_argument('--total_pixels', type=int, default=_DEFAULT_TOTAL_PIXELS,
                        help='Total pixel budget for Qwen-family models')
    parser.add_argument('--min_pixels', type=int, default=_DEFAULT_MIN_PIXELS)

    # --- Generation ---
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)

    # --- Subset sampling (for API cost control) ---
    parser.add_argument('--sample_n', type=int, default=None,
                        help='Evaluate on a stratified random subset of N annotations')
    parser.add_argument('--sample_seed', type=int, default=42,
                        help='Random seed for stratified subset sampling')

    args = parser.parse_args(argv)
    if args.no_video and args.visual_ablation != 'original':
        parser.error('--no_video cannot be combined with --visual_ablation')
    if args.visual_ablation != 'original' and not args.interleaved_timestamps:
        parser.error('non-original --visual_ablation requires --interleaved_timestamps')
    if args.visual_ablation != 'original' and (args.vcot_visual != 'none' or args.vcot_text != 'none'):
        parser.error('--visual_ablation cannot be combined with visual CoT inputs')
    if args.cot_type is not None:
        args.tcot_type = normalize_tcot_type(args.cot_type)
    args.tcot_type = normalize_tcot_type(args.tcot_type)
    selected_context_features = parse_context_features(args.context_features)
    if not hasattr(args, 'context_feature_fps'):
        args.context_feature_fps = 'auto'
    if not hasattr(args, 'context_prompt_mode'):
        args.context_prompt_mode = 'interleaved' if (args.interleaved_timestamps and selected_context_features) else 'preface'
    elif args.context_prompt_mode is None:
        args.context_prompt_mode = 'interleaved' if args.interleaved_timestamps else 'preface'
    if not hasattr(args, 'context_feature_interpretation'):
        args.context_feature_interpretation = 'none'
    if not hasattr(args, 'context_feature_format'):
        args.context_feature_format = 'detailed'
    if not hasattr(args, 'vcot_labels'):
        args.vcot_labels = 'automated vehicle,white circle'
    if not hasattr(args, 'vcot_cache_dir'):
        args.vcot_cache_dir = '.cache/visual_cot'
    if not hasattr(args, 'vcot_render_sampled_video'):
        args.vcot_render_sampled_video = False
    if not hasattr(args, 'vcot_detector'):
        args.vcot_detector = 'groundingdinotiny'
    if not hasattr(args, 'vcot_online_if_missing'):
        args.vcot_online_if_missing = False
    if not hasattr(args, 'vrdata_dir'):
        args.vrdata_dir = 'data/vrdata'

    # Build output directory path: <log_dir>/<model_short>/<run_tag>
    model_short = args.model_name.split('/')[-1]
    args.video_sample_fps = max(1, int(args.num_frames / args.video_duration))
    args.raw_video_root = args.video_root
    if not args.no_video and args.vcot_visual in {
        'gaze_dot', 'gaze_rainbow', 'gaze_bone', 
        'bbox_overlay', 'som_overlay_bg', 'som_overlay_nobg'}:
        args.video_root = resolve_vcot_video_root(args.video_root, args)
    tcot_tag = str(args.tcot_type) if args.tcot_type else 'none'
    interleaved_tag = 'interleaved' if args.interleaved_timestamps else 'nointerleave'
    run_tag = f"tcot{tcot_tag}_f{args.num_frames}_{interleaved_tag}"
    if args.vcot_visual != 'none' or args.vcot_text != 'none':
        run_tag += f"_vcot{args.vcot_visual}+{args.vcot_text}"
    if args.prompt_variant:
        run_tag += f'_{args.prompt_variant}'
    if args.context_features != 'none':
        cf_tag = args.context_features.replace(',', '+')
        run_tag += f'_ctx{cf_tag}_cfps{args.context_feature_fps}_{args.context_prompt_mode}_{args.context_feature_format}'
        if args.context_feature_interpretation != 'none':
            run_tag += f'_interp{args.context_feature_interpretation}'
    if args.no_video:
        run_tag += '_novideo'
    if args.visual_ablation != 'original':
        run_tag += f'_abl-{args.visual_ablation}-s{args.ablation_seed}'
    if args.lora_adapter:
        run_tag += '_finetuned'
    if args.quantize:
        run_tag += '_int8'

    if args.sample_n is not None:
        args.log_dir += f'_debugging'
    args.outdir = os.path.join(args.log_dir, model_short, f'seed_{args.seed}', run_tag)
    
    os.makedirs(args.outdir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.outdir, f'{ts}.log')
    setup_logging(log_file)
    logging.info("eval_vlm arguments: %s", vars(args))
    if args.cot_type is not None:
        logging.warning("--cot_type is deprecated; use --tcot_type instead")

    enable_strict_determinism(args.seed)

    adapter = get_adapter(args.model_name)
    adapter.load(args.model_name, quantize=args.quantize)
    if args.lora_adapter:
        if not args.model_name.startswith('Qwen/'):
            parser.error('--lora_adapter is currently supported only for local Qwen models')
        from peft import PeftModel
        logging.info('Loading LoRA adapter from %s', args.lora_adapter)
        adapter.model = PeftModel.from_pretrained(adapter.model, args.lora_adapter)
        adapter.model.eval()

    try:
        run_evaluation(adapter, args.ann_file, args, ts)
    finally:
        adapter.unload()


if __name__ == '__main__':
    main()
