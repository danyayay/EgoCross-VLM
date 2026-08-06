#!/usr/bin/env python3
"""
Batch evaluate Qwen model on all auxiliary tasks.

Evaluates a Qwen model on all auxiliary task annotation files (vehicle_speed, vehicle_approach,
ped_moving, ped_direction, gaze_vehicle, ehmi, head_turning, vehicle_proximity, crossing_proximity).

Generates accuracy and F1 scores for each task, with an aggregated summary.

Usage:
    python -m training.eval_auxtasks \
        --model_name Qwen/Qwen3-VL-2B-Instruct \
        --split test \
        --tcot_type none
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import numpy as np

from models.vlm_adapters import MODEL_NAMES, get_adapter
from utils.analyze_groundvqa_results import analyze, pretty_print_report
from utils.eval_utils import (
    _FRAMES_REQUIRED_MODELS,
    _DEFAULT_TOTAL_PIXELS,
    _DEFAULT_MIN_PIXELS,
    _sample_stratified,
    build_prompt_from_annotation,
    infer_task_from_path,
    get_video_frames,
    _parse_model_output,
)
from utils.util import enable_strict_determinism, setup_logging


def run_evaluation(adapter, ann_file: str, task_name: str, args, ts: str) -> dict:
    """Evaluate model on a single annotation file."""
    annos = json.loads(Path(ann_file).read_text())
    
    if args.sample_n and args.sample_n < len(annos):
        annos = _sample_stratified(annos, args.sample_n, args.seed)
        logging.info("Using stratified subset of %d annotations (seed=%d)",
                     len(annos), args.seed)
    
    results = []
    inference_time = 0.0

    logging.info("Starting evaluation on task %s: %d samples", task_name, len(annos))

    for anno in tqdm(annos, desc=f'Evaluating {task_name}'):
        video_id = anno.get('video_id')
        video_path = os.path.join(args.video_root, video_id + '.mp4')
        if not os.path.exists(video_path):
            results.append({'video_id': video_id, 'error': 'video_not_found', 'video_path': video_path})
            continue

        # Pre-extract frames when needed
        frames, timestamps = None, None
        if not args.no_video:
            _, frames, timestamps = get_video_frames(
                video_path, num_frames=args.num_frames, cache_dir=args.cache_dir)

        prompt_text, context_text, question_text, options_map, correct_answer = build_prompt_from_annotation(
            anno, tcot_type=args.tcot_type, prompt_variant=args.prompt_variant,
            task_name=task_name, context_features='none',
            context_feature_fps='auto', timestamps=timestamps,
            num_frames=args.num_frames, video_duration=args.video_duration,
            context_feature_interpretation='none')

        try:
            messages = adapter.build_messages(
                video_path, prompt_text,
                context=context_text,
                question=question_text,
                frames=frames,
                timestamps=timestamps,
                interleaved=args.interleaved_timestamps,
                no_video=args.no_video,
                total_pixels=args.total_pixels,
                min_pixels=args.min_pixels,
                max_frames=args.max_frames,
                video_sample_fps=args.video_sample_fps,
            )
            output_text, meta = adapter.run_inference(
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            retry_count = 0
            while not meta['is_finished'] and retry_count <= 2:
                retry_count += 1
                logging.info("Truncated output for %s, retrying (attempt %d)", video_id, retry_count)
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
            'options': options_map,
            'tcot_type': args.tcot_type,
            'task_name': task_name,
            'answer': correct_answer,
            'pred_answer': pred_answer,
            'reasoning': reasoning,
            'output_text': output_text,
            'is_finished': meta['is_finished'],
            'duration': meta['duration'],
        })
        inference_time += meta['duration']

    logging.info("Inference time: %.2f seconds", inference_time)
    logging.info("Average inference time per sample: %.2f seconds", inference_time / len(annos) if annos else 0)

    output_filename = f'{ts}_{task_name}_responses.json'
    output_path = os.path.join(args.outdir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info("Saved %d results to %s", len(results), output_path)

    report = analyze(output_path)
    pretty_print_report(report)
    report_path = os.path.join(args.outdir, output_filename.split('.')[0] + '_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logging.info(
        "Task: %s | Samples: %d, Correct: %d/%d, Accuracy: %.4f, Macro F1: %.4f",
        task_name,
        report['num_samples'],
        report['classification']['correct'],
        report['classification']['total'],
        report['classification']['accuracy'],
        report['classification']['macro_f1'],
    )
    logging.info("Wrote report to %s", report_path)
    
    return report


def find_auxtask_annotation_files(feature_dir='features/groundvqa_auxtask', split='test'):
    """Find all auxiliary task annotation files for a given split."""
    feature_path = Path(feature_dir)
    pattern = f'annotations.VRbinary__aux_*__{split}_close.json'
    files = sorted(feature_path.glob(pattern))
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Batch evaluate VLM on all auxiliary tasks'
    )

    # --- Input / Output ---
    parser.add_argument('--feature_dir', default='features/groundvqa_auxtask',
                        help='Directory containing auxiliary task annotation files')
    parser.add_argument('--split', default='test',
                        choices=['train', 'val', 'test', 'full'],
                        help='Which split to evaluate on')
    parser.add_argument('--video_root', default='data/videodata_256/clips')
    parser.add_argument('--log_dir', default='logs/auxtask_eval')
    parser.add_argument('--cache_dir', default='.cache')

    # --- Model ---
    parser.add_argument('--model_name', default='Qwen/Qwen3-VL-2B-Instruct',
                        choices=MODEL_NAMES)
    parser.add_argument('--quantize', action='store_true',
                        help='Load local model with 8-bit quantization (bitsandbytes)')

    # --- Prompt ---
    parser.add_argument('--tcot_type', default='none',
                        choices=['none', 'tcot1', 'tcot2', 'tcot3', 'tcot4', 'tcot5', 
                                 'tcot6', 'tcot7', 'tcot8', 'tcot9', 'tcot10'],
                        help='Text-only chain-of-thought prompt variant')
    parser.add_argument('--prompt_variant', default='p6',
                        choices=['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'])
    parser.add_argument('--no_video', action='store_true',
                        help='Text-only baseline')

    # --- Frame sampling ---
    parser.add_argument('--num_frames', type=int, default=8,
                        help='Number of frames to sample per clip')
    parser.add_argument('--max_frames', type=int, default=2048,
                        help='Maximum number of frames to process (Qwen token limit)')
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

    # --- Subset sampling (for cost control) ---
    parser.add_argument('--sample_n', type=int, default=None,
                        help='Evaluate on a stratified random subset of N annotations')

    args = parser.parse_args(argv)

    # Build output directory path: <log_dir>/<model_short>/<run_tag>
    model_short = args.model_name.split('/')[-1]
    args.video_sample_fps = max(1, int(args.num_frames / args.video_duration))
    
    tcot_tag = str(args.tcot_type) if args.tcot_type else 'none'
    run_tag = f"tcot{tcot_tag}_f{args.num_frames}"
    interleaved_tag = 'interleaved' if args.interleaved_timestamps else 'nointerleave'
    run_tag += f"_{interleaved_tag}"
    if args.prompt_variant:
        run_tag += f'_{args.prompt_variant}'
    if args.no_video:
        run_tag += '_novideo'
    if args.quantize:
        run_tag += '_int8'

    if args.sample_n is not None and args.sample_n > 0:
        args.log_dir += f'_debugging'
    args.outdir = os.path.join(args.log_dir, model_short, run_tag)
    os.makedirs(args.outdir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.outdir, f'{ts}.log')
    setup_logging(log_file)
    logging.info("eval_auxtasks arguments: %s", vars(args))

    enable_strict_determinism(args.seed)

    # Find annotation files
    ann_files = find_auxtask_annotation_files(args.feature_dir, args.split)
    if not ann_files:
        logging.error(f"No annotation files found in {args.feature_dir} for split {args.split}")
        sys.exit(1)
    
    logging.info(f"Found {len(ann_files)} auxiliary task annotation files for split '{args.split}'")
    for f in ann_files:
        logging.info(f"  - {f.name}")

    adapter = get_adapter(args.model_name)
    adapter.load(args.model_name, quantize=args.quantize)

    try:
        # Evaluate on each auxiliary task
        all_reports = {}
        for ann_file in ann_files:
            try:
                task_name = infer_task_from_path(str(ann_file))
                report = run_evaluation(adapter, str(ann_file), task_name, args, ts)
                all_reports[task_name] = report
            except Exception as e:
                logging.exception(f"Error evaluating {ann_file}")
                continue
        
        # Write aggregate summary
        summary_path = os.path.join(args.outdir, f'{ts}_auxtask_summary.json')
        summary = {
            'timestamp': ts,
            'model': args.model_name,
            'tcot_type': args.tcot_type,
            'split': args.split,
            'tasks': all_reports,
        }
        
        # Calculate aggregate metrics
        if all_reports:
            accuracies = [r['classification']['accuracy'] for r in all_reports.values() if 'classification' in r]
            f1_scores = [r['classification']['macro_f1'] for r in all_reports.values() if 'classification' in r]
            
            summary['aggregate'] = {
                'num_tasks': len(all_reports),
                'mean_accuracy': np.mean(accuracies) if accuracies else 0.0,
                'std_accuracy': np.std(accuracies) if accuracies else 0.0,
                'mean_macro_f1': np.mean(f1_scores) if f1_scores else 0.0,
                'std_macro_f1': np.std(f1_scores) if f1_scores else 0.0,
            }
            
            logging.info("="*80)
            logging.info("AUXILIARY TASK EVALUATION SUMMARY")
            logging.info("="*80)
            logging.info(f"Model: {args.model_name}")
            logging.info(f"TCOT Type: {args.tcot_type}")
            logging.info(f"Split: {args.split}")
            logging.info(f"Number of Tasks: {summary['aggregate']['num_tasks']}")
            logging.info(f"Mean Accuracy: {summary['aggregate']['mean_accuracy']:.4f} ± {summary['aggregate']['std_accuracy']:.4f}")
            logging.info(f"Mean Macro F1: {summary['aggregate']['mean_macro_f1']:.4f} ± {summary['aggregate']['std_macro_f1']:.4f}")
            logging.info("="*80)
            logging.info("Per-Task Results:")
            for task_name, report in sorted(all_reports.items()):
                if 'classification' in report:
                    c = report['classification']
                    logging.info(f"  {task_name:25s}: Acc={c['accuracy']:.4f}, F1={c['macro_f1']:.4f}")
                    # Show label distribution
                    dist = {cls: c['per_class'][cls]['support'] for cls in c['classes']}
                    logging.info(f"    Label distribution: {dist}")
            logging.info("="*80)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logging.info(f"Wrote summary to {summary_path}")
    finally:
        adapter.unload()


if __name__ == '__main__':
    main()
