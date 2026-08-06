"""
Simple LoRA fine-tuning script for VLMs using the Adapter pattern.

Supports multiple --ft_type options controlling which modules receive LoRA adapters.
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import json
import logging
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.vlm_adapters import MODEL_REGISTRY, get_adapter
from utils.analyze_groundvqa_results import classification_metrics
from utils.util import cleanup_intermediate_checkpoints, enable_strict_determinism, setup_logging
from utils.eval_utils import (
    build_context_feature_interpretation,
    build_interleaved_context_preface,
    build_interleaved_context_texts,
    build_prompt_from_annotation,
    get_video_frames,
    parse_context_features,
    _parse_model_output,
    _sample_stratified,
)

import random


TRAINABLE_MODEL_NAMES = [
    name for name in MODEL_REGISTRY
    if name not in {'gemini-2.0-flash', 'gpt-4o'}
]


def _timestamp_value(ts) -> float:
    try:
        return float(ts.mean())
    except Exception:
        try:
            values = list(ts)
            return float(sum(values) / len(values)) if values else 0.0
        except Exception:
            return float(ts)


def describe_model_input(video_path, context_text, question_text, prompt_text, args,
                         frames=None, timestamps=None, interleaved=False,
                         frame_contexts=None):
    """Create a JSON-safe description of the multimodal input saved with results."""
    frame_contexts = frame_contexts or []
    if interleaved and frames is not None and timestamps is not None:
        frame_items = []
        for i, ts in enumerate(timestamps):
            frame_items.append({
                'type': 'image',
                'role': 'frame',
                'index': i,
                'timestamp_sec': round(_timestamp_value(ts), 4),
                'placeholder': f'<image frame {i}>',
            })
        sequence = []
        if context_text:
            sequence.append({'type': 'text', 'role': 'context', 'text': context_text})
        for item in frame_items:
            ts_text = f"[t={item['timestamp_sec']:.2f}s]"
            frame_context = frame_contexts[item['index']] if item['index'] < len(frame_contexts) else ''
            if frame_context:
                ts_text = ts_text + "\n" + frame_context
            sequence.append({'type': 'text', 'role': 'timestamp', 'text': ts_text})
            sequence.append(item)
        sequence.append({'type': 'text', 'role': 'question', 'text': question_text})
        visual_input = {
            'mode': 'sampled_frames',
            'video_path': video_path,
            'input_source': 'sampled_frames',
            'num_frames_requested': args.num_frames,
            'num_frames_actual': len(frame_items),
            'timestamps_sec': [item['timestamp_sec'] for item in frame_items],
            'max_frames': args.max_frames,
            'video_duration_sec': args.video_duration,
            'video_sample_fps': args.video_sample_fps,
            'interleaved': True,
        }
    else:
        sequence = [
            {'type': 'video', 'video_path': video_path, 'placeholder': '<video clip>'},
            {'type': 'text', 'role': 'prompt', 'text': prompt_text},
        ]
        visual_input = {
            'mode': 'video_file',
            'video_path': video_path,
            'input_source': 'raw_video',
            'num_frames_requested': args.num_frames,
            'max_frames': args.max_frames,
            'video_duration_sec': args.video_duration,
            'video_sample_fps': args.video_sample_fps,
            'interleaved': False,
            'placeholder': '<video clip>',
        }

    return {
        'visual_input': visual_input,
        'text_input': {
            'context': context_text,
            'question': question_text,
            'prompt_text_flat': prompt_text,
        },
        'sequence': sequence,
    }


def build_context_prompt_for_training(anno, args, timestamps=None):
    context_features_for_preface = args.context_features
    frame_contexts = None
    if args.context_prompt_mode == 'interleaved' and timestamps is not None:
        selected = parse_context_features(args.context_features)
        non_demo = [f for f in selected if f != 'demographics']
        demo = [f for f in selected if f == 'demographics']
        context_features_for_preface = ','.join(demo) if demo else 'none'
        frame_contexts = build_interleaved_context_texts(
            anno, timestamps, non_demo,
            context_feature_format=args.context_feature_format)

    prompt_interpretation = 'none' if args.context_prompt_mode == 'interleaved' else args.context_feature_interpretation
    prompt_text, context_text, question_text, options_map, correct_answer = build_prompt_from_annotation(
        anno, cot_type=args.cot_type, prompt_variant=args.prompt_variant,
        task_name=args.task_type, context_features=context_features_for_preface,
        context_feature_fps=args.context_feature_fps, timestamps=timestamps,
        num_frames=args.num_frames, video_duration=args.video_duration,
        context_feature_interpretation=prompt_interpretation,
        context_feature_format=args.context_feature_format)

    if args.context_prompt_mode == 'interleaved':
        interleaved_preface = build_interleaved_context_preface(
            anno, timestamps, args.context_features,
            context_feature_format=args.context_feature_format)
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

    return prompt_text, context_text, question_text, options_map, correct_answer, frame_contexts


def parameter_counts(model):
    """Log total, trainable, and non-trainable parameter counts."""
    total = trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    logging.info(
        "Model parameters: total=%d, trainable=%d, non_trainable=%d",
        total, trainable, total - trainable)
    return total, trainable, total - trainable


def is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or 'cuda out of memory' in msg


class VideoQADataset(Dataset):
    """Dataset that holds annotation dicts and returns them for collating."""
    def __init__(self, annos, video_root):
        self.annos = annos
        self.video_root = video_root

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        return self.annos[idx]


TASK_VOCAB = {
    'crossing_intention': {'cross', 'yield'},
    'vehicle_speed':      {'moving', 'static'},
    'vehicle_approach':   {'moving closer to me', 'moving away from me'},
}


def evaluate(adapter, ann_file, args, mode='test', save_dir=None):
    valid_answers = TASK_VOCAB.get(args.task_type, set())
    y_true, y_pred = [], []
    results = []
    start_time = time.time()
    for anno in tqdm(ann_file, desc='evaluating'):
        video_path = os.path.join(args.video_root, f"{anno.get('video_id')}.mp4")
        frames, timestamps = None, None
        needs_frames = (args.interleaved_timestamps
                        or hasattr(adapter, 'num_image_token')
                        or (args.context_prompt_mode == 'interleaved'
                            and bool(parse_context_features(args.context_features))))
        if needs_frames:
            _, frames, timestamps = get_video_frames(video_path, num_frames=args.num_frames, cache_dir=args.cache_dir)

        prompt_text, context_text, question_text, options_map, correct_answer, frame_contexts = build_context_prompt_for_training(
            anno, args, timestamps=timestamps)

        input_description = describe_model_input(
            video_path, context_text, question_text, prompt_text, args,
            frames=frames, timestamps=timestamps,
            interleaved=args.interleaved_timestamps,
            frame_contexts=frame_contexts)

        messages = adapter.build_messages(
            video_path, prompt_text,
            context=context_text, 
            question=question_text,
            frames=frames, 
            timestamps=timestamps, 
            interleaved=args.interleaved_timestamps,
            total_pixels=args.total_pixels, 
            min_pixels=args.min_pixels,
            max_frames=args.max_frames, 
            video_sample_fps=args.video_sample_fps,
            frame_contexts=frame_contexts
        )
        try:
            out, meta = adapter.run_inference(
                messages, max_new_tokens=args.eval_max_new_tokens,
                temperature=0.0 if args.eval_deterministic else 0.7)
            raw_out = out
            pred_answer, reasoning = _parse_model_output(
                raw_out, options_map, cot_type=args.cot_type)
            out = (pred_answer or 'others').strip().lower()
            if out == 'ross':
                out = 'cross'
            if valid_answers and out not in valid_answers:
                out = 'others'
            y_pred.append(out)
            results.append({
                'video_id': anno.get('video_id'),
                'context': context_text,
                'question': question_text,
                'prompt': input_description['sequence'],
                'options': options_map,
                'cot_type': args.cot_type,
                'prompt_variant': args.prompt_variant,
                'task_name': args.task_type,
                'context_features': args.context_features,
                'context_feature_fps': args.context_feature_fps,
                'context_prompt_mode': args.context_prompt_mode,
                'context_feature_interpretation': args.context_feature_interpretation,
                'context_feature_format': args.context_feature_format,
                'visual_input': input_description['visual_input'],
                'text_input': input_description['text_input'],
                'answer': correct_answer,
                'pred_answer': out,
                'reasoning': reasoning,
                'output_text': raw_out,
                'is_finished': meta.get('is_finished', True),
                'duration': meta.get('duration'),
                'token_count': meta.get('token_count'),
                'tokens_per_sec': meta.get('tokens_per_sec'),
                'temperature': meta.get('temperature'),
                'max_new_tokens': meta.get('max_new_tokens'),
            })
        except Exception:
            raise
        y_true.append(correct_answer)
    logging.info("Inference time: %.2f seconds", time.time() - start_time)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f'{mode}_results.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info("Saved %d results to %s", len(results), out_path)

    cls_metrics = classification_metrics(y_true, y_pred)
    cls_metrics['macro_f1'] = (
        sum(m['f1'] for m in cls_metrics['per_class'].values()) / len(cls_metrics['per_class']))
    logging.info(
        "Correct: %d/%d, Accuracy: %.4f, Macro F1: %.4f",
        cls_metrics['correct'], cls_metrics['total'],
        cls_metrics['accuracy'], cls_metrics['macro_f1'])
    logging.info("Per-class metrics:")
    logging.info("%-30s %-9s %-9s %-9s %-7s", 'class', 'precision', 'recall', 'f1', 'support')
    for cls in cls_metrics['classes']:
        m = cls_metrics['per_class'][cls]
        logging.info("%-30s %.4f    %.4f    %.4f    %7d",
                     cls, m['precision'], m['recall'], m['f1'], m['support'])
    classes = cls_metrics['classes']
    logging.info("Confusion matrix (rows=gt, cols=pred):")
    logging.info("%-22s %s", "", "  ".join(f"{c[:10]:>10}" for c in classes))
    for gt in classes:
        row = cls_metrics['confusion'].get(gt, {})
        logging.info("%-22s %s", gt[:20], "  ".join(f"{row.get(pred, 0):10}" for pred in classes))

    return {f'{mode}_acc': round(float(cls_metrics['accuracy']), 4),
            f'{mode}_macro_f1': round(float(cls_metrics['macro_f1']), 4)}


def test(adapter, test_annos, args):
    logging.info("Running test evaluation with LoRA adapter from: %s", args.adapter_ckpt_path)
    ckpt_path = os.path.abspath(args.adapter_ckpt_path)
    if isinstance(adapter.model, PeftModel):
        # Model is already wrapped (just finished training) — load the best checkpoint weights in-place.
        adapter.model.load_adapter(ckpt_path, adapter_name="default", is_trainable=False)
        adapter.model.set_adapter("default")
    else:
        adapter.model = PeftModel.from_pretrained(adapter.model, ckpt_path, adapter_name="default")
    adapter.model.eval()
    test_metrics = evaluate(adapter, test_annos, args, mode='test', save_dir=args.log_dir)
    logging.info("Final test results: %s", test_metrics)


def maybe_sample_annotations(annos, args, split_name, seed_offset=0):
    if args.sample_n is None or args.sample_n <= 0 or args.sample_n >= len(annos):
        return annos
    sampled = _sample_stratified(annos, args.sample_n, args.sample_seed + seed_offset)
    logging.info(
        "Using stratified %s subset of %d/%d annotations (seed=%d)",
        split_name, len(sampled), len(annos), args.sample_seed + seed_offset)
    return sampled


def load_data(args):
    annos_test = json.loads(Path(args.ann_file_test).read_text())
    annos_test = maybe_sample_annotations(annos_test, args, 'test', seed_offset=2)
    if args.mode == 'eval':
        logging.info("Loaded test=%d", len(annos_test))
        return [], [], annos_test

    annos_train = json.loads(Path(args.ann_file_train).read_text())
    annos_val = json.loads(Path(args.ann_file_val).read_text())
    annos_train = maybe_sample_annotations(annos_train, args, 'train', seed_offset=0)
    annos_val = maybe_sample_annotations(annos_val, args, 'val', seed_offset=1)
    logging.info("Loaded train=%d, val=%d, test=%d",
                 len(annos_train), len(annos_val), len(annos_test))
    return annos_train, annos_val, annos_test


def train(adapter, train_annos, val_annos, args):
    target_modules = adapter.get_peft_target_modules(args.ft_type)

    if args.init_adapter_path:
        logging.info("Loading pretrained LoRA adapter from %s for initialization",
                     args.init_adapter_path)
        adapter.model = PeftModel.from_pretrained(adapter.model, os.path.abspath(args.init_adapter_path))
        adapter.model = adapter.model.merge_and_unload()
        logging.info("Merged pretrained adapter into base model weights")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        bias='none',
        task_type='CAUSAL_LM' if 'InternVL' not in args.model_name else None,
    )
    adapter.model = get_peft_model(adapter.model, lora_config)
    adapter.model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, adapter.model.parameters()), lr=args.lr)

    parameter_counts(adapter.model)

    # Weighted sampler to balance class frequency
    labels = []
    for a in train_annos:
        correct = a.get('answer') or 'OTHER'
        labels.append(str(correct))
    counts = Counter(labels)
    logging.info("Training class distribution: %s", counts)
    weights = [1.0 / counts[lbl] for lbl in labels]

    train_dataset = VideoQADataset(train_annos, video_root=args.video_root)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(train_dataset), replacement=True)
    
    collate = adapter.make_collate_fn(args)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            sampler=sampler, collate_fn=collate, num_workers=0)

    global_step = 0
    best_metric = -1.0
    best_step = 0
    best_checkpoints = []  # list of (metric, step, path), sorted desc

    for epoch in range(args.epochs):
        logging.info("Starting epoch %d/%d", epoch + 1, args.epochs)
        loop = tqdm(dataloader, desc=f'epoch {epoch}')
        for batch in loop:
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            try:
                outputs = adapter.model(**batch)
                loss = outputs.loss
            except Exception as e:
                if is_cuda_oom(e):
                    logging.error("Model forward failed for batch: %s", e)
                    raise
                logging.warning("Model forward failed for batch: %s", e)
                continue
            if not loss.requires_grad:
                raise RuntimeError(
                    f"Loss is detached for ft_type={args.ft_type}; selected LoRA target modules "
                    "did not participate in this forward pass. Check visual/text input fields "
                    "and target module names.")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            loop.set_postfix(loss=loss.item())

        if len(val_annos) > 0:
            adapter.model.eval()
            val_metrics = evaluate(adapter, val_annos, args, mode='val')
            adapter.model.train()
            
            val_metric = val_metrics.get(args.monitor)
            if val_metric is None:
                logging.warning("Validation metric %s not found: %s", args.monitor, val_metrics)
            else:
                worst_metric = (min(m for m, _, _ in best_checkpoints)
                                if best_checkpoints else None)
                should_save = (len(best_checkpoints) < args.top_k
                               or val_metric > worst_metric)
                if should_save:
                    best_step = global_step
                    best_metric = max(val_metric, best_metric)
                    best_save_dir = os.path.join(
                        args.log_dir,
                        f'best-{args.monitor}-{val_metric:.4f}-step-{best_step}')
                    os.makedirs(best_save_dir, exist_ok=True)
                    adapter.model.save_pretrained(best_save_dir)
                    logging.info("Saved checkpoint to %s (%s=%.4f)",
                                 best_save_dir, args.monitor, val_metric)
                    best_checkpoints.append((val_metric, best_step, best_save_dir))
                    best_checkpoints.sort(key=lambda x: x[0], reverse=True)
                    while len(best_checkpoints) > args.top_k:
                        removed_metric, removed_step, removed_path = best_checkpoints.pop(-1)
                        try:
                            if os.path.exists(removed_path):
                                shutil.rmtree(removed_path)
                        except Exception as e:
                            logging.warning("Failed to remove checkpoint %s: %s",
                                            removed_path, e)
            logging.info("Epoch %d val metrics: %s", epoch + 1, val_metrics)

    logging.info("Top-%d checkpoints: %s",
                 args.top_k,
                 [(round(m, 4), s, p) for m, s, p in best_checkpoints])

    if best_checkpoints:
        best_ckpt_path = best_checkpoints[0][2]
        if os.path.exists(best_ckpt_path):
            for f in Path(best_ckpt_path).iterdir():
                shutil.copy(f, args.log_dir)
        logging.info("Training completed. Adapter saved to %s", best_ckpt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--ann_file_template',
                        default='features/groundvqa/annotations.VRbinary__crossing_intention__mode_close.json')
    parser.add_argument('--video_root', default='data/videodata_256/clips')
    parser.add_argument('--cache_dir', default='.cache')
    parser.add_argument('--log_dir', type=str, default='logs/vlm_training_')
    parser.add_argument('--adapter_ckpt_path', default="",
                        help='Path to LoRA adapter checkpoint for evaluation (required if mode=eval)')
    parser.add_argument('--init_adapter_path', default='',
                        help='Path to a pretrained LoRA adapter to initialize from. '
                             'The adapter is merged into base weights before attaching a fresh LoRA, '
                             'implementing sequential auxiliary-task pretraining.')

    parser.add_argument('--model_name', default='Qwen/Qwen3-VL-2B-Instruct',
                        choices=TRAINABLE_MODEL_NAMES)
    parser.add_argument('--quantize', action='store_true',
                        help='Load local model with 8-bit quantization (bitsandbytes)')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--top_k', type=int, default=1,
                        help='Number of top checkpoints to keep during training')
    parser.add_argument('--keep_intermediate_checkpoints', action='store_true',
                        help='Keep best-* checkpoint directories after successful final evaluation')
    parser.add_argument('--random_seed', type=int, default=42)

    parser.add_argument('--prompt_variant', default='p0',
                        choices=['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'])
    parser.add_argument('--cot_type', type=str, default='none',
                        help='Chain of Thought prompt variant to use')
    parser.add_argument('--task_type', type=str, default='crossing_intention',
                        choices=list(TASK_VOCAB.keys()),
                        help='Task vocabulary for output parsing during evaluation')
    parser.add_argument('--context_features', default='none',
                        help='Comma-separated context features: none,ego_motion,vehicle_motion,gaze_direction,gaze_direction_change,gaze_on_screen,gaze_on_screen_ratio,gaze_target,pose,demographics,all')
    parser.add_argument('--context_feature_fps', default='auto',
                        help='Text feature sampling rate, e.g. auto, 1, 2, 4.')
    parser.add_argument('--context_prompt_mode', default=None, choices=['preface', 'interleaved'],
                        help='Place feature text before visual input or next to interleaved frames. Default follows --interleaved_timestamps.')
    parser.add_argument('--context_feature_interpretation', default='none', choices=['none', 'brief', 'detailed'],
                        help='Add an interpretation/legend for selected structured context features.')
    parser.add_argument('--context_feature_format', default='legacy',
                        choices=['detailed', 'legacy', 'compact', 'schema', 'summary'],
                        help='Text serialization for structured context features.')
                        
    parser.add_argument('--lora_rank', type=int, default=2)
    parser.add_argument('--lora_alpha', type=int, default=8)
    parser.add_argument('--ft_type', type=str, default='lora_vlm_bridger',
                        choices=['lora_llm_attn_qv', 'lora_llm_attn_qkvo', 'lora_llm_mlp',
                                 'lora_llm_attn_mlp', 'lora_llm_vlm_bridger', 'lora_vlm_bridger'])

    parser.add_argument('--monitor', type=str, default='val_acc',
                        choices=['val_macro_f1', 'val_acc'])
    parser.add_argument('--eval_deterministic', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--eval_max_new_tokens', type=int, default=512)
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help='Reserved for future batched generation; train_vlm currently evaluates sequentially')
    parser.add_argument('--sample_n', type=int, default=None,
                        help='Use a stratified random subset of N annotations per split')
    parser.add_argument('--sample_seed', type=int, default=42,
                        help='Random seed for stratified subset sampling')

    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--max_frames', type=int, default=2048,
                        help='Maximum number of frames to process (Qwen token limit, not a frame-count control)')
    parser.add_argument('--video_duration', type=float, default=2.0,
                        help='Duration of each video clip in seconds (for fps calculation)')
    parser.add_argument('--interleaved_timestamps', action='store_true',
                        help='Interleave frame images with [t=Xs] timestamp text')
    parser.add_argument('--total_pixels', type=int, default=20480 * 32 * 32)
    parser.add_argument('--min_pixels', type=int, default=64 * 32 * 32)
    args = parser.parse_args()

    if args.mode == 'eval' and not args.adapter_ckpt_path:
        parser.error('--adapter_ckpt_path is required when --mode eval')
    if args.epochs < 1:
        parser.error('--epochs must be >= 1')
    if args.batch_size < 1:
        parser.error('--batch_size must be >= 1')
    if args.lr <= 0:
        parser.error('--lr must be > 0')
    if args.top_k < 1:
        parser.error('--top_k must be >= 1 because train mode evaluates the best checkpoint')
    if args.lora_rank < 1:
        parser.error('--lora_rank must be >= 1')
    if args.lora_alpha < 1:
        parser.error('--lora_alpha must be >= 1')
    if args.eval_max_new_tokens < 1:
        parser.error('--eval_max_new_tokens must be >= 1')
    if args.eval_batch_size < 1:
        parser.error('--eval_batch_size must be >= 1')
    if args.sample_n is not None and args.sample_n < 1:
        parser.error('--sample_n must be >= 1 when provided')
    if args.num_frames < 1:
        parser.error('--num_frames must be >= 1')
    if args.max_frames < 1:
        parser.error('--max_frames must be >= 1')
    if args.video_duration <= 0:
        parser.error('--video_duration must be > 0')
    if args.total_pixels < 1:
        parser.error('--total_pixels must be >= 1')
    if args.min_pixels < 1:
        parser.error('--min_pixels must be >= 1')
    if args.min_pixels > args.total_pixels:
        parser.error('--min_pixels must be <= --total_pixels')

    selected_context_features = parse_context_features(args.context_features)
    if args.context_prompt_mode is None:
        args.context_prompt_mode = 'interleaved' if (args.interleaved_timestamps and selected_context_features) else 'preface'

    args.ann_file_train = args.ann_file_template.replace('mode', 'train')
    args.ann_file_val = args.ann_file_template.replace('mode', 'val')
    args.ann_file_test = args.ann_file_template.replace('mode', 'test')
    args.video_sample_fps = max(1, int(round(args.num_frames / args.video_duration)))

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_tag = ''
    if args.context_features != 'none':
        cf_tag = args.context_features.replace(',', '+')
        run_tag = f'_ctx{cf_tag}_cfps{args.context_feature_fps}_{args.context_prompt_mode}_{args.context_feature_format}'
        if args.context_feature_interpretation != 'none':
            run_tag += f'_interp{args.context_feature_interpretation}'
    args.log_dir = os.path.join(args.log_dir, args.model_name.split('/')[-1], ts + run_tag)
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f'{args.mode}.log')
    setup_logging(log_file)
    logging.info("Arguments: %s", args)

    enable_strict_determinism(args.random_seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    adapter = get_adapter(args.model_name)
    adapter.load(args.model_name, quantize=args.quantize)

    train_annos, val_annos, test_annos = load_data(args)

    if args.mode == 'train':
        logging.info("Starting training; log_file=%s", log_file)
        train(adapter, train_annos, val_annos, args)
        args.adapter_ckpt_path = args.log_dir
        logging.info("Starting test evaluation with best adapter from %s", args.adapter_ckpt_path)
        args.eval_deterministic = True
        logging.info('### Deterministic generation:')
        test(adapter, test_annos, args)
        if not args.keep_intermediate_checkpoints:
            removed = cleanup_intermediate_checkpoints(args.log_dir)
            logging.info("Removed %d intermediate checkpoint directories after successful evaluation: %s",
                         len(removed), removed)
    else:
        test(adapter, test_annos, args)


if __name__ == '__main__':
    main()
