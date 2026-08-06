"""
Simple LoRA fine-tuning script for Qwen3-VL using PEFT.

Supports multiple --ft_type options controlling which modules receive LoRA adapters.

Usage (example):
  python -m training.train_qwen \\
    --ann_file_template features/groundvqa/annotations.VRbinary_00000_mode_close.json \\
    --output_dir adapters/qwen_text_lora --epochs 3 --batch_size 2 --lr 1e-4
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
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from utils.analyze_groundvqa_results import classification_metrics
from utils.qwen_utils import build_message
from utils.util import cleanup_intermediate_checkpoints, enable_strict_determinism, setup_logging

import random


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


def build_prompt_from_annotation(anno, video_dir):
    """Build a multiple-choice prompt from an annotation dict.

    Returns (video_path, prompt_text, options_map, correct_answer).
    """
    video_id = anno.get('video_id')
    video_path = os.path.join(video_dir, f"{video_id}.mp4")

    question = anno.get('question')
    correct = anno.get('answer')
    wrongs = anno.get('wrong_answers')
    if isinstance(wrongs, list):
        wrongs = [w.strip() for w in wrongs if w.strip()]
    options = [str(correct)] + [str(w) for w in wrongs]
    assert len(options) > 1, "Annotation must have at least one correct and one wrong answer"

    random.shuffle(options)
    letters = [chr(ord('A') + i) for i in range(len(options))]
    options_map = {letters[i]: options[i] for i in range(len(options))}
    options_str = ' '.join([f"({k}) {v}" for k, v in options_map.items()])
    prompt_text = f"{question} Choose one option: {options_str}"
    return video_path, prompt_text, options_map, correct


class VideoQADataset(Dataset):
    """Dataset that holds annotation dicts and returns them for collating."""
    def __init__(self, annos, video_root):
        self.annos = annos
        self.video_root = video_root

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        return self.annos[idx]


def _build_sample_messages(anno, args):
    """Build processor messages and return (messages, correct_answer) for one annotation."""
    video_path, prompt_text, _, correct_answer = build_prompt_from_annotation(
        anno, video_dir=args.video_root)
    messages = build_message(
        video_path, prompt_text,
        total_pixels=args.total_pixels, min_pixels=args.min_pixels,
        max_frames=args.num_frames, fps=args.sample_fps)
    return messages, correct_answer


def make_multimodal_collate(processor, tokenizer, args):
    """Return a collate_fn that converts a list of annotation dicts into model training tensors."""
    def collate(batch):
        messages_list = []
        answers = []
        for anno in batch:
            messages, correct_answer = _build_sample_messages(anno, args)
            messages_list.append(messages)
            answers.append(correct_answer)

        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in messages_list]

        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages_list, return_video_kwargs=True, image_patch_size=16,
                return_video_metadata=True)
        except Exception as e:
            logging.warning("process_vision_info failed in collate: %s", e)
            image_inputs = video_inputs = None
            video_kwargs = {}

        if video_inputs is not None:
            try:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
            except Exception:
                video_metadatas = None
        else:
            video_metadatas = None

        inputs = processor(
            text=texts, images=image_inputs, videos=video_inputs,
            video_metadata=video_metadatas, **(video_kwargs or {}),
            do_resize=False, return_tensors='pt', padding=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        batch_input_ids = inputs['input_ids']
        batch_attention = inputs.get('attention_mask', torch.ones_like(batch_input_ids))

        answer_id_seqs = [
            tokenizer(a + (tokenizer.eos_token or ''), add_special_tokens=False,
                      return_tensors='pt')['input_ids'][0]
            for a in answers
        ]

        new_input_ids, new_attention, new_labels = [], [], []
        for i in range(batch_input_ids.size(0)):
            prompt_ids = batch_input_ids[i]
            prompt_att = batch_attention[i]
            answer_ids = answer_id_seqs[i]
            concatenated = torch.cat([prompt_ids, answer_ids], dim=0)
            concatenated_att = torch.cat(
                [prompt_att, torch.ones(answer_ids.size(0), dtype=prompt_att.dtype)], dim=0)
            labels = concatenated.clone()
            labels[:prompt_ids.size(0)] = -100  # mask prompt; train only on answer tokens
            new_input_ids.append(concatenated)
            new_attention.append(concatenated_att)
            new_labels.append(labels)

        B = batch_input_ids.size(0)
        max_len = max(t.size(0) for t in new_input_ids)
        padded_inputs = torch.full((B, max_len), tokenizer.pad_token_id or 0, dtype=torch.long)
        padded_att = torch.zeros((B, max_len), dtype=batch_attention.dtype)
        padded_labels = torch.full((B, max_len), -100, dtype=torch.long)
        for i in range(B):
            seq_len = new_input_ids[i].size(0)
            padded_inputs[i, :seq_len] = new_input_ids[i]
            padded_att[i, :seq_len] = new_attention[i]
            padded_labels[i, :seq_len] = new_labels[i]

        return {
            'input_ids': padded_inputs,
            'attention_mask': padded_att,
            'labels': padded_labels,
            'pixel_values_videos': inputs.get('pixel_values_videos', None),
            'video_grid_thw': inputs.get('video_grid_thw', None),
        }

    return collate


def inference(model, processor, video, prompt, generation_style=None):
    """Run multimodal inference and return the generated text."""
    messages = build_message(
        video, prompt,
        total_pixels=generation_style['total_pixels'],
        min_pixels=generation_style['min_pixels'],
        max_frames=generation_style['max_frames'],
        fps=generation_style['fps'])
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages], return_video_kwargs=True, image_patch_size=16, return_video_metadata=True)
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        video_metadata=video_metadatas, **video_kwargs,
        do_resize=False, return_tensors="pt", padding=True)
    inputs = inputs.to('cuda')
    output_ids = model.generate(
        **inputs,
        max_new_tokens=generation_style['max_new_tokens'],
        do_sample=generation_style['do_sample'])
    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    is_finished = (processor.tokenizer.eos_token_id == generated_ids[-1][-1].item())
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0], is_finished


TASK_VOCAB = {
    'crossing_intention': {'cross', 'yield'},
    'vehicle_speed':      {'moving', 'static'},
    'vehicle_approach':   {'moving closer to me', 'moving away from me'},
}


def evaluate(model, processor, ann_file, video_dir, mode='test', generation_style=None,
             task_type='crossing_intention', save_dir=None):
    valid_answers = TASK_VOCAB.get(task_type, set())
    y_true, y_pred = [], []
    results = []
    start_time = time.time()
    for anno in tqdm(ann_file, desc='evaluating'):
        video_path, prompt_text, options_map, correct_answer = build_prompt_from_annotation(
            anno, video_dir)
        try:
            out, is_finished = inference(model, processor, video_path, prompt_text,
                                         generation_style=generation_style)
            retry_count = 0
            while not is_finished and retry_count < 2:
                retry_count += 1
                retry_style = {**generation_style,
                               'max_new_tokens': generation_style['max_new_tokens'] * retry_count,
                               'do_sample': True}
                out, is_finished = inference(model, processor, video_path, prompt_text,
                                             generation_style=retry_style)
            raw_out = out
            out = out.strip().lower()
            if any(ans in out for ans in valid_answers):
                out = out.lstrip('(a)').lstrip('(b)').lstrip('(c)').lstrip('(d)').strip()
                if out == 'ross': out = 'cross'
            else:
                out = 'others'
            # else:
            #     out = options_map.get(out) or options_map.get(out.upper())
            # if out is None:
            #     out = 'others'
            y_pred.append(out)
            results.append({
                'video_id': anno.get('video_id'),
                'answer': correct_answer,
                'pred_answer': out,
                'output_text': raw_out,
                'is_finished': is_finished,
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


def test(model, processor, test_annos, video_dir, adapter_ckpt_path, generation_style,
         task_type='crossing_intention', save_dir=None):
    logging.info("Running test evaluation with LoRA adapter from: %s", adapter_ckpt_path)
    model = PeftModel.from_pretrained(model, os.path.abspath(adapter_ckpt_path),
                                      adapter_name="default")
    test_metrics = evaluate(model, processor, test_annos, video_dir=video_dir,
                            mode='test', generation_style=generation_style,
                            task_type=task_type, save_dir=save_dir)
    logging.info("Final test results: %s", test_metrics)


def load_data(args):
    annos_train = json.loads(Path(args.ann_file_train).read_text())
    annos_val = json.loads(Path(args.ann_file_val).read_text())
    annos_test = json.loads(Path(args.ann_file_test).read_text())
    logging.info("Loaded train=%d, val=%d, test=%d",
                 len(annos_train), len(annos_val), len(annos_test))
    return annos_train, annos_val, annos_test


def train(model, processor, train_annos, val_annos, args, generation_style):
    target_modules_by_type = {
        'lora_llm_attn_qv':       ['q_proj', 'v_proj'],
        'lora_llm_attn_qkvo':     ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        'lora_llm_mlp':           ['gate_proj', 'up_proj', 'down_proj'],
        'lora_llm_attn_mlp':      ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                   'gate_proj', 'up_proj', 'down_proj'],
        'lora_llm_vlm_bridger':   ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                   'gate_proj', 'up_proj', 'down_proj',
                                   'linear_fc1', 'linear_fc2'],
        'lora_vlm_bridger':       ['linear_fc1', 'linear_fc2'],
    }
    if args.ft_type not in target_modules_by_type:
        raise ValueError(f"Unsupported ft_type {args.ft_type}")
    target_modules = target_modules_by_type[args.ft_type]

    if args.init_adapter_path:
        logging.info("Loading pretrained LoRA adapter from %s for initialization",
                     args.init_adapter_path)
        model = PeftModel.from_pretrained(model, os.path.abspath(args.init_adapter_path))
        model = model.merge_and_unload()
        logging.info("Merged pretrained adapter into base model weights")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_config)
    model.train()

    tokenizer = processor.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    parameter_counts(model)

    # Weighted sampler to balance class frequency
    labels = []
    for a in train_annos:
        try:
            _, _, _, correct = build_prompt_from_annotation(a, video_dir=args.video_root)
        except Exception:
            correct = a.get('answer') or 'OTHER'
        labels.append(str(correct))
    counts = Counter(labels)
    logging.info("Training class distribution: %s", counts)
    weights = [1.0 / counts[lbl] for lbl in labels]

    train_dataset = VideoQADataset(train_annos, video_root=args.video_root)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(train_dataset), replacement=True)
    collate = make_multimodal_collate(processor, tokenizer, args)
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
            batch = {k: v.to(args.device) for k, v in batch.items()}
            try:
                outputs = model(**batch)
                loss = outputs.loss
            except Exception as e:
                logging.warning("Model forward failed for batch: %s", e)
                continue
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            loop.set_postfix(loss=loss.item())

        if len(val_annos) > 0:
            val_metrics = evaluate(model, processor, val_annos,
                                   video_dir=args.video_root, mode='val',
                                   generation_style=generation_style,
                                   task_type=args.task_type)
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
                    model.save_pretrained(best_save_dir)
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
    parser.add_argument('--ann_file_template',
                        default='features/groundvqa_qn3/annotations.VRbinary_00000_mode_close.json')
    parser.add_argument('--mode', choices=['train', 'eval'], default='eval')
    parser.add_argument('--adapter_ckpt_path', default="logs/qwen_training_intention/Qwen3-VL-2B-Instruct/20260415_085039",
                        help='Path to LoRA adapter checkpoint for evaluation (required if mode=eval)')
    parser.add_argument('--init_adapter_path', default='logs/qwen_training_auxtask/Qwen3-VL-2B-Instruct/20260414_202017/',
                        help='Path to a pretrained LoRA adapter to initialize from. '
                             'The adapter is merged into base weights before attaching a fresh LoRA, '
                             'implementing sequential auxiliary-task pretraining.')

    parser.add_argument('--model_name', default='Qwen/Qwen3-VL-2B-Instruct',
                        choices=['Qwen/Qwen3-VL-2B-Instruct', 'Qwen/Qwen3-VL-4B-Instruct-FP8',
                                 'Qwen/Qwen3-VL-8B-Instruct', 'Qwen/Qwen3-VL-2B-Instruct-FP8'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--lora_rank', type=int, default=2)
    parser.add_argument('--lora_alpha', type=int, default=8)
    parser.add_argument('--ft_type', type=str, default='lora_llm_vlm_bridger',
                        choices=['lora_llm_attn_qv', 'lora_llm_attn_qkvo', 'lora_llm_mlp',
                                 'lora_llm_attn_mlp', 'lora_llm_vlm_bridger', 'lora_vlm_bridger'])

    parser.add_argument('--log_dir', type=str, default='logs/qwen_training_intention')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--monitor', type=str, default='val_acc',
                        choices=['val_macro_f1', 'val_acc'])
    parser.add_argument('--top_k', type=int, default=1,
                        help='Number of top checkpoints to keep during training')
    parser.add_argument('--keep_intermediate_checkpoints', action='store_true',
                        help='Keep best-* checkpoint directories after successful final evaluation')

    parser.add_argument('--eval_deterministic', action='store_true', default=True)
    parser.add_argument('--eval_max_new_tokens', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=16)

    parser.add_argument('--task_type', type=str, default='crossing_intention',
                        choices=list(TASK_VOCAB.keys()),
                        help='Task vocabulary for output parsing during evaluation')
    parser.add_argument('--video_root', default='data/videodata_256/clips')
    parser.add_argument('--num_frames', type=int, default=64)
    parser.add_argument('--total_pixels', type=int, default=20480 * 32 * 32)
    parser.add_argument('--min_pixels', type=int, default=64 * 32 * 32)
    parser.add_argument('--sample_fps', type=float, default=2.0)
    args = parser.parse_args()

    args.ann_file_train = args.ann_file_template.replace('mode', 'train')
    args.ann_file_val = args.ann_file_template.replace('mode', 'val')
    args.ann_file_test = args.ann_file_template.replace('mode', 'test')

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.log_dir = os.path.join(args.log_dir, args.model_name.split('/')[-1], ts)
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f'{args.mode}.log')
    setup_logging(log_file)
    logging.info("Arguments: %s", args)

    enable_strict_determinism(args.random_seed)
    generation_style = {
        'max_new_tokens': args.eval_max_new_tokens,
        'do_sample': not args.eval_deterministic,
        'total_pixels': args.total_pixels,
        'min_pixels': args.min_pixels,
        'max_frames': args.num_frames,
        'fps': args.sample_fps,
    }

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype="auto", device_map='auto')
    processor = AutoProcessor.from_pretrained(args.model_name)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_annos, val_annos, test_annos = load_data(args)

    if args.mode == 'train':
        logging.info("Starting training; log_file=%s", log_file)
        train(model, processor, train_annos, val_annos, args, generation_style=generation_style)
        args.adapter_ckpt_path = args.log_dir
        logging.info("Starting test evaluation with best adapter from %s", args.adapter_ckpt_path)
        generation_style['do_sample'] = False
        logging.info('### Deterministic generation:')
        test(model, processor, test_annos, args.video_root, args.adapter_ckpt_path, generation_style,
             task_type=args.task_type, save_dir=args.log_dir)
        if not args.keep_intermediate_checkpoints:
            removed = cleanup_intermediate_checkpoints(args.log_dir)
            logging.info("Removed %d intermediate checkpoint directories after successful evaluation: %s",
                         len(removed), removed)
    else:
        test(model, processor, test_annos, args.video_root, args.adapter_ckpt_path, generation_style,
             task_type=args.task_type, save_dir=args.log_dir)


if __name__ == '__main__':
    main()
