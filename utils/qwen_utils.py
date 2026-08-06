"""Shared utilities for Qwen3-VL training and evaluation scripts."""

from __future__ import annotations

import logging
import os
import random
from typing import Dict, List

import numpy as np
import torch




def make_qwen_collate_fn(processor, tokenizer, args):
    """Return a collate_fn for Qwen models."""
    from utils.eval_utils import (
        build_context_feature_interpretation,
        build_interleaved_context_preface,
        build_interleaved_context_texts,
        build_prompt_from_annotation,
        get_video_frames,
        parse_context_features,
    )
    from qwen_vl_utils import process_vision_info
    
    def collate(batch):
        messages_list = []
        answers = []
        for anno in batch:
            video_path = os.path.join(args.video_root, f"{anno.get('video_id')}.mp4")
            frames, timestamps = None, None
            selected_context_features = parse_context_features(getattr(args, 'context_features', 'none'))
            needs_frames = (getattr(args, 'interleaved_timestamps', False)
                            or (getattr(args, 'context_prompt_mode', 'preface') == 'interleaved'
                                and bool(selected_context_features)))
            if needs_frames:
                _, frames, timestamps = get_video_frames(
                    video_path, num_frames=args.num_frames, cache_dir=args.cache_dir)

            context_features_for_preface = getattr(args, 'context_features', 'none')
            frame_contexts = None
            if getattr(args, 'context_prompt_mode', 'preface') == 'interleaved' and timestamps is not None:
                non_demo = [f for f in selected_context_features if f != 'demographics']
                demo = [f for f in selected_context_features if f == 'demographics']
                context_features_for_preface = ','.join(demo) if demo else 'none'
                frame_contexts = build_interleaved_context_texts(
                    anno, timestamps, non_demo,
                    context_feature_format=getattr(args, 'context_feature_format', 'detailed'))

            prompt_interpretation = (
                'none' if getattr(args, 'context_prompt_mode', 'preface') == 'interleaved'
                else getattr(args, 'context_feature_interpretation', 'none'))
            prompt_text, context_text, question_text, _, correct_answer = build_prompt_from_annotation(
                anno, cot_type=args.cot_type,
                prompt_variant=getattr(args, 'prompt_variant', 'p0'),
                task_name=getattr(args, 'task_type', None),
                context_features=context_features_for_preface,
                context_feature_fps=getattr(args, 'context_feature_fps', 'auto'),
                timestamps=timestamps,
                num_frames=getattr(args, 'num_frames', None),
                video_duration=getattr(args, 'video_duration', None),
                context_feature_interpretation=prompt_interpretation,
                context_feature_format=getattr(args, 'context_feature_format', 'detailed'))

            if getattr(args, 'context_prompt_mode', 'preface') == 'interleaved':
                interleaved_preface = build_interleaved_context_preface(
                    anno, timestamps, getattr(args, 'context_features', 'none'),
                    context_feature_format=getattr(args, 'context_feature_format', 'detailed'))
                if interleaved_preface and interleaved_preface not in context_text:
                    context_text = (context_text + "\n" + interleaved_preface).strip() if context_text else interleaved_preface
                    prompt_text = (context_text + ' ' + question_text).strip()

            if (getattr(args, 'context_prompt_mode', 'preface') == 'interleaved'
                    and getattr(args, 'context_feature_interpretation', 'none') != 'none'):
                interp_text = build_context_feature_interpretation(
                    getattr(args, 'context_features', 'none'),
                    getattr(args, 'context_feature_interpretation', 'none'),
                    context_feature_format=getattr(args, 'context_feature_format', 'detailed'))
                if interp_text and interp_text not in context_text:
                    context_text = (context_text + "\n" + interp_text).strip() if context_text else interp_text
                    prompt_text = (context_text + ' ' + question_text).strip()

            # Support Qwen message formats. When requested, mirror eval_vlm's
            # timestamp-interleaved ordering: context -> (timestamp, frame) * N -> question.
            if getattr(args, 'interleaved_timestamps', False):
                messages = build_interleaved_message(
                    frames, timestamps, prompt_text,
                    total_pixels=args.total_pixels, min_pixels=args.min_pixels,
                    context=context_text, question=question_text,
                    frame_contexts=frame_contexts)
            else:
                messages = build_message(
                    video_path, prompt_text,
                    total_pixels=args.total_pixels, min_pixels=args.min_pixels,
                    max_frames=args.num_frames, video_sample_fps=args.video_sample_fps,
                    context=context_text, question=question_text)
            messages_list.append(messages)
            answers.append(str(correct_answer))

        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in messages_list]

        patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", 14)
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages_list, return_video_kwargs=True, image_patch_size=patch_size,
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

        if getattr(tokenizer, 'pad_token', None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        batch_input_ids = inputs['input_ids']
        batch_attention = inputs.get('attention_mask', torch.ones_like(batch_input_ids))
        batch_mm_token_type_ids = inputs.get('mm_token_type_ids', None)

        answer_id_seqs = [
            tokenizer(a + (tokenizer.eos_token or ''), add_special_tokens=False,
                      return_tensors='pt')['input_ids'][0]
            for a in answers
        ]

        new_input_ids, new_attention, new_labels = [], [], []
        new_mm_ids = [] if batch_mm_token_type_ids is not None else None
        
        for i in range(batch_input_ids.size(0)):
            prompt_ids = batch_input_ids[i]
            prompt_att = batch_attention[i]
            answer_ids = answer_id_seqs[i]
            concatenated = torch.cat([prompt_ids, answer_ids], dim=0)
            concatenated_att = torch.cat(
                [prompt_att, torch.ones(answer_ids.size(0), dtype=prompt_att.dtype)], dim=0)
            labels = concatenated.clone()
            labels[:prompt_ids.size(0)] = -100  # mask prompt; train only on answer tokens
            
            if new_mm_ids is not None:
                prompt_mm = batch_mm_token_type_ids[i]
                concatenated_mm = torch.cat([prompt_mm, torch.zeros(answer_ids.size(0), dtype=prompt_mm.dtype)], dim=0)
                new_mm_ids.append(concatenated_mm)
                
            new_input_ids.append(concatenated)
            new_attention.append(concatenated_att)
            new_labels.append(labels)

        B = batch_input_ids.size(0)
        max_len = max(t.size(0) for t in new_input_ids)
        padded_inputs = torch.full((B, max_len), tokenizer.pad_token_id or 0, dtype=torch.long)
        padded_att = torch.zeros((B, max_len), dtype=batch_attention.dtype)
        padded_labels = torch.full((B, max_len), -100, dtype=torch.long)
        padded_mm_ids = torch.zeros((B, max_len), dtype=torch.long) if new_mm_ids is not None else None
        
        for i in range(B):
            seq_len = new_input_ids[i].size(0)
            padded_inputs[i, :seq_len] = new_input_ids[i]
            padded_att[i, :seq_len] = new_attention[i]
            padded_labels[i, :seq_len] = new_labels[i]
            if padded_mm_ids is not None:
                padded_mm_ids[i, :seq_len] = new_mm_ids[i]

        batch_dict = {
            'input_ids': padded_inputs,
            'attention_mask': padded_att,
            'labels': padded_labels,
        }
        # Non-interleaved Qwen video inputs use video tensors; interleaved
        # timestamp/frame inputs use image tensors. Preserve whichever visual
        # fields the processor produced so vision/projector LoRA participates
        # in the forward pass.
        for key in (
            'pixel_values',
            'image_grid_thw',
            'pixel_values_videos',
            'video_grid_thw',
            'second_per_grid_ts',
        ):
            value = inputs.get(key, None)
            if value is not None:
                batch_dict[key] = value
        if padded_mm_ids is not None:
            batch_dict['mm_token_type_ids'] = padded_mm_ids
        return batch_dict
    return collate


def build_message(video, prompt_text: str, total_pixels: int, min_pixels: int,
                  max_frames: int, video_sample_fps: float,
                  context: str = '', question: str = '') -> List[Dict]:
    """Build a single-turn user message dict for the Qwen3-VL processor.

    Ordering (non-interleaved): video → context → question.

    Args:
        video: Video file path or dict accepted by qwen_vl_utils.
        prompt_text: Full prompt string (context + question concatenated).
          Used as fallback when ``context`` and ``question`` are both empty.
        context: Scene/task description to place before the question.
        question: Multiple-choice question + options (+ CoT suffix).
        total_pixels: Total pixel budget for Qwen-family models.
        min_pixels: Minimum pixels per frame.
        max_frames: Maximum number of frames the model will decode.
        video_sample_fps: Target frame sampling rate.
    """
    # Build text portion: context first, then question, after the video.
    if context or question:
        text_parts = []
        if context:
            text_parts.append(context)
        text_parts.append(question or prompt_text)
        text_content = ' '.join(text_parts)
    else:
        text_content = prompt_text

    return [{"role": "user", "content": [
        {"video": video, "total_pixels": total_pixels, "min_pixels": min_pixels,
         "max_frames": max_frames, "video_sample_fps": video_sample_fps},
        {"type": "text", "text": text_content},
    ]}]


def build_interleaved_message(frames: np.ndarray, timestamps: np.ndarray,
                              prompt_text: str, total_pixels: int,
                              min_pixels: int,
                              context: str = '', question: str = '',
                              frame_contexts: list[str] | None = None) -> List[Dict]:
    """Build a single-turn user message with frames interleaved with timestamp text.

    Ordering (interleaved): context → (timestamp, image)×N → question.

    Each frame is inserted as an image content block preceded by a
    ``[t=X.XXs]`` text block, giving the model explicit temporal grounding.
    Intended for use with Qwen3-VL / Qwen2.5-VL via process_vision_info.

    Note: when this message format is used, process_vision_info returns
    ``image_inputs`` (not ``video_inputs``).  The calling inference function
    must handle the case where ``video_inputs is None`` and pass
    ``images=image_inputs`` to the processor instead.

    Args:
        frames: uint8 RGB array of shape [N, H, W, 3].
        timestamps: float array of shape [N] with per-frame timestamps in seconds.
        prompt_text: Full prompt string (context + question concatenated).
          Used as fallback when ``context`` and ``question`` are both empty.
        total_pixels: Total pixel budget shared across all frames (Qwen parameter).
        min_pixels: Minimum pixels per frame (Qwen parameter).
        context: Scene/task description to place *before* the first frame.
        question: Multiple-choice question + options to place *after* all frames.
    """
    from PIL import Image
    content = []

    # Context goes first, before any visual content
    if context:
        content.append({"type": "text", "text": context})

    # Interleaved: timestamp → image for each frame
    frame_contexts = frame_contexts or []
    for i, (frame, ts) in enumerate(zip(frames, timestamps)):
        ts_text = f"[t={float(np.mean(ts)):.2f}s]"
        if i < len(frame_contexts) and frame_contexts[i]:
            ts_text = ts_text + "\n" + frame_contexts[i]
        content.append({"type": "text", "text": ts_text})
        content.append({"type": "image", "image": Image.fromarray(frame),
                        "total_pixels": total_pixels, "min_pixels": min_pixels})

    # Question goes after all frames
    content.append({"type": "text", "text": question if question else prompt_text})
    return [{"role": "user", "content": content}]
