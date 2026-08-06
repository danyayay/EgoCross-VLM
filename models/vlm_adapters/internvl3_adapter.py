"""InternVL3 adapter (OpenGVLab/InternVL3-2B, OpenGVLab/InternVL3-8B).

InternVL3 differs from Qwen-family in three key ways:
  1. Uses AutoModel (not a named generation class)
  2. No Processor — frames are passed as pixel tensors via dynamic_preprocess
  3. Video is represented as a sequence of <image> tokens in the prompt string

Preprocessing follows the InternVL3 HuggingFace model card.
"""

import time
import logging
import math
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from models.vlm_adapters.base import VLMAdapter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_SIZE = 448  # InternVL3 native resolution
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'


def _build_transform(input_size: int = IMG_SIZE):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 6,
                         image_size: int = IMG_SIZE) -> list[Image.Image]:
    """Tile the image into at most max_num patches for dynamic high-res processing."""
    orig_w, orig_h = image.size
    aspect = orig_w / orig_h
    best_ratio_n, best_ratio_d = 1, 1
    best_area = 0
    for n in range(min_num, max_num + 1):
        for d in range(1, n + 1):
            if n % d == 0:
                ratio = n / d
                area = min(ratio / aspect, aspect / ratio) * n
                if area > best_area and d <= orig_w / image_size and n // d <= orig_h / image_size:
                    best_ratio_n, best_ratio_d = n, d
                    best_area = area
    # fallback: 1×1 (single tile)
    target_w = best_ratio_n * image_size
    target_h = best_ratio_d * image_size
    resized = image.resize((target_w, target_h))
    tiles = []
    for row in range(best_ratio_d):
        for col in range(best_ratio_n):
            box = (col * image_size, row * image_size,
                   (col + 1) * image_size, (row + 1) * image_size)
            tiles.append(resized.crop(box))
    # Append a thumbnail for global context
    tiles.append(image.resize((image_size, image_size)))
    return tiles


def _load_image_tensor(image: Image.Image, max_num: int = 6,
                        image_size: int = IMG_SIZE,
                        no_thumbnail: bool = False) -> torch.Tensor:
    """Return a [num_patches, 3, H, W] float tensor for one image."""
    transform = _build_transform(image_size)
    patches = _dynamic_preprocess(image, max_num=max_num, image_size=image_size)
    if no_thumbnail and len(patches) > 1:
        patches = patches[:-1]  # drop the appended thumbnail
    return torch.stack([transform(p) for p in patches])


class InternVL3Adapter(VLMAdapter):

    def load(self, model_name: str, quantize: bool = False) -> None:
        self.model_name = model_name
        quantization_config = None
        if quantize:
            # NF4 4-bit corrupts generation on InternVL3-8B (outputs degenerate to
            # repeated "!" tokens) — the larger Qwen2.5 LLM backbone has activation
            # outliers that blockwise NF4 quantization doesn't handle. LLM.int8()
            # with threshold=0 forces all outlier features to fp16 and fixes it,
            # at ~10GB peak vs ~7GB for NF4. Verified both InternVL3-2B and -8B
            # produce correct output under this config.
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=0.0,
            )
            logging.info("Loading %s with 8-bit quantization (int8 threshold=0)", model_name)
        else:
            logging.info("Loading %s in BF16", model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Patch: InternVL3's cached modeling code predates the all_tied_weights_keys
        # API in newer transformers. Patch all call sites that access this attribute.
        import transformers.modeling_utils as _mu
        import transformers.integrations.accelerate as _accel_mod
        import transformers.quantizers.base as _qbase

        _orig_move = _mu.PreTrainedModel._move_missing_keys_from_meta_to_device
        _orig_init_infer = _accel_mod._init_infer_auto_device_map
        _orig_get_keys = _qbase.get_keys_to_not_convert

        def _safe_move(self_model, *args, **kwargs):
            if not hasattr(type(self_model), 'all_tied_weights_keys') and \
               'all_tied_weights_keys' not in self_model.__dict__:
                self_model.__dict__['all_tied_weights_keys'] = {}
            return _orig_move(self_model, *args, **kwargs)

        def _safe_init_infer(model, *args, **kwargs):
            if not hasattr(type(model), 'all_tied_weights_keys') and \
               'all_tied_weights_keys' not in model.__dict__:
                model.__dict__['all_tied_weights_keys'] = {}
            return _orig_init_infer(model, *args, **kwargs)

        def _safe_get_keys(model, *args, **kwargs):
            if not hasattr(type(model), 'all_tied_weights_keys') and \
               'all_tied_weights_keys' not in model.__dict__:
                model.__dict__['all_tied_weights_keys'] = {}
            return _orig_get_keys(model, *args, **kwargs)

        _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _safe_move
        _accel_mod._init_infer_auto_device_map = _safe_init_infer
        _qbase.get_keys_to_not_convert = _safe_get_keys
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if not quantize else torch.float16,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=quantization_config,
            ).eval()
        finally:
            _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _orig_move
            _accel_mod._init_infer_auto_device_map = _orig_init_infer
            _qbase.get_keys_to_not_convert = _orig_get_keys

        # Use the model's own num_image_token (accounts for downsample_ratio)
        self.num_image_token = self.model.num_image_token

    def build_messages(self, video_path: str, prompt: str,
                       context: str = '',
                       question: str = '',
                       frames=None, timestamps=None,
                       interleaved: bool = False,
                       no_video: bool = False,
                       **kwargs) -> dict:
        """Return a dict with 'pixel_values', 'num_patches_list', and 'question'."""
        if no_video:
            return {'pixel_values': None, 'num_patches_list': [], 'question': prompt}
        if frames is None:
            raise ValueError("InternVL3Adapter requires pre-extracted frames (pass frames=...)")

        all_tensors = []
        num_patches_list = []
        image_placeholders = []

        frame_contexts = kwargs.get('frame_contexts') or []
        for i, frame in enumerate(frames):
            pil_img = Image.fromarray(frame)
            # max_num=1, no_thumbnail: 256×256 frames are already smaller than IMG_SIZE=448;
            # the thumbnail adds no information and doubles the token count.
            tensor = _load_image_tensor(pil_img, max_num=1, no_thumbnail=True)  # [1, 3, H, W]
            all_tensors.append(tensor)
            num_patches_list.append(tensor.shape[0])

            if interleaved and timestamps is not None:
                ts = float(timestamps[i].mean())
                ts_text = f"[t={ts:.2f}s]"
                if i < len(frame_contexts) and frame_contexts[i]:
                    ts_text = ts_text + "\n" + frame_contexts[i]
                image_placeholders.append(f"{ts_text}<image>")
            else:
                image_placeholders.append("<image>")

        pixel_values = torch.cat(all_tensors, dim=0)  # [total_patches, 3, H, W]
        # Use <image> placeholders — model.chat() replaces each with the correct
        # IMG_CONTEXT_TOKEN sequence using num_patches_list internally.
        frames_str = "\n".join(image_placeholders)

        if interleaved:
            # Ordering: context → (timestamp, image)×N → question
            parts = []
            if context:
                parts.append(context)
            parts.append(frames_str)
            parts.append(question if question else prompt)
            full_question = "\n".join(p for p in parts if p)
        else:
            # Ordering (non-interleaved): frames → context → question
            text_after = (context + " " + (question if question else prompt)).strip() \
                if context else (question if question else prompt)
            full_question = f"{frames_str}\n{text_after}"

        return {
            'pixel_values': pixel_values,
            'num_patches_list': num_patches_list,
            'question': full_question,
        }

    def run_inference(self, messages: dict, max_new_tokens: int = 512,
                      temperature: float = 0.0) -> tuple[str, dict]:
        pixel_values = messages['pixel_values']
        if pixel_values is not None:
            # Match pixel_values dtype to the vision model's patch_embedding weight
            vit_dtype = self.model.vision_model.embeddings.patch_embedding.weight.dtype
            pixel_values = pixel_values.to(vit_dtype).to('cuda')
        question = messages['question']
        num_patches_list = messages['num_patches_list'] or None

        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0.0}
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        start = time.perf_counter()
        output_text = self.model.chat(
            self.tokenizer, pixel_values, question,
            generation_config=gen_kwargs,
            num_patches_list=num_patches_list,
            history=None, return_history=False,
        )
        duration = time.perf_counter() - start

        token_count = len(self.tokenizer.encode(output_text, add_special_tokens=False))
        is_finished = not output_text.endswith('...')

        return output_text, {
            'is_finished': is_finished,
            'duration': duration,
            'token_count': token_count,
            'tokens_per_sec': token_count / duration if duration > 0 else float('inf'),
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
        }

    def get_peft_target_modules(self, ft_type: str) -> list[str]:
        target_modules_by_type = {
            'lora_llm_attn_qv':       ['q_proj', 'v_proj'],
            'lora_llm_attn_qkvo':     ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'lora_llm_mlp':           ['gate_proj', 'up_proj', 'down_proj'],
            'lora_llm_attn_mlp':      ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                       'gate_proj', 'up_proj', 'down_proj'],
            'lora_llm_vlm_bridger':   ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                       'gate_proj', 'up_proj', 'down_proj',
                                       'mlp1.1', 'mlp1.3'],
            'lora_vlm_bridger':       ['mlp1.1', 'mlp1.3'],
        }
        if ft_type not in target_modules_by_type:
            raise ValueError(f"Unsupported ft_type {ft_type} for InternVL3Adapter")
        return target_modules_by_type[ft_type]

    def make_collate_fn(self, args):
        from utils.eval_utils import (
            build_context_feature_interpretation,
            build_interleaved_context_preface,
            build_interleaved_context_texts,
            build_prompt_from_annotation,
            get_video_frames,
            parse_context_features,
        )
        import torch
        import os
        
        def collate(batch):
            input_ids_list = []
            labels_list = []
            pixel_values_list = []
            num_patches_list = []

            for anno in batch:
                video_path = os.path.join(args.video_root, f"{anno.get('video_id')}.mp4")
                _, frames, timestamps = get_video_frames(video_path, num_frames=args.num_frames, cache_dir=args.cache_dir)

                selected_context_features = parse_context_features(getattr(args, 'context_features', 'none'))
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
                
                messages = self.build_messages(
                    video_path, prompt_text, context=context_text, question=question_text,
                    frames=frames, timestamps=timestamps,
                    interleaved=args.interleaved_timestamps,
                    frame_contexts=frame_contexts)
                
                pixel_values = messages['pixel_values']
                if pixel_values is not None:
                    vit_dtype = self.model.vision_model.embeddings.patch_embedding.weight.dtype
                    pixel_values = pixel_values.to(vit_dtype)
                    pixel_values_list.append(pixel_values)
                    num_patches_list.extend(messages['num_patches_list'])

                question = messages['question']
                # InternVL3 training template:
                # User: <question>\nAssistant: <answer>
                # The model uses Qwen2.5 tokenizer which has <|im_start|> user\n...<|im_end|>\n<|im_start|>assistant\n
                prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
                answer_text = f"{correct_answer}<|im_end|>\n"
                
                prompt_ids = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
                answer_ids = self.tokenizer(answer_text, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
                
                input_ids = torch.cat([prompt_ids, answer_ids], dim=0)
                labels = torch.cat([torch.full_like(prompt_ids, -100), answer_ids], dim=0)
                
                input_ids_list.append(input_ids)
                labels_list.append(labels)

            max_len = max(len(ids) for ids in input_ids_list)
            padded_input_ids = torch.full((len(batch), max_len), self.tokenizer.pad_token_id or 0, dtype=torch.long)
            padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
            
            for i in range(len(batch)):
                seq_len = len(input_ids_list[i])
                padded_input_ids[i, :seq_len] = input_ids_list[i]
                padded_labels[i, :seq_len] = labels_list[i]
                attention_mask[i, :seq_len] = True

            batch_dict = {
                'input_ids': padded_input_ids,
                'labels': padded_labels,
                'attention_mask': attention_mask,
            }
            if pixel_values_list:
                batch_dict['pixel_values'] = torch.cat(pixel_values_list, dim=0)
                batch_dict['image_flags'] = torch.ones(len(batch_dict['pixel_values']), dtype=torch.long)
            
            return batch_dict

        return collate

adapter_class = InternVL3Adapter
