"""Qwen3-VL adapter (Qwen/Qwen3-VL-2B-Instruct, Qwen/Qwen3-VL-8B-Instruct)."""

import time
import logging

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

from models.vlm_adapters.base import VLMAdapter
from utils.qwen_utils import build_message, build_interleaved_message


class Qwen3VLAdapter(VLMAdapter):

    def load(self, model_name: str, quantize: bool = False) -> None:
        self.model_name = model_name
        quantization_config = None
        if quantize:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logging.info("Loading %s with 8-bit quantization", model_name)
        else:
            logging.info("Loading %s in BF16", model_name)

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto" if not quantize else torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
        )
        self.model.eval()

    def build_messages(self, video_path: str, prompt: str,
                       context: str = '',
                       question: str = '',
                       frames=None, timestamps=None,
                       interleaved: bool = False,
                       no_video: bool = False,
                       total_pixels: int = 20480 * 28 * 28,
                       min_pixels: int = 16 * 28 * 28,
                       max_frames: int = 2048,
                       video_sample_fps: float = 2.0,
                       frame_contexts=None,
                       **kwargs):
        if no_video:
            return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        if interleaved and frames is not None and timestamps is not None:
            return build_interleaved_message(frames, timestamps, prompt,
                                             total_pixels, min_pixels,
                                             context=context, question=question,
                                             frame_contexts=frame_contexts)
        return build_message(video_path, prompt, total_pixels, min_pixels,
                             max_frames, video_sample_fps,
                             context=context, question=question)

    def run_inference(self, messages, max_new_tokens: int = 512,
                      temperature: float = 0.0) -> tuple[str, dict]:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages], return_video_kwargs=True,
            image_patch_size=patch_size, return_video_metadata=True)

        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            video_metadata=video_metadatas, **video_kwargs,
            return_tensors="pt")
        inputs = inputs.to('cuda')

        gen_kwargs = {"max_new_tokens": max_new_tokens}
        if temperature == 0.0:
            gen_kwargs.update({"do_sample": False, "temperature": None,
                               "top_p": None, "top_k": None})
        else:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True

        start = time.perf_counter()
        output_ids = self.model.generate(**inputs, **gen_kwargs)
        duration = time.perf_counter() - start

        generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        gen_token_count = sum(len(ids) for ids in generated_ids)
        is_finished = self.processor.tokenizer.eos_token_id == generated_ids[-1][-1]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0]

        return output_text, {
            'is_finished': is_finished.item(),
            'duration': duration,
            'token_count': gen_token_count,
            'tokens_per_sec': gen_token_count / duration if duration > 0 else float('inf'),
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
                                       'linear_fc1', 'linear_fc2'],
            'lora_vlm_bridger':       ['linear_fc1', 'linear_fc2'],
        }
        if ft_type not in target_modules_by_type:
            raise ValueError(f"Unsupported ft_type {ft_type} for Qwen3VLAdapter")
        return target_modules_by_type[ft_type]

    def make_collate_fn(self, args):
        from utils.qwen_utils import make_qwen_collate_fn
        return make_qwen_collate_fn(self.processor, self.processor.tokenizer, args)

adapter_class = Qwen3VLAdapter
