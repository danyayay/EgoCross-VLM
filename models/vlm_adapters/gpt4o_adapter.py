"""GPT-4o adapter (API-based).

Requires: pip install openai
API key: set OPENAI_API_KEY environment variable.

GPT-4o does not accept video files directly. Frames are sent as base64 JPEG
image_url content blocks. Low-detail mode is used (85 tokens/image) since
the source frames are already 256×256 — high-detail would upscale and add
tokens with no meaningful benefit.
"""

import base64
import io
import logging
import os
import time
from typing import Any

import numpy as np
from PIL import Image

from models.vlm_adapters.base import VLMAdapter


def _frame_to_b64(frame: np.ndarray) -> str:
    """Encode a uint8 RGB [H,W,3] numpy frame as a base64 JPEG string."""
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


class GPT4oAdapter(VLMAdapter):

    def load(self, model_name: str = "gpt-4o", quantize: bool = False) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Set OPENAI_API_KEY environment variable")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        logging.info("GPT-4o adapter ready (%s)", model_name)

    def build_messages(self, video_path: str, prompt: str,
                       context: str = '',
                       question: str = '',
                       frames=None, timestamps=None,
                       interleaved: bool = False,
                       no_video: bool = False,
                       **kwargs) -> list:
        """Return OpenAI messages list.

        Interleaved ordering:  context → (timestamp, image)×N → question.
        Non-interleaved ordering: all frames → context + question (full prompt).
        """
        if no_video:
            return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        if frames is None:
            raise ValueError("GPT4oAdapter requires pre-extracted frames (pass frames=...)")

        content = []

        if interleaved:
            # Interleaved: context first, then (timestamp + image) pairs, then question
            if context:
                content.append({"type": "text", "text": context})
            frame_contexts = kwargs.get('frame_contexts') or []
            for i, frame in enumerate(frames):
                if timestamps is not None:
                    ts = float(timestamps[i].mean())
                    ts_text = f"[t={ts:.2f}s]"
                    if i < len(frame_contexts) and frame_contexts[i]:
                        ts_text = ts_text + "\n" + frame_contexts[i]
                    content.append({"type": "text", "text": ts_text})
                b64 = _frame_to_b64(frame)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                })
            content.append({"type": "text", "text": question if question else prompt})
        else:
            # Non-interleaved: all frames first, then the full prompt
            for frame in frames:
                b64 = _frame_to_b64(frame)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                })
            content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]

    def run_inference(self, messages: list, max_new_tokens: int = 512,
                      temperature: float = 0.0) -> tuple[str, dict]:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        duration = time.perf_counter() - start

        output_text = response.choices[0].message.content or ""
        token_count = response.usage.completion_tokens if response.usage else 0
        is_finished = response.choices[0].finish_reason == "stop"

        return output_text, {
            'is_finished': is_finished,
            'duration': duration,
            'token_count': token_count,
            'tokens_per_sec': token_count / duration if duration > 0 else float('inf'),
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
        }

    def unload(self) -> None:
        pass  # API-based, nothing to unload


adapter_class = GPT4oAdapter
