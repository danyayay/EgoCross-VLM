"""Gemini 2.0 Flash adapter (API-based).

Requires: pip install google-genai
API key: set GEMINI_API_KEY environment variable.

Videos are sent as inline base64-encoded parts (suitable for ≤20 MB clips).
Frames can optionally be sent as interleaved image+text parts.
"""

import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from models.vlm_adapters.base import VLMAdapter


class GeminiAdapter(VLMAdapter):

    def load(self, model_name: str = "gemini-2.0-flash", quantize: bool = False) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Install google-genai: pip install google-genai")

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("Set GEMINI_API_KEY environment variable")

        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.model_name = model_name
        logging.info("Gemini adapter ready (%s)", model_name)

    def build_messages(self, video_path: str, prompt: str,
                       context: str = '',
                       question: str = '',
                       frames=None, timestamps=None,
                       interleaved: bool = False,
                       no_video: bool = False,
                       **kwargs) -> list:
        """Return a list of Gemini content parts.

        Interleaved ordering:  context → (image, timestamp)×N → question.
        Non-interleaved ordering: video/frames → context + question (full prompt).
        """
        types = self.types

        if no_video:
            return [types.Part.from_text(text=prompt)]

        if interleaved and frames is not None and timestamps is not None:
            from PIL import Image as PILImage
            parts = []
            if context:
                parts.append(types.Part.from_text(text=context))
            frame_contexts = kwargs.get('frame_contexts') or []
            for i, (frame, ts) in enumerate(zip(frames, timestamps)):
                buf = io.BytesIO()
                PILImage.fromarray(frame).save(buf, format='JPEG', quality=85)
                jpeg_bytes = buf.getvalue()
                parts.append(types.Part.from_bytes(data=jpeg_bytes, mime_type='image/jpeg'))
                ts_text = f"[t={float(np.mean(ts)):.2f}s]"
                if i < len(frame_contexts) and frame_contexts[i]:
                    ts_text = ts_text + "\n" + frame_contexts[i]
                parts.append(types.Part.from_text(text=ts_text))
            parts.append(types.Part.from_text(text=question if question else prompt))
            return parts

        if frames is not None:
            # Non-interleaved: send frames as images, then full prompt
            from PIL import Image as PILImage
            parts = []
            for frame in frames:
                buf = io.BytesIO()
                PILImage.fromarray(frame).save(buf, format='JPEG', quality=85)
                jpeg_bytes = buf.getvalue()
                parts.append(types.Part.from_bytes(data=jpeg_bytes, mime_type='image/jpeg'))
            parts.append(types.Part.from_text(text=prompt))
            return parts

        # Send raw video file inline
        video_bytes = Path(video_path).read_bytes()
        mime = 'video/mp4'
        parts = [
            types.Part.from_bytes(data=video_bytes, mime_type=mime),
            types.Part.from_text(text=prompt),
        ]
        return parts

    def run_inference(self, messages: list, max_new_tokens: int = 512,
                      temperature: float = 0.0) -> tuple[str, dict]:
        config = self.types.GenerateContentConfig(
            max_output_tokens=max_new_tokens,
            temperature=temperature if temperature > 0.0 else 0.0,
        )
        start = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=messages,
            config=config,
        )
        duration = time.perf_counter() - start
        output_text = response.text or ""
        token_count = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
        is_finished = response.candidates[0].finish_reason.name == "STOP" if response.candidates else True

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


adapter_class = GeminiAdapter
