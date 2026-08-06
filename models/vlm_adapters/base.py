"""Abstract base class for VLM adapters."""

from abc import ABC, abstractmethod
from typing import Any


class VLMAdapter(ABC):
    """Common interface for all VLM adapters used in eval_vlm.py.

    Each adapter is responsible for:
      - Loading/unloading the model and processor
      - Building the message dict from a video + prompt
      - Running a single inference call and returning (output_text, meta_dict)

    The ``meta_dict`` must contain at minimum:
      ``is_finished`` (bool), ``duration`` (float), ``token_count`` (int),
      ``tokens_per_sec`` (float), ``temperature`` (float|None),
      ``max_new_tokens`` (int).
    """

    @abstractmethod
    def load(self, model_name: str, quantize: bool = False) -> None:
        """Load model weights and processor into memory."""

    @abstractmethod
    def build_messages(self, video_path: str, prompt: str,
                       context: str = '',
                       question: str = '',
                       frames=None, timestamps=None,
                       interleaved: bool = False,
                       no_video: bool = False,
                       total_pixels: int = 20480 * 28 * 28,
                       min_pixels: int = 16 * 28 * 28,
                       **kwargs) -> Any:
        """Build the model-specific message structure.

        Args:
            video_path: Path to the video file.
            prompt: Full text prompt (``context + question`` concatenated).
              Provided for backwards-compatibility; prefer the split fields.
            context: Scene/task description that should appear *before* the
              visual content.  Empty string when not present in the annotation.
            question: Multiple-choice question + options (+ CoT suffix) that
              should appear *after* the visual content.
            frames: Pre-extracted frames as np.ndarray [N,H,W,3] (optional).
            timestamps: Timestamps for each frame in seconds (optional).
            interleaved: If True, interleave frame images with timestamp text.
              Ordering: context → (timestamp, image)×N → question.
            no_video: If True, send text-only (no frames or video encoded).
            total_pixels: Total pixel budget (Qwen-family only).
            min_pixels: Minimum pixels per frame (Qwen-family only).
        """

    @abstractmethod
    def run_inference(self, messages: Any, max_new_tokens: int = 512,
                      temperature: float = 0.0) -> tuple[str, dict]:
        """Run inference and return (output_text, meta_dict)."""

    def get_peft_target_modules(self, ft_type: str) -> list[str]:
        """Return the list of target modules for LoRA based on ft_type."""
        raise NotImplementedError("Training not implemented for this adapter.")

    def make_collate_fn(self, args) -> Any:
        """Return a collate function for DataLoader."""
        raise NotImplementedError("Training not implemented for this adapter.")

    def unload(self) -> None:
        """Release model from GPU memory (optional override)."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        import torch
        torch.cuda.empty_cache()
