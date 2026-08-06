"""VLM adapter registry for multi-model zero-shot evaluation."""

from models.vlm_adapters.base import VLMAdapter

MODEL_REGISTRY = {
    "Qwen/Qwen3-VL-2B-Instruct":       "models.vlm_adapters.qwen3vl_adapter",
    "Qwen/Qwen3-VL-8B-Instruct":       "models.vlm_adapters.qwen3vl_adapter",
    "Qwen/Qwen3-VL-8B-Thinking":       "models.vlm_adapters.qwen3vl_adapter",
    "Qwen/Qwen2.5-VL-7B-Instruct":     "models.vlm_adapters.qwen25vl_adapter",
    "OpenGVLab/InternVL3-2B":           "models.vlm_adapters.internvl3_adapter",
    "OpenGVLab/InternVL3-8B":           "models.vlm_adapters.internvl3_adapter",
    "gemini-2.0-flash":                 "models.vlm_adapters.gemini_adapter",
    "gpt-4o":                           "models.vlm_adapters.gpt4o_adapter",
}

MODEL_NAMES = list(MODEL_REGISTRY.keys())


def get_adapter(model_name: str) -> VLMAdapter:
    """Import and return the adapter instance for the given model name."""
    import importlib
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {MODEL_NAMES}")
    module = importlib.import_module(MODEL_REGISTRY[model_name])
    return module.adapter_class()
