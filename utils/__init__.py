"""Utility modules for data loading, preprocessing, and training helpers."""

import importlib

from utils.util import seed_everything

_VR_DATASET_EXPORTS = {
    "VRDataset",
    "collate_fn",
    "parse_person_from_feat_id",
    "split_person_groups",
}


def __getattr__(name):
    """Lazily import heavier optional utilities."""
    if name in _VR_DATASET_EXPORTS:
        vr_dataset = importlib.import_module("utils.vr_dataset")
        value = getattr(vr_dataset, name)
        globals()[name] = value
        return value
    if name == "qwen_utils":
        module = importlib.import_module("utils.qwen_utils")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'utils' has no attribute {name!r}")

__all__ = [
    "seed_everything",
    "VRDataset",
    "collate_fn",
    "parse_person_from_feat_id",
    "split_person_groups",
    "qwen_utils",
]
