"""Utility functions for training and evaluation.

This module provides common utility functions used across the project,
including seed setting for reproducibility and other helper functions.
"""

import os
import random
import numpy as np
import torch
from pathlib import Path
import shutil


def seed_everything(seed: int) -> None:
    """Set random seeds for all libraries to ensure reproducibility.
    
    Sets seeds for Python's random module, NumPy, and PyTorch (CPU and CUDA).
    Also configures PyTorch to use deterministic algorithms.
    
    Args:
        seed (int): Random seed value to use across all libraries.
        
    Returns:
        None
        
    Example:
        >>> seed_everything(42)
        >>> # All random operations will now be deterministic
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import logging

class _NoDecordFilter(logging.Filter):
    def filter(self, record):
        return 'decord' not in record.name.lower() and 'decord' not in record.getMessage().lower()


def setup_logging(log_file: str) -> logging.Logger:
    """Set up file + stream logging, filtering out noisy decord messages.

    Args:
        log_file: Full path to the log file. The parent directory must already exist.
    """
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()
    ignore_decord = _NoDecordFilter()
    file_handler.addFilter(ignore_decord)
    stream_handler.addFilter(ignore_decord)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[file_handler, stream_handler],
        force=True,
    )
    logging.getLogger('decord').setLevel(logging.WARNING)
    return logging.getLogger(__name__)

def enable_strict_determinism(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup_intermediate_checkpoints(run_dir: str, prefix: str = "best-") -> list[str]:
    """Remove direct checkpoint children after successful final evaluation.

    Final adapter files copied into ``run_dir`` and logs/results are preserved.
    Call this only after evaluation succeeds so failed runs remain resumable.
    """
    root = Path(run_dir).resolve()
    if not root.is_dir():
        return []
    removed = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        resolved = child.resolve()
        if resolved.parent != root:
            continue
        shutil.rmtree(resolved)
        removed.append(str(resolved))
    return removed
