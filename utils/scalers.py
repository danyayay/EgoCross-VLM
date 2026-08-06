"""Helpers to compute and persist per-modality mean/std statistics from a DataLoader.

This module implements a simple batch-accumulation (sum, sumsq, count) approach
to compute mean and std without storing all data in memory. The returned stats are
plain numpy arrays and can be saved/loaded with torch.save / torch.load.
"""
from typing import Dict, Any, Iterable
import os
import numpy as np
import torch


def _accumulate_stats(acc, x: np.ndarray):
    # x: (N, D)
    n = x.shape[0]
    s = x.sum(axis=0)
    s2 = (x * x).sum(axis=0)
    acc['count'] += n
    acc['sum'] += s
    acc['sumsq'] += s2


def compute_stats_from_dataloader(dataloader, eps: float = 1e-8) -> Dict[str, Any]:
    """Compute mean/std for pose, gaze, motion and goal distance from the given dataloader.

    Expects dataloader batches to be tuples: (vis, pose, gaze, motion, goal, labels)
    where modalities may be None. goal is expected shape (B, T, 3) with [d, sin, cos].
    Returns a dict with keys 'pose','gaze','motion','goal_dist' each mapping to {'mean': np.array, 'std': np.array}
    for the feature dimension. If a modality is absent in the data, it will be omitted.
    """
    # accumulators per modality
    accs = {}
    # mapping: modality -> batch index in tuple returned by collate_fn
    # collate_fn returns (vis, pose, gaze, motion, vel, goal, labels)
    mapping = {'pose': 1, 'gaze': 2, 'motion': 3, 'vel': 4, 'goal': 5}

    for batch in dataloader:
        # ensure tuple/list
        for mod, idx in mapping.items():
            if idx >= len(batch):
                continue
            x = batch[idx]
            if x is None:
                continue
            # x is torch.Tensor with shape (B, T, D)
            if isinstance(x, torch.Tensor):
                xb = x.cpu().numpy()
            else:
                xb = np.asarray(x)
            if xb.size == 0:
                continue
            B, T = xb.shape[0], xb.shape[1]
            D = xb.shape[2]
            flat = xb.reshape(B * T, D)
            key = mod
            if key not in accs:
                accs[key] = {'count': 0, 'sum': np.zeros((D,), dtype=np.float64), 'sumsq': np.zeros((D,), dtype=np.float64)}
            _accumulate_stats(accs[key], flat)

    stats = {}
    for k, v in accs.items():
        cnt = v['count']
        if cnt == 0:
            continue
        mean = (v['sum'] / cnt).astype(np.float32)
        var = (v['sumsq'] / cnt) - (mean.astype(np.float64) ** 2)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var).astype(np.float32)
        if k == 'goal':
            # only keep distance stats for goal (first column)
            stats['goal_dist'] = {'mean': mean[0:1].astype(np.float32), 'std': std[0:1].astype(np.float32)}
        else:
            stats[k] = {'mean': mean, 'std': std}

    return stats


def save_stats(stats: Dict[str, Any], work_dir: str):
    os.makedirs(os.path.join(work_dir, 'scalers'), exist_ok=True)
    path = os.path.join(work_dir, 'scalers', 'stats.pt')
    # convert numpy arrays to torch tensors for more robust saving
    torch_stats = {}
    for k, v in stats.items():
        torch_stats[k] = {'mean': torch.from_numpy(v['mean']), 'std': torch.from_numpy(v['std'])}
    torch.save(torch_stats, path)
    return path


def load_stats(path: str) -> Dict[str, Any]:
    data = torch.load(path, map_location='cpu')
    stats = {}
    for k, v in data.items():
        stats[k] = {'mean': v['mean'].numpy().astype(np.float32), 'std': v['std'].numpy().astype(np.float32)}
    return stats
