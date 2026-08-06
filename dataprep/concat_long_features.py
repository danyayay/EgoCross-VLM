#!/usr/bin/env python3
"""
Concatenate long-form features into short/segment features for the same person.

Usage examples:
  python scripts/concat_long_features.py --dry-run
  python scripts/concat_long_features.py --out-dir features/unified_enhanced --overwrite

Behavior & assumptions:
- Short/segment features are in `features/unified` and often named like `P12S10-0.pt`.
- Long features are in `features_long/unified` named like `P12S9.pt` (one file per trial/session).
- For each short file `PxxSyy(-idx).pt`, if there exist long files for the same Pxx with session < yy,
  they will be concatenated (in chronological order) and then joined with the short feature.
- Concatenation strategy is flexible:
  * By default we try to concat along the last dimension (feature axis). If shapes don't match,
    the script will try reasonable fallbacks (pooling long features over time, repeating a 1D long
    vector along time, or concatenating along time axis).
  * You can force axis with --axis (0 or 1), where 0 is time/rows and 1 is features/cols.

Important: this script requires PyTorch to load/save `.pt` tensors. If PyTorch isn't available,
the script will exit with an instruction to install it.

The script writes outputs to `features/unified_enhanced/` by default to avoid overwriting originals.
"""
import argparse
import os
import re
from pathlib import Path
import sys

try:
    import torch
except Exception as e:
    print("This script requires PyTorch to load/save .pt files. Please install torch (pip install torch) and retry.")
    raise


RE_SHORT = re.compile(r'^(P\d+S(\d+))(?:-(\d+))?\.pt$')


def list_long_sessions(long_dir, person):
    # returns dict session_num -> Path
    found = {}
    p = Path(long_dir)
    if not p.exists():
        return found
    for f in p.iterdir():
        m = re.match(rf'^{person}S?(\d+)\.pt$', f.name)
        if m:
            try:
                s = int(m.group(1))
            except:
                continue
            found[s] = f
    # the above regex isn't ideal; use a generic parse
    for f in p.iterdir():
        m = RE_SHORT.match(f.name)
        if m and m.group(1).startswith(person):
            s = int(m.group(2))
            found[s] = f
    return found


def parse_short_name(name):
    m = RE_SHORT.match(name)
    if not m:
        return None
    base = m.group(1)  # PxxSyy
    session = int(m.group(2))
    idx = m.group(3)
    return base, session, idx


def safe_load(path: Path):
    # torch.load may return tensors, dicts, etc. Handle tensor and numpy-like
    obj = torch.load(str(path))
    if hasattr(obj, 'ndim'):
        return obj
    # common pattern: saved as a dict {'features': tensor}
    if isinstance(obj, dict):
        for k in ('features','feat','x'):
            if k in obj and hasattr(obj[k], 'ndim'):
                return obj[k]
    raise ValueError(f'Unsupported .pt content in {path}: {type(obj)}')


def ensure_out_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def concat_with_fallback(short_t, long_list, axis):
    """
    short_t: tensor (e.g., [T, D])
    long_list: list of tensors (each may be [T',D'] or [D'] or [1,D'])
    axis: requested concat axis (0 or 1)
    Returns the concatenated tensor.
    """
    # if no long, return short
    if not long_list:
        return short_t

    # make a single long tensor: concatenate long_list along axis=0 (stack temporally)
    try:
        long_cat = torch.cat([l if isinstance(l, torch.Tensor) else torch.tensor(l) for l in long_list], dim=0)
    except Exception:
        # fall back to stacking then flatten/pool
        long_cat = torch.stack(long_list, dim=0)

    # If user asked axis=1 (feature axis), try to produce a [T, D_long]
    if axis == 1:
        # if long_cat is 1D -> expand to (T, D_long)
        if long_cat.ndim == 1:
            D_long = long_cat.shape[0]
            T = short_t.shape[0] if short_t.ndim >= 1 else 1
            long_exp = long_cat.unsqueeze(0).expand(T, D_long).contiguous()
            return torch.cat([short_t, long_exp], dim=1)
        if long_cat.ndim == 2:
            # if time length matches, concat along features
            if short_t.ndim == 2 and long_cat.shape[0] == short_t.shape[0]:
                return torch.cat([short_t, long_cat], dim=1)
            # else pool long_cat over time -> (D_long,), expand
            pooled = long_cat.mean(dim=0)
            T = short_t.shape[0]
            long_exp = pooled.unsqueeze(0).expand(T, -1).contiguous()
            return torch.cat([short_t, long_exp], dim=1)

    # If axis == 0 (time axis), try to concat along rows
    if axis == 0:
        # if short and long have same feature dim, and are 2D, concat
        if short_t.ndim == 2 and long_cat.ndim == 2 and short_t.shape[1] == long_cat.shape[1]:
            return torch.cat([long_cat, short_t], dim=0)
        # if long is 1D but short is 2D -> expand long to (T_long, D_short) by repeating pooled value
        if long_cat.ndim == 1 and short_t.ndim == 2:
            D = short_t.shape[1]
            pooled = long_cat
            if pooled.shape[0] != D:
                # reduce/expand via mean and repeat
                pooled = pooled.mean().unsqueeze(0).expand(D)
            long_exp = pooled.unsqueeze(0).expand(1, -1)
            return torch.cat([long_exp, short_t], dim=0)

    # If nothing matched, as last resort, flatten both and concat along last dim
    sflat = short_t.reshape(-1)
    lflat = long_cat.reshape(-1)
    out = torch.cat([sflat, lflat], dim=0)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--short-dir', default='features/unified', help='directory with short/segment .pt files')
    p.add_argument('--long-dir', default='features_long/unified', help='directory with long .pt files (one per trial)')
    p.add_argument('--out-dir', default='features_mem2/unified', help='where to write concatenated outputs')
    p.add_argument('--axis', type=int, choices=[0,1], default=0, help='force concat axis: 0=time, 1=features; if omitted try feature axis first')
    p.add_argument('--memory-length', type=int, default=2, help='limit to most recent t earlier trials (0 = use all earlier sessions)')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--overwrite', action='store_true', default=True)
    p.add_argument('--verbose', action='store_true', default=True)
    args = p.parse_args()

    short_dir = Path(args.short_dir)
    long_dir = Path(args.long_dir)
    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)

    short_files = sorted([f for f in short_dir.iterdir() if f.suffix == '.pt'])
    total = 0
    updated = 0
    skipped = 0
    copied = 0

    # Build index of long sessions per person
    long_index = {}
    for f in long_dir.iterdir():
        if not f.is_file():
            continue
        m = RE_SHORT.match(f.name)
        if not m:
            # try simpler parse PxxSyy.pt
            m2 = re.match(r'^(P\d+)S(\d+)\.pt$', f.name)
            if m2:
                person = m2.group(1)
                s = int(m2.group(2))
                long_index.setdefault(person, {})[s] = f
            continue
        full = m.group(1)  # PxxSyy
        person = re.match(r'^(P\d+)', full).group(1)
        s = int(m.group(2))
        long_index.setdefault(person, {})[s] = f

    for sf in short_files:
        total += 1
        parsed = parse_short_name(sf.name)
        if not parsed:
            if args.verbose:
                print('skip malformed name', sf)
            skipped += 1
            continue
        base, session, idx = parsed
        person = re.match(r'^(P\d+)', base).group(1)

        # find long sessions for this person that are < session
        person_longs = long_index.get(person, {})
        if not len(person_longs):
            if args.verbose:
                print(f'no long features for {sf.name} (person {person}) -> skip')
            skipped += 1
            continue

        earlier = [s for s in person_longs.keys() if s < session]
        if not earlier:
            skipped += 1
            # copy the original ones
            short_t = safe_load(sf)
            out_path = out_dir / sf.name
            torch.save(short_t, str(out_path))
            if args.verbose:
                print(f'no earlier long for {sf.name} (person {person}) -> copied')
                print(f'saving copied feature to {out_path}, shape: {short_t.shape}')
            copied += 1
            continue

        earlier_sorted = sorted(earlier)
        # if memory-length > 0, keep only the most recent `t` earlier sessions
        if args.memory_length and args.memory_length > 0:
            mem = int(args.memory_length)
            if mem < len(earlier_sorted):
                earlier_sorted = earlier_sorted[-mem:]
        long_paths = [person_longs[s] for s in earlier_sorted]

        if args.verbose:
            print(f'processing {sf.name}, will use long sessions: {earlier_sorted}')

        if args.dry_run:
            updated += 1
            continue

        short_t = safe_load(sf)
        long_tensors = []
        for lp in long_paths:
            try:
                long_tensors.append(safe_load(lp))
            except Exception as e:
                print(f'warning: failed to load {lp}: {e}')

        # try axis heuristics
        if args.axis is None:
            # try feature axis first
            try_axes = [1, 0]
        else:
            try_axes = [args.axis]

        out_tensor = None
        for ax in try_axes:
            try:
                out_tensor = concat_with_fallback(short_t, long_tensors, axis=ax)
                break
            except Exception as e:
                if args.verbose:
                    print(f'concat failed with axis {ax} for {sf.name}: {e}')
                out_tensor = None

        if out_tensor is None:
            print(f'failed to concat for {sf.name}, skipping')
            skipped += 1
            continue

        out_path = out_dir / sf.name
        if out_path.exists() and not args.overwrite:
            print(f'skipping existing {out_path} (use --overwrite to replace)')
            skipped += 1
            continue

        if args.verbose:
            print(f'saving concatenated feature to {out_path}, shape: {out_tensor.shape}')
        torch.save(out_tensor, str(out_path))
        updated += 1

    print(f'done. total short files scanned: {total}, updated: {updated}, skipped: {skipped}, copied: {copied}')


if __name__ == '__main__':
    main()
