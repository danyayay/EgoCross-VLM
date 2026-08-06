"""VR dataset loader: reads annotation JSON, loads frames via OpenCV, and extracts
head-pose/gaze from per-video CSVs. Returns either flattened per-frame pixels or
raw resized frames (T,H,W,3) to be used with a backbone.

Compact, single-definition implementation (no duplicated content).
"""
import re 
import json
import os
import warnings
from PIL import Image
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from decord import VideoReader, cpu, gpu
from torchvision.transforms._transforms_video import NormalizeVideo
from torchvision import transforms
import torch
from torch.utils.data import Dataset


class VRDataset(Dataset):
    """Dataset for egocentric pedestrian-intention prediction.

    Loads annotation JSON files and returns per-sample tuples of multimodal
    tensors: visual frames, head pose, gaze, body-frame motion, velocity, goal,
    and a class label.

    Args:
        annotations_path: Path to the JSON annotation file.
        videodata_dir: Root directory containing ``.mp4`` video clips.
        vrdata_dir: Directory with per-recording ``.csv`` files (head pose,
            gaze, motion). Pass ``None`` to skip all scalar modalities.
        pose_scaler: Optional ``{'mean': np.ndarray, 'std': np.ndarray}`` dict
            used to normalise head-pose features.
        gaze_scaler: Same format as ``pose_scaler``, for gaze features.
        motion_scaler: Same format, for motion features.
        vel_scaler: Same format, for velocity features.
        goal_dist_scaler: Same format applied only to the distance column (first
            column) of goal features.
        T: Number of temporal frames to sample per clip.
        frame_size: ``(H, W)`` to resize frames to.
        person_ids: Subset of person IDs to include. Auto-detected when ``None``.
        visual_dim: Expected visual feature dimensionality (used only as a hint;
            pass ``None`` for auto).
        pose_cols: CSV column names for head pose. ``None`` = auto-detect,
            ``[]`` = disable.
        gaze_cols: Same as ``pose_cols`` for gaze.
        motion_cols: Same as ``pose_cols`` for motion.
        vel_cols: Same as ``pose_cols`` for velocity.
        goal_cols: Same as ``pose_cols`` for goal.
        visual_encoder: One of ``'pixels'``, ``'cnn'``, ``'clip'``, or ``None``.
        return_raw_frames: If ``True`` return ``(T, H, W, C)`` tensors instead
            of flattened ``(T, H*W*C)`` arrays.
        crop_anchor: Frame crop strategy: ``'center'``, ``'saliency'``, or
            ``'padding'``.
        skip_missing: If ``True``, silently skip samples with missing files.
    """

    def __init__(
        self,
        annotations_path: str,
        videodata_dir: str,
        vrdata_dir: Optional[str] = None,
        pose_scaler: Optional[dict] = None,
        gaze_scaler: Optional[dict] = None,
        motion_scaler: Optional[dict] = None,
        vel_scaler: Optional[dict] = None,
        goal_dist_scaler: Optional[dict] = None,
        T: int = 8,
        frame_size: Tuple[int, int] = (32, 32),
        person_ids: List[str] = None,
        visual_dim: Optional[int] = None,
        pose_cols: Optional[List[str]] = None,
        gaze_cols: Optional[List[str]] = None,
        motion_cols: Optional[List[str]] = None,
        vel_cols: Optional[List[str]] = None,
        goal_cols: Optional[List[str]] = None,
        visual_encoder: str = "pixels",
        return_raw_frames: bool = False,
        crop_anchor: str = "center",
        skip_missing: bool = True,
        vr_meta_csv: Optional[str] = None,
    ):
        self.annotations_path = annotations_path
        self.videodata_dir = videodata_dir
        self.vrdata_dir = vrdata_dir
        self.T = int(T)
        self.frame_size = tuple(frame_size)
        self.skip_missing = skip_missing
        self.return_raw_frames = bool(return_raw_frames)
        # crop_anchor: 'center' or 'saliency' or 'padding - if 'gaze' and gaze coords are available,
        # crop windows will try to center at the gaze coordinates (per-frame). When
        # gaze is near the border and a full crop cannot be placed, the center is
        # moved towards the image center so the crop stays inside the frame as
        # much as possible.
        if crop_anchor not in ("center", "saliency", "padding"):
            raise ValueError("crop_anchor must be 'center' or 'saliency' or 'padding'")
        self.crop_anchor = crop_anchor
        self._meta_df = pd.read_csv(vr_meta_csv) if vr_meta_csv is not None else None

        if visual_encoder == 'clip':
            self.norm_mean = (0.48145466, 0.4578275, 0.40821073)
            self.norm_std = (0.26862954, 0.26130258, 0.27577711)
        else:
            self.norm_mean = (0.485, 0.456, 0.406)
            self.norm_std = (0.229, 0.224, 0.225)
        self.norm_mean_pixel = (124, 116, 104) # in pixel values [0,255]
        # allow disabling visual modality (set visual_encoder=None to skip video input)
        self.visual_encoder = visual_encoder

        # store user-specified column configs (None means auto-detect, [] means disable)
        # Represent as: None -> auto-detect, False -> disabled (empty list), True -> explicit list provided
        self._user_visual_dim = True if (visual_dim is not None and visual_dim > 0) else False
        self._user_pose_cols = None if pose_cols is None else (True if len(pose_cols) > 0 else False)
        self._user_gaze_cols = None if gaze_cols is None else (True if len(gaze_cols) > 0 else False)
        self._user_motion_cols = None if motion_cols is None else (True if len(motion_cols) > 0 else False)
        self._user_vel_cols = None if vel_cols is None else (True if len(vel_cols) > 0 else False)
        self._user_goal_cols = None if goal_cols is None else (True if len(goal_cols) > 0 else False)

        self.pose_cols = pose_cols
        self.gaze_cols = gaze_cols
        self.motion_cols = motion_cols
        self.vel_cols = vel_cols
        self.goal_cols = goal_cols

        # optional scalers (dict with 'mean' and 'std' numpy arrays)
        self.pose_scaler = pose_scaler
        self.gaze_scaler = gaze_scaler
        self.motion_scaler = motion_scaler
        self.vel_scaler = vel_scaler
        self.goal_dist_scaler = goal_dist_scaler

        # load annotations
        with open(annotations_path, "r") as f:
            ann_list = json.load(f)
        
        if person_ids is None:
            person_ids = set()
            for entry in ann_list:
                sample_id = entry['video_id']
                pid = sample_id.split('S')[0]
                person_ids.add(pid)
            person_ids = sorted(person_ids)
            print(f"Auto-detected person ids from annotations: {person_ids}")

        self.samples = []
        for entry in ann_list:                
            sample_id = entry['video_id']
            pid = sample_id.split('S')[0]
            video_id = sample_id.split('-')[0]
            if pid in person_ids:
                video_path = os.path.join(videodata_dir, str(sample_id) + '.mp4') if videodata_dir is not None else None
                csv_path = os.path.join(vrdata_dir, str(video_id) + '.csv') if vrdata_dir is not None else None

                video_start_frame = int(entry['video_start_frame'])
                video_end_frame = int(entry['video_end_frame'])
                clip_start_frame = int(entry['moment_start_frame'])
                clip_end_frame = int(entry['moment_end_frame'])

                label = entry['answer']

                self.samples.append({"video_path": video_path, "csv_path": csv_path,
                                     "video_id": sample_id,
                                     "video_start_frame": video_start_frame, "video_end_frame": video_end_frame, 
                                     "clip_start_frame": clip_start_frame, "clip_end_frame": clip_end_frame,
                                     "label": label})

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found in annotations")

        # map string labels to ints if necessary
        labels = [s["label"] for s in self.samples]
        if any(isinstance(x, str) for x in labels):
            uniq = sorted(set(labels))
            self.label_map = {v: i for i, v in enumerate(uniq)}
            for s in self.samples:
                s["label"] = self.label_map[s["label"]]
        else:
            self.label_map = None
            for s in self.samples:
                s["label"] = int(s["label"])

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def _read_frames(self, video_path: str, start: int, end: int, csv_path: str = '') -> np.ndarray:
        frame_idxs = np.linspace(start, max(start + 1, end), num=self.T, endpoint=False, dtype=int)

        # If visual modality is disabled, return an empty visual tensor (T, 0)
        if self.visual_encoder is None or video_path is None:
            return np.zeros((self.T, 0), dtype=np.float32)

        # read full-resolution frames using decord (works reliably and fast)
        with open(video_path, 'rb') as f:
            vr = VideoReader(f, ctx=cpu(0))
        frames = vr.get_batch(frame_idxs).asnumpy()  # (T, H, W, 3) uint8, return in RGB format
        _, height, width, _ = frames.shape  # height = 256, width = 396 for VR_256
        frames = torch.from_numpy(frames)
        normalize = NormalizeVideo(mean=self.norm_mean, std=self.norm_std)

        if self.crop_anchor == 'center':
            transforms_compose = transforms.Compose([
                    transforms.Resize(self.frame_size),
                    transforms.CenterCrop(self.frame_size),
                    normalize,
                ])
            frames = frames.float() / 255.0
            frames = frames.permute(3, 0, 1, 2) # (C, T, Hresize, Wresize)
            frames = transforms_compose(frames)  # -> (C, T, H, W)
            frames = frames.permute(1, 2, 3, 0).numpy()
            
        elif self.crop_anchor == 'padding':
            # normalize first
            transforms_compose1 = transforms.Compose([normalize])
            frames = frames.float() / 255.0
            frames = frames.permute(3, 0, 1, 2) # (C, T, Hresize, Wresize)
            frames = transforms_compose1(frames)  # -> (C, T, H, W)
            # padding 
            half = int(np.floor((width - height) / 2))
            frames = torch.nn.functional.pad(frames, (0, 0, half, half), mode='constant', value=0.0) 
            # resize 
            transforms_compose2 = transforms.Compose([
                    transforms.Resize(self.frame_size),
                    transforms.CenterCrop(self.frame_size),
                ])
            frames = transforms_compose2(frames)  # -> (C, T, H, W)
            frames = frames.permute(1, 2, 3, 0).numpy() # (T, H, W, C)

        elif self.crop_anchor == 'saliency':

            height_raw, width_raw = 1068, 1536
            T_read, height, width, channel = frames.shape
            half = int(np.floor(height / 2))

            # get the gaze data
            gaze_for_crop = pd.read_csv(csv_path)[['px_u', 'px_v']].iloc[start:end] # (T, 2)
            px_u, px_v = gaze_for_crop['px_u'].values, gaze_for_crop['px_v'].values
            px_u = px_u * width / width_raw
            px_v = px_v * height / height_raw

            # Create output container
            frames_holder = np.zeros((T_read, half*2, half*2, channel), dtype=np.float32)

            for i in range(T_read):
                frame = frames[i]

                top_x = np.clip(px_u[i] - half, 0, width - half*2)
                top_y = np.clip(px_v[i] - half, 0, height - half*2)
                frames_holder[i] = frame[int(top_y):int(top_y)+half*2, int(top_x):int(top_x)+half*2, :]

            transforms_compose = transforms.Compose([
                    transforms.Resize(self.frame_size),
                    transforms.CenterCrop(self.frame_size),
                    normalize,
                ])
            frames = torch.Tensor(frames_holder / 255.0)
            frames = frames.permute(3, 0, 1, 2) # (C, T, Hresize, Wresize)
            frames = transforms_compose(frames)  # -> (C, T, H, W)
            frames = frames.permute(1, 2, 3, 0).numpy()

        if self.return_raw_frames:
            return frames
        T, H, W, C = frames.shape
        return frames.reshape(T, H * W * C)
    
    def _read_vr_csv(self, csv_path: str, start: int, end: int):
        """Read per-frame CSV and return sampled (pose, gaze, motion, goal_ts)

        goal_ts is (T,3) with columns [d, sin(beta), cos(beta)] when angle/dist available.
        """
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
        except Exception:
            try:
                data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
                df = None
            except Exception:
                return None, None, None, None, None

        if df is not None:
            cols = df.columns.tolist()
            arr = df.values
            # Determine which columns to use. Behavior:
            # - if user passed None for a modality, try auto-detect from CSV columns
            # - if user passed an empty list [], treat modality as disabled
            # - if user passed an explicit list, use those column names
            if self._user_gaze_cols is None:
                gaze_cols = [c for c in cols if c.lower().startswith("gaze") or c.lower() in ("gx", "gy", "gaze_x", "gaze_y", "px_u", "px_v")]
            elif len(self.gaze_cols) == 0:
                gaze_cols = []
            else:
                gaze_cols = [c for c in cols if c in self.gaze_cols]

            if self._user_pose_cols is None:
                pose_cols = [c for c in cols if c.lower().startswith("head") or c.lower().startswith("pose") or c.lower() in ("px", "py", "pz", "rx", "ry", "rz")]
            elif len(self.pose_cols) == 0:
                pose_cols = []
            else:
                pose_cols = [c for c in cols if c in self.pose_cols]

            if self._user_motion_cols is None:
                motion_cols = [c for c in cols if c.lower().startswith("ped") and ("loc" in c.lower() or "location" in c.lower())]
            elif len(self.motion_cols) == 0:
                motion_cols = []
            else:
                motion_cols = [c for c in cols if c in self.motion_cols]
            
            if self._user_vel_cols is None:
                vel_cols = [c for c in cols if c.lower().startswith("ped") and ("loc" in c.lower() or "location" in c.lower())]
            elif len(self.vel_cols) == 0:
                vel_cols = []
            else:
                vel_cols = [c for c in cols if c in self.vel_cols]
        else:
            arr = data
            ncols = arr.shape[1]
            pose_cols = list(range(min(6, ncols)))
            gaze_cols = list(range(min(2, ncols)))
            vel_cols = list(range(min(2, ncols)))

        frame_idx_col = None
        if df is not None:
            for c in cols:
                if c.lower() in ("frame"):
                    frame_idx_col = cols.index(c)
                    break

        if frame_idx_col is not None and df is not None:
            frames = df.iloc[:, frame_idx_col].values.astype(int)
            mask = (frames >= start) & (frames < end)
            sub = df[mask].values
        else:
            arr_len = arr.shape[0]
            s = max(0, min(arr_len - 1, start))
            e = max(0, min(arr_len, end))
            sub = arr[s:e]

        if sub.shape[0] == 0:
            return None, None, None, None, None

        idxs = np.linspace(0, sub.shape[0] - 1, num=self.T, dtype=int)
        sampled = sub[idxs]

        pose = None
        gaze = None
        motion = None
        goal_ts = None
        vel = None
        try:
            if df is not None:
                if len(pose_cols) > 0:
                    pose_idx = [cols.index(c) for c in pose_cols if c in cols]
                    if len(pose_idx) > 0:
                        pose = sampled[:, pose_idx]
                if len(gaze_cols) > 0:
                    gaze_idx = [cols.index(c) for c in gaze_cols if c in cols]
                    if len(gaze_idx) > 0:
                        gaze = sampled[:, gaze_idx]
                if len(motion_cols) > 0:
                    motion_idx = [cols.index(c) for c in motion_cols if c in cols]
                    if len(motion_idx) > 0:
                        motion = sampled[:, motion_idx]
                if len(vel_cols) > 0:
                    vel_idx = [cols.index(c) for c in vel_cols if c in cols]
                    if len(vel_idx) > 0:
                        vel = sampled[:, vel_idx]

                # goal extraction: angle and distance (only if goal modality enabled or auto, and meta CSV provided)
                if ((self._user_goal_cols is None) or (len(self.goal_cols) > 0)) and self._meta_df is not None:
                    pid, sid = re.match(r'P(\d+)S(\d+)', os.path.basename(csv_path).split('.')[0]).groups()
                    angle = self._meta_df[(self._meta_df['pid'] == int(pid)) & (self._meta_df['sid'] == int(sid))]['angle'].values[0]
                    goal_deg_map = {0: -45.0, 1: -90.0, 2: -135.0}
                    goal_loc_map = {0: (150, -150), 1: (0, -150), 2: (-150, -150)}
                    # ensure required columns exist
                    if 'Ped_Location_x_smoothed' in cols and 'Ped_Location_y_smoothed' in cols and 'rot_z' in cols:
                        dist_x = goal_loc_map[angle][0] - df.Ped_Location_x_smoothed.values
                        dist_y = goal_loc_map[angle][1] - df.Ped_Location_y_smoothed.values
                        dist_vals = np.sqrt(dist_x ** 2 + dist_y ** 2)[idxs]
                        angle_vals = (goal_deg_map[angle] - df.rot_z.values)[idxs] # in degrees
                        goal_ts = np.concatenate([
                            dist_vals[:, None], np.sin(np.deg2rad(angle_vals[:, None])), np.cos(np.deg2rad(angle_vals[:, None]))
                            ], axis=1).astype(np.float32) # np.sin/cos takes radians instead of degrees
                    else:
                        goal_ts = None
            else:
                pose = sampled[:, :min(6, sampled.shape[1])] if sampled.shape[1] >= 1 else None
                gaze = sampled[:, :min(2, sampled.shape[1])] if sampled.shape[1] >= 2 else None
                motion = sampled[:, :min(2, sampled.shape[1])] if sampled.shape[1] >= 2 else None
                vel = sampled[:, :min(2, sampled.shape[1])] if sampled.shape[1] >= 2 else None
        except Exception:
            pose = None
            gaze = None
            motion = None
            goal_ts = None
            vel = None

        if pose is not None:
            pose = np.asarray(pose, dtype=np.float32)
        if gaze is not None:
            gaze = np.asarray(gaze, dtype=np.float32)
        if motion is not None:
            motion = np.asarray(motion, dtype=np.float32)
        if vel is not None:
            vel = np.asarray(vel, dtype=np.float32)

        if goal_ts is not None:
            goal_ts = np.asarray(goal_ts, dtype=np.float32)

        return pose, gaze, motion, vel, goal_ts

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Return a single sample as a tuple of tensors and an integer label.

        Returns:
            Tuple of ``(vis, pose, gaze, motion, vel, goal, label)`` where each
            modality tensor has shape ``(T, D)`` and ``label`` is an int.
        """
        s = self.samples[idx]
        video_path = s["video_path"]
        csv_path = s["csv_path"]
        video_start_frame = s["video_start_frame"]
        video_end_frame = s["video_end_frame"]
        clip_start_frame = s["clip_start_frame"]
        clip_end_frame = s["clip_end_frame"]
        label = int(s["label"])

        # read scalar modalities first so gaze can be used for gaze-based cropping
        pose, gaze, motion, vel, goal_ts = self._read_vr_csv(csv_path, video_start_frame, video_end_frame)

        # read visual frames only if visual modality is enabled
        if self.visual_encoder is None:
            visual = np.zeros((self.T, 0), dtype=np.float32)
        else:
            visual = self._read_frames(video_path, clip_start_frame, clip_end_frame, csv_path)

        # the current gaze on screen is already linearly interpolated during data preparation
        # apply optional scalers (fit on training set only)
        def _apply_scaler(arr, scaler):
            if arr is None or scaler is None:
                return arr
            mean = scaler.get('mean')
            std = scaler.get('std')
            if mean is None or std is None:
                return arr
            return (arr - mean) / (std + 1e-6)

        if pose is not None and self.pose_scaler is not None:
            try:
                pose = _apply_scaler(pose, self.pose_scaler)
            except Exception:
                pass
        if gaze is not None and self.gaze_scaler is not None:
            try:
                gaze = _apply_scaler(gaze, self.gaze_scaler)
            except Exception:
                pass
        if motion is not None and self.motion_scaler is not None:
            try:
                motion = _apply_scaler(motion, self.motion_scaler)
            except Exception:
                pass
        if vel is not None and self.vel_scaler is not None:
            try:
                vel = _apply_scaler(vel, self.vel_scaler)
            except Exception:
                pass
        if goal_ts is not None and self.goal_dist_scaler is not None:
            try:
                g = goal_ts.copy()
                g[:, 0:1] = _apply_scaler(g[:, 0:1], self.goal_dist_scaler)
                goal_ts = g
            except Exception:
                pass

        vis_t = torch.from_numpy(visual.astype(np.float32))
        pose_t = torch.from_numpy(pose.astype(np.float32)) if pose is not None else torch.empty(self.T, 0, dtype=torch.float32)
        gaze_t = torch.from_numpy(gaze.astype(np.float32)) if gaze is not None else torch.empty(self.T, 0, dtype=torch.float32)
        motion_t = torch.from_numpy(motion.astype(np.float32)) if motion is not None else torch.empty(self.T, 0, dtype=torch.float32)
        vel_t = torch.from_numpy(vel.astype(np.float32)) if vel is not None else torch.empty(self.T, 0, dtype=torch.float32)
        goal_t = torch.from_numpy(goal_ts.astype(np.float32)) if goal_ts is not None else torch.empty(self.T, 0, dtype=torch.float32)

        # ensure at least one modality is present
        modalities_present = sum([
            vis_t.numel() > 0,
            pose_t.numel() > 0,
            gaze_t.numel() > 0,
            motion_t.numel() > 0,
            vel_t.numel() > 0,
            goal_t.numel() > 0,
        ])
        if modalities_present == 0:
            raise RuntimeError("No modalities available: visual disabled and no scalar modalities found. At least one modality is required.")

        return vis_t, pose_t, gaze_t, motion_t, vel_t, goal_t, label

    def get_label_weights_basedon_count(self, n_classes: Optional[int] = None) -> np.ndarray:
        # get counts
        n_class = len(self.label_map.keys())
        assert n_classes == n_class, f"Input n_classes ({n_classes}) must match number of unique labels ({n_class})"
        counts = np.zeros(n_class, dtype=np.int64)
        for s in self.samples:
            lbl = s['label']
            binc = np.bincount([lbl], minlength=n_class)
            counts[: len(binc)] += binc
        
        # inverse frequency, normalized by number of classes so mean weight ~=1
        counts_safe = counts.copy().astype(np.float32)
        counts_safe[counts_safe == 0] = 1.0
        total = int(counts.sum()) if counts.sum() > 0 else 0
        weights = total / (n_class * counts_safe)
        weights[counts == 0] = 0.0
        return weights


def collate_fn(batch: List) -> Tuple:
    """Collate a list of dataset samples into batched tensors.

    Modalities that are empty for all samples in the batch are returned as
    ``None`` so downstream models can detect and skip them.

    Args:
        batch: List of ``(vis, pose, gaze, motion, vel, goal, label)`` tuples
            returned by :class:`VRDataset`.

    Returns:
        Tuple of ``(vis, pose, gaze, motion, vel, goal, labels)`` where each
        modality is a stacked tensor or ``None``, and ``labels`` is a
        ``torch.long`` tensor.
    """
    vis, pose, gaze, motion, vel, goal, labels = zip(*batch)
    vis = torch.stack(vis, dim=0)
    pose = torch.stack(pose, dim=0) if pose[0].numel() > 0 else None
    gaze = torch.stack(gaze, dim=0) if gaze[0].numel() > 0 else None
    motion = torch.stack(motion, dim=0) if motion[0].numel() > 0 else None
    vel = torch.stack(vel, dim=0) if vel[0].numel() > 0 else None
    goal = torch.stack(goal, dim=0) if goal[0].numel() > 0 else None
    labels = torch.tensor(labels, dtype=torch.long)
    return vis, pose, gaze, motion, vel, goal, labels


def parse_person_from_feat_id(fid) -> Optional[str]:
    """Extract a person ID string (e.g. ``'P10'``) from a feature ID entry.

    Args:
        fid: Feature ID value; may be bytes, a string, or any type that can be
            converted to a string.

    Returns:
        The extracted person ID (e.g. ``'P10'``) or ``None`` if not found.
    """
    if fid is None:
        return None
    if isinstance(fid, bytes):
        try:
            fid = fid.decode('utf-8')
        except Exception:
            fid = str(fid)
    if not isinstance(fid, str):
        try:
            # try to stringify (e.g., numpy bytes)
            fid = str(fid)
        except Exception:
            return None
    m = re.search(r'(P\d+)', fid)
    if m:
        return m.group(1)
    # try alternate pattern 'p\d+'
    m = re.search(r'(p\d+)', fid)
    if m:
        return m.group(1).upper()
    return None


def split_person_groups(feat_ids=None, annotation_path=None, seed=42, ratios=(6, 1, 3)):
    """Split persons into train/val/test groups by person-level splitting with given ratios.
    Returns dict with keys 'train','val','test' -> index lists and 'persons' lists.
    """
    if annotation_path:
        # get all feat_ids
        with open(annotation_path, "r") as f:
            raw = json.load(f)
        feat_ids = [entry['video_uid'] for entry in raw]

    if feat_ids is None:
        raise ValueError('Either feat_ids or annotation_path must be provided to split_person_groups')
    
    # extract person id per sample
    persons_per_sample = [parse_person_from_feat_id(x) for x in feat_ids]
    unique_persons = sorted([p for p in set(persons_per_sample) if p is not None])
    if len(unique_persons) == 0:
        raise ValueError('No person ids found in feat_id to perform group split')
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_persons)

    total = sum(ratios)
    n = len(unique_persons)
    n_train = max(1, int(n * (ratios[0] / total)))
    n_val = max(1, int(n * (ratios[1] / total)))
    # ensure sum <= n
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    n_test = n - n_train - n_val

    train_persons = unique_persons[:n_train]
    val_persons = unique_persons[n_train:n_train + n_val]
    test_persons = unique_persons[n_train + n_val:]
    print('Split persons:')
    print(f'  train ({len(train_persons)}): {train_persons}')
    print(f'  val   ({len(val_persons)}): {val_persons}')
    print(f'  test  ({len(test_persons)}): {test_persons}')

    idx_train = [i for i, p in enumerate(persons_per_sample) if p in train_persons]
    idx_val = [i for i, p in enumerate(persons_per_sample) if p in val_persons]
    idx_test = [i for i, p in enumerate(persons_per_sample) if p in test_persons]

    return {
        'train_idx': np.array(idx_train, dtype=int),
        'val_idx': np.array(idx_val, dtype=int),
        'test_idx': np.array(idx_test, dtype=int),
        'train_persons': train_persons,
        'val_persons': val_persons,
        'test_persons': test_persons,
    }


if __name__ == "__main__":
    split_person_groups(
        annotation_path="features/groundvqa/annotations.VRbinary_00000_full_close.json",
        seed=42, ratios=(6, 1, 3))