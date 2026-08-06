"""
Simple gaze overlay script that preserves original video quality by
piping frames to ffmpeg and encoding losslessly (libx264 -crf 0) into MP4.

Usage example:
python dataprep/v3overlay_gaze.py --input_video P1S1-0.mp4 \
    --data_dir_from data/videodata_256/clips \
    --data_dir_to data/videodata_256/clips_overlay \
    --vrdata_dir data/vrdata

Requirements: decord, ffmpeg, numpy, opencv-python, pandas
"""
import os
import tqdm
import argparse
import json
import subprocess
import shlex

import numpy as np
import pandas as pd
import cv2

try:
    from decord import VideoReader, cpu
except Exception:
    VideoReader = None
    cpu = None



################## overlay related functions ##################
def parse_overlay_color(color: str) -> tuple[int, int, int]:
    """Parse an RGB color string and return BGR for OpenCV drawing."""
    r, g, b = [int(x) for x in color.split(",")]
    return (b, g, r)


def draw_gaze_overlay_on_bgr_frame(
    frame,
    gaze,
    frame_idx: int,
    overlay_style: str = "rainbow",
    radius: int = 5,
    color_bgr: tuple[int, int, int] = (0, 0, 255),
    heatmap_alpha: float = 0.3,
    heatmap_radius: int = 30,
    heatmap_sigma: float = 10.0,
):
    """Draw one gaze overlay frame using the same style as the CLI pipeline."""
    out = frame.copy()
    if overlay_style == "dot":
        if frame_idx < len(gaze):
            x, y = gaze[frame_idx]
            if x is not None and y is not None:
                try:
                    if not np.isnan(x) and not np.isnan(y):
                        cv2.circle(out, (int(x), int(y)), radius, color_bgr, -1)
                except Exception:
                    pass
        return out

    if overlay_style in ["rainbow", "bone"]:
        color_map = cv2.COLORMAP_JET if overlay_style == "rainbow" else cv2.COLORMAP_BONE
        gaze_up_to = gaze[: frame_idx + 1]
        saliency = gaze_temporal_map(out.shape, gaze_up_to, r=heatmap_radius, sigma=heatmap_sigma)
        cmap = cv2.applyColorMap(saliency, color_map)
        if out.dtype != cmap.dtype:
            cmap = cmap.astype(out.dtype)
        return cv2.addWeighted(out, 1.0 - heatmap_alpha, cmap, heatmap_alpha, 0)

    raise ValueError(f"Unsupported overlay style: {overlay_style}")


def overlay_frames(
    frames,
    gaze,
    overlay_style: str = "rainbow",
    radius: int = 5,
    color: str = "255,0,0",
    heatmap_alpha: float = 0.3,
    heatmap_radius: int = 30,
    heatmap_sigma: float = 10.0,
):
    """Return RGB frames with gaze overlays drawn on them."""
    color_bgr = parse_overlay_color(color)
    rendered = []
    for i, frame_rgb in enumerate(frames):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        overlaid_bgr = draw_gaze_overlay_on_bgr_frame(
            frame_bgr, gaze, i, overlay_style=overlay_style, radius=radius,
            color_bgr=color_bgr, heatmap_alpha=heatmap_alpha,
            heatmap_radius=heatmap_radius, heatmap_sigma=heatmap_sigma)
        rendered.append(cv2.cvtColor(overlaid_bgr, cv2.COLOR_BGR2RGB))
    return np.asarray(rendered, dtype=np.uint8)


def encode_rgb_frames_to_video(frames, out_path: str, fps: float) -> str:
    """Encode RGB uint8 frames to a lossless H.264 MP4."""
    if len(frames) == 0:
        raise ValueError("Cannot encode an empty frame sequence")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    h, w = frames[0].shape[:2]
    ffmpeg_cmd = (
        f"ffmpeg -y -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - "
        f"-c:v libx264 -preset veryslow -crf 0 -pix_fmt yuv420p {shlex.quote(out_path)}"
    )
    proc = subprocess.Popen(shlex.split(ffmpeg_cmd), stdin=subprocess.PIPE)
    try:
        for frame_rgb in frames:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            proc.stdin.write(frame_bgr.tobytes())
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
    return out_path


def render_sampled_gaze_video(
    frames,
    gaze,
    out_path: str,
    fps: float,
    overlay_style: str = "rainbow",
    radius: int = 5,
    color: str = "255,0,0",
    heatmap_alpha: float = 0.3,
    heatmap_radius: int = 30,
    heatmap_sigma: float = 10.0,
) -> str:
    """Render sampled RGB frames with gaze overlay and encode them as MP4."""
    rendered = overlay_frames(
        frames, gaze, overlay_style=overlay_style, radius=radius, color=color,
        heatmap_alpha=heatmap_alpha, heatmap_radius=heatmap_radius,
        heatmap_sigma=heatmap_sigma)
    return encode_rgb_frames_to_video(rendered, out_path, fps)

def overlay_and_encode(args, gaze):
    video_path = os.path.join(args.data_dir_from, args.input_video)
    if VideoReader is None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {video_path} with decord or OpenCV")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_iter = None
    else:
        vr = VideoReader(video_path, ctx=cpu(0))
        try:
            fps = float(vr.get_avg_fps())
        except Exception:
            fps = 30.0
        sample = vr[0].asnumpy()
        h, w = int(sample.shape[0]), int(sample.shape[1])
        frame_iter = vr

    out_fname = os.path.splitext(args.input_video)[0] + ".mp4"
    out_path = os.path.join(args.data_dir_to, out_fname)
    ffmpeg_cmd = (
        f"ffmpeg -y -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - "
        f"-c:v libx264 -preset veryslow -crf 0 -pix_fmt yuv420p {shlex.quote(out_path)}"
    )

    print("Encoding to:", out_path)
    print("FFmpeg cmd:", ffmpeg_cmd)

    proc = subprocess.Popen(shlex.split(ffmpeg_cmd), stdin=subprocess.PIPE)
    written = 0
    try:
        if frame_iter is None:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = draw_gaze_overlay_on_bgr_frame(
                    frame, gaze, written, overlay_style=args.overlay_style,
                    radius=args.radius, color_bgr=args.color_bgr,
                    heatmap_alpha=args.heatmap_alpha,
                    heatmap_radius=args.heatmap_radius,
                    heatmap_sigma=args.heatmap_sigma)
                proc.stdin.write(frame.tobytes())
                written += 1
            cap.release()
        else:
            for i, fr in enumerate(frame_iter):
                frame = cv2.cvtColor(fr.asnumpy(), cv2.COLOR_RGB2BGR)
                frame = draw_gaze_overlay_on_bgr_frame(
                    frame, gaze, i, overlay_style=args.overlay_style,
                    radius=args.radius, color_bgr=args.color_bgr,
                    heatmap_alpha=args.heatmap_alpha,
                    heatmap_radius=args.heatmap_radius,
                    heatmap_sigma=args.heatmap_sigma)
                proc.stdin.write(frame.tobytes())
                written += 1
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    print(f"Wrote {written} frames to {out_path}")
    return out_path

def overlay_for_single_clip_or_slice(args):
    # ffmpeg expects BGR ordering when we send bgr24 frames
    args.color_bgr = parse_overlay_color(args.color)

    gaze = load_gaze(args)

    _ = overlay_and_encode(args, gaze)


################ saliency map related functions ################
def gaze_temporal_map(image_shape, gaze_points, r=30, sigma=10):
    """
    Temporal salience map: encodes old→new gaze progression.
    """
    H, W = image_shape[:2]
    S = np.zeros((H, W), dtype=np.float32)
    n = len(gaze_points)

    # Weight linearly increases with time (newer = brighter)
    weights = np.linspace(0.1, 1.0, n)

    for i, (x, y) in enumerate(gaze_points):
        if x is not None:
            w = weights[i]
            if np.isnan(x) or np.isnan(y):
                continue
            x, y = int(x), int(y)

            yy, xx = np.mgrid[-r:r+1, -r:r+1]
            mask = w * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

            x1, x2, y1, y2 = x - r, x + r + 1, y - r, y + r + 1
            if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                continue
            
            # Later fixations overwrite earlier ones
            S_patch = S[y1:y2, x1:x2]

            S[y1:y2, x1:x2] = np.maximum(S_patch, mask)

    # Smooth transitions
    ksize = int(2 * sigma + 1) # more constrast
    S = cv2.GaussianBlur(S, (ksize, ksize), sigma)
    S = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX)
    return S.astype(np.uint8)


def gaze_intensity_map(image_shape, gaze_points, r=30, sigma=10):
    """
    Frequency-based salience map: encodes how often gaze falls on each area.
    """
    H, W = image_shape[:2]
    S = np.zeros((H, W), dtype=np.float32)

    for (x, y) in gaze_points:
        if x is not None:
            x, y = int(x), int(y)
            yy, xx = np.mgrid[-r:r+1, -r:r+1]
            mask = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

            x1, x2, y1, y2 = x - r, x + r + 1, y - r, y + r + 1
            if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                continue

            S[y1:y2, x1:x2] += mask  # Additive accumulation
            
    ksize = int(6 * sigma + 1)
    S = cv2.GaussianBlur(S, (ksize, ksize), sigma)
    S = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX)
    return S.astype(np.uint8)


def generate_saliency_map(args):
    gaze = load_gaze(args)
    saliency_map = gaze_temporal_map((args.resize_height, get_resized_width(args)), gaze, )
    # save the saliency map as a grayscale image
    cv2.imwrite(os.path.join(args.data_dir_to, f'{args.input_video}.png'), saliency_map)
    return saliency_map


def get_resized_width(args):
    if args.is_resized:
        ratio = args.original_height / args.resize_height
        return int(args.original_width / ratio)
    else:
        return args.original_width


def load_gaze(args):
    if '-' in args.input_video:
        video_id = args.input_video.split('.')[0]

        # read annotation and find temporal slice
        with open(args.input_annofile, 'r') as f:
            annos = json.load(f)

        matching = [a for a in annos if a.get('video_id') == video_id]
        if len(matching) != 1:
            raise ValueError(f'Expected exactly one annotation for video_id {video_id}, found {len(matching)}')
        anno = matching[0]
        start_f, end_f = anno['video_start_frame'], anno['video_end_frame']

        clip_id = args.input_video.split('-')[0]
        csv_path = os.path.join(args.vrdata_dir, clip_id + '.csv')
        df = pd.read_csv(csv_path)
        gaze = df.iloc[start_f: end_f + 1][['px_u', 'px_v', 'mask_attention']].copy()

    else:
        video_id = args.input_video.split('.')[0]

        df = pd.read_csv(os.path.join(args.vrdata_dir, video_id + '.csv'))
        gaze = df[['px_u', 'px_v', 'mask_attention']].copy()
        
    if args.is_resized:
        ratio = args.original_height / args.resize_height
        gaze['px_u'] = gaze['px_u'] / ratio
        gaze['px_v'] = gaze['px_v'] / ratio
    gaze.loc[~gaze['mask_attention'], ['px_u', 'px_v']] = None  # filter by mask_attention

    return gaze[['px_u', 'px_v']].values.tolist()


def main():
    parser = argparse.ArgumentParser(description='Overlay gaze points and write lossless video')
    parser.add_argument('--gaze_format', type=str, default='overlay', choices=['overlay', 'saliency'])
    parser.add_argument('--overlay_style', type=str, default='rainbow', choices=['dot', 'rainbow', 'bone'],
                        help='When using overlay format, draw a single dot (dot) or a rainbow heatmap (rainbow)')
    parser.add_argument('--data_dir_from', type=str, default='data/videodata_256/full_scale')
    parser.add_argument('--vrdata_dir', type=str, default='data/vrdata')
    parser.add_argument('--input_annofile', type=str, default='features/groundvqa/annotations.VRbinary_00000_full_close.json')
    # parser.add_argument('--input_video', type=str, default='P1S1.mp4')
    parser.add_argument('--input_video', type=str, default='all')
    parser.add_argument('--is_resized', action='store_true', default=True)
    parser.add_argument('--original_height', type=int, default=1068)
    parser.add_argument('--original_width', type=int, default=1536)
    parser.add_argument('--resize_height', type=int, default=256)
    # width=1536, height=1068
    parser.add_argument('--radius', type=int, default=5, help='radius of gaze dot')
    parser.add_argument('--color', type=str, default='255,0,0', help='gaze color as R,G,B')
    # heatmap/blending params for rainbow style
    parser.add_argument('--heatmap_alpha', type=float, default=0.3, help='alpha blending for heatmap overlay (0-1)')
    parser.add_argument('--heatmap_radius', type=int, default=30, help='radius used when creating heatmap blobs')
    parser.add_argument('--heatmap_sigma', type=float, default=10.0, help='sigma used for heatmap Gaussian')
    args = parser.parse_args()

    if args.gaze_format == 'saliency':
        args.data_dir_to = args.data_dir_from + '_saliency'
        os.makedirs(args.data_dir_to, exist_ok=True)
        if args.input_video == 'all':
            # process all videos in the data_dir_from
            # video_files = [f for f in os.listdir(args.data_dir_from) if f.endswith('.mp4')]
            # video_files.sort()
            # video_files = ['P1S1-0.mp4', 'P1S1-1.mp4', 'P1S2-0.mp4', 'P1S2-1.mp4', 'P2S1-0.mp4', 'P2S1-1.mp4', 'P2S2-0.mp4', 'P2S2-1.mp4']
            for vid in tqdm.tqdm(video_files):
                print(f'Processing {vid}...')
                args.input_video = vid
                generate_saliency_map(args)
        else:
            generate_saliency_map(args)
    elif args.gaze_format == 'overlay':
        args.data_dir_to = args.data_dir_from + f'_{args.overlay_style}'
        os.makedirs(args.data_dir_to, exist_ok=True)
        if args.input_video == 'all':
            # process all videos in the data_dir_from
            video_files = [f for f in os.listdir(args.data_dir_from) if f.endswith('.mp4')]
            video_files.sort()
            # video_files = ['P1S1-0.mp4', 'P1S1-1.mp4', 'P1S2-0.mp4', 'P1S2-1.mp4', 'P2S1-0.mp4', 'P2S1-1.mp4', 'P2S2-0.mp4', 'P2S2-1.mp4']
            for vid in tqdm.tqdm(video_files):
                print(f'Processing {vid}...')
                args.input_video = vid
                overlay_for_single_clip_or_slice(args)
        else:
            overlay_for_single_clip_or_slice(args)


if __name__ == '__main__':
    main()