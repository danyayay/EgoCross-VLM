"""
Generate auxiliary task annotation files for pre-training experiments.

Two tasks are generated from the same video clips as the target crossing intention task:

  Task A (aux_vehicle_speed):
    Binary classification of whether the leader vehicle is moving or static
    during the 2-second observation window.
    Labels: 'vehicle moving' / 'vehicle static'

  Task B (aux_vehicle_approach):
    Binary classification of whether the leader vehicle is moving closer to me,
    the ego-pedestrian, during the 2-second observation window.
    Labels: 'vehicle moving closer to me' / 'vehicle moving away from me'

Both tasks reuse the same video clip IDs (video_id = psid-i) so no new video
extraction is needed. The annotation schema matches the existing groundvqa format.

Usage:
    python -m dataprep.d1prepare_annos_aux
        --vrdata_dir data/vrdata
        --json_dir features/groundvqa_auxtask
        [--past_dur_sec 2.0] [--stride 0.5] [--fps 30] [--seed 42]
        [--sample_n 10]
"""

import argparse
import json
import random
from pathlib import Path

from dataprep.d1prepare_annos import filter_csv, get_binary_annotations_at_time


# ---------------------------------------------------------------------------
# Label functions
# ---------------------------------------------------------------------------

def get_leader_speed_label(df, clip_start, clip_end, vel_thr=10.0):
    """Return 'vehicle static' if the leader vehicle barely moved, else 'vehicle moving'.

    Args:
        df: full session DataFrame (from filter_csv)
        clip_start: integer index of first frame (inclusive) of observation window
        clip_end: integer index of last frame (exclusive) of observation window
        vel_thr: velocity threshold in cm/s; max velocity below this → static
    """
    leader_vel = df['leader_vel_x'].iloc[clip_start:clip_end]
    if leader_vel.isnull().all():
        return 'vehicle moving'  # fallback: treat unknown as moving
    max_vel = leader_vel.max()
    return 'vehicle static' if max_vel < vel_thr else 'vehicle moving'


def get_leader_approach_label(df, clip_start, clip_end, vel_thr=10.0):
    """Return 'vehicle moving closer to me' if the leader vehicle moves toward the pedestrian.

    Approaching = vehicle has some velocity AND the distance to the pedestrian
    is decreasing over the observation window.

    Args:
        df: full session DataFrame (from filter_csv)
        clip_start: integer index of first frame (inclusive) of observation window
        clip_end: integer index of last frame (exclusive) of observation window
        vel_thr: minimum mean velocity (cm/s) to consider vehicle as moving
    """
    leader_vel = df['leader_vel_x'].iloc[clip_start:clip_end]
    leader_loc = df['leader_loc_x'].iloc[clip_start:clip_end]
    ped_loc = df['ped_loc_x'].iloc[clip_start:clip_end]

    if leader_vel.isnull().all() or leader_loc.isnull().all():
        return 'vehicle moving away from me'  # fallback

    mean_vel = leader_vel.mean()
    if mean_vel < vel_thr:
        return 'vehicle moving away from me'  # vehicle is nearly static

    # Distance: |leader_loc_x - ped_loc_x|
    dist = (leader_loc - ped_loc).abs()
    dist_start = dist.iloc[0]
    dist_end = dist.iloc[-1]
    return 'vehicle moving closer to me' if dist_end < dist_start else 'vehicle moving away from me'


def get_ped_moving_label(df, clip_start, clip_end, vel_thr=30.0):
    """Return 'walking' if the pedestrian is moving, else 'standing still'.

    Args:
        df: full session DataFrame (from filter_csv)
        clip_start: integer index of first frame (inclusive) of observation window
        clip_end: integer index of last frame (exclusive) of observation window
        vel_thr: mean velocity threshold in cm/s; above this → walking
    """
    mean_vel = df['ped_vel'].iloc[clip_start:clip_end].mean()
    return 'walking' if mean_vel >= vel_thr else 'standing still'


def get_ped_direction_label(df, clip_start, clip_end, vel_thr=30.0):
    """Return 'moving toward the crossing' if the pedestrian approaches the crossing center.

    Args:
        df: full session DataFrame (from filter_csv)
        clip_start: integer index of first frame (inclusive) of observation window
        clip_end: integer index of last frame (exclusive) of observation window
        vel_thr: minimum mean velocity to count as moving (cm/s)
    """
    ped_vel = df['ped_vel'].iloc[clip_start:clip_end]
    ped_loc_y = df['ped_loc_y'].iloc[clip_start:clip_end].abs()
    if ped_vel.mean() < vel_thr:
        return 'moving away from the crossing'
    return 'moving toward the crossing' if ped_loc_y.iloc[-1] < ped_loc_y.iloc[0] else 'moving away from the crossing'


def get_gaze_vehicle_label(df, clip_start, clip_end, min_gaze_frames=3):
    """Return 'looking at the vehicle' if gaze hits the leader vehicle for enough frames.

    Args:
        df: full session DataFrame (from filter_csv)
        clip_start: integer index of first frame (inclusive) of observation window
        clip_end: integer index of last frame (exclusive) of observation window
        min_gaze_frames: minimum number of valid gaze-on-vehicle frames required
    """
    valid = df['mask_attention'].iloc[clip_start:clip_end].astype(str).str.lower() == 'true'
    on_pod = df['hit_object'].iloc[clip_start:clip_end] == 'pod_leader'
    return 'looking at the vehicle' if (valid & on_pod).sum() >= min_gaze_frames else 'not looking at the vehicle'


def get_ehmi_label(df, clip_start, clip_end):
    """Return the EHMI signal label for the leader vehicle.

    green = shuttle yields to pedestrian; red = shuttle does NOT yield.
    Returns None if no EHMI data is present (skip this clip).

    Args:
        df: full session DataFrame (from filter_csv)
        clip_start: integer index of first frame (inclusive) of observation window
        clip_end: integer index of last frame (exclusive) of observation window
    """
    ehmi = df['leader_ehmi'].iloc[clip_start:clip_end]
    ehmi_str = ehmi.astype(str).str.strip()
    nonempty = ehmi[~ehmi_str.isin(['', 'nan', 'NaN', 'None'])]
    if nonempty.empty:
        return None
    mode_result = nonempty.mode()
    if mode_result.empty:
        return None
    val = mode_result.iloc[0]
    if val == 'green':
        return 'the shuttle will yield to me'
    if val == 'red':
        return 'the shuttle will not yield to me'
    return None


def get_head_turning_label(df, clip_start, clip_end, turn_thr=20.0):
    """Return head turning direction based on mean head angular velocity.

    Args:
        df: full session DataFrame (from filter_csv)
        clip_start: integer index of first frame (inclusive) of observation window
        clip_end: integer index of last frame (exclusive) of observation window
        turn_thr: angular velocity threshold in deg/s; below this → looking straight ahead
    """
    head_vel = df['head_vel_rel'].iloc[clip_start:clip_end]
    mean_vel = head_vel.mean()
    if abs(mean_vel) < turn_thr:
        return 'looking straight ahead'
    return 'turning head right' if mean_vel > 0 else 'turning head left'


def get_vehicle_proximity_label(df, clip_start, clip_end, close_thr_cm=500.0):
    """Return 'close' if the leader vehicle is within threshold distance of pedestrian.

    Returns None if no leader vehicle data is present.

    Args:
        df: full session DataFrame (from filter_csv)
        clip_start: integer index of first frame (inclusive) of observation window
        clip_end: integer index of last frame (exclusive) of observation window
        close_thr_cm: distance threshold in cm; below this → close
    """
    leader_loc = df['leader_loc_x'].iloc[clip_start:clip_end]
    ped_loc = df['ped_loc_x'].iloc[clip_start:clip_end]
    if leader_loc.isnull().all():
        return None
    rel_dist = (leader_loc - ped_loc).abs().mean()
    return 'close (within 5 meters)' if rel_dist < close_thr_cm else 'far (more than 5 meters away)'


def get_crossing_proximity_label(df, clip_start, clip_end, close_thr_cm=200.0):
    """Return 'close to the crossing point' if pedestrian is near the crossing center.

    Args:
        df: full session DataFrame (from filter_csv)
        clip_start: integer index of first frame (inclusive) of observation window
        clip_end: integer index of last frame (exclusive) of observation window
        close_thr_cm: distance threshold in cm; below this → close
    """
    mean_dist = df['dist_ped2center'].iloc[clip_start:clip_end].mean()
    return 'close to the crossing point' if mean_dist < close_thr_cm else 'far from the crossing point'


# ---------------------------------------------------------------------------
# Annotation generation per session
# ---------------------------------------------------------------------------

def _make_anno(psid, clip_uid, clip_start, clip_end, past_dur_sec, fps, label, all_classes, rng, task_name=None):
    """Build a single groundvqa-format annotation dict."""
    wrong = [c for c in all_classes if c != label]
    options = [label] + wrong
    rng.shuffle(options)
    anno = {
        "video_uid": psid,
        "video_start_sec": clip_start / fps,
        "video_end_sec": clip_end / fps,
        "video_start_frame": clip_start,
        "video_end_frame": clip_end,
        "video_id": clip_uid,
        "sample_id": clip_uid,
        "task": {"name": task_name or "aux_unknown", "future_sec": 1.0},
        "answer": label,
        "moment_start_frame": 0,
        "moment_end_frame": clip_end - clip_start,
        "wrong_answers": wrong,
    }
    return anno


def get_aux_annotations_for_session(
        vrdata_dir, psid,
        past_dur_sec=2.0, fut_dur_sec=1.0, stride=0.5, fps=30,
        seed=42, sample_n=None):
    """Generate auxiliary task annotations for one session.

    Returns a dict of task_name → list of annotation dicts in groundvqa format.
    """
    rng = random.Random(seed)

    df = filter_csv(Path(vrdata_dir), psid)
    total_frames = len(df)
    past_frames = int(past_dur_sec * fps)
    fut_frames = int(fut_dur_sec * fps)
    stride_frames = int(stride * fps)

    vehicle_speed_classes = ['vehicle moving', 'vehicle static']
    vehicle_approach_classes = ['vehicle moving closer to me', 'vehicle moving away from me']
    ped_moving_classes = ['walking', 'standing still']
    ped_direction_classes = ['moving toward the crossing', 'moving away from the crossing']
    gaze_vehicle_classes = ['looking at the vehicle', 'not looking at the vehicle']
    ehmi_classes = ['the shuttle will yield to me', 'the shuttle will not yield to me']
    head_turning_classes = ['turning head left', 'turning head right', 'looking straight ahead']
    vehicle_proximity_classes = ['close (within 5 meters)', 'far (more than 5 meters away)']
    crossing_proximity_classes = ['close to the crossing point', 'far from the crossing point']

    annos = {
        'vehicle_speed': [],
        'vehicle_approach': [],
        'ped_moving': [],
        'ped_direction': [],
        'gaze_vehicle': [],
        'ehmi': [],
        'head_turning': [],
        'vehicle_proximity': [],
        'crossing_proximity': [],
    }

    clip_idx = 0
    is_recovered_crossing = False
    for clip_start in range(0, total_frames - past_frames - fut_frames + 1, stride_frames):
        if sample_n is not None and clip_idx >= sample_n:
            break
        clip_end = clip_start + past_frames

        # Mirror target task: stop after first crossing following a pod pass
        is_last_pod_pass = df['is_last_pod_pass'].iloc[clip_start:clip_end].any()
        if is_last_pod_pass:
            if is_recovered_crossing:
                break
            clip_pred = clip_end + fut_frames
            fut_state = get_binary_annotations_at_time(
                df['ped_vel'].iloc[clip_end:clip_pred],
                df['ped_loc_y'].iloc[clip_end:clip_pred],
                df['time'].iloc[clip_end:clip_pred],
            )
            if fut_state == 'cross':
                is_recovered_crossing = True

        clip_uid = f"{psid}-{clip_idx}"
        clip_idx += 1

        # --- Task: vehicle speed ---
        speed_label = get_leader_speed_label(df, clip_start, clip_end)
        annos['vehicle_speed'].append(_make_anno(
            psid, clip_uid, clip_start, clip_end, past_dur_sec, fps,
            speed_label, vehicle_speed_classes, rng, task_name='aux_vehicle_speed'))

        # --- Task: vehicle approach ---
        approach_label = get_leader_approach_label(df, clip_start, clip_end)
        annos['vehicle_approach'].append(_make_anno(
            psid, clip_uid, clip_start, clip_end, past_dur_sec, fps,
            approach_label, vehicle_approach_classes, rng, task_name='aux_vehicle_approach'))

        # --- Task: pedestrian moving ---
        ped_moving_label = get_ped_moving_label(df, clip_start, clip_end)
        annos['ped_moving'].append(_make_anno(
            psid, clip_uid, clip_start, clip_end, past_dur_sec, fps,
            ped_moving_label, ped_moving_classes, rng, task_name='aux_ped_moving'))

        # --- Task: pedestrian direction ---
        ped_direction_label = get_ped_direction_label(df, clip_start, clip_end)
        annos['ped_direction'].append(_make_anno(
            psid, clip_uid, clip_start, clip_end, past_dur_sec, fps,
            ped_direction_label, ped_direction_classes, rng, task_name='aux_ped_direction'))

        # --- Task: gaze on vehicle ---
        gaze_label = get_gaze_vehicle_label(df, clip_start, clip_end)
        annos['gaze_vehicle'].append(_make_anno(
            psid, clip_uid, clip_start, clip_end, past_dur_sec, fps,
            gaze_label, gaze_vehicle_classes, rng, task_name='aux_gaze_vehicle'))

        # --- Task: EHMI signal (skip if no EHMI data) ---
        ehmi_label = get_ehmi_label(df, clip_start, clip_end)
        if ehmi_label is not None:
            annos['ehmi'].append(_make_anno(
                psid, clip_uid, clip_start, clip_end, past_dur_sec, fps,
                ehmi_label, ehmi_classes, rng, task_name='aux_ehmi'))

        # --- Task: head turning ---
        head_label = get_head_turning_label(df, clip_start, clip_end)
        annos['head_turning'].append(_make_anno(
            psid, clip_uid, clip_start, clip_end, past_dur_sec, fps,
            head_label, head_turning_classes, rng, task_name='aux_head_turning'))

        # --- Task: vehicle proximity (skip if no leader data) ---
        veh_prox_label = get_vehicle_proximity_label(df, clip_start, clip_end)
        if veh_prox_label is not None:
            annos['vehicle_proximity'].append(_make_anno(
                psid, clip_uid, clip_start, clip_end, past_dur_sec, fps,
                veh_prox_label, vehicle_proximity_classes, rng, task_name='aux_vehicle_proximity'))

        # --- Task: crossing proximity ---
        cross_prox_label = get_crossing_proximity_label(df, clip_start, clip_end)
        annos['crossing_proximity'].append(_make_anno(
            psid, clip_uid, clip_start, clip_end, past_dur_sec, fps,
            cross_prox_label, crossing_proximity_classes, rng, task_name='aux_crossing_proximity'))

        print(f"  {clip_uid}: vehicle_speed={speed_label}, vehicle_approach={approach_label}, "
              f"ped_moving={ped_moving_label}, ped_dir={ped_direction_label}, "
              f"gaze={gaze_label}, head={head_label}, "
              f"veh_prox={veh_prox_label}, cross_prox={cross_prox_label}")

    return annos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate auxiliary task annotations for all subtasks')
    parser.add_argument('--vrdata_dir', type=str, default='data/vrdata')
    parser.add_argument('--json_dir', type=str, default='features/groundvqa_auxtask',
                        help='Output directory for all annotation files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for option order shuffling')
    parser.add_argument('--psid', type=str, default=None,
                        help='If provided, only process this single session (for testing)')
    parser.add_argument('--sample_n', type=int, default=None,
                        help='Debug mode: stop after generating this many base clips')
    
    parser.add_argument('--past_dur_sec', type=float, default=2.0)
    parser.add_argument('--fut_dur_sec', type=float, default=1.0,
                        help='Future window in seconds (must match target task, default: 1.0)')
    parser.add_argument('--stride', type=float, default=0.5,
                        help='Stride in seconds between clips (default: 0.5, same as target task)')
    parser.add_argument('--fps', type=int, default=30)
    
    args = parser.parse_args()

    vrdata_dir = Path(args.vrdata_dir)
    json_dir = Path(args.json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)

    # Collect all session IDs
    if args.psid:
        psids = [args.psid]
    else:
        psids = sorted([f.stem for f in vrdata_dir.glob('P*.csv')])

    all_annos = {
        'vehicle_speed': [], 'vehicle_approach': [], 'ped_moving': [], 'ped_direction': [],
        'gaze_vehicle': [], 'ehmi': [], 'head_turning': [],
        'vehicle_proximity': [], 'crossing_proximity': [],
    }

    for psid in psids:
        print(f"Processing {psid} ...")
        try:
            remaining = None
            if args.sample_n is not None:
                remaining = args.sample_n - len(all_annos['vehicle_speed'])
                if remaining <= 0:
                    break
            session_annos = get_aux_annotations_for_session(
                vrdata_dir, psid,
                past_dur_sec=args.past_dur_sec,
                fut_dur_sec=args.fut_dur_sec,
                stride=args.stride,
                fps=args.fps,
                seed=args.seed,
                sample_n=remaining,
            )
            for task, annos in session_annos.items():
                all_annos[task].extend(annos)
            if args.sample_n is not None and len(all_annos['vehicle_speed']) >= args.sample_n:
                print(f"Reached --sample_n={args.sample_n}; stopping auxiliary generation early.")
                break
        except Exception as e:
            print(f"  ERROR processing {psid}: {e}")
            continue

    # Save all annotation files
    from collections import Counter
    for task, annos in all_annos.items():
        out_path = json_dir / f'annotations.VRbinary__aux_{task}__full_close.json'
        with open(out_path, 'w') as f:
            json.dump(annos, f, indent=2)
        dist = Counter(a['answer'] for a in annos)
        print(f"Wrote {len(annos):5d} {task} annotations to {out_path}")
        print(f"  Label distribution: {dict(dist)}")


if __name__ == '__main__':
    main()
