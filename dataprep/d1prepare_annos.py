from html import parser
import os
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


def write_json_with_progress(data, json_path, desc=None, indent=4):
    """Write JSON while showing progress for large generated annotation files."""
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    desc = desc or f"Writing {path.name}"
    with path.open('w', encoding='utf-8') as f:
        if isinstance(data, list):
            f.write('[\n')
            for i, item in enumerate(tqdm(data, desc=desc, unit='item')):
                if i:
                    f.write(',\n')
                text = json.dumps(item, ensure_ascii=False, indent=indent)
                f.write(' ' * indent + text.replace('\n', '\n' + ' ' * indent))
            f.write('\n]')
            return

        encoder = json.JSONEncoder(ensure_ascii=False, indent=indent)
        for chunk in tqdm(encoder.iterencode(data), desc=desc, unit='chunk'):
            f.write(chunk)


def filter_csv(vrdata_dir, psid):
    df = pd.read_csv(vrdata_dir / f"{psid}.csv")

    # filter out if all pods have already passed, then no conflicts are present
    # important for "yielding" scenarios
    df['is_last_pod_pass'] = (((df['ped_loc_x'] - df['leader_loc_x']) < 0) & (df['ped_loc_y'] > 0)).astype(bool)
    if not df['follower_loc_x'].isnull().all():
        df['is_last_pod_pass'] = (((df['ped_loc_x'] - df['follower_loc_x']) < 0) & (df['ped_loc_y'] > 0)).astype(bool) & df['is_last_pod_pass']
    
    idx_pod_pass = df[df.is_last_pod_pass == True].first_valid_index()
    if df['is_last_pod_pass'].any():
        idx_pass_conflict = df[df['ped_loc_y'] < 0].first_valid_index()
        assert idx_pass_conflict > idx_pod_pass, f"Pod pass index {idx_pod_pass} should be before conflict pass index {idx_pass_conflict}"
        print('Filtering out', df.iloc[idx_pass_conflict].shape[0], 'frames due to pods passed, at index', idx_pass_conflict)
        df = df.iloc[:idx_pass_conflict] 
    
    # filter out if the person has already resolved the conflicts by arriving the other side
    # important for "crossing" scenarios
    df['is_resolved'] = ((df['ped_loc_y']) < -100).astype(bool)
    idx_resolved = df[df.is_resolved == True].first_valid_index()
    if idx_resolved:
        print('Filtered out', df.iloc[idx_resolved:].shape[0], 'frames due to resolved conflicts, at index', idx_resolved)
        df = df.iloc[:idx_resolved]

    return df


def get_video_fullscale_annotation(vrdata_dir, psid):
    df = pd.read_csv(vrdata_dir / f"{psid}.csv")
    start_sec, start_frame = 0, 0
    end_sec = float(df['video_time'].iloc[-1])
    end_frame = int(df['frame'].iloc[-1])

    groundvqa_anno = [{
        "video_uid": psid,
        "video_start_sec": start_sec,
        "video_end_sec": end_sec,
        "video_start_frame": start_frame,
        "video_end_frame": end_frame,
        "video_id": psid,
        "sample_id": psid,
        "question": None,
        "answer": None,
        "moment_start_frame": start_frame,
        "moment_end_frame": end_frame,
        "wrong_answers": None
    }]
    
    egovlp_anno = [{
        "clip_uid": psid,
        "video_start_sec": start_sec,
        "video_end_sec": end_sec,
        "video_start_frame": start_frame,
        "video_end_frame": end_frame,
        "clip_start_sec": start_sec,
        "clip_end_sec": end_sec,
        "clip_start_frame": start_frame,
        "clip_end_frame": end_frame,
        "source_clip_uid": psid,
        "annotations": [
            {
                "language_queries": [
                    {
                        "clip_start_sec": start_sec,
                        "clip_end_sec": end_sec,
                        "video_start_sec": start_sec,
                        "video_end_sec": end_sec,
                        "video_start_frame": start_frame,
                        "video_end_frame": end_frame,
                        "template": None,
                        "query": None,
                    }
                ],
                "annotation_uid": psid,
                "request_uid": psid
            }
        ]
    }]
    egovlp_anno = {
        "video_uid": psid,
        "clips": egovlp_anno
    }
    return groundvqa_anno, egovlp_anno


def extract_extra_features_for_frame(df, frame_idx, last_frame_idx, dts_df=None, psid_str=None, requested=None):
    """Extract features at a specific frame index from the dataframe.
    requested: list of feature names to extract (strings)
    returns dict of feature_name -> value (or None)"""
    if requested is None:
        return {}
    row = df.iloc[frame_idx]
    time = df['time'].iloc[frame_idx] - df['time'].iloc[last_frame_idx]
    features = {'time': round(time, 4)}

    # Location: x,y
    if 'location' in requested:
        loc = [row['ped_loc_x'], row['ped_loc_y']]
        features['location'] = {'x': round(loc[0], 4), 'y': round(loc[1], 4)} if loc is not None else None

    # Head pose: yaw, pitch, roll (best-effort)
    if 'head_pose' in requested:
        features['head_pose'] = {'yaw': round(row['head_rot_z'], 4)}

    # Gaze orientation: try vector components or angles
    if 'gaze_orientation' in requested:
        features['gaze_orientation'] = {'yaw': round(row['eye_rot_z'], 4)}

    # Gaze on screen: direct column or heuristic (best-effort)
    if 'gaze_on_screen' in requested:
        eos = [row['px_u'], row['px_v']] # TODO: transform to normalized & scaled screen coordinates if needed
        features['gaze_on_screen'] = {'x': round(eos[0], 4), 'y': round(eos[1], 4)}

    # Demographics: lookup from dts_df using psid or pid/sid
    demo = None
    if 'demographics' in requested:
        if psid_str is not None:
            # parse numbers from P10S1 style
            import re
            pid, sid = re.match(r'P(\d+)S(\d+)', psid_str).groups()
            # try common column names
            row_demo = dts_df[(dts_df['pid'] == int(pid)) & (dts_df['sid'] == int(sid))]
            if not row_demo.empty:
                demo = row_demo.iloc[0].to_dict()
        # mapping
        gender_map = {0: 'male', 1: 'female'}
        education_map = {0: "High school or equivalent", 1: "Associate degree or equivalent", 
                            2: "Bachelor’s degree or equivalent", 3: "Master’s degree or equivalent", 
                            4: "Doctoral degree or equivalent"}
        dominant_hand_map = {0: "left hand", 1: "right hand"}
        walk_frequency_map = {0: "never", 1: "rarely (less than 1 day per week)", 
                                2: "1 day per week", 3: "2 to 3 days per week", 
                                4: "4 to 5 days per week", 5: "every day (7 days)"}
        vr_frequency_map = {0: "never", 1: "seldom", 2: "sometimes", 3: "often", 4: "very often"}
        as_familiarity_map = {0: 'not at all', 1: 'slightly familiar', 2: 'moderate familiar', 
                                3: 'very familiar', 4: 'extremely familiar'}
        as_experience_map = {0: 'no experience', 1: 'little experience', 2: 'some experience', 
                                3: 'a lot of experience', 4: 'extensive experience'}
        # keep only these fields with mapping
        if demo is not None:
            demo = {
                'age': int(demo['age']) if 'age' in demo else None,
                'gender': gender_map.get(int(demo['gender'])) if 'gender' in demo else None,
                'highest education level': education_map.get(int(demo['education'])) if 'education' in demo else None,
                'dominant hand in daily life': dominant_hand_map.get(int(demo['dominant_hand'])) if 'dominant_hand' in demo else None,
                'walk frequency in daily life': walk_frequency_map.get(int(demo['walk_frequency'])) if 'walk_frequency' in demo else None,
                'VR frequency': vr_frequency_map.get(int(demo['vr_frequency'])) if 'vr_frequency' in demo else None,
                'familiarity with automated shuttles': as_familiarity_map.get(int(demo['as_familiarity'])) if 'as_familiarity' in demo else None,
                'previous experience with automated shuttles': as_experience_map.get(int(demo['as_experience'])) if 'as_experience' in demo else None,
                'trust score': float(demo['trust_score']) if 'trust_score' in demo and not pd.isna(demo['trust_score']) else None,
                'violation score': float(demo['v_score']) if 'v_score' in demo and not pd.isna(demo['v_score']) else None,
                'lapse score': float(demo['l_score']) if 'l_score' in demo and not pd.isna(demo['l_score']) else None,
                'positive behavior score': float(demo['p_score']) if 'p_score' in demo and not pd.isna(demo['p_score']) else None
            }

    return features, demo


def extract_extra_features_for_frame_as_tuple(df, frame_idx, last_frame_idx, dts_df=None, psid_str=None, requested=None):
    """Extract features at a specific frame index from the dataframe.
    requested: list of feature names to extract (strings)
    returns dict of feature_name -> value (or None)"""
    if requested is None:
        return {}
    row = df.iloc[frame_idx]
    time = df['time'].iloc[frame_idx] - df['time'].iloc[last_frame_idx]
    features = (round(time, 4),)

    # Location: x,y
    if 'location' in requested:
        loc = [row['ped_loc_x'], row['ped_loc_y']]
        features += ( {'x': round(loc[0]), 'y': round(loc[1])} if loc is not None else None,)

    # Head pose: yaw, pitch, roll (best-effort)
    if 'head_pose' in requested:
        features += ( {'yaw': round(row['head_rot_z'], 2)},)

    # Gaze orientation: try vector components or angles
    if 'gaze_orientation' in requested:
        features += ( {'yaw': round(row['eye_rot_z'], 4)},) # if row['mask_attention'] else (None,)

    # Gaze on screen: direct column or heuristic (best-effort)
    if 'gaze_on_screen' in requested:
        # ratio = 1068/256
        eos = [row['px_u'], row['px_v']] # TODO: transform to normalized & scaled screen coordinates if needed
        features += ( {'x': round(eos[0], 4), 'y': round(eos[1], 4)} if eos is not None else None,) # if eos is not None and row['mask_attention'] else None,)

    # Demographics: lookup from dts_df using psid or pid/sid
    demo, demo_str = None, None
    if 'demographics' in requested:
        if psid_str is not None:
            # parse numbers from P10S1 style
            import re
            pid, sid = re.match(r'P(\d+)S(\d+)', psid_str).groups()
            # try common column names
            row_demo = dts_df[(dts_df['pid'] == int(pid)) & (dts_df['sid'] == int(sid))]
            if not row_demo.empty:
                demo = row_demo.iloc[0].to_dict()
        # mapping
        gender_map = {0: 'male', 1: 'female'}
        education_map = {0: "High school or equivalent", 1: "Associate degree or equivalent", 
                            2: "Bachelor degree or equivalent", 3: "Master degree or equivalent", 
                            4: "Doctoral degree or equivalent"}
        dominant_hand_map = {0: "left hand", 1: "right hand"}
        walk_frequency_map = {0: "never", 1: "rarely (less than 1 day per week)", 
                                2: "1 day per week", 3: "2 to 3 days per week", 
                                4: "4 to 5 days per week", 5: "every day"}
        vr_frequency_map = {0: "never", 1: "seldom", 2: "sometimes", 3: "often", 4: "very often"}
        as_familiarity_map = {0: 'not at all', 1: 'slightly familiar', 2: 'moderate familiar', 
                                3: 'very familiar', 4: 'extremely familiar'}
        as_experience_map = {0: 'no experience', 1: 'little experience', 2: 'some experience', 
                                3: 'a lot of experience', 4: 'extensive experience'}
        # keep only these fields with mapping
        if demo is not None:
            demo = {
                'age': int(demo['age']) if 'age' in demo else None,
                'gender': gender_map.get(int(demo['gender'])) if 'gender' in demo else None,
                'highest education level': education_map.get(int(demo['education'])) if 'education' in demo else None,
                'dominant hand in daily life': dominant_hand_map.get(int(demo['dominant_hand'])) if 'dominant_hand' in demo else None,
                'walk frequency in daily life': walk_frequency_map.get(int(demo['walk_frequency'])) if 'walk_frequency' in demo else None,
                'VR frequency': vr_frequency_map.get(int(demo['vr_frequency'])) if 'vr_frequency' in demo else None,
                'familiarity with automated shuttles': as_familiarity_map.get(int(demo['as_familiarity'])) if 'as_familiarity' in demo else None,
                'previous experience with automated shuttles': as_experience_map.get(int(demo['as_experience'])) if 'as_experience' in demo else None,
                'trust score': float(demo['trust_score']) if 'trust_score' in demo and not pd.isna(demo['trust_score']) else None,
                'violation score': float(demo['v_score']) if 'v_score' in demo and not pd.isna(demo['v_score']) else None,
                'lapse score': float(demo['l_score']) if 'l_score' in demo and not pd.isna(demo['l_score']) else None,
                'positive behavior score': float(demo['p_score']) if 'p_score' in demo and not pd.isna(demo['p_score']) else None
            }
            demo_str = f'You are a {demo["gender"]} of {demo["age"]} years old, with {demo["highest education level"]} education level. ' #+\
                # f'Your dominant hand in daily life is {demo["dominant hand in daily life"]}. ' +\
                # f'Your walk frequency in daily life is {demo["walk frequency in daily life"]}. ' +\
                # f'Your VR frequency is {demo["VR frequency"]}. ' +\
                # f'Your familiarity with automated shuttles is {demo["familiarity with automated shuttles"]}, ' +\
                # f'and your previous experience with automated shuttles is {demo["previous experience with automated shuttles"]}.'

    return features, demo_str


def _json_clean(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 4)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _safe_col(row, name):
    return _json_clean(row[name]) if name in row.index else None


def _stream_from_rows(df_clip, source_fps, columns, builders):
    samples = []
    if df_clip.empty:
        return {
            "source_fps": source_fps,
            "window_start_sec": 0.0,
            "window_end_sec": 0.0,
            "columns": columns,
            "samples": samples,
        }
    t0 = float(df_clip['time'].iloc[0]) if 'time' in df_clip else 0.0
    for _, row in df_clip.iterrows():
        t = float(row['time'] - t0) if 'time' in row.index else len(samples) / float(source_fps)
        item = {"t": round(t, 4)}
        for key, fn in builders.items():
            item[key] = fn(row)
        samples.append(item)
    return {
        "source_fps": source_fps,
        "window_start_sec": 0.0,
        "window_end_sec": round(samples[-1]["t"], 4) if samples else 0.0,
        "columns": columns,
        "samples": samples,
    }


def _demographics_value(dts_df=None, psid_str=None):
    if dts_df is None or psid_str is None:
        return None
    try:
        import re
        pid, sid = re.match(r'P(\d+)S(\d+)', psid_str).groups()
        row_demo = dts_df[(dts_df['pid'] == int(pid)) & (dts_df['sid'] == int(sid))]
        if row_demo.empty:
            return None
        demo = row_demo.iloc[0].to_dict()
    except Exception:
        return None
    gender_map = {0: 'male', 1: 'female'}
    education_map = {0: "High school or equivalent", 1: "Associate degree or equivalent",
                     2: "Bachelor degree or equivalent", 3: "Master degree or equivalent",
                     4: "Doctoral degree or equivalent"}
    dominant_hand_map = {0: "left hand", 1: "right hand"}
    walk_frequency_map = {0: "never", 1: "rarely (less than 1 day per week)",
                          2: "1 day per week", 3: "2 to 3 days per week",
                          4: "4 to 5 days per week", 5: "every day"}
    vr_frequency_map = {0: "never", 1: "seldom", 2: "sometimes", 3: "often", 4: "very often"}
    as_familiarity_map = {0: 'not at all', 1: 'slightly familiar', 2: 'moderate familiar',
                          3: 'very familiar', 4: 'extremely familiar'}
    as_experience_map = {0: 'no experience', 1: 'little experience', 2: 'some experience',
                         3: 'a lot of experience', 4: 'extensive experience'}
    return {
        'age': int(demo['age']) if 'age' in demo and not pd.isna(demo['age']) else None,
        'gender': gender_map.get(int(demo['gender'])) if 'gender' in demo and not pd.isna(demo['gender']) else None,
        'highest education level': education_map.get(int(demo['education'])) if 'education' in demo and not pd.isna(demo['education']) else None,
        'dominant hand in daily life': dominant_hand_map.get(int(demo['dominant_hand'])) if 'dominant_hand' in demo and not pd.isna(demo['dominant_hand']) else None,
        'walk frequency in daily life': walk_frequency_map.get(int(demo['walk_frequency'])) if 'walk_frequency' in demo and not pd.isna(demo['walk_frequency']) else None,
        'VR frequency': vr_frequency_map.get(int(demo['vr_frequency'])) if 'vr_frequency' in demo and not pd.isna(demo['vr_frequency']) else None,
        'familiarity with automated shuttles': as_familiarity_map.get(int(demo['as_familiarity'])) if 'as_familiarity' in demo and not pd.isna(demo['as_familiarity']) else None,
        'previous experience with automated shuttles': as_experience_map.get(int(demo['as_experience'])) if 'as_experience' in demo and not pd.isna(demo['as_experience']) else None,
        'trust score': float(demo['trust_score']) if 'trust_score' in demo and not pd.isna(demo['trust_score']) else None,
        'violation score': float(demo['v_score']) if 'v_score' in demo and not pd.isna(demo['v_score']) else None,
        'lapse score': float(demo['l_score']) if 'l_score' in demo and not pd.isna(demo['l_score']) else None,
        'positive behavior score': float(demo['p_score']) if 'p_score' in demo and not pd.isna(demo['p_score']) else None,
    }


def build_clip_features(df_clip, fps, dts_df=None, psid_str=None):
    """Build structured, native-rate clip facts for prompt-time selection."""
    if df_clip.empty:
        return {}
    motion = _stream_from_rows(
        df_clip, fps,
        [
            'ped_loc_x', 'ped_loc_y', 'ped_vel', 'ped_vel_x', 'ped_vel_y',
            'leader_loc_x', 'leader_vel_x', 'follower_loc_x', 'dist_ped2center'
        ],
        {
            'ped_x': lambda r: _safe_col(r, 'ped_loc_x'),
            'ped_y': lambda r: _safe_col(r, 'ped_loc_y'),
            'ped_speed': lambda r: _safe_col(r, 'ped_vel'),
            'ped_vx': lambda r: _safe_col(r, 'ped_vel_x'),
            'ped_vy': lambda r: _safe_col(r, 'ped_vel_y'),
            'leader_x': lambda r: _safe_col(r, 'leader_loc_x'),
            'leader_vx': lambda r: _safe_col(r, 'leader_vel_x'),
            'follower_x': lambda r: _safe_col(r, 'follower_loc_x'),
            'dist_to_crossing': lambda r: _safe_col(r, 'dist_ped2center'),
            'leader_rel_x': lambda r: _json_clean(r['leader_loc_x'] - r['ped_loc_x']) if {'leader_loc_x', 'ped_loc_x'}.issubset(r.index) else None,
        })
    gaze_direction = _stream_from_rows(
        df_clip, fps, ['eye_rot_z'], {'yaw': lambda r: _safe_col(r, 'eye_rot_z')})
    gaze_on_screen = _stream_from_rows(
        df_clip, fps, ['px_u', 'px_v', 'mask_attention'],
        {
            'x': lambda r: _safe_col(r, 'px_u'),
            'y': lambda r: _safe_col(r, 'px_v'),
            'valid': lambda r: _safe_col(r, 'mask_attention'),
        })
    gaze_target = _stream_from_rows(
        df_clip, fps, ['hit_object', 'mask_attention'],
        {
            'hit_object': lambda r: _safe_col(r, 'hit_object'),
            'valid': lambda r: _safe_col(r, 'mask_attention'),
        })
    pose = _stream_from_rows(
        df_clip, fps, ['head_rot_z', 'head_vel_rel'],
        {
            'head_yaw': lambda r: _safe_col(r, 'head_rot_z'),
            'head_vel': lambda r: _safe_col(r, 'head_vel_rel'),
        })
    features = {
        'motion': motion,
        'gaze_direction': gaze_direction,
        'gaze_on_screen': gaze_on_screen,
        'gaze_target': gaze_target,
        'pose': pose,
    }
    demo = _demographics_value(dts_df=dts_df, psid_str=psid_str)
    if demo is not None:
        features['demographics'] = {'value': demo}
    return features


def get_binary_annotations_at_time(
        vel, y_ped2center, time, 
        acc_thr_unsure=10, acc_thr_sure=25, 
        vel_thr_cross=50, vel_thr_still=30, state_list=['cross', 'yield']
    ):
    # whether person will yield to automated pod in the future window
    acc = vel.diff().dropna() / time.diff().dropna()
    acc_mean = acc.mean()

    n_vel_move = (vel >= vel_thr_cross).sum()
    n_vel_still = (vel < vel_thr_cross).sum()

    vel_mean = vel.mean()
    direction = 'backward' if y_ped2center.diff().mean() > 0 else 'forward'
    dist_diff = y_ped2center.abs().diff().dropna()
    if dist_diff.mean() < 0 and y_ped2center.iloc[-1] >= 0: 
        in_conflict = True
    else:
        in_conflict = False

    if np.abs(acc_mean) <= acc_thr_unsure and vel_mean >= vel_thr_cross:
        return "cross"
    if direction == 'backward':
        return "yield"
    if acc_mean > 0:
        if n_vel_move >= n_vel_still:
            return "cross"
        else: # n_vel_move < n_vel_still
            return "yield"
    else:
        if in_conflict:
            return "yield"
        else:
            if vel_mean < vel_thr_still:
                return "yield"
            else:
                return "cross"


def get_multi_annotations_at_time(
        vel, y_ped2center, time, 
        acc_thr_unsure=10, acc_thr_sure=25, 
        vel_thr_cross=50, vel_thr_still=30,
        state_list=['speeding up', 'slowing down', 'stepping back', 'constant speed', 'stopping']
    ):
        acc = vel.diff().dropna() / time.diff().dropna()
        vel_mean = vel.mean()
        acc_mean = acc.mean()
        if vel_mean >= 0:
            if acc_mean >= acc_thr_sure:
                return state_list[0]
            elif acc_mean <= -acc_thr_sure:
                return state_list[1]
            elif vel_mean >= vel_thr_cross:
                return state_list[3]
            else: # 0 < vel_mean < vel_thr_cross
                return state_list[4]
        else:
            return state_list[2]


def get_video_annotation(
        vrdata_dir, psid, anno_type, problem_type, 
        past_dur_sec, fut_dur_sec, stride, fps=30, 
        acc_thr_unsure=10, acc_thr_sure=25, vel_thr_cross=50, vel_thr_still=30, 
        ml_feat_list=[], 
        extra_feature_names=None, dts_df=None, feat_every_n_frames=None, legacy_questions=False):
    df = filter_csv(vrdata_dir, psid)
    extra_feature_names = extra_feature_names or []

    ### answers
    # state_list_binary = ['yes', 'no']
    state_list_binary = ['cross', 'yield']
    
    # state_list_multi = ['accelerating', 'decelerating', 'moving backward', 'maintaining speed', 'staying still']
    # state_list_multi = ['speed up', 'slow down', 'step back', 'constant speed', 'stop']
    state_list_multi = ['speeding up', 'slowing down', 'stepping back', 'constant speed', 'stopping']
    # state_list_multi = ['accelerating', 'decelerating', 'stepping back', 'constant speed', 'stopping']
    # state_list_multi = ['accelerating forward', 'decelerating forward', 'moving backward', 'maintaining forward speed', 'stationary']
    # state_list_multi = ['going', 'stopping', 'moving backward', 'going', 'standing']
    
    get_annotation_at_time_dict = {
        'binary': get_binary_annotations_at_time,
        'multi': get_multi_annotations_at_time
    }

    total_num_frames = len(df)  
    past_num_frames = int(past_dur_sec * fps)
    fut_num_frames = int(fut_dur_sec * fps)
    stride_dur = int(stride * fps)
    is_recovered_crossing = False

    if problem_type == 'ml':
        feat_id, feat = [], []
    elif problem_type == 'qa':
        groundvqa_annos, egovlp_annos = [], []

    for i, clip_start_frame_in_trial in enumerate(range(0, total_num_frames - past_num_frames - fut_num_frames + 1, stride_dur)):
        clip_end_frame_in_trial = clip_start_frame_in_trial + past_num_frames
        clip_pred_frame_in_trial = clip_start_frame_in_trial + past_num_frames + fut_num_frames
        clip_start_sec_in_trial = clip_start_frame_in_trial / fps
        clip_end_sec_in_trial = clip_end_frame_in_trial / fps
        
        fut_vel = df['ped_vel'].iloc[clip_end_frame_in_trial:clip_pred_frame_in_trial]
        fut_y_ped2center = df['ped_loc_y'].iloc[clip_end_frame_in_trial:clip_pred_frame_in_trial]
        fut_time = df['time'].iloc[clip_end_frame_in_trial:clip_pred_frame_in_trial]
        fut_state = get_annotation_at_time_dict[anno_type](
            fut_vel, fut_y_ped2center, fut_time, 
            acc_thr_unsure=acc_thr_unsure, acc_thr_sure=acc_thr_sure, 
            vel_thr_cross=vel_thr_cross, vel_thr_still=vel_thr_still, 
            state_list=state_list_binary if anno_type=='binary' else state_list_multi
        )
        
        ## only keep the first crossing before pod pass
        is_last_pod_pass = df['is_last_pod_pass'].iloc[clip_start_frame_in_trial:clip_end_frame_in_trial].any()
        if is_last_pod_pass:
            if is_recovered_crossing: break
            if fut_state == 'cross': is_recovered_crossing = True
        
        clip_uid = psid + '-' + str(i)
        clip_start_sec_in_clip = 0.0
        clip_end_sec_in_clip = past_dur_sec
        clip_start_frame_in_clip = 0
        clip_end_frame_in_clip = past_num_frames
        print(f"sample_id={clip_uid}, Past frame={clip_start_frame_in_trial}-{clip_end_frame_in_trial}, Future frame={clip_end_frame_in_trial}-{clip_pred_frame_in_trial}: {fut_state}")

        if problem_type == 'ml':
            # get features
            feat_i = []
            if 'traj' in ml_feat_list:
                past_traj_xy = df[['ped_loc_x', 'ped_loc_y']].iloc[clip_start_frame_in_trial:clip_end_frame_in_trial].to_numpy()
                feat_i.append(past_traj_xy)
            if 'vel' in ml_feat_list:
                past_vel_xy = df[['ped_vel_x', 'ped_vel_y']].iloc[clip_start_frame_in_trial:clip_end_frame_in_trial].to_numpy()
                feat_i.append(past_vel_xy)
            if 'pod' in ml_feat_list:
                df['distdiff_ped2leader'] = df['ped_loc_x'] - df['leader_loc_x']
                df['distdiff_ped2follower'] = df['ped_loc_x'] - df['follower_loc_x']
                feat_i.append(df[['distdiff_ped2leader']].iloc[clip_start_frame_in_trial:clip_end_frame_in_trial].to_numpy())
                # TODO: fix the dist of pod follower
            feat_i = np.concatenate(feat_i, axis=1)  # (past_num_frames, feat_dim)

            feat_id.append([
                clip_uid, 
                fut_state, 
                clip_start_frame_in_trial, 
                clip_end_frame_in_trial, 
                clip_start_sec_in_trial, 
                clip_end_sec_in_trial,
                clip_start_frame_in_clip, 
                clip_end_frame_in_clip, 
                clip_start_sec_in_clip, 
                clip_end_sec_in_clip
            ])
            feat.append(feat_i)
        elif problem_type == 'qa':
            if anno_type == 'binary':
                # question = f" Question: What is the most likely action of the person in the next {fut_dur_sec} second given what's seen in the past {past_dur_sec} seconds?" ## qn0 
                # question = f"Question: What is your most likely action in the next {fut_dur_sec} second given what's seen in the past {past_dur_sec} seconds?" ## qn1 
                ### qn1/2??
                # question = "You are walking relaxed in a shared space environment where pedestrians and vehicles share the same area without strict traffic rules." +\
                #     "Some automated shuttles may include external displays to communicate: Green pedestrian sign means it can stop in front of you. Red pedestrian sign means it cannot stop in front of you. But you don't need to follow the display." +\
                #     "Your goal is to reach the white circle safely." +\
                #     f"Question: What is your most likely action in the next {fut_dur_sec} second given what's seen in the past {past_dur_sec} seconds?" ## qn2 
                ### qn3
                # question = "You are a pedestrian in a shared space environment where pedestrians and vehicles share the same area without strict traffic rules. " +\
                #     f"You need to get to the white circle safely. Based on what you saw in the egocentric video for the past {past_dur_sec} seconds, what is your most likely action in the next {fut_dur_sec} second?"
                ### qn5
                # question = "You are a pedestrian in a shared space environment where pedestrians and vehicles share the same area without strict traffic rules. " +\
                #     f"You need to get to the white circle safely. In the first-person videos, your gaze point is marked with a red dot. Based on the scene and gaze in the egocentric video for the past {past_dur_sec} seconds, what is your most likely action in the next {fut_dur_sec} second?"
                
                ### qn5_bone
                question = "You are a pedestrian in a shared space environment where pedestrians and vehicles share the same area without strict traffic rules. " +\
                    f"You need to get to the white circle safely. Based on what you saw in the egocentric video and the overlaid gaze fixation heatmap for the past {past_dur_sec} seconds, what is your most likely action in the next {fut_dur_sec} second?"
                
                ### qn6
                # question = "You are a pedestrian in a shared space environment where pedestrians and vehicles share the same area without strict traffic rules. " +\
                #     f"You need to get to the white circle safely. Based on the scene and gaze in the egocentric video for the past {past_dur_sec} seconds, what is your most likely action in the next {fut_dur_sec} second?"
                
                # question = f"I provide you with a video that contains gaze information. For each frame, the gaze point will be marked on the image with a red circle spot. Choose the correct option based on the first-person perspective scene question. Question: What is the most likely action of the person in the next {fut_dur_sec} second given what's seen in the past {past_dur_sec} seconds?"
                # question = f"What will the person do in the next {fut_dur_sec} second given what's seen in the past {past_dur_sec} seconds?"
                # question = f"Will this camera wearer yield to automated pod in the next {fut_dur_sec} second given what's seen in the past {past_dur_sec} seconds?"
                # question = f"Will the person let the coming automated pod pass first in the next {fut_dur_sec} second given what's seen in the past {past_dur_sec} seconds?"
                # question = f"Will the person cross before the approaching automated pod in the next {fut_dur_sec} second given what's seen in the past {past_dur_sec} seconds?"
                template = "Predict: What will happen?"
                wrong_answers = [x for x in state_list_binary if x != fut_state]
            else:
                # question = f"What is the person's movement state in the next {fut_dur_sec} second given what's seen in the past {past_dur_sec} seconds?"
                question = f" Question: What is the person's movement state in the next {fut_dur_sec} second given what's seen in the past {past_dur_sec} seconds?"
                template = "Predict: What is the person's movement state?"
                wrong_answers = random.sample([x for x in state_list_multi if x != fut_state], 3)
            
            df_clip = df.iloc[clip_start_frame_in_trial:clip_pred_frame_in_trial]
            # attach extra time series features if requested
            feat_repr = 'tuple1'
            assert feat_repr in ['statement', 'tuple1', 'tuple2'], f"Invalid feat_repr {feat_repr}"
            extra_ts_feats_names = [i for i in extra_feature_names if i != 'demographics'] if extra_feature_names else []
            if extra_ts_feats_names:
                # pick a representative frame to extract features at (use clip_end_frame_in_trial, bounded)
                if feat_repr == 'statement':
                    extra_ts_feat_str = ''
                    for frame_idx in range(0, past_num_frames+1, feat_every_n_frames):
                        if frame_idx >= clip_end_frame_in_trial:
                            break
                        extra_ts_feat, _ = extract_extra_features_for_frame(df_clip, frame_idx, past_num_frames-1, dts_df=dts_df, psid_str=psid, requested=extra_feature_names)
                        # optionally embed features into question string
                        if len(extra_ts_feats_names) > 0:
                            extra_ts_feat_str = extra_ts_feat_str + " " + ", ".join(f"{k}: {v}" for k, v in extra_ts_feat.items())
                    question = question + ' Observed time series features are ' + extra_ts_feat_str
                elif feat_repr == 'tuple1':
                    extra_ts_feat_str = 'Context: The following time series are observed pairs in the past 2 seconds of '
                    dict_name = {
                        'location': 'location of pedestrian in [x, y]', 
                        'head_pose': 'head pose in degree', 
                        'gaze_orientation': 'gaze orientation in degree', 
                        'gaze_on_screen': 'gaze on screen in [x, y]'
                        }
                    extra_ts_feat_str += f'[time in second, {", ".join([dict_name[i] for i in extra_ts_feats_names])}]: '
                    # 'location', 'head_pose', 'gaze_orientation', 'gaze_on_screen', 'demographics'
                    for frame_idx in range(0, past_num_frames+1, feat_every_n_frames):
                        if frame_idx >= clip_end_frame_in_trial:
                            break
                        extra_ts_feat, _ = extract_extra_features_for_frame_as_tuple(df_clip, frame_idx, past_num_frames-1, dts_df=dts_df, psid_str=psid, requested=extra_feature_names)
                        if len(extra_ts_feats_names) > 0:
                            extra_ts_feat_str += f'{extra_ts_feat}, '.replace("(", "[").replace(")", "]")
                    question = extra_ts_feat_str + question
                elif feat_repr == 'tuple2':
                    dict_name = {
                        'location': 'location of pedestrian in [x, y]', 
                        'head_pose': 'head pose in degree', 
                        'gaze_orientation': 'gaze orientation in degree', 
                        'gaze_on_screen': 'gaze fixation point coordinates (x, y in pixels)'
                        }
                    extra_ts_feat_str = f' Your past {dict_name[extra_feature_names[0]]} on the egocentric videos are: '
                    if 'gaze_on_screen' in extra_ts_feats_names:
                        is_data = False
                        for frame_idx in range(0, past_num_frames+1, feat_every_n_frames):
                            if frame_idx >= clip_end_frame_in_trial:
                                break
                            extra_ts_feat, _ = extract_extra_features_for_frame_as_tuple(df_clip, frame_idx, past_num_frames-1, dts_df=dts_df, psid_str=psid, requested=extra_feature_names)
                            if len(extra_ts_feats_names) > 0 and extra_ts_feat[1] is not None:
                                extra_ts_feat_str += f'Time {extra_ts_feat[0]:.2f}s: Gaze({extra_ts_feat[1]["x"]}, {extra_ts_feat[1]["y"]}), '
                                is_data = True
                        if is_data:
                            extra_ts_feat_str = extra_ts_feat_str.strip(', ')
                            extra_ts_feat_str += '. The coordinate is from left to right for x-axis and from top to bottom for y-axis.'
                        else: 
                            extra_ts_feat_str = ' No valid gaze fixation is detected.'
                    if 'gaze_orientation' in extra_ts_feats_names:
                        is_data = False
                        for frame_idx in range(0, past_num_frames+1, feat_every_n_frames):
                            if frame_idx >= clip_end_frame_in_trial:
                                break
                            extra_ts_feat, _ = extract_extra_features_for_frame_as_tuple(df_clip, frame_idx, past_num_frames-1, dts_df=dts_df, psid_str=psid, requested=extra_feature_names)
                            if len(extra_ts_feats_names) > 0 and extra_ts_feat[1] is not None:
                                extra_ts_feat_str += f'Time {extra_ts_feat[0]:.2f}s: Gaze ({extra_ts_feat[1]["yaw"]:.2f}), '
                                is_data = True
                        if is_data:
                            extra_ts_feat_str = extra_ts_feat_str.strip(', ')
                            extra_ts_feat_str += '. The gaze orientation is represented by yaw angle in degree, where negative means looking more to the left.'
                        else:
                            extra_ts_feat_str = ' No valid gaze orientation is detected.'
                    ins_index = question.index(' Based on the scene and gaze')
                    question = question[:ins_index] + extra_ts_feat_str + question[ins_index:]
            # add extra demographics feature if requested
            if 'demographics' in extra_feature_names:
                _, demo_str = extract_extra_features_for_frame_as_tuple(df_clip, 0, past_num_frames-1, dts_df=dts_df, psid_str=psid, requested=extra_feature_names)
                # question = 'Context: Demographic info of the camera wearer is ' + ", ".join(f"{k}: {v}" for k, v in demo.items()) + question
                question = 'Context: ' + demo_str + question

            groundvqa_anno = {
                "video_uid": psid,
                "video_start_sec": clip_start_sec_in_trial,
                "video_end_sec": clip_end_sec_in_trial,
                "video_start_frame": clip_start_frame_in_trial,
                "video_end_frame": clip_end_frame_in_trial,
                "video_id": clip_uid,
                "sample_id": clip_uid,
                "task": {"name": "crossing_intention", "future_sec": fut_dur_sec},
                "answer": fut_state,
                "moment_start_frame": clip_start_frame_in_clip,
                "moment_end_frame": clip_end_frame_in_clip,
                "wrong_answers": wrong_answers,
                "clip_features": build_clip_features(
                    df.iloc[clip_start_frame_in_trial:clip_end_frame_in_trial],
                    fps=fps, dts_df=dts_df, psid_str=psid),
            }
            if legacy_questions:
                groundvqa_anno["question"] = question
            groundvqa_annos.append(groundvqa_anno)

            egovlp_anno = {
                "clip_uid": clip_uid,
                "video_start_sec": clip_start_sec_in_trial,
                "video_end_sec": clip_end_sec_in_trial,
                "video_start_frame": clip_start_frame_in_trial,
                "video_end_frame": clip_end_frame_in_trial,
                "clip_start_sec": clip_start_sec_in_clip,
                "clip_end_sec": clip_end_sec_in_clip,
                "clip_start_frame": clip_start_frame_in_clip,
                "clip_end_frame": clip_end_frame_in_clip,
                "source_clip_uid": clip_uid,
                "annotations": [
                    {
                        "language_queries": [
                            {
                                "clip_start_sec": clip_start_sec_in_clip,
                                "clip_end_sec": clip_end_sec_in_clip,
                                "video_start_sec": clip_start_sec_in_trial,
                                "video_end_sec": clip_end_sec_in_trial,
                                "video_start_frame": clip_start_frame_in_trial,
                                "video_end_frame": clip_end_frame_in_trial,
                                "template": template,
                                "query": question,
                            }
                        ],
                        "annotation_uid": clip_uid,
                        "request_uid": clip_uid
                    }
                ]
            }
            egovlp_annos.append(egovlp_anno)
    
    if problem_type == 'ml':
        feat = np.stack(feat, axis=0)  # (num_samples, past_num_frames, feat_dim)
        return feat_id, feat
    
    elif problem_type == 'qa':
        egovlp_annos = {
            "video_uid": psid,
            "clips": egovlp_annos
        }
        return groundvqa_annos, egovlp_annos


def visualize_anno(vrdata_dir, psid, anno, fut_sec, anno_type, fig_dir='figures/anno_viz'):
    df = filter_csv(Path(vrdata_dir), psid)

    x = df.ped_loc_x.values
    y = df.ped_loc_y.values
    vel = df.ped_vel.values
    y_ped2center = df.ped_loc_y.values
    time = df.time.values
    acc = np.diff(vel, prepend=vel[0]) / np.diff(time, prepend=time[0])

    dist_diff = np.diff(np.abs(y_ped2center))
    dist_diff = np.concatenate(([dist_diff[0]], dist_diff))  # add a 0 at the beginning to align with original length

    multi_state_to_color = {'speeding up': 'red', 'slowing down': 'blue', 'stepping back': 'green',
                                 'constant speed': 'purple', 'stopping': 'orange'}
    # binary_state_to_color = {'yes': 'red', 'no': 'green'}
    binary_state_to_color = {'cross': 'green', 'yield': 'red'}
    state_to_color_dict = {
        'binary': binary_state_to_color,
        'multi': multi_state_to_color
    }

    fut_num_frames = int(fut_sec * 30)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].scatter(x, y, c='lightgray', marker='.', s=1)  # plot trajectory in light gray
    axs[0].scatter(x[0], y[0], c='black', marker='o', label='start')  # mark start point
    for i, a in enumerate(anno):
        frame_curr = a['video_end_frame']
        fut_state = a['answer']
        axs[0].scatter(x[frame_curr], y[frame_curr], c='k', marker='*')
        axs[0].scatter(x[frame_curr:frame_curr+fut_num_frames], y[frame_curr:frame_curr+fut_num_frames], c=state_to_color_dict[anno_type][fut_state], marker='.', alpha=0.3)
    axs[0].scatter(x[-1], y[-1], c='black', marker='x', label='end')
    axs[0].set_xlabel('x (cm)')
    axs[0].set_ylabel('y (cm)')
    # add legend for states
    for state, color in state_to_color_dict[anno_type].items():
        axs[0].scatter([], [], c=color, marker='.', label=state)
    axs[0].legend(scatterpoints=1, frameon=True, labelspacing=1, title='States')

    # plot velocity over time
    acc_color = 'magenta'
    vel_color = 'cyan'
    axs[1].plot(time, vel, color=vel_color, label='Velocity (cm/s)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Velocity (cm/s)')
    # add vertical lines for annotation moments
    for a in anno:
        t_curr = a['video_end_sec']
        frame_curr = a['video_end_frame']
        vel_mean = vel[frame_curr:frame_curr+fut_num_frames].mean()
        in_conflict = ((dist_diff[frame_curr:frame_curr+fut_num_frames].mean() < 0) & (y_ped2center[-1] >= 0)).astype(bool)
        axs[1].axvline(x=t_curr, color='red' if in_conflict else 'gray', linestyle='--', alpha=0.5)
        axs[1].axhline(y=0, color=vel_color, linestyle='--', alpha=0.5)
        axs[1].plot([t_curr, t_curr+fut_sec], [vel_mean, vel_mean], color=vel_color, alpha=0.7)
    axs[1].yaxis.label.set_color(vel_color)
    
    # twin axes for acceleration
    ax2 = axs[1].twinx()
    ax2.plot(time, acc, color=acc_color, label='Acceleration (cm/s²)')
    for a in anno:
        t_curr = a['video_end_sec']
        frame_curr = a['video_end_frame']
        acc_mean = acc[frame_curr:frame_curr+fut_num_frames].mean()
        ax2.axhline(y=0, color=acc_color, linestyle='--', alpha=0.5)
        ax2.plot([t_curr, t_curr+fut_sec], [acc_mean, acc_mean], color=acc_color, alpha=0.7)
    ax2.set_ylabel('Acceleration (cm/s²)')
    ax2.yaxis.label.set_color(acc_color)

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    png_filename = f'{fig_dir}/{psid}_{anno_type}.png'
    plt.savefig(png_filename)
    print(f"Saved annotation visualization to {png_filename}")


def main(argv=None):
    # psid = 'P1S2'  # hardcoded for now

    parser = argparse.ArgumentParser(description="Prepare GT samples from VR data")
    parser.add_argument("--vrdata_dir", type=str, default="data/vrdata", help="VR data directory containing *.csv files")
    parser.add_argument("--problem_type", type=str, default='qa', choices=['qa', 'ml'], help="type of problem to prepare annotations for")
    parser.add_argument("--dts_csv", type=str, default='data/dts_qn.csv', help="Path to dts_qn.csv for demographics lookup")

    # time series parameters
    parser.add_argument("--past_sec", type=float, default=2, help="past window length in seconds")
    parser.add_argument("--fut_sec", type=float, default=1, help="future window length in seconds")
    parser.add_argument("--stride", type=float, default=0.5, help="stride between sample center times (seconds)")
    parser.add_argument("--fps", type=int, default=30, help="frame rate of the videos (frames per second)")

    # labeling parameters
    parser.add_argument("--anno_type", type=str, default="binary", choices=["multi", "binary", "long"], help="type of annotation to prepare")
    parser.add_argument("--acc_thr_unsure", type=float, default=10, help="Acceleration threshold for unsure state")
    parser.add_argument("--acc_thr_sure", type=float, default=25, help="Acceleration threshold for sure state")
    parser.add_argument("--vel_thr_cross", type=float, default=50, help="Velocity threshold for crossing")
    parser.add_argument("--vel_thr_still", type=float, default=30, help="Velocity threshold for stillness")
    parser.add_argument("--sample_n", type=int, default=None, help="Debug mode: stop after generating at least N samples/feature rows")
        
    # Keep the public arg space clean: inspect only --problem_type first,
    # then register the relevant mode-specific arguments on the real parser.
    prelim = argparse.ArgumentParser(add_help=False)
    prelim.add_argument("--problem_type", type=str, default='qa', choices=['qa', 'ml'])
    known_args, _ = prelim.parse_known_args(argv)

    if known_args.problem_type == 'ml':
        parser.add_argument("--ml_feat_list", type=str, default=["traj", "vel", "pod"], nargs="+", help="list of features to extract")
        parser.add_argument("--ml_feat_dir", type=str, default='features/ml', help="Path to save the extracted ML features")
    elif known_args.problem_type == 'qa':
        # parser.add_argument("--whichones", default=['P1', 'P2'], help="whether processing all VR data files")
        parser.add_argument("--whichones", default='all', help="whether processing all VR data files")
        parser.add_argument("--anno_dir", type=str, default='data/annotations', help="VR data directory containing *.csv files")
        parser.add_argument("--json_dir", type=str, default='features/groundvqa_', help="Path to save the prepared annotation JSON files")
        parser.add_argument("--viz_state", type=bool, default=False, help="visualize state transitions")
        # old 
        parser.add_argument("--extra_feature_names", nargs='+', default=[],
                            # default=['location', 'head_pose', 'gaze_orientation', 'gaze_on_screen', 'demographics'],
            help="Comma-separated extra features to include: location,head_pose,gaze_orientation,gaze_on_screen,demographics")
        parser.add_argument("--feat_every_n_frames", type=int, default=2, help="Legacy prompt feature stride; ignored by new structured annotations")
        parser.add_argument("--legacy_questions", action="store_true", help="Keep legacy question/context text in generated JSON")

    args = parser.parse_args(argv)

    if args.problem_type == 'ml':
        vrdata_files = [f.split('.')[0] for f in os.listdir(args.vrdata_dir) if f.endswith('.csv')]
        print(f"Found {len(vrdata_files)} VR data files.")

        ### process each file
        feat_id_annos = [] 
        feat_annos = []
        for psid in vrdata_files:
            print(f"Processing {psid}...")
            feat_id, feat = get_video_annotation(
                vrdata_dir=Path(args.vrdata_dir), 
                psid=psid, 
                anno_type=args.anno_type, 
                problem_type=args.problem_type,
                past_dur_sec=args.past_sec, 
                fut_dur_sec=args.fut_sec,
                stride=args.stride, 
                fps=args.fps,
                acc_thr_unsure=args.acc_thr_unsure,
                acc_thr_sure=args.acc_thr_sure,
                vel_thr_cross=args.vel_thr_cross,
                vel_thr_still=args.vel_thr_still,
                ml_feat_list=args.ml_feat_list
            )
            feat_id_annos.extend(feat_id)
            feat_annos.append(feat)
            if args.sample_n is not None and len(feat_id_annos) >= args.sample_n:
                print(f"Reached --sample_n={args.sample_n}; stopping ML generation early.")
                break
        feat_id = np.stack(feat_id_annos, axis=0)  # (num_samples, 9)
        feat = np.concatenate(feat_annos, axis=0)  # (num_samples, past_num_frames, feat_dim)
        if args.sample_n is not None:
            feat_id = feat_id[:args.sample_n]
            feat = feat[:args.sample_n]
        print(f"feats.shape={feat.shape}, feats_id.shape={feat_id.shape}")

        # save small JSON summary
        if not os.path.exists(args.ml_feat_dir):
            os.makedirs(args.ml_feat_dir)
        feat_path = os.path.join(args.ml_feat_dir, f'{args.anno_type}_{"".join(args.ml_feat_list)}.npz')
        
        np.savez_compressed(feat_path, feat_id=feat_id, feat=feat)
        print(f"Wrote {len(feat_id)} samples to {feat_path}")

    elif args.problem_type == 'qa':
        if not os.path.exists(args.anno_dir):
            os.mkdir(args.anno_dir)
        if not os.path.exists(args.json_dir):
            os.mkdir(args.json_dir)

        ### find all files
        if args.whichones == 'all':
            # vrdata_files = [f.split('.')[0] for f in os.listdir(args.vrdata_dir) if f.endswith('.csv') and (f.startswith('P1S') or f.startswith('P2S'))]
            vrdata_files = [f.split('.')[0] for f in os.listdir(args.vrdata_dir) if f.endswith('.csv')]
        elif isinstance(args.whichones, list):
            vrdata_files = [f.split('.')[0] for f in os.listdir(args.vrdata_dir) if f.endswith('.csv') and f.split('S')[0] in args.whichones]
        else:
            raise ValueError("whichones should be 'all' or a list of psids")
        # vrdata_files = ['P39S11']
        print(f"Found {len(vrdata_files)} VR data files.")

        ### process each file
        if args.anno_type != 'long':
            groundvqa_annos = [] 
            egovlp_annos = {
                "version": "1.0",
                "date": "251216",
                "description": "NLQ Annotation for VR test",
                "manifest": None,
                "videos": []
            }
            # add suffix to output json filenames based on the extra_feature_names
            ename = ''
            for feat_name in ['location', 'head_pose', 'gaze_orientation', 'gaze_on_screen', 'demographics']:
                if feat_name in args.extra_feature_names:
                    ename += '1'
                else:
                    ename += '0'
            if np.array([i in args.extra_feature_names for i in ['location', 'head_pose', 'gaze_orientation', 'gaze_on_screen']]).any():
                ename += f'_every{args.feat_every_n_frames}'
            print(f"Extra feature encoding: {ename}")

            # load dts csv if requested
            dts_df = None
            if os.path.exists(args.dts_csv):
                dts_df = pd.read_csv(args.dts_csv)
            elif 'demographics' in args.extra_feature_names:
                raise FileNotFoundError(f"'{args.dts_csv}' not found; demographics will be None")

            for psid in vrdata_files:
                print(f"Processing {psid}...")
                groundvqa_anno, egovlp_anno = get_video_annotation(
                    vrdata_dir=Path(args.vrdata_dir), 
                    psid=psid, 
                    anno_type=args.anno_type, 
                    problem_type=args.problem_type,
                    past_dur_sec=args.past_sec, 
                    fut_dur_sec=args.fut_sec, 
                    stride=args.stride, 
                    fps=args.fps,
                    acc_thr_unsure=args.acc_thr_unsure,
                    acc_thr_sure=args.acc_thr_sure,
                    vel_thr_cross=args.vel_thr_cross,
                    vel_thr_still=args.vel_thr_still,
                    extra_feature_names=args.extra_feature_names, 
                    dts_df=dts_df, 
                    feat_every_n_frames=args.feat_every_n_frames,
                    legacy_questions=args.legacy_questions)
                # visualize state transitions if needed
                if args.viz_state:
                    visualize_anno(args.vrdata_dir, psid, groundvqa_anno, args.fut_sec, args.anno_type)
                if args.sample_n is not None:
                    remaining = args.sample_n - len(groundvqa_annos)
                    if remaining <= 0:
                        break
                    groundvqa_annos.extend(groundvqa_anno[:remaining])
                    kept_ids = {a["video_id"] for a in groundvqa_anno[:remaining]}
                    egovlp_anno = dict(egovlp_anno)
                    egovlp_anno["clips"] = [c for c in egovlp_anno.get("clips", []) if c.get("clip_uid") in kept_ids]
                    egovlp_annos['videos'].append(egovlp_anno)
                    if len(groundvqa_annos) >= args.sample_n:
                        print(f"Reached --sample_n={args.sample_n}; stopping QA generation early.")
                        break
                else:
                    groundvqa_annos.extend(groundvqa_anno)
                    egovlp_annos['videos'].append(egovlp_anno)

            # save small JSON summary
            if args.whichones == 'all':
                testname = 'full'
            else:
                testname = 'testing'

            json_path = f'{args.json_dir}/annotations.VR{args.anno_type}__crossing_intention__{testname}_close.json'
            write_json_with_progress(groundvqa_annos, json_path, desc='Writing GroundVQA annotations')
            print(f"Wrote {len(groundvqa_annos)} samples to {json_path}")

            # write a small summary of label counts alongside the annotations JSON
            try:
                from collections import Counter
                label_counts = Counter([a.get('answer') for a in groundvqa_annos])
                summary = {
                    'total_samples': len(groundvqa_annos),
                    'label_counts': dict(label_counts)
                }
                print(summary)
            except Exception as e:
                print(f"Failed to write label summary: {e}")

            json_path = os.path.join(args.anno_dir, args.anno_type)
            if not os.path.exists(json_path):
                os.mkdir(json_path)
            json_path = os.path.join(json_path, f'nlq_crossing_intention_{testname}.json')
            write_json_with_progress(egovlp_annos, json_path, desc='Writing EgoVLP annotations')
            print(f"Wrote {len(egovlp_annos['videos'])} samples to {json_path}")
        else:
            # long annotations: fullscale clips
            groundvqa_annos = [] 
            egovlp_annos = {
                "version": "1.0",
                "date": "251216",
                "description": "Fullscale Annotation for VR test",
                "manifest": None,
                "videos": []
            }
            for psid in vrdata_files:
                groundvqa_anno, elovlp_anno = get_video_fullscale_annotation(Path(args.vrdata_dir), psid)
                if args.sample_n is not None and len(groundvqa_annos) >= args.sample_n:
                    break
                groundvqa_annos.extend(groundvqa_anno)
                egovlp_annos['videos'].append(elovlp_anno)

            json_path = 'features/features_long/annotations.VR_test.json'
            write_json_with_progress(groundvqa_annos, json_path, desc='Writing GroundVQA annotations')
            print(f"Wrote {len(groundvqa_annos)} samples to {json_path}")

            # also write a label summary for the fullscale annotations
            try:
                from collections import Counter
                label_counts = Counter([a.get('answer') for a in groundvqa_annos])
                summary = {
                    'total_samples': len(groundvqa_annos),
                    'label_counts': dict(label_counts)
                }
                print(summary)
            except Exception as e:
                print(f"Failed to write label summary: {e}")
            
            json_path = os.path.join(args.anno_dir, 'long', 'nlq_val.json')
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            write_json_with_progress(egovlp_annos, json_path, desc='Writing EgoVLP annotations')
            print(f"Wrote {len(egovlp_annos['videos'])} fullscale annotations to {json_path}")


if __name__ == '__main__':
    main()
