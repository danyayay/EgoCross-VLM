#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import re
from pathlib import Path
from utils.vr_dataset import split_person_groups

anno_dir = Path('features/groundvqa_auxtask')

filenames = [
    'annotations.VRbinary__aux_vehicle_speed__full_close.json',
    'annotations.VRbinary__aux_vehicle_approach__full_close.json',
    'annotations.VRbinary__aux_ped_moving__full_close.json',
    'annotations.VRbinary__aux_ped_direction__full_close.json',
    'annotations.VRbinary__aux_gaze_vehicle__full_close.json',
    'annotations.VRbinary__aux_ehmi__full_close.json',
    'annotations.VRbinary__aux_head_turning__full_close.json',
    'annotations.VRbinary__aux_vehicle_proximity__full_close.json',
    'annotations.VRbinary__aux_crossing_proximity__full_close.json',
]

for anno_filename in filenames:
    print(f'Processing {anno_filename}...')
    anno_filepath = anno_dir / anno_filename
    
    try:
        videos = json.loads(anno_filepath.read_text())
    except Exception as e:
        print(f'  Skipping (not valid JSON): {e}')
        continue
    
    if not isinstance(videos, list):
        print(f'  Unsupported JSON structure')
        continue
    
    orig_count = len(videos)
    
    # Get split persons
    splits_dict = split_person_groups(
        annotation_path=str(anno_filepath),
        seed=42, ratios=(6, 1, 3))
    
    # Split data
    for split in ['train', 'val', 'test']:
        data = [v for v in videos if isinstance(v.get('video_uid'), str) and 
                re.match(r'(P\d+)S\d+', v['video_uid']).groups()[0] in (splits_dict[f'{split}_persons'])]
        kept_count = len(data)
        dst_path = str(anno_filepath).replace('full', split)
        with open(dst_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f'  Wrote {dst_path}: kept {kept_count}/{orig_count}')
    
    print()
