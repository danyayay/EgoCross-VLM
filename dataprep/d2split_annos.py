#!/usr/bin/env python3
import os
import re
import json
import argparse
from pathlib import Path
from utils.vr_dataset import split_person_groups


def split_full_annos_to_train_val_test_groundvqa(
        anno_dir, anno_filename, splits_dict):
    if not os.path.exists(anno_dir):
        print(f"Features directory not found: {anno_dir}")
        raise SystemExit(2)

    # PREFIXES_TRAIN = splits_dict['train_persons']  # e.g. ['P0', 'P2', 'P4']
    # PREFIXES_TEST = splits_dict['test_persons']  # e.g. ['P1', 'P3', 'P5']
    # PREFIXES_VAL = splits_dict['val_persons']  # e.g. ['P2', 'P4']

    # read files 
    anno_filepath = anno_dir / anno_filename
    print(f"Processing {anno_filepath}...")
    try:
        videos = json.loads(anno_filepath.read_text())
    except Exception as e:
        print(f"Skipping (not valid JSON): {e}")
        return
    
    # support two common shapes:
    # 1) dict with 'videos' list (like nlq files)
    # 2) top-level list of entries (like features annotations)
    if isinstance(videos, list):
        orig_count = len(videos)
        for split in ['train', 'val', 'test']:
            re.match('P\d+', videos[0]['video_uid'])
            data = [v for v in videos if isinstance(v.get('video_uid'), str) and \
                    re.match(r'(P\d+)S\d+', v['video_uid']).groups()[0] in (splits_dict[f'{split}_persons'])]
            kept_count = len(data)
            dst_path = f"{anno_dir}/{anno_filename.replace('full', split)}"
            with open(dst_path, 'w') as f:
                f.write(json.dumps(data, indent=4))
            print(f"  Wrote {dst_path}: kept {kept_count}/{orig_count}")
    else:
        print(f"  Unsupported JSON structure: {type(data)}")
        return False
    
    print('Done')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Split full annotations JSON file into train/val/test based on person-level splitting')
    parser.add_argument('--anno_dir', type=Path, default='features/groundvqa', help='Comma-separated list of feature IDs to use for splitting (e.g. "P0,P2,P4"). If not provided, will read from anno_filepath.')
    parser.add_argument('--anno_filename', type=str, default='annotations.VRbinary__crossing_intention__full_close.json', help='Name of the full annotations JSON file to split (should be in features/groundvqa)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility of the person-level splitting')
    parser.add_argument('--ratios', type=str, default='6,1,3', help='Comma-separated ratios for train,val,test splits (e.g. "6,1,3")')
    args = parser.parse_args()

    # get split persons
    splits_dict = split_person_groups(
        annotation_path=os.path.join(args.anno_dir, args.anno_filename),
        seed=args.seed, ratios=tuple(map(int, args.ratios.split(','))))

    # get split data
    split_full_annos_to_train_val_test_groundvqa(args.anno_dir, args.anno_filename, splits_dict)