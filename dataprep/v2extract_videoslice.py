import json
import os
import pandas as pd
import subprocess
import argparse


def extract_video_clips(videodata_dir, anno_filepath):
    in_video_dir = f'{videodata_dir}/full_scale_rainbow'
    out_video_dir = f'{videodata_dir}/clips_rainbow'
    if not os.path.exists(out_video_dir):
        os.makedirs(out_video_dir)

    with open(anno_filepath, 'r') as f:
        anno_data = json.load(f)

    for anno in anno_data:
        in_video_name = anno['video_uid']
        print('Processing video:', in_video_name)

        video_start_sec = anno['video_start_sec']
        video_end_sec = anno['video_end_sec']
        out_video_name = anno['video_id']

        # output video
        output_path = f"{out_video_dir}/{out_video_name}.mp4"
        command = [
            "ffmpeg",
            "-y",
            "-i", f'{in_video_dir}/{in_video_name}.mp4',
            "-ss", str(video_start_sec),
            "-to", str(video_end_sec),
            output_path
        ]

        completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if completed.returncode != 0:
            print(f"ffmpeg failed for {in_video_name}: returncode={completed.returncode}")
            print(completed.stderr.decode(errors='ignore'))
        else:
            print(f"Clipped frames {video_start_sec}–{video_end_sec} saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video clips from raw videos based on trial metadata")
    parser.add_argument("--anno_filepath", type=str, default='features/groundvqa/annotations.VRbinary_00000_full_close.json', help="Path to save the extracted video clips")
    parser.add_argument("--videodata_dir", type=str, default='data/videodata_256', help="Path to the raw video directory")

    args = parser.parse_args()

    extract_video_clips(args.videodata_dir, args.anno_filepath)