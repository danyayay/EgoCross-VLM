import os
import argparse
import subprocess
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Video resizing script')
parser.add_argument('--input_folder', type=str, default='data/videodata/full_scale', help='Path to the input folder')
parser.add_argument('--output_folder', type=str, default='data/videodata_256/full_scale', help='Path to the output folder')
args = parser.parse_args()

folder_path = args.input_folder
output_path = args.output_folder

if not os.path.exists(output_path):
    os.makedirs(output_path)


def videos_resize(videoinfos):
    global count

    videoid, videoname = videoinfos

    if os.path.exists(os.path.join(output_path, videoname)):
        print(f'{videoname} is resized.')
        return

    inname = folder_path + '/' + videoname
    outname = output_path + '/' + videoname

    cmd = "ffmpeg -y -i {} -filter:v scale=\"trunc(oh*a/2)*2:256\" -c:a copy {}".format( inname, outname)
    subprocess.call(cmd, shell=True)

    return


if __name__ == "__main__":
    file_list = []
    mp4_list = [item for item in os.listdir(folder_path) if item.endswith('.mp4')]
    mp4_list.sort()
    for id, video in enumerate(mp4_list):
        file_list.append([id, video])

    pool = Pool(4)
    pool.map(videos_resize, tuple(file_list))