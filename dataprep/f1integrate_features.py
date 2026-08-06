import pickle
import torch
import os
import json
import h5py


feature_dir = 'features'
gaze_format='overlay_' # ['overlay_', 'raw_']
egovlp_path = f'{feature_dir}/{gaze_format}egovlp'
internvideo_verb_path = f'{feature_dir}/{gaze_format}internvideo_verb'
internvideo_noun_path = f'{feature_dir}/{gaze_format}internvideo_noun'

anno_filepath = f'{feature_dir}/groundvqa/annotations.VRbinary_00000_full_close.json'
# anno_filepath = f'{feature_dir}/annotations.VR_test.json'
comb_feat_path = f'{feature_dir}/{gaze_format}unified'

hdf5_path = f'{feature_dir}/groundvqa'


def find_unique_files():
    with open(anno_filepath, 'r') as f:
        annotations = json.load(f)
    clipname_lists = [a['video_id'] for a in annotations]
    clipname_lists = list(set(clipname_lists))
    return clipname_lists


def integrate_features(clipname_lists):
    if not os.path.exists(comb_feat_path):
        os.makedirs(comb_feat_path)
    all_obtained = True

    # load features
    for clipname in clipname_lists:
        if os.path.exists(os.path.join(comb_feat_path, clipname + '.pt')):
            print(f"{clipname}'s features are already combined.")
            continue
        egovlp_feat = torch.load(f'{egovlp_path}/{clipname}.pt')
        internvideo_verb_feat = torch.load(f'{internvideo_verb_path}/{clipname}.pt')
        internvideo_noun_feat = torch.load(f'{internvideo_noun_path}/{clipname}.pt')

        # order: egovlp, verb, noun
        size = min(egovlp_feat.size(0), internvideo_verb_feat.size(0), internvideo_noun_feat.size(0))
        combined_feat = torch.cat([egovlp_feat[:size], internvideo_verb_feat[:size], internvideo_noun_feat[:size]], dim=1)

        torch.save(combined_feat, f'{comb_feat_path}/{clipname}.pt')
        all_obtained = False

    print("All features are already combined!" if all_obtained else "Some features were combined.")


def construct_hdf5(clipname_lists):
    with h5py.File(f"{hdf5_path}/egovlp_internvideo_{gaze_format.split('_')[0]}.hdf5", "w") as f:
        for clipname in clipname_lists:
            iv = torch.load(f"{comb_feat_path}/{clipname}.pt", map_location='cpu').numpy()
            f.create_dataset(clipname, data=iv, dtype="<f4")


def get_preextracted_features_given_lists(clipname_lists):
    in_path = f"{hdf5_path}/egovlp_internvideo_15000groundvqa.hdf5"
    out_path = f"{hdf5_path}/egovlp_internvideo_pre.hdf5"
    keep_keys = clipname_lists

    with h5py.File(in_path, "r") as infile, h5py.File(out_path, "w") as outfile:
        for key in infile.keys():
            if key in keep_keys:
                infile.copy(key, outfile)
    infile.close()
    outfile.close()


if __name__ == '__main__':
    clipname_lists = find_unique_files()
    integrate_features(clipname_lists)
    construct_hdf5(clipname_lists)
    # get_preextracted_features_given_lists(clipname_lists)