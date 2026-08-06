#!/usr/bin/env python3
import json
from pathlib import Path


def filter_file(path: Path, prefixes=('P12','P13','P2')):
    print(f"Processing {path}")
    data = json.loads(path.read_text())
    videos = data.get('videos', [])
    orig_count = len(videos)
    prefixes_t = tuple(prefixes)
    filtered = [v for v in videos if isinstance(v.get('video_uid'), str) and v['video_uid'].startswith(prefixes_t)]
    kept_count = len(filtered)
    # prepare output file (do not overwrite the original nlq_val.json)
    out_path = path.with_name('nlq_test.json')
    # # if an existing nlq_test.json is present, back it up
    # if out_path.exists():
    #     bak_out = out_path.with_suffix(out_path.suffix + '.bak')
    #     bak_out.write_text(out_path.read_text())
    #     print(f"  Existing {out_path.name} backed up to {bak_out.name}")

    out_data = dict(data)
    out_data['videos'] = filtered
    out_path.write_text(json.dumps(out_data, indent=4))
    if kept_count == orig_count:
        print(f"  Written {out_path.name}: no change ({kept_count}/{orig_count})")
    else:
        print(f"  Written {out_path.name}: kept {kept_count}/{orig_count}")


def main():
    base = Path(__file__).resolve().parents[1]
    # targets = [base / 'annotations' / 'binary' / 'nlq_val.json',
    #            base / 'annotations' / 'multiclass' / 'nlq_val.json']
    targets = [base / 'annotations' / 'multiclass' / 'nlq_val.json']
    for t in targets:
        if not t.exists():
            print(f"Warning: {t} not found")
            continue
        filter_file(t)

if __name__ == '__main__':
    main()
