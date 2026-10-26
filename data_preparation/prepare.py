"""Offline image indexing, dataset sampling, and image preparation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess

import numpy as np
from PIL import Image

DATASETS = {'dl3dv', 'realestate10k', 'acid', 'tanks', 'mipnerf360'}
MIP_SCENES = {'bicycle', 'bonsai', 'counter', 'flowers', 'garden', 'kitchen', 'room', 'stump', 'treehill'}
SIZES = {'train-low': (320, 512), 'train-high': (576, 1024), 'eval': (512, 512)}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never silently overwrite a previous split or preprocessing result.
    with path.open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(base, path):
    return (Path(base).parent / path).resolve()


def natural_key(path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', str(path))]


def check_camera(K, c2w):
    K, c2w = np.asarray(K, dtype=float), np.asarray(c2w, dtype=float)
    if K.shape != (3, 3) or c2w.shape != (4, 4):
        raise ValueError('Expected K[3,3], OpenCV c2w[4,4]')
    if not np.isfinite(K).all() or not np.isfinite(c2w).all():
        raise ValueError('Non-finite camera')
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError('Invalid pinhole intrinsics')
    if not np.allclose(c2w[3], [0, 0, 0, 1]):
        raise ValueError('Invalid homogeneous pose')
    R = c2w[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(R), 1, atol=1e-3):
        raise ValueError('Pose rotation must be in SO(3)')


def index_images(args):
    root = Path(args.images).resolve()
    files = sorted([p for p in root.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}], key=natural_key)
    if not files:
        raise ValueError('No images')
    cameras = None
    if args.cameras:
        with np.load(args.cameras, allow_pickle=False) as z:
            cameras = {k: z[k].copy() for k in ('K', 'c2w', 'names')}
        if list(cameras['names']) != [p.name for p in files]:
            raise ValueError('Camera names must exactly match natural-sorted image filenames')
        if len(cameras['K']) != len(files) or len(cameras['c2w']) != len(files):
            raise ValueError('Camera/image count mismatch')
    frames = []
    for i, path in enumerate(files):
        with Image.open(path) as im:
            w, h = im.size
        row = {'path': str(path), 'source_index': i, 'width': w, 'height': h}
        if cameras:
            check_camera(cameras['K'][i], cameras['c2w'][i])
            row.update(K=cameras['K'][i].tolist(), c2w=cameras['c2w'][i].tolist())
        frames.append(row)
    write_json(args.out, {'version': 1, 'ordering': 'natural_filename', 'camera_convention': 'opencv_c2w', 'frames': frames})


def parse_re10k(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError('Empty camera metadata')
    rows = []
    for line in lines[1:]:
        cols = line.split()
        if len(cols) != 19:
            raise ValueError('Expected 19 columns: timestamp, 6 intrinsics fields, 12 extrinsics fields')
        timestamp = int(cols[0])
        values = np.asarray(cols[1:], dtype=float)
        w2c = np.eye(4)
        w2c[:3] = values[-12:].reshape(3, 4)
        rows.append((timestamp, values[:4], np.linalg.inv(w2c)))
    if any(rows[i][0] >= rows[i+1][0] for i in range(len(rows)-1)):
        raise ValueError('Metadata timestamps must be strictly increasing; do not reorder benchmark indices')
    return lines[0], rows


def index_re10k(args):
    url, rows = parse_re10k(args.metadata)
    root = Path(args.images).resolve()
    root.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, (ts, intr, c2w) in enumerate(rows):
        path = root / f'{ts}.png'
        if not path.exists() and args.video:
            # Timestamp is microseconds, never a frame number. Local video only.
            subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-i', str(Path(args.video).resolve()),
                            '-ss', f'{ts / 1e6:.6f}', '-frames:v', '1', str(path)], check=True)
        if not path.is_file():
            raise FileNotFoundError(f'Missing timestamp image {path}; no frame is skipped or reindexed')
        with Image.open(path) as im:
            w, h = im.size
        fx, fy, cx, cy = intr
        K = np.array([[fx*w, 0, cx*w], [0, fy*h, cy*h], [0, 0, 1]])
        check_camera(K, c2w)
        frames.append({'path': str(path), 'source_index': i, 'timestamp_us': ts,
                       'width': w, 'height': h, 'K': K.tolist(), 'c2w': c2w.tolist()})
    write_json(args.out, {'version': 1, 'source_url': url, 'metadata_sha256': sha256(args.metadata),
                          'ordering': 'official_metadata_rows', 'camera_convention': 'opencv_c2w', 'frames': frames})


def validate_catalog(path, paper_counts=False):
    catalog = read_json(path)
    scenes = catalog['scenes']
    if not scenes:
        raise ValueError('Catalog is empty')
    seen, sources, files = set(), {}, {}
    eval_names = {d: set() for d in DATASETS}
    for s in scenes:
        ds, sid, split = s['dataset'], s['scene_id'], s['split']
        if ds not in DATASETS or split not in {'train', 'eval'}:
            raise ValueError('Unknown dataset or split')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', sid) or sid in {'.', '..'}:
            raise ValueError('scene_id must be a safe single path component')
        if (ds, sid) in seen:
            raise ValueError(f'Duplicate scene or train/eval overlap: {ds}/{sid}')
        seen.add((ds, sid))
        # source_id should identify the original video/physical scene, not just a clip.
        source = s['source_id']
        if not isinstance(source, str) or not source.strip():
            raise ValueError('source_id is required for video/scene leakage checks')
        if source in sources and sources[source] != split:
            raise ValueError(f'Source video/scene crosses train/eval: {source}')
        sources[source] = split
        official = s['source_split']
        if ds == 'dl3dv' and official != ('DL3DV-10K' if split == 'train' else 'DL3DV-Evaluation'):
            raise ValueError('DL3DV eval must use independent DL3DV-Evaluation, never DL3DV-Benchmark')
        if ds in {'acid', 'realestate10k'} and official != ('train' if split == 'train' else 'test'):
            raise ValueError('Preserve official train/test split for RE10K and ACID')
        if ds in {'tanks', 'mipnerf360'} and split != 'eval':
            raise ValueError('Out-of-domain data cannot be used for training')
        if split == 'eval':
            eval_names[ds].add(sid)
        seqpath = resolve(path, s['sequence'])
        seq = read_json(seqpath)
        frames = seq['frames']
        if not frames:
            raise ValueError(f'Empty sequence {seqpath}')
        if seq.get('camera_convention') != 'opencv_c2w':
            raise ValueError('Explicit opencv_c2w camera convention required')
        local_paths = set()
        for i, f in enumerate(frames):
            p = resolve(seqpath, f['path'])
            if p in local_paths:
                raise ValueError(f'Duplicate frame path: {p}')
            local_paths.add(p)
            if not p.is_file() or f['source_index'] != i:
                raise ValueError(f'Missing image or non-contiguous source_index in {seqpath}')
            if f['width'] <= 0 or f['height'] <= 0:
                raise ValueError('Invalid image dimensions')
            if p in files and files[p] != split:
                raise ValueError(f'Image file crosses train/eval: {p}')
            files[p] = split
            if 'K' in f or 'c2w' in f:
                check_camera(f['K'], f['c2w'])
    if paper_counts:
        for ds, n in [('dl3dv', 20), ('tanks', 7), ('mipnerf360', 9)]:
            if len(eval_names[ds]) != n:
                raise ValueError(f'Paper requires {n} eval scenes for {ds}; got {len(eval_names[ds])}')
        if eval_names['mipnerf360'] != MIP_SCENES:
            raise ValueError('Mip-NeRF 360 must contain all nine canonical scenes')
        for ds in ('acid', 'realestate10k'):
            if not eval_names[ds]:
                raise ValueError(f'Missing {ds} Easy Set evaluation scenes')
    return scenes


def training_samples(scene, n_frames, seed, count=1, num_refs=2, sampling=None):
    sampling = sampling or {}
    init_frames = sampling.get("initialization_frames", 50)
    video_frames = sampling.get("video_frames", 25)
    stride_min = sampling.get("stride_min", 2)
    stride_max = sampling.get("stride_max", 5)
    if not 2 <= num_refs <= video_frames <= init_frames or count < 1 or not 1 <= stride_min <= stride_max:
        raise ValueError('Invalid frame counts, reference count, or stride range')
    # Per-scene seed makes results independent of catalog ordering.
    key = f"{seed}:{scene['dataset']}:{scene['scene_id']}"
    rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
    strides = [s for s in range(stride_min, stride_max+1) if (init_frames-1)*s < n_frames]
    if not strides:
        raise ValueError(f"{scene['scene_id']}: insufficient frames for the configured initialization clip")
    result = []
    for clip in range(count):
        stride = rng.choice(strides)
        start = rng.randrange(n_frames - (init_frames-1)*stride)
        init = [start + i*stride for i in range(init_frames)]
        offset = rng.randrange(init_frames-video_frames+1)
        target = init[offset:offset+video_frames]
        refs = sorted(rng.sample(target, num_refs))
        result.append({'sample_id': f'train_{clip:05d}', 'initialization_indices': init,
                       'reference_indices': refs, 'point_source_indices': refs,
                       'target_indices': target, 'stride': stride,
                       'protocol': 'training'})
    return result


def check_eval_record(row, n_frames, dataset, view_counts=(2, 3, 6), frame_stride=10):
    refs, target = row['reference_indices'], row['target_indices']
    for name, ids in [('reference_indices', refs), ('target_indices', target)]:
        if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 or i >= n_frames for i in ids):
            raise ValueError(f'Invalid {name}')
    if set(refs) & set(target):
        raise ValueError('Evaluation targets must be held out from reference images')
    if dataset in {'acid', 'realestate10k'}:
        if len(refs) != 2 or row.get('protocol') != 'reconx_easy' or not row.get('index_source'):
            raise ValueError('RE10K/ACID require original ReconX Easy Set indices and index_source provenance')
    else:
        if len(refs) not in view_counts or any(b-a != frame_stride for a, b in zip(refs, refs[1:])):
            raise ValueError(f'Expected view counts {view_counts} at stride {frame_stride}')
    sid = row['sample_id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', sid):
        raise ValueError('Unsafe sample_id')


def make_plan(args):
    scenes = validate_catalog(args.catalog, args.paper_counts)
    records = read_json(args.eval_index)['samples'] if args.eval_index else []
    consumed = set()
    samples = []
    for scene in scenes:
        seqpath = resolve(args.catalog, scene['sequence'])
        seq = read_json(seqpath)
        if scene['split'] == 'train':
            rows = training_samples(scene, len(seq['frames']), args.seed, args.clips_per_scene, args.num_refs, getattr(args, "sampling", None))
        else:
            rows = []
            for j, r in enumerate(records):
                if (r['dataset'], r['scene_id']) == (scene['dataset'], scene['scene_id']):
                    check_eval_record(r, len(seq['frames']), scene['dataset'],
                                      getattr(args, 'view_counts', (2, 3, 6)), getattr(args, 'frame_stride', 10))
                    rows.append({**r, 'initialization_indices': r['reference_indices'],
                                 'point_source_indices': r['reference_indices']})
                    consumed.add(j)
            if not rows:
                raise ValueError(f"Missing explicit eval index for {scene['dataset']}/{scene['scene_id']}")
            if args.paper_counts and scene['dataset'] not in {'acid', 'realestate10k'}:
                if {len(r['reference_indices']) for r in rows} != {2, 3, 6}:
                    raise ValueError('Each paper scene needs all three view-count settings')
        for row in rows:
            samples.append({**row, **{k: scene[k] for k in ('dataset', 'scene_id', 'split', 'source_id', 'source_split')},
                            'sequence': str(seqpath), 'sequence_sha256': sha256(seqpath)})
    if len(consumed) != len(records):
        raise ValueError('Eval index contains scenes absent from catalog or assigned to training')
    ids = [(r['dataset'], r['scene_id'], r['sample_id']) for r in samples]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate sample_id within a scene')
    write_json(args.out, {'version': 1, 'seed': args.seed, 'catalog_sha256': sha256(args.catalog),
                         'eval_index_sha256': sha256(args.eval_index) if args.eval_index else None,
                         'paper_scene_counts_checked': args.paper_counts,
                         'resolutions': getattr(args, 'resolutions', SIZES), 'samples': samples})


def materialize(args):
    plan = read_json(args.plan)
    h, w = plan.get('resolutions', SIZES)[args.stage]
    selected = [r for r in plan['samples'] if (r['split'] == 'eval') == (args.stage == 'eval')]
    if not selected:
        raise ValueError('No samples for this stage')
    root = Path(args.out).resolve()
    if root.exists():
        raise FileExistsError(f'Output must be a new directory: {root}')
    root.mkdir(parents=True)
    for row in selected:
        seqpath = Path(row['sequence'])
        if sha256(seqpath) != row['sequence_sha256']:
            raise ValueError('Sequence changed since sampling plan was generated')
        seq = read_json(seqpath)
        dest = root / row['dataset'] / row['scene_id'] / row['sample_id']
        dest.mkdir(parents=True)
        out = {**row, 'stage': args.stage, 'height': h, 'width': w, 'resize': 'stretch_bilinear',
               'camera_convention': 'opencv_c2w', 'plan_sha256': sha256(args.plan), 'groups': {}}
        # Eval initialization has ONLY sparse references. Targets live in a separate directory.
        groups = {'initialization': row['initialization_indices'], 'reference': row['reference_indices'],
                  'target_gt': row['target_indices']}
        for group, indices in groups.items():
            (dest / group).mkdir()
            output_frames = []
            for i in indices:
                frame = seq['frames'][i]
                src = resolve(seqpath, frame['path'])
                relpath = f'{group}/{i:06d}.png'
                with Image.open(src) as im:
                    if im.size != (frame['width'], frame['height']):
                        raise ValueError(f'Image dimensions changed: {src}')
                    im.convert('RGB').resize((w, h), Image.Resampling.BILINEAR).save(dest / relpath)
                result = {**frame, 'path': relpath, 'original_path': str(src), 'width': w, 'height': h,
                          'original_sha256': sha256(src)}
                if 'K' in frame:
                    # Pixel-center affine transform consistent with PIL resizing.
                    sx, sy = w/frame['width'], h/frame['height']
                    A = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]])
                    result['K'] = (A @ np.asarray(frame['K'])).tolist()
                    result['pixel_transform'] = A.tolist()
                output_frames.append(result)
            out['groups'][group] = output_frames
        write_json(dest / 'sample.json', out)
    write_json(root / 'preparation.json', {'stage': args.stage, 'samples': len(selected), 'plan_sha256': sha256(args.plan)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('index-images'); q.add_argument('--images', required=True); q.add_argument('--cameras'); q.add_argument('--out', required=True); q.set_defaults(func=index_images)
    q = sub.add_parser('index-re10k'); q.add_argument('--metadata', required=True); q.add_argument('--images', required=True); q.add_argument('--video', help='Optional LOCAL video; ffmpeg extraction only'); q.add_argument('--out', required=True); q.set_defaults(func=index_re10k)
    q = sub.add_parser('validate'); q.add_argument('--catalog', required=True); q.add_argument('--paper-counts', action='store_true'); q.set_defaults(func=lambda a: print(f"Validated {len(validate_catalog(a.catalog, a.paper_counts))} scenes"))
    q = sub.add_parser('plan'); q.add_argument('--catalog', required=True); q.add_argument('--eval-index'); q.add_argument('--paper-counts', action='store_true'); q.add_argument('--seed', type=int, default=123); q.add_argument('--clips-per-scene', type=int, default=1); q.add_argument('--num-refs', type=int, default=2); q.add_argument('--out', required=True); q.set_defaults(func=make_plan)
    q = sub.add_parser('materialize'); q.add_argument('--plan', required=True); q.add_argument('--stage', choices=SIZES, required=True); q.add_argument('--out', required=True); q.set_defaults(func=materialize)
    args = p.parse_args()
    try:
        args.func(args)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as e:
        p.exit(2, f'ERROR: {e}\n')


if __name__ == '__main__':
    main()
