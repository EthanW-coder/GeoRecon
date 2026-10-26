"""Build dataset manifests from an editable YAML configuration."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml

try:
    from . import prepare
except ImportError:
    import prepare


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError('Expected a YAML mapping')
    sample = config['sampling']
    init, frames = sample['initialization_frames'], sample['video_frames']
    refs, clips = sample['reference_views'], sample['clips_per_scene']
    numbers = [init, frames, refs, clips, sample['stride_min'], sample['stride_max']]
    if any(type(v) is not int or v <= 0 for v in numbers):
        raise ValueError('Sampling parameters must be positive integers')
    if not 2 <= refs <= frames <= init or sample['stride_min'] > sample['stride_max']:
        raise ValueError('Invalid reference count, video length, or sampling stride range')
    evaluation = config['evaluation']
    if not evaluation['view_counts'] or any(type(v) is not int or v < 2 for v in evaluation['view_counts']):
        raise ValueError('Evaluation view counts must be integers >= 2')
    if type(evaluation['frame_stride']) is not int or evaluation['frame_stride'] < 1:
        raise ValueError('Evaluation frame_stride must be positive')
    for dataset, settings in evaluation['datasets'].items():
        if dataset not in prepare.DATASETS:
            raise ValueError(f'Unknown dataset: {dataset}')
        if type(settings['enabled']) is not bool:
            raise ValueError('enabled must be a YAML boolean')
        ids = settings['scene_ids']
        if not isinstance(ids, list) or any(not isinstance(s, str) for s in ids) or len(set(ids)) != len(ids):
            raise ValueError(f'Invalid or duplicate scene_ids for {dataset}')
        if settings['enabled']:
            if not ids:
                raise ValueError(f'Configure scene_ids for {dataset}')
            expected = settings.get('expected_scenes')
            if evaluation['enforce_scene_counts'] and expected is not None and len(ids) != expected:
                raise ValueError(f'{dataset}: expected {expected} scenes, got {len(ids)}')
    for stage, size in config['resolutions'].items():
        if stage not in prepare.SIZES or len(size) != 2 or any(type(x) is not int or x < 8 or x % 8 for x in size):
            raise ValueError('Resolutions must be [height,width], positive multiples of 8')
    return config


def build_manifests(path):
    path = Path(path).resolve()
    config = load_config(path)
    catalog_path = prepare.resolve(path, config['catalog'])
    catalog = prepare.read_json(catalog_path)
    selections = config['evaluation']['datasets']
    scenes, found = [], set()
    for scene in catalog['scenes']:
        ds, sid = scene['dataset'], scene['scene_id']
        if scene['split'] == 'train':
            selected = True
        else:
            settings = selections.get(ds, {})
            selected = settings.get('enabled', False) and sid in settings.get('scene_ids', [])
        if selected:
            scenes.append({**scene, 'sequence': str(prepare.resolve(catalog_path, scene['sequence']))})
            if scene['split'] == 'eval':
                found.add((ds, sid))
    records = []
    for ds, settings in selections.items():
        if not settings['enabled']:
            continue
        for sid in settings['scene_ids']:
            if (ds, sid) not in found:
                raise ValueError(f'Evaluation scene absent from catalog: {ds}/{sid}')
        rows = list(settings.get('samples', []))
        source = str(path)
        if settings.get('index_file'):
            source_path = prepare.resolve(path, settings['index_file'])
            source = f'{source_path.name}:sha256:{prepare.sha256(source_path)}'
            external = prepare.read_json(source_path)
            rows.extend(external['samples'])
        if not rows:
            raise ValueError(f'Configure samples or index_file for {ds}')
        for row in rows:
            if row.get('dataset', ds) != ds:
                raise ValueError(f'Index dataset does not match {ds}')
            if row['scene_id'] not in settings['scene_ids']:
                continue
            records.append({**row, 'dataset': ds, 'protocol': settings.get('protocol', row.get('protocol', 'stride')),
                            'index_source': row.get('index_source', source)})
    output = prepare.resolve(path, config['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    catfile, idxfile = output/'catalog.json', output/'eval_index.json'
    prepare.write_json(catfile, {'scenes': scenes})
    prepare.write_json(idxfile, {'samples': records})
    prepare.make_plan(SimpleNamespace(
        catalog=catfile, eval_index=idxfile, paper_counts=False,
        seed=config['seed'], clips_per_scene=config['sampling']['clips_per_scene'],
        num_refs=config['sampling']['reference_views'], sampling=config['sampling'],
        view_counts=config['evaluation']['view_counts'], frame_stride=config['evaluation']['frame_stride'],
        out=output/'plan.json', resolutions=config['resolutions'],
    ))
    prepare.write_json(output/'configuration.json', config)
    return output/'plan.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--check', action='store_true', help='Validate YAML settings without reading image files')
    args = parser.parse_args()
    try:
        if args.check:
            load_config(args.config)
            print('Configuration valid')
        else:
            print(build_manifests(args.config))
    except (ValueError, KeyError, OSError, TypeError, yaml.YAMLError) as exc:
        parser.exit(2, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
