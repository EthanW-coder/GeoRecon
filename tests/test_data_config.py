"""Dataset configuration and split-selection tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import yaml

from data_preparation.configure import load_config, build_manifests
from data_preparation.prepare import materialize
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


class DataConfigTests(unittest.TestCase):
    def test_default_configuration(self):
        config = load_config(ROOT/'configs/data_processing.yaml')
        self.assertEqual(config['evaluation']['datasets']['dl3dv']['expected_scenes'], 20)
        self.assertEqual(config['evaluation']['datasets']['tanks']['expected_scenes'], 7)

    def test_scene_count_validation(self):
        config = load_config(ROOT/'configs/data_processing.yaml')
        config['evaluation']['datasets']['dl3dv'].update(enabled=True, scene_ids=['one'])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'config.yaml'; path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, 'expected 20'):
                load_config(path)

    def test_editable_indices_and_resolutions(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            frames = []
            for i in range(12):
                image = root/f'{i}.png'; Image.new('RGB', (8, 8)).save(image)
                frames.append({'path': image.name, 'source_index': i, 'width': 8, 'height': 8})
            (root/'sequence.json').write_text(json.dumps({'frames': frames, 'camera_convention': 'opencv_c2w'}))
            (root/'catalog.json').write_text(json.dumps({'scenes': [dict(dataset='dl3dv', scene_id='chosen',
                source_id='video', split='eval', source_split='DL3DV-Evaluation', sequence='sequence.json')]}))
            cfg = load_config(ROOT/'configs/data_processing.yaml')
            cfg.update(catalog='catalog.json', output_dir='manifests')
            cfg['evaluation']['enforce_scene_counts'] = False
            cfg['evaluation']['frame_stride'] = 3
            cfg['evaluation']['datasets']['dl3dv'].update(enabled=True, scene_ids=['chosen'], samples=[dict(
                scene_id='chosen', sample_id='pair', reference_indices=[0, 3], target_indices=[1, 2])])
            cfg['resolutions']['eval'] = [16, 24]
            path = root/'config.yaml'; path.write_text(yaml.safe_dump(cfg))
            plan = build_manifests(path)
            content = json.loads(plan.read_text())
            self.assertEqual(content['samples'][0]['initialization_indices'], [0, 3])
            materialize(SimpleNamespace(plan=plan, stage='eval', out=root/'prepared'))
            with Image.open(root/'prepared/dl3dv/chosen/pair/reference/000000.png') as im:
                self.assertEqual(im.size, (24, 16))

    def test_easy_set_requires_two_inputs(self):
        from data_preparation.prepare import check_eval_record
        row = dict(reference_indices=[0, 5, 10], target_indices=[1], sample_id='easy',
                   protocol='reconx_easy', index_source='index.json')
        with self.assertRaises(ValueError):
            check_eval_record(row, 20, 'realestate10k')


if __name__ == '__main__':
    unittest.main()
