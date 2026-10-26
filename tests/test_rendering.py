"""Gaussian rasterizer interface tests."""
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
from modules.gs_renderer import GSRenderer


class RenderingTests(unittest.TestCase):
    def test_official_rasterizer_pair_and_screen_gradients(self):
        extension = ModuleType('diff_gaussian_rasterization')
        extension.GaussianRasterizationSettings = lambda **kwargs: kwargs
        received = {}

        class Rasterizer:
            def __init__(self, raster_settings):
                self.settings = raster_settings

            def __call__(self, **kwargs):
                received.update(kwargs)
                value = kwargs['means3D'].sum() + kwargs['means2D'].sum() + kwargs['shs'].sum()
                return value.expand(3, 4, 4), torch.ones(2)

        extension.GaussianRasterizer = Rasterizer
        xyz = torch.ones(2, 3, requires_grad=True)
        features = torch.ones(2, 16, 3, requires_grad=True)
        with patch.dict(sys.modules, {'diff_gaussian_rasterization': extension}):
            result = GSRenderer().render(
                means3D=xyz, opacities=torch.ones(2, 1), scales=torch.ones(2, 3),
                rotations=torch.tensor([[1., 0, 0, 0]]).repeat(2, 1), sh_features=features,
                camera_center=torch.zeros(3), world_view_transform=torch.eye(4),
                full_proj_transform=torch.eye(4), image_height=4, image_width=4)
        result['render'].sum().backward()
        self.assertEqual(tuple(result['render'].shape), (4, 4, 3))
        self.assertEqual(tuple(received['means2D'].shape), (2, 3))
        self.assertIs(received['shs'], features)
        self.assertIsNotNone(result['viewspace_points'].grad)
        self.assertIsNotNone(features.grad)
        self.assertTrue(result['visibility_filter'].all())


if __name__ == '__main__':
    unittest.main()
