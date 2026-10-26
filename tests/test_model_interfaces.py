"""Local unit tests for geometry and video-model interfaces."""
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from models.vggt_interface import VGGTInterface
from models.depth_video_diffusion import DepthVideoDiffusion, VideoDiffusionConfig
from models.checkpoints import load_state_dict_file


class ImageTokens(nn.Module):
    def forward(self, images):
        return images.mean((1, 2, 3))[:, None, None].expand(-1, 16, 1024)


class VideoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_stage_model = nn.Linear(1, 1)
        self.cond_stage_model = nn.Linear(1, 1)
        self.embedder = ImageTokens()
        self.image_proj_model = nn.Identity()
        self.denoiser = nn.Conv3d(8, 4, 1)
        self.parameterization = 'v'
        self.num_timesteps = 4
        self.use_dynamic_rescale = True
        self.register_buffer('alphas_cumprod', torch.tensor([.9, .6, .3, 0.]))
        self.register_buffer('scale_arr', torch.tensor([1., .9, .8, .7]))
        self.context_lengths = []

    def encode_first_stage(self, images):
        return F.avg_pool2d(images, 8).mean(1, keepdim=True).expand(-1, 4, -1, -1)

    def get_learned_conditioning(self, captions):
        return self.alphas_cumprod.new_zeros(len(captions), 77, 1024)

    def apply_model(self, images, steps, cond, fs=None):
        self.context_lengths.append(cond['c_crossattn'][0].shape[1])
        semantic = cond['c_crossattn'][0].mean((1, 2))[:, None, None, None, None]
        return self.denoiser(torch.cat([images, cond['c_concat'][0]], dim=1)) + semantic

    def decode_first_stage(self, latents):
        b, _, t, h, w = latents.shape
        frames = latents[:, :3].permute(0, 2, 1, 3, 4).reshape(b*t, 3, h, w)
        frames = F.interpolate(frames, scale_factor=8)
        return frames.reshape(b, t, 3, h*8, w*8).permute(0, 2, 1, 3, 4)


class ModelInterfaceTests(unittest.TestCase):
    def test_missing_weights_fail_locally(self):
        with self.assertRaises(FileNotFoundError):
            VGGTInterface('/nonexistent/vggt/model.pt', device='cpu').load_model()
        with self.assertRaises(ValueError):
            DepthVideoDiffusion()

    def test_local_checkpoint_formats(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'weights.pt'
            torch.save({'state_dict': {'weight': torch.ones(2)}, 'step': 5}, path)
            self.assertTrue(torch.equal(load_state_dict_file(path)['weight'], torch.ones(2)))

    def test_vggt_resolution_and_camera_convention(self):
        class Geometry(nn.Module):
            def forward(self, images):
                b, n, _, h, w = images.shape
                return {'pose_enc': torch.zeros(b, n, 9),
                        'depth': torch.full((b, n, h, w, 1), 2.),
                        'depth_conf': torch.full((b, n, h, w), 4.)}
        def cameras(enc, size):
            h, w = size
            n = enc.shape[1]
            w2c = torch.eye(4)[None, None, :3].repeat(1, n, 1, 1)
            w2c[..., 0, 3] = 1
            K = torch.tensor([[20., 0, (w-1)/2], [0, 20., (h-1)/2], [0, 0, 1.]])
            return w2c, K[None, None].repeat(1, n, 1, 1)
        util = ModuleType('vggt.utils.pose_enc'); util.pose_encoding_to_extri_intri = cameras
        with patch.dict(sys.modules, {'vggt': ModuleType('vggt'), 'vggt.utils': ModuleType('vggt.utils'), 'vggt.utils.pose_enc': util}):
            interface = VGGTInterface(device='cpu', image_size=28)
            interface.model = Geometry()
            result = interface.run_reconstruction(torch.zeros(2, 3, 12, 16))
        self.assertEqual(tuple(result['depth'].shape), (2, 12, 16))
        self.assertTrue(torch.all(result['depth'] == 2))
        self.assertTrue(torch.all(result['poses'][:, 0, 3] == -1))
        self.assertAlmostEqual(result['intrinsics'][0, 0, 2].item(), 7.5, places=5)
        self.assertAlmostEqual(result['intrinsics'][0, 1, 2].item(), 5.5, places=5)
        self.assertTrue(result['confidence_mask'].all())
        self.assertEqual(tuple(result['points3d'].shape), (2, 12, 16, 3))

    def inputs(self):
        return dict(I_render=torch.rand(1, 3, 3, 16, 16), D_mask=torch.ones(1, 3, 1, 16, 16),
                    I_con=torch.rand(1, 2, 3, 16, 16), T_novel=torch.zeros(1, 3, 6))

    def test_backbone_training_and_frozen_encoders(self):
        model = DepthVideoDiffusion(VideoDiffusionConfig(condition_dropout=0), VideoBackbone()).train()
        inputs = self.inputs()
        result = model(**inputs, I_gt=torch.rand_like(inputs['I_render']))
        result['loss'].backward()
        self.assertTrue(torch.isfinite(result['loss']))
        self.assertIsNotNone(model.geometric_encoder.zero_conv.conv.weight.grad)
        self.assertIsNotNone(model.backbone.denoiser.weight.grad)
        self.assertFalse(model.backbone.first_stage_model.training)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.first_stage_model.parameters()))
        self.assertEqual(model.backbone.context_lengths[-1], 77+32+3)

    def test_ddim_zero_terminal_alpha_cfg_and_rgb(self):
        model = DepthVideoDiffusion(base_diffusion_model=VideoBackbone()).eval()
        inputs = self.inputs()
        torch.manual_seed(20)
        first = model.sample(**inputs, num_inference_steps=3, guidance_scale=1)
        torch.manual_seed(20)
        second = model.sample(**inputs, num_inference_steps=3, guidance_scale=3)
        self.assertEqual(tuple(first.shape), (1, 3, 3, 16, 16))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(((first >= 0) & (first <= 1)).all())
        self.assertFalse(torch.allclose(first, second))


if __name__ == '__main__':
    unittest.main()
