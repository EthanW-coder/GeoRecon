"""VGGT geometry inference with camera-to-world output in input-image coordinates."""
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F


class VGGTInterface:
    """Load a local VGGT checkpoint and reconstruct all input views jointly."""

    def __init__(self, model_path='./checkpoints/vggt/model.pt', device='cuda',
                 min_conf_thr=3.0, image_size=518):
        if image_size < 14 or image_size % 14:
            raise ValueError('VGGT image_size must be a positive multiple of 14')
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.min_conf_thr = min_conf_thr
        self.image_size = image_size
        self.model = None

    def load_model(self):
        """Initialize VGGT from local weights without network access."""
        if not self.model_path.is_file():
            raise FileNotFoundError(f'VGGT checkpoint not found: {self.model_path}')
        try:
            from vggt.models.vggt import VGGT
        except ImportError as exc:
            raise ImportError('Install VGGT using the instructions in README.md') from exc
        from models.checkpoints import load_state_dict_file
        model = VGGT()
        model.load_state_dict(load_state_dict_file(self.model_path), strict=True)
        self.model = model.to(self.device).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[0] == 0 or images.shape[1] != 3:
            raise ValueError('Expected nonempty RGB images with shape [N,3,H,W]')
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError('VGGT input images must be finite RGB values in [0,1]')
        if self.model is None:
            self.load_model()
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        images = images.to(self.device, dtype=torch.float32)
        n, _, h, w = images.shape
        model_w = self.image_size
        model_h = max(14, round(h * model_w / w / 14) * 14)
        resized = F.interpolate(images, size=(model_h, model_w), mode='bilinear', align_corners=False)
        if self.device.type == 'cuda':
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            context = torch.autocast(device_type='cuda', dtype=dtype)
        else:
            context = nullcontext()
        with context:
            predictions = self.model(resized.unsqueeze(0))
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            predictions['pose_enc'].float(), (model_h, model_w))
        w2c = torch.eye(4, device=self.device).repeat(n, 1, 1)
        w2c[:, :3, :] = extrinsics[0].float()
        poses = torch.linalg.inv(w2c)

        sx, sy = model_w / w, model_h / h
        transform = images.new_tensor([[sx, 0, (sx - 1) / 2],
                                       [0, sy, (sy - 1) / 2], [0, 0, 1]])
        K = torch.linalg.inv(transform).unsqueeze(0) @ intrinsics[0].float()
        depth = predictions['depth'][0, ..., 0].float()
        confidence = predictions['depth_conf'][0].float()
        depth = F.interpolate(depth[:, None], (h, w), mode='bilinear', align_corners=False)[:, 0]
        confidence = F.interpolate(confidence[:, None], (h, w), mode='bilinear', align_corners=False)[:, 0]
        yy, xx = torch.meshgrid(torch.arange(h, device=self.device),
                                torch.arange(w, device=self.device), indexing='ij')
        pixels = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).float()
        rays = torch.einsum('nij,hwj->nhwi', torch.linalg.inv(K), pixels)
        camera_points = rays * depth[..., None]
        world_points = torch.einsum('nij,nhwj->nhwi', poses[:, :3, :3], camera_points)
        world_points = world_points + poses[:, None, None, :3, 3]
        return {'depth': depth, 'poses': poses, 'intrinsics': K,
                'confidence': confidence, 'points3d': world_points}

    def run_reconstruction(self, images: torch.Tensor, clean_pc=True) -> Dict[str, torch.Tensor]:
        result = self(images)
        if clean_pc:
            result['confidence_mask'] = (
                (result['confidence'] >= self.min_conf_thr)
                & torch.isfinite(result['depth']) & (result['depth'] > 0))
        return result
