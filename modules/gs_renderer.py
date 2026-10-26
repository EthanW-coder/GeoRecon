"""
Differentiable 3D Gaussian Splatting renderer.

Provides the rendering interface for 3DGS optimization with optional
diffusion-based novel view supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class GSRenderConfig:
    """Configuration for Gaussian Splatting rendering."""
    sh_degree: int = 3
    white_background: bool = False
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    use_confidence: bool = False
    debug: bool = False


class GSRenderer:
    """
    Differentiable renderer for 3D Gaussian Splatting.

    Wraps the CUDA diff_gaussian_rasterization backend.
    """

    def __init__(self, config: GSRenderConfig = None):
        self.config = config or GSRenderConfig()

    def render(
        self,
        means3D: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
        sh_features: Optional[torch.Tensor] = None,
        camera_center: Optional[torch.Tensor] = None,
        world_view_transform: Optional[torch.Tensor] = None,
        full_proj_transform: Optional[torch.Tensor] = None,
        fov_x: float = 90.0,
        fov_y: float = 90.0,
        image_height: int = 512,
        image_width: int = 512,
        bg_color: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Render Gaussians to an image.

        Render using the installed CUDA rasterizer.

        Returns:
            dict with render, radii, viewspace_points, and visibility_filter
        """
        if bg_color is None:
            bg_color = torch.zeros(3, device=means3D.device)
            if self.config.white_background:
                bg_color = torch.ones(3, device=means3D.device)

        return self._render_cuda(
            means3D, opacities, scales, rotations, colors, sh_features,
            camera_center, world_view_transform, full_proj_transform,
            fov_x, fov_y, image_height, image_width, bg_color,
        )

    def _render_cuda(self, means3D, opacities, scales, rotations, colors,
                     sh_features, camera_center, world_view_transform,
                     full_proj_transform, fov_x, fov_y, H, W, bg_color):
        """Render using CUDA backend."""
        from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings

        tanfovx = np.tan(np.deg2rad(fov_x / 2))
        tanfovy = np.tan(np.deg2rad(fov_y / 2))

        raster_settings = GaussianRasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=self.config.sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=self.config.debug,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        if colors is None and sh_features is None:
            raise ValueError('Gaussian colors or spherical harmonics are required')
        screenspace = torch.zeros_like(means3D, requires_grad=True)
        if torch.is_grad_enabled():
            screenspace.retain_grad()
        outputs = rasterizer(
            means3D=means3D, means2D=screenspace,
            shs=sh_features if colors is None else None, colors_precomp=colors,
            opacities=opacities, scales=scales, rotations=rotations,
            cov3D_precomp=None,
        )
        rendered_image, radii = outputs[:2]
        return {
            'render': rendered_image.permute(1, 2, 0),
            'radii': radii, 'viewspace_points': screenspace,
            'visibility_filter': radii > 0,
        }


def render_gaussians(
    gaussian_model,
    camera,
    config: GSRenderConfig = None,
    bg_color: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Render a pre-defined Gaussian model from a camera viewpoint.

    Args:
        gaussian_model: GaussianModel instance or dict with GS parameters
        camera: camera object with FoV and pose attributes
        config: renderer configuration
        bg_color: background color

    Returns:
        dict with render, radii, viewspace_points, and visibility_filter
    """
    renderer = GSRenderer(config)

    if hasattr(gaussian_model, 'get_xyz'):
        means3D = gaussian_model.get_xyz
        opacities = gaussian_model.get_opacity
        scales = gaussian_model.get_scaling
        rotations = gaussian_model.get_rotation
        sh_features = gaussian_model.get_features
        colors = None
    elif isinstance(gaussian_model, dict):
        means3D = gaussian_model['means3D']
        opacities = gaussian_model['opacities']
        scales = gaussian_model['scales']
        rotations = gaussian_model['rotations']
        colors = gaussian_model.get('colors')
        sh_features = gaussian_model.get('sh_features')
    else:
        raise ValueError(f"Unsupported gaussian_model type: {type(gaussian_model)}")

    return renderer.render(
        means3D=means3D,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        colors=colors,
        sh_features=sh_features,
        camera_center=camera.camera_center,
        world_view_transform=camera.world_view_transform,
        full_proj_transform=camera.full_proj_transform,
        fov_x=camera.FoVx,
        fov_y=camera.FoVy,
        image_height=camera.image_height,
        image_width=camera.image_width,
        bg_color=bg_color,
    )
