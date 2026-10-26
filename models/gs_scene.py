"""
3D Gaussian Splatting scene representation and optimization.

Defines the Gaussian model (learnable parameters) and scene representation
(combining model, cameras, and rendering pipeline).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from modules.gs_renderer import GSRenderer, GSRenderConfig


@dataclass
class Camera:
    """Camera representation for Gaussian Splatting rendering."""
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    R: torch.Tensor  # (3, 3) rotation
    T: torch.Tensor  # (3,) translation
    world_view_transform: torch.Tensor  # (4, 4) world-to-view
    projection_matrix: torch.Tensor  # (4, 4) projection
    full_proj_transform: torch.Tensor  # (4, 4) combined
    camera_center: torch.Tensor  # (3,) world-space center
    image: Optional[torch.Tensor] = None  # (3, H, W) ground truth image
    image_name: str = ""


@dataclass
class PseudoCamera(Camera):
    """Camera with diffusion-generated pseudo-ground-truth for novel view supervision."""
    pseudo_gt: Optional[torch.Tensor] = None  # (3, H, W) generated image
    mask: Optional[torch.Tensor] = None  # (H, W) valid region mask
    depth: Optional[torch.Tensor] = None  # (H, W) depth map


class GaussianModel(nn.Module):
    """
    3D Gaussian Splatting model with learnable parameters.

    Attributes (learnable):
        _xyz: (N, 3) 3D positions
        _features_dc: (N, 1, 3) DC component of SH colors
        _features_rest: (N, 15, 3) higher-order SH coefficients
        _scaling: (N, 3) per-axis scale factors
        _rotation: (N, 4) quaternion rotations
        _opacity: (N, 1) opacity values
    """

    def __init__(
        self,
        sh_degree: int = 3,
        white_background: bool = False,
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        # Learnable parameters (initialized later)
        self._xyz = nn.Parameter(torch.empty(0, 3))
        self._features_dc = nn.Parameter(torch.empty(0, 1, 3))
        self._features_rest = nn.Parameter(torch.empty(0, 15, 3))
        self._scaling = nn.Parameter(torch.empty(0, 3))
        self._rotation = nn.Parameter(torch.empty(0, 4))
        self._opacity = nn.Parameter(torch.empty(0, 1))

        # Non-learnable tracking
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0, 1)
        self.denom = torch.empty(0, 1)

        self.optimizer = None
        self.spatial_lr_scale = 1.0

    def create_from_point_cloud(
        self,
        points: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
        spatial_lr_scale: float = 1.0,
    ):
        """
        Initialize Gaussians from a point cloud.

        Args:
            points: (N, 3) 3D point positions
            colors: (N, 3) RGB colors in [0, 1]
            spatial_lr_scale: spatial learning rate scale
        """
        N = points.shape[0]
        device = points.device
        self.spatial_lr_scale = spatial_lr_scale

        # Initialize positions
        self._xyz = nn.Parameter(points.float().to(device))

        # Initialize colors as SH DC component
        if colors is None:
            # Random colors
            fused_color = (torch.rand(N, 3, device=device) * 0.5 + 0.25).clamp(0, 1)
        else:
            fused_color = colors.float().to(device).clamp(0, 1)

        # RGB -> SH(0) = (RGB - 0.5) / 0.28209479
        sh0 = (fused_color - 0.5) / 0.28209479177387814
        self._features_dc = nn.Parameter(sh0.unsqueeze(1).float())
        self._features_rest = nn.Parameter(
            torch.zeros(N, (self.max_sh_degree + 1) ** 2 - 1, 3, device=device)
        )

        # Compute nearest neighbor distances for scale initialization
        try:
            from simple_knn._C import distCUDA2
            dists = distCUDA2(points.float().cuda())
            dists = dists.to(device)
        except ImportError:
            if N > 10000:
                idx = torch.randperm(N, device=device)[:5000]
                pts_sample = points[idx]
                dists = torch.cdist(points, pts_sample).min(dim=1)[0]
            else:
                dists = torch.cdist(points, points).topk(k=2, largest=False, dim=1)[0][:, 1]

        scales = torch.log(torch.sqrt(dists)).unsqueeze(-1).repeat(1, 3)
        self._scaling = nn.Parameter(scales)

        # Quaternion identity
        rots = torch.zeros(N, 4, device=device)
        rots[:, 0] = 1.0
        self._rotation = nn.Parameter(rots)

        # Initialize opacity to ~0.1
        inv_sigmoid_val = torch.log(torch.tensor(0.1) / (1 - torch.tensor(0.1)))
        self._opacity = nn.Parameter(torch.full((N, 1), inv_sigmoid_val, device=device))

        # Tracking buffers
        self.max_radii2D = torch.zeros(N, device=device)
        self.xyz_gradient_accum = torch.zeros(N, 1, device=device)
        self.denom = torch.zeros(N, 1, device=device)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return F.normalize(self._rotation, dim=1)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat([features_dc, features_rest], dim=1)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def get_covariance(self, scaling_modifier: float = 1.0):
        """Compute 3D covariance matrices for all Gaussians."""
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    @staticmethod
    def covariance_activation(scales, scaling_modifier, rotations):
        """Convert scales and quaternions to 3D covariance matrices."""
        from utils.geometry_utils import build_scaling_rotation_matrix

        L = build_scaling_rotation_matrix(scales * scaling_modifier, rotations)
        actual_covariance = L @ L.transpose(1, 2)
        symm = actual_covariance[:, [0, 0, 1], [0, 1, 1]]
        return actual_covariance, symm

    def setup_optimizer(
        self,
        position_lr_init: float = 0.00016,
        position_lr_final: float = 0.0000016,
        position_lr_delay_mult: float = 0.01,
        position_lr_max_steps: int = 10000,
        feature_lr: float = 0.0025,
        opacity_lr: float = 0.05,
        scaling_lr: float = 0.005,
        rotation_lr: float = 0.001,
    ):
        """Setup Adam optimizer with per-parameter learning rates."""
        l = [
            {'params': [self._xyz],
             'lr': position_lr_init * self.spatial_lr_scale,
             "name": "xyz"},
            {'params': [self._features_dc],
             'lr': feature_lr, "name": "f_dc"},
            {'params': [self._features_rest],
             'lr': feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity],
             'lr': opacity_lr, "name": "opacity"},
            {'params': [self._scaling],
             'lr': scaling_lr, "name": "scaling"},
            {'params': [self._rotation],
             'lr': rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = (
            position_lr_init, position_lr_final,
            position_lr_delay_mult, position_lr_max_steps
        )

    def update_learning_rate(self, iteration: int):
        """Update position learning rate with exponential scheduling."""
        if self.optimizer is None:
            return

        init_lr, final_lr, delay_mult, max_steps = self.xyz_scheduler_args
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                if iteration < delay_mult * max_steps:
                    progress = iteration / (delay_mult * max_steps)
                    lr = init_lr * (1 - progress) + final_lr * progress
                else:
                    progress = min(
                        (iteration - delay_mult * max_steps) / ((1 - delay_mult) * max_steps), 1.0
                    )
                    lr = init_lr * (1 - progress) + final_lr * progress
                param_group['lr'] = lr * self.spatial_lr_scale

    def add_densification_stats(
        self,
        viewspace_point_tensor: torch.Tensor,
        update_filter: torch.Tensor,
    ):
        """
        Accumulate gradient statistics for densification decisions.

        Args:
            viewspace_point_tensor: (N, 2) points in view space
            update_filter: (N,) boolean mask of visible points
        """
        grad_norm = torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.xyz_gradient_accum[update_filter] += grad_norm
        self.denom[update_filter] += 1

    def densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        extent: float,
        max_screen_size: Optional[int] = None,
        iteration: int = 0,
    ):
        """
        Perform densification (clone + split) and pruning.

        Args:
            max_grad: gradient threshold for densification
            min_opacity: minimum opacity for pruning
            extent: scene extent
            max_screen_size: optional max screen size for pruning
            iteration: current training iteration
        """
        grads = self.xyz_gradient_accum / self.denom.clamp(min=1)
        grads[grads.isnan()] = 0.0

        # Densify
        self._densify_and_clone(grads, max_grad, extent)
        self._densify_and_split(grads, max_grad, extent)

        # Prune
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size is not None:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = prune_mask | big_points_vs
        self._prune_points(prune_mask)

    def _densify_and_clone(self, grads, grad_threshold, extent):
        """Clone Gaussians with high gradient and small scale."""
        selected_pts_mask = torch.norm(grads, dim=-1) >= grad_threshold
        size_threshold = 0.01 * extent

        scales = self.get_scaling
        small_mask = torch.max(scales, dim=1)[0] <= size_threshold

        clone_mask = selected_pts_mask & small_mask
        if clone_mask.sum() == 0:
            return

        self._duplicate_points(clone_mask)

    def _densify_and_split(self, grads, grad_threshold, extent):
        """Split Gaussians with high gradient and large scale into two."""
        N = self._xyz.shape[0]
        selected_pts_mask = torch.norm(grads, dim=-1) >= grad_threshold
        size_threshold = 0.01 * extent

        scales = self.get_scaling
        large_mask = torch.max(scales, dim=1)[0] > size_threshold
        split_mask = selected_pts_mask & large_mask

        if split_mask.sum() == 0:
            return

        n_clones = split_mask.sum().item()
        new_xyz = self._xyz[split_mask].clone()
        new_features_dc = self._features_dc[split_mask].clone()
        new_features_rest = self._features_rest[split_mask].clone()
        new_opacity = self._opacity[split_mask].clone()
        new_scaling = self._scaling[split_mask].clone() - torch.log(torch.tensor(1.6))
        new_rotation = self._rotation[split_mask].clone()

        # Append new parameters
        self._xyz = nn.Parameter(torch.cat([self._xyz, new_xyz], dim=0))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, new_features_dc], dim=0))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, new_features_rest], dim=0))
        self._opacity = nn.Parameter(torch.cat([self._opacity, new_opacity], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling, new_scaling], dim=0))
        self._rotation = nn.Parameter(torch.cat([self._rotation, new_rotation], dim=0))

        # Update tracking buffers
        device = self._xyz.device
        new_max_radii2D = torch.zeros(n_clones, device=device)
        new_xyz_gradient = torch.zeros(n_clones, 1, device=device)
        new_denom = torch.zeros(n_clones, 1, device=device)
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D])
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, new_xyz_gradient])
        self.denom = torch.cat([self.denom, new_denom])

        # Update optimizer
        if self.optimizer is not None:
            self._add_params_to_optimizer(n_clones)

    def _duplicate_points(self, mask):
        """Duplicate existing Gaussians."""
        n_dupes = mask.sum().item()
        new_xyz = self._xyz[mask].clone()
        new_features_dc = self._features_dc[mask].clone()
        new_features_rest = self._features_rest[mask].clone()
        new_opacity = self._opacity[mask].clone()
        new_scaling = self._scaling[mask].clone()
        new_rotation = self._rotation[mask].clone()

        self._xyz = nn.Parameter(torch.cat([self._xyz, new_xyz], dim=0))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, new_features_dc], dim=0))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, new_features_rest], dim=0))
        self._opacity = nn.Parameter(torch.cat([self._opacity, new_opacity], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling, new_scaling], dim=0))
        self._rotation = nn.Parameter(torch.cat([self._rotation, new_rotation], dim=0))

        device = self._xyz.device
        self.max_radii2D = torch.cat([
            self.max_radii2D, torch.zeros(n_dupes, device=device)
        ])
        self.xyz_gradient_accum = torch.cat([
            self.xyz_gradient_accum, torch.zeros(n_dupes, 1, device=device)
        ])
        self.denom = torch.cat([
            self.denom, torch.zeros(n_dupes, 1, device=device)
        ])

        if self.optimizer is not None:
            self._add_params_to_optimizer(n_dupes)

    def _prune_points(self, mask):
        """Remove Gaussians that satisfy the pruning mask."""
        valid_mask = ~mask
        if mask.sum() == 0:
            return

        self._xyz = nn.Parameter(self._xyz[valid_mask])
        self._features_dc = nn.Parameter(self._features_dc[valid_mask])
        self._features_rest = nn.Parameter(self._features_rest[valid_mask])
        self._opacity = nn.Parameter(self._opacity[valid_mask])
        self._scaling = nn.Parameter(self._scaling[valid_mask])
        self._rotation = nn.Parameter(self._rotation[valid_mask])

        self.max_radii2D = self.max_radii2D[valid_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_mask]
        self.denom = self.denom[valid_mask]

        if self.optimizer is not None:
            self._remove_params_from_optimizer(mask)

    def _add_params_to_optimizer(self, n_new: int):
        """Extend optimizer state for newly added parameters."""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                for key in stored_state:
                    if key == 'step':
                        continue
                    if isinstance(stored_state[key], torch.Tensor):
                        stored_state[key] = torch.cat([
                            stored_state[key], torch.zeros_like(stored_state[key][:n_new])
                        ], dim=0)

        # Extend param groups
        for group, new_param in zip(
            self.optimizer.param_groups,
            [self._xyz, self._features_dc, self._features_rest,
             self._opacity, self._scaling, self._rotation]
        ):
            stored_state = self.optimizer.state.get(group['params'][0], None)
            group['params'][0] = new_param
            if stored_state is not None:
                self.optimizer.state[group['params'][0]] = stored_state

    def _remove_params_from_optimizer(self, mask):
        """Shrink optimizer state for pruned parameters."""
        valid_mask = ~mask
        for group, new_param in zip(
            self.optimizer.param_groups,
            [self._xyz, self._features_dc, self._features_rest,
             self._opacity, self._scaling, self._rotation]
        ):
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                for key in stored_state:
                    if stored_state[key].shape[0] == mask.shape[0]:
                        stored_state[key] = stored_state[key][valid_mask]
            group['params'][0] = new_param
            if stored_state is not None:
                self.optimizer.state[group['params'][0]] = stored_state

    def reset_opacity(self):
        """Reset opacity for all Gaussians to a lower value."""
        inv_sigmoid_val = torch.log(torch.tensor(0.01) / (1 - torch.tensor(0.01)))
        opacities_new = inv_sigmoid_val * torch.ones_like(self._opacity)
        opacities_new = torch.min(self._opacity, opacities_new)
        self._opacity = nn.Parameter(opacities_new)

    def append_points_from_pcd(
        self,
        new_points: torch.Tensor,
        new_colors: Optional[torch.Tensor] = None,
    ):
        """
        Append new Gaussians from an additional point cloud (e.g., from diffusion depths).

        Args:
            new_points: (M, 3) new 3D positions
            new_colors: (M, 3) RGB colors
        """
        if new_points.shape[0] == 0:
            return

        M = new_points.shape[0]
        N_old = self._xyz.shape[0]
        device = self._xyz.device

        # Append positions
        self._xyz = nn.Parameter(torch.cat([self._xyz, new_points.float().to(device)], dim=0))

        # Colors
        if new_colors is None:
            new_colors = torch.rand(M, 3, device=device) * 0.5 + 0.25
        sh0 = (new_colors.float().to(device).clamp(0, 1) - 0.5) / 0.28209479177387814
        self._features_dc = nn.Parameter(torch.cat([
            self._features_dc, sh0.unsqueeze(1)
        ], dim=0))
        self._features_rest = nn.Parameter(torch.cat([
            self._features_rest,
            torch.zeros(M, (self.max_sh_degree + 1) ** 2 - 1, 3, device=device)
        ], dim=0))

        # Default scale and rotation
        self._scaling = nn.Parameter(torch.cat([
            self._scaling,
            torch.zeros(M, 3, device=device)
        ], dim=0))
        new_rots = torch.zeros(M, 4, device=device)
        new_rots[:, 0] = 1.0
        self._rotation = nn.Parameter(torch.cat([self._rotation, new_rots], dim=0))

        # Opacity
        inv_sig = torch.log(torch.tensor(0.1) / (1 - torch.tensor(0.1)))
        self._opacity = nn.Parameter(torch.cat([
            self._opacity,
            torch.full((M, 1), inv_sig, device=device)
        ], dim=0))

        # Tracking buffers
        self.max_radii2D = torch.cat([
            self.max_radii2D, torch.zeros(M, device=device)
        ])
        self.xyz_gradient_accum = torch.cat([
            self.xyz_gradient_accum, torch.zeros(M, 1, device=device)
        ])
        self.denom = torch.cat([
            self.denom, torch.zeros(M, 1, device=device)
        ])

        # Update optimizer
        if self.optimizer is not None:
            self._add_params_to_optimizer(M)


class SceneRepresentation:
    """
    Complete scene representation combining Gaussian model, cameras, and rendering.

    Manages the full 3DGS optimization workflow:
    - Training view selection
    - Rendering
    - Pseudo-camera (diffusion-guided) supervision
    - Densification scheduling
    """

    def __init__(
        self,
        gaussian_model: GaussianModel,
        train_cameras: List[Camera],
        test_cameras: Optional[List[Camera]] = None,
        render_config: GSRenderConfig = None,
    ):
        self.gaussian_model = gaussian_model
        self.train_cameras = train_cameras
        self.test_cameras = test_cameras or []
        self.render_config = render_config or GSRenderConfig()
        self.renderer = GSRenderer(self.render_config)

        # Pseudo-camera stack for diffusion-guided supervision
        self.pseudo_camera_stack: List[PseudoCamera] = []

        # Point cloud (initialization reference)
        self.point_cloud: Optional[torch.Tensor] = None

    def add_pseudo_cameras(self, pseudo_cams: List[PseudoCamera]):
        """Add diffusion-generated pseudo cameras for novel-view supervision."""
        self.pseudo_camera_stack.extend(pseudo_cams)

    def get_pseudo_camera(self) -> Optional[PseudoCamera]:
        """Pop a pseudo-camera from the stack for supervision."""
        if len(self.pseudo_camera_stack) == 0:
            return None
        return self.pseudo_camera_stack.pop(0)

    def render_view(
        self,
        camera: Camera,
        bg_color: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Render the scene from a given camera viewpoint.

        Args:
            camera: camera parameters
            bg_color: background color

        Returns:
            dict with 'render', 'depth', 'alpha', 'radii'
        """
        return self.renderer.render(
            means3D=self.gaussian_model.get_xyz,
            opacities=self.gaussian_model.get_opacity,
            scales=self.gaussian_model.get_scaling,
            rotations=self.gaussian_model.get_rotation,
            sh_features=self.gaussian_model.get_features,
            camera_center=camera.camera_center,
            world_view_transform=camera.world_view_transform,
            full_proj_transform=camera.full_proj_transform,
            fov_x=camera.FoVx,
            fov_y=camera.FoVy,
            image_height=camera.image_height,
            image_width=camera.image_width,
            bg_color=bg_color,
        )

    def create_camera_from_pose(
        self,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        image_height: int,
        image_width: int,
        image: Optional[torch.Tensor] = None,
        image_name: str = "",
    ) -> Camera:
        """
        Create a Camera object from pose and intrinsics matrices.

        Args:
            c2w: (4, 4) camera-to-world matrix
            intrinsics: (3, 3) camera intrinsics
            image_height: image height
            image_width: image width
            image: optional ground truth image
            image_name: identifier for the camera

        Returns:
            Camera object
        """
        from utils.geometry_utils import get_world_to_view_matrix, get_projection_matrix, get_fov

        device = c2w.device

        # Extract R, T from camera-to-world
        w2c = torch.inverse(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]

        # Compute FoV
        fov_x, fov_y = get_fov(intrinsics, (image_width, image_height))

        # Build transforms
        world_view_transform = get_world_to_view_matrix(R, T).T
        projection_matrix = get_projection_matrix(
            znear=0.01, zfar=100.0, fov_x=fov_x, fov_y=fov_y
        ).to(device).T
        full_proj_transform = world_view_transform @ projection_matrix

        camera_center = c2w[:3, 3]

        return Camera(
            image_height=image_height,
            image_width=image_width,
            FoVx=fov_x,
            FoVy=fov_y,
            R=R,
            T=T,
            world_view_transform=world_view_transform,
            projection_matrix=projection_matrix,
            full_proj_transform=full_proj_transform,
            camera_center=camera_center,
            image=image,
            image_name=image_name,
        )
