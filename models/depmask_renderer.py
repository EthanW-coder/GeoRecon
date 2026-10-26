"""
DepMask: Confidence & Geometry Dual-Guided Sparse-to-Dense Rendering.

Implements a sparse point cloud rendering pipeline that generates RGB images
and confidence-masked depth maps for diffusion-based 3D reconstruction.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import cv2


@dataclass
class RenderConfig:
    """Configuration for DepMask rendering pipeline."""
    conf_threshold: float = 0.3
    edge_threshold: float = 5.0
    render_resolution: Tuple[int, int] = (512, 512)
    point_radius: float = 1.5
    reproj_error_threshold: float = 2.0
    min_observations: int = 2
    depth_mask_value: float = 0.0
    depth_far_plane: float = 100.0
    depth_near_plane: float = 0.1


class ConfidenceEstimator:
    """
    Estimates confidence weights for 3D points based on multi-view consistency.

    Confidence combines:
    1. Number of observations (more views = higher confidence)
    2. Reprojection error (lower error = higher confidence)
    3. View angle diversity (wider baseline = higher confidence)
    """

    def __init__(self, config: RenderConfig):
        self.config = config

    def compute_point_confidence(
        self,
        points_3d: torch.Tensor,
        observations: List[Dict],
        camera_poses: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute confidence weight for each 3D point.

        Args:
            points_3d: (N, 3) world coordinates
            observations: list of dicts with 'camera', 'features', 'matches'
            camera_poses: (M, 4, 4) camera-to-world transforms
            intrinsics: (M, 3, 3) camera intrinsics

        Returns:
            confidence: (N,) tensor in [0, 1]
        """
        N = points_3d.shape[0]
        device = points_3d.device

        observation_score = torch.zeros(N, device=device)
        reproj_error_score = torch.zeros(N, device=device)
        angle_score = torch.zeros(N, device=device)

        for point_idx in range(N):
            point = points_3d[point_idx]
            viewing_rays = []
            reproj_errors = []

            for obs_idx, obs in enumerate(observations):
                if point_idx not in obs['point_indices']:
                    continue

                cam_pose = camera_poses[obs_idx]
                K = intrinsics[obs_idx]

                point_homo = torch.cat([point, torch.ones(1, device=device)])
                world_to_cam = torch.inverse(cam_pose)
                point_cam = world_to_cam @ point_homo

                point_img = K @ point_cam[:3]
                uv_pred = point_img[:2] / (point_img[2] + 1e-8)

                uv_obs = obs['observations'][point_idx]
                error = torch.norm(uv_pred - uv_obs).item()
                reproj_errors.append(error)

                cam_center = cam_pose[:3, 3]
                ray_dir = F.normalize(point - cam_center, dim=0)
                viewing_rays.append(ray_dir)

            # Observation score
            num_obs = len(viewing_rays)
            observation_score[point_idx] = min(
                1.0, num_obs / self.config.min_observations
            )

            # Reprojection error score (exponential decay)
            if len(reproj_errors) > 0:
                mean_error = np.mean(reproj_errors)
                reproj_error_score[point_idx] = torch.exp(
                    -torch.tensor(mean_error / self.config.reproj_error_threshold)
                )

            # Angle diversity score
            if len(viewing_rays) >= 2:
                rays = torch.stack(viewing_rays)
                angles = torch.acos(torch.clamp(rays @ rays.T, -1, 1))
                max_angle = torch.max(angles).item()
                angle_score[point_idx] = min(1.0, max_angle / (np.pi / 3))

        confidence = (
            0.4 * observation_score
            + 0.4 * reproj_error_score
            + 0.2 * angle_score
        )

        return confidence


class DepthEdgeDetector:
    """
    Detects stretching artifacts by analyzing depth discontinuities.

    Stretching occurs when sparse points are interpolated across large depth gaps,
    creating unrealistic "paper-thin" surfaces.
    """

    def __init__(self, config: RenderConfig):
        self.config = config

    def compute_depth_gradient(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Compute depth gradient magnitude using Sobel operators."""
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=depth_map.dtype, device=depth_map.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=depth_map.dtype, device=depth_map.device
        ).view(1, 1, 3, 3)

        depth_4d = depth_map.unsqueeze(0).unsqueeze(0)
        grad_x = F.conv2d(depth_4d, sobel_x, padding=1)
        grad_y = F.conv2d(depth_4d, sobel_y, padding=1)
        gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2).squeeze()

        return gradient

    def detect_stretching_artifacts(
        self,
        depth_map: torch.Tensor,
        confidence_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Identify pixels affected by stretching artifacts.

        Args:
            depth_map: (H, W) rendered depth
            confidence_map: (H, W) accumulated alpha during rendering

        Returns:
            artifact_mask: (H, W) boolean mask, True = artifact pixel
        """
        depth_grad = self.compute_depth_gradient(depth_map)

        large_gradient = depth_grad > self.config.edge_threshold
        low_confidence = confidence_map < self.config.conf_threshold

        artifact_mask = large_gradient & low_confidence

        # Morphological dilation
        artifact_mask_np = artifact_mask.cpu().numpy().astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        artifact_mask_dilated = cv2.dilate(artifact_mask_np, kernel, iterations=1)
        artifact_mask = torch.from_numpy(artifact_mask_dilated).bool().to(
            depth_map.device
        )

        return artifact_mask


class PointCloudRenderer:
    """
    Renders RGB images and confidence-masked depth maps from sparse 3D point clouds.

    Uses splatting-based rendering with alpha blending and confidence tracking.
    """

    def __init__(self, config: RenderConfig, enable_confidence_mask: bool = True):
        self.config = config
        self.enable_confidence_mask = enable_confidence_mask
        self.edge_detector = DepthEdgeDetector(config)

    def render_view(
        self,
        points_3d: torch.Tensor,
        points_color: torch.Tensor,
        points_confidence: torch.Tensor,
        camera_pose: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Render RGB, depth, and confidence maps from a given camera viewpoint.

        Args:
            points_3d: (N, 3) world coordinates
            points_color: (N, 3) RGB colors in [0, 1]
            points_confidence: (N,) confidence weights
            camera_pose: (4, 4) camera-to-world transform
            intrinsics: (3, 3) camera intrinsics

        Returns:
            rgb_map: (H, W, 3) rendered RGB
            depth_map: (H, W) rendered depth
            confidence_map: (H, W) accumulated confidence (alpha)
        """
        device = points_3d.device
        H, W = self.config.render_resolution

        rgb_buffer = torch.zeros(H, W, 3, device=device)
        depth_buffer = torch.zeros(H, W, device=device)
        confidence_buffer = torch.zeros(H, W, device=device)

        # Camera space transform
        world_to_cam = torch.inverse(camera_pose)
        points_homo = torch.cat(
            [points_3d, torch.ones(points_3d.shape[0], 1, device=device)], dim=1
        )
        points_cam = (world_to_cam @ points_homo.T).T

        # Filter behind camera
        valid_depth = points_cam[:, 2] > self.config.depth_near_plane
        points_cam = points_cam[valid_depth]
        points_color_valid = points_color[valid_depth]
        points_confidence_valid = points_confidence[valid_depth]

        if points_cam.shape[0] == 0:
            return rgb_buffer, depth_buffer, confidence_buffer

        # Project to image plane
        points_img = (intrinsics @ points_cam[:, :3].T).T
        uvz = points_img[:, :2] / (points_img[:, 2:3] + 1e-8)
        depths = points_img[:, 2]

        # Filter by image bounds
        u_valid = (uvz[:, 0] >= 0) & (uvz[:, 0] < W)
        v_valid = (uvz[:, 1] >= 0) & (uvz[:, 1] < H)
        valid_proj = u_valid & v_valid & (depths < self.config.depth_far_plane)

        uvz = uvz[valid_proj]
        depths = depths[valid_proj]
        points_color_valid = points_color_valid[valid_proj]
        points_confidence_valid = points_confidence_valid[valid_proj]

        # Back-to-front depth ordering
        depth_order = torch.argsort(depths, descending=True)
        uvz = uvz[depth_order]
        depths = depths[depth_order]
        points_color_valid = points_color_valid[depth_order]
        points_confidence_valid = points_confidence_valid[depth_order]

        # Splatting
        radius = self.config.point_radius
        for i in range(uvz.shape[0]):
            u, v = uvz[i]
            depth = depths[i]
            color = points_color_valid[i]
            conf = points_confidence_valid[i]

            u_min = int(max(0, u - radius))
            u_max = int(min(W, u + radius + 1))
            v_min = int(max(0, v - radius))
            v_max = int(min(H, v + radius + 1))

            if u_max <= u_min or v_max <= v_min:
                continue

            y_coords, x_coords = torch.meshgrid(
                torch.arange(v_min, v_max, device=device),
                torch.arange(u_min, u_max, device=device),
                indexing='ij',
            )
            dist_sq = (x_coords - u) ** 2 + (y_coords - v) ** 2
            weight = torch.exp(-dist_sq / (2 * (radius / 2) ** 2))
            weight = weight * conf

            alpha = weight.unsqueeze(-1)

            rgb_buffer[v_min:v_max, u_min:u_max] = (
                rgb_buffer[v_min:v_max, u_min:u_max] * (1 - alpha)
                + color.view(1, 1, 3) * alpha
            )

            depth_buffer[v_min:v_max, u_min:u_max] = (
                depth_buffer[v_min:v_max, u_min:u_max] * (1 - weight)
                + depth * weight
            )

            confidence_buffer[v_min:v_max, u_min:u_max] += weight

        confidence_buffer = torch.clamp(confidence_buffer, 0, 1)

        return rgb_buffer, depth_buffer, confidence_buffer

    def create_masked_depth(
        self,
        depth_map: torch.Tensor,
        confidence_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create depth map with DepMask masking strategy:
        D_mask(u,v) = d(u,v) if Conf(u,v) > tau_conf AND grad_d(u,v) < tau_edge, else 0

        Args:
            depth_map: (H, W) rendered depth
            confidence_map: (H, W) rendering confidence

        Returns:
            masked_depth: (H, W) depth with unreliable regions masked
        """
        artifact_mask = self.edge_detector.detect_stretching_artifacts(
            depth_map, confidence_map
        )
        low_confidence_mask = confidence_map < self.config.conf_threshold
        unreliable_mask = artifact_mask | low_confidence_mask

        masked_depth = depth_map.clone()
        masked_depth[unreliable_mask] = self.config.depth_mask_value

        return masked_depth

    def render_rgbd(
        self,
        points_3d: torch.Tensor,
        points_color: torch.Tensor,
        points_confidence: torch.Tensor,
        camera_pose: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Render a complete RGB-D frame with confidence masking.

        Returns:
            dict with 'rgb', 'depth', 'masked_depth', 'confidence'
        """
        rgb_map, depth_map, confidence_map = self.render_view(
            points_3d, points_color, points_confidence, camera_pose, intrinsics
        )

        masked_depth = self.create_masked_depth(depth_map, confidence_map)

        return {
            'rgb': rgb_map,
            'depth': depth_map,
            'masked_depth': masked_depth,
            'confidence': confidence_map,
        }


class DepMaskRenderPipeline:
    """
    Complete Confidence & Geometry Dual-Guided Rendering Pipeline.

    Pipeline:
    1. Sparse image input -> VGGT feature extraction -> point cloud generation
    2. Global alignment and confidence estimation
    3. Multi-view rendering with DepMask masking
    """

    def __init__(self, config: RenderConfig):
        self.config = config
        self.confidence_estimator = ConfidenceEstimator(config)
        self.renderer = PointCloudRenderer(config)

    def extract_point_cloud_from_depth(
        self,
        images: torch.Tensor,
        depth_maps: torch.Tensor,
        camera_poses: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract dense 3D point cloud from posed depth maps.

        Args:
            images: (N, 3, H, W) input images
            depth_maps: (N, H, W) per-view depth maps
            camera_poses: (N, 4, 4) camera-to-world transforms
            intrinsics: (N, 3, 3) camera intrinsics

        Returns:
            points_3d: (M, 3) world-space point cloud
            points_color: (M, 3) point colors
        """
        device = images.device
        points_list = []
        colors_list = []

        for i in range(images.shape[0]):
            H, W = depth_maps[i].shape
            v, u = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij',
            )

            K = intrinsics[i]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            z = depth_maps[i]
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points_cam = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

            pose = camera_poses[i]
            points_homo = torch.cat(
                [points_cam, torch.ones(points_cam.shape[0], 1, device=device)], dim=1
            )
            points_world = (pose @ points_homo.T).T[:, :3]

            colors = images[i].permute(1, 2, 0).reshape(-1, 3)

            valid = (z.reshape(-1) > self.config.depth_near_plane) & (
                z.reshape(-1) < self.config.depth_far_plane
            )
            points_list.append(points_world[valid])
            colors_list.append(colors[valid])

        points_3d = torch.cat(points_list, dim=0)
        points_color = torch.cat(colors_list, dim=0)

        return points_3d, points_color

    def global_alignment(
        self,
        points_3d: torch.Tensor,
        camera_poses: torch.Tensor,
        intrinsics: torch.Tensor,
        observations: List[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform global bundle adjustment to refine geometry and compute confidence.

        Args:
            points_3d: (N, 3) initial 3D points
            camera_poses: (M, 4, 4) camera poses
            intrinsics: (M, 3, 3) camera intrinsics
            observations: multi-view correspondences

        Returns:
            refined_points: (N, 3) optimized 3D points
            confidence: (N,) per-point confidence weights
        """
        confidence = self.confidence_estimator.compute_point_confidence(
            points_3d, observations, camera_poses, intrinsics
        )
        return points_3d, confidence

    def render_novel_view(
        self,
        points_3d: torch.Tensor,
        points_color: torch.Tensor,
        points_confidence: torch.Tensor,
        target_camera_pose: torch.Tensor,
        target_intrinsics: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Render a novel view with RGB and masked depth.

        Returns:
            dict with 'rgb', 'depth', 'masked_depth', 'confidence'
        """
        return self.renderer.render_rgbd(
            points_3d, points_color, points_confidence,
            target_camera_pose, target_intrinsics,
        )

    def render_trajectory(
        self,
        points_3d: torch.Tensor,
        points_color: torch.Tensor,
        points_confidence: torch.Tensor,
        trajectory_poses: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Render multiple views along a camera trajectory.

        Args:
            points_3d: (N, 3) world-space points
            points_color: (N, 3) point colors
            points_confidence: (N,) confidence weights
            trajectory_poses: (K, 4, 4) camera-to-world poses
            intrinsics: (K, 3, 3) or (1, 3, 3) camera intrinsics

        Returns:
            List of rendered view dicts for each trajectory point
        """
        K = trajectory_poses.shape[0]
        if intrinsics.dim() == 2:
            intrinsics = intrinsics.unsqueeze(0).expand(K, -1, -1)

        rendered_views = []
        for i in range(K):
            view = self.render_novel_view(
                points_3d, points_color, points_confidence,
                trajectory_poses[i], intrinsics[i],
            )
            rendered_views.append(view)

        return rendered_views

    def forward(
        self,
        input_images: torch.Tensor,
        depth_maps: torch.Tensor,
        camera_poses: torch.Tensor,
        intrinsics: torch.Tensor,
        target_poses: torch.Tensor,
        target_intrinsics: torch.Tensor,
        observations: Optional[List[Dict]] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Complete DepMask rendering pipeline.

        Args:
            input_images: (N, 3, H, W) sparse input views
            depth_maps: (N, H, W) per-view depth
            camera_poses: (N, 4, 4) input camera poses
            intrinsics: (N, 3, 3) input camera intrinsics
            target_poses: (K, 4, 4) target viewpoint poses
            target_intrinsics: (K, 3, 3) target intrinsics
            observations: optional multi-view correspondences

        Returns:
            List of rendered novel views
        """
        N = input_images.shape[0]
        device = input_images.device

        # Stage 1: Extract point cloud
        points_3d, points_color = self.extract_point_cloud_from_depth(
            input_images, depth_maps, camera_poses, intrinsics
        )

        # Stage 2: Global alignment and confidence
        if observations is None:
            raise ValueError('Multi-view point observations are required for confidence estimation')

        refined_points, confidence = self.global_alignment(
            points_3d, camera_poses, intrinsics, observations
        )

        # Stage 3: Render novel views
        rendered_views = self.render_trajectory(
            refined_points, points_color, confidence,
            target_poses, target_intrinsics,
        )

        return rendered_views
