"""
3D geometric utility functions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def build_rotation_matrix(r: torch.Tensor) -> torch.Tensor:
    """
    Build a 3x3 rotation matrix from quaternion.

    Args:
        r: (N, 4) quaternion (w, x, y, z)

    Returns:
        R: (N, 3, 3) rotation matrices
    """
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] +
                      r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
    q = r / norm.unsqueeze(1)
    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r_w = q[:, 0]
    r_x = q[:, 1]
    r_y = q[:, 2]
    r_z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (r_y * r_y + r_z * r_z)
    R[:, 0, 1] = 2 * (r_x * r_y - r_w * r_z)
    R[:, 0, 2] = 2 * (r_x * r_z + r_w * r_y)
    R[:, 1, 0] = 2 * (r_x * r_y + r_w * r_z)
    R[:, 1, 1] = 1 - 2 * (r_x * r_x + r_z * r_z)
    R[:, 1, 2] = 2 * (r_y * r_z - r_w * r_x)
    R[:, 2, 0] = 2 * (r_x * r_z - r_w * r_y)
    R[:, 2, 1] = 2 * (r_y * r_z + r_w * r_x)
    R[:, 2, 2] = 1 - 2 * (r_x * r_x + r_y * r_y)

    return R


def build_scaling_rotation_matrix(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Build scaling-rotation matrix.

    Args:
        s: (N, 3) scaling factors
        r: (N, 4) quaternion

    Returns:
        L: (N, 3, 3) scaling-rotation matrix
    """
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation_matrix(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Compute inverse sigmoid."""
    return torch.log(x / (1 - x + 1e-8) + 1e-8)


def get_world_to_view_matrix(
    R: torch.Tensor,
    T: torch.Tensor,
    trans: torch.Tensor = torch.tensor([0., 0., 0.]),
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Build a world-to-view (view) transformation matrix.

    Args:
        R: (3, 3) rotation matrix
        T: (3,) translation vector
        trans: (3,) additional translation offset
        scale: scaling factor

    Returns:
        w2v: (4, 4) world-to-view matrix
    """
    device = R.device
    Rt = torch.zeros((4, 4), device=device)
    Rt[:3, :3] = R
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + trans.to(device=device, dtype=cam_center.dtype)) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)

    return Rt


def get_projection_matrix(
    znear: float = 0.01,
    zfar: float = 100.0,
    fov_x: float = 90.0,
    fov_y: float = 90.0,
) -> torch.Tensor:
    """
    Build an OpenGL-style projection matrix.

    Args:
        znear: near clipping plane
        zfar: far clipping plane
        fov_x: horizontal field of view in degrees
        fov_y: vertical field of view in degrees

    Returns:
        proj: (4, 4) projection matrix
    """
    tan_half_fov_x = np.tan(np.deg2rad(fov_x / 2))
    tan_half_fov_y = np.tan(np.deg2rad(fov_y / 2))

    top = tan_half_fov_y * znear
    bottom = -top
    right = tan_half_fov_x * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P


def get_fov(
    intrinsics: torch.Tensor,
    image_size: Tuple[int, int],
) -> Tuple[float, float]:
    """
    Compute horizontal and vertical field of view from intrinsics.

    Args:
        intrinsics: (3, 3) camera intrinsics matrix
        image_size: (W, H) image dimensions

    Returns:
        fov_x, fov_y in degrees
    """
    W, H = image_size
    fx = intrinsics[0, 0].item()
    fy = intrinsics[1, 1].item()

    fov_x = 2 * np.arctan(W / (2 * fx)) * 180 / np.pi
    fov_y = 2 * np.arctan(H / (2 * fy)) * 180 / np.pi

    return fov_x, fov_y


def focal_to_fov(focal: float, pixels: float) -> float:
    """Convert focal length (in pixels) to field of view (in degrees)."""
    return 2 * np.arctan(pixels / (2 * focal)) * 180 / np.pi


def fov_to_focal(fov: float, pixels: float) -> float:
    """Convert field of view (in degrees) to focal length (in pixels)."""
    return pixels / (2 * np.tan(np.deg2rad(fov) / 2))


def compute_nearest_neighbor_distance(
    points: torch.Tensor,
) -> torch.Tensor:
    """
    Compute nearest neighbor distance for each point using CUDA-based metric.

    Args:
        points: (N, 3) 3D point positions

    Returns:
        dists: (N,) distances to nearest neighbor
    """
    try:
        from simple_knn._C import distCUDA2
        return distCUDA2(points.float().cuda())
    except ImportError:
        # Fallback: compute pairwise distances on GPU
        N = points.shape[0]
        if N > 10000:
            # Too many points for full pairwise, use random sampling
            idx = torch.randperm(N, device=points.device)[:5000]
            points_sample = points[idx]
            dists = torch.cdist(points, points_sample).min(dim=1)[0]
        else:
            dists = torch.cdist(points, points).topk(k=2, largest=False, dim=1)[0][:, 1]
        return dists


def world_to_camera(
    points_3d: torch.Tensor,
    camera_pose: torch.Tensor,
) -> torch.Tensor:
    """
    Transform points from world to camera coordinates.

    Args:
        points_3d: (N, 3) world coordinates
        camera_pose: (4, 4) camera-to-world matrix

    Returns:
        points_cam: (N, 3) camera coordinates
    """
    world_to_cam = torch.inverse(camera_pose)
    points_homo = torch.cat([
        points_3d, torch.ones(points_3d.shape[0], 1, device=points_3d.device)
    ], dim=1)
    points_cam = (world_to_cam @ points_homo.T).T[:, :3]
    return points_cam


def project_points(
    points_3d: torch.Tensor,
    camera_pose: torch.Tensor,
    intrinsics: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D points to image coordinates.

    Args:
        points_3d: (N, 3) world coordinates
        camera_pose: (4, 4) camera-to-world matrix
        intrinsics: (3, 3) camera intrinsics

    Returns:
        uv: (N, 2) pixel coordinates
        depths: (N,) depth values (positive = in front of camera)
    """
    points_cam = world_to_camera(points_3d, camera_pose)
    depths = points_cam[:, 2]

    points_img = (intrinsics @ points_cam.T).T
    uv = points_img[:, :2] / (points_img[:, 2:3] + 1e-8)

    return uv, depths


def rigid_transform_3d(
    A: torch.Tensor,
    B: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute optimal rigid transformation from A to B using SVD.

    Args:
        A: (N, 3) source points
        B: (N, 3) target points

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    centroid_A = A.mean(dim=0)
    centroid_B = B.mean(dim=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = torch.linalg.svd(H)

    R = Vt.T @ U.T

    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    return R, t
