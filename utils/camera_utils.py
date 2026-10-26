"""
Camera pose and trajectory generation utilities.
"""

import copy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def rotate_theta(
    c2ws: torch.Tensor,
    theta: float,
    phi: float,
    r: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Rotate camera poses on a sphere centered at [0, 0, -r].

    Args:
        c2ws: (N, 4, 4) camera-to-world matrices
        theta: elevation angle delta in degrees
        phi: azimuthal angle delta in degrees
        r: sphere radius offset
        device: torch device

    Returns:
        rotated_c2ws: (N, 4, 4) rotated camera-to-world matrices
    """
    c2ws = copy.deepcopy(c2ws)
    c2ws[:, 2, 3] = c2ws[:, 2, 3] + r

    theta = torch.deg2rad(torch.tensor(theta)).to(device)
    phi = torch.deg2rad(torch.tensor(phi)).to(device)

    v = torch.tensor([0, torch.cos(theta), torch.sin(theta)]).to(device)
    v_x = torch.zeros(3, 3).to(device)
    v_x[0, 1] = -v[2]
    v_x[0, 2] = v[1]
    v_x[1, 0] = v[2]
    v_x[1, 2] = -v[0]
    v_x[2, 0] = -v[1]
    v_x[2, 1] = v[0]

    v_x_square = torch.matmul(v_x, v_x)
    R_mat = (
        torch.eye(3).to(device)
        + torch.sin(phi) * v_x
        + (1 - torch.cos(phi)) * v_x_square
    )

    R_h = torch.eye(4).to(device)
    R_h[:3, :3] = R_mat

    c2ws = torch.matmul(R_h, c2ws)
    c2ws[:, 2, 3] = c2ws[:, 2, 3] - r

    return c2ws


def sphere_to_pose(
    c2ws: torch.Tensor,
    theta: float,
    phi: float,
    r: float,
    device: torch.device,
    x: Optional[float] = None,
    y: Optional[float] = None,
) -> torch.Tensor:
    """
    Convert spherical coordinates to camera pose matrix.

    Args:
        c2ws: (1, 4, 4) reference camera-to-world matrix
        theta: elevation angle in degrees
        phi: azimuthal angle in degrees
        r: spherical radius
        device: torch device
        x: optional x translation
        y: optional y translation

    Returns:
        c2ws_new: (1, 4, 4) new camera-to-world matrix
    """
    c2ws_new = copy.deepcopy(c2ws)
    c2ws_new[:, 2, 3] += r

    if y is not None:
        c2ws_new[:, 1, 3] += y
    if x is not None:
        c2ws_new[:, 0, 3] += x

    theta_rad = torch.deg2rad(torch.tensor(theta)).to(device)
    phi_rad = torch.deg2rad(torch.tensor(phi)).to(device)

    sin_v = torch.sin(theta_rad)
    cos_v = torch.cos(theta_rad)
    sin_h = torch.sin(phi_rad)
    cos_h = torch.cos(phi_rad)

    R_mat = torch.tensor([
        [cos_h, sin_h * sin_v, -sin_h * cos_v],
        [0, cos_v, sin_v],
        [sin_h, -cos_h * sin_v, cos_h * cos_v],
    ]).float().to(device)

    R_h = torch.eye(4).to(device)
    R_h[:3, :3] = R_mat

    c2ws_new = torch.matmul(R_h, c2ws_new)
    c2ws_new[:, 2, 3] = c2ws_new[:, 2, 3] - r

    return c2ws_new


def generate_trajectory(
    c2w_ref: torch.Tensor,
    num_views: int,
    elevation: float = 5.0,
    center_scale: float = 1.0,
    d_theta: float = 10.0,
    d_phi: float = 30.0,
    d_r: float = -0.2,
    d_x: float = 0.0,
    d_y: float = 0.0,
    device: torch.device = torch.device('cuda'),
) -> torch.Tensor:
    """
    Generate a camera trajectory around a reference pose.

    Args:
        c2w_ref: (1, 4, 4) reference camera-to-world matrix
        num_views: number of trajectory views
        elevation: elevation angle offset in degrees
        center_scale: scale factor for spherical radius
        d_theta: theta angle increment per step
        d_phi: phi angle increment per step
        d_r: radius increment per step (fractional)
        d_x: x translation increment per step
        d_y: y translation increment per step
        device: torch device

    Returns:
        trajectory: (num_views, 4, 4) camera-to-world matrices
    """
    trajectory = []
    for i in range(num_views):
        pose = sphere_to_pose(
            c2w_ref,
            theta=elevation + i * d_theta,
            phi=i * d_phi,
            r=center_scale * (1 + i * d_r),
            x=i * d_x,
            y=i * d_y,
            device=device,
        )
        trajectory.append(pose)
    return torch.cat(trajectory, dim=0)


def generate_trajectory_interp(
    c2w_ref: torch.Tensor,
    c2w_target: torch.Tensor,
    num_views: int,
    device: torch.device = torch.device('cuda'),
) -> torch.Tensor:
    """
    Interpolate camera poses between reference and target poses.

    Args:
        c2w_ref: (1, 4, 4) starting camera-to-world matrix
        c2w_target: (1, 4, 4) ending camera-to-world matrix
        num_views: number of interpolated views
        device: torch device

    Returns:
        trajectory: (num_views, 4, 4) interpolated camera-to-world matrices
    """
    trajectory = []
    for i in range(num_views):
        t = i / max(num_views - 1, 1)
        pose = interpolate_poses(c2w_ref, c2w_target, t, device)
        trajectory.append(pose)
    return torch.cat(trajectory, dim=0)


def interpolate_poses(
    pose_a: torch.Tensor,
    pose_b: torch.Tensor,
    t: float,
    device: torch.device = torch.device('cuda'),
) -> torch.Tensor:
    """
    Linearly interpolate between two camera poses.

    Args:
        pose_a: (1, 4, 4) starting pose
        pose_b: (1, 4, 4) ending pose
        t: interpolation factor in [0, 1]
        device: torch device

    Returns:
        interp_pose: (1, 4, 4) interpolated pose
    """
    pose_a = pose_a.squeeze(0).cpu().numpy()
    pose_b = pose_b.squeeze(0).cpu().numpy()

    R_a = R.from_matrix(pose_a[:3, :3])
    R_b = R.from_matrix(pose_b[:3, :3])

    # Slerp for rotation
    if t <= 0:
        R_interp = R_a
    elif t >= 1:
        R_interp = R_b
    else:
        key_times = [0, 1]
        key_rots = R.from_matrix(np.stack([pose_a[:3, :3], pose_b[:3, :3]]))
        slerp = Slerp(key_times, key_rots)
        R_interp = slerp([t])[0]

    # Linear interpolation for translation
    T_interp = pose_a[:3, 3] * (1 - t) + pose_b[:3, 3] * t

    result = np.eye(4)
    result[:3, :3] = R_interp.as_matrix()
    result[:3, 3] = T_interp

    return torch.from_numpy(result).float().unsqueeze(0).to(device)


def generate_trajectory_from_txt(
    c2w_ref: torch.Tensor,
    txt_path: str,
    device: torch.device = torch.device('cuda'),
) -> torch.Tensor:
    """
    Load camera trajectory from a text file.

    File format: each line contains [theta, phi, r] or [theta, phi, r, x, y]

    Args:
        c2w_ref: (1, 4, 4) reference camera-to-world matrix
        txt_path: path to trajectory text file
        device: torch device

    Returns:
        trajectory: (N, 4, 4) camera-to-world matrices
    """
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    params = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            values = [float(v) for v in line.split()]
            params.append(values)

    trajectory = []
    for param in params:
        theta, phi, r = param[0], param[1], param[2]
        x = param[3] if len(param) > 3 else 0.0
        y = param[4] if len(param) > 4 else 0.0

        pose = sphere_to_pose(
            c2w_ref, theta=theta, phi=phi, r=r,
            x=x, y=y, device=device,
        )
        trajectory.append(pose)

    return torch.cat(trajectory, dim=0)


def generate_candidate_poses(
    c2w_ref: torch.Tensor,
    num_candidates: int = 36,
    theta_range: Tuple[float, float] = (-20, 20),
    phi_range: Tuple[float, float] = (-30, 30),
    r_range: Tuple[float, float] = (0.5, 1.5),
    device: torch.device = torch.device('cuda'),
) -> torch.Tensor:
    """
    Generate candidate camera poses around a reference pose for NBV planning.

    Args:
        c2w_ref: (1, 4, 4) reference pose
        num_candidates: number of candidate poses to generate
        theta_range: (min, max) elevation angle range in degrees
        phi_range: (min, max) azimuthal angle range in degrees
        r_range: (min, max) radius range (multiplicative factor)
        device: torch device

    Returns:
        candidate_poses: (num_candidates, 4, 4) camera-to-world matrices
    """
    theta_vals = torch.linspace(theta_range[0], theta_range[1], 6, device=device)
    phi_vals = torch.linspace(phi_range[0], phi_range[1], 6, device=device)
    r_vals = torch.linspace(r_range[0], r_range[1], 3, device=device)

    # Sample candidates
    candidates = []
    indices = torch.randperm(len(theta_vals) * len(phi_vals))[:num_candidates]
    for idx in indices:
        ti = idx // len(phi_vals)
        pi = idx % len(phi_vals)
        ri = torch.randint(0, len(r_vals), (1,)).item()

        pose = sphere_to_pose(
            c2w_ref,
            theta=theta_vals[ti].item(),
            phi=phi_vals[pi].item(),
            r=r_vals[ri].item(),
            device=device,
        )
        candidates.append(pose)

    return torch.cat(candidates, dim=0)


def compute_camera_center(c2w: torch.Tensor) -> torch.Tensor:
    """
    Extract camera center from camera-to-world matrix.

    Args:
        c2w: (..., 4, 4) camera-to-world matrices

    Returns:
        centers: (..., 3) camera centers in world coordinates
    """
    return c2w[..., :3, 3]


def compute_look_at_matrix(
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor = torch.tensor([0., 1., 0.]),
) -> torch.Tensor:
    """
    Compute a look-at camera-to-world matrix.

    Args:
        eye: (3,) camera position
        center: (3,) look-at target
        up: (3,) up vector

    Returns:
        c2w: (4, 4) camera-to-world matrix
    """
    device = eye.device
    up = up.to(device)

    z = F.normalize(eye - center, dim=0)
    x = F.normalize(torch.cross(up, z), dim=0)
    y = torch.cross(z, x)

    c2w = torch.eye(4, device=device)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = eye

    return c2w


def compute_relative_pose(
    source_c2w: torch.Tensor,
    target_c2w: torch.Tensor,
) -> torch.Tensor:
    """
    Compute relative pose from source to target.

    Args:
        source_c2w: (..., 4, 4) source camera-to-world
        target_c2w: (..., 4, 4) target camera-to-world

    Returns:
        rel_pose: (..., 4, 4) relative transformation
    """
    source_w2c = torch.inverse(source_c2w)
    rel_pose = target_c2w @ source_w2c
    return rel_pose
