"""
Camera trajectory encoder implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class SimpleCameraEncoder(nn.Module):
    """
    Simple camera trajectory encoder using MLP.

    Encodes 6-DoF camera parameters (rotation + translation) or
    4x4 transformation matrices into embedding vectors.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
            in_dim = out_dim

        self.encoder = nn.Sequential(*layers)

    def forward(self, camera_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            camera_params: (B, T, C) camera trajectory
                           C = 6: [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
                           C = 16: flattened 4x4 matrices

        Returns:
            embeddings: (B, T, output_dim) camera embeddings
        """
        B, T, C = camera_params.shape
        camera_flat = camera_params.reshape(B * T, C)
        embeddings_flat = self.encoder(camera_flat)
        embeddings = embeddings_flat.reshape(B, T, -1)
        return embeddings


class TransformerCameraEncoder(nn.Module):
    """
    Transformer-based camera trajectory encoder.

    Uses self-attention to model the temporal relationship between
    camera poses along a trajectory.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, camera_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            camera_params: (B, T, C) camera trajectory

        Returns:
            embeddings: (B, T, output_dim) camera embeddings
        """
        B, T, C = camera_params.shape

        x = self.input_proj(camera_params)
        x = x + self.pos_embed[:, :T, :]
        x = self.transformer(x)
        x = self.output_proj(x)

        return x


def poses_to_6dof(poses: torch.Tensor) -> torch.Tensor:
    """
    Convert 4x4 camera-to-world matrices to 6-DoF parameters (R6 representation).

    Args:
        poses: (B, T, 4, 4) or (T, 4, 4) camera-to-world matrices

    Returns:
        dof6: (B, T, 6) or (T, 6) 6-DoF parameters [rx, ry, rz, tx, ty, tz]
    """
    squeeze_dim = False
    if poses.dim() == 3:
        poses = poses.unsqueeze(0)
        squeeze_dim = True

    B, T = poses.shape[:2]

    # Extract translation
    t = poses[..., :3, 3]

    # Extract rotation as axis-angle
    R = poses[..., :3, :3]

    # Rodriguez formula inverse: rotation matrix -> axis-angle
    # angle = arccos((trace(R) - 1) / 2)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))

    # Avoid division by zero
    sin_angle = torch.sin(angle) + 1e-8

    rx = (R[..., 2, 1] - R[..., 1, 2]) / (2 * sin_angle)
    ry = (R[..., 0, 2] - R[..., 2, 0]) / (2 * sin_angle)
    rz = (R[..., 1, 0] - R[..., 0, 1]) / (2 * sin_angle)

    # Scale axis by angle
    rx = rx * angle
    ry = ry * angle
    rz = rz * angle

    dof6 = torch.stack([rx, ry, rz, t[..., 0], t[..., 1], t[..., 2]], dim=-1)

    if squeeze_dim:
        dof6 = dof6.squeeze(0)

    return dof6


def dof6_to_poses(dof6: torch.Tensor) -> torch.Tensor:
    """
    Convert 6-DoF parameters back to 4x4 camera-to-world matrices.

    Args:
        dof6: (B, T, 6) or (T, 6) 6-DoF parameters [rx, ry, rz, tx, ty, tz]

    Returns:
        poses: (B, T, 4, 4) or (T, 4, 4) camera-to-world matrices
    """
    squeeze_dim = False
    if dof6.dim() == 2:
        dof6 = dof6.unsqueeze(0)
        squeeze_dim = True

    B, T = dof6.shape[:2]
    device = dof6.device

    rx, ry, rz = dof6[..., 0], dof6[..., 1], dof6[..., 2]
    tx, ty, tz = dof6[..., 3], dof6[..., 4], dof6[..., 5]

    # Axis-angle to rotation matrix (Rodriguez formula)
    angle = torch.sqrt(rx**2 + ry**2 + rz**2 + 1e-8)
    axis = torch.stack([rx / angle, ry / angle, rz / angle], dim=-1)

    K = torch.zeros(B, T, 3, 3, device=device)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]

    K2 = K @ K

    sin_a = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
    cos_a = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)

    I = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)
    R = I + sin_a * K + (1 - cos_a) * K2

    poses = torch.zeros(B, T, 4, 4, device=device)
    poses[..., :3, :3] = R
    poses[..., 0, 3] = tx
    poses[..., 1, 3] = ty
    poses[..., 2, 3] = tz
    poses[..., 3, 3] = 1.0

    if squeeze_dim:
        poses = poses.squeeze(0)

    return poses
