"""
Semantic-Camera Modulator for cross-attention conditioning.

Modulates U-Net features via cross-attention with combined semantic (CLIP) and
camera trajectory embeddings.

Cross-Attention mechanism:
    Q: U-Net spatial features
    K, V: Concatenated [CLIP embeddings, Camera embeddings]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SemanticCameraModulator(nn.Module):
    """
    Modulates U-Net features via cross-attention with semantic and camera conditions.

    Extracts appearance features from condition images using CLIP and camera
    motion features from the trajectory using a camera encoder, then concatenates
    them for use as cross-attention keys/values.
    """

    def __init__(
        self,
        clip_embed_dim: int = 1024,
        camera_embed_dim: int = 256,
        clip_encoder: Optional[nn.Module] = None,
        camera_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.clip_embed_dim = clip_embed_dim
        self.camera_embed_dim = camera_embed_dim
        self.clip_encoder = clip_encoder
        self.camera_encoder = camera_encoder

        # Projection layers to align dimensions
        self.clip_proj = nn.Linear(clip_embed_dim, clip_embed_dim)
        self.camera_proj = nn.Linear(camera_embed_dim, clip_embed_dim)

        if isinstance(clip_encoder, nn.Module):
            for param in self.clip_encoder.parameters():
                param.requires_grad = False

    def encode_appearance(self, I_con: torch.Tensor) -> torch.Tensor:
        """
        Extract appearance features from condition views using CLIP.

        Args:
            I_con: (B, 3, H, W) - Condition view (e.g., first frame)

        Returns:
            clip_features: (B, L, D) - CLIP image embeddings
        """
        if self.clip_encoder is not None:
            with torch.no_grad():
                clip_features = self.clip_encoder(I_con)
        else:
            raise RuntimeError('A CLIP image encoder is required')

        clip_features = self.clip_proj(clip_features)
        return clip_features

    def encode_camera(self, T_novel: torch.Tensor) -> torch.Tensor:
        """
        Extract camera motion features from trajectory.

        Args:
            T_novel: (B, T, 6) camera trajectory (r_x, r_y, r_z, t_x, t_y, t_z)
                     or (B, T, 4, 4) camera transformation matrices

        Returns:
            camera_features: (B, T, D) camera embeddings
        """
        if self.camera_encoder is not None:
            camera_features = self.camera_encoder(T_novel)
        else:
            raise RuntimeError('A camera encoder is required')

        camera_features = self.camera_proj(camera_features)
        return camera_features

    def forward(
        self,
        I_con: torch.Tensor,
        T_novel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create combined conditioning for cross-attention.

        Args:
            I_con: (B, 3, H, W) - Condition view
            T_novel: (B, T, 6) or (B, T, 4, 4) - Camera trajectory

        Returns:
            combined_conditioning: (B, L+T, D) - Combined embeddings for cross-attention
        """
        clip_features = self.encode_appearance(I_con)
        camera_features = self.encode_camera(T_novel)

        # Concatenate along sequence dimension
        combined_conditioning = torch.cat([clip_features, camera_features], dim=1)

        return combined_conditioning
