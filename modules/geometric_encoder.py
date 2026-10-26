"""
Geometric condition encoder: encodes RGB renders and masked depth for video diffusion conditioning.

Architecture:
    I_render -> VAE Encoder -> z_render (B, 4, T, H/8, W/8)
    D_mask -> Downsample -> z_mask (B, 1, T, H/8, W/8)
    Concatenation: [z_render, z_mask] -> (B, 5, T, H/8, W/8)
    Zero-Conv: (B, 5, T, H/8, W/8) -> (B, 4, T, H/8, W/8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional


class ZeroConv(nn.Module):
    """
    Zero-initialized 2D convolution for stable training.
    Ensures geometric conditions do not disrupt pre-trained weights during early fine-tuning.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GeometricConditionEncoder(nn.Module):
    """
    Encodes explicit geometric conditions (RGB renders and masked depth) into latent space.

    The encoded conditions are projected through a zero-initialized convolution
    and added to the noisy latent.
    """

    def __init__(
        self,
        unet_in_channels: int = 4,
        render_channels: int = 4,
        depth_channels: int = 1,
        zero_conv_init: bool = True,
        vae_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.unet_in_channels = unet_in_channels
        self.vae_encoder = vae_encoder

        # Depth downsampling from (B, 1, T, H, W) to (B, 1, T, H/8, W/8)
        self.depth_downsample = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(1, 4, 4), stride=(1, 4, 4), padding=0),
            nn.SiLU(),
            nn.Conv3d(16, 1, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
        )

        # Input: render(4) + depth(1) = 5 channels
        total_in = render_channels + depth_channels
        if zero_conv_init:
            self.zero_conv = ZeroConv(in_channels=total_in, out_channels=unet_in_channels)
        else:
            self.zero_conv = nn.Conv2d(total_in, unet_in_channels, kernel_size=1)

    def encode_render_rgb(self, I_render: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB render views using VAE encoder.

        Args:
            I_render: (B, T, 3, H, W) - RGB render sequence

        Returns:
            z_render: (B, 4, T, H/8, W/8) - Latent render features
        """
        B, T, C, H, W = I_render.shape

        if self.vae_encoder is not None:
            I_render_flat = rearrange(I_render, 'b t c h w -> (b t) c h w')
            with torch.no_grad():
                z_render_flat = self.vae_encoder(I_render_flat)
                if hasattr(z_render_flat, 'sample'):
                    z_render_flat = z_render_flat.sample()
            z_render = rearrange(z_render_flat, '(b t) c h w -> b c t h w', b=B, t=T)
        else:
            raise RuntimeError('A VAE encoder is required for RGB conditioning')

        return z_render

    def encode_masked_depth(self, D_mask: torch.Tensor) -> torch.Tensor:
        """
        Downsample masked depth to latent resolution.

        Args:
            D_mask: (B, T, 1, H, W) - Masked depth maps

        Returns:
            z_mask: (B, 1, T, H/8, W/8) - Downsampled depth features
        """
        D_mask = rearrange(D_mask, 'b t c h w -> b c t h w')
        z_mask = self.depth_downsample(D_mask)
        return z_mask

    def forward(
        self,
        z_t: torch.Tensor,
        I_render: torch.Tensor,
        D_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine geometric conditions with noisy latent.

        Args:
            z_t: (B, 4, T, H_lat, W_lat) - Noisy latent from diffusion process
            I_render: (B, T, 3, H, W) - RGB render views
            D_mask: (B, T, 1, H, W) - Masked depth views

        Returns:
            conditioned_latent: (B, 4, T, H_lat, W_lat) - Geometrically conditioned latent
        """
        # Encode geometric conditions
        z_render = self.encode_render_rgb(I_render)
        z_mask = self.encode_masked_depth(D_mask)

        # Match spatial dimensions
        if z_render.shape[-2:] != z_t.shape[-2:]:
            z_render = F.interpolate(
                rearrange(z_render, 'b c t h w -> (b t) c h w'),
                size=z_t.shape[-2:], mode='bilinear', align_corners=False
            )
            z_render = rearrange(z_render, '(b t) c h w -> b c t h w', b=z_t.shape[0], t=z_t.shape[2])

        if z_mask.shape[-2:] != z_t.shape[-2:]:
            z_mask = F.interpolate(
                rearrange(z_mask, 'b c t h w -> (b t) c h w'),
                size=z_t.shape[-2:], mode='bilinear', align_corners=False
            )
            z_mask = rearrange(z_mask, '(b t) c h w -> b c t h w', b=z_t.shape[0], t=z_t.shape[2])

        # Concatenate geometric conditions along the channel dimension.
        z_concat = torch.cat([z_render, z_mask], dim=1)

        # Process through zero-conv per-frame
        B, C, T, H, W = z_concat.shape
        z_concat_flat = rearrange(z_concat, 'b c t h w -> (b t) c h w')
        conditioned_flat = self.zero_conv(z_concat_flat)
        conditioned_latent = rearrange(conditioned_flat, '(b t) c h w -> b c t h w', b=B, t=T)

        # Residual connection
        output = z_t + conditioned_latent

        return output
