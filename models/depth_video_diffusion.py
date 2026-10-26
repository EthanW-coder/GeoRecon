"""Depth-conditioned video diffusion built on the DynamiCrafter backbone."""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from modules.geometric_encoder import GeometricConditionEncoder, ZeroConv
from modules.semantic_modulator import SemanticCameraModulator
from modules.camera_encoder import SimpleCameraEncoder


@dataclass
class VideoDiffusionConfig:
    vae_latent_channels: int = 4
    vae_scale_factor: int = 8
    unet_in_channels: int = 4
    render_channels: int = 4
    depth_channels: int = 1
    clip_embed_dim: int = 1024
    camera_embed_dim: int = 256
    num_frames: int = 25
    condition_dropout: float = 0.1


class DepthVideoDiffusion(nn.Module):
    """Geometric residual conditioning and semantic-camera cross-attention."""

    def __init__(self, config=None, base_diffusion_model=None):
        super().__init__()
        if base_diffusion_model is None:
            raise ValueError('A loaded DynamiCrafter backbone is required')
        self.config = config or VideoDiffusionConfig()
        if not 0 <= self.config.condition_dropout <= 1:
            raise ValueError('condition_dropout must be in [0,1]')
        self.backbone = base_diffusion_model
        if self.backbone.parameterization not in ('eps', 'v'):
            raise ValueError('Supported prediction types: eps, v')
        self.num_timesteps = self.backbone.num_timesteps
        self.geometric_encoder = GeometricConditionEncoder(
            unet_in_channels=self.config.unet_in_channels,
            render_channels=self.config.render_channels,
            depth_channels=self.config.depth_channels,
            vae_encoder=self._encode_frames,
        )
        self.semantic_camera_modulator = SemanticCameraModulator(
            clip_embed_dim=self.config.clip_embed_dim,
            camera_embed_dim=self.config.camera_embed_dim,
            clip_encoder=self._encode_reference,
            camera_encoder=SimpleCameraEncoder(output_dim=self.config.camera_embed_dim),
        )
        self._freeze_encoders()

    def _freeze_encoders(self):
        for module in (self.backbone.first_stage_model, self.backbone.cond_stage_model,
                       self.backbone.embedder, self.backbone.image_proj_model):
            module.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_encoders()
        return self

    @torch.no_grad()
    def _encode_frames(self, images):
        return self.backbone.encode_first_stage(images * 2 - 1)

    @torch.no_grad()
    def _encode_reference(self, images):
        if images.ndim == 4:
            images = images[:, None]
        if images.ndim != 5 or images.shape[1] == 0:
            raise ValueError('Expected reference images [B,N,3,H,W]')
        b, n = images.shape[:2]
        embeddings = self.backbone.embedder(rearrange(images * 2 - 1, 'b n c h w -> (b n) c h w'))
        tokens = self.backbone.image_proj_model(embeddings)
        return rearrange(tokens, '(b n) l c -> b (n l) c', b=b, n=n)

    def _conditions(self, I_render, D_mask, I_con, T_novel):
        if I_render.ndim != 5 or I_render.shape[2] != 3:
            raise ValueError('Expected rendered RGB [B,T,3,H,W]')
        b, t, _, h, w = I_render.shape
        if h % 8 or w % 8:
            raise ValueError('Video dimensions must be divisible by 8')
        if D_mask.shape != (b, t, 1, h, w) or T_novel.shape != (b, t, 6):
            raise ValueError('Depth and camera trajectory must match the rendered sequence')
        z_render = self.geometric_encoder.encode_render_rgb(I_render)
        z_mask = self.geometric_encoder.encode_masked_depth(D_mask)
        if z_mask.shape[-2:] != z_render.shape[-2:]:
            raise ValueError('Depth and VAE latent resolutions disagree')
        joined = rearrange(torch.cat((z_render, z_mask), dim=1), 'b c t h w -> (b t) c h w')
        residual = rearrange(self.geometric_encoder.zero_conv(joined), '(b t) c h w -> b c t h w', b=b, t=t)
        semantic = self.semantic_camera_modulator(I_con, T_novel)
        # Keep a shared image/camera token sequence across the temporal axis.
        # DynamiCrafter reserves 77 tokens for text and recognizes T*16 image tokens.
        if semantic.shape[1] == t * 16:
            semantic = torch.cat((semantic, semantic.new_zeros(b, 1, semantic.shape[-1])), dim=1)
        with torch.no_grad():
            text = self.backbone.get_learned_conditioning([''] * b)
        if text.shape[1] != 77 or text.shape[-1] != semantic.shape[-1]:
            raise ValueError('DynamiCrafter text and semantic context dimensions disagree')
        return residual, z_render, text, semantic

    def _predict(self, x, timestep, conditions, keep=None, frame_stride=10):
        residual, render, text, semantic = conditions
        if keep is not None:
            residual = residual * keep[:, None, None, None, None]
            render = render * keep[:, None, None, None, None]
            semantic = semantic * keep[:, None, None]
        cond = {'c_concat': [render], 'c_crossattn': [torch.cat((text, semantic), dim=1)]}
        fs = torch.full((x.shape[0],), frame_stride, device=x.device, dtype=torch.long)
        return self.backbone.apply_model(x + residual, timestep, cond, fs=fs)

    def _scale(self, timestep, shape):
        if getattr(self.backbone, 'use_dynamic_rescale', False):
            return self.backbone.scale_arr[timestep].reshape(shape)
        return torch.ones(shape, device=timestep.device)

    def forward(self, I_render, D_mask, I_con, T_novel, I_gt=None, frame_stride=10):
        if I_gt is None or I_gt.shape != I_render.shape:
            raise ValueError('Ground-truth video must match the rendered RGB shape')
        b, t = I_gt.shape[:2]
        z0 = self._encode_frames(rearrange(I_gt, 'b t c h w -> (b t) c h w'))
        z0 = rearrange(z0, '(b t) c h w -> b c t h w', b=b, t=t)
        steps = torch.randint(self.num_timesteps, (b,), device=z0.device)
        alpha = self.backbone.alphas_cumprod[steps].reshape(b, 1, 1, 1, 1)
        z0 = z0 * self._scale(steps, (b, 1, 1, 1, 1))
        noise = torch.randn_like(z0)
        noisy = alpha.sqrt() * z0 + (1 - alpha).sqrt() * noise
        conditions = self._conditions(I_render, D_mask, I_con, T_novel)
        keep = (torch.rand(b, device=z0.device) >= self.config.condition_dropout).to(z0.dtype)
        prediction = self._predict(noisy, steps, conditions, keep, frame_stride)
        target = noise if self.backbone.parameterization == 'eps' else alpha.sqrt()*noise - (1-alpha).sqrt()*z0
        return {'loss': F.mse_loss(prediction, target), 'prediction': prediction, 'target': target}

    @torch.no_grad()
    def sample(self, I_render, D_mask, I_con, T_novel, num_inference_steps=50,
               guidance_scale=7.5, eta=0., frame_stride=10):
        if not 1 <= num_inference_steps <= self.num_timesteps or not 0 <= eta <= 1:
            raise ValueError('Invalid DDIM step count or eta')
        conditions = self._conditions(I_render, D_mask, I_con, T_novel)
        x = torch.randn_like(conditions[1])
        b = x.shape[0]
        timesteps = torch.linspace(self.num_timesteps-1, 0, num_inference_steps, device=x.device).long()
        for i, step in enumerate(timesteps):
            ts = step.expand(b)
            output = self._predict(x, ts, conditions, frame_stride=frame_stride)
            if guidance_scale != 1:
                uncond = self._predict(x, ts, conditions, x.new_zeros(b), frame_stride)
                output = uncond + guidance_scale * (output - uncond)
            alpha = self.backbone.alphas_cumprod[step]
            previous = timesteps[i+1] if i+1 < len(timesteps) else None
            alpha_prev = self.backbone.alphas_cumprod[previous] if previous is not None else x.new_tensor(1.)
            if self.backbone.parameterization == 'v':
                x0 = alpha.sqrt()*x - (1-alpha).sqrt()*output
                eps = alpha.sqrt()*output + (1-alpha).sqrt()*x
            else:
                if alpha <= 0:
                    raise ValueError('Epsilon prediction requires a nonzero terminal alpha')
                eps = output
                x0 = (x - (1-alpha).sqrt()*eps) / alpha.sqrt()
            if getattr(self.backbone, 'use_dynamic_rescale', False):
                prev_scale = self.backbone.scale_arr[previous] if previous is not None else x.new_tensor(1.)
                x0 = x0 * (prev_scale / self.backbone.scale_arr[step])
            sigma = eta * (((1-alpha_prev)/(1-alpha).clamp_min(1e-12)) * (1-alpha/alpha_prev)).clamp_min(0).sqrt()
            x = alpha_prev.sqrt()*x0 + (1-alpha_prev-sigma.square()).clamp_min(0).sqrt()*eps
            if eta:
                x = x + sigma*torch.randn_like(x)
        decoded = self.backbone.decode_first_stage(x)
        return rearrange((decoded + 1).mul(.5).clamp(0, 1), 'b c t h w -> b t c h w')
