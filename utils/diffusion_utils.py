"""
Diffusion sampling utilities.
"""

import importlib
import torch
import numpy as np
from collections import OrderedDict
from einops import rearrange
from typing import Optional, Dict, Any


def count_params(model, verbose: bool = False) -> int:
    """Count total parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config: Dict[str, Any]):
    """Instantiate a class from a config dictionary with 'target' and 'params' keys."""
    if "target" not in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string: str, reload: bool = False):
    """Dynamically import and return a class/function from a dotted path string."""
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_model_checkpoint(model, ckpt_path: str):
    """
    Load model weights from checkpoint file.

    Handles:
    - Standard PyTorch checkpoints with 'state_dict' key
    - DeepSpeed checkpoints
    - Direct state_dict checkpoints
    """
    state_dict = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            new_pl_sd = OrderedDict()
            for k, v in state_dict.items():
                new_pl_sd[k] = v
            # Handle key renaming for compatibility
            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    elif "module" in state_dict:
        # DeepSpeed checkpoint
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]] = state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    else:
        model.load_state_dict(state_dict, strict=False)

    print(f">>> Model checkpoint loaded from {ckpt_path}.")
    return model


def get_latent_z(model, videos: torch.Tensor) -> torch.Tensor:
    """
    Encode video frames to latent space.

    Args:
        model: diffusion model with encode_first_stage method
        videos: (B, C, T, H, W) video tensor

    Returns:
        z: (B, c, T, h, w) latent representation
    """
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def prepare_diffusion_conditioning(
    model,
    prompts,
    images,
    condition_index,
    fs: int = 10,
    text_input: bool = True,
) -> Dict[str, list]:
    """
    Prepare conditioning dict for diffusion sampling.

    Args:
        model: diffusion model
        prompts: list of text prompts
        images: (B, C, T, H_cond, W_cond) conditioning video frames
        condition_index: list of frame indices used for conditioning
        fs: frame stride
        text_input: whether to use text conditioning

    Returns:
        cond: conditioning dictionary for DDIM sampler
    """
    batch_size = images.shape[0]

    if not text_input:
        prompts = [""] * batch_size

    # Image embedding
    img = images[:, :, condition_index[0]]  # b c h w
    img_emb = model.embedder(img)  # b l c
    img_emb = model.image_proj_model(img_emb)

    # Text embedding
    cond_emb = model.get_learned_conditioning(prompts)

    cond = {
        "c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]
    }

    # Hybrid conditioning (latent concatenation)
    if hasattr(model.model, 'conditioning_key') and model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, images)
        img_cat_cond = z
        cond["c_concat"] = [img_cat_cond]

    return cond


def prepare_unconditional_conditioning(
    model,
    images,
    prompt_emb_dim: int,
    uncond_type: str = "zero_embed",
) -> Dict[str, list]:
    """
    Prepare unconditional conditioning for classifier-free guidance.

    Args:
        model: diffusion model
        images: conditioning images
        prompt_emb_dim: dimension of prompt embeddings
        uncond_type: "empty_seq" or "zero_embed"

    Returns:
        uc: unconditional conditioning dictionary
    """
    batch_size = images.shape[0]

    if uncond_type == "empty_seq":
        uc_emb = model.get_learned_conditioning([""] * batch_size)
    elif uncond_type == "zero_embed":
        # Get a reference embedding first for shape
        ref_emb = model.get_learned_conditioning([""])
        uc_emb = torch.zeros(batch_size, *ref_emb.shape[1:], device=images.device)
    else:
        raise ValueError(f"Unknown uncond_type: {uncond_type}")

    img = torch.zeros_like(images[:, :, 0])
    uc_img_emb = model.embedder(img)
    uc_img_emb = model.image_proj_model(uc_img_emb)

    uc = {
        "c_crossattn": [torch.cat([uc_emb, uc_img_emb], dim=1)]
    }

    if hasattr(model.model, 'conditioning_key') and model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, images)
        uc["c_concat"] = [z]

    return uc


def image_guided_synthesis(
    model,
    prompts,
    videos,
    noise_shape,
    n_samples: int = 1,
    ddim_steps: int = 50,
    ddim_eta: float = 1.0,
    unconditional_guidance_scale: float = 1.0,
    cfg_img: Optional[float] = None,
    fs: int = 10,
    text_input: bool = True,
    multiple_cond_cfg: bool = False,
    timestep_spacing: str = 'uniform',
    guidance_rescale: float = 0.0,
    condition_index: Optional[list] = None,
    **kwargs,
):
    """
    Perform image-guided DDIM synthesis for novel view generation.

    Args:
        model: diffusion model
        prompts: text prompts
        videos: (B, C, T, H, W) conditioning video frames
        noise_shape: shape of noise tensor (B, C, T, H/8, W/8)
        n_samples: number of samples to generate
        ddim_steps: DDIM sampling steps
        ddim_eta: DDIM eta (0 = deterministic)
        unconditional_guidance_scale: CFG scale
        cfg_img: separate image CFG scale for multi-condition mode
        fs: frame stride
        text_input: whether to use text conditioning
        multiple_cond_cfg: use multi-condition CFG
        timestep_spacing: timestep spacing strategy
        guidance_rescale: guidance rescale factor
        condition_index: frame indices for conditioning

    Returns:
        batch_variants: (n_samples, B, C, T, H, W) generated videos
    """
    from lvdm.models.samplers.ddim import DDIMSampler
    from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSamplerMultiCond

    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSamplerMultiCond(model)
    batch_size = noise_shape[0]
    fs_tensor = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    if not text_input:
        prompts = [""] * batch_size
    if condition_index is None:
        condition_index = [0]

    # Prepare conditioning
    img = videos[:, :, condition_index[0]]
    img_emb = model.embedder(img)
    img_emb = model.image_proj_model(img_emb)

    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}

    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, videos)
        cond["c_concat"] = [z]

    # Unconditional conditioning for CFG
    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            uc_emb = model.get_learned_conditioning(batch_size * [""])
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        else:
            uc_emb = model.get_learned_conditioning(batch_size * [""])

        uc_img_emb = model.embedder(torch.zeros_like(img))
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb, uc_img_emb], dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [z]
    else:
        uc = None

    # Multi-condition CFG
    if multiple_cond_cfg and cfg_img is not None and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb, img_emb], dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [z]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    batch_variants = []
    for _ in range(n_samples):
        samples, _ = ddim_sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=batch_size,
            shape=noise_shape[1:],
            verbose=False,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            cfg_img=cfg_img,
            mask=None,
            x0=None,
            fs=fs_tensor,
            timestep_spacing=timestep_spacing,
            guidance_rescale=guidance_rescale,
            **kwargs,
        )

        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)

    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)
