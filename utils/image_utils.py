"""
Image processing utilities.
"""

import os
from typing import List, Optional, Tuple, Union

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def save_video(
    data: Union[np.ndarray, torch.Tensor, List[str]],
    output_path: str,
    fps: int = 8,
    quality: int = 8,
    folder: Optional[List[str]] = None,
):
    """
    Save video frames as an MP4/GIF file.

    Args:
        data: video data as (T, H, W, 3) numpy/torch array or list of image paths
        output_path: path to save the video
        fps: frames per second
        quality: compression quality
        folder: optional base folders for image paths
    """
    if isinstance(data, np.ndarray):
        tensor_data = (data * 255).astype(np.uint8) if data.max() <= 1.0 else data.astype(np.uint8)
    elif isinstance(data, torch.Tensor):
        tensor_data = (data.detach().cpu().float() * 255).clamp(0, 255).to(torch.uint8).numpy()
    elif isinstance(data, list):
        if folder is not None:
            images = [np.array(Image.open(os.path.join(f, p))) for f, p in zip(folder, data)]
        else:
            images = [np.array(Image.open(p)) for p in data]
        tensor_data = np.stack(images, axis=0).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    imageio.mimwrite(output_path, tensor_data, fps=fps, quality=quality)


def save_image(
    tensor: torch.Tensor,
    path: str,
    normalize: bool = True,
):
    """
    Save a single image tensor to disk.

    Args:
        tensor: (C, H, W) or (1, C, H, W) image tensor
        path: output file path
        normalize: if True, clamp to [0, 1] range
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    if normalize:
        tensor = tensor.clamp(0, 1)

    from torchvision.utils import save_image as tv_save_image
    tv_save_image(tensor, path)


def resize_image(
    image: Union[torch.Tensor, np.ndarray],
    target_size: Tuple[int, int],
    mode: str = 'bilinear',
) -> Union[torch.Tensor, np.ndarray]:
    """
    Resize an image to target size.

    Args:
        image: (C, H, W) or (H, W, C) image
        target_size: (H, W) target dimensions
        mode: interpolation mode for torch tensors

    Returns:
        resized image
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            if image.shape[0] in [1, 3]:
                image = image.unsqueeze(0)
            else:
                image = image.unsqueeze(0).permute(0, 3, 1, 2)

        return F.interpolate(
            image, size=target_size, mode=mode, align_corners=False
        ).squeeze(0)
    else:
        return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LANCZOS4)


def center_crop_image(
    image: torch.Tensor,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Center crop an image tensor.

    Args:
        image: (..., H, W) tensor
        target_size: (H, W) crop dimensions

    Returns:
        cropped tensor
    """
    h, w = image.shape[-2], image.shape[-1]
    th, tw = target_size

    y1 = (h - th) // 2
    x1 = (w - tw) // 2

    return image[..., y1:y1 + th, x1:x1 + tw]


def normalize_image(
    image: torch.Tensor,
    mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
    std: Tuple[float, ...] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """
    Normalize image to mean=0, std=1 using given statistics.

    Args:
        image: (C, H, W) or (B, C, H, W) image in [0, 1]
        mean: channel-wise mean
        std: channel-wise standard deviation

    Returns:
        normalized image
    """
    mean = torch.tensor(mean, device=image.device).view(-1, 1, 1)
    std = torch.tensor(std, device=image.device).view(-1, 1, 1)

    if image.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return (image - mean) / std


def denormalize_image(
    image: torch.Tensor,
    mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
    std: Tuple[float, ...] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """Reverse image normalization."""
    mean = torch.tensor(mean, device=image.device).view(-1, 1, 1)
    std = torch.tensor(std, device=image.device).view(-1, 1, 1)

    if image.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return image * std + mean


def numpy_to_tensor(image: np.ndarray, device: str = 'cuda') -> torch.Tensor:
    """
    Convert numpy image to torch tensor.

    Args:
        image: (H, W, C) uint8 numpy array
        device: target device

    Returns:
        (C, H, W) float32 tensor in [0, 1]
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    tensor = torch.from_numpy(image).float().to(device)
    if tensor.dim() == 3 and tensor.shape[-1] in [1, 3]:
        tensor = tensor.permute(2, 0, 1)

    return tensor


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert torch tensor to numpy image.

    Args:
        tensor: (C, H, W) float32 tensor in [0, 1]

    Returns:
        (H, W, C) uint8 numpy array
    """
    image = tensor.detach().cpu().float().clamp(0, 1).numpy()
    if image.ndim == 3:
        image = image.transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    return image


def load_image(path: str, device: str = 'cuda') -> torch.Tensor:
    """
    Load an image from disk as a torch tensor.

    Args:
        path: image file path
        device: target device

    Returns:
        (C, H, W) float32 tensor in [0, 1]
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return numpy_to_tensor(img, device=device)
