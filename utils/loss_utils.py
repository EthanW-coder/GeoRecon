"""
Loss functions for 3D reconstruction and novel view synthesis.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard L1 loss."""
    return F.l1_loss(pred, target)


def l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard L2 / MSE loss."""
    return F.mse_loss(pred, target)


def l1_loss_masked(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Masked L1 loss.

    Args:
        pred: (..., H, W) predicted image
        target: (..., H, W) target image
        mask: (H, W) boolean mask, True = valid

    Returns:
        scalar loss
    """
    if mask is None:
        return F.l1_loss(pred, target)

    diff = torch.abs(pred - target)
    if mask.dim() < diff.dim():
        mask = mask.unsqueeze(0).unsqueeze(0)

    return (diff * mask.float()).sum() / (mask.float().sum() + 1e-8)


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    """
    Compute SSIM-based loss (1 - SSIM).

    Args:
        pred: (B, C, H, W) predicted image
        target: (B, C, H, W) target image
        window_size: size of gaussian kernel
        size_average: whether to average over batch

    Returns:
        ssim loss: 1 - SSIM score
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Gaussian window
    def gaussian(window_size, sigma):
        gauss = torch.tensor([
            math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    window_1d = gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float()
    window = window_2d.unsqueeze(0).unsqueeze(0).to(pred.device)
    window = window.expand(pred.size(1), 1, window_size, window_size)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=pred.size(1))
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=pred.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=pred.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=pred.size(1)) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=pred.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map.mean([1, 2, 3])


def lpips_loss(pred: torch.Tensor, target: torch.Tensor, lpips_model=None) -> torch.Tensor:
    """
    Compute LPIPS perceptual loss.

    Args:
        pred: (B, C, H, W) predicted image in [0, 1]
        target: (B, C, H, W) target image
        lpips_model: pretrained LPIPS model (if None, returns 0)

    Returns:
        scalar LPIPS loss
    """
    if lpips_model is None:
        return torch.tensor(0.0, device=pred.device)

    if pred.ndim == 3:
        pred, target = pred.unsqueeze(0), target.unsqueeze(0)
    return lpips_model(pred, target, normalize=True).mean()


def vgg_perceptual_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    vgg_model,
) -> torch.Tensor:
    """
    Compute VGG-based perceptual loss.

    Args:
        pred: (B, C, H, W) predicted image
        target: (B, C, H, W) target image
        vgg_model: pretrained VGG network

    Returns:
        scalar perceptual loss
    """
    if pred.min() < 0:
        pred = (pred + 1) / 2
        target = (target + 1) / 2

    pred_features = vgg_model(pred)
    target_features = vgg_model(target)

    loss = 0.0
    for pf, tf in zip(pred_features, target_features):
        loss += F.l1_loss(pf, tf)

    return loss


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """
    Total variation loss to encourage smoothness.

    Args:
        img: (B, C, H, W) image tensor

    Returns:
        scalar TV loss
    """
    diff_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    diff_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    return diff_h + diff_w


def depth_smoothness_loss(depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """
    Edge-aware depth smoothness loss.

    Encourages depth to be smooth while respecting image edges.

    Args:
        depth: (B, 1, H, W) depth map
        image: (B, 3, H, W) reference image

    Returns:
        scalar smoothness loss
    """
    depth_dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
    depth_dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]

    image_dx = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), dim=1, keepdim=True)
    image_dy = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), dim=1, keepdim=True)

    weight_x = torch.exp(-image_dx)
    weight_y = torch.exp(-image_dy)

    loss = (weight_x * torch.abs(depth_dx)).mean() + (weight_y * torch.abs(depth_dy)).mean()
    return loss


def rendering_loss(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    lambda_ssim: float = 0.2,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Combined rendering loss: L1 + lambda * (1 - SSIM).

    Args:
        pred_rgb: (C, H, W) or (B, C, H, W) predicted RGB
        gt_rgb: same shape, ground truth RGB
        lambda_ssim: SSIM loss weight
        mask: optional boolean mask

    Returns:
        total_loss: scalar loss
        metrics: dict with individual loss components
    """
    if pred_rgb.dim() == 3:
        pred_rgb = pred_rgb.unsqueeze(0)
        gt_rgb = gt_rgb.unsqueeze(0)

    Ll1 = l1_loss_masked(pred_rgb, gt_rgb, mask)
    Lssim = ssim_loss(pred_rgb, gt_rgb) if lambda_ssim > 0 else torch.tensor(0.0, device=pred_rgb.device)

    total = (1.0 - lambda_ssim) * Ll1 + lambda_ssim * Lssim

    return total, {"l1": Ll1.item(), "ssim": Lssim.item() if isinstance(Lssim, torch.Tensor) else Lssim}


def confidence_weighted_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    """
    Confidence-weighted L1 loss for region-adaptive optimization (Eq. 11 numerator).

    Computes: sum_u( C(u) * |pred(u) - target(u)|_1 ) / ( sum_u( C(u) ) + delta )

    Args:
        pred: (C, H, W) or (B, C, H, W) predicted image
        target: same shape, target image
        confidence: (H, W) or (B, 1, H, W) confidence map in [0, 1]

    Returns:
        scalar loss
    """
    delta = 1e-8

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    # Ensure confidence has batch and channel dims: (B, 1, H, W)
    if confidence.dim() == 2:
        confidence = confidence.unsqueeze(0).unsqueeze(0)
    elif confidence.dim() == 3:
        confidence = confidence.unsqueeze(1)

    # Per-pixel L1 averaged over color channels: (B, 1, H, W)
    diff = torch.abs(pred - target).mean(dim=1, keepdim=True)

    weighted_diff = diff * confidence
    loss = weighted_diff.sum() / (confidence.sum() + delta)

    return loss


def novel_view_loss(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    confidence: torch.Tensor,
    lambda_lpips: float = 0.5,
    lpips_model=None,
) -> Tuple[torch.Tensor, dict]:
    """
    Novel-view loss L_novel from Eq. 11.

    L_novel = confidence_weighted_L1 + lambda_lpips * LPIPS

    Args:
        pred_rgb: (C, H, W) or (B, C, H, W) rendered GS output
        gt_rgb: same shape, diffusion-generated novel view
        confidence: (H, W) DepMask confidence map Cm(u)
        lambda_lpips: weight for LPIPS term
        lpips_model: pretrained LPIPS model

    Returns:
        total_loss: scalar
        metrics: dict with components
    """
    l1_conf = confidence_weighted_l1_loss(pred_rgb, gt_rgb, confidence)
    l_lpips = lpips_loss(pred_rgb, gt_rgb, lpips_model) if lambda_lpips > 0 else torch.tensor(0.0, device=pred_rgb.device)

    total = l1_conf + lambda_lpips * l_lpips

    metrics = {
        "l1_weighted": l1_conf.item(),
        "lpips": l_lpips.item() if isinstance(l_lpips, torch.Tensor) else l_lpips,
    }
    return total, metrics


def reference_view_loss(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """
    Reference-view loss L_ref from Eq. 12.

    L_ref = D-SSIM = 1 - SSIM(pred, gt)

    Args:
        pred_rgb: (C, H, W) or (B, C, H, W) rendered GS output
        gt_rgb: same shape, observed reference image

    Returns:
        total_loss: scalar
        metrics: dict with components
    """
    if pred_rgb.ndim == 3:
        pred_rgb, gt_rgb = pred_rgb.unsqueeze(0), gt_rgb.unsqueeze(0)
    dssim = ssim_loss(pred_rgb, gt_rgb)
    return dssim, {"dssim": dssim.item()}


def region_adaptive_loss(
    novel_pred: torch.Tensor,
    novel_target: torch.Tensor,
    confidence: torch.Tensor,
    ref_pred: torch.Tensor,
    ref_target: torch.Tensor,
    lambda_lpips: float = 0.5,
    lambda_novel: float = 0.3,
    lambda_ref: float = 0.7,
    lpips_model=None,
) -> Tuple[torch.Tensor, dict]:
    """
    Region-adaptive loss L_ra from Eq. 13.

    L_ra = lambda_novel * L_novel + lambda_ref * L_ref

    Computed per-iteration from one novel view and one reference view.

    Args:
        novel_pred: (C, H, W) rendered GS output for novel view
        novel_target: (C, H, W) diffusion-generated novel view frame
        confidence: (H, W) DepMask confidence map for the novel view
        ref_pred: (C, H, W) rendered GS output for reference view
        ref_target: (C, H, W) observed reference image
        lambda_lpips: LPIPS weight within L_novel
        lambda_novel: novel-view loss weight
        lambda_ref: reference-view loss weight
        lpips_model: pretrained LPIPS model

    Returns:
        total_loss: scalar
        metrics: dict with all loss components
    """
    L_novel, novel_metrics = novel_view_loss(
        novel_pred, novel_target, confidence,
        lambda_lpips=lambda_lpips, lpips_model=lpips_model,
    )
    L_ref, ref_metrics = reference_view_loss(ref_pred, ref_target)

    total = lambda_novel * L_novel + lambda_ref * L_ref

    metrics = {
        "L_novel": L_novel.item(),
        "L_ref": L_ref.item(),
        "L_ra": total.item(),
        **{f"novel_{k}": v for k, v in novel_metrics.items()},
        **{f"ref_{k}": v for k, v in ref_metrics.items()},
    }
    return total, metrics
