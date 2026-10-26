"""Generate RGB, depth, and confidence conditions from tracked scene geometry."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
try:
    from .prepare import check_camera, sha256, write_json, read_json
except ImportError:
    from prepare import check_camera, sha256, write_json, read_json


def point_confidence(points, poses, Ks, obs_uv, visible, n_min=2, rho=2.0):
    if n_min < 1 or rho <= 0:
        raise ValueError('n_min and rho must be positive')
    V, N = visible.shape
    if obs_uv.shape != (V, N, 2) or points.shape != (N, 3):
        raise ValueError('Observation/point shapes disagree')
    counts = np.zeros(N)
    errors = np.zeros(N)
    rays = []
    for j in range(V):
        check_camera(Ks[j], poses[j])
        cam = (points - poses[j, :3, 3]) @ poses[j, :3, :3]
        projected = cam @ Ks[j].T
        valid = visible[j]
        if np.any(valid & ((cam[:, 2] <= 0) | ~np.isfinite(obs_uv[j]).all(axis=1))):
            raise ValueError('Visible observations require positive depth and finite measured UV')
        uv = projected[:, :2] / np.where(projected[:, 2:3] != 0, projected[:, 2:3], 1)
        counts += valid
        errors += np.where(valid, np.linalg.norm(uv - np.nan_to_num(obs_uv[j]), axis=1), 0)
        ray = points - poses[j, :3, 3]
        rays.append(ray / np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-12))
    max_angle = np.zeros(N)
    for j in range(V):
        for k in range(j):
            angle = np.arccos(np.clip(np.sum(rays[j]*rays[k], axis=1), -1, 1))
            max_angle = np.maximum(max_angle, np.where(visible[j] & visible[k], angle, 0))
    s_obs = np.minimum(1, counts/n_min)
    s_rep = np.where(counts > 0, np.exp(-errors/(np.maximum(counts, 1)*rho)), 0)
    s_ang = max_angle / np.pi
    return (0.4*s_obs + 0.4*s_rep + 0.2*s_ang).astype(np.float32)


def render(points, colors, weights, pose, K, height, width, sigma=1.5, support=3.0):
    if sigma <= 0 or support <= 0 or height < 1 or width < 1:
        raise ValueError('Invalid rendering dimensions or Gaussian parameters')
    check_camera(K, pose)
    cam = (points - pose[:3, 3]) @ pose[:3, :3]
    positive = cam[:, 2] > 0
    cam, colors, weights = cam[positive], colors[positive], weights[positive]
    proj = cam @ K.T
    uv = proj[:, :2] / proj[:, 2:3]
    radius = int(np.ceil(sigma*support))
    keep = (uv[:, 0] >= -radius) & (uv[:, 0] < width+radius) & (uv[:, 1] >= -radius) & (uv[:, 1] < height+radius)
    cam, colors, weights, uv = cam[keep], colors[keep], weights[keep], uv[keep]
    sums = np.zeros((height*width, 5), dtype=np.float64)
    for start in range(0, len(cam), 8192):
        xy = uv[start:start+8192]
        base = np.floor(xy).astype(np.int64)
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                pixels = base + [dx, dy]
                d2 = np.sum((pixels-xy)**2, axis=1)
                valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height) & (d2 <= (support*sigma)**2)
                if not valid.any():
                    continue
                weight = weights[start:start+8192][valid] * np.exp(-d2[valid]/(2*sigma*sigma))
                values = np.column_stack([colors[start:start+8192][valid], cam[start:start+8192, 2][valid], np.ones(valid.sum())])
                ids = pixels[valid, 1]*width + pixels[valid, 0]
                np.add.at(sums, ids, values*weight[:, None])
    sums = sums.reshape(height, width, 5)
    denom = sums[..., 4]
    averaged = np.divide(sums[..., :4], denom[..., None], out=np.zeros_like(sums[..., :4]), where=denom[..., None] > 0)
    return averaged[..., :3].astype(np.float32), averaged[..., 3].astype(np.float32), np.minimum(1, denom).astype(np.float32)


def mask_depth(depth, confidence, tau_conf=.3, tau_edge=5.):
    padded = np.pad(depth, 1, mode='constant')
    h, w = depth.shape
    gx, gy = np.zeros_like(depth), np.zeros_like(depth)
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    for y in range(3):
        for x in range(3):
            gx += sx[y, x]*padded[y:y+h, x:x+w]
            gy += sx.T[y, x]*padded[y:y+h, x:x+w]
    valid = (confidence >= tau_conf) & (np.hypot(gx, gy) <= tau_edge)
    return np.where(valid, depth, 0).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', required=True)
    p.add_argument('--sample', required=True, help='Prepared sample.json for provenance/index validation')
    p.add_argument('--out', required=True)
    p.add_argument('--height', type=int, required=True)
    p.add_argument('--width', type=int, required=True)
    p.add_argument('--n-min', type=int, required=True, help='Minimum observation count for confidence normalization')
    p.add_argument('--support-sigma', type=float, default=3.0)
    a = p.parse_args()
    with np.load(a.geometry, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in ['points', 'colors', 'observing_c2w', 'observing_K', 'obs_uv', 'visible', 'target_c2w', 'target_K', 'observing_indices', 'point_source_indices', 'target_indices']}
    sample = read_json(a.sample)
    for key, expected in [('observing_indices', sample['initialization_indices']),
                          ('point_source_indices', sample['point_source_indices']),
                          ('target_indices', sample['target_indices'])]:
        if data[key].tolist() != expected:
            p.error(f'{key} differs from the prepared sample; possible split/target leakage')
    if sample['height'] != a.height or sample['width'] != a.width:
        p.error('Render resolution must match prepared sample and camera pixel coordinates')
    if sample['split'] == 'eval' and set(data['observing_indices'].tolist()) & set(data['target_indices'].tolist()):
        p.error('Evaluation target images must not be geometry observations')
    points, colors = data['points'], data['colors']
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape or not len(points):
        p.error('Expected nonempty points/colors [N,3]')
    if not np.isfinite(points).all() or not np.isfinite(colors).all() or np.any((colors < 0) | (colors > 1)):
        p.error('Geometry must be finite and colors in [0,1]')
    visible = data['visible']
    if visible.dtype != bool:
        p.error('visible must be boolean')
    if len(data['observing_c2w']) != len(visible) or len(data['observing_K']) != len(visible) or len(data['observing_indices']) != len(visible):
        p.error('Observing camera counts disagree')
    if len(data['target_c2w']) != len(data['target_K']) or len(data['target_c2w']) != len(data['target_indices']) or len(data['target_c2w']) == 0:
        p.error('Target camera counts disagree or empty trajectory')
    weights = point_confidence(points, data['observing_c2w'], data['observing_K'], data['obs_uv'], visible, a.n_min)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / 'point_confidence.npy', weights)
    for i, (pose, K) in enumerate(zip(data['target_c2w'], data['target_K'])):
        rgb, depth, confidence = render(points, colors, weights, pose, K, a.height, a.width, support=a.support_sigma)
        np.savez_compressed(out / f'{i:06d}.npz', rgb=rgb, depth=depth, confidence=confidence,
                            masked_depth=mask_depth(depth, confidence), c2w=pose, K=K)
    write_json(out / 'metadata.json', {'geometry_sha256': sha256(a.geometry), 'sample_sha256': sha256(a.sample), 'height': a.height, 'width': a.width,
        'alpha_beta_gamma': [.4, .4, .2], 'rho_pixels': 2., 'sigma_pixels': 1.5, 'tau_conf': .3,
        'tau_edge': 5., 'n_min': a.n_min, 'support_sigma': a.support_sigma, 'sobel_padding': 'zero',
        'implementation': 'numpy', 'frames': len(data['target_c2w'])})


if __name__ == '__main__':
    main()
