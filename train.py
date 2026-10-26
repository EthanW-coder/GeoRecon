"""Train GeoRecon video diffusion or optimize a Gaussian scene."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from configs.config import get_parser, opts_to_config
from georecon import GeoRecon


class ConditionDataset(Dataset):
    """Load RGB-depth conditions and observed target videos from NPZ files."""

    def __init__(self, manifest):
        manifest = Path(manifest).resolve()
        self.paths = [(manifest.parent / path).resolve() for path in json.loads(manifest.read_text())['samples']]
        if not self.paths:
            raise ValueError('Training manifest contains no samples')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with np.load(self.paths[index], allow_pickle=False) as arrays:
            result = {key: torch.from_numpy(arrays[key].copy()).float()
                      for key in ('I_render', 'D_mask', 'I_con', 'T_novel', 'I_gt')}
        video = result['I_gt']
        if video.ndim != 4 or video.shape[1] != 3 or video.shape != result['I_render'].shape:
            raise ValueError('Training videos must have shape [T,3,H,W]')
        refs = result['I_con']
        if refs.ndim != 4 or refs.shape[0] < 2 or refs.shape[1] != 3:
            raise ValueError('Training requires at least two reference images [N,3,H,W]')
        for key, value in result.items():
            if not torch.isfinite(value).all():
                raise ValueError(f'Non-finite training input: {key}')
            if key in ('I_render', 'I_con', 'I_gt') and (value.min() < 0 or value.max() > 1):
                raise ValueError(f'{key} must be RGB in [0,1]')
        return result


def train_diffusion(config, manifest, steps, learning_rate, batch_size, resume=False):
    if not manifest:
        raise ValueError('--training_manifest is required for diffusion training')
    if steps < 1 or learning_rate <= 0 or batch_size < 1:
        raise ValueError('Steps, learning rate, and batch size must be positive')
    dataset = ConditionDataset(manifest)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    pipeline = GeoRecon(config)
    pipeline.setup_diffusion(load_georecon_weights=resume)
    model = pipeline.diffusion_model.train()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    iterator = iter(loader)
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(pipeline.device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        result = model(**batch, frame_stride=config.frame_stride)
        result['loss'].backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Step {step+1}/{steps}: loss={result['loss'].item():.6f}")
    destination = Path(config.out_dir) / 'diffusion_model.pth'
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'step': steps}, destination)
    return str(destination)


def main():
    parser = get_parser()
    parser.add_argument('--train_mode', choices=['diffusion', 'gs', 'full'], default='gs')
    parser.add_argument('--training_manifest', help='JSON list of training NPZ samples')
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--resume_diffusion', action='store_true', help='Initialize from the specified GeoRecon checkpoint')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    config = opts_to_config(args)
    if args.train_mode in ('diffusion', 'full'):
        checkpoint = train_diffusion(config, args.training_manifest, args.steps,
                                     args.learning_rate, args.batch_size, args.resume_diffusion)
        config.diffusion_ckpt_path = checkpoint
        config.diffusion_sparse_ckpt_path = checkpoint
    if args.train_mode in ('gs', 'full'):
        GeoRecon(config).run_training()


if __name__ == '__main__':
    main()
