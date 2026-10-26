"""
GeoRecon Inference Script.

Supports:
1. Single-view novel view synthesis
2. Sparse multi-view novel view synthesis with interpolation
3. Full 3D reconstruction with 3DGS optimization

Usage:
    python inference.py --image_dir ./test/images/scene.png --mode single_view
    python inference.py --image_dir ./test/images/scene/ --mode sparse_view
    python inference.py --image_dir ./test/images/scene/ --mode train
"""

import os
import sys
import torch
import argparse
from pathlib import Path

from configs.config import get_parser, opts_to_config
from georecon import GeoRecon


def main():
    parser = get_parser()

    opts = parser.parse_args()
    config = opts_to_config(opts)

    # Resolve experiment name
    if config.exp_name is None:
        if os.path.isfile(config.image_dir):
            config.exp_name = Path(config.image_dir).stem
        else:
            config.exp_name = Path(config.image_dir).name
    config.out_dir = os.path.join(config.out_dir, config.exp_name)

    print("=" * 60)
    print("GeoRecon: Geometry-Aware 3D Reconstruction")
    print("=" * 60)
    print(f"Mode: {config.mode}")
    print(f"Input: {config.image_dir}")
    print(f"Output: {config.out_dir}")
    print(f"Device: {config.device}")
    print(f"VGGT model: {config.vggt_model_path}")
    print(f"Diffusion checkpoint: {config.diffusion_ckpt_path}")
    print("=" * 60)

    georecon = GeoRecon(config)
    georecon.run()


if __name__ == "__main__":
    main()
