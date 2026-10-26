# Third-party notices

GeoRecon's own source code is released under the MIT License (see [LICENSE](LICENSE)).

GeoRecon is a pipeline that runs on top of third-party components. Those components are **not**
distributed with this repository: the installation steps in [README.md](README.md) point to their
official upstream sources, and they remain governed by their own licenses. Their terms are not
superseded by GeoRecon's MIT license, and they restrict what you may do with the combined system.

Review each upstream license before any use beyond academic research.

| Component | Upstream | License | Restrictions |
| --- | --- | --- | --- |
| VGGT | https://github.com/facebookresearch/vggt | Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) | Non-commercial use only |
| 3D Gaussian Splatting | https://github.com/graphdeco-inria/gaussian-splatting | Gaussian-Splatting License (Inria / MPII) | Research and evaluation purposes only; non-commercial; no sublicensing |
| diff-gaussian-rasterization | https://github.com/graphdeco-inria/diff-gaussian-rasterization | Gaussian-Splatting License (Inria / MPII) | Same as above |
| simple-knn | https://github.com/graphdeco-inria/gaussian-splatting | Gaussian-Splatting License (Inria / MPII) | Same as above |
| DynamiCrafter | https://github.com/Doubiiu/DynamiCrafter | Apache-2.0 (code) | Verify the license of the specific model weights you download before use |

## Practical consequence

Because the geometry backbone (VGGT, CC BY-NC 4.0) and the rasterization backend
(diff-gaussian-rasterization, Inria research license) are required at runtime, **the assembled
GeoRecon pipeline cannot be used for commercial purposes** even though this repository's own code
is MIT-licensed.

## Model checkpoints

Model weights are not included in this repository. GeoRecon checkpoints are fine-tuned from the
DynamiCrafter backbone, so any redistribution of those weights is additionally subject to the terms
attached to the DynamiCrafter and Stable Diffusion model weights. Check the upstream terms before
publishing weights.

## Datasets

Datasets used by the preparation scripts (DL3DV-10K, DL3DV-Evaluation, RealEstate10K, ACID,
Tanks and Temples, Mip-NeRF 360) are not distributed here and carry their own terms. See
[data_preparation/README.md](data_preparation/README.md) for upstream links.