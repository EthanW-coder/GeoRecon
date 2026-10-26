# GeoRecon: Confidence-Aware Geometric Guidance for Sparse-View 3D Reconstruction with Video Diffusion
GeoRecon combines confidence-aware RGB-depth guidance, reference-image semantics, and camera motion to synthesize novel views from sparse images. Reference views and generated views supervise a region-adaptive Gaussian scene representation.

## Installation

Run the following commands from the repository root. Use Python 3.10 and a CUDA-enabled PyTorch installation for reconstruction and video generation.

```bash
conda create -n georecon python=3.10 -y
conda activate georecon
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Install the [official VGGT implementation](https://github.com/facebookresearch/vggt):

```bash
mkdir -p third_party
git clone https://github.com/facebookresearch/vggt.git third_party/vggt
pip install -e third_party/vggt --no-deps
```

Download the [VGGT-1B checkpoint](https://huggingface.co/facebook/VGGT-1B):

```bash
mkdir -p checkpoints/vggt
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='facebook/VGGT-1B', filename='model.pt', local_dir='checkpoints/vggt')"
```

Clone the [official DynamiCrafter implementation](https://github.com/Doubiiu/DynamiCrafter). Its runtime dependencies are included in this repository's `requirements.txt`.

```bash
git clone https://github.com/Doubiiu/DynamiCrafter.git third_party/DynamiCrafter
```

Download the [DynamiCrafter 512 backbone checkpoint](https://huggingface.co/Doubiiu/DynamiCrafter_512):

```bash
mkdir -p checkpoints/dynamicrafter_512
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Doubiiu/DynamiCrafter_512', filename='model.ckpt', local_dir='checkpoints/dynamicrafter_512')"
```

Keep the two types of checkpoints in separate directories:

```text
checkpoints/
  vggt/model.pt
  dynamicrafter_512/model.ckpt
  georecon/model.ckpt
  georecon/model_sparse.ckpt
```

### Gaussian rasterization

Install the CUDA extensions from [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) using a CUDA toolkit compatible with the installed PyTorch build:

```bash
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git third_party/gaussian-splatting
pip install ./third_party/gaussian-splatting/submodules/diff-gaussian-rasterization
pip install ./third_party/gaussian-splatting/submodules/simple-knn
```

## Data preparation

Edit [configs/data_processing.yaml](configs/data_processing.yaml) to configure:

- DL3DV evaluation scene IDs, with an expected count of 20.
- Tanks and Temples evaluation scene IDs, with an expected count of 7.
- Mip-NeRF 360 scenes.
- Per-scene reference and target frame indices.
- RealEstate10K and ACID Easy Set index files or inline records.
- Initialization clips, frame strides, reference counts, and image resolutions.

Dataset download links, index schemas, and preprocessing commands are provided in [data_preparation/README.md](data_preparation/README.md).

```bash
python data_preparation/configure.py --config configs/data_processing.yaml --check
python data_preparation/configure.py --config configs/data_processing.yaml
```


## Inference

```bash
python inference.py \
  --image_dir ./data/scene/images \
  --mode sparse_view \
  --vggt_model_path ./checkpoints/vggt/model.pt \
  --dynamicrafter_ckpt_path ./checkpoints/dynamicrafter_512/model.ckpt \
  --diffusion_sparse_ckpt_path ./checkpoints/georecon/model_sparse.ckpt \
  --height 512 --width 512 \
  --out_dir ./output
```

## Training

Video diffusion training.

```bash
python train.py \
  --train_mode diffusion \
  --training_manifest ./data/training.json \
  --dynamicrafter_ckpt_path ./checkpoints/dynamicrafter_512/model.ckpt \
  --steps 50000 --learning_rate 1e-5 --batch_size 1 \
  --out_dir ./output/diffusion
```
3DGS training.

```bash
python train.py \
  --train_mode gs \
  --image_dir ./data/scene/images \
  --diffusion_ckpt_path ./output/diffusion/diffusion_model.pth \
  --out_dir ./output/scene
```


## Acknowledgments

We gratefully acknowledge the projects that helped make GeoRecon possible, including:
- [VGGT](https://github.com/facebookresearch/vggt)
- [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)

## License

This repository's own source code is released under the [MIT License](LICENSE).



