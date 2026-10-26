# Data preparation


## Official datasets

| Dataset | Download | Use |
| --- | --- | --- |
| DL3DV-10K | [Official repository](https://github.com/DL3DV-10K/Dataset), [videos](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-video), [960P frames](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P) | Training |
| DL3DV-Evaluation | [Official evaluation split](https://huggingface.co/datasets/DL3DV/DL3DV-Evaluation) | 20 evaluation scenes |
| RealEstate10K | [Download page](https://google.github.io/realestate10k/download.html), [camera metadata](https://storage.cloud.google.com/realestate10k-public-files/RealEstate10K.tar.gz) | Training and two-view Easy Set evaluation |
| ACID | [Project page](https://infinite-nature.github.io/), [ACID v1 metadata](https://storage.googleapis.com/gresearch/aerial-coastline-imagery-dataset/acid_v1_release.tar.gz) | Training and two-view Easy Set evaluation |
| Tanks and Temples | [Official downloads](https://www.tanksandtemples.org/download/) | 7 out-of-domain evaluation scenes |
| Mip-NeRF 360 | [Project page](https://jonbarron.info/mipnerf360/), [part 1](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip), [part 2](https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip) | All 9 scenes, evaluation only |

DL3DV-Evaluation is independent of DL3DV-10K. Use it for evaluation rather than the 140-scene DL3DV-Benchmark subset. RE10K and ACID metadata contain video URLs, timestamps, and camera parameters; prepare the corresponding videos or extracted frames locally. Use ACID v1 for the ACID experiments.

## Configuration

Edit [configs/data_processing.yaml](../configs/data_processing.yaml). All paths in that file are resolved relative to the YAML file.

| Setting | Purpose |
| --- | --- |
| `catalog` | Local scene catalog |
| `output_dir` | Destination for selected manifests and sampling plan |
| `sampling` | Initialization frame count, video length, frame-stride range, references, clips per scene |
| `resolutions` | `[height, width]` for each processing stage |
| `evaluation.view_counts` | Input view counts, default `[2, 3, 6]` |
| `evaluation.frame_stride` | Input frame stride, default `10` |
| `evaluation.enforce_scene_counts` | Check each enabled dataset's configured scene count |
| `evaluation.datasets.<name>.enabled` | Include this dataset's evaluation scenes |
| `evaluation.datasets.<name>.scene_ids` | Explicit scene selection |
| `evaluation.datasets.<name>.expected_scenes` | Expected number of evaluation scenes |
| `evaluation.datasets.<name>.samples` | Inline per-scene frame indices |
| `evaluation.datasets.<name>.index_file` | External JSON index, including Easy Set indices |

Set the DL3DV 20-scene and Tanks 7-scene lists in `scene_ids`. Enable each evaluation dataset after configuring its scenes and indices. The Mip-NeRF 360 list contains `bicycle`, `bonsai`, `counter`, `flowers`, `garden`, `kitchen`, `room`, `stump`, and `treehill`.

Example dataset entry:

```yaml
dl3dv:
  enabled: true
  expected_scenes: 20
  scene_ids: [scene_01, scene_02]  # Extend this list to the selected 20 scenes.
  index_file: ../data/indices/dl3dv.json
  samples: []
```

For a smaller experiment, change `expected_scenes` or disable `enforce_scene_counts`. The frame selection is explicit and remains stable across runs.

Inline frame records use zero-based indices into the scene's indexed frame sequence:

```yaml
samples:
  - scene_id: scene_01
    sample_id: two_view
    reference_indices: [0, 10]
    target_indices: [1, 2, 3, 4, 5, 6, 7, 8, 9]
  - scene_id: scene_01
    sample_id: three_view
    reference_indices: [0, 10, 20]
    target_indices: [1, 5, 11, 15, 19]
```

Configure the target frames and all desired input-view settings explicitly. Reference and target sets must be disjoint during evaluation. RE10K and ACID use `protocol: reconx_easy` with two reference views. Set their `index_file` to the selected Easy Set JSON, or enter its frame indices under `samples`. Easy Set records retain their supplied reference separation; the generic stride setting applies to DL3DV, Tanks, and Mip-NeRF 360.

External index format:

```json
{
  "samples": [
    {
      "scene_id": "clip_id",
      "sample_id": "easy_pair",
      "reference_indices": [12, 72],
      "target_indices": [24, 36, 48, 60],
      "index_source": "reconx_easy"
    }
  ]
}
```

An optional `dataset` field must match the YAML dataset entry. If an external index uses timestamps, map them to the indexed sequence's `timestamp_us` values before entering frame indices. Dataset-specific index files can contain additional scenes; the YAML `scene_ids` list determines which are included.

## Install preprocessing dependencies

```bash
pip install -r data_preparation/requirements.txt
```

FFmpeg is required only when extracting frames from local videos. These commands do not download data or model weights.

## Index local images

For DL3DV, Tanks, and Mip-NeRF 360, index each scene's image directory:

```bash
python data_preparation/prepare.py index-images \
  --images ./data/raw/dl3dv/scene_01/images \
  --out ./data/sequences/dl3dv_scene_01.json
```

Files are ordered naturally by filename, so `1.png`, `2.png`, `10.png` appear in that order. Keep depth maps, thumbnails, and alternate resolutions outside the indexed image directory. The frame stride refers to this indexed sequence; preserve the extraction rate used for the experiment.

Optional camera data can be supplied with `--cameras cameras.npz`:

| Key | Shape | Convention |
| --- | --- | --- |
| `names` | `[F]` Unicode | Filenames matching the indexed image order |
| `K` | `[F,3,3]` | Pixel-coordinate intrinsics at the source image resolution |
| `c2w` | `[F,4,4]` | OpenCV camera-to-world: x right, y down, z forward |

Convert COLMAP world-to-camera matrices by inversion. Convert OpenGL camera-to-world matrices by right-multiplying `diag(1,-1,-1,1)`. Use undistorted images with their corresponding intrinsics. External cameras are optional when preparing images for VGGT.

## Index RealEstate10K and ACID

Frames are named `<timestamp_us>.png` and follow the official metadata row order:

```bash
python data_preparation/prepare.py index-re10k \
  --metadata ./data/raw/realestate10k/train/clip_id.txt \
  --images ./data/frames/realestate10k/clip_id \
  --out ./data/sequences/realestate10k_clip_id.json
```

Add `--video ./data/videos/source.mp4` to extract missing timestamp frames from a local video. The parser converts microseconds to seconds for FFmpeg, expands normalized intrinsics to pixel coordinates, and inverts the provided world-to-camera matrices. Missing frames produce an error rather than changing the index ordering. Use the same frame-extraction convention for training and evaluation indices.

## Scene catalog

Create `data/catalog.json` with the local scenes:

```json
{
  "scenes": [
    {
      "dataset": "dl3dv",
      "scene_id": "train_scene",
      "source_id": "dl3dv:source_video_1",
      "split": "train",
      "source_split": "DL3DV-10K",
      "sequence": "sequences/dl3dv_train_scene.json"
    },
    {
      "dataset": "dl3dv",
      "scene_id": "scene_01",
      "source_id": "dl3dv:eval_scene_01",
      "split": "eval",
      "source_split": "DL3DV-Evaluation",
      "sequence": "sequences/dl3dv_scene_01.json"
    }
  ]
}
```

Dataset names are `dl3dv`, `realestate10k`, `acid`, `tanks`, and `mipnerf360`. For RE10K and ACID, preserve `source_split: train` or `test`; for Tanks and Mip use `split: eval` and record the official source group in `source_split`.

Use the same `source_id` for clips from the same original video or physical scene. The validator checks scene IDs, source IDs, and image paths across training and evaluation. Training scenes in the catalog are included automatically; evaluation scenes are selected by the YAML configuration.

## Create manifests and process images

```bash
python data_preparation/configure.py --config configs/data_processing.yaml --check
python data_preparation/configure.py --config configs/data_processing.yaml

python data_preparation/prepare.py materialize \
  --plan data/manifests/plan.json --stage train-low --out data/prepared_low

python data_preparation/prepare.py materialize \
  --plan data/manifests/plan.json --stage train-high --out data/prepared_high

python data_preparation/prepare.py materialize \
  --plan data/manifests/plan.json --stage eval --out data/prepared_eval
```

Default training sampling selects 50 initialization frames at a constant random stride from 2 to 5. A 25-frame window provides the training targets, with at least two reference views selected from that window. The seed is derived per scene, keeping selections independent of catalog order. Short sequences are rejected when no configured stride can produce the initialization clip.

The output contains `initialization/`, `reference/`, `target_gt/`, and `sample.json`. Evaluation initialization contains only reference images. Evaluation targets are kept in a separate directory and are not inputs to geometry estimation or reference-image encoding.

The processing stages use 320x512, 576x1024, and 512x512 by default. RGB images are resized with bilinear interpolation. Intrinsics are transformed by the same pixel-center affine map, including the half-pixel offset. Image sizes, cameras, frame indices, and source hashes are retained in `sample.json`. Existing output directories are not overwritten.

## Geometry and RGB-depth conditions

Run VGGT jointly on the initialization images and retain its camera and image-coordinate transforms. Use the selected point-source frames to assemble the point cloud. Compute reprojection errors from measured multi-view tracks, not from the point's own projection.

Export geometry as NPZ:

| Key | Shape |
| --- | --- |
| `points`, `colors` | `[P,3]`; world points and RGB in `[0,1]` |
| `observing_c2w`, `observing_K` | `[V,4,4]`, `[V,3,3]` |
| `obs_uv`, `visible` | `[V,P,2]`, `[V,P]` boolean |
| `target_c2w`, `target_K` | `[T,4,4]`, `[T,3,3]` |
| `observing_indices` | Initialization frame indices from `sample.json` |
| `point_source_indices` | Selected point-source frame indices |
| `target_indices` | Target frame indices |

Cameras and observed pixels must use the processed image resolution. Keep a common depth scale across the clip. Target cameras in training must correspond to the observed target video frames.

```bash
python data_preparation/render_conditions.py \
  --geometry ./data/geometry/scene.npz \
  --sample ./data/prepared_low/dl3dv/scene/train_00000/sample.json \
  --height 320 --width 512 --n-min 2 \
  --out ./data/conditions/scene/train_00000
```

The renderer uses confidence weights `(0.4,0.4,0.2)`, reprojection decay `rho=2`, Gaussian width `sigma=1.5`, confidence threshold `0.3`, and Sobel threshold `5`. Gaussian support defaults to 3 sigma and Sobel uses zero padding. Set `--n-min` to the chosen minimum observation count. RGB and depth use normalized weighted accumulation; confidence is the accumulated weight clipped to one.

Each frame contains `rgb`, `depth`, `masked_depth`, `confidence`, `c2w`, and `K`. Generate conditions separately for each output resolution.

## Training cache

`train.py --training_manifest` reads a JSON file containing paths to NPZ samples:

```json
{"samples": ["cache/scene_01.npz", "cache/scene_02.npz"]}
```

Each sample stores float arrays:

| Key | Shape | Content |
| --- | --- | --- |
| `I_render` | `[T,3,H,W]` | Rendered RGB in `[0,1]` |
| `D_mask` | `[T,1,H,W]` | Masked depth |
| `I_con` | `[N,3,H,W]`, `N>=2` | Reference RGB in `[0,1]` |
| `T_novel` | `[T,6]` | Axis-angle rotation followed by translation |
| `I_gt` | `[T,3,H,W]` | Observed target RGB in `[0,1]` |

Assemble each cache from the corresponding prepared sample and condition files. Keep reference count, frame count, and resolution consistent within a batch. `I_gt` contains the observed video frames; generated views are used separately for Gaussian optimization. The model adapter performs VAE and CLIP normalization and keeps those encoders frozen.
