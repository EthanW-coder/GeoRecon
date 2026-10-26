"""
Configuration system for GeoRecon.
Provides argument parsing and structured configuration for all pipeline stages.
"""

import os
import argparse
from dataclasses import dataclass, field
from typing import Tuple, Optional, List


@dataclass
class RenderConfig:
    """Configuration for DepMask rendering pipeline."""
    conf_threshold: float = 0.3
    edge_threshold: float = 5.0
    render_resolution: Tuple[int, int] = (512, 512)
    point_radius: float = 1.5
    reproj_error_threshold: float = 2.0
    min_observations: int = 2
    depth_mask_value: float = 0.0
    depth_far_plane: float = 100.0
    depth_near_plane: float = 0.1


@dataclass
class DiffusionConfig:
    """Configuration for depth-aware video diffusion."""
    vae_latent_channels: int = 4
    vae_scale_factor: int = 8
    unet_in_channels: int = 4
    unet_model_channels: int = 320
    render_channels: int = 4
    depth_channels: int = 1
    zero_conv_init: bool = True
    clip_embed_dim: int = 1024
    camera_embed_dim: int = 256
    num_frames: int = 25
    frame_height: int = 576
    frame_width: int = 1024
    num_timesteps: int = 1000


@dataclass
class GSOptimizationConfig:
    """Configuration for 3D Gaussian Splatting optimization."""
    iterations: int = 10000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 10000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 10000
    densify_grad_threshold: float = 0.0005
    prune_threshold: float = 0.005
    sh_degree: int = 3
    white_background: bool = False
    # Region-adaptive optimization loss weights (Eq. 11-13)
    lambda_lpips: float = 0.5      # LPIPS weight within L_novel (Eq. 11)
    lambda_novel: float = 0.3      # novel-view loss weight (Eq. 13)
    lambda_ref: float = 0.7        # reference-view loss weight (Eq. 13)
    # Pseudo-camera (diffusion guidance) parameters
    start_sample_pseudo: int = 2000
    end_sample_pseudo: int = 9500
    sample_pseudo_interval: int = 1
    pseudo_cam_weight: float = 0.05
    pseudo_cam_lpips: bool = True
    pseudo_cam_lpips_weight: float = 0.1
    guidance_interval: int = 260
    guidance_ddim_steps: int = 50
    guidance_center_scale: float = 1.0
    use_trajectory_pool: bool = True
    append_pcd_from_diffusion: bool = False


@dataclass
class GeoReconConfig:
    """Master configuration for GeoRecon pipeline."""
    # General
    image_dir: str = ""
    out_dir: str = "./output"
    device: str = "cuda:0"
    exp_name: Optional[str] = None
    seed: int = 123

    # Mode
    mode: str = "single_view"  # single_view, sparse_view, train

    # Model weight paths
    vggt_model_path: str = "./checkpoints/vggt/model.pt"
    dynamicrafter_root: str = "./third_party/DynamiCrafter"
    dynamicrafter_ckpt_path: str = "./checkpoints/dynamicrafter_512/model.ckpt"
    diffusion_ckpt_path: str = "./checkpoints/georecon/model.ckpt"
    diffusion_sparse_ckpt_path: str = "./checkpoints/georecon/model_sparse.ckpt"
    diffusion_config_path: str = "./configs/model_config.yaml"

    # Camera trajectory
    elevation: float = 5.0
    center_scale: float = 1.0
    d_theta: float = 10.0
    d_phi: float = 30.0
    d_r: float = -0.2
    d_x: float = 0.0
    d_y: float = 0.0
    traj_txt: Optional[str] = None

    # Renderer
    mask_pc: bool = True
    mask_image: bool = False
    reduce_pc: bool = False
    bg_threshold: float = 0.0
    dpt_threshold: float = 1.0

    # Diffusion sampling
    ddim_steps: int = 50
    ddim_eta: float = 1.0
    batch_size: int = 1
    height: int = 512
    width: int = 512
    video_length: int = 25
    frame_stride: int = 10
    unconditional_guidance_scale: float = 7.5
    cfg_img: Optional[float] = None
    timestep_spacing: str = "uniform_trailing"
    guidance_rescale: float = 0.7
    perframe_ae: bool = True
    n_samples: int = 1
    prompt: str = ""
    negative_prompt: str = ""

    # VGGT
    vggt_image_size: int = 518
    vggt_min_conf_thr: float = 3.0

    # Sub-configs
    render: RenderConfig = field(default_factory=RenderConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    gs_opt: GSOptimizationConfig = field(default_factory=GSOptimizationConfig)


def get_parser() -> argparse.ArgumentParser:
    """Create argument parser for GeoRecon."""
    parser = argparse.ArgumentParser(description="GeoRecon: Geometry-Aware 3D Reconstruction")

    # General
    parser.add_argument("--image_dir", type=str, default="./test/images",
                        help="Input image directory or file path")
    parser.add_argument("--out_dir", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed")

    # Mode
    parser.add_argument("--mode", type=str, default="single_view",
                        choices=["single_view", "sparse_view", "train"],
                        help="Operating mode")

    # Model weight paths
    parser.add_argument("--vggt_model_path", type=str,
                        default="./checkpoints/vggt/model.pt",
                        help="Path to VGGT model weights")
    parser.add_argument("--diffusion_ckpt_path", type=str,
                        default="./checkpoints/georecon/model.ckpt",
                        help="Path to diffusion model checkpoint")
    parser.add_argument("--diffusion_sparse_ckpt_path", type=str,
                        default="./checkpoints/georecon/model_sparse.ckpt",
                        help="Path to sparse-view diffusion checkpoint")
    parser.add_argument("--dynamicrafter_root", default="./third_party/DynamiCrafter",
                        help="Local DynamiCrafter source directory")
    parser.add_argument("--dynamicrafter_ckpt_path", default="./checkpoints/dynamicrafter_512/model.ckpt",
                        help="Local DynamiCrafter backbone checkpoint")
    parser.add_argument("--diffusion_config_path", type=str,
                        default="./configs/model_config.yaml",
                        help="Path to diffusion model YAML config")

    # Camera trajectory
    parser.add_argument("--elevation", type=float, default=5.0,
                        help="Elevation angle in degrees")
    parser.add_argument("--center_scale", type=float, default=1.0,
                        help="Scale factor for spherical radius")
    parser.add_argument("--d_theta", type=float, default=10.0,
                        help="Delta theta angle")
    parser.add_argument("--d_phi", type=float, default=30.0,
                        help="Delta phi angle")
    parser.add_argument("--d_r", type=float, default=-0.2,
                        help="Delta radius factor")
    parser.add_argument("--d_x", type=float, default=0.0,
                        help="Delta x translation")
    parser.add_argument("--d_y", type=float, default=0.0,
                        help="Delta y translation")
    parser.add_argument("--traj_txt", type=str, default=None,
                        help="Path to trajectory text file")

    # Renderer
    parser.add_argument("--mask_pc", type=bool, default=True,
                        help="Whether to mask point cloud")
    parser.add_argument("--mask_image", type=bool, default=False,
                        help="Whether to mask rendered images")
    parser.add_argument("--reduce_pc", type=bool, default=False,
                        help="Whether to reduce point cloud")
    parser.add_argument("--bg_threshold", type=float, default=0.0,
                        help="Background removal threshold")
    parser.add_argument("--dpt_threshold", type=float, default=1.0,
                        help="Maximum depth multiplier threshold")
    parser.add_argument("--conf_threshold", type=float, default=0.3,
                        help="DepMask confidence threshold")
    parser.add_argument("--edge_threshold", type=float, default=5.0,
                        help="DepMask edge/depth gradient threshold")
    parser.add_argument("--point_radius", type=float, default=1.5,
                        help="Point splatting radius")

    # Diffusion sampling
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="Number of DDIM sampling steps")
    parser.add_argument("--ddim_eta", type=float, default=1.0,
                        help="DDIM eta parameter")
    parser.add_argument("--height", type=int, default=512,
                        help="Output image height")
    parser.add_argument("--width", type=int, default=512,
                        help="Output image width")
    parser.add_argument("--video_length", type=int, default=25,
                        help="Number of video frames")
    parser.add_argument("--frame_stride", type=int, default=10,
                        help="Frame stride for conditioning")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--cfg_img", type=float, default=None,
                        help="Image guidance scale for multi-condition CFG")
    parser.add_argument("--timestep_spacing", type=str, default="uniform_trailing",
                        help="Timestep spacing strategy")
    parser.add_argument("--guidance_rescale", type=float, default=0.7,
                        help="Guidance rescale factor")
    parser.add_argument("--perframe_ae", type=bool, default=True,
                        help="Use per-frame autoencoder decoding")
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of samples per prompt")
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt for diffusion")

    # VGGT
    parser.add_argument("--vggt_image_size", type=int, default=518,
                        help="VGGT processing width, divisible by 14")
    parser.add_argument("--vggt_min_conf_thr", type=float, default=3.0,
                        help="VGGT minimum confidence threshold")

    # GS optimization
    parser.add_argument("--gs_iterations", type=int, default=10000,
                        help="Number of GS optimization iterations")
    parser.add_argument("--gs_lambda_dssim", type=float, default=0.2,
                        help="SSIM weight in GS loss")
    parser.add_argument("--gs_position_lr_init", type=float, default=0.00016,
                        help="Initial position learning rate")
    parser.add_argument("--gs_feature_lr", type=float, default=0.0025,
                        help="SH feature learning rate")
    parser.add_argument("--gs_opacity_lr", type=float, default=0.05,
                        help="Opacity learning rate")
    parser.add_argument("--gs_scaling_lr", type=float, default=0.005,
                        help="Scale learning rate")
    parser.add_argument("--gs_rotation_lr", type=float, default=0.001,
                        help="Rotation learning rate")
    parser.add_argument("--gs_sh_degree", type=int, default=3,
                        help="Spherical harmonics degree")
    parser.add_argument("--gs_densify_from_iter", type=int, default=500,
                        help="Start iteration for densification")
    parser.add_argument("--gs_densify_until_iter", type=int, default=10000,
                        help="End iteration for densification")
    parser.add_argument("--gs_densify_grad_threshold", type=float, default=0.0005,
                        help="Gradient threshold for densification")
    parser.add_argument("--gs_prune_threshold", type=float, default=0.005,
                        help="Min opacity for pruning")
    parser.add_argument("--gs_pseudo_cam_weight", type=float, default=0.05,
                        help="Pseudo-camera loss weight")
    parser.add_argument("--gs_guidance_interval", type=int, default=260,
                        help="Run diffusion guidance every N iterations")
    parser.add_argument("--gs_start_sample_pseudo", type=int, default=2000,
                        help="Start pseudo-camera supervision at iteration")
    parser.add_argument("--gs_end_sample_pseudo", type=int, default=9500,
                        help="End pseudo-camera supervision at iteration")
    # Region-adaptive optimization loss weights
    parser.add_argument("--gs_lambda_lpips", type=float, default=0.5,
                        help="LPIPS weight in novel-view loss (Eq. 11)")
    parser.add_argument("--gs_lambda_novel", type=float, default=0.3,
                        help="Novel-view loss weight (Eq. 13)")
    parser.add_argument("--gs_lambda_ref", type=float, default=0.7,
                        help="Reference-view loss weight (Eq. 13)")

    return parser


def opts_to_config(opts: argparse.Namespace) -> GeoReconConfig:
    """Convert parsed arguments to GeoReconConfig dataclass."""
    config = GeoReconConfig()

    # Copy direct attributes
    direct_attrs = [
        "image_dir", "out_dir", "device", "exp_name", "seed", "mode",
        "vggt_model_path", "diffusion_ckpt_path", "diffusion_sparse_ckpt_path",
        "dynamicrafter_root", "dynamicrafter_ckpt_path", "diffusion_config_path",
        "elevation", "center_scale", "d_theta", "d_phi", "d_r", "d_x", "d_y",
        "traj_txt", "mask_pc", "mask_image", "reduce_pc", "bg_threshold",
        "dpt_threshold", "ddim_steps", "ddim_eta", "height", "width",
        "video_length", "frame_stride", "unconditional_guidance_scale",
        "cfg_img", "timestep_spacing", "guidance_rescale", "perframe_ae",
        "n_samples", "prompt", "vggt_image_size", "vggt_min_conf_thr",
    ]
    for attr in direct_attrs:
        if hasattr(opts, attr):
            setattr(config, attr, getattr(opts, attr))

    # Update render config
    render_attrs = {
        "conf_threshold": "conf_threshold",
        "edge_threshold": "edge_threshold",
        "point_radius": "point_radius",
    }
    for opt_attr, render_attr in render_attrs.items():
        if hasattr(opts, opt_attr):
            setattr(config.render, render_attr, getattr(opts, opt_attr))

    # Update GS optimization config
    gs_attrs = {
        "gs_iterations": "iterations",
        "gs_lambda_dssim": "lambda_dssim",
        "gs_position_lr_init": "position_lr_init",
        "gs_feature_lr": "feature_lr",
        "gs_opacity_lr": "opacity_lr",
        "gs_scaling_lr": "scaling_lr",
        "gs_rotation_lr": "rotation_lr",
        "gs_sh_degree": "sh_degree",
        "gs_densify_from_iter": "densify_from_iter",
        "gs_densify_until_iter": "densify_until_iter",
        "gs_densify_grad_threshold": "densify_grad_threshold",
        "gs_prune_threshold": "prune_threshold",
        "gs_pseudo_cam_weight": "pseudo_cam_weight",
        "gs_guidance_interval": "guidance_interval",
        "gs_start_sample_pseudo": "start_sample_pseudo",
        "gs_end_sample_pseudo": "end_sample_pseudo",
        # Region-adaptive loss weights
        "gs_lambda_lpips": "lambda_lpips",
        "gs_lambda_novel": "lambda_novel",
        "gs_lambda_ref": "lambda_ref",
    }
    for opt_attr, gs_attr in gs_attrs.items():
        if hasattr(opts, opt_attr):
            setattr(config.gs_opt, gs_attr, getattr(opts, opt_attr))

    config.render.render_resolution = (config.height, config.width)
    return config
