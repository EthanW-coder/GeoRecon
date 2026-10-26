"""
GeoRecon: Geometry-Aware 3D Reconstruction with Depth-Video Diffusion Prior.

Main orchestrator that integrates:
1. VGGT-based sparse 3D reconstruction
2. DepMask (Confidence & Geometry Dual-Guided) rendering
3. Depth-aware video diffusion for novel view synthesis
4. 3D Gaussian Splatting optimization with diffusion guidance
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from configs.config import GeoReconConfig
from models.depmask_renderer import DepMaskRenderPipeline, RenderConfig as DepMaskRenderConfig
from models.depth_video_diffusion import DepthVideoDiffusion, VideoDiffusionConfig
from models.vggt_interface import VGGTInterface
from models.gs_scene import GaussianModel, Camera, PseudoCamera, SceneRepresentation
from modules.gs_renderer import GSRenderConfig
from utils.camera_utils import (
    generate_trajectory, generate_trajectory_interp, generate_trajectory_from_txt,
)
from utils.loss_utils import region_adaptive_loss
from utils.image_utils import save_video, load_image
from utils.geometry_utils import get_world_to_view_matrix, get_projection_matrix, get_fov


class GeoRecon:
    """
    GeoRecon: Complete geometry-aware 3D reconstruction pipeline.

    Pipeline stages:
    1. Sparse image input -> VGGT feature extraction -> point cloud + camera poses
    2. DepMask renderer -> RGB renders + confidence-masked depth from sparse points
    3. Depth-video diffusion -> novel view video generation
    4. 3DGS optimization -> high-quality 3D Gaussian representation
    """

    def __init__(self, config: GeoReconConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Set seed
        seed_everything(config.seed)

        # Pipeline components
        self.vggt_interface: Optional[VGGTInterface] = None
        self.depmask_pipeline: Optional[DepMaskRenderPipeline] = None
        self.diffusion_model: Optional[DepthVideoDiffusion] = None
        self.gaussian_model: Optional[GaussianModel] = None
        self.scene: Optional[SceneRepresentation] = None

        # State
        self.images: Optional[torch.Tensor] = None
        self.points_3d: Optional[torch.Tensor] = None
        self.points_color: Optional[torch.Tensor] = None
        self.points_confidence: Optional[torch.Tensor] = None
        self.camera_poses: Optional[torch.Tensor] = None
        self.intrinsics: Optional[torch.Tensor] = None
        self.is_initialized = False

    def setup_vggt(self):
        """Initialize VGGT interface for sparse 3D reconstruction."""
        self.vggt_interface = VGGTInterface(
            model_path=self.config.vggt_model_path,
            device=str(self.device),
            image_size=self.config.vggt_image_size,
            min_conf_thr=self.config.vggt_min_conf_thr,
        )
        self.vggt_interface.load_model()

    def setup_depmask_renderer(self):
        """Initialize DepMask rendering pipeline."""
        render_cfg = DepMaskRenderConfig(
            conf_threshold=self.config.render.conf_threshold,
            edge_threshold=self.config.render.edge_threshold,
            render_resolution=self.config.render.render_resolution,
            point_radius=self.config.render.point_radius,
        )
        self.depmask_pipeline = DepMaskRenderPipeline(render_cfg)

    def setup_diffusion(self, load_georecon_weights=True):
        """Load the DynamiCrafter backbone and GeoRecon control weights."""
        from models.dynamicrafter_interface import load_dynamicrafter
        from models.checkpoints import load_state_dict_file

        ckpt_path = self.config.diffusion_ckpt_path
        if 'sparse' in self.config.mode:
            ckpt_path = self.config.diffusion_sparse_ckpt_path
        if load_georecon_weights and not Path(ckpt_path).is_file():
            raise FileNotFoundError(f'GeoRecon checkpoint not found: {ckpt_path}')
        settings = OmegaConf.load(self.config.diffusion_config_path)
        diff_cfg = VideoDiffusionConfig(**OmegaConf.to_container(settings.model, resolve=True))
        backbone = load_dynamicrafter(
            self.config.dynamicrafter_root, self.config.dynamicrafter_ckpt_path,
            self.config.diffusion_config_path, self.device,
        )
        self.diffusion_model = DepthVideoDiffusion(
            config=diff_cfg, base_diffusion_model=backbone,
        ).to(self.device)
        if load_georecon_weights:
            self.diffusion_model.load_state_dict(load_state_dict_file(ckpt_path), strict=True)

    def setup_gs_model(self):
        """Initialize 3D Gaussian Splatting model."""
        gs_config = self.config.gs_opt
        self.gaussian_model = GaussianModel(
            sh_degree=gs_config.sh_degree,
            white_background=gs_config.white_background,
        )

    def load_input_images(self, image_path: str) -> torch.Tensor:
        """
        Load input images from a single file or directory.

        Args:
            image_path: path to a single image file or directory of images

        Returns:
            images: (N, 3, H, W) tensor
        """
        if os.path.isfile(image_path):
            img = load_image(image_path, device=str(self.device))
            images = img.unsqueeze(0)  # (1, 3, H, W)
        elif os.path.isdir(image_path):
            image_files = sorted([
                f for f in os.listdir(image_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
            ])
            images = []
            for fname in image_files:
                img = load_image(os.path.join(image_path, fname), device=str(self.device))
                images.append(img)
            images = torch.stack(images)  # (N, 3, H, W)
        else:
            raise FileNotFoundError(f"Image path not found: {image_path}")

        # Resize to target resolution if needed
        H, W = images.shape[-2:]
        target_H, target_W = self.config.height, self.config.width
        if H != target_H or W != target_W:
            images = F.interpolate(
                images, size=(target_H, target_W),
                mode='bilinear', align_corners=False,
            )

        return images

    def reconstruct_sparse(self, images: torch.Tensor):
        """
        Stage 1: Reconstruct sparse 3D structure from input images using VGGT.

        Args:
            images: (N, 3, H, W) input images
        """
        print(">>> Stage 1: Sparse 3D reconstruction with VGGT...")
        self.images = images

        if self.vggt_interface is None:
            self.setup_vggt()

        vggt_output = self.vggt_interface.run_reconstruction(images)

        self.camera_poses = vggt_output['poses']
        self.intrinsics = vggt_output['intrinsics']
        depth_maps = vggt_output['depth']
        if 'confidence_mask' in vggt_output:
            depth_maps = torch.where(vggt_output['confidence_mask'], depth_maps, torch.zeros_like(depth_maps))

        # Extract point cloud from depth maps
        if self.depmask_pipeline is None:
            self.setup_depmask_renderer()

        self.points_3d, self.points_color = self.depmask_pipeline.extract_point_cloud_from_depth(
            images, depth_maps, self.camera_poses, self.intrinsics
        )
        self.points_confidence = torch.ones(
            self.points_3d.shape[0], device=self.device
        )

        print(f"    Extracted {self.points_3d.shape[0]} 3D points")
        print(f"    Camera poses: {self.camera_poses.shape[0]}")

    def render_trajectory(
        self,
        target_poses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stage 2: Render RGB, masked depth and confidence maps using DepMask renderer.

        Args:
            target_poses: (K, 4, 4) target camera poses

        Returns:
            render_rgb: (K, 3, H, W) rendered RGB frames
            masked_depth: (K, 1, H, W) masked depth maps
            confidence: (K, H, W) per-view confidence maps Cm(u)
        """
        print(f">>> Stage 2: DepMask rendering of {target_poses.shape[0]} views...")

        if self.depmask_pipeline is None:
            self.setup_depmask_renderer()

        # Ensure intrinsics match target view count
        if self.intrinsics.shape[0] == 1:
            target_intrinsics = self.intrinsics.expand(target_poses.shape[0], -1, -1)
        elif self.intrinsics.shape[0] < target_poses.shape[0]:
            target_intrinsics = self.intrinsics[0:1].expand(target_poses.shape[0], -1, -1)
        else:
            target_intrinsics = self.intrinsics[:target_poses.shape[0]]

        rendered_views = self.depmask_pipeline.render_trajectory(
            self.points_3d, self.points_color, self.points_confidence,
            target_poses, target_intrinsics,
        )

        render_rgb = torch.stack([v['rgb'] for v in rendered_views])  # (K, H, W, 3)
        masked_depth = torch.stack([v['masked_depth'] for v in rendered_views])  # (K, H, W)
        confidence = torch.stack([v['confidence'] for v in rendered_views])  # (K, H, W)

        # Rearrange to (K, C, H, W)
        render_rgb = render_rgb.permute(0, 3, 1, 2)
        masked_depth = masked_depth.unsqueeze(1)

        return render_rgb, masked_depth, confidence

    def generate_novel_views(
        self,
        render_rgb: torch.Tensor,
        masked_depth: torch.Tensor,
        target_poses: torch.Tensor,
        condition_view: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Stage 3: Generate high-quality novel views using depth-aware video diffusion.

        Args:
            render_rgb: (K, 3, H, W) DepMask-rendered RGB frames
            masked_depth: (K, 1, H, W) DepMask-masked depth maps
            target_poses: (K, 4, 4) target camera poses
            condition_view: (3, H, W) reference condition view

        Returns:
            generated_video: (K, 3, H, W) diffusion-refined video frames
        """
        print(f">>> Stage 3: Depth-video diffusion for novel view generation...")

        if self.diffusion_model is None:
            self.setup_diffusion()

        if condition_view is None and self.images is not None:
            condition_view = self.images

        # Prepare inputs for the diffusion model
        I_render = render_rgb.unsqueeze(0)  # (1, K, 3, H, W)
        D_mask = masked_depth.unsqueeze(0)  # (1, K, 1, H, W)
        I_con = condition_view[None, None] if condition_view.dim() == 3 else condition_view.unsqueeze(0)

        # Convert poses to 6-DoF format
        from modules.camera_encoder import poses_to_6dof
        T_novel = poses_to_6dof(target_poses.unsqueeze(0))  # (1, K, 6)

        # Generate
        self.diffusion_model.eval()
        with torch.no_grad():
            generated = self.diffusion_model.sample(
                I_render=I_render,
                D_mask=D_mask,
                I_con=I_con,
                T_novel=T_novel,
                num_inference_steps=self.config.ddim_steps,
                guidance_scale=self.config.unconditional_guidance_scale,
                frame_stride=self.config.frame_stride,
                eta=self.config.ddim_eta,
            )

        # Squeeze batch dimension
        generated_video = generated.squeeze(0)  # (K, 3, H, W)

        print(f"    Generated {generated_video.shape[0]} frames")

        return generated_video.clamp(0, 1)

    def optimize_gs(
        self,
        generated_frames: torch.Tensor,
        target_poses: torch.Tensor,
        ground_truth_rgb: Optional[torch.Tensor] = None,
        confidence_maps: Optional[torch.Tensor] = None,
    ):
        """
        Stage 4: Region-adaptive 3DGS optimization (paper Section "Region-adaptive Optimization").

        Implements Eqs. 11-13:
          L_novel = confidence-weighted L1 + lambda_lpips * LPIPS
          L_ref   = D-SSIM
          L_ra    = lambda_novel * L_novel + lambda_ref * L_ref

        Args:
            generated_frames: (K, 3, H, W) diffusion-generated novel views Inovel
            target_poses: (K, 4, 4) corresponding novel-view camera poses
            ground_truth_rgb: optional ground truth for validation
            confidence_maps: (K, H, W) DepMask confidence maps Cm(u) for novel views
        """
        print(f">>> Stage 4: Region-adaptive 3DGS optimization...")

        if self.gaussian_model is None:
            self.setup_gs_model()

        gs_config = self.config.gs_opt

        # Initialize Gaussians from sparse point cloud (Eq. 10)
        self.gaussian_model.create_from_point_cloud(
            self.points_3d, self.points_color,
            spatial_lr_scale=1.0,
        )

        # Setup optimizer
        self.gaussian_model.setup_optimizer(
            position_lr_init=gs_config.position_lr_init,
            position_lr_final=gs_config.position_lr_final,
            position_lr_delay_mult=gs_config.position_lr_delay_mult,
            position_lr_max_steps=gs_config.position_lr_max_steps,
            feature_lr=gs_config.feature_lr,
            opacity_lr=gs_config.opacity_lr,
            scaling_lr=gs_config.scaling_lr,
            rotation_lr=gs_config.rotation_lr,
        )

        # Setup LPIPS model for novel-view perceptual loss
        lpips_model = None
        if gs_config.lambda_lpips > 0:
            try:
                import lpips as lpips_lib
                lpips_model = lpips_lib.LPIPS(net='vgg').to(self.device)
                lpips_model.eval()
                print("    LPIPS model loaded for novel-view perceptual loss")
            except ImportError:
                print("    WARNING: lpips not installed, skipping LPIPS loss")
            except Exception as e:
                print(f"    WARNING: Failed to load LPIPS model: {e}")

        H, W = self.config.height, self.config.width

        # --- Build novel-view cameras (with confidence maps) ---
        novel_cameras: List[Camera] = []
        K = min(generated_frames.shape[0], target_poses.shape[0])
        for i in range(K):
            ref_intrinsics = self.intrinsics[0]
            cam = self._create_camera_from_pose(
                target_poses[i], ref_intrinsics,
                H, W,
                image=generated_frames[i] if ground_truth_rgb is None else ground_truth_rgb[i],
            )
            # Attach confidence map to the camera for region-adaptive loss
            if confidence_maps is not None and i < confidence_maps.shape[0]:
                cam._confidence = confidence_maps[i].to(self.device)  # (H, W)
            else:
                cam._confidence = torch.ones(H, W, device=self.device)
            novel_cameras.append(cam)

        # --- Build reference-view cameras (observed input images) ---
        ref_cameras: List[Camera] = []
        if self.camera_poses is not None and self.images is not None:
            for i in range(self.images.shape[0]):
                cam = self._create_camera_from_pose(
                    self.camera_poses[i],
                    self.intrinsics[i] if self.intrinsics.shape[0] > i else self.intrinsics[0],
                    H, W, image=self.images[i],
                )
                ref_cameras.append(cam)

        if len(novel_cameras) == 0:
            raise RuntimeError("No novel-view cameras available for GS optimization")
        if len(ref_cameras) == 0:
            print("    WARNING: No reference-view cameras, using novel-view only")
            ref_cameras = novel_cameras[:1]  # fallback: use first novel view as reference

        print(f"    Novel views: {len(novel_cameras)}, Reference views: {len(ref_cameras)}")

        # Create scene representation (combine all cameras for rendering)
        all_train_cameras = novel_cameras + ref_cameras
        render_cfg = GSRenderConfig(
            sh_degree=gs_config.sh_degree,
            white_background=gs_config.white_background,
        )
        self.scene = SceneRepresentation(
            self.gaussian_model, all_train_cameras, render_config=render_cfg
        )
        self.scene.point_cloud = self.points_3d

        bg_color = torch.zeros(3, device=self.device)
        if gs_config.white_background:
            bg_color = torch.ones(3, device=self.device)

        # Index stacks for random sampling
        novel_indices = list(range(len(novel_cameras)))
        ref_indices = list(range(len(ref_cameras)))

        import random
        from tqdm import tqdm

        progress_bar = tqdm(range(1, gs_config.iterations + 1), desc="GS Optimization")
        metrics = {'L_ra': 0.0, 'novel_l1_weighted': 0.0, 'novel_lpips': 0.0, 'ref_dssim': 0.0}

        for iteration in progress_bar:
            # --- Sample one novel view ---
            if len(novel_indices) == 0:
                novel_indices = list(range(len(novel_cameras)))
            novel_idx = novel_indices.pop(random.randint(0, len(novel_indices) - 1))
            novel_cam = novel_cameras[novel_idx]

            # --- Sample one reference view ---
            if len(ref_indices) == 0:
                ref_indices = list(range(len(ref_cameras)))
            ref_idx = ref_indices.pop(random.randint(0, len(ref_indices) - 1))
            ref_cam = ref_cameras[ref_idx]

            # --- Render both views ---
            novel_render = self.scene.render_view(novel_cam, bg_color)
            ref_render = self.scene.render_view(ref_cam, bg_color)

            novel_image = novel_render['render'].permute(2, 0, 1)  # (C, H, W)
            ref_image = ref_render['render'].permute(2, 0, 1)      # (C, H, W)

            # --- Region-adaptive loss (Eq. 13) ---
            novel_conf = getattr(novel_cam, '_confidence', torch.ones(H, W, device=self.device))
            loss, loss_components = region_adaptive_loss(
                novel_pred=novel_image,
                novel_target=novel_cam.image,
                confidence=novel_conf,
                ref_pred=ref_image,
                ref_target=ref_cam.image,
                lambda_lpips=gs_config.lambda_lpips,
                lambda_novel=gs_config.lambda_novel,
                lambda_ref=gs_config.lambda_ref,
                lpips_model=lpips_model,
            )

            # --- Backward pass ---
            loss.backward()

            # Update metrics
            metrics['L_ra'] = loss_components.get('L_ra', loss.item())
            metrics['novel_l1_weighted'] = loss_components.get('novel_l1_weighted', 0.0)
            metrics['novel_lpips'] = loss_components.get('novel_lpips', 0.0)
            metrics['ref_dssim'] = loss_components.get('ref_dssim', 0.0)

            # --- Densification and pruning ---
            if gs_config.densify_from_iter <= iteration <= gs_config.densify_until_iter:
                if iteration % gs_config.densification_interval == 0:
                    self.gaussian_model.densify_and_prune(
                        max_grad=gs_config.densify_grad_threshold,
                        min_opacity=gs_config.prune_threshold,
                        extent=1.0,
                        iteration=iteration,
                    )

                # Opacity reset
                if iteration % gs_config.opacity_reset_interval == 0:
                    self.gaussian_model.reset_opacity()

            # --- Optimizer step ---
            self.gaussian_model.optimizer.step()
            self.gaussian_model.optimizer.zero_grad(set_to_none=True)

            # --- LR scheduling ---
            self.gaussian_model.update_learning_rate(iteration)

            progress_bar.set_postfix(metrics)

        print(f"    GS optimization complete. Total Gaussians: {self.gaussian_model._xyz.shape[0]}")

    def run_single_view(self):
        """Run single-view novel view synthesis pipeline."""
        image_path = self.config.image_dir
        images = self.load_input_images(image_path)
        print(f"Loaded {images.shape[0]} input images")

        # Stage 1: Sparse reconstruction
        self.reconstruct_sparse(images)

        # Generate camera trajectory
        ref_pose = self.camera_poses[0:1]
        if self.config.traj_txt:
            target_poses = generate_trajectory_from_txt(
                ref_pose, self.config.traj_txt, self.device
            )
        else:
            target_poses = generate_trajectory(
                ref_pose,
                num_views=self.config.video_length,
                elevation=self.config.elevation,
                center_scale=self.config.center_scale,
                d_theta=self.config.d_theta,
                d_phi=self.config.d_phi,
                d_r=self.config.d_r,
                device=self.device,
            )

        # Stage 2: DepMask rendering
        render_rgb, masked_depth, confidence = self.render_trajectory(target_poses)

        # Stage 3: Diffusion generation
        generated_video = self.generate_novel_views(
            render_rgb, masked_depth, target_poses,
            condition_view=self.images[0],
        )

        # Save outputs
        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        save_video(
            generated_video.permute(0, 2, 3, 1).cpu(),
            str(out_dir / "generated_video.mp4"),
        )
        save_video(
            render_rgb.permute(0, 2, 3, 1).cpu(),
            str(out_dir / "depmask_render.mp4"),
        )

        print(f"Output saved to {out_dir}")

        return generated_video

    def run_sparse_view(self):
        """Run sparse multi-view reconstruction pipeline."""
        image_path = self.config.image_dir
        images = self.load_input_images(image_path)
        print(f"Loaded {images.shape[0]} input images")

        # Stage 1: Reconstruction from sparse views
        self.reconstruct_sparse(images)

        # Generate interpolated trajectory between views
        N = self.camera_poses.shape[0]
        all_poses = []
        for i in range(N - 1):
            interp = generate_trajectory_interp(
                self.camera_poses[i:i+1],
                self.camera_poses[i+1:i+2],
                num_views=max(2, self.config.video_length // N),
                device=self.device,
            )
            all_poses.append(interp)
        target_poses = torch.cat(all_poses, dim=0)

        # Stage 2: DepMask rendering
        render_rgb, masked_depth, confidence = self.render_trajectory(target_poses)

        # Stage 3: Diffusion generation
        generated_video = self.generate_novel_views(
            render_rgb, masked_depth, target_poses,
        )

        # Stage 4: GS optimization (optional)
        if self.config.mode == "train":
            self.optimize_gs(generated_video, target_poses, confidence_maps=confidence)

        # Save
        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        save_video(
            generated_video.permute(0, 2, 3, 1).cpu(),
            str(out_dir / "generated_video.mp4"),
        )

        return generated_video

    def run_training(self):
        """Run training pipeline with 3DGS optimization."""
        image_path = self.config.image_dir
        images = self.load_input_images(image_path)

        # Reconstruct
        self.reconstruct_sparse(images)

        # Generate trajectory
        ref_pose = self.camera_poses[0:1]
        target_poses = generate_trajectory(
            ref_pose, num_views=self.config.video_length,
            device=self.device,
        )

        # DepMask rendering
        render_rgb, masked_depth, confidence = self.render_trajectory(target_poses)

        # Generate novel views
        generated_video = self.generate_novel_views(
            render_rgb, masked_depth, target_poses,
        )

        # Optimize 3DGS with region-adaptive loss
        self.optimize_gs(generated_video, target_poses, confidence_maps=confidence)

        # Save Gaussian model
        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.gaussian_model.state_dict(),
            str(out_dir / "gaussian_model.pth"),
        )

    def run(self):
        """Main entry point - dispatches to the appropriate mode."""
        if self.config.mode == "single_view":
            return self.run_single_view()
        elif self.config.mode == "sparse_view":
            return self.run_sparse_view()
        elif self.config.mode == "train":
            return self.run_training()
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

    def _create_camera_from_pose(
        self,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        height: int,
        width: int,
        image: Optional[torch.Tensor] = None,
    ) -> Camera:
        """Create Camera object from pose and intrinsics."""
        device = c2w.device

        w2c = torch.inverse(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]

        fov_x, fov_y = get_fov(intrinsics, (width, height))

        world_view_transform = get_world_to_view_matrix(R, T).T
        projection_matrix = get_projection_matrix(
            0.01, 100.0, fov_x, fov_y
        ).to(device).T
        full_proj_transform = world_view_transform @ projection_matrix

        return Camera(
            image_height=height,
            image_width=width,
            FoVx=fov_x,
            FoVy=fov_y,
            R=R,
            T=T,
            world_view_transform=world_view_transform,
            projection_matrix=projection_matrix,
            full_proj_transform=full_proj_transform,
            camera_center=c2w[:3, 3],
            image=image,
        )

    def _create_pseudo_camera(
        self,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        height: int,
        width: int,
        pseudo_gt: torch.Tensor,
        confidence_mask: Optional[torch.Tensor] = None,
    ) -> PseudoCamera:
        """Create PseudoCamera with diffusion-generated image and optional DepMask confidence mask."""
        cam = self._create_camera_from_pose(c2w, intrinsics, height, width)
        if confidence_mask is None:
            confidence_mask = torch.ones(height, width, device=pseudo_gt.device, dtype=torch.bool)
        return PseudoCamera(
            image_height=height,
            image_width=width,
            FoVx=cam.FoVx,
            FoVy=cam.FoVy,
            R=cam.R,
            T=cam.T,
            world_view_transform=cam.world_view_transform,
            projection_matrix=cam.projection_matrix,
            full_proj_transform=cam.full_proj_transform,
            camera_center=cam.camera_center,
            pseudo_gt=pseudo_gt,
            mask=confidence_mask,
            depth=None,
        )
