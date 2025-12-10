import json
import math
import os
import time
import threading
from dataclasses import dataclass, field
from functools import wraps
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from examples.vs_env import set_vc_envs; set_vc_envs()

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras

import torch
import torch.nn.functional as F
import torch.nn as nn
from adan import Adan
import numpy as np
import imageio
import tqdm
import tyro
import viser
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from examples.datasets.normalize_3d import apply_scene_normalization
from examples.sky_only_trainer_2dgs_viewer import skysphere_renderer
from examples.datasets.scene_prepare import prepare_scene
from examples.datasets.waymo import WaymoParser
from examples.datasets.colmap import Parser as ColmapParser
from examples import my_datasets
from examples.datasets.dataset import Dataset
from examples.skysphere_model_parametrized import SkysphereModelParametrized
from gsplat import rasterization_2dgs, RasterizationMode2DGS
from gsplat.strategy.epoch_stats import EpochStatistics, training_data_generator
from gsplat.strategy.ops import scaling_activation, opacity_activation, opacity_inverse_activation
from gsplat_viewer_2dgs import GsplatViewer
from nerfview import CameraState


@dataclass
class SkyOnlySimpleConfig:
    # Dataset configuration
    dataset: my_datasets.DatasetConfig = field(default_factory=lambda: my_datasets.DATASET_SEGMENT_102751)

    # Downsample factor for the dataset
    data_factor: int = 4
    # Target resolution (alternative to factor): int for max side, tuple for (max_w, max_h)
    target_resolution: int | tuple[int, int] | None = None
    # Every N images there is a test image
    test_every: int = 8
    # Preload all images into memory
    preload_images: bool = True
    # Disable viewer
    disable_viewer: bool = False
    # Port for the viewer server
    port: int = 8080
    
    # Training parameters
    max_steps: int = 6_000
    batch_size: int = 24
    eval_steps: List[int] = field(default_factory=lambda: [SkyOnlySimpleConfig.max_steps//3, SkyOnlySimpleConfig.max_steps])
    save_steps: List[int] = field(default_factory=lambda: [SkyOnlySimpleConfig.max_steps//3, SkyOnlySimpleConfig.max_steps])
    
    # Skysphere parameters
    skysphere_radius_multiplier: float = 20.0
    full_skysphere_points: int = 1_000_000
    init_opacity: float = 0.1
    init_scale: float = 1.0
    
    # Loss weights
    ssim_lambda: float = 0.2
    
    # Rendering parameters
    near_plane: float = 0.2
    far_plane: float = 20000000
    
    # Tensorboard
    tb_every: int = 100
    
    # Bilateral grid parameters
    use_bilateral_grid: bool = True
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    use_fused_bilagrid: bool = True
    
    # Color correction for evaluation
    use_color_correct: bool = False

    # Simple relocation parameters
    relocation_start_epoch: int = 10
    relocation_every_epochs: int = 3
    relocation_min_opacity: float = 0.005
    relocation_end_epoch: int | None = 500
    # Learning rate scheduler settings
    lr_scheduler: str = "cosine_warm_restarts"  # "exponential" or "cosine_warm_restarts"
    # For cosine_warm_restarts: T_mult - period multiplier after each restart
    cosine_T_mult: int = 1
    # For cosine_warm_restarts: minimum LR as ratio of initial (eta_min = lr * ratio)
    cosine_eta_min_ratio: float = 0.01
    # Period for cosine restarts (in epochs)
    add_every_epochs: int = 12

    # Camera optimizer configuration (from nerfstudio)
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))

    # Camera intrinsics optimization
    optimize_intrinsics: bool = True
    # Learning rate for intrinsics optimization
    intrinsics_lr: float = 0.1
    # Whether to optimize principal point (cx, cy)
    optimize_principal_point: bool = True
    # Whether to tie fx and fy together (single focal length)
    tie_focal_lengths: bool = False


def with_lock(lock_name):
    """Decorator to make method thread-safe with specified lock."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            lock = getattr(self, lock_name)
            with lock:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


@torch.no_grad()
def simple_relocate_quaternion(
    params: torch.nn.ParameterDict,
    optimizers: dict,
    dead_mask: torch.Tensor,
    min_opacity: float = 0.005,
):
    """Simple relocation for quaternion-parametrized skysphere.
    
    Copies parameters from alive gaussians to dead ones.
    Averages opacities of sampled gaussians.
    
    Args:
        params: ParameterDict with 'quats', 'scales', 'opacities', 'colors'
        optimizers: Dict of optimizers
        dead_mask: Boolean mask indicating dead gaussians (n_touched == 0)
        min_opacity: Minimum opacity threshold
    """
    dead_indices = dead_mask.nonzero(as_tuple=True)[0]
    alive_indices = (~dead_mask).nonzero(as_tuple=True)[0]
    n_dead = len(dead_indices)
    
    if n_dead == 0 or len(alive_indices) == 0:
        return 0
    
    # Sample alive gaussians uniformly
    sample_indices = alive_indices[torch.randint(len(alive_indices), (n_dead,), device=dead_mask.device)]
    
    # Copy parameters from sampled to dead positions
    for name, param in params.items():
        if name == 'opacities':
            # Average opacity: divide by 2 for both source and target
            new_opacity = opacity_activation(param[sample_indices]) / 2.0
            new_opacity = torch.clamp(new_opacity, min=min_opacity, max=1.0 - 1e-5)
            param.data[sample_indices] = opacity_inverse_activation(new_opacity)
            param.data[dead_indices] = param.data[sample_indices]
        else:
            param.data[dead_indices] = param.data[sample_indices]
        
        # Reset optimizer state for affected indices
        if name in optimizers:
            for state in optimizers[name].state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.shape[0] == param.shape[0]:
                        v[sample_indices] = 0
    
    return n_dead


class SkyOnlySimpleRunner:
    """Simplified trainer for sky-only training without 2D gaussians."""
    
    def __init__(self, cfg: SkyOnlySimpleConfig) -> None:
        self.cfg = cfg
        self.device = "cuda"
        self.rasterize_lock = threading.Lock()

        # Default result_dir if not specified
        if cfg.dataset.output_dir is None:
            cfg.dataset.output_dir = str(Path(cfg.dataset.dataset_dir).with_suffix(".result") / "sky-only")

        # Setup output directories
        os.makedirs(cfg.dataset.output_dir, exist_ok=True)
        self.ckpt_dir = f"{cfg.dataset.output_dir}/skysphere/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.dataset.output_dir}/skysphere/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.dataset.output_dir}/skysphere/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.dataset.output_dir}/skysphere/tb")

        # Load data based on dataset type
        if isinstance(cfg.dataset, my_datasets.WaymoDatasetConfig):
            result_dir = Path(cfg.dataset.output_dir)
            self.parser = WaymoParser(
                data_dir=cfg.dataset.dataset_dir,
                camera_angles=cfg.dataset.waymo_camera_angles,
                frame_range=cfg.dataset.waymo_frame_range,
                load_lidar=cfg.dataset.waymo_load_lidar,
                output_dir=result_dir,
                waymo_calib_dir=cfg.dataset.waymo_calib_dir,
            )
        elif isinstance(cfg.dataset, my_datasets.ColmapDatasetConfig):
            result_dir = Path(cfg.dataset.output_dir)
            self.parser = ColmapParser(
                data_dir=cfg.dataset.dataset_dir,
                output_dir=result_dir,
            )
        else:
            raise ValueError(f"Unknown dataset type: {type(cfg.dataset)}")

        scene_fullscale = self.parser.scene

        if cfg.dataset.normalize_world_space:
            apply_scene_normalization(scene_fullscale)

        scene = prepare_scene(
            scene_fullscale,
            factor=cfg.data_factor,
            target_resolution=cfg.target_resolution,
        )
        
        # Create datasets
        if cfg.preload_images:
            from datasets.preloaded_dataset import PreloadedDataset
            self.trainset = PreloadedDataset(
                scene,
                split="train",
                patch_size=None,
                load_depths=False,
                device=self.device,
                to_gpu=True,
                require_sky_mask=True,
                invert_sky_mask=cfg.dataset.invert_mask,
                soft_sky_mask=cfg.dataset.soft_mask,
                load_aux_keys=["mask", "sky_heat"]
            )
            self.valset = PreloadedDataset(
                scene,
                split="val",
                patch_size=None,
                load_depths=False,
                device=self.device,
                to_gpu=False,
                require_sky_mask=True,
                invert_sky_mask=cfg.dataset.invert_mask,
                soft_sky_mask=cfg.dataset.soft_mask,
            )
        else:
            self.trainset = Dataset(
                scene,
                split="train",
                patch_size=None,
                load_depths=False,
                require_sky_mask=True,
                invert_sky_mask=cfg.dataset.invert_mask,
                soft_sky_mask=cfg.dataset.soft_mask,
                load_aux_keys=["mask", "sky_heat"]
            )
            self.valset = Dataset(scene, split="val")
        
        self.scene_scale = scene.scene_scale * 1.1
        print(f"Scene scale: {self.scene_scale}")
        print(f"Number of training images: {len(self.trainset)}")
        print(f"Number of validation images: {len(self.valset)}")

        self.skysphere_model = SkysphereModelParametrized(self.scene_scale, trainable_opacities=True)
        
        # Initialize skysphere from trainset
        self.skysphere_model.initialize_from_trainset(
            trainset=self.trainset,
            full_skysphere_N_points=cfg.full_skysphere_points,
            # init_opacity=cfg.init_opacity,
            init_scale=cfg.init_scale,
        )
        
        if self.skysphere_model.is_empty:
            raise ValueError("No sky masks found in dataset! Cannot train sky-only model.")
        
        print(f"Skysphere initialized with {self.skysphere_model.n_points} points")
        
        # Create optimizers for skysphere
        self.skysphere_optimizers = self.skysphere_model.create_optimizers(
            batch_size=cfg.batch_size,
            sparse_grad=False,
        )
        
        # Initialize bilateral grid if enabled
        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3,# * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]
        
        # Initialize epoch statistics for tracking
        self.epoch_stats = None  # Will be initialized at first epoch
        
        # Initialize camera optimizer from nerfstudio
        self.camera_optimizer: CameraOptimizer = cfg.camera_optimizer.setup(
            num_cameras=len(self.trainset), device=self.device
        )
        self.pose_optimizers = []  # Will be populated in train() if camera optimization is enabled

        # Initialize intrinsics optimization if enabled
        self.intrinsics_optimizers = []
        self.optimized_Ks = None
        if cfg.optimize_intrinsics:
            # Create optimizable K matrices for each camera
            Ks_list = []
            for i in range(len(self.trainset)):
                # Get original K matrix for this camera
                cam_data = self.trainset[i]
                K_orig = cam_data["K"].clone()

                # Create optimizable parameters
                if cfg.tie_focal_lengths:
                    # Single focal length parameter
                    focal = (K_orig[0, 0] + K_orig[1, 1]) / 2.0
                    focal_param = torch.nn.Parameter(torch.tensor([focal], device=self.device))
                    if cfg.optimize_principal_point:
                        principal_point = torch.nn.Parameter(K_orig[0:1, 2:3].clone().to(self.device))
                        cy_param = torch.nn.Parameter(K_orig[1:2, 2:3].clone().to(self.device))
                        Ks_list.append((focal_param, focal_param, principal_point, cy_param))
                    else:
                        Ks_list.append((focal_param, focal_param, None, None))
                else:
                    # Separate fx, fy parameters
                    fx_param = torch.nn.Parameter(torch.tensor([K_orig[0, 0]], device=self.device))
                    fy_param = torch.nn.Parameter(torch.tensor([K_orig[1, 1]], device=self.device))
                    if cfg.optimize_principal_point:
                        cx_param = torch.nn.Parameter(torch.tensor([K_orig[0, 2]], device=self.device))
                        cy_param = torch.nn.Parameter(torch.tensor([K_orig[1, 2]], device=self.device))
                        Ks_list.append((fx_param, fy_param, cx_param, cy_param))
                    else:
                        Ks_list.append((fx_param, fy_param, None, None))

            self.optimized_Ks = torch.nn.ParameterList([
                param for params_tuple in Ks_list
                for param in params_tuple if param is not None
            ])
            self.Ks_structure = Ks_list  # Store structure for reconstruction

        # Metrics
        from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        
        # Viewer
        if not cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.dataset.output_dir),
                mode="training",
            )
    
    @with_lock('rasterize_lock')
    def rasterize_sky(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        track_domination: bool = False,
        override_colors: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict]:
        """Rasterize sky using WEIGHTED_SUM mode."""
        
        sky_splats = self.skysphere_model.get_splats()
        
        means = sky_splats["means"]  # [N, 3]
        quats = sky_splats["quats"]  # [N, 4]
        scales = scaling_activation(sky_splats["scales"])  # [N, 3]
        opacities = opacity_activation(sky_splats["opacities"])  # [N,] - already 1.0 constants
        
        # Use override colors if provided, otherwise use sky colors
        if override_colors is not None:
            colors = override_colors
        else:
            colors = sky_splats["colors"]  # [N, 3] - direct RGB values
        
        batch_size = camtoworlds.shape[0]
        # Expand colors for batch processing
        colors_batch = colors.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, 3]
        
        # Rasterize with WEIGHTED_SUM mode
        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors_batch,  # [B, N, 3]
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            render_mode="RGB",
            rasterization_mode=RasterizationMode2DGS.WEIGHTED_SUM,  # Use WEIGHTED_SUM
            packed=False,
            sparse_grad=False,
            track_domination=track_domination,
        )
        
        return render_colors, render_alphas, info
    
    def train(self):
        cfg = self.cfg
        device = self.device
        
        max_steps = cfg.max_steps
        
        # Create schedulers
        n_cameras_per_epoch = len(self.trainset)
        T_0 = cfg.add_every_epochs * n_cameras_per_epoch

        schedulers = []
        for opt_name, optimizer in self.skysphere_optimizers.items():
            lr = optimizer.param_groups[0]["lr"]
            # Apply cosine scheduler to scales and quats (quats control position on sphere)
            if cfg.lr_scheduler == "cosine_warm_restarts" and opt_name in ("scales", "quats"):
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=T_0, T_mult=cfg.cosine_T_mult, eta_min=lr * cfg.cosine_eta_min_ratio,
                )
            else:
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.01 ** (1.0 / max_steps)
                )
            schedulers.append(scheduler)
        
        # Add schedulers for bilateral grid if enabled
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )
        
        # Get camera optimizer's optimizers and add schedulers for them
        camera_optimizers = {}
        self.camera_optimizer.get_param_groups(camera_optimizers)
        for opt_name, opt_params in camera_optimizers.items():
            if opt_params:  # Only if there are parameters to optimize
                optimizer = Adan(
                    opt_params,
                    lr=1e-5 * math.sqrt(cfg.batch_size),  # Default lr, can be adjusted
                    weight_decay=1e-6,
                    betas=(0.98/8, 0.92/8, 0.99/8),
                )
                schedulers.append(
                    torch.optim.lr_scheduler.ExponentialLR(
                        optimizer, gamma=0.01 ** (1.0 / max_steps)
                    )
                )
                # Store optimizer for later use
                if not hasattr(self, 'pose_optimizers'):
                    self.pose_optimizers = []
                self.pose_optimizers.append(optimizer)

        # Create optimizer for intrinsics if enabled
        if cfg.optimize_intrinsics and self.optimized_Ks is not None:
            intrinsics_optimizer = Adan(
                self.optimized_Ks.parameters(),
                lr=cfg.intrinsics_lr * math.sqrt(cfg.batch_size),
                eps=1e-15 / math.sqrt(cfg.batch_size),
                betas=(0.98/4, 0.92/4, 0.99/4),
            )
            self.intrinsics_optimizers = [intrinsics_optimizer]
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    intrinsics_optimizer, gamma=0.01 ** (1.0 / max_steps)
                )
            )

        # Create dataloader
        if cfg.preload_images:
            from datasets.preloaded_dataset import PreloadedDataLoader
            trainloader = PreloadedDataLoader(
                self.trainset,
                batch_size=cfg.batch_size,
                shuffle=True,
                device=device,
            )
        else:
            trainloader = torch.utils.data.DataLoader(
                self.trainset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
            )

        # Training loop
        global_tic = time.time()
        pbar = tqdm.tqdm(range(max_steps))

        n_cameras = len(trainloader)
        
        for step, data, epoch_ctx in training_data_generator(trainloader, pbar):
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()
            
            # Initialize epoch statistics at the start of each epoch
            if epoch_ctx.epoch_start:
                n_gaussian = self.skysphere_model.n_points
                if self.epoch_stats is None or len(self.epoch_stats.count) != n_gaussian:
                    self.epoch_stats = EpochStatistics(n_gaussian, device)
            
            camtoworlds = data["camtoworld"].to(device)  # [B, 4, 4]
            Ks = data["K"].to(device)  # [B, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [B, H, W, 3]
            batch_size_actual = pixels.shape[0]
            height, width = pixels.shape[1:3]
            
            # Get image IDs for camera optimization
            image_ids = data["image_id"].to(device)  # [B]

            # Use optimized intrinsics if enabled
            if cfg.optimize_intrinsics and self.optimized_Ks is not None:
                # Reconstruct K matrices from optimized parameters for each camera in batch
                batch_size_actual = image_ids.shape[0]
                Ks_opt = torch.zeros(batch_size_actual, 3, 3, device=device)

                for b_idx in range(batch_size_actual):
                    cam_idx = image_ids[b_idx].item()
                    fx, fy, cx, cy = self.Ks_structure[cam_idx]

                    Ks_opt[b_idx, 0, 0] = fx if fx is not None else Ks[b_idx, 0, 0]
                    Ks_opt[b_idx, 1, 1] = fy if fy is not None else Ks[b_idx, 1, 1]
                    Ks_opt[b_idx, 0, 2] = cx if cx is not None else Ks[b_idx, 0, 2]
                    Ks_opt[b_idx, 1, 2] = cy if cy is not None else Ks[b_idx, 1, 2]
                    Ks_opt[b_idx, 2, 2] = 1.0

                Ks = Ks_opt  # [B, 3, 3]

            # Apply camera optimization
            if self.camera_optimizer.config.mode != "off":
                # Create a Cameras object for nerfstudio camera optimizer
                camera = Cameras(
                    camera_to_worlds=camtoworlds,
                    fx=Ks[:, 0, 0],
                    fy=Ks[:, 1, 1],
                    cx=Ks[:, 0, 2],
                    cy=Ks[:, 1, 2],
                    width=torch.tensor([width], device=device),
                    height=torch.tensor([height], device=device),
                    metadata={"cam_idx": image_ids[0].item()} if image_ids.numel() == 1 else None,
                )
                optimized_c2w = self.camera_optimizer.apply_to_camera(camera)
                # apply_to_camera returns [B, 3, 4], need to convert to [B, 4, 4]
                # оно ебанутое. если батч больше одного, оно внутри само применяет, а если 1 - возвращает 3х4
                if optimized_c2w.shape[0] == 1:
                    # Add the homogeneous row [0, 0, 0, 1]
                    bottom_row = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=device)
                    camtoworlds = torch.cat([optimized_c2w, bottom_row], dim=1)
                else:
                    camtoworlds = optimized_c2w

            # Get sky mask (1 = sky, 0 = world)
            if data.get("sky_mask") is None:
                raise ValueError("Sky mask not found in data!")
            sky_mask = data["sky_mask"].to(device).float()  # [B, H, W]
            sky_heat = data["sky_heat"].to(device).float()  # [B, H, W]

            # Render sky
            sky_colors, sky_wsum, info = self.rasterize_sky(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                track_domination=True,
            )

            # Penalty for exceeding 0-1 per channel
            overflow_penalty = (sky_colors - 1.0).clamp(min=0).mean() + (-sky_colors).clamp(min=0).mean()
            
            # Apply bilateral grid if enabled
            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                grid_xy = grid_xy.expand(batch_size_actual, -1, -1, -1)
                sky_colors = slice(
                    self.bil_grids,
                    grid_xy,
                    sky_colors,
                    image_ids.unsqueeze(-1),
                )["rgb"]
            
            # Apply sky mask - only compute loss where sky_mask is 1
            # Mask both rendered and ground truth
            sky_colors_masked = sky_colors * sky_mask.unsqueeze(-1)
            pixels_masked = pixels * sky_mask.unsqueeze(-1)
            
            # Compute losses only on sky regions
            l1loss = F.l1_loss(sky_colors_masked, pixels_masked)


            # SSIM loss with masking
            # Permute for SSIM computation
            pixels_perm = pixels_masked.permute(0, 3, 1, 2)  # [B, 3, H, W]
            colors_perm = sky_colors_masked.permute(0, 3, 1, 2)  # [B, 3, H, W]
            ssimloss = 1.0 - self.ssim(colors_perm, pixels_perm)
            
            # Combined loss
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            loss += 0.5 * overflow_penalty

            outmasked_penality = (sky_wsum * (1-sky_heat.unsqueeze(-1))).clamp(0).mean()
            loss += 0.1 * outmasked_penality

            # Add total variation loss for bilateral grid
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # Add loss from camera optimizer
            loss_dict = {}
            self.camera_optimizer.get_loss_dict(loss_dict)
            for loss_name, loss_value in loss_dict.items():
                loss += loss_value

            # Backward pass
            loss.backward()

            # Update epoch statistics with rendering info
            if self.epoch_stats is not None:
                # Add width and height to info for statistics update
                info["width"] = width
                info["height"] = height
                info["camtoworlds"] = camtoworlds
                info["Ks"] = Ks
                info["n_cameras"] = n_cameras

                # Create dummy params dict with skysphere parameters
                # sky_params = self.skysphere_model.get_splats()
                sky_params = None

                # Update statistics
                self.epoch_stats.update_from_info(
                    info=info,
                    params=sky_params,
                    key_for_gradient="gradient_2dgs",
                    packed=info.get("packed", False)
                )

            # Optimize
            for optimizer in self.skysphere_optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Optimize pose parameters
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Optimize bilateral grid
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Optimize intrinsics if enabled
            for optimizer in self.intrinsics_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            for scheduler in schedulers:
                scheduler.step()
            
            # Reset epoch statistics at the end of epoch
            if epoch_ctx.epoch_end and self.epoch_stats is not None:
                # Simple relocation of dead splats (n_touched == 0)
                should_relocate = (
                    epoch_ctx.i_epoch >= cfg.relocation_start_epoch and
                    epoch_ctx.i_epoch % cfg.relocation_every_epochs == 0
                    and (cfg.relocation_end_epoch is None or
                         epoch_ctx.i_epoch <= cfg.relocation_end_epoch)
                )
                if should_relocate:
                    # Consider splats dead if they were never touched OR have zero importance (no gradients)
                    dead_mask = (self.epoch_stats.n_touched_accum == 0) #| (self.epoch_stats.importance  == 0)
                    # if self.skysphere_model.trainable_opacities:
                    #     dead_mask |= (opacity_activation(self.skysphere_model.params["opacities"]) < cfg.relocation_min_opacity*4)
                    #
                    # # Mark smallest 0.5% by scale as dead
                    # scales_activated = scaling_activation(self.skysphere_model.params["scales"]).mean(dim=-1)
                    # scale_threshold = torch.quantile(scales_activated, 0.005)
                    # dead_mask |= (scales_activated < scale_threshold)

                    n_relocated = simple_relocate_quaternion(
                        params=self.skysphere_model.params,
                        optimizers=self.skysphere_optimizers,
                        dead_mask=dead_mask,
                        min_opacity=cfg.relocation_min_opacity,
                    )
                    print(f"Epoch {epoch_ctx.i_epoch}: Relocated {n_relocated} dead splats")

                self.epoch_stats.reset()
            
            # Update viewer
            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_rays_per_step = height * width * batch_size_actual
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
                self.viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
                self.viewer.update(step, num_train_rays_per_step)
            
            # Logging
            desc = f"epoch={epoch_ctx.i_epoch} ({100.0 * (epoch_ctx.i + 1) / epoch_ctx.epoch_len:.1f}%) | "
            desc += f"loss={loss.item():.3f} | l1={l1loss.item():.3f} | ssim={ssimloss.item():.4f}"
            if cfg.use_bilateral_grid:
                desc += f" | tv={tvloss.item():.4f}"
            pbar.set_description(desc)
            
            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_sky_GS", self.skysphere_model.n_points, step)
                self.writer.flush()
            
            # Save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                # Create checkpoint directory for this step
                step_dir = f"{self.ckpt_dir}/step_{step:06d}"
                os.makedirs(step_dir, exist_ok=True)

                # Save skysphere model
                self.skysphere_model.save_checkpoint(f"{step_dir}/skysphere.pt")

                # torch.save({
                #     "step": step,
                #     "skysphere": self.skysphere_model.state_dict(),
                #     "epoch": epoch_ctx.i_epoch,
                # }, f"{step_dir}/skysphere.pt")
                
                print(f"Saved checkpoint at step {step}")
            
            # Evaluation
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step)
    
    @torch.no_grad()
    def eval(self, step: int):
        """Evaluate on validation set."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        
        # Create validation dataloader
        if cfg.preload_images:
            from datasets.preloaded_dataset import PreloadedDataLoader
            valloader = PreloadedDataLoader(
                self.valset,
                batch_size=1,
                shuffle=False,
                device=device,
            )
        else:
            valloader = torch.utils.data.DataLoader(
                self.valset, batch_size=1, shuffle=False, num_workers=1
            )
        
        metrics = {"psnr": [], "ssim": []}
        
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            
            # Get sky mask
            if "sky_mask" in data:
                sky_mask = data["sky_mask"].to(device).float()
            else:
                # If no sky mask in validation, use full image
                sky_mask = torch.ones((1, height, width), device=device)
            
            # Render sky
            sky_colors, _, _ = self.rasterize_sky(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
            )
            
            # sky_colors = sky_colors.clamp(0, 1)
            
            # Save rendered image
            canvas = torch.cat([pixels, sky_colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_step{step}.png",
                (canvas * 255).astype(np.uint8)
            )
            
            # Compute metrics only on sky regions
            sky_colors_masked = sky_colors * sky_mask.unsqueeze(-1)
            pixels_masked = pixels * sky_mask.unsqueeze(-1)
            
            pixels_perm = pixels_masked.permute(0, 3, 1, 2)
            colors_perm = sky_colors_masked.permute(0, 3, 1, 2)
            
            metrics["psnr"].append(self.psnr(colors_perm, pixels_perm))
            metrics["ssim"].append(self.ssim(colors_perm, pixels_perm))
        
        # Aggregate metrics
        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "num_sky_GS": self.skysphere_model.n_points,
        }
        
        print(f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}")
        
        # Save stats
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        
        # Log to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()
    
    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state
    ):
        """Render function for viewer."""
                
        # Prepare all needed parameters for external function
        device = self.device
        rasterize_fn = self.rasterize_sky
        epoch_stats = self.epoch_stats
        trainset_len = len(self.trainset)
        grow_grad2d = self.cfg.grow_grad2d if hasattr(self.cfg, 'grow_grad2d') else 0.0002
        n_points = self.skysphere_model.n_points

        return skysphere_renderer(device, rasterize_fn, epoch_stats, trainset_len, grow_grad2d, n_points, self.skysphere_model, camera_state, render_tab_state)


def main(cfg: SkyOnlySimpleConfig):
    global BilateralGrid, slice, total_variation_loss, color_correct

    # Import BilateralGrid and related functions based on configuration
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
    
    runner = SkyOnlySimpleRunner(cfg)
    runner.train()
    print("Training completed!")
    
    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(SkyOnlySimpleConfig)
    main(cfg)