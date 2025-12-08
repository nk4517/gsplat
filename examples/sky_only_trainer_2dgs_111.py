import json
import math
import os
import time
import threading
from dataclasses import dataclass, field
from functools import wraps
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image

from examples.sky_only_trainer_2dgs_viewer import skysphere_renderer
from examples.vs_env import set_vc_envs; set_vc_envs()

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import imageio
import tqdm
import tyro
import viser
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from datasets.colmap import Dataset, Parser
from examples.skysphere_model_parametrized import SkysphereModelParametrized
from gsplat import rasterization_2dgs, RasterizationMode2DGS
from gsplat.strategy.epoch_stats import EpochStatistics, training_data_generator
from gsplat_viewer_2dgs import GsplatViewer
from nerfview import CameraState

# Bilateral grid imports will be added dynamically in main()

# Import LIG 2D Gaussian model
from gaussian2d_model import Gaussian2D

@dataclass
class SkyOnlyConfig:
    # Path to dataset
    data_dir: str = r"x:\_ai\_demos\_gsplat\_datasets\segment-102751"
    # data_dir: str = r"y:\_gopro_kv92\2025-10-06-1\3-good-park"
    # data_dir: str = r"x:\_ai\_demos\_gsplat\_datasets\youtube01"
    # Directory to save results
    result_dir: str = r"x:\_ai\_my_nerfstudio_results\sky_only"

    # Downsample factor for the dataset
    data_factor: int = 4
    # Every N images there is a test image
    test_every: int = 8
    # Preload all images into memory
    preload_images: bool = True
    # Disable viewer
    disable_viewer: bool = False
    # Port for the viewer server
    port: int = 8080
    
    # Training parameters
    max_steps: int = 2_000
    batch_size: int = 8
    eval_steps: List[int] = field(default_factory=lambda: [5_000, 10_000])
    save_steps: List[int] = field(default_factory=lambda: [5_000, 10_000])
    
    # Skysphere parameters
    skysphere_radius_multiplier: float = 20.0
    skysphere_points: int = 250_000
    init_opacity: float = 0.1
    init_scale: float = 1.0
    
    # 2D Gaussian parameters
    gaussian2d_points: int = 10000
    gaussian2d_lr: float = 0.018
    
    # Loss weights
    ssim_lambda: float = 0.2
    
    # Rendering parameters
    near_plane: float = 0.2
    far_plane: float = 20000000
    
    # Tensorboard
    tb_every: int = 100
    
    # Bilateral grid parameters
    use_bilateral_grid: bool = False
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    use_fused_bilagrid: bool = True
    
    # Color correction for evaluation
    use_color_correct: bool = False


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


class SkyOnlyRunner:
    """Simplified trainer for sky-only training."""
    
    def __init__(self, cfg: SkyOnlyConfig) -> None:
        self.cfg = cfg
        self.device = "cuda"
        self.rasterize_lock = threading.Lock()
        
        # Setup output directories
        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")
        
        # Load data
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=True,
            test_every=cfg.test_every,
        )
        
        # Create datasets
        if cfg.preload_images:
            from datasets.preloaded_dataset import PreloadedDataset
            self.trainset = PreloadedDataset(
                self.parser,
                split="train",
                patch_size=None,
                load_depths=False,
                device=self.device,
                to_gpu=True,
                require_sky_mask=True,
                invert_sky_mask=True,
            )
            self.valset = PreloadedDataset(
                self.parser,
                split="val",
                patch_size=None,
                load_depths=False,
                device=self.device,
                to_gpu=False,
                require_sky_mask=True,
                invert_sky_mask=True,
            )
        else:
            self.trainset = Dataset(
                self.parser,
                split="train",
                patch_size=None,
                load_depths=False,
            )
            self.valset = Dataset(self.parser, split="val")
        
        self.scene_scale = self.parser.scene_scale * 1.1
        print(f"Scene scale: {self.scene_scale}")
        print(f"Number of training images: {len(self.trainset)}")
        print(f"Number of validation images: {len(self.valset)}")


        self.skysphere_model = SkysphereModelParametrized(self.parser.scene_scale)
        
        # Initialize skysphere from trainset
        self.skysphere_model.initialize_from_trainset(
            trainset=self.trainset,
            num_points=cfg.skysphere_points,
            init_opacity=cfg.init_opacity,
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
        
        # Initialize 2D Gaussian model and embeddings
        self.num_cameras = len(self.trainset)
        self.gaussian2d_points = cfg.gaussian2d_points
        
        # Initialize 2D Gaussian models only if enabled
        if cfg.use_2d_gaussians:
            # Get image dimensions from first item
            sample_data = self.trainset[0]
            sample_image = sample_data["image"]
            if isinstance(sample_image, torch.Tensor):
                self.img_height = sample_image.shape[0]
                self.img_width = sample_image.shape[1]
            else:
                self.img_height = sample_image.shape[0]
                self.img_width = sample_image.shape[1]
            
            # Initialize per-camera 2D Gaussian models
            self.gaussian2d_models = {}
            self.gaussian2d_uncertainties = {}  # Will store nn.Parameters
            self.uncertainty_optimizers = {}  # Optimizers for uncertainty masks
            
            for i in range(self.num_cameras):
                # Create model for this camera
                model = Gaussian2D(
                    loss_type="L2",
                    opt_type="adam",
                    num_points=self.gaussian2d_points,
                    H=self.img_height,
                    W=self.img_width,
                    BLOCK_H=16,
                    BLOCK_W=16,
                    device=self.device,
                    lr=cfg.gaussian2d_lr,
                ).to(self.device)
                self.gaussian2d_models[i] = model
                
                # Initialize uncertainty mask logits for this camera
                # 0.0 logit -> 0.5 after sigmoid
                uncertainty_logits = nn.Parameter(torch.zeros(
                    (self.img_height, self.img_width), 
                    device=self.device
                ))  # Start with 0.0 logits (0.5 after sigmoid)
                self.gaussian2d_uncertainties[i] = uncertainty_logits
                
                # Create optimizer for uncertainty mask
                self.uncertainty_optimizers[i] = torch.optim.Adam(
                    [uncertainty_logits],
                    lr=1e-3,  # Learning rate for uncertainty masks
                )
        else:
            self.gaussian2d_models = {}
            self.gaussian2d_uncertainties = {}
            self.uncertainty_optimizers = {}
        
        # Initialize epoch statistics for tracking
        self.epoch_stats = None  # Will be initialized at first epoch
        
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
                output_dir=Path(cfg.result_dir),
                mode="training",
            )
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load checkpoint from directory."""
        # Load skysphere model
        skysphere_path = f"{checkpoint_dir}/skysphere.pt"
        if os.path.exists(skysphere_path):
            checkpoint = torch.load(skysphere_path, map_location=self.device)
            self.skysphere_model.load_state_dict(checkpoint["skysphere"])
            print(f"Loaded skysphere from {skysphere_path}")
        
        # Load each 2D gaussian model and uncertainty mask
        for camera_id in range(self.num_cameras):
            # Load model
            model_path = f"{checkpoint_dir}/gaussian2d_{camera_id:04d}.pt"
            if os.path.exists(model_path):
                self.gaussian2d_models[camera_id].load_checkpoint(model_path)
            
            # Load uncertainty mask from 8-bit grayscale PNG
            uncertainty_path = f"{checkpoint_dir}/uncertainty_{camera_id:04d}.png"
            if os.path.exists(uncertainty_path):
                # Load 8-bit grayscale PNG
                img = Image.open(uncertainty_path)
                uncertainty_8bit = np.array(img, dtype=np.uint8)
                # Convert back to [0, 1] range then to logits
                uncertainty = torch.from_numpy(uncertainty_8bit.astype(np.float32) / 255.0)
                # Convert probability to logits: logit = log(p / (1 - p))
                # Clamp to avoid log(0) or log(inf)
                uncertainty_clamped = torch.clamp(uncertainty, 1e-6, 1 - 1e-6)
                uncertainty_logits = torch.log(uncertainty_clamped / (1 - uncertainty_clamped))
                # Update the parameter data
                self.gaussian2d_uncertainties[camera_id].data = uncertainty_logits.to(self.device)
        
        print(f"Loaded checkpoint from {checkpoint_dir}")
    
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
        scales = sky_splats["scales"]  # [N, 3]
        opacities = sky_splats["opacities"]  # [N,] - already 1.0 constants
        
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
        
        return render_colors, info
    
    def render_2d_gaussians(
        self,
        image_ids: Tensor,
        height: int,
        width: int,
    ) -> Tuple[Tensor, Tensor]:
        """Render 2D gaussians for specific cameras."""
        batch_size = image_ids.shape[0]
        
        # Process each camera in batch
        renders = []
        masks = []
        
        for i in range(batch_size):
            camera_id = image_ids[i].item()
            
            # Get model for this camera
            model = self.gaussian2d_models[camera_id]
            
            # Render
            output = model()
            render = output["render"]  # [1, 3, H, W]
            
            # Get uncertainty mask for this camera (apply sigmoid to logits)
            mask_logits = self.gaussian2d_uncertainties[camera_id]  # [H, W]
            mask = torch.sigmoid(mask_logits)  # Convert logits to probabilities
            
            renders.append(render)
            masks.append(mask)
        
        # Stack results
        renders = torch.cat(renders, dim=0)  # [B, 3, H, W]
        masks = torch.stack(masks, dim=0)  # [B, H, W]
        
        # Convert to [B, H, W, 3] format
        renders = renders.permute(0, 2, 3, 1)  # [B, H, W, 3]
        masks = masks.unsqueeze(-1)  # [B, H, W, 1]
        
        return renders, masks
    
    def train(self):
        cfg = self.cfg
        device = self.device
        
        max_steps = cfg.max_steps
        
        # Create schedulers
        schedulers = []
        for opt_name, optimizer in self.skysphere_optimizers.items():
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.01 ** (1.0 / max_steps)
                )
            )
        
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
        
        # Add schedulers for 2D gaussian optimizers
        # Note: Each Gaussian2D model has its own scheduler already
        # We'll step them individually during training
        
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
            
            # Get sky mask (1 = sky, 0 = world)
            if "sky_mask" not in data:
                raise ValueError("Sky mask not found in data!")
            sky_mask = data["sky_mask"].to(device).float()  # [B, H, W]
            
            # Get image IDs for 2D gaussian rendering
            image_ids = data["image_id"].to(device)  # [B]
            
            # Render 2D gaussians (foreground/world)
            render_2d, uncertainty_mask = self.render_2d_gaussians(
                image_ids=image_ids,
                height=height,
                width=width,
            )
            
            # Render sky
            sky_colors, info = self.rasterize_sky(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                track_domination=False,
            )

            # sky_colors = sky_colors.clamp(0, 1)
            
            # Blend 2D gaussians and sky using uncertainty mask
            # final = r_2d + (1 - uncertainty_mask) * r_sphere
            final_render = render_2d + (1 - uncertainty_mask) * sky_colors
            final_render = final_render.clamp(0, 1)
            
            # Apply bilateral grid if enabled
            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                grid_xy = grid_xy.expand(batch_size_actual, -1, -1, -1)
                final_render = slice(
                    self.bil_grids,
                    grid_xy,
                    final_render,
                    image_ids.unsqueeze(-1),
                )["rgb"]
            
            # Apply sky mask - only compute loss where sky_mask is 1
            # Mask both rendered and ground truth
            final_render_masked = final_render * sky_mask.unsqueeze(-1)
            pixels_masked = pixels * sky_mask.unsqueeze(-1)
            
            # Compute losses only on sky regions
            l1loss = F.l1_loss(final_render_masked, pixels_masked)
            
            # SSIM loss with masking
            # Permute for SSIM computation
            pixels_perm = pixels_masked.permute(0, 3, 1, 2)  # [B, 3, H, W]
            colors_perm = final_render_masked.permute(0, 3, 1, 2)  # [B, 3, H, W]
            ssimloss = 1.0 - self.ssim(colors_perm, pixels_perm)
            
            # Combined loss
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            
            # Add total variation loss for bilateral grid
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss
            
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
            
            # Optimize 2D gaussians if enabled
            if cfg.use_2d_gaussians:
                for i in range(batch_size_actual):
                    camera_id = image_ids[i].item()
                    model = self.gaussian2d_models[camera_id]
                    model.optimizer.step()
                    model.optimizer.zero_grad(set_to_none=True)
                    
                    # Optimize uncertainty mask for this camera
                    self.uncertainty_optimizers[camera_id].step()
                    self.uncertainty_optimizers[camera_id].zero_grad(set_to_none=True)
            
            # Optimize bilateral grid
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            for scheduler in schedulers:
                scheduler.step()
            
            # Reset epoch statistics at the end of epoch
            if epoch_ctx.epoch_end and self.epoch_stats is not None:
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
                torch.save({
                    "step": step,
                    "skysphere": self.skysphere_model.state_dict(),
                    "epoch": epoch_ctx.i_epoch,
                    "use_2d_gaussians": cfg.use_2d_gaussians,
                }, f"{step_dir}/skysphere.pt")
                
                # Save 2D gaussian models if enabled
                if cfg.use_2d_gaussians:
                    for camera_id, model in self.gaussian2d_models.items():
                        # Save model
                        model.save_checkpoint(f"{step_dir}/gaussian2d_{camera_id:04d}.pt")
                        
                        # Save uncertainty mask as 8-bit grayscale PNG (convert logits to probabilities first)
                        uncertainty_logits = self.gaussian2d_uncertainties[camera_id]
                        uncertainty = torch.sigmoid(uncertainty_logits)
                        # Convert to 8-bit range (0-255)
                        uncertainty_8bit = (uncertainty.detach().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(uncertainty_8bit, mode='L').save(
                            f"{step_dir}/uncertainty_{camera_id:04d}.png"
                        )
                
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
            
            # Get image IDs for 2D gaussian rendering
            if "image_id" in data:
                image_ids = data["image_id"].to(device)
                
                # Render 2D gaussians if enabled
                if self.cfg.use_2d_gaussians:
                    # Check if we have models for these validation images
                    camera_id = image_ids[0].item()
                    if camera_id in self.gaussian2d_models:
                        # Render 2D gaussians
                        render_2d, uncertainty_mask = self.render_2d_gaussians(
                            image_ids=image_ids,
                            height=height,
                            width=width,
                        )
                    else:
                        # No model for this validation camera
                        render_2d = torch.zeros_like(pixels)
                        uncertainty_mask = torch.zeros((1, height, width, 1), device=device)
                else:
                    # 2D gaussians disabled
                    render_2d = torch.zeros_like(pixels)
                    uncertainty_mask = torch.zeros((1, height, width, 1), device=device)
            else:
                # If no image_id, skip 2D rendering
                render_2d = torch.zeros_like(pixels)
                uncertainty_mask = torch.zeros((1, height, width, 1), device=device)
            
            # Render sky
            sky_colors, _ = self.rasterize_sky(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
            )

            # Blend 2D and sky
            final_render = render_2d + (1 - uncertainty_mask) * sky_colors
            final_render = final_render.clamp(0, 1)
            
            # Save rendered image
            canvas = torch.cat([pixels, final_render], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_step{step}.png",
                (canvas * 255).astype(np.uint8)
            )
            
            # Compute metrics only on sky regions
            final_render_masked = final_render * sky_mask.unsqueeze(-1)
            pixels_masked = pixels * sky_mask.unsqueeze(-1)
            
            pixels_perm = pixels_masked.permute(0, 3, 1, 2)
            colors_perm = final_render_masked.permute(0, 3, 1, 2)
            
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

        return skysphere_renderer(device, rasterize_fn, epoch_stats, trainset_len, grow_grad2d, n_points, camera_state, render_tab_state)


def main(cfg: SkyOnlyConfig):
    # Import BilateralGrid and related functions based on configuration
    global BilateralGrid, slice, total_variation_loss, color_correct
    if cfg.use_bilateral_grid:
        if cfg.use_fused_bilagrid:
            pass
        else:
            pass
    
    runner = SkyOnlyRunner(cfg)
    runner.train()
    print("Training completed!")
    
    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(SkyOnlyConfig)
    main(cfg)