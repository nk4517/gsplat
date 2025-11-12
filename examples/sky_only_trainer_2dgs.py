import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import imageio
import tqdm
import tyro
import viser
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from datasets.colmap import Dataset, Parser
from examples.skysphere_model import SkysphereModel
from examples.skysphere_model_parametrized import SkysphereModelParametrized
from gsplat import rasterization_2dgs, RasterizationMode2DGS
from gsplat.strategy.ops import opacity_activation
from gsplat_viewer_2dgs import GsplatViewer
from nerfview import CameraState

# Bilateral grid imports will be added dynamically in main()

@dataclass
class SkyOnlyConfig:
    # Path to dataset
    data_dir: str = r"x:\_ai\_demos\_gsplat\_datasets\segment-102751"
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
    max_steps: int = 10_000
    batch_size: int = 8
    eval_steps: List[int] = field(default_factory=lambda: [5_000, 10_000])
    save_steps: List[int] = field(default_factory=lambda: [5_000, 10_000])
    
    # Skysphere parameters
    skysphere_radius_multiplier: float = 20.0
    skysphere_points: int = 50_000
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


class SkyOnlyRunner:
    """Simplified trainer for sky-only training."""
    
    def __init__(self, cfg: SkyOnlyConfig) -> None:
        self.cfg = cfg
        self.device = "cuda"
        
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
            )
            self.valset = PreloadedDataset(
                self.parser,
                split="val",
                patch_size=None,
                load_depths=False,
                device=self.device,
                to_gpu=False,
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
        
        # Initialize skysphere model
        self.skysphere_model = SkysphereModelParametrized(
            trainset=self.trainset,
            scene_scale=self.scene_scale,
            radius_multiplier=cfg.skysphere_radius_multiplier,
            num_points=cfg.skysphere_points,
            init_opacity=cfg.init_opacity,
            init_scale=cfg.init_scale,
            device=self.device,
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
    
    def rasterize_sky(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """Rasterize sky using WEIGHTED_SUM mode."""
        
        sky_splats = self.skysphere_model.get_splats()
        
        means = sky_splats["means"]  # [N, 3]
        quats = sky_splats["quats"]  # [N, 4]
        scales = sky_splats["scales"]  # [N, 3]
        opacities = sky_splats["opacities"]  # [N,] - already 1.0 constants
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
        )
        
        return render_colors, render_alphas, info
    
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
        data_iter = iter(trainloader)
        
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()
            
            # Get next batch
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(trainloader)
                data = next(data_iter)
            
            camtoworlds = data["camtoworld"].to(device)  # [B, 4, 4]
            Ks = data["K"].to(device)  # [B, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [B, H, W, 3]
            batch_size_actual = pixels.shape[0]
            height, width = pixels.shape[1:3]
            
            # Get sky mask (1 = sky, 0 = world)
            if "sky_mask" not in data:
                raise ValueError("Sky mask not found in data!")
            sky_mask = data["sky_mask"].to(device).float()  # [B, H, W]
            
            # Render sky
            sky_colors, sky_alphas, info = self.rasterize_sky(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
            )
            
            # Apply bilateral grid if enabled
            if cfg.use_bilateral_grid:
                image_ids = data["image_id"].to(device)
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
            
            # Add alpha regularization to encourage sky coverage
            # Penalize low alpha in sky regions
            alpha_loss = F.mse_loss(sky_alphas[..., 0] * sky_mask, sky_mask)
            loss += alpha_loss * 0.1
            
            # Add strong radius regularization to keep gaussians on sphere
            radius_loss = self.skysphere_model.radius_regularization_loss()
            loss += radius_loss * 0.001

            # Add total variation loss for bilateral grid
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # Backward pass
            loss.backward()
            
            # Optimize
            for optimizer in self.skysphere_optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Optimize bilateral grid
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            for scheduler in schedulers:
                scheduler.step()
            
            # Update viewer
            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_rays_per_step = height * width * batch_size_actual
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
                self.viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
                self.viewer.update(step, num_train_rays_per_step)
            
            # Logging
            desc = f"loss={loss.item():.3f} | l1={l1loss.item():.3f} | ssim={ssimloss.item():.4f} | alpha={alpha_loss.item():.4f} | radius={radius_loss.item():.4f}"
            if cfg.use_bilateral_grid:
                desc += f" | tv={tvloss.item():.4f}"
            pbar.set_description(desc)
            
            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/alpha_loss", alpha_loss.item(), step)
                self.writer.add_scalar("train/radius_loss", radius_loss.item(), step)
                self.writer.add_scalar("train/num_sky_GS", self.skysphere_model.n_points, step)
                self.writer.flush()
            
            # Save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                checkpoint_data = {
                    "step": step,
                    "skysphere": self.skysphere_model.state_dict(),
                }
                torch.save(
                    checkpoint_data,
                    f"{self.ckpt_dir}/ckpt_{step}.pt",
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
            
            # Render sky
            sky_colors, sky_alphas, _ = self.rasterize_sky(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
            )
            
            # Composite with black background for world regions
            composite = sky_colors * sky_alphas
            
            # Save rendered image
            canvas = torch.cat([pixels, composite], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_step{step}.png",
                (canvas * 255).astype(np.uint8)
            )
            
            # Compute metrics only on sky regions
            sky_colors_masked = composite * sky_mask.unsqueeze(-1)
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
        width = render_tab_state.viewer_width
        height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)
        
        # Render sky
        sky_colors, sky_alphas, _ = self.rasterize_sky(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
        )
        
        # Composite with black background
        renders = (sky_colors * sky_alphas).squeeze(0).clamp(0, 1)
        
        # Update render tab state
        render_tab_state.total_gs_count = self.skysphere_model.n_points
        render_tab_state.rendered_gs_count = self.skysphere_model.n_points
        
        return renders.cpu().numpy()


def main(cfg: SkyOnlyConfig):
    # Import BilateralGrid and related functions based on configuration
    global BilateralGrid, slice, total_variation_loss, color_correct
    if cfg.use_bilateral_grid:
        if cfg.use_fused_bilagrid:
            from fused_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
        else:
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

    runner = SkyOnlyRunner(cfg)
    runner.train()
    print("Training completed!")
    
    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(SkyOnlyConfig)
    main(cfg)