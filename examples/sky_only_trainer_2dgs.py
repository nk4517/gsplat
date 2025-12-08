import json
import os
import time
import threading
from dataclasses import dataclass, field
from functools import wraps
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from examples.sky_only_trainer_2dgs_viewer import skysphere_renderer
from examples.vs_env import set_vc_envs;

set_vc_envs()

import torch
import torch.nn.functional as F
import numpy as np
import imageio
import tqdm
import tyro
import viser
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from datasets.colmap import Parser
from examples.datasets.dataset import Dataset
from examples.skysphere_model_parametrized import SkysphereModelParametrized
from gsplat import rasterization_2dgs, RasterizationMode2DGS
from gsplat.strategy.epoch_stats import EpochStatistics, training_data_generator
from gsplat import MCMCStrategy
from gsplat.strategy.ops import scaling_activation, opacity_activation
from gsplat_viewer_2dgs import GsplatViewer
from nerfview import CameraState

# Bilateral grid imports will be added dynamically in main()

@dataclass
class SkyOnlyConfig:
    # Path to dataset
    data_dir: str = r"x:\_ai\_demos\_gsplat\_datasets\segment-102751"
    # data_dir: str = r"y:\_gopro_kv92\2025-10-06-1\3-good-park"
    # data_dir: str = r"x:\_ai\_demos\_gsplat\_datasets\youtube01"
    # Directory to save results
    result_dir: str = r"x:\_ai\_my_nerfstudio_results\sky_only"

    invert_sky_mask: bool = True  # Invert sky mask (1=sky, 0=world) vs (0=sky, 1=world)

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
    max_steps: int = 10_000
    batch_size: int = 8
    eval_steps: List[int] = field(default_factory=lambda: [5_000, 10_000])
    save_steps: List[int] = field(default_factory=lambda: [5_000, 10_000])

    # Skysphere parameters
    skysphere_radius_multiplier: float = 20.0
    full_skysphere_N_points: int = 500_000
    init_opacity: float = 0.1
    init_scale: float = 1.0
    trainable_opacities: bool = True

    # 2D Gaussian parameters
    gaussian2d_points: int = 10000
    gaussian2d_lr: float = 0.018
    use_2d_gaussians: bool = False  # Enable/disable 2D gaussians and uncertainty masks

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
    use_color_correct: bool = True

    # SH background parameters

    use_sh_background: bool = True
    sh_background_degree: int = 4
    sh_background_lr: float = 5e-3

    sh_bg_base_weight: float = 0.5  # Base weight for SH background blending
    sh_bg_threshold: float = 15  # Threshold above which SH background is not visible

    wsum_penalty_lambda: float = 5e-3  # Penalty for high wsum values to encourage SH background
    regularize_lambda: float = 5e-3  # Penalty for high wsum values to encourage SH background

    # MCMC strategy parameters
    use_mcmc_strategy: bool = True
    mcmc_cap_max: int = 25_000
    mcmc_noise_lr: float = 5e5
    mcmc_refine_start_epochs: int = 20
    mcmc_refine_stop_epochs: int = 500
    mcmc_refine_every_epochs: int = 10
    mcmc_add_every_epochs: int = 20
    mcmc_min_opacity: float = 0.005
    mcmc_growth_factor: float = 1.05


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
            target_resolution=cfg.target_resolution,
            normalize=True,
            test_every=cfg.test_every,
        )

        # Create datasets
        if cfg.preload_images:
            from datasets.preloaded_dataset import PreloadedDataset
            self.trainset = PreloadedDataset(
                self.parser.scene,
                split="train",
                patch_size=None,
                load_depths=False,
                device=self.device,
                to_gpu=True,
                require_sky_mask=True,
                invert_sky_mask=cfg.invert_sky_mask,
            )
            self.valset = PreloadedDataset(
                self.parser.scene,
                split="val",
                patch_size=None,
                load_depths=False,
                device=self.device,
                to_gpu=False,
                require_sky_mask=True,
                invert_sky_mask=cfg.invert_sky_mask,
            )
        else:
            self.trainset = Dataset(
                self.parser.scene,
                split="train",
                patch_size=None,
                load_depths=False,
            )
            self.valset = Dataset(self.parser.scene, split="val")

        self.scene_scale = self.parser.scene.scene_scale * 1.1
        print(f"Scene scale: {self.scene_scale}")
        print(f"Number of training images: {len(self.trainset)}")
        print(f"Number of validation images: {len(self.valset)}")


        self.skysphere_model = SkysphereModelParametrized(
            self.parser.scene_scale,
            trainable_opacities=cfg.trainable_opacities,
            use_sh_background=cfg.use_sh_background,
            sh_degree=cfg.sh_background_degree,
            sh_bg_base_weight=cfg.sh_bg_base_weight,
            sh_bg_threshold=cfg.sh_bg_threshold,
        )

        # Initialize skysphere from trainset
        self.skysphere_model.initialize_from_trainset(
            trainset=self.trainset,
            full_skysphere_N_points=cfg.full_skysphere_N_points,
            # init_opacity=cfg.init_opacity,
            init_scale=cfg.init_scale,
            # use_dog=False,
        )

        if self.skysphere_model.is_empty:
            raise ValueError("No sky masks found in dataset! Cannot train sky-only model.")

        print(f"Skysphere initialized with {self.skysphere_model.n_points} points")

        # Create optimizers for skysphere
        self.skysphere_optimizers = self.skysphere_model.create_optimizers(
            batch_size=cfg.batch_size,
            sparse_grad=False,
            sh_background_lr=cfg.sh_background_lr,
        )

        # Initialize MCMC strategy if enabled
        self.strategy = None
        self.strategy_state = None
        if cfg.use_mcmc_strategy:
            self.strategy = MCMCStrategy(
                cap_max=cfg.mcmc_cap_max,
                noise_lr=cfg.mcmc_noise_lr,
                refine_start_epochs=cfg.mcmc_refine_start_epochs,
                refine_stop_epochs=cfg.mcmc_refine_stop_epochs,
                refine_every_epochs=cfg.mcmc_refine_every_epochs,
                add_every_epochs=cfg.mcmc_add_every_epochs,
                min_opacity=cfg.mcmc_min_opacity,
                growth_factor=cfg.mcmc_growth_factor,
                verbose=True,
                model_type="quaternion",  # Use quaternion-specific functions
            )
            self.strategy_state = self.strategy.initialize_state(
                scene_scale=self.skysphere_model.radius
            )
            print(f"MCMC strategy initialized with quaternion model type")
        else:
            print("No strategy enabled - manual training mode")

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
                    lr=2e-3,  # * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

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

    @with_lock('rasterize_lock')
    def rasterize_sky(
            self,
            camtoworlds: Tensor,
            Ks: Tensor,
            width: int,
            height: int,
            track_domination: bool = False,
            override_colors: Optional[Tensor] = None,
            **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """Rasterize sky using WEIGHTED_SUM mode.
        
        Returns:
            render_colors: [B, H, W, 3]
            wsum: [B, H, W, 1] - sum of weights (not alpha in WEIGHTED_SUM mode)
            info: dict with rendering info
        """

        sky_splats = self.skysphere_model.get_splats()

        means = sky_splats["means"]  # [N, 3]
        quats = sky_splats["quats"]  # [N, 4]

        scales = scaling_activation(sky_splats["scales"])  # [N, 3]
        opacities = opacity_activation(sky_splats["opacities"])  # [N,]

        drop_rate = kwargs.pop("drop_rate", None)
        if drop_rate is not None:
            opacities = F.dropout(opacities, p=drop_rate, training=True)

        # Use override colors if provided, otherwise use sky colors
        if override_colors is not None:
            colors = override_colors
        else:
            colors = opacity_activation(sky_splats["colors"])  # [N, 3] - RGB

        batch_size = camtoworlds.shape[0]
        # Expand colors for batch processing
        colors_batch = colors.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, 3]

        # Rasterize with WEIGHTED_SUM mode
        (
            render_colors,
            render_wsum,
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

        return render_colors, render_wsum, info

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
                shuffle=False,
                device=device,
            )
        else:
            trainloader = torch.utils.data.DataLoader(
                self.trainset,
                batch_size=cfg.batch_size,
                shuffle=False,
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
                if not cfg.use_mcmc_strategy and (self.epoch_stats is None or len(self.epoch_stats.count) != n_gaussian):
                    self.epoch_stats = EpochStatistics(n_gaussian, device)

                # Call strategy epoch start if enabled
                if self.strategy is not None:
                    self.strategy.step_epoch_start(
                        params=self.skysphere_model.params,
                        optimizers=self.skysphere_optimizers,
                        state=self.strategy_state,
                        step=step,
                        epoch_ctx=epoch_ctx,
                    )

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
            sky_colors, sky_wsum, info = self.rasterize_sky(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                track_domination=True,
                drop_rate=None,
            )

            sky_colors = self.skysphere_model.compose_with_background(
                sky_colors, sky_wsum, camtoworlds, Ks, width, height
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

            # Penalty for exceeding 0-1 per channel
            overflow_penalty = (sky_colors_masked - 1.0).clamp(min=0).mean() + (1.0 - sky_colors_masked).clamp(min=0).mean()
            loss += 0.1 * overflow_penalty

            if cfg.use_sh_background:
                # Wsum penalty - encourage lower wsum to allow SH background
                wsum_penalty = cfg.wsum_penalty_lambda * sky_wsum.mean()

                scales = scaling_activation(self.skysphere_model.params['scales'][..., :2])
                regularize_penality = cfg.regularize_lambda * (scales-scales.mean()).abs().mean()

                # Combined loss
                loss += wsum_penalty
                loss += regularize_penality

            # Add total variation loss for bilateral grid
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # Call strategy pre_backward if enabled
            if self.strategy is not None:
                self.strategy.step_pre_backward(
                    params=self.skysphere_model.params,
                    optimizers=self.skysphere_optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    epoch_ctx=epoch_ctx,
                )

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

                # Update statistics
                self.epoch_stats.update_from_info(
                    info=info,
                    params=self.skysphere_model.params,
                    key_for_gradient="gradient_2dgs",
                    packed=info.get("packed", False)
                )

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

            # Call strategy post_backward if enabled
            if self.strategy is not None:
                # Get current learning rate from first scheduler
                current_lr = schedulers[0].get_last_lr()[0] if schedulers else 1e-3
                self.strategy.step_post_backward(
                    params=self.skysphere_model.params,
                    optimizers=self.skysphere_optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=current_lr,
                    epoch_ctx=epoch_ctx,
                )

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
                if cfg.use_sh_background:
                    self.writer.add_scalar("train/wsum_penalty", wsum_penalty.item(), step)
                self.writer.add_scalar("train/num_sky_GS", self.skysphere_model.n_points, step)
                self.writer.flush()

            # Save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                # Create checkpoint directory for this step
                step_dir = f"{self.ckpt_dir}/step_{step:06d}"
                os.makedirs(step_dir, exist_ok=True)

                # Save skysphere model
                self.skysphere_model.save_checkpoint(f"{step_dir}/skysphere.pt")

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
            sky_colors, sky_wsum, _ = self.rasterize_sky(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
            )

            sky_colors = sky_colors.clamp(0, 1)

            # Compose with SH background if enabled
            sky_colors = self.skysphere_model.compose_with_background(
                sky_colors, sky_wsum, camtoworlds, Ks, width, height
            )

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
        epoch_stats = self.strategy_state["epoch_stats"] if self.strategy_state is not None else self.epoch_stats
        trainset_len = len(self.trainset)
        grow_grad2d = self.cfg.grow_grad2d if hasattr(self.cfg, 'grow_grad2d') else 0.0002
        n_points = self.skysphere_model.n_points
        cfg = self.cfg

        return skysphere_renderer(device, rasterize_fn, epoch_stats, trainset_len, grow_grad2d, n_points, self.skysphere_model, camera_state, render_tab_state)


def main(cfg: SkyOnlyConfig):
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

    runner = SkyOnlyRunner(cfg)
    runner.train()
    print("Training completed!")

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(SkyOnlyConfig)
    main(cfg)
