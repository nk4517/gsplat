import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import Literal, assert_never
from pathlib import Path

from examples.vs_env import set_vc_envs; set_vc_envs()


print("import 111")


# импортировать всё, что связано c torch только после этого

from examples.lib_compose import CompositingOrder, compose_renders
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras

from examples.simple_trainer_2dgs_viewer import render_inner

import imageio
import numpy as np
import torch
from adan import Adan

# GPU поддерживает TensorFloat32 (TF32) tensor cores для ускорения матричных умножений с float32, но PyTorch не использует их по умолчанию.
torch.set_float32_matmul_precision('high')
# torch.autograd.set_detect_anomaly(True)

import torch.nn.functional as F
import tqdm
import tyro
import viser
# from tiny_renderer.minigui import CUDARenderer
from datasets.colmap import Dataset, Parser
from datasets.traj import generate_interpolated_path
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from examples.utils import normalize_robust
from gsplat import MCMCStrategy
from gsplat.strategy.ops import scaling_inverse_activation, opacity_inverse_activation, scaling_activation, opacity_activation
from gsplat.antialias_2dgs import apply_flat_smoothing, update_max_sampling_rate

from examples.utils import (
    AppearanceOptModule,
    knn,
    rgb_to_sh,
    set_random_seed,
)
from gsplat_viewer_2dgs import GsplatViewer, GsplatRenderTabState
from gsplat import rasterization_2dgs
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.epoch_stats import training_data_generator
from nerfview import CameraState, RenderTabState, apply_float_colormap

@torch.jit.script
def binary_cross_entropy_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * torch.log(input) + (1 - target) * torch.log(1 - input)).mean()

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    use_viser: bool = True
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None

    # Path to the Mip-NeRF 360 dataset
    # data_dir: str = r"x:\_ai\_demos\_gsplat\_datasets\youtube05\towel"
    # data_dir: str = r"X:\_ai\_gsplat\datasets\garden"
    # data_dir: str = r"x:\_ai\_gsplat\datasets\bicycle"
    # data_dir: str = r"X:\_ai\_gsplat\datasets\kitchen"
    # data_dir: str = r"x:\_ai\_gsplat\datasets\fb_colmap_res"
    # data_dir: str = r"y:\_gopro_kv92\extracted_keyframes\GOPR6996_colmap"
    # data_dir: str = r"x:\_ai\_demos\_gsplat\_datasets\segment-102751"
    data_dir: str = r"x:\_ai\_demos\_gsplat\_datasets\youtube01"
    # data_dir: str = r"x:\_ai\_glomap\data\south-building"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    # result_dir: str = r"x:\_ai\_my_nerfstudio_results\koneva1"
    result_dir: str = r"x:\_ai\_my_nerfstudio_results\bicycle"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Preload all images into memory for faster training
    preload_images: bool = True

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [500, 7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm" # "random" # "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 0
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.2
    # Far plane clipping distance
    far_plane: float = 20000000

    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.05
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1

    # Start refining GSs after this epoch
    refine_start_epochs: int = 0
    # Stop refining GSs after this epoch
    refine_stop_epochs: int = 500
    # Refine GSs every this many epochs
    refine_every_epochs: int = 1
    # Add new GSs every this many epochs (for MCMCStrategy)
    add_every_epochs: int = 2
    # Start resetting opacities after this epoch
    reset_start_epochs: int = 100
    # Stop resetting opacities after this epoch
    reset_end_epochs: int = 10000
    # Reset opacities every this many epochs
    reset_every_epochs: int = 20
    # Pause refining for this many epochs after reset
    pause_refine_after_reset_epochs: int = 1

    # Auto-calculate epoch parameters from legacy step-based values
    auto_epoch_params: bool = False
    # Legacy step-based values for auto-calculation
    legacy_refine_start_iter: int = 1
    legacy_refine_stop_iter: int = 25_000
    legacy_reset_every_iter: int = 3000
    legacy_refine_every_iter: int = 100

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    # Whether to use revised opacity heuristic from arXiv:2404.06109 (experimental)
    revised_opacity: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = True

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

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = True
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = True
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Enable normal consistency loss. (Currently for 2DGS only)
    normal_loss: bool = True
    # Weight for normal loss
    normal_lambda: float = 5e-2
    # Iteration to start normal consistency regulerization
    normal_start_iter: int = 7_000

    # Distortion loss. (experimental)
    dist_loss: bool = True
    # Weight for distortion loss
    dist_lambda: float = 1e-2
    # Iteration to start distortion loss regulerization
    dist_start_iter: int = 3_000

    # Opacity entropy regularization (penalizes partial transparency)
    opacity_entropy_loss: bool = True
    # Weight for opacity entropy loss
    opacity_entropy_lambda: float = 1e-3
    # Iteration to start opacity entropy regularization
    opacity_entropy_start_iter: int = 1_000

    # Elongation regularization (penalizes stretched gaussians)
    elongation_loss: bool = False
    # Weight for elongation loss
    elongation_lambda: float = 1e-2
    # Threshold for elongation ratio (penalty starts after this ratio)
    elongation_threshold: float = 4.0
    # Iteration to start elongation regularization
    elongation_start_iter: int = 0

    # L1 opacity regularization (penalizes high opacity)
    opacity_l1_loss: bool = False
    # Weight for L1 opacity loss
    opacity_l1_lambda: float = 1e-5
    # Epoch to start L1 opacity regularization
    opacity_l1_start_epoch: int = 3

    # Scale percentile regularization (penalizes extreme sizes)
    scale_percentile_loss: bool = False
    # Weight for scale percentile loss
    scale_percentile_lambda: float = 1e-5
    # Lower percentile threshold (penalize scales below this)
    scale_percentile_lower: float = 0.05
    # Upper percentile threshold (penalize scales above this)
    scale_percentile_upper: float = 99.0
    # Iteration to start scale percentile regularization
    scale_percentile_start_iter: int = 1_000

    # Effective rank regularization (penalizes low-rank gaussians). from arXiv:2406.11672
    erank_loss: bool = False
    # Weight for effective rank loss
    erank_lambda: float = 1e-3
    # Iteration to start effective rank regularization
    erank_start_iter: int = 0

    # Model for splatting.
    model_type: Literal["2dgs", "2dgs-inria"] = "2dgs"

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = True

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=lambda: MCMCStrategy()
    )

    # AA-2DGS parameters
    use_aa_smoothing: bool = False  # Enable AA-2DGS smoothing
    aa_smoothing_reg: float = 0.1  # s_reg parameter from paper
    aa_compute_every: int = 1  # Recompute compute_min_depth_normalized_sq every N epochs

    # Importance-based pruning parameters (Speedy-Splat style)
    importance_prune_enabled: bool = False  # Enable importance-based pruning
    importance_prune_start_epoch: int = 8  # Start importance pruning after this epoch
    importance_prune_end_epoch: int = 10000  # Stop importance pruning after this epoch
    importance_prune_every_epochs: int = 1  # Perform importance pruning every this many epochs
    importance_prune_ratio: float = 0.005  # Fraction to prune (0.3 = remove 30% least important)

    # Split parameters for gaussians that dominate or touch too many pixels
    split_big_dominated_pct: float = 0.001  # Split gaussians dominating more than this percentage of pixels in one view
    split_big_touched_pct: float = 0.0025  # Split gaussians touching more than this percentage of pixels in one view

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        # self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
    
    def align_steps_to_epochs(self, n_cameras_per_epoch: int):
        """Align eval and save steps to epoch boundaries."""
        def align_to_epoch_end(step: int) -> int:
            """Round step to the nearest epoch end."""
            epoch = math.ceil(step / n_cameras_per_epoch)
            return epoch * n_cameras_per_epoch - 1  # -1 because steps are 0-indexed
        
        self.eval_steps = [align_to_epoch_end(step) for step in self.eval_steps]
        self.save_steps = [align_to_epoch_end(step) for step in self.save_steps]
        # self.ply_steps = [align_to_epoch_end(step) for step in self.ply_steps]
        self.max_steps = align_to_epoch_end(self.max_steps)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    N = points.shape[0]
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = scaling_inverse_activation(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = opacity_inverse_activation(torch.full((N,), init_opacity))  # [N,]

    # Initialize max_sampling_rate as zero (will be computed later if AA is enabled)
    max_sampling_rate = torch.zeros((N,))  # [N,] - maximum sampling rate for AA-2DGS

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 5e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
        # max_sampling_rate doesn't need gradients or optimizer (marked with None lr)
        ("max_sampling_rate", torch.nn.Parameter(max_sampling_rate, requires_grad=False), None),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1

    # Create optimizers for each parameter group
    optimizers = {}
    for name, _, lr in params:
        if lr is None:  # Skip parameters without learning rate
            continue

        # Scaled learning rate and hyperparameters based on batch size
        scaled_lr = lr * math.sqrt(batch_size)
        scaled_eps = 1e-15 / math.sqrt(batch_size)
        scaled_betas = (1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.99))#, 1 - batch_size * (1 - 0.99))

        # Choose optimizer based on sparse_grad setting
        if sparse_grad:
            # Use SparseAdam for sparse gradients
            optimizer = torch.optim.SparseAdam(
                [{"params": splats[name], "lr": scaled_lr}],
                eps=scaled_eps,
                betas=scaled_betas,
            )
        else:
            # Use Adan optimizer (drop-in replacement for Adam with better performance)
            optimizer = torch.optim.Adam(
                [{"params": splats[name], "lr": lr }],
                eps=scaled_eps,
                betas=scaled_betas,
                fused=True,  # Enable fused operations for better performance
            )

        optimizers[name] = optimizer

    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        
        # Choose between preloaded and regular dataset
        if cfg.preload_images:
            from datasets.preloaded_dataset import PreloadedDataset
            self.trainset = PreloadedDataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
                device=self.device,
                to_gpu=True,  # Store directly on GPU for fastest access
            )
            self.valset = PreloadedDataset(
                self.parser,
                split="val",
                patch_size=None,
                load_depths=False,
                device=self.device,
                to_gpu=False, # Store directly on GPU for fastest access
            )
        else:
            self.trainset = Dataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
            )
            self.valset = Dataset(self.parser, split="val")
        
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Auto-calculate epoch parameters from legacy step-based values
        n_cameras_per_epoch = len(self.trainset)
        print(f"Cameras per epoch: {n_cameras_per_epoch}")
        
        # Align eval and save steps to epoch boundaries
        cfg.align_steps_to_epochs(n_cameras_per_epoch)
        print(f"Aligned eval_steps to epochs: {cfg.eval_steps}")
        print(f"Aligned save_steps to epochs: {cfg.save_steps}")
        print(f"Aligned max_steps: {cfg.max_steps} ({cfg.max_steps // n_cameras_per_epoch} epochs)")
        
        if cfg.auto_epoch_params:
            # Legacy step-based defaults (from original 3DGS)
            legacy_refine_start_iter = cfg.legacy_refine_start_iter
            legacy_refine_stop_iter = cfg.legacy_refine_stop_iter
            legacy_reset_every_iter = cfg.legacy_reset_every_iter
            legacy_refine_every_iter = cfg.legacy_refine_every_iter
            
            # Auto-calculate epoch parameters
            cfg.refine_start_epochs = max(1, math.ceil(legacy_refine_start_iter / n_cameras_per_epoch))
            cfg.refine_stop_epochs = max(1, math.ceil(legacy_refine_stop_iter / n_cameras_per_epoch))
            cfg.reset_every_epochs = max(1, math.ceil(legacy_reset_every_iter / n_cameras_per_epoch))
            cfg.refine_every_epochs = max(1, math.ceil(legacy_refine_every_iter / n_cameras_per_epoch))
            
            # For MCMC strategy, add_every_epochs is 3x refine_every_epochs by default
            cfg.add_every_epochs = cfg.refine_every_epochs * 3
            
            # Adjust reset_start_epochs to be after some refinement cycles
            cfg.reset_start_epochs = cfg.refine_start_epochs + cfg.refine_every_epochs * 2
            
            print(f"Auto-calculated epoch parameters from legacy step-based values:")
        else:
            print(f"Using manually set epoch parameters:")
        
        print(f"  refine_start={cfg.refine_start_epochs} epochs, "
              f"refine_stop={cfg.refine_stop_epochs} epochs")
        if isinstance(cfg.strategy, MCMCStrategy):
            print(f"  refine_every={cfg.refine_every_epochs} epochs, "
                  f"add_every={cfg.add_every_epochs} epochs")
        else:
            print(f"  reset_every={cfg.reset_every_epochs} epochs, "
                  f"refine_every={cfg.refine_every_epochs} epochs")
            print(f"  reset_start={cfg.reset_start_epochs} epochs, "
                  f"reset_end={cfg.reset_end_epochs} epochs")

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        self.model_type = cfg.model_type

        if self.model_type == "2dgs":
            key_for_gradient = "gradient_2dgs"
        else:
            key_for_gradient = "means2d"

        # Densification Strategy
        # self.strategy = DefaultStrategy(
        #     verbose=True,
        #     prune_opa=cfg.prune_opa,
        #     grow_grad2d=cfg.grow_grad2d,
        #     grow_scale3d=cfg.grow_scale3d,
        #     prune_scale3d=cfg.prune_scale3d,
        #     # refine_scale2d_stop_iter=4000, # splatfacto behavior
        #     refine_start_iter=cfg.refine_start_iter,
        #     refine_stop_iter=cfg.refine_stop_iter,
        #     reset_every=cfg.reset_every,
        #     refine_every=max(cfg.refine_every, len(self.parser.train_cameras)),
        #     absgrad=cfg.absgrad,
        #     revised_opacity=cfg.revised_opacity,
        #     key_for_gradient=key_for_gradient,
        # )

        self.cfg.strategy.verbose = True
        
        # Update strategy with epoch parameters from config
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.cfg.strategy.refine_start_epochs = cfg.refine_start_epochs
            self.cfg.strategy.refine_stop_epochs = cfg.refine_stop_epochs
            self.cfg.strategy.reset_start_epochs = cfg.reset_start_epochs
            self.cfg.strategy.reset_end_epochs = cfg.reset_end_epochs
            self.cfg.strategy.reset_every_epochs = cfg.reset_every_epochs
            self.cfg.strategy.refine_every_epochs = cfg.refine_every_epochs
            self.cfg.strategy.pause_refine_after_reset_epochs = cfg.pause_refine_after_reset_epochs
            self.cfg.strategy.prune_opa = cfg.prune_opa
            self.cfg.strategy.grow_grad2d = cfg.grow_grad2d
            self.cfg.strategy.grow_scale3d = cfg.grow_scale3d
            self.cfg.strategy.prune_scale3d = cfg.prune_scale3d
            self.cfg.strategy.absgrad = cfg.absgrad
            self.cfg.strategy.revised_opacity = cfg.revised_opacity
            self.cfg.strategy.key_for_gradient = key_for_gradient
            # Importance-based pruning parameters
            self.cfg.strategy.importance_prune_enabled = cfg.importance_prune_enabled
            self.cfg.strategy.importance_prune_start_epoch = cfg.importance_prune_start_epoch
            self.cfg.strategy.importance_prune_end_epoch = cfg.importance_prune_end_epoch
            self.cfg.strategy.importance_prune_every_epochs = cfg.importance_prune_every_epochs
            self.cfg.strategy.importance_prune_ratio = cfg.importance_prune_ratio
            # Split parameters for gaussians that dominate or touch too many pixels
            self.cfg.strategy.split_big_dominated_pct = cfg.split_big_dominated_pct
            self.cfg.strategy.split_big_touched_pct = cfg.split_big_touched_pct
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            # Update MCMCStrategy with epoch parameters
            self.cfg.strategy.refine_start_epochs = cfg.refine_start_epochs
            self.cfg.strategy.refine_stop_epochs = cfg.refine_stop_epochs
            self.cfg.strategy.refine_every_epochs = cfg.refine_every_epochs
            self.cfg.strategy.add_every_epochs = cfg.add_every_epochs
            self.cfg.strategy.min_opacity = cfg.prune_opa
            self.cfg.strategy.growth_factor = 1.15
            self.cfg.strategy.verbose = True
            self.cfg.strategy.model_type = cfg.model_type

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        else:
            assert_never(self.cfg.strategy)

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

        self.app_optimizers = []
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]

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
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        if cfg.use_aa_smoothing:
            update_max_sampling_rate(
                self.splats,
                self.trainset,
                self.cfg.near_plane,
                self.cfg.far_plane,
                self.device,
            )

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )

        # Viewer
        if not self.cfg.disable_viewer and self.cfg.use_viser:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

        elif not self.cfg.disable_viewer and not self.cfg.use_viser:
            self.tinyrenderr = CUDARenderer()

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        splats: Optional[Dict[str, Tensor]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        # Use provided splats or fall back to self.splats
        if splats is None:
            splats = self.splats

        means = splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = splats["quats"]  # [N, 4]
        scales = scaling_activation(splats["scales"])  # [N, 3]
        opacities = opacity_activation(splats["opacities"])  # [N,]

        drop_rate = kwargs.pop("drop_rate", None)
        if drop_rate is not None:
            opacities = F.dropout(opacities, p=drop_rate, training=True)

        # Check for non-finite values before rasterization
        param_checks = [
            ("means", means),
            ("quats", quats),
            ("scales", scales),
            ("opacities", opacities),
        ]

        if kwargs.get("extra_features") is not None:
            param_checks.append(("extra", kwargs["extra_features"]))

        image_ids = kwargs.pop("image_ids", None)
        override_colors = kwargs.pop("override_colors", None)
        overmax_opacity = kwargs.pop("overmax_opacity", False)
        f_orig = kwargs.pop("f_orig", None)
        blur_mod = kwargs.pop("blur_mod", None)

        if override_colors is not None:
            # Use provided override colors (e.g., for colormapped visualizations)
            colors = override_colors
        elif self.cfg.app_opt:
            colors = self.app_module(
                features=splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]

        assert self.cfg.antialiased is False, "Antialiased is not supported for 2DGS"

        if self.model_type == "2dgs":
            # Apply AA-2DGS smoothing for 2DGS
            cfg_aa = self.cfg.use_aa_smoothing
            gui_aa = rasterize_mode == "antialiased"
            not_gui = rasterize_mode is None
            if cfg_aa:
                if (gui_aa or (not_gui and cfg_aa)) and "max_sampling_rate" in self.splats:

                    # Get focal length from K matrix
                    focal = float((Ks[0, 0, 0] + Ks[0, 1, 1]) / 2.0)
                    if f_orig is None:
                        f_orig = focal

                    # Apply AA-2DGS smoothing
                    scales, opacities = apply_flat_smoothing(
                        scales, opacities, self.splats["max_sampling_rate"].detach(),
                        s_reg=self.cfg.aa_smoothing_reg, focal=focal, f_orig=f_orig, blur_mod=blur_mod)

            if overmax_opacity:  # self._scale_modifier <= 0.02:
                opacities = torch.full_like(opacities, fill_value=1e3)

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
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                packed=self.cfg.packed,
                # absgrad=self.cfg.absgrad, # для gsplat2dgs он всегда считается
                sparse_grad=self.cfg.sparse_grad,
                **kwargs,
            )
        elif self.model_type == "2dgs-inria":
            raise NotImplementedError("2dgs-inria disabled")

        return (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        )

    def train(self):
        cfg = self.cfg
        device = self.device

        # Dump cfg.
        # with open(f"{cfg.result_dir}/cfg.json", "w") as f:
        #     json.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]

        # Get camera optimizer's optimizers and add schedulers for them
        camera_optimizers = {}
        self.camera_optimizer.get_param_groups(camera_optimizers)
        for opt_name, opt_params in camera_optimizers.items():
            if opt_params:  # Only if there are parameters to optimize
                optimizer = torch.optim.Adam(
                    opt_params,
                    lr=1e-5 * math.sqrt(cfg.batch_size),  # Default lr, can be adjusted
                    weight_decay=1e-6,
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
            intrinsics_optimizer = torch.optim.Adam(
                self.optimized_Ks.parameters(),
                lr=cfg.intrinsics_lr * math.sqrt(cfg.batch_size),
                eps=1e-15 / math.sqrt(cfg.batch_size),
            )
            self.intrinsics_optimizers = [intrinsics_optimizer]
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    intrinsics_optimizer, gamma=0.01 ** (1.0 / max_steps)
                )
            )

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

        # Create dataloader based on preload_images setting
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

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))

        for step, data, epoch_ctx in training_data_generator(trainloader, pbar):
            if not cfg.disable_viewer and cfg.use_viser:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            # Call batch start callback at the beginning of each epoch
            if epoch_ctx.epoch_start:
                self.cfg.strategy.step_epoch_start(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    epoch_ctx=epoch_ctx,
                )

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            
            # Get image_ids early for intrinsics optimization
            image_ids = data["image_id"].to(device)
            
            # Use optimized intrinsics if enabled
            if cfg.optimize_intrinsics and self.optimized_Ks is not None:
                # Reconstruct K matrix from optimized parameters
                cam_idx = image_ids[0].item()
                fx, fy, cx, cy = self.Ks_structure[cam_idx]
                
                K_opt = torch.zeros(3, 3, device=device)
                K_opt[0, 0] = fx if fx is not None else Ks[0, 0, 0]
                K_opt[1, 1] = fy if fy is not None else Ks[0, 1, 1]
                K_opt[0, 2] = cx if cx is not None else Ks[0, 0, 2]
                K_opt[1, 2] = cy if cy is not None else Ks[0, 1, 2]
                K_opt[2, 2] = 1.0
                
                Ks = K_opt.unsqueeze(0)  # [1, 3, 3]
            
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )

            this_pass_depth_loss = cfg.depth_loss and "points" in data and "depths" in data

            if this_pass_depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

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
                # apply_to_camera returns [1, 3, 4], need to convert to [1, 4, 4]
                optimized_c2w = self.camera_optimizer.apply_to_camera(camera)
                # Add the homogeneous row [0, 0, 0, 1]
                bottom_row = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=device)
                camtoworlds = torch.cat([optimized_c2w, bottom_row], dim=1)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            gamma = 0.2  # Scaling factor из статьи arXiv:2504.00773
            drop_rate = gamma * (step / max_steps)

            # forward
            (
                world_renders,
                world_alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                info,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if this_pass_depth_loss else "RGB+D",
                distloss=self.cfg.dist_loss,
                track_domination=True,
                drop_rate=drop_rate,
            )

            # Use world renders directly
            render_alphas = world_alphas
            render_colors = world_renders

            # Add camtoworlds to info for distance computation in strategy
            info["camtoworlds"] = camtoworlds
            info["Ks"] = Ks

            # Add AA-2DGS parameters if enabled
            if cfg.use_aa_smoothing:
                info["aa_params"] = {
                    "trainset": self.trainset,
                    "near_plane": cfg.near_plane,
                    "far_plane": cfg.far_plane,
                    "device": self.device,
                    "aa_compute_every": cfg.aa_compute_every,
                }

            if render_colors.shape[-1] == 4:
                colors, depths = render_colors[..., 0:3], render_colors[..., 3:4]
            else:
                colors, depths = render_colors, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(
                    self.bil_grids,
                    grid_xy.expand(colors.shape[0], -1, -1, -1),
                    colors,
                    image_ids.unsqueeze(-1),
                )["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - render_alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                epoch_ctx=epoch_ctx,
            )
            masks = data["mask"].to(device) if "mask" in data else None
            if masks is not None:
                pixels = pixels * masks[..., None]
                colors = colors * masks[..., None]

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if this_pass_depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]

                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = torch.abs(disp - disp_gt).mean()
                depthloss = depthloss * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            if cfg.normal_loss:
                if step > cfg.normal_start_iter:
                    curr_normal_lambda = cfg.normal_lambda
                else:
                    curr_normal_lambda = 0.0
                # normal consistency loss
                normals = normals.squeeze(0).permute((2, 0, 1))
                normals_from_depth *= world_alphas.squeeze(0).detach()
                if len(normals_from_depth.shape) == 4:
                    normals_from_depth = normals_from_depth.squeeze(0)
                normals_from_depth = normals_from_depth.permute((2, 0, 1))
                normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
                normalloss = curr_normal_lambda * normal_error.mean()
                loss += normalloss

            if cfg.dist_loss:
                if step > cfg.dist_start_iter:
                    curr_dist_lambda = cfg.dist_lambda
                else:
                    curr_dist_lambda = 0.0
                distloss = render_distort.mean()
                loss += distloss * curr_dist_lambda

            if cfg.opacity_entropy_loss:
                if step > cfg.opacity_entropy_start_iter:
                    curr_opacity_entropy_lambda = cfg.opacity_entropy_lambda
                else:
                    curr_opacity_entropy_lambda = 0.0
                # Binary cross-entropy of opacities with themselves (entropy regularization)
                # This penalizes partial transparency (values near 0.5)
                activated_opacities = opacity_activation(self.splats["opacities"])
                # Clamp to avoid log(0)
                activated_opacities_clamped = torch.clamp(activated_opacities, 1e-7, 1 - 1e-7)
                opacity_entropy = -(activated_opacities_clamped * torch.log(activated_opacities_clamped) +
                                   (1 - activated_opacities_clamped) * torch.log(1 - activated_opacities_clamped)).mean()
                loss += opacity_entropy * curr_opacity_entropy_lambda

            if cfg.elongation_loss:
                if step > cfg.elongation_start_iter:
                    curr_elongation_lambda = cfg.elongation_lambda
                else:
                    curr_elongation_lambda = 0.0
                # Calculate elongation in log space (scales are stored in log form)
                log_scales = self.splats["scales"][..., :2]  # [N, 2] - 2DGS, only x,y scales (no activation)
                log_elongation_ratio = torch.abs(log_scales[:, 0] - log_scales[:, 1])  # [N] - abs(log_x - log_y) = log(max/min)

                excess_ratio = torch.clamp(torch.exp(log_elongation_ratio) - cfg.elongation_threshold, min=0.0)
                elongation_loss = (excess_ratio ** 2).mean()
                loss += elongation_loss * curr_elongation_lambda

            if cfg.scale_percentile_loss:
                if step > cfg.scale_percentile_start_iter:
                    curr_scale_percentile_lambda = cfg.scale_percentile_lambda
                else:
                    curr_scale_percentile_lambda = 0.0
                
                # Get activated scales (real sizes, not log)
                activated_scales = scaling_activation(self.splats["scales"])  # [N, 3]
                # For 2DGS, use area of 1-sigma ellipse as size metric
                # Area = π * σ_x * σ_y
                scale_areas = torch.pi * activated_scales[:, 0] * activated_scales[:, 1]  # [N]
                
                # Compute percentiles
                lower_percentile = torch.quantile(scale_areas, cfg.scale_percentile_lower / 100.0)
                upper_percentile = torch.quantile(scale_areas, cfg.scale_percentile_upper / 100.0)
                
                # Penalize scales outside percentile range
                # Quadratic penalty for being below lower percentile
                below_penalty = torch.clamp(lower_percentile - scale_areas, min=0.0) ** 2
                # Quadratic penalty for being above upper percentile
                above_penalty = torch.clamp(scale_areas - upper_percentile, min=0.0) ** 2
                
                scale_percentile_loss = (below_penalty + above_penalty).mean()
                loss += scale_percentile_loss * curr_scale_percentile_lambda

            if cfg.opacity_l1_loss:
                if epoch_ctx.i_epoch >= cfg.opacity_l1_start_epoch:
                    curr_opacity_l1_lambda = cfg.opacity_l1_lambda
                else:
                    curr_opacity_l1_lambda = 0.0
                # L1 regularization on activated opacities
                activated_opacities = opacity_activation(self.splats["opacities"])
                opacity_l1_loss = (activated_opacities).mean()
                loss += opacity_l1_loss * curr_opacity_l1_lambda

            if cfg.erank_loss:
                if step > cfg.erank_start_iter:
                    curr_erank_lambda = cfg.erank_lambda
                else:
                    curr_erank_lambda = 0.0
                
                # Calculate effective rank for 2DGS
                activated_scales = scaling_activation(self.splats["scales"])[..., :2]  # [N, 2]
                
                # Square the scales (variance proportional to squared std dev)
                scales_squared = activated_scales ** 2  # [N, 2]
                
                # Calculate proportions: p_i = scale_i^2 / sum(scales^2)
                sum_scales_squared = scales_squared.sum(dim=1, keepdim=True)  # [N, 1]
                proportions = scales_squared / (sum_scales_squared + 1e-10)  # [N, 2]
                
                # Calculate entropy: H = -sum(p_i * log(p_i))
                proportions_safe = torch.clamp(proportions, min=1e-10)
                entropy = -(proportions_safe * torch.log(proportions_safe)).sum(dim=1)  # [N]
                
                # Effective rank = exp(entropy)
                erank = torch.exp(entropy)  # [N]
                
                # Penalty = max(-log(erank - 1), 0)
                # Add small epsilon to avoid log(0) when erank is close to 1
                erank_penalty = torch.clamp(-torch.log(torch.clamp(erank - 1, min=1e-10)), min=0.0)
                erank_loss = erank_penalty.mean()
                loss += erank_loss * curr_erank_lambda

            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # Add loss from camera optimizer
            loss_dict = {}
            self.camera_optimizer.get_loss_dict(loss_dict)
            for loss_name, loss_value in loss_dict.items():
                loss += loss_value

            assert self.splats["means"].isfinite().all()

            # Comprehensive loss component checking
            loss_components = {
                "l1loss": l1loss,
                "ssimloss": ssimloss,
                "total_loss": loss,
            }

            if cfg.depth_loss and this_pass_depth_loss:
                loss_components["depthloss"] = depthloss

            if cfg.normal_loss and step > cfg.normal_start_iter:
                loss_components["normalloss"] = normalloss

            if cfg.dist_loss and step > cfg.dist_start_iter:
                loss_components["distloss"] = distloss

            if cfg.opacity_entropy_loss and step > cfg.opacity_entropy_start_iter:
                loss_components["opacity_entropy"] = opacity_entropy

            if cfg.elongation_loss and step > cfg.elongation_start_iter:
                loss_components["elongation_loss"] = elongation_loss

            if cfg.scale_percentile_loss and step > cfg.scale_percentile_start_iter:
                loss_components["scale_percentile_loss"] = scale_percentile_loss

            if cfg.erank_loss and step > cfg.erank_start_iter:
                loss_components["erank_loss"] = erank_loss

            if cfg.opacity_l1_loss and epoch_ctx.i_epoch >= cfg.opacity_l1_start_epoch:
                loss_components["opacity_l1_loss"] = opacity_l1_loss

            if cfg.use_bilateral_grid:
                loss_components["tvloss"] = tvloss

            loss.backward()


            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            desc += f"epoch={epoch_ctx.i_epoch} ({100.0 * (epoch_ctx.i + 1) / epoch_ctx.epoch_len:.1f}%)| "
            if this_pass_depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.dist_loss:
                desc += f"dist loss={distloss.item():.6f}"
            if cfg.opacity_entropy_loss and step > cfg.opacity_entropy_start_iter:
                desc += f" ent={opacity_entropy.item():.4f}"
            if cfg.elongation_loss and step > cfg.elongation_start_iter:
                desc += f" elong={elongation_loss.item():.4f}"
            if cfg.scale_percentile_loss and step > cfg.scale_percentile_start_iter:
                desc += f" scale_percentile={scale_percentile_loss.item():.4f}"
            if cfg.erank_loss and step > cfg.erank_start_iter:
                desc += f" erank={erank_loss.item():.4f}"
            if cfg.opacity_l1_loss and epoch_ctx.i_epoch >= cfg.opacity_l1_start_epoch:
                desc += f" op_l1={opacity_l1_loss.item():.4f}"
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if this_pass_depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.normal_loss:
                    self.writer.add_scalar("train/normalloss", normalloss.item(), step)
                if cfg.dist_loss:
                    self.writer.add_scalar("train/distloss", distloss.item(), step)
                if cfg.opacity_entropy_loss:
                    self.writer.add_scalar("train/opacity_entropy_loss", opacity_entropy.item(), step)
                if cfg.elongation_loss and step > cfg.elongation_start_iter:
                    self.writer.add_scalar("train/elongation_loss", elongation_loss.item(), step)
                if cfg.scale_percentile_loss and step > cfg.scale_percentile_start_iter:
                    self.writer.add_scalar("train/scale_percentile_loss", scale_percentile_loss.item(), step)
                if cfg.opacity_l1_loss and epoch_ctx.i_epoch >= cfg.opacity_l1_start_epoch:
                    self.writer.add_scalar("train/opacity_l1_loss", opacity_l1_loss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.optimize_intrinsics and self.optimized_Ks is not None:
                    # Log first camera's intrinsics as example
                    fx, fy, cx, cy = self.Ks_structure[0]
                    if fx is not None:
                        self.writer.add_scalar("train/intrinsics/fx", fx.item(), step)
                    if fy is not None:
                        self.writer.add_scalar("train/intrinsics/fy", fy.item(), step)
                    if cx is not None:
                        self.writer.add_scalar("train/intrinsics/cx", cx.item(), step)
                    if cy is not None:
                        self.writer.add_scalar("train/intrinsics/cy", cy.item(), step)
                if cfg.tb_save_image:
                    canvas = (
                        torch.cat([pixels, colors[..., :3]], dim=2)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()


            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize
            for k, optimizer in self.optimizers.items():
                # print(k)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # Optimize intrinsics if enabled
            for optimizer in self.intrinsics_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()


            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                checkpoint_data = {
                    "step": step,
                    "splats": self.splats.state_dict(),
                }
                # Save optimized intrinsics if enabled
                if cfg.optimize_intrinsics and self.optimized_Ks is not None:
                    checkpoint_data["optimized_Ks"] = self.optimized_Ks.state_dict()
                    checkpoint_data["Ks_structure"] = self.Ks_structure
                torch.save(
                    checkpoint_data,
                    f"{self.ckpt_dir}/ckpt_{step}.pt",
                )

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step)
                self.render_traj(step)

            if not cfg.disable_viewer and cfg.use_viser:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        # Create validation dataloader based on preload_images setting
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

        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        if cfg.use_bilateral_grid:
            metrics.update({"cc_psnr": [], "cc_ssim": [], "cc_lpips": []})
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            (
                colors,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                _,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 3]
            colors = torch.clamp(colors, 0.0, 1.0)
            colors = colors[..., :3]  # Take RGB channels

            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}.png", (canvas * 255).astype(np.uint8)
            )

            # write median depths
            render_median = normalize_robust(render_median)
            # render_median = render_median.detach().cpu().squeeze(0).unsqueeze(-1).repeat(1, 1, 3).numpy()
            render_median = (
                apply_float_colormap(render_median).detach().cpu().squeeze(0).numpy()
            )

            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_median_depth_{step}.png",
                (render_median * 255).astype(np.uint8),
            )

            # write normals
            normals = (normals * 0.5 + 0.5).squeeze(0).cpu().numpy()
            normals_output = (normals * 255).astype(np.uint8)
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_normal_{step}.png", normals_output
            )

            # write normals from depth
            normals_from_depth *= alphas.squeeze(0).detach()
            normals_from_depth = (normals_from_depth * 0.5 + 0.5).cpu().numpy()
            normals_from_depth = (normals_from_depth - np.min(normals_from_depth)) / (
                np.max(normals_from_depth) - np.min(normals_from_depth)
            )
            normals_from_depth_output = (normals_from_depth * 255).astype(np.uint8)
            if len(normals_from_depth_output.shape) == 4:
                normals_from_depth_output = normals_from_depth_output.squeeze(0)
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_normals_from_depth_{step}.png",
                normals_from_depth_output,
            )

            # write distortions

            render_dist = render_distort
            render_dist = normalize_robust(render_dist)
            render_dist = (
                apply_float_colormap(render_dist).detach().cpu().squeeze(0).numpy()
            )
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_distortions_{step}.png",
                (render_dist * 255).astype(np.uint8),
            )

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))
            if cfg.use_bilateral_grid:
                cc_colors = color_correct(colors.permute(0, 2, 3, 1), pixels.permute(0, 2, 3, 1))
                cc_colors = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["cc_psnr"].append(self.psnr(cc_colors, pixels))
                metrics["cc_ssim"].append(self.ssim(cc_colors, pixels))
                metrics["cc_lpips"].append(self.lpips(cc_colors, pixels))

        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()

        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(self.splats["means"]),
        }

        if cfg.use_bilateral_grid:
            cc_psnr = torch.stack(metrics["cc_psnr"]).mean()
            cc_ssim = torch.stack(metrics["cc_ssim"]).mean()
            cc_lpips = torch.stack(metrics["cc_lpips"]).mean()
            stats.update({
                "cc_psnr": cc_psnr.item(),
                "cc_ssim": cc_ssim.item(),
                "cc_lpips": cc_lpips.item(),
            })
            print(
                f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
                f"CC_PSNR: {cc_psnr.item():.3f}, CC_SSIM: {cc_ssim.item():.4f}, CC_LPIPS: {cc_lpips.item():.3f} "
                f"Time: {ellipse_time:.3f}s/image "
                f"Number of GS: {len(self.splats['means'])}"
            )
        else:
            print(
                f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
                f"Time: {ellipse_time:.3f}s/image "
                f"Number of GS: {len(self.splats['means'])}"
            )

        # save stats as json
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds = self.parser.camtoworlds[5:-5]
        camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, alphas, _, surf_normals, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]

            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = normalize_robust(depths)
            surf_normals = normalize_robust(surf_normals)

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        
        # Extract all self references at the beginning for future extraction
        device = self.device
        splats = self.splats
        parser_Ks_dict = self.parser.Ks_dict
        cfg = self.cfg
        strategy_state = self.strategy_state
        trainset = self.trainset
        rasterize_splats_fn = self.rasterize_splats
        epoch_stats = strategy_state.get("epoch_stats", None) if strategy_state else None

        n_cameras = len(trainset)
        
        return render_inner(camera_state=camera_state, render_tab_state=render_tab_state, device=device,
                            splats=splats, parser_Ks_dict=parser_Ks_dict, cfg=cfg, n_cameras=n_cameras,
                            rasterize_splats_fn=rasterize_splats_fn,
                            epoch_stats=epoch_stats)


def main(cfg: Config):
    runner = Runner(cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            if k in ckpt["splats"]:
                runner.splats[k].data = ckpt["splats"][k]
            elif k == "max_sampling_rate" and k not in ckpt["splats"]:
                # Initialize max_sampling_rate if not in checkpoint
                N = len(runner.splats["means"])
                runner.splats[k].data = torch.zeros(N, device=runner.device)
        
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    
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
    
    main(cfg)

