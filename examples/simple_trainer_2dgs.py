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

from examples.lib_skysphere import reproject_skysphere, compute_skysphere_geometry
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras

import imageio
import numpy as np
import torch

# GPU поддерживает TensorFloat32 (TF32) tensor cores для ускорения матричных умножений с float32, но PyTorch не использует их по умолчанию.
torch.set_float32_matmul_precision('high')

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

from examples.utils import normalize_robust, index_map_to_pseudocolor
from gsplat import MCMCStrategy
from gsplat.strategy.ops import scaling_inverse_activation, opacity_inverse_activation, scaling_activation, opacity_activation
from gsplat.antialias_2dgs import apply_flat_smoothing, calc_sigma_sq, update_max_sampling_rate

from examples.utils import (
    AppearanceOptModule,
    CameraOptModule,
    knn,
    rgb_to_sh,
    set_random_seed,
    scalar_to_colormap,
    skyness_to_colormap,
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
    data_dir: str = r"X:\_ai\_gsplat\datasets\garden"
    # data_dir: str = r"x:\_ai\_gsplat\datasets\bicycle"
    # data_dir: str = r"x:\_ai\_demos\_gsplat\_datasets\mip360\kitchen"
    # data_dir: str = r"x:\_ai\_gsplat\datasets\fb_colmap_res"
    # data_dir: str = r"y:\_gopro_kv92\extracted_keyframes\GOPR6996_colmap"
    # data_dir: str = r"x:\_ai\_demos\_gsplat\_datasets\segment-102751"
    # data_dir: str = r"x:\_ai\_demos\_gsplat\_datasets\youtube01"
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
    preload_images: bool = False

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
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
    grow_grad2d: float = 0.0008
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1

    # Start refining GSs after this epoch
    refine_start_epochs: int = 3
    # Stop refining GSs after this epoch
    refine_stop_epochs: int = 500
    # Refine GSs every this many epochs
    refine_every_epochs: int = 1
    # Start resetting opacities after this epoch
    reset_start_epochs: int = 100
    # Stop resetting opacities after this epoch
    reset_end_epochs: int = 10000
    # Reset opacities every this many epochs
    reset_every_epochs: int = 20
    # Pause refining for this many epochs after reset
    pause_refine_after_reset_epochs: int = 1

    # Auto-calculate epoch parameters from legacy step-based values
    auto_epoch_params: bool = True
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

    # Skysphere parameters
    skysphere_enabled: bool = False
    # Radius of skysphere as multiple of scene extent
    skysphere_radius_multiplier: float = 20.0
    # Number of points to sample on skysphere
    skysphere_points: int = 50_000
    # Learning rate for skyness attribute
    skyness_lr: float = 0.01
    # Regularization weight for skyness
    skyness_reg: float = 0.01
    # Enable skyness supervision from sky masks
    skyness_supervision: bool = True
    # Weight for skyness supervision loss
    skyness_supervision_lambda: float = 0.1
    # Weight for skysphere deviation loss
    skysphere_radius_reg: float = 0.05

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
    opacity_entropy_loss: bool = False
    # Weight for opacity entropy loss
    opacity_entropy_lambda: float = 1e-3
    # Iteration to start opacity entropy regularization
    opacity_entropy_start_iter: int = 1_000

    # Elongation regularization (penalizes stretched gaussians)
    elongation_loss: bool = True
    # Weight for elongation loss
    elongation_lambda: float = 1e-3
    # Threshold for elongation ratio (penalty starts after this ratio)
    elongation_threshold: float = 4.0
    # Iteration to start elongation regularization
    elongation_start_iter: int = 1_000

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
        default_factory=lambda: DefaultStrategy()
    )

    # AA-2DGS parameters
    use_aa_smoothing: bool = True  # Enable AA-2DGS smoothing
    aa_smoothing_reg: float = 0.1  # s_reg parameter from paper
    aa_compute_every: int = 1  # Recompute compute_min_depth_normalized_sq every N epochs

    # Importance-based pruning parameters (Speedy-Splat style)
    importance_prune_enabled: bool = False  # Enable importance-based pruning
    importance_prune_start_epoch: int = 8  # Start importance pruning after this epoch
    importance_prune_end_epoch: int = 10000  # Stop importance pruning after this epoch
    importance_prune_every_epochs: int = 1  # Perform importance pruning every this many epochs
    importance_prune_ratio: float = 0.005  # Fraction to prune (0.3 = remove 30% least important)

    # Split parameters for gaussians that dominate or touch too many pixels
    split_big_dominated_pct: float = 0.0005  # Split gaussians dominating more than this percentage of pixels
    split_big_touched_pct: float = 0.001  # Split gaussians touching more than this percentage of pixels

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
    skyness_lr: float = 0.01,
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
    
    # SKYNESS SYSTEM:
    # Skyness is a learnable per-gaussian attribute that indicates whether a gaussian belongs to the sky/background.
    # It's stored in logit space (before sigmoid) for stable optimization:
    #   - skyness = 0 (logit) -> sigmoid(0) = 0.5 probability (uncertain)
    #   - skyness > 0 (logit) -> sigmoid(skyness) > 0.5 (likely sky)
    #   - skyness < 0 (logit) -> sigmoid(skyness) < 0.5 (likely world object)
    # 
    # The system works as follows:
    # 1. Regular SfM points start with skyness=0 (neutral, will be learned during training)
    # 2. Skysphere points are initialized with skyness=1.5 (high confidence they're sky)
    # 3. During training, skyness is optimized along with other parameters
    # 4. Densification strategy uses skyness to treat sky/world gaussians differently
    # 5. Sky gaussians (skyness > 0.75) can have different pruning/splitting behavior
    skyness = torch.zeros((N,))  # [N,] - logit space, 0 = 0.5 probability (neutral/uncertain)

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
        ("skyness", torch.nn.Parameter(skyness), skyness_lr),
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
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size)}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    }
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
            
            # Adjust reset_start_epochs to be after some refinement cycles
            cfg.reset_start_epochs = cfg.refine_start_epochs + cfg.refine_every_epochs * 2
            
            print(f"Auto-calculated epoch parameters from legacy step-based values:")
        else:
            print(f"Using manually set epoch parameters:")
        
        print(f"  refine_start={cfg.refine_start_epochs} epochs, "
              f"refine_stop={cfg.refine_stop_epochs} epochs")
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
            skyness_lr=cfg.skyness_lr,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        self.model_type = cfg.model_type

        # Initialize skysphere if enabled
        if cfg.skysphere_enabled:
            self._initialize_skysphere()

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
            self.cfg.strategy.growth_factor = 1.15
            self.cfg.strategy.verbose = True

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Initialize camera optimizer from nerfstudio
        self.camera_optimizer: CameraOptimizer = cfg.camera_optimizer.setup(
            num_cameras=len(self.trainset), device=self.device
        )
        self.pose_optimizers = []  # Will be populated in train() if camera optimization is enabled

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
                self.strategy_state,
                self.trainset,
                self.cfg.near_plane,
                self.cfg.far_plane,
                self.device
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

    def _initialize_skysphere(self):
        """Initialize skysphere points from camera views.
        
        SKYSPHERE INITIALIZATION:
        The skysphere is a sphere of gaussians placed far from the scene center to represent
        the sky/background. These gaussians are initialized by:
        1. Creating a fibonacci sphere of points at radius = scene_scale * radius_multiplier
        2. For each training camera, projecting these points to the image
        3. Sampling colors from the image at projected locations
        4. If sky masks are available, only using points that project to sky regions
        5. Setting skyness=1.5 for these points (high confidence they're sky)
        6. Orienting their normals to point towards the scene center
        
        This provides a good initialization for sky regions that can be further optimized.
        """
        cfg = self.cfg
        device = self.device

        # Get skysphere geometry from lib_skysphere
        new_points, new_colors, new_quats = compute_skysphere_geometry(
            self.trainset,
            self.scene_scale * cfg.skysphere_radius_multiplier,
            cfg.skysphere_points,
            device
        )

        if new_points is not None:
            N_sky = new_points.shape[0]
            
            # Create parameters for skysphere points
            dist2_avg = (knn(new_points, min(4, N_sky))[:, 1:] ** 2).mean(dim=-1)
            dist_avg = torch.sqrt(dist2_avg)
            new_scales = scaling_inverse_activation(dist_avg * self.cfg.init_scale).unsqueeze(-1).repeat(1, 3)
            
            new_opacities = opacity_inverse_activation(torch.full((N_sky,), self.cfg.init_opa, device=device))
            # Initialize skyness in logit space: logit(0.75) ≈ 1.1 for sky points
            new_skyness = torch.full((N_sky,), torch.logit(torch.tensor(0.75)), device=device)  # High confidence it's sky
            
            # Add SH coefficients for colors
            new_sh0 = torch.zeros((N_sky, 1, 3), device=device)
            new_sh0[:, 0, :] = rgb_to_sh(new_colors)
            new_shN = torch.zeros((N_sky, (self.cfg.sh_degree + 1) ** 2 - 1, 3), device=device)
            
            # Update skyness for existing points to indicate they're likely world objects (25% probability)
            # when skysphere is added
            existing_skyness = self.splats["skyness"].data
            existing_skyness[:] = torch.logit(torch.tensor(0.25))  # logit(0.25) ≈ -1.1
            
            # Concatenate with existing splats
            self.splats["means"] = torch.nn.Parameter(
                torch.cat([self.splats["means"], new_points], dim=0)
            )
            self.splats["scales"] = torch.nn.Parameter(
                torch.cat([self.splats["scales"], new_scales], dim=0)
            )
            self.splats["quats"] = torch.nn.Parameter(
                torch.cat([self.splats["quats"], new_quats], dim=0)
            )
            self.splats["opacities"] = torch.nn.Parameter(
                torch.cat([self.splats["opacities"], new_opacities], dim=0)
            )
            self.splats["skyness"] = torch.nn.Parameter(
                torch.cat([self.splats["skyness"], new_skyness], dim=0)
            )
            self.splats["sh0"] = torch.nn.Parameter(
                torch.cat([self.splats["sh0"], new_sh0], dim=0)
            )
            self.splats["shN"] = torch.nn.Parameter(
                torch.cat([self.splats["shN"], new_shN], dim=0)
            )
            
            # Update optimizers with new parameters
            for name, optimizer in self.optimizers.items():
                param = self.splats[name]
                optimizer.param_groups[0]["params"] = [param]

            print(f"Added {N_sky} skysphere points. Total GS: {len(self.splats['means'])}")

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
            if (gui_aa or (not_gui and cfg_aa)) and "max_sampling_rate_sq" in self.strategy_state:

                # Get focal length from K matrix
                focal = float((Ks[0, 0, 0] + Ks[0, 1, 1]) / 2.0)
                if f_orig is None:
                    f_orig = focal

                # Apply AA-2DGS smoothing
                scales, opacities = apply_flat_smoothing(
                    scales, opacities, self.strategy_state["max_sampling_rate_sq"],
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
                absgrad=self.cfg.absgrad,
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
                # # нужно до сброса. и оно до начала тренировки высчитывается
                # # Recompute compute_min_depth_normalized_sq for AA-2DGS at the beginning of each epoch
                # if self.cfg.use_aa_smoothing and epoch_ctx.i_epoch > 0 and epoch_ctx.i_epoch % self.cfg.aa_compute_every == 0:
                #     self._compute_max_sampling_rate_sq()

                self.cfg.strategy.step_epoch_start(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    epoch_ctx=epoch_ctx,
                )

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)

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

            # Prepare extra features for rendering (e.g., skyness)
            extra_features = None
            sky_mask_rendered = None
            if cfg.skysphere_enabled:
                # Add skyness as extra feature to be rendered alongside colors
                skyness_values = torch.sigmoid(self.splats["skyness"]).unsqueeze(-1)  # [N, 1]
                extra_features = skyness_values

            # forward
            (
                renders,
                alphas,
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
                extra_features=extra_features,  # Pass extra features to render
            )

            # Extract rendered skyness mask if available
            if cfg.skysphere_enabled and "rendered_extras" in info:
                sky_mask_rendered = info["rendered_extras"][0, :, :, 0]  # [H, W] - skyness probability
                # Create world mask (inverse of sky mask) for regularizations
                world_mask = 1.0 - sky_mask_rendered  # Higher values for world objects
            else:
                world_mask = torch.ones((height, width), device=device)

            # Add camtoworlds to info for distance computation in strategy
            info["camtoworlds"] = camtoworlds
            info["Ks"] = Ks

            # Add skyness info if enabled
            if cfg.skysphere_enabled:
                # Convert skyness from logit space to probability space for use in densification
                # Gaussians with skyness > 0.75 are considered sky, < 0.25 are world objects
                info["skyness"] = torch.sigmoid(self.splats["skyness"]).detach()

            # Add AA-2DGS parameters if enabled
            if cfg.use_aa_smoothing:
                info["aa_params"] = {
                    "trainset": self.trainset,
                    "near_plane": cfg.near_plane,
                    "far_plane": cfg.far_plane,
                    "device": self.device,
                    "aa_compute_every": cfg.aa_compute_every,
                }

            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

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
                colors = colors + bkgd * (1.0 - alphas)

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

                # Sample world mask at the same points if skysphere is enabled
                if cfg.skysphere_enabled and "rendered_extras" in info:
                    world_mask_sampled = F.grid_sample(
                        world_mask.unsqueeze(0).unsqueeze(0), grid, align_corners=True
                    )  # [1, 1, M, 1]
                    world_mask_sampled = world_mask_sampled.squeeze(3).squeeze(1)  # [1, M]
                else:
                    world_mask_sampled = torch.ones_like(depths)  # No masking if skysphere disabled

                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                # Apply world mask to depth loss - only compute loss for world objects
                depthloss = (torch.abs(disp - disp_gt) * world_mask_sampled).sum() / (world_mask_sampled.sum() + 1e-6)
                depthloss = depthloss * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            if cfg.normal_loss:
                if step > cfg.normal_start_iter:
                    curr_normal_lambda = cfg.normal_lambda
                else:
                    curr_normal_lambda = 0.0
                # normal consistency loss
                normals = normals.squeeze(0).permute((2, 0, 1))
                normals_from_depth *= alphas.squeeze(0).detach()
                if len(normals_from_depth.shape) == 4:
                    normals_from_depth = normals_from_depth.squeeze(0)
                normals_from_depth = normals_from_depth.permute((2, 0, 1))
                normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
                # Apply world mask to normal loss - only compute loss for world objects
                normal_error_masked = normal_error * world_mask.unsqueeze(0)
                normalloss = curr_normal_lambda * normal_error_masked.sum() / (world_mask.sum() + 1e-6)
                loss += normalloss

            if cfg.dist_loss:
                if step > cfg.dist_start_iter:
                    curr_dist_lambda = cfg.dist_lambda
                else:
                    curr_dist_lambda = 0.0
                # Apply world mask to distortion loss - only compute loss for world objects
                render_distort_masked = render_distort * world_mask.unsqueeze(0).unsqueeze(-1)
                distloss = render_distort_masked.sum() / (world_mask.sum() + 1e-6)
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
                # Calculate elongation ratio (max scale / min scale for each gaussian)
                activated_scales = scaling_activation(self.splats["scales"])[..., :2]  # [N, 3]
                max_scales, _ = activated_scales.max(dim=1)  # [N]
                min_scales, _ = activated_scales.min(dim=1)  # [N]
                elongation_ratio = max_scales / (min_scales + 1e-8)  # [N] - avoid division by zero
                
                # Apply quadratic penalty for ratios above threshold
                excess_ratio = torch.clamp(elongation_ratio - cfg.elongation_threshold, min=0.0)
                elongation_loss = (excess_ratio ** 2).mean()
                loss += elongation_loss * curr_elongation_lambda

            if cfg.skysphere_enabled and cfg.skyness_reg > 0:
                # SKYNESS REGULARIZATION:
                # This loss encourages gaussians to commit to being either sky or world objects,
                # penalizing uncertain values near 0.5 probability.
                # The entropy is minimized when skyness is close to 0 or 1 (after sigmoid).
                # This helps the model make clear decisions about which gaussians represent sky.
                skyness_probs = torch.sigmoid(self.splats["skyness"])
                skyness_clamped = torch.clamp(skyness_probs, 1e-7, 1 - 1e-7)
                skyness_entropy = -(skyness_clamped * torch.log(skyness_clamped) +
                                   (1 - skyness_clamped) * torch.log(1 - skyness_clamped)).mean()
                loss += skyness_entropy * cfg.skyness_reg

            # SKY SPHERE RADIUS REGULARIZATION:
            # Penalize sky gaussians for deviating from the skysphere radius
            skysphere_radius_loss = None
            if cfg.skysphere_enabled and cfg.skysphere_radius_reg > 0:
                skyness_probs = torch.sigmoid(self.splats["skyness"])
                # Only apply to gaussians with high skyness (> 0.75 probability)
                sky_mask = skyness_probs > 0.75
                if sky_mask.any():
                    sky_positions = self.splats["means"][sky_mask]
                    # Calculate distance from origin
                    distances = torch.norm(sky_positions, dim=1)
                    # Target radius for skysphere
                    target_radius = self.scene_scale * cfg.skysphere_radius_multiplier
                    # L2 loss for deviation from target radius
                    radius_deviation = (distances - target_radius) ** 2
                    # Weight by skyness probability (stronger penalty for higher skyness)
                    weighted_deviation = radius_deviation * skyness_probs[sky_mask]
                    skysphere_radius_loss = weighted_deviation.mean()
                    loss += skysphere_radius_loss * cfg.skysphere_radius_reg

            # SKYNESS SUPERVISION FROM SKY MASKS:
            # If sky masks are available, use them as ground truth to supervise skyness learning
            # This creates a cross-entropy loss between rendered skyness and ground truth sky mask
            skyness_supervision_loss = torch.Tensor([0,])
            if cfg.skyness_supervision and cfg.skysphere_enabled and "sky_mask" in data:
                sky_mask_gt = data["sky_mask"].to(device).float()  # [H, W] binary mask

                # Extract skyness from rendered_extras in info
                # The skyness was passed as a single channel extra feature
                if "rendered_extras" in info:
                    skyness_rendered = info["rendered_extras"][0, :, :, 0]  # [H, W] - first channel contains skyness

                    # Compute binary cross-entropy between rendered skyness and ground truth mask
                    skyness_rendered_clamped = torch.clamp(skyness_rendered, 1e-7, 1 - 1e-7)
                    skyness_supervision_loss = -(
                        sky_mask_gt[0, ...] * torch.log(skyness_rendered_clamped) +
                        (1 - sky_mask_gt[0, ...]) * torch.log(1 - skyness_rendered_clamped)
                    ).mean()

                    loss += skyness_supervision_loss * cfg.skyness_supervision_lambda

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

            if cfg.skysphere_enabled:
                if cfg.skyness_reg > 0:
                    loss_components["skyness_entropy"] = skyness_entropy
                if cfg.skyness_supervision and "sky_mask" in data and "rendered_extras" in info:
                    loss_components["skyness_supervision"] = skyness_supervision_loss
                if cfg.skysphere_radius_reg > 0 and skysphere_radius_loss is not None:
                    loss_components["skysphere_radius"] = skysphere_radius_loss

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
                if cfg.skysphere_enabled and cfg.skyness_reg > 0:
                    skyness_probs = torch.sigmoid(self.splats["skyness"])
                    self.writer.add_scalar("train/skyness_supervision_loss", skyness_supervision_loss.item(), step)
                    self.writer.add_scalar("train/skyness_entropy", skyness_entropy.item(), step)
                    self.writer.add_scalar("train/skyness_sky_count", (skyness_probs > 0.75).sum().item(), step)
                    self.writer.add_scalar("train/skyness_world_count", (skyness_probs < 0.25).sum().item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
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
            for scheduler in schedulers:
                scheduler.step()


            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                # Add skyness-aware densification parameters
                if cfg.skysphere_enabled and "skyness" in info:
                    # Sky points (skyness > 0.75) should have different densification behavior
                    skyness_probs = info["skyness"]
                    info["is_sky"] = skyness_probs > 0.75
                    info["is_world"] = skyness_probs < 0.25
                    # Sky points should be pruned more aggressively if they're too close
                    info["skysphere_radius"] = self.scene_scale * cfg.skysphere_radius_multiplier

                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    epoch_ctx=epoch_ctx,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                    epoch_ctx=epoch_ctx,
                )
            else:
                assert_never(self.cfg.strategy)

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
                torch.save(
                    {
                        "step": step,
                        "splats": self.splats.state_dict(),
                    },
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
            renders, _, _, surf_normals, _, _, _ = self.rasterize_splats(
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
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        # Create detached copy of splats for viewer rendering
        viewer_splats = {k: v.clone().detach() for k, v in self.splats.items()}

        focal = float(K[0, 0] + K[1, 1])/2  # Use focal length from K matrix
        K_orig = list(self.parser.Ks_dict.values())[0]
        f_orig = float(K_orig[0, 0] + K_orig[1, 1])/2

        blur_mod = render_tab_state.blur_mod

        # Prepare override colors for colormapped visualization
        override_colors = None
        overmax_opacity = False

        if render_tab_state.render_mode == "max_sampling_rate" and "max_sampling_rate" in self.splats:
            max_sampling_rate = self.splats["max_sampling_rate"].detach()
            override_colors = scalar_to_colormap(
                max_sampling_rate,
                colormap=render_tab_state.colormap,
                inverse=render_tab_state.inverse,
                explicit_min=10,
                explicit_max=1000,
            ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]

        elif render_tab_state.render_mode == "accumulated_max_sampling_rate":
            # Use accumulated max_sampling_rate from epoch statistics if available
            if "epoch_stats" in self.strategy_state and hasattr(self.strategy_state["epoch_stats"], "max_sampling_rate"):
                accumulated_max_sampling = self.strategy_state["epoch_stats"].max_sampling_rate.clone().detach()
                override_colors = scalar_to_colormap(
                    accumulated_max_sampling,
                    colormap=render_tab_state.colormap,
                    inverse=render_tab_state.inverse,
                    explicit_min=10,
                    explicit_max=1000,
                ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]

        elif render_tab_state.render_mode == "sigma_smooth" and "max_sampling_rate" in self.splats:
            max_sampling_rate = self.splats["max_sampling_rate"].clone().detach()
            # Calculate smoothing sigma squared
            # изменения вблизи очень слабозаметны, хотя и применяются правильно
            sigma_smooth = torch.sqrt(calc_sigma_sq(max_sampling_rate, self.cfg.aa_smoothing_reg, focal, f_orig))
            scales = scaling_activation(viewer_splats["scales"])  # [N, 3]
            min_scales = scales[:, :2].min(dim=1).values  # [N]
            relative_change = sigma_smooth / min_scales  # [N]
            override_colors = scalar_to_colormap(
                relative_change,
                colormap=render_tab_state.colormap,
                inverse=render_tab_state.inverse,
                # explicit_min=0.001,
                # explicit_max=2.0
            ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]

        elif render_tab_state.render_mode == "n_cameras_visible":
            # Use n_cameras_visible_from from epoch statistics if available
            if "epoch_stats" in self.strategy_state and hasattr(self.strategy_state["epoch_stats"], "n_cameras_visible_from"):
                n_cameras = self.strategy_state["epoch_stats"].n_cameras_visible_from.clone().float()
                n_cameras[n_cameras == 0] = -len(self.trainset) # чтобы палитра начиналась с середины, а ноль выделялся цветом
                override_colors = scalar_to_colormap(
                    n_cameras,
                    colormap=render_tab_state.colormap,
                    inverse=render_tab_state.inverse,
                    explicit_min=-len(self.trainset),
                    explicit_max=len(self.trainset),
                ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]

        elif render_tab_state.render_mode == "skyness":
            # Visualize skyness probability
            skyness_probs = torch.sigmoid(viewer_splats["skyness"])
            override_colors = skyness_to_colormap(skyness_probs).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]

        elif render_tab_state.render_mode == "elongation":
            # Calculate elongation in log space (scales are stored in log form)
            log_scales = viewer_splats["scales"][..., :2]  # [N, 2] - 2DGS, only x,y scales (no activation)
            log_elongation_ratio = torch.abs(log_scales[:, 0] - log_scales[:, 1])  # [N] - abs(log_x - log_y) = log(max/min)
            elongation_ratio = torch.exp(log_elongation_ratio)  # Convert from log space to actual ratio
            override_colors = scalar_to_colormap(
                elongation_ratio,
                colormap=render_tab_state.colormap,
                inverse=render_tab_state.inverse,
                explicit_min=1.0,
                explicit_max=10.0,
            ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]
            # overmax_opacity = True  # Use maximum opacity for better visibility

        elif render_tab_state.render_mode == "grad2d_accum":
            # Visualize accumulated gradient magnitudes
            if "grad2d_abs" in self.strategy_state and self.strategy_state["grad2d_abs"] is not None:
                grad2d = self.strategy_state["grad2d_abs"].clone()
                count = self.strategy_state["count"].clone()
                # Average gradient per visibility count
                avg_grad = torch.where(count > 0, grad2d / count.clamp_min(1), torch.zeros_like(grad2d))
                override_colors = scalar_to_colormap(
                    avg_grad,
                    colormap=render_tab_state.colormap,
                    inverse=render_tab_state.inverse,
                    explicit_min=0.0,
                    explicit_max=self.cfg.grow_grad2d * 2,  # Scale relative to grow threshold
                ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]
            else:
                # Fallback if no gradient data available
                override_colors = torch.zeros((len(viewer_splats["means"]), 1, 3), device=self.device)
        
        elif render_tab_state.render_mode == "grad2d_count":
            # Visualize visibility count (how many times each gaussian was visible)
            if "count" in self.strategy_state and self.strategy_state["count"] is not None:
                count = self.strategy_state["count"].clone()
                override_colors = scalar_to_colormap(
                    count,
                    colormap=render_tab_state.colormap,
                    inverse=render_tab_state.inverse,
                    explicit_min=0,
                    explicit_max=len(self.trainset),  # Max is number of training views
                ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]
            else:
                # Fallback if no count data available
                override_colors = torch.zeros((len(viewer_splats["means"]), 1, 3), device=self.device)
        
        elif render_tab_state.render_mode == "gcr":
            # Visualize Gradient Consistency Ratio (GCR) from GDAGS
            if ("grad2d" in self.strategy_state and self.strategy_state["grad2d"] is not None and
                "grad2d_abs" in self.strategy_state and self.strategy_state["grad2d_abs"] is not None):
                grad2d = self.strategy_state["grad2d"].clone()
                grad2d_abs = self.strategy_state["grad2d_abs"].clone()
                count = self.strategy_state["count"].clone()
                # Average gradients per visibility count
                avg_grad = torch.where(count > 0, grad2d / count.clamp_min(1), torch.zeros_like(grad2d))
                avg_grad_abs = torch.where(count > 0, grad2d_abs / count.clamp_min(1), torch.zeros_like(grad2d_abs))
                # Compute GCR = grad / grad_abs
                gcr = (avg_grad + 1e-8) / (avg_grad_abs + 1e-8)
                gcr = torch.clamp(gcr, 0.0, 1.0)  # Clamp to [0, 1]
                override_colors = scalar_to_colormap(
                    gcr,
                    colormap=render_tab_state.colormap,
                    inverse=render_tab_state.inverse,
                    explicit_min=0.0,
                    explicit_max=1.0,
                ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]
            else:
                # Fallback if no gradient data available
                override_colors = torch.zeros((len(viewer_splats["means"]), 1, 3), device=self.device)
        
        elif render_tab_state.render_mode == "gdags_weight":
            # Visualize GDAGS weight: w = 0.8 + 25 * (1 - GCR)^15
            if ("grad2d" in self.strategy_state and self.strategy_state["grad2d"] is not None and
                "grad2d_abs" in self.strategy_state and self.strategy_state["grad2d_abs"] is not None):
                grad2d = self.strategy_state["grad2d"].clone()
                grad2d_abs = self.strategy_state["grad2d_abs"].clone()
                count = self.strategy_state["count"].clone()
                # Average gradients per visibility count
                avg_grad = torch.where(count > 0, grad2d / count.clamp_min(1), torch.zeros_like(grad2d))
                avg_grad_abs = torch.where(count > 0, grad2d_abs / count.clamp_min(1), torch.zeros_like(grad2d_abs))
                # Compute GCR = grad / grad_abs
                gcr = (avg_grad + 1e-8) / (avg_grad_abs + 1e-8)
                gcr = torch.clamp(gcr, 0.0, 1.0)  # Clamp to [0, 1]
                # Compute GDAGS weight
                weight = 0.8 + 25 * torch.pow(1 - gcr, 15)
                override_colors = scalar_to_colormap(
                    weight,
                    colormap=render_tab_state.colormap,
                    inverse=render_tab_state.inverse,
                    explicit_min=0.8,
                    explicit_max=25.8,
                ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]
            else:
                # Fallback if no gradient data available
                override_colors = torch.zeros((len(viewer_splats["means"]), 1, 3), device=self.device)
        
        elif render_tab_state.render_mode == "grad2d_gcr_combined":
            # Combined visualization: grad2d_abs determines intensity, gcr determines hue
            if ("grad2d" in self.strategy_state and self.strategy_state["grad2d"] is not None and
                "grad2d_abs" in self.strategy_state and self.strategy_state["grad2d_abs"] is not None):
                grad2d = self.strategy_state["grad2d"].clone()
                grad2d_abs = self.strategy_state["grad2d_abs"].clone()
                count = self.strategy_state["count"].clone()
                
                # Average gradients per visibility count
                avg_grad = torch.where(count > 0, grad2d / count.clamp_min(1), torch.zeros_like(grad2d))
                avg_grad_abs = torch.where(count > 0, grad2d_abs / count.clamp_min(1), torch.zeros_like(grad2d_abs))
                
                # Compute GCR = grad / grad_abs
                gcr = (avg_grad + 1e-8) / (avg_grad_abs + 1e-8)
                gcr = torch.clamp(gcr, 0.0, 1.0)  # Clamp to [0, 1]
                
                # Normalize grad2d_abs to [0, 1]
                grad_norm = torch.clamp(avg_grad_abs / (self.cfg.grow_grad2d * 2), 0.0, 1.0)
                
                # Create color mapping:
                # Low grad_norm → pastel blue (0.7, 0.85, 1.0)
                # High grad_norm + low gcr → pastel red (1.0, 0.7, 0.7)
                # High grad_norm + high gcr → pastel green (0.7, 1.0, 0.7)
                
                # Base pastel blue
                base_color = torch.tensor([0.3, 0.4, 1.0], device=self.device)
                # Target colors based on gcr
                red_color = torch.tensor([1.0, 0.3, 0.3], device=self.device)
                green_color = torch.tensor([0.3, 1.0, 0.3], device=self.device)
                
                # Interpolate between red and green based on gcr
                target_color = red_color * (1 - gcr).unsqueeze(-1) + green_color * gcr.unsqueeze(-1)
                
                # Interpolate between base and target based on grad_norm
                colors = base_color * (1 - grad_norm).unsqueeze(-1) + target_color * grad_norm.unsqueeze(-1)
                
                override_colors = colors.unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]
            else:
                # Fallback if no gradient data available
                override_colors = torch.zeros((len(viewer_splats["means"]), 1, 3), device=self.device)
        
        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            splats=viewer_splats,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            render_mode="RGB+ED",
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device) / 255.0,
            track_domination=True,
            distloss=render_tab_state.render_mode == "distort",
            override_colors=override_colors,
            overmax_opacity=overmax_opacity,
            rasterize_mode=render_tab_state.rasterize_mode,
            f_orig=f_orig,
            blur_mod=blur_mod,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(viewer_splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode in ("depth(expected)", "depth(dominating)"):
            if render_tab_state.render_mode == "depth(dominating)":
                depth = info["dominating_depths"][0, ..., None]
            else:
                depth = render_median[0, ..., 0:1]
            # normalize depth to [0, 1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
                depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
                depth_norm = torch.clip(depth_norm, 0, 1)
            else:
                depth_norm = normalize_robust(depth)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "normal":
            render_normals = render_normals[0, ..., 0:3] * 0.5 + 0.5  # normalize to [0, 1]
            renders = render_normals.cpu().numpy()
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        elif render_tab_state.render_mode == "domination":
            renders = (
                index_map_to_pseudocolor(info["dominating_gauss_ids"][0, ...])
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "distort":
            dist = render_distort[0, ..., 0:1]
            # normalize distortion to [0, 1]
            if render_tab_state.normalize_nearfar:
                # Use near/far plane for normalization
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
                dist_norm = (dist - near_plane) / (far_plane - near_plane + 1e-10)
                dist_norm = torch.clip(dist_norm, 0, 1)
            else:
                # Use robust normalization to exclude outliers
                dist_norm = normalize_robust(dist)
            if render_tab_state.inverse:
                dist_norm = 1 - dist_norm
            renders = (
                apply_float_colormap(dist_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        else:
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        return renders


def main(cfg: Config):
    runner = Runner(cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            if k in ckpt["splats"]:
                runner.splats[k].data = ckpt["splats"][k]
            elif k == "skyness" and k not in ckpt["splats"]:
                # Initialize skyness if not in checkpoint
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

