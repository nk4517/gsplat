import torch
import torch.nn as nn
import math
from typing import Dict
from gsplat.strategy.ops import scaling_inverse_activation, opacity_inverse_activation, opacity_activation, scaling_activation, rotation_activation
from examples.utils import knn
from examples.lib_skysphere import compute_skysphere_geometry, compute_skysphere_geometry_dog
from examples.sh_background_model import SHBackgroundModel

"""
!!!CRITICAL IMPLEMENTATION DETAILS - LLM MUST READ CAREFULLY!!!

This skysphere model uses QUATERNION PARAMETRIZATION instead of direct 3D positions:

1. PARAMETRIZATION:
   - Instead of storing means (3D positions) directly as learnable parameters
   - We store ONLY quaternions (4D rotation representations) 
   - 3D positions are COMPUTED from quaternions: means = radius * direction_from_quaternion
   - This ensures splats ALWAYS stay on the sphere surface at fixed radius
   
2. QUATERNION TO POSITION CONVERSION:
   - Quaternion encodes the splat orientation for rendering, so negative Z-axis (0, 0, -1) to the normal direction
   - Since normals point TO center (inward), positions are in OPPOSITE direction (outward)
   - Formula: Rotate (0, 0, -1) by quaternion, then multiply by radius
   - Direct formula used: dir = 2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 2(qx² + qy²) - 1
   - Final position: means = radius * directions
   
3. RENDERING MODE:
   - Uses RasterizationMode2DGS.WEIGHTED_SUM (NOT CLASSIC mode)
   - Opacities default to CONSTANT 1.0 (can be made trainable via trainable_opacities flag)
   - Trainable opacities allow individual splats to be dimmed during training (instead of shrinking to points via scale reduction)
   - Colors are direct RGB values (no SH, no view-dependency)
   - WEIGHTED_SUM directly sums weighted colors without alpha accumulation
   
4. OPTIMIZATION:
   - Only quaternions, scales, and colors are optimized
   - Means are NOT stored or optimized directly - they're computed from quaternions
   - Quaternion learning rate controls both orientation AND position on sphere
   - Radius is fixed (not learnable)

5. KEY INVARIANTS:
   - All splats remain at exactly radius distance from origin
   - Splat normals always point toward center (encoded in quaternion)
   - No means parameter exists - only quaternions

!!!FAILURE TO UNDERSTAND THIS WILL CAUSE INCORRECT MODIFICATIONS!!!
"""

# @torch.compile
def compute_means_from_quats(quats: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """Compute 3D positions from quaternions.
    
    The quaternion encodes the splat orientation for rendering:
    - It rotates Z-axis (0,0,1) to point TOWARD center (inward normal)
    - For rendering, this quaternion defines the splat's normal direction
    
    To compute splat position on sphere (opposite to normal):
    - We rotate (0,0,-1) instead of (0,0,1) by the same quaternion
    - This gives us the direction FROM center (outward)
    - Multiply by radius to get the position on sphere surface
    """
    qw = quats[:, 0]
    qx = quats[:, 1]
    qy = quats[:, 2]
    qz = quats[:, 3]
    
    # Direct formula for outward direction
    dir_x = 2 * (qx * qz - qw * qy)
    dir_y = 2 * (qy * qz + qw * qx)
    # Compute negated z directly: -[1 - 2(qx² + qy²)] = 2(qx² + qy²) - 1
    dir_z = 2 * (qx * qx + qy * qy) - 1
    
    directions = torch.stack([dir_x, dir_y, dir_z], dim=1)
    
    # Multiply by radius to get positions
    means = radius * directions
    
    return means


class SkysphereModelParametrized(nn.Module):
    """Skysphere model with means parametrized through quaternions.
    
    Instead of storing means directly, we use the existing quaternions (which already encoding orientation) to compute means as:
    means = radius * direction_from_quaternion
    
    This ensures splats stay on the sphere and their normals point to center.
    """
    
    def __init__(
        self,
        scene_scale: float,
        radius_multiplier: float = 20.0,
        trainable_opacities: bool = False,
        use_sh_background: bool = False,
        sh_degree: int = 4,
        sh_bg_base_weight: float = 0.01,
        sh_bg_threshold: float = 0.5,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.scene_scale = scene_scale
        self.radius_multiplier = radius_multiplier
        self.radius = scene_scale * radius_multiplier
        self.device = device
        self.trainable_opacities = trainable_opacities
        self.bg_base_weight = sh_bg_base_weight
        self.bg_threshold = sh_bg_threshold
        
        # Initialize as empty by default
        self.n_points = 0
        self.is_empty = True
        
        # Store radius as buffer
        self.register_buffer('radius_buffer', torch.tensor(self.radius, device=device))
        
        # Initialize params as ParameterDict for direct passing to strategy
        self.params = nn.ParameterDict()
        
        # Initialize SH background if enabled
        if use_sh_background:
            self.sh_background = SHBackgroundModel(
                sh_degree=sh_degree,
                device=device,
            )
            print(f"SH background initialized with degree {sh_degree}")
        else:
            self.sh_background = None
    
    def initialize_from_trainset(
        self,
        trainset,
        full_skysphere_N_points: int = 50_000,
        init_scale: float = 1.0,
    ):
        """Initialize skysphere from training set data."""
        # Initialize skysphere geometry
        points, colors, quats = compute_skysphere_geometry(
            trainset, self.radius, full_skysphere_N_points, self.device
        )
        
        if points is None:
            # No sky masks available, create empty skysphere
            self.n_points = 0
            self.is_empty = True
            return
        
        self.is_empty = False
        self.n_points = points.shape[0]
        
        # Initialize scales based on nearest neighbors
        dist2_avg = (knn(points, min(4, self.n_points))[:, 1:] ** 2).mean(dim=-1)
        dist_avg = torch.sqrt(dist2_avg)
        scales = scaling_inverse_activation(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)
        
        # Use simple RGB colors (no view-dependent effects for distant sky)
        
        # Register parameters in params dict - NO means, only quaternions
        self.params['scales'] = nn.Parameter(scales)
        self.params['quats'] = nn.Parameter(quats)
        # Store colors as logits for unconstrained optimization
        colors_clamped = colors.clamp(1e-5, 1 - 1e-5)  # Avoid inf in logit
        colors_logit = opacity_inverse_activation(colors_clamped)
        self.params['colors'] = nn.Parameter(colors_logit)
        
        # Update radius buffer if needed
        self.radius_buffer = torch.tensor(self.radius, device=self.device)
        
        opacities_init = torch.ones((self.n_points,), device=self.device)
        opacities = opacity_inverse_activation(opacities_init.clamp(1e-5, 1 - 1e-5))
        # Always store opacities in params dict, trainable_opacities controls requires_grad
        self.params['opacities'] = nn.Parameter(opacities, requires_grad=self.trainable_opacities)
        
    def get_splats(self) -> Dict[str, torch.Tensor]:
        """Get splat parameters in format compatible with rasterization."""
        if self.is_empty:
            return {}
        
        rs = {
            "scales": self.params['scales'],
            "quats": self.params['quats'],
            "opacities": self.params['opacities'],
            "colors": self.params['colors'],  # Convert from logit to RGB
        }
        rs["means"] = compute_means_from_quats(rotation_activation(rs["quats"]), self.radius_buffer)
        return rs
    
    def compose_with_background(
        self,
        sky_colors: torch.Tensor,
        sky_wsum: torch.Tensor,
        camtoworlds: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
    ) -> torch.Tensor:
        """Compose sky colors with SH background if enabled.
        
        Args:
            sky_colors: Rendered sky colors [B, H, W, 3]
            sky_wsum: Sky weight sum [B, H, W, 1]
            camtoworlds: Camera to world matrices [B, 4, 4]
            Ks: Camera intrinsics [B, 3, 3]
            width: Image width
            height: Image height
            
        Returns:
            Final composed colors [B, H, W, 3]
        """
        if self.sh_background is None:
            return sky_colors
        
        sh_bg = self.sh_background.render(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
        )
        
        # transition_range = self.bg_threshold - self.bg_base_weight
        # bg_weight = self.bg_base_weight * torch.clamp((self.bg_threshold - sky_wsum) / transition_range, 0, None)
        # total_weight = sky_wsum + bg_weight
        # return (sky_colors + sh_bg * bg_weight) / (total_weight + 1e-8)
        bg_weight = self.bg_base_weight  # всегда 0.1
        total_weight = sky_wsum + bg_weight
        return (sky_colors + sh_bg * bg_weight) / (total_weight + 1e-8)

    def create_optimizers(
            self,
            sh_background_lr=1e-3,
            batch_size: int = 1,
            sparse_grad: bool = False) -> Dict[str, torch.optim.Optimizer]:
        """Create optimizers for skysphere parameters."""
        if self.is_empty:
            return {}
        
        from adan import Adan
        from torch.optim import Adam
        
        # Learning rates for skysphere (no means to optimize)
        lr_config = {
            "scales": 3e-2,
            "quats": 2.5e-4,  # Quaternions control both orientation AND position
            "colors": 2.5e-2,
        }
        
        # Add opacities if trainable
        if self.trainable_opacities:
            lr_config["opacities"] = 2.5e-3
        
        optimizers = {}
        for name, lr in lr_config.items():
            param: nn.Parameter = self.params[name]
            if not param.requires_grad:
                continue
            
            # Scale learning rate based on batch size
            scaled_lr = lr * math.sqrt(batch_size)
            scaled_eps = 1e-15 / math.sqrt(batch_size)
            
            if sparse_grad:
                scaled_betas = (1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.99))
                optimizer = torch.optim.SparseAdam(
                    [{"params": param, "lr": scaled_lr}],
                    eps=scaled_eps,
                    betas=scaled_betas,
                )
            else:
                scaled_betas = (1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.99))
                optimizer = Adan(
                    [{"params": param, "lr": lr}],
                    eps=scaled_eps,
                    # betas=(0.98/8, 0.92/8, 0.99/8),
                    fused=True,
                )
            
            optimizers[name] = optimizer
        
        # Add SH background optimizer if present
        if self.sh_background is not None:
            optimizers['sh_background'] = self.sh_background.create_optimizer(lr=sh_background_lr)
        
        return optimizers

    def save_checkpoint(self, path: str):
        """Save model checkpoint to file."""
        if self.is_empty:
            # Save minimal checkpoint for empty skysphere
            checkpoint = {
                'is_empty': True,
                'scene_scale': self.scene_scale,
                'radius_multiplier': self.radius_multiplier,
                'trainable_opacities': self.trainable_opacities,
                'bg_base_weight': self.bg_base_weight,
                'bg_threshold': self.bg_threshold,
                'has_sh_background': self.sh_background is not None,
            }
        else:
            checkpoint = {
                'is_empty': False,
                'scene_scale': self.scene_scale,
                'radius_multiplier': self.radius_multiplier,
                'trainable_opacities': self.trainable_opacities,
                'bg_base_weight': self.bg_base_weight,
                'bg_threshold': self.bg_threshold,
                'n_points': self.n_points,
                'state_dict': self.state_dict(),
                'has_sh_background': self.sh_background is not None,
            }
            if self.sh_background is not None:
                checkpoint['sh_background_state'] = self.sh_background.state_dict()
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, frozen=False):
        """Load model checkpoint from file."""
        checkpoint = torch.load(path, map_location=self.device)

        self.scene_scale = checkpoint['scene_scale']
        self.radius_multiplier = checkpoint['radius_multiplier']
        self.radius = self.scene_scale * self.radius_multiplier
        self.is_empty = checkpoint['is_empty']
        self.trainable_opacities = checkpoint.get('trainable_opacities', False)
        self.bg_base_weight = checkpoint.get('bg_base_weight', 0.01)
        self.bg_threshold = checkpoint.get('bg_threshold', 0.5)
        
        # Create SH background BEFORE load_state_dict (it's a submodule, its state is in state_dict)
        if checkpoint.get('has_sh_background', False):
            if self.sh_background is None:
                sh_coeffs_shape = checkpoint.get('sh_background_state', {}).get('sh_coeffs', torch.zeros(25)).shape[0]
                sh_degree = int(sh_coeffs_shape ** 0.5) - 1
                self.sh_background = SHBackgroundModel(
                    sh_degree=sh_degree,
                    device=self.device,
                )
        if not self.is_empty:
            self.n_points = checkpoint['n_points']
            n = self.n_points
            self.params['scales'] = nn.Parameter(torch.zeros(n, 3, device=self.device))
            self.params['quats'] = nn.Parameter(torch.zeros(n, 4, device=self.device))
            self.params['colors'] = nn.Parameter(torch.zeros(n, 3, device=self.device))
            self.params['opacities'] = nn.Parameter(torch.zeros(n, device=self.device))
            self.load_state_dict(checkpoint['state_dict'])
            if frozen:
                self.params["scales"].requires_grad = not frozen
                self.params["quats"].requires_grad = not frozen
                self.params["colors"].requires_grad = not frozen
                self.params["opacities"].requires_grad = self.trainable_opacities and not frozen
        
        # Load SH background state if saved separately (new format)
        if self.sh_background is not None and 'sh_background_state' in checkpoint:
            self.sh_background.load_state_dict(checkpoint['sh_background_state'])
            if frozen:
                self.sh_background.sh_coeffs.requires_grad = False
