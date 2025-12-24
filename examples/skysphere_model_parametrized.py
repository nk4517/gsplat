import torch
import torch.nn as nn
import math
from typing import Dict
from gsplat.strategy.ops import scaling_inverse_activation
from examples.utils import knn
from examples.lib_skysphere import compute_skysphere_geometry, compute_skysphere_geometry_dog

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
   - Opacities are CONSTANT 1.0 (not learnable, not activated)
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
        device: str = "cuda",
    ):
        super().__init__()
        
        self.scene_scale = scene_scale
        self.radius_multiplier = radius_multiplier
        self.radius = scene_scale * radius_multiplier
        self.device = device
        
        # Initialize as empty by default
        self.n_points = 0
        self.is_empty = True
        
        # Store radius as buffer
        self.register_buffer('radius_buffer', torch.tensor(self.radius, device=device))
    
    def initialize_from_trainset(
        self,
        trainset,
        num_points: int = 50_000,
        init_opacity: float = 0.1,
        init_scale: float = 1.0,
        use_dog: bool = False,
        dog_sigma: float = 2.0,
        dog_k: float = 3.6,
    ):
        """Initialize skysphere from training set data."""
        # Initialize skysphere geometry
        if use_dog:
            points, colors, quats = compute_skysphere_geometry_dog(
                trainset, self.radius, num_points, self.device, dog_sigma, dog_k
            )
        else:
            points, colors, quats = compute_skysphere_geometry(
                trainset, self.radius, num_points, self.device
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
        
        # Register parameters - NO means, only quaternions
        self.scales = nn.Parameter(scales)
        self.quats = nn.Parameter(quats)
        # Store colors as logits for unconstrained optimization
        colors_clamped = colors.clamp(1e-5, 1 - 1e-5)  # Avoid inf in logit
        colors_logit = torch.logit(colors_clamped)
        self.colors = nn.Parameter(colors_logit)
        
        # Update radius buffer if needed
        self.radius_buffer = torch.tensor(self.radius, device=self.device)
        
        # Opacities are constant 1.0 for WEIGHTED_SUM mode
        self.register_buffer('opacities', torch.ones((self.n_points,), device=self.device))
        
    
    def get_splats(self) -> Dict[str, torch.Tensor]:
        """Get splat parameters in format compatible with rasterization."""
        if self.is_empty:
            return {}
        
        return {
            "means": compute_means_from_quats(self.quats, self.radius_buffer),
            "scales": self.scales,
            "quats": self.quats,
            "opacities": self.opacities,
            "colors": torch.sigmoid(self.colors),  # Convert from logit to RGB
        }
    
    def create_optimizers(self, batch_size: int = 1, sparse_grad: bool = False) -> Dict[str, torch.optim.Optimizer]:
        """Create optimizers for skysphere parameters."""
        if self.is_empty:
            return {}
        
        from adan import Adan
        from torch.optim import Adam
        
        # Learning rates for skysphere (no means to optimize)
        lr_config = {
            "scales": 5e-2,
            "quats": 5e-4,  # Quaternions control both orientation AND position
            "colors": 2.5e-2,
        }
        
        optimizers = {}
        for name, lr in lr_config.items():
            param = getattr(self, name)
            
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
                optimizer = Adan(
                    [{"params": param, "lr": lr}],
                    eps=scaled_eps,
                    fused=True,
                )
            
            optimizers[name] = optimizer
        
        return optimizers

    def save_checkpoint(self, path: str):
        """Save model checkpoint to file."""
        if self.is_empty:
            # Save minimal checkpoint for empty skysphere
            checkpoint = {
                'is_empty': True,
                'scene_scale': self.scene_scale,
                'radius_multiplier': self.radius_multiplier,
            }
        else:
            checkpoint = {
                'is_empty': False,
                'scene_scale': self.scene_scale,
                'radius_multiplier': self.radius_multiplier,
                'n_points': self.n_points,
                'state_dict': self.state_dict(),
            }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint from file."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.scene_scale = checkpoint['scene_scale']
        self.radius_multiplier = checkpoint['radius_multiplier']
        self.radius = self.scene_scale * self.radius_multiplier
        self.is_empty = checkpoint['is_empty']
        
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
            self.load_state_dict(checkpoint['state_dict'])