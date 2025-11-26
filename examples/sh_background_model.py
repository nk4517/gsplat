import torch
import torch.nn as nn
from typing import Optional

from adan import Adan


class SHBackgroundModel(nn.Module):
    """Spherical harmonics background model with learnable coefficients."""
    
    def __init__(
        self,
        sh_degree: int = 2,
        bg_base_weight: float = 0.005,
        bg_threshold: float = 0.05,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.sh_degree = sh_degree
        self.device = device
        self.bg_base_weight = bg_base_weight
        self.bg_threshold = bg_threshold
        
        # Number of SH coefficients for given degree
        self.n_coeffs = (sh_degree + 1) ** 2
        
        # Initialize SH coefficients as learnable parameters [K, 3]
        # Start with small random values
        sh_coeffs = torch.randn(self.n_coeffs, 3, device=device) * 0.1
        # Set DC component to mid-gray
        sh_coeffs[0] = 0.5
        
        self.sh_coeffs = nn.Parameter(sh_coeffs)
    
    def render(
        self,
        camtoworlds: torch.Tensor,  # [B, 4, 4]
        Ks: torch.Tensor,  # [B, 3, 3]
        width: int,
        height: int,
    ) -> torch.Tensor:
        """Render SH background for given camera parameters.
        
        Returns:
            RGB image [B, H, W, 3]
        """
        from gsplat.cuda._wrapper import spherical_harmonics

        batch_size = camtoworlds.shape[0]

        # Generate pixel coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=self.device, dtype=torch.float32),
            torch.arange(width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )

        # Pixel centers
        pixels_x = grid_x + 0.5  # [H, W]
        pixels_y = grid_y + 0.5  # [H, W]

        # Process each camera in batch
        renders = []
        for b in range(batch_size):
            K = Ks[b]  # [3, 3]
            c2w = camtoworlds[b]  # [4, 4]

            # Unproject pixels to camera rays
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            # Ray directions in camera space
            dirs_cam_x = (pixels_x - cx) / fx  # [H, W]
            dirs_cam_y = (pixels_y - cy) / fy  # [H, W]
            dirs_cam_z = torch.ones_like(dirs_cam_x)  # [H, W]

            # Stack to [H, W, 3]
            dirs_cam = torch.stack([dirs_cam_x, dirs_cam_y, dirs_cam_z], dim=-1)

            # Transform to world space
            R = c2w[:3, :3]  # [3, 3]
            dirs_world = torch.matmul(dirs_cam, R.T)  # [H, W, 3]

            # Normalize directions
            dirs_world = dirs_world / (torch.norm(dirs_world, dim=-1, keepdim=True) + 1e-8)

            # Flatten for SH evaluation
            dirs_flat = dirs_world.reshape(-1, 3)  # [H*W, 3]

            # Expand coefficients for all pixels
            coeffs_expanded = self.sh_coeffs.unsqueeze(0).expand(height * width, -1, -1)  # [H*W, K, 3]

            # Evaluate SH
            colors_flat = spherical_harmonics(
                self.sh_degree,
                dirs_flat,
                coeffs_expanded,
            )  # [H*W, 3]

            # Reshape back to image
            colors = colors_flat.reshape(height, width, 3)  # [H, W, 3]

            # Clamp to valid range
            colors = torch.clamp(colors, 0.0, 1.0)

            renders.append(colors)

        # Stack batch
        return torch.stack(renders, dim=0)  # [B, H, W, 3]

    def blend_with_sky(
        self,
        sky_colors: torch.Tensor,  # [B, H, W, 3]
        sky_wsum: torch.Tensor,    # [B, H, W, 1]
        sh_bg: torch.Tensor,       # [B, H, W, 3]
    ) -> torch.Tensor:
        """Blend SH background with sky colors using weighted mixing."""
        transition_range = self.bg_threshold - self.bg_base_weight
        bg_weight = self.bg_base_weight * torch.clamp((self.bg_threshold - sky_wsum) / transition_range, 0, None)
        total_weight = sky_wsum + bg_weight
        return (sky_colors + sh_bg * bg_weight) / (total_weight + 1e-8)
    
    def create_optimizer(self, lr: float = 1e-3) -> torch.optim.Optimizer:
        """Create optimizer for SH coefficients."""
        return Adan([self.sh_coeffs], lr=lr, eps=1e-15)