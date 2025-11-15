from gsplat2d.project_gaussians import project_gaussians
from gsplat2d.rasterize import rasterize_gaussians
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def loss_fn(pred, target, loss_type="L2", lambda_value=0.7):
    """Loss function for 2D gaussians."""
    if loss_type == "L2":
        return F.mse_loss(pred, target)
    elif loss_type == "L1":
        return F.l1_loss(pred, target)
    elif loss_type == "L1+SSIM":
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
        l1_loss = F.l1_loss(pred, target)
        ssim_loss = 1.0 - ssim(pred.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2))
        return l1_loss * (1.0 - lambda_value) + ssim_loss * lambda_value
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class Gaussian2D(nn.Module):
    """2D Gaussian model for per-camera rendering."""
    
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        
        self.B_SIZE = 16
        
        self.device = kwargs["device"]
        
        self.last_size = (self.H, self.W)
        
        # Initialize parameters
        w_init = torch.rand(self.init_num_points, 1, device=self.device) * self.W
        h_init = torch.rand(self.init_num_points, 1, device=self.device) * self.H
        self.means = nn.Parameter(torch.cat((w_init, h_init), dim=1))
        
        self.cov2d = nn.Parameter(torch.rand(self.init_num_points, 3, device=self.device))
        d = 3
        self.rgbs = nn.Parameter(torch.zeros(self.init_num_points, d, device=self.device))
        
        self.means.requires_grad = True
        self.cov2d.requires_grad = True
        self.rgbs.requires_grad = True
        
        # Create optimizer
        if kwargs.get("opt_type") == "adam":
            self.optimizer = torch.optim.Adam([
                {'params': self.rgbs, 'lr': kwargs["lr"]},
                {'params': self.means, 'lr': kwargs["lr"] * 2},
                {'params': self.cov2d, 'lr': kwargs["lr"] * 5}
            ])
        else:
            from adan import Adan
            self.optimizer = Adan([
                {'params': self.rgbs, 'lr': kwargs["lr"]},
                {'params': self.means, 'lr': kwargs["lr"] * 2},
                {'params': self.cov2d, 'lr': kwargs["lr"] * 5}
            ], fused=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=70000, gamma=0.7)
    
    def forward(self):
        (
            xys,
            radii,
            conics,
            num_tiles_hit,
        ) = project_gaussians(
            self.cov2d,
            self.means,
            self.H,
            self.W,
            self.B_SIZE,
        )
        out_img, dx, dy, dxy = rasterize_gaussians(
                xys,
                radii,
                conics,
                num_tiles_hit,
                self.rgbs,
                self.H,
                self.W,
                self.B_SIZE,
            )
        
        out_img = torch.clamp(out_img[..., :3], 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        
        # Gradients can be negative, don't clamp them
        dx = dx[..., :3]
        dx = dx.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        
        dy = dy[..., :3]
        dy = dy.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        
        dxy = dxy[..., :3]
        dxy = dxy.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        
        return {"render": out_img, "dx": dx, "dy": dy, "dxy": dxy}
    
    def train_iter(self, gt_image):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        
        self.scheduler.step()
        return loss, psnr
    
    def save_checkpoint(self, path: str):
        """Save model parameters to file."""
        checkpoint = {
            'means': self.means.data,
            'cov2d': self.cov2d.data,
            'rgbs': self.rgbs.data,
            'H': self.H,
            'W': self.W,
            'num_points': self.init_num_points,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model parameters from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.means.data = checkpoint['means']
        self.cov2d.data = checkpoint['cov2d']
        self.rgbs.data = checkpoint['rgbs']
        self.H = checkpoint['H']
        self.W = checkpoint['W']
        self.init_num_points = checkpoint['num_points']