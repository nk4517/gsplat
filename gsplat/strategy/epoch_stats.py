from typing import Any, Dict, Optional
import torch
from torch import Tensor


@torch.no_grad()
def _accumulate_by_ids(
        target_tensor: torch.Tensor,
        gs_ids: torch.Tensor,
        values: torch.Tensor,
        packed: bool,
):
    """Generic accumulation method that handles both packed and non-packed cases.

    Args:
        target_tensor: Tensor to accumulate into
        gs_ids: Gaussian IDs for indexing
        values: Values to accumulate
        packed: Whether the data is in packed format
    """
    if packed:
        # Direct index_add for packed case
        target_tensor.index_add_(0, gs_ids, values)
    else:
        # For non-packed case, need to handle per-camera data
        if values.dim() == 2:  # [C, N] format
            # Sum across cameras for each gaussian
            values_sum = values.sum(dim=0)  # [N]
            target_tensor += values_sum
        else:  # [N] format
            target_tensor += values


class EpochStatistics:
    """Manages epoch-based statistics for Gaussian Splatting optimization."""
    
    def __init__(self, n_gaussian: int, device: torch.device):
        """Initialize epoch statistics.
        
        Args:
            n_gaussian: Number of Gaussians
            device: Device to store tensors on
        """
        self.n_cameras_visible_from = torch.zeros(n_gaussian, dtype=torch.int32, device=device)
        self.max_touchedPct = torch.zeros(n_gaussian, device=device)
        self.max_dominatedPct = torch.zeros(n_gaussian, device=device)
        self.n_touched_accum = torch.zeros(n_gaussian, dtype=torch.int32, device=device)
        self.n_dominated_accum = torch.zeros(n_gaussian, dtype=torch.int32, device=device)
        self.max_sampling_rate = torch.zeros(n_gaussian, device=device)  # Initialize with zero for maximum tracking
        
        # For gradient accumulation (used by both strategies)
        self.grad2d = torch.zeros(n_gaussian, device=device)
        self.grad2d_abs = torch.zeros(n_gaussian, device=device)
        self.count = torch.zeros(n_gaussian, device=device)
        self.importance = torch.zeros(n_gaussian, device=device)
        self.radii = None  # Optional, initialized when needed
 
    
    @torch.no_grad()
    def update_domination_stats(
            self,
            n_touched: torch.Tensor,
            n_dominated: torch.Tensor,
            width: int, height: int,
            gs_ids: torch.Tensor, packed: bool = False
    ):
        """Update domination statistics for the current camera view.
        
        Args:
            gs_ids: Gaussian IDs that were rendered
            n_touched: Number of pixels touched by each gaussian
            n_dominated: Number of pixels dominated by each gaussian
            width: Image width
            height: Image height
            packed: Whether the data is in packed format
        """
        total_px = width * height
        device = self.n_cameras_visible_from.device
        n_gaussian = self.n_cameras_visible_from.shape[0]
        
        # Create update filter based on which gaussians are being updated
        update_filter = torch.zeros(n_gaussian, dtype=torch.bool, device=device)
        update_filter[gs_ids] = True
        
        # Calculate percentages of touched camera space for updated gaussians
        ntoched_upd = torch.zeros(n_gaussian, dtype=torch.int32, device=device)
        ndom_upd = torch.zeros(n_gaussian, dtype=torch.int32, device=device)
        
        # Handle both packed and full-size tensors based on packed flag
        if packed:
            # Packed format: n_touched/n_dominated contain only values for rendered gaussians
            ntoched_upd[gs_ids] = n_touched
            ndom_upd[gs_ids] = n_dominated
        else:
            # Non-packed format: n_touched/n_dominated are full-size, index by gs_ids
            ntoched_upd[gs_ids] = n_touched[gs_ids]
            ndom_upd[gs_ids] = n_dominated[gs_ids]
        
        # Update epoch max percentages
        self.max_touchedPct[update_filter] = torch.max(
            self.max_touchedPct[update_filter], 
            ntoched_upd[update_filter] / total_px
        )
        self.max_dominatedPct[update_filter] = torch.max(
            self.max_dominatedPct[update_filter], 
            ndom_upd[update_filter] / total_px
        )
        
        # Increment counter of cameras from which each gaussian was visible in this epoch
        if packed:
            visible_mask = torch.zeros(n_gaussian, dtype=torch.bool, device=device)
            visible_mask[gs_ids] = n_touched > 0
            self.n_cameras_visible_from[visible_mask] += 1
        else:
            self.n_cameras_visible_from[n_touched > 0] += 1
        
        # Accumulate touched and dominated counts for this epoch
        _accumulate_by_ids(self.n_touched_accum, gs_ids, n_touched, packed)
        _accumulate_by_ids(self.n_dominated_accum, gs_ids, n_dominated, packed)

    @torch.no_grad()
    def update_max_sampling_rate(
            self,
            sampling_rates: torch.Tensor,
            visible_mask: torch.Tensor,
    ):
        """Update maximum sampling rate for each gaussian across all views.
        
        Args:
            sampling_rates: [C, N] or [N] - Sampling rates (f/d) for gaussians
            visible_mask: [C, N] or [N] - Boolean mask of visible gaussians
        """
        if sampling_rates.dim() == 2:  # [C, N] format
            # For each gaussian, find maximum sampling rate across all cameras where it's visible
            for c in range(sampling_rates.shape[0]):
                visible = visible_mask[c]
                if visible.any():
                    self.max_sampling_rate[visible] = torch.maximum(
                        self.max_sampling_rate[visible],
                        sampling_rates[c, visible]
                    )
        else:  # [N] format
            if visible_mask.any():
                self.max_sampling_rate[visible_mask] = torch.maximum(
                    self.max_sampling_rate[visible_mask],
                    sampling_rates[visible_mask]
                )
    
    @torch.no_grad()
    def update_gradient_stats(
            self,
            grads: torch.Tensor,
            grads_abs: torch.Tensor,
            importance_grads: torch.Tensor,
            gs_ids: torch.Tensor,
            radii: Optional[torch.Tensor] = None,
            width: Optional[int] = None,
            height: Optional[int] = None,
    ):
        """Update gradient accumulation statistics.
        
        Args:
            grads: Gradient norms for each gaussian
            grads_abs: Absolute gradient norms for each gaussian
            importance_grads: Importance gradients (vG^2) for each gaussian
            gs_ids: Gaussian IDs that were rendered
            radii: Optional radii values for scale tracking
            width: Optional image width for radii normalization
            height: Optional image height for radii normalization
        """
        self.grad2d.index_add_(0, gs_ids, grads)
        self.grad2d_abs.index_add_(0, gs_ids, grads_abs)
        self.importance.index_add_(0, gs_ids, importance_grads)
        self.count.index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
        
        if radii is not None and width is not None and height is not None:
            if self.radii is None:
                self.radii = torch.zeros_like(self.grad2d)
            # Should be ideally using scatter max
            self.radii[gs_ids] = torch.maximum(
                self.radii[gs_ids],
                # normalize radii to [0, 1] screen space
                radii / float(max(width, height)),
            )
   

    @torch.no_grad()
    def update_from_info(
            self,
            info: Dict[str, Any],
            params: Dict[str, Any],
            key_for_gradient: str = "gradient_2dgs",
            packed: bool = False,
    ):
        """Update statistics from rasterization info dict.
        
        Args:
            info: Info dict from rasterization containing gradients and domination stats
            params: Parameters dictionary containing means and other gaussian parameters
            key_for_gradient: Which gradient key to use ("gradient_2dgs" or "means2d")
            packed: Whether the data is in packed format
        """
        device = self.n_cameras_visible_from.device
        n_gaussian = self.n_cameras_visible_from.shape[0]
        
        # Update domination statistics if available
        if "n_touched" in info and "n_dominated" in info:
            gs_ids = info.get("gaussian_ids")
            packed_data = gs_ids is not None and len(info["n_touched"]) == len(gs_ids)
            self.update_domination_stats(
                n_touched=info["n_touched"],
                n_dominated=info["n_dominated"],
                width=info.get("width", 1920),  # Default if not provided
                height=info.get("height", 1080),  # Default if not provided
                gs_ids=gs_ids if gs_ids is not None else torch.arange(n_gaussian, device=device),
                packed=packed_data
            )
        
        # Update gradient statistics if gradients are available
        if key_for_gradient in info and info[key_for_gradient].grad is not None:
            if key_for_gradient == "gradient_2dgs":
                # For gradient_2dgs, extract appropriate elements
                gradient_2dgs = info[key_for_gradient].grad.clone()
                grads = gradient_2dgs[..., :2]
                grads_abs = gradient_2dgs[..., 2:4]
                importance_grads = gradient_2dgs[..., 4:5]
            else:
                # For means2d (not currently used in 2DGS)
                raise NotImplementedError(f"Gradient key {key_for_gradient} not implemented")
            
            # Normalize gradients
            width = info.get("width", 1920)
            height = info.get("height", 1080)
            n_cameras = info.get("n_cameras", 1)
            grads[..., 0] *= width / 2.0 * n_cameras
            grads[..., 1] *= height / 2.0 * n_cameras
            grads_abs[..., 0] *= width / 2.0 * n_cameras
            grads_abs[..., 1] *= height / 2.0 * n_cameras
            
            # Get gaussian IDs and filter by visibility
            if packed:
                # Packed format: gradients already correspond to visible gaussians
                gs_ids = info["gaussian_ids"]  # [nnz]
                radii = info.get("radii")
                if radii is not None:
                    radii = radii.max(dim=-1).values  # [nnz]
            else:
                # Non-packed format: need to filter by visibility
                radii = info.get("radii")
                if radii is not None:
                    sel = (radii > 0.0).all(dim=-1)  # [C, N]
                    gs_ids = torch.where(sel)[1]  # [nnz]
                    grads = grads[sel]  # [nnz, 2]
                    grads_abs = grads_abs[sel]  # [nnz, 2]
                    importance_grads = importance_grads[sel]  # [nnz, 1]
                    radii = radii[sel].max(dim=-1).values  # [nnz]
                else:
                    gs_ids = torch.arange(n_gaussian, device=device)
                    radii = None
            
            # Update gradient statistics
            self.update_gradient_stats(
                grads=grads.norm(dim=-1),
                grads_abs=grads_abs.norm(dim=-1),
                importance_grads=importance_grads.squeeze(-1),
                gs_ids=gs_ids,
                radii=radii,
                width=width if radii is not None else None,
                height=height if radii is not None else None,
            )
        
        # Update sampling rate statistics if available
        if "camtoworlds" in info and "Ks" in info:
            # Compute euclidean distances from camera to gaussians
            camtoworlds = info["camtoworlds"]
            Ks = info["Ks"]
            
            # Get means from params
            means = params["means"]
            
            # Euclidean distance from camera to gaussian center
            cam_positions = camtoworlds[:, :3, 3].unsqueeze(1)  # [C, 1, 3]
            means = means.unsqueeze(0)  # [1, N, 3]
            distances = torch.norm(cam_positions - means, dim=2)  # [C, N]
            
            # Extract focal lengths from camera intrinsics
            # Use average of fx and fy
            focals = (Ks[:, 0, 0] + Ks[:, 1, 1]) / 2.0  # [C]
            # Compute sampling rates f/d
            sampling_rates = focals.unsqueeze(1) / distances  # [C, N]
            
            # Update max_sampling_rate tracking
            # Use n_touched for visibility - accounts for transparency and actual pixel coverage
            if "n_touched" in info:
                visible_mask = info["n_touched"] > 0  # [C, N] or [N] - gaussians that touched at least one pixel
            else:
                # Fallback to radii if n_touched not available
                visible_mask = info["radii"][..., 0] > 0
            self.update_max_sampling_rate(sampling_rates, visible_mask)
    
    @torch.no_grad()
    def reset(self):
        """Reset all statistics to zero."""
        self.n_cameras_visible_from = self.n_cameras_visible_from.contiguous()
        self.n_cameras_visible_from.zero_()

        self.max_touchedPct = self.max_touchedPct.contiguous()
        self.max_touchedPct.zero_()

        self.max_dominatedPct = self.max_dominatedPct.contiguous()
        self.max_dominatedPct.zero_()

        self.n_touched_accum = self.n_touched_accum.contiguous()
        self.n_touched_accum.zero_()

        self.n_dominated_accum = self.n_dominated_accum.contiguous()
        self.n_dominated_accum.zero_()

        self.max_sampling_rate = self.max_sampling_rate.contiguous()
        self.max_sampling_rate.zero_()  # Reset to zero for maximum tracking
        
        # Reset gradient stats
        self.grad2d.zero_()
        self.grad2d_abs.zero_()
        self.count.zero_()
        self.importance.zero_()
        if self.radii is not None:
            self.radii.zero_()


class EpochFlag(Enum):
    """Flags indicating the position within an epoch."""
    START = "start"
    CONTINUE = "continue"
    END = "end"
    START_AND_END = "start_and_end"


@dataclass
class EpochContext:
    """Context information about the current epoch."""
    i_epoch: int = 0
    epoch_flag: Optional[EpochFlag] = None
    i: int = 0  # index within epoch
    epoch_len: int = 0  # total length of epoch

    @property
    def epoch_end(self) -> bool:
        """Check if this is the end of an epoch."""
        return self.epoch_flag in (EpochFlag.END, EpochFlag.START_AND_END)

    @property
    def epoch_start(self) -> bool:
        """Check if this is the start of an epoch."""
        return self.epoch_flag in (EpochFlag.START, EpochFlag.START_AND_END)


def training_data_generator(trainloader, pbar):
    """Generator that yields training data with epoch boundary flags."""
    i_epoch = 0
    pbar_iter = pbar.__iter__()
    while True:
        epoch_len = len(trainloader)
        for i, data in enumerate(trainloader):
            try:
                step = next(pbar_iter)
            except StopIteration:
                return

            if epoch_len == 1:
                epoch_flag = EpochFlag.START_AND_END
            elif i == 0:
                epoch_flag = EpochFlag.START
            elif i == epoch_len - 1:
                epoch_flag = EpochFlag.END
            else:
                epoch_flag = EpochFlag.CONTINUE

            epoch_ctx = EpochContext(
                i_epoch=i_epoch,
                epoch_flag=epoch_flag,
                i=i,
                epoch_len=epoch_len
            )

            yield step, data, epoch_ctx

        i_epoch += 1
