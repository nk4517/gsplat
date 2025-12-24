from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
import torch


"""
nnz — стандартная аббревиатура для "number of non-zero elements" (количество ненулевых элементов) в разреженных матрицах и тензорах.

В gsplat промежуточные результаты в meta словаре содержат:

gaussian_ids - индексы гауссианов (для nnz, иначе None)
camera_ids - индексы камер (для nnz, иначе None)
isect_ids - индексы пересечений (гауссиан, тайл)
flatten_ids - линейные индексы для развёртки
Для прямого доступа к парам (камера, гауссиан) в packed режиме используются camera_ids и gaussian_ids из возвращаемого meta словаря.


# После рендеринга
colors, alphas, meta = rasterization(..., packed=True)

# Индексы активных пар
camera_indices = meta['camera_ids']    # [nnz]
gaussian_indices = meta['gaussian_ids'] # [nnz]
"""

"""
ВАЖНО ДЛЯ LLM: Семантика touched и dominated статистик

n_touched и n_dominated в packed формате:
- Размер [nnz], где каждый элемент соответствует паре (камера, гауссиан)
- n_touched[i] - количество пикселей камеры cam_ids[i], лучи которых затронули гауссиан gs_ids[i]
  (луч может НЕ затронуть гауссиан если прошёл далеко от него, или если гауссиан загорожен другими и на него не хватило прозрачности)
- n_dominated[i] - количество пикселей камеры cam_ids[i], где гауссиан gs_ids[i] съел самую большую долю прозрачности среди всех гауссианов

touched_pct и dominated_pct:
- Это проценты от общего количества пикселей КОНКРЕТНОЙ камеры (width * height)
- Для каждой пары (камера, гауссиан) процент считается независимо
- max_touchedPct[gs_id] хранит максимальный процент по всем камерам, из которых был виден гауссиан gs_id
- max_dominatedPct[gs_id] аналогично

n_cameras_visible_from:
- Считает количество уникальных камер, из которых гауссиан был виден (n_touched > 0) за всю эпоху
- В батче с несколькими камерами каждая уникальная камера считается отдельно

n_touched_accum и n_dominated_accum:
- Накапливают общее количество пикселей за эпоху для каждого гауссиана
- При батчевой обработке суммируются значения от всех пар (камера, гауссиан) с одинаковым gs_id
"""

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


@torch.no_grad()
def _accumulate_max(
        target_tensor: torch.Tensor,
        gs_ids: torch.Tensor | None,
        values: torch.Tensor,
        packed: bool,
):
    """Generic maximum accumulation method that handles both packed and non-packed cases.

    Args:
        target_tensor: Tensor to update with maximum values [N]
        gs_ids: Gaussian IDs for indexing (for packed format) [nnz]
        values: Values to take maximum of [nnz] for packed, [C, N] or [N] for non-packed
        packed: Whether the data is in packed format
    """
    if packed:
        # Use index_reduce with 'amax' for packed case
        target_tensor.index_reduce_(0, gs_ids, values, 'amax', include_self=True)
    else:
        # For non-packed case, handle per-camera data
        if values.dim() == 2:  # [C, N] format
            # Maximum across cameras for each gaussian
            max_values = values.max(dim=0).values  # [N]
            target_tensor.copy_(torch.maximum(target_tensor, max_values))
        else:  # [N] format
            target_tensor.copy_(torch.maximum(target_tensor, values))


@torch.no_grad()
def _accumulate_count(
        target_tensor: torch.Tensor,
        gs_ids: torch.Tensor | None,
        mask: torch.Tensor,
        packed: bool,
):
    """Generic count accumulation method that handles both packed and non-packed cases.

    Args:
        target_tensor: Tensor to accumulate counts into [N]
        gs_ids: Gaussian IDs for indexing (for packed format) [nnz]
        mask: Boolean or integer mask to count [nnz] for packed, [C, N] or [N] for non-packed
        packed: Whether the data is in packed format
    """
    if packed:
        # Convert boolean mask to int if needed
        count_values = mask.int() if mask.dtype == torch.bool else mask
        target_tensor.index_add_(0, gs_ids, count_values)
    else:
        # For non-packed case, handle per-camera data
        if mask.dim() == 2:  # [C, N] format
            # Sum counts across cameras for each gaussian
            count_sum = mask.sum(dim=0)  # [N]
            if mask.dtype == torch.bool:
                count_sum = count_sum.int()
            target_tensor += count_sum
        else:  # [N] format
            if mask.dtype == torch.bool:
                target_tensor += mask.int()
            else:
                target_tensor += mask


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
            gs_ids: torch.Tensor|None,
            cam_ids: torch.Tensor | None,
            packed: bool = False
    ):
        """Update domination statistics.
        
        Args:
            n_touched: [nnz] for packed, [C, N] or [N] for non-packed - Number of pixels touched
            n_dominated: [nnz] for packed, [C, N] or [N] for non-packed - Number of pixels dominated
            width: Image width
            height: Image height
            gs_ids: [nnz] Gaussian IDs (for packed format), None for non-packed
            cam_ids: [nnz] Camera IDs (for packed format), None for non-packed
            packed: Whether the data is in packed format
        """
        if len(n_touched) == 0:
            return

        # Unify [N] format as [1, N] for non-packed processing
        if not packed and n_touched.dim() == 1:
            n_touched = n_touched.unsqueeze(0)  # [N] -> [1, N]
            n_dominated = n_dominated.unsqueeze(0)  # [N] -> [1, N]
        
        # Calculate percentages
        total_px = width * height
        touched_pct = n_touched.float() / total_px
        dominated_pct = n_dominated.float() / total_px
        
        # Update max percentages
        _accumulate_max(self.max_touchedPct, gs_ids, touched_pct, packed=packed)
        _accumulate_max(self.max_dominatedPct, gs_ids, dominated_pct, packed=packed)
        
        # Count cameras from which each gaussian was visible
        if packed:
            visible_mask = n_touched > 0  # [nnz]
        else:
            # For non-packed: sum visibility across cameras for each gaussian
            visible_mask = (n_touched > 0).sum(dim=0) if n_touched.dim() == 2 else (n_touched > 0).int()
        
        _accumulate_count(self.n_cameras_visible_from, gs_ids, visible_mask, packed=packed)
        
        # Accumulate total touched and dominated pixels across all cameras
        _accumulate_by_ids(self.n_touched_accum, gs_ids, n_touched, packed=packed)
        _accumulate_by_ids(self.n_dominated_accum, gs_ids, n_dominated, packed=packed)

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
        # Set invisible elements to zero so they don't affect maximum
        masked_rates = sampling_rates * visible_mask.float()
        _accumulate_max(self.max_sampling_rate, None, masked_rates, packed=False)
    
    @torch.no_grad()
    def update_gradient_stats(
            self,
            width: int,
            height: int,
            grads: torch.Tensor,
            grads_abs: torch.Tensor,
            importance_grads: torch.Tensor,
            gs_ids: Optional[torch.Tensor],
            n_touched: Optional[torch.Tensor] = None,
            radii: Optional[torch.Tensor] = None,
            packed: bool = False,
    ):
        """Update gradient accumulation statistics.
        
        Args:
            grads: [nnz] for packed, [C, N] or [N] for non-packed - Gradient norms
            grads_abs: [nnz] for packed, [C, N] or [N] for non-packed - Absolute gradient norms
            importance_grads: [nnz] for packed, [C, N] or [N] for non-packed - Importance gradients (vG^2)
            gs_ids: [nnz] Gaussian IDs (for packed format), None for non-packed
            n_touched: [nnz] for packed, [C, N] or [N] for non-packed - Number of touched pixels (more reliable than radii)
            radii: [nnz] for packed, [C, N] or [N] for non-packed - Optional radii values
            width: Image width for radii normalization
            height: Image height for radii normalization
            packed: Whether the data is in packed format
        """
        
        # Accumulate gradients
        _accumulate_by_ids(self.grad2d, gs_ids, grads, packed=packed)
        _accumulate_by_ids(self.grad2d_abs, gs_ids, grads_abs, packed=packed)
        _accumulate_by_ids(self.importance, gs_ids, importance_grads, packed=packed)
        
        # Count how many times each gaussian was rendered  
        # Priority: n_touched > radii > assume all visible
        # Determine visibility mask
        if n_touched is not None and len(n_touched):
            visible_mask = (n_touched > 0).float()
        elif radii is not None:
            visible_mask = (radii[..., 0] > 0).float()
        elif packed:
            visible_mask = torch.ones_like(gs_ids, dtype=torch.float32)
        else:
            # Without visibility info, assume all gaussians processed
            n_cameras = grads.shape[0] if grads.dim() == 2 else 1
            self.count += n_cameras
            visible_mask = None
        
        if visible_mask is not None:
            _accumulate_count(self.count, gs_ids, visible_mask, packed=packed)
        
        if radii is not None:
            if self.radii is None:
                self.radii = torch.zeros_like(self.grad2d)
            # Normalize radii to [0, 1] screen space
            normalized_radii = torch.norm(radii.float(), dim=-1) / float(max(width, height))
            _accumulate_max(self.radii, gs_ids, normalized_radii, packed=packed)

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
            self.update_domination_stats(
                n_touched=info["n_touched"],
                n_dominated=info["n_dominated"],
                width=info.get("width", 1920),  # Default if not provided
                height=info.get("height", 1080),  # Default if not provided
                gs_ids=info["gaussian_ids"],
                cam_ids=info["camera_ids"],
                packed=packed
            )

        # Update gradient statistics if gradients are available
        if key_for_gradient in info and info[key_for_gradient].grad is not None:
            if key_for_gradient == "gradient_2dgs":
                # For gradient_2dgs, extract appropriate elements
                gradient_2dgs = info[key_for_gradient].grad.detach().clone()
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

            # Update gradient statistics
            self.update_gradient_stats(
                width=width,
                height=height,
                grads=grads.norm(dim=-1),
                grads_abs=grads_abs.norm(dim=-1),
                importance_grads=importance_grads.squeeze(-1),
                gs_ids=info["gaussian_ids"],
                n_touched=info.get("n_touched"),
                radii=info.get("radii"),
                packed=packed,
            )

        # Update sampling rate statistics if available
        if params is not None and "camtoworlds" in info and "Ks" in info:
            # Compute euclidean distances from camera to gaussians
            camtoworlds = info["camtoworlds"]
            Ks = info["Ks"]

            # Get means from params
            means = params["means"].detach()

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
