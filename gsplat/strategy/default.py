from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from typing_extensions import Literal

from .base import Strategy, EpochContext, StateWrapper
from .ops import (
    duplicate, remove, reset_opa, split, split_n_2dgs, opacity_activation, scaling_activation
)
from ..antialias_2dgs import update_max_sampling_rate


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



@torch.jit.script
def stoch1(prune_mask: torch.Tensor, pct: float = 0.75) -> torch.Tensor:
    prune_mask &= torch.rand(prune_mask.shape[0], device=prune_mask.device) < pct
    return prune_mask


@dataclass
class DefaultStrategy(Strategy):
    """A default strategy that follows the original 3DGS paper:

    `3D Gaussian Splatting for Real-Time Radiance Field Rendering <https://arxiv.org/abs/2308.04079>`_

    The strategy will:

    - Periodically duplicate GSs with high image plane gradients and small scales.
    - Periodically split GSs with high image plane gradients and large scales.
    - Periodically prune GSs with low opacity.
    - Periodically reset GSs to a lower opacity.

    If `absgrad=True`, it will use the absolute gradients instead of average gradients
    for GS duplicating & splitting, following the AbsGS paper:

    `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_

    Which typically leads to better results but requires to set the `grow_grad2d` to a
    higher value, e.g., 0.0008. Also, the :func:`rasterization` function should be called
    with `absgrad=True` as well so that the absolute gradients are computed.

    Args:
        prune_opa (float): GSs with opacity below this value will be pruned. Default is 0.005.
        grow_grad2d (float): GSs with image plane gradient above this value will be
          split/duplicated. Default is 0.0002.
        grow_scale3d (float): GSs with 3d scale (normalized by scene_scale) below this
          value will be duplicated. Above will be split. Default is 0.01.
        grow_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be split. Default is 0.05.
        prune_scale3d (float): GSs with 3d scale (normalized by scene_scale) above this
          value will be pruned. Default is 0.1.
        prune_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be pruned. Default is 0.15.
        refine_scale2d_stop_iter (int): Stop refining GSs based on 2d scale after this
          iteration. Default is 0. Set to a positive value to enable this feature.
        refine_start_epochs (int): Start refining GSs after this many epochs. Default is 15.
        refine_stop_epochs (int): Stop refining GSs after this many epochs. Default is 500.
        reset_every_epochs (int): Reset opacities every this many epochs. Default is 30.
        reset_start_epochs (int): Start resetting opacities after this many epochs. Default is 0.
        reset_end_epochs (int): Stop resetting opacities after this many epochs. Default is 10000.
        refine_every_epochs (int): Refine GSs every this many epochs. Default is 3.
        pause_refine_after_reset_epochs (int): Pause refining GSs for this many epochs after
          reset. Default is 1.
        absgrad (bool): Use absolute gradients for GS splitting. Default is False.
        revised_opacity (bool): Whether to use revised opacity heuristic from
          arXiv:2404.06109 (experimental). Default is False.
        verbose (bool): Whether to print verbose information. Default is False.
        key_for_gradient (str): Which variable uses for densification strategy.
          3DGS uses "means2d" gradient and 2DGS uses a similar gradient which stores
          in variable "gradient_2dgs".

    Examples:

        >>> from gsplat import DefaultStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = DefaultStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info)

    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_epochs: int = 15  # Start refining GSs after this many epochs
    refine_stop_epochs: int = 500  # Stop refining GSs after this many epochs
    reset_start_epochs: int = 100  # Start resetting opacities after this many epochs
    reset_end_epochs: int = 10000  # Stop resetting opacities after this many epochs
    reset_every_epochs: int = 20  # Reset opacities every this many epochs
    refine_every_epochs: int = 5  # Refine GSs every this many epochs
    pause_refine_after_reset_epochs: int = 1  # Pause refining for this many epochs after reset
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"
    split_big_dominated_pct: float = 0.0005  # Split gaussians dominating more than this percentage of pixels
    split_big_touched_pct: float = 0.001  # Split gaussians touching more than this percentage of pixels
    # Importance-based pruning parameters (Speedy-Splat style)
    importance_prune_enabled: bool = False  # Enable importance-based pruning
    importance_prune_start_epoch: int = 200  # Start importance pruning after this epoch
    importance_prune_end_epoch: int = 500  # Stop importance pruning after this epoch
    importance_prune_every_epochs: int = 50  # Perform importance pruning every this many epochs
    importance_prune_ratio: float = 0.3  # Fraction of gaussians to prune based on importance (0.3 = remove 30% least important)

    @torch.no_grad()
    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        # - importance: running accum of importance scores (vG^2) for each GS.
        # - radii: the radii of the GSs (normalized by the image resolution).
        state = {
            "grad2d": None, 
            "count": None, 
            "importance": None,  # For importance-based pruning (vG^2)
            "scene_scale": scene_scale,
            "last_reset_epoch": -1000,
        }
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None
        return state

    @torch.no_grad()
    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        epoch_ctx: EpochContext,
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        assert (
            self.key_for_gradient in info
        ), "The 2D means of the Gaussians is required but missing."
        info[self.key_for_gradient].retain_grad()

    @torch.no_grad()
    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        epoch_ctx: EpochContext,
        packed: bool = False,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        # Check if refinement should be stopped based on epochs
        if epoch_ctx.i_epoch >= self.refine_stop_epochs:
            return

        self._update_state(params, state, info, packed=packed)

        if not epoch_ctx.epoch_end:
            return


        # нужно вызывать именно здесь. после последнего обновления статов но до opacity/refine/prune
        if epoch_ctx.i_epoch > 0 and "aa_params" in info:
            aa_params = info["aa_params"]
            if epoch_ctx.i_epoch % aa_params["aa_compute_every"] == 0:
                update_max_sampling_rate(
                    params,
                    aa_params["trainset"],
                    aa_params["near_plane"],
                    aa_params["far_plane"],
                    aa_params["device"],
                    state,
                )

        # Check if we should reset opacities at this epoch
        should_reset = (self.reset_every_epochs > 0 and 
                       epoch_ctx.i_epoch > 0 and 
                       epoch_ctx.i_epoch % self.reset_every_epochs == 0 and
                       epoch_ctx.i_epoch >= self.reset_start_epochs and
                       epoch_ctx.i_epoch <= self.reset_end_epochs)
        
        if should_reset:
            # Create mask for non-sky gaussians if skyness is available
            reset_mask = None
            # if "skyness" in params:
            #     skyness_probs = torch.sigmoid(params["skyness"])
            #     reset_mask = skyness_probs < 0.66  # Only reset gaussians with skyness < 66%
            #
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
                mask=reset_mask,
            )
            state["last_reset_epoch"] = epoch_ctx.i_epoch
            if self.verbose:
                n_reset = reset_mask.sum().item() if reset_mask is not None else len(params["opacities"])
                print(f"Epoch {epoch_ctx.i_epoch} (Step {step}): Reset opacities to {self.prune_opa * 2.0} for {n_reset} gaussians")

        # Check if we should refine at this epoch
        should_refine = not should_reset and (
            epoch_ctx.i_epoch >= self.refine_start_epochs and
            epoch_ctx.i_epoch % self.refine_every_epochs == 0 and
            (epoch_ctx.i_epoch - state["last_reset_epoch"]) >= self.pause_refine_after_reset_epochs
        )

        if should_refine:

            # grow GSs
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step, epoch_ctx)
            if self.verbose:
                print(
                    f"Epoch {epoch_ctx.i_epoch} (Step {step}): {n_dupli} GSs duplicated, {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            # prune GSs
            n_prune = self._prune_gs(params, optimizers, state, step, epoch_ctx, n_dupli+n_split)
            if self.verbose:
                print(
                    f"Epoch {epoch_ctx.i_epoch} (Step {step}): {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )

            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            state["importance"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()

        # оно всегда, но на всякий случай чтобы понятно было
        if epoch_ctx.epoch_end:
            if "epoch_stats" in state:
                state["epoch_stats"].reset()

        if should_refine or should_reset or epoch_ctx.epoch_end:
            torch.cuda.empty_cache()


    @torch.no_grad()
    def step_epoch_start(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        epoch_ctx: EpochContext,
    ):
        """Callback function to be executed before forward pass of the first camera in batch."""
        # Reset epoch-based statistics at the start of each epoch
        n_gaussian = len(list(params.values())[0])
        device = list(params.values())[0].device

        # Initialize epoch statistics block
        if "epoch_stats" not in state:
            state["epoch_stats"] = EpochStatistics(n_gaussian, device)

    @torch.no_grad()
    def _update_state(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            state: Dict[str, Any],
            info: Dict[str, Any],
            packed: bool = False,
    ):
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
            # Take element 4 for importance (vG^2)
            importance_grads = gradient_2dgs[..., 4:5]
        else:
            grads = info[self.key_for_gradient].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize state on the first run
        n_gaussian = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if state["importance"] is None:
            state["importance"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            assert "radii" in info, "radii is required but missing."
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"].max(dim=-1).values  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
            importance_grads = importance_grads[sel]  # [nnz, 1]
            radii = info["radii"][sel].max(dim=-1).values  # [nnz]
        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["importance"].index_add_(0, gs_ids, importance_grads.squeeze(-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )
        if self.refine_scale2d_stop_iter > 0:
            # Should be ideally using scatter max
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                # normalize radii to [0, 1] screen space
                radii / float(max(info["width"], info["height"])),
            )

        if "epoch_stats" in state:
            # ========== Epoch-based statistics block ==========
            if "n_touched" in info and "n_dominated" in info:
                state["epoch_stats"].update_domination_stats(
                    n_touched=info["n_touched"], n_dominated=info["n_dominated"],
                    width=info["width"], height=info["height"],
                    gs_ids=gs_ids, packed=packed)


            # Update minimum depth statistics if available
            if "camtoworlds" in info and "Ks" in info:
                # Compute euclidean distances from camera to gaussians
                camtoworlds = info["camtoworlds"]

                # Euclidean distance from camera to gaussian center
                cam_positions = camtoworlds[:, :3, 3].unsqueeze(1)  # [C, 1, 3]
                means = params["means"].unsqueeze(0)  # [1, N, 3]
                distances = torch.norm(cam_positions - means, dim=2)  # [C, N]
                
                # Compute sampling rates f/d for each camera
                # Extract focal lengths from camera intrinsics
                Ks = info["Ks"]  # [C, 3, 3]
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
                state["epoch_stats"].update_max_sampling_rate(sampling_rates, visible_mask)

            # ========== End of epoch-based statistics block ==========

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        epoch_ctx: EpochContext,
    ) -> Tuple[int, int]:
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        n_before = len(params["opacities"])
        is_split = torch.zeros(n_before, dtype=torch.bool, device=device)
        is_split_huge = torch.zeros(n_before, dtype=torch.bool, device=device)
        is_dupli = torch.zeros(n_before, dtype=torch.bool, device=device)

        # Filter out sky gaussians if skyness is available
        is_not_sky = torch.ones(n_before, dtype=torch.bool, device=device)
        if "skyness" in params:
            skyness_probs = torch.sigmoid(params["skyness"].detach())
            is_not_sky = skyness_probs < 0.66  # Only process gaussians with skyness < 66%

        is_certain_sky = ~is_not_sky

        # is_grad_high = grads > self.grow_grad2d
        # is_small = (
        #     torch.exp(params["scales"]).max(dim=-1).values
        #     <= self.grow_scale3d * state["scene_scale"]
        # )
        # is_dupli = is_grad_high & is_small
        # Apply skyness filter to duplication
        is_dupli = is_dupli & is_not_sky
        n_dupli = is_dupli.sum().item()

        # is_large = ~is_small

        # is_dupli |= is_grad_high_for_split
        # if self.refine_scale2d_stop_iter > 0 and step < self.refine_scale2d_stop_iter:
        #     is_split |= state["radii"] > self.grow_scale2d

        # is_dupli = stoch1(is_dupli, 0.33)

        n_dupli = is_dupli.sum().item()

        # Split gaussians that dominate too many pixels (using current epoch statistics)
        if "epoch_stats" in state and hasattr(state["epoch_stats"], "max_touchedPct"):
            # Use split_n for very large gaussians (> 2% of image)
            # is_split_huge = (state["epoch_stats"].max_touchedPct > 0.005) & is_not_sky
            # is_split_huge |= (state["epoch_stats"].max_touchedPct > 0.05) & is_certain_sky

            # Use regular split for moderately large gaussians
            split_by_domination = (state["epoch_stats"].max_dominatedPct > self.split_big_dominated_pct) & ~is_split_huge
            split_by_big_touch = ((state["epoch_stats"].max_touchedPct > self.split_big_touched_pct) & ~split_by_domination) & is_not_sky
            split_by_big_touch |= ((state["epoch_stats"].max_touchedPct > self.split_big_touched_pct * 10) & ~split_by_domination) & is_certain_sky

            print("split_n by huge touch pct:", is_split_huge.sum().item())
            print("split by domination pct:", split_by_domination.sum().item())
            print("split by touch pct:", split_by_big_touch.sum().item())
            
            is_split |= split_by_domination
            is_split_huge |= split_by_big_touch

            is_large = state["epoch_stats"].max_touchedPct > self.split_big_touched_pct / 5

            is_split |= is_grad_high_for_split & ~is_large


        # Apply skyness filter to splitting
        # is_split = is_split & is_not_sky[:len(is_split)]
        # is_split_huge = is_split_huge & is_not_sky[:len(is_split_huge)]

        # is_split = stoch1(is_split, 0.33)
        
        n_split = is_split.sum().item()
        n_split_huge = is_split_huge.sum().item()

        # first duplicate
        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)
            # Resize epoch statistics after duplication

        # new GSs added by duplication will not be split

        is_split_huge = torch.cat(
            [
                is_split_huge,
                torch.zeros(len(params["opacities"]) - len(is_split_huge), dtype=torch.bool, device=device),
            ]
        )

        # then split_n for very large gaussians
        HUGE_SPLIN_COUNT = 16
        if n_split_huge > 0:
            split_n_2dgs(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split_huge,
                revised_opacity=self.revised_opacity,
                n_splits=6,
                size_scale=0.6,
                distribution_scale=0.8,
            )

        is_split = torch.cat(
            [
                is_split,
                torch.zeros(len(params["opacities"]) - len(is_split), dtype=torch.bool, device=device),
            ]
        )

        # then regular split
        if n_split > 0:
            split_n_2dgs(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
                n_splits=2,
                size_scale=0.5,
                distribution_scale=0.6,
            )

        n_after = len(params["opacities"])

        n_new_total = n_after - n_before
        return n_dupli, n_new_total-n_dupli

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        epoch_ctx: EpochContext,
            n_new: int
    ) -> int:
        device = params["opacities"].device
        n_total = len(params["opacities"])

        # Initialize prune mask for all gaussians
        is_prune = torch.zeros(n_total, dtype=torch.bool, device=device)

        # Filter out sky gaussians if skyness is available
        is_not_sky = torch.ones(n_total, dtype=torch.bool, device=device)
        if "skyness" in params:
            skyness_probs = torch.sigmoid(params["skyness"].detach())
            is_not_sky = skyness_probs < 0.66  # Only process gaussians with skyness < 66%

        # Apply pruning criteria only to old gaussians

        # Opacity-based pruning for old gaussians
        is_prune |= opacity_activation(params["opacities"].detach().flatten()) < self.prune_opa
        
        # Apply skyness filter - sky gaussians should not be pruned
        # is_prune = is_prune & is_not_sky

        # Only prune by size after first reset epoch
        # if (epoch_ctx.i_epoch - state.get("last_reset_epoch", -1000)) > 0:
        #     is_too_big = (
        #         scaling_activation(params["scales"][:n_old]).max(dim=-1).values
        #         > self.prune_scale3d * state["scene_scale"]
        #     )
        #     # The official code also implements sreen-size pruning but
        #     # it's actually not being used due to a bug:
        #     # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
        #     # We implement it here for completeness but set `refine_scale2d_stop_iter`
        #     # to 0 by default to disable it.
        #     if step < self.refine_scale2d_stop_iter:
        #         is_too_big |= state["radii"][:n_old] > self.prune_scale2d
        #
        #     is_prune[:n_old] = is_prune[:n_old] | is_too_big

        # Prune gaussians that were never visible from any camera (using current epoch statistics)
        epoch_stats = state["epoch_stats"]
        if "epoch_stats" in state and hasattr(epoch_stats, "n_cameras_visible_from") and (epoch_ctx.i_epoch - state["last_reset_epoch"]) > 1:
            # после ресета часть будет прозрачной и невидимой, нужно дать время чтобы набралась видимость
            is_never_visible = epoch_stats.n_cameras_visible_from == 0

            is_never_visible = stoch1(is_never_visible, 0.33)

            # is_never_visible |= epoch_stats.max_touchedPct[:n_old] < 1. / (1920 * 1080)
            print("removing lost splats", is_never_visible.sum().item())
            is_prune |= is_never_visible # & is_not_sky

        # Importance-based pruning (Speedy-Splat style https://arxiv.org/abs/2412.00578)
        if (self.importance_prune_enabled and
                self.importance_prune_start_epoch <= epoch_ctx.i_epoch < self.importance_prune_end_epoch and
            epoch_ctx.i_epoch % self.importance_prune_every_epochs == 0 and
            "importance" in state and state["importance"] is not None):

            scores = state["importance"]
            count = state["count"]

            # Normalize scores by number of cameras where gaussian was visible
            # to avoid bias against gaussians visible in fewer views
            normalized_scores = torch.where(
                count > 0,
                scores / count.clamp_min(1),
                torch.zeros_like(scores)
            )

            # Find threshold for pruning based on percentile
            if normalized_scores.numel() > 0:
                sorted_scores, _ = torch.sort(normalized_scores)
                threshold_idx = int(self.importance_prune_ratio * len(sorted_scores))
                threshold = sorted_scores[threshold_idx] if threshold_idx < len(sorted_scores) else sorted_scores[-1]

                # Mark gaussians below threshold for pruning
                is_low_importance = normalized_scores <= threshold

                # Combine with existing prune mask (only for old gaussians)
                # Apply skyness filter to importance-based pruning
                is_prune |= is_low_importance & is_not_sky

                if self.verbose:
                    print(f"Importance pruning: marking {is_low_importance.sum().item()} gaussians "
                          f"with importance <= {threshold:.6f}")

        # is_prune[-n_new:] = False
        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune