import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor

from .base import Strategy
from .ops import inject_noise_to_position, relocate, sample_add, opacity_activation, scaling_activation, remove, relocate_quaternion, sample_add_quaternion, inject_noise_to_quats
from .epoch_stats import EpochStatistics, EpochContext


    opacities = opacity_activation(params["opacities"].flatten())

def _calc_importance(params, state):
    """Calculate importance scores based on epoch_stats importance/count."""
    opacities = opacity_activation(params["opacities"].flatten())
    if "epoch_stats" in state and hasattr(state["epoch_stats"], "importance"):
        epoch_stats = state["epoch_stats"]
        importance = epoch_stats.importance
        count = epoch_stats.count
        probs = torch.where(
            count > 0,
            importance / count.clamp_min(1),
            torch.zeros_like(importance)
        )
    else:
        probs = opacities.clone()
    return probs


@dataclass
class MCMCStrategy(Strategy):
    """Strategy that follows the paper:

    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/abs/2404.09591>`_

    This strategy will:

    - Periodically teleport GSs with low opacity to a place that has high opacity.
    - Periodically introduce new GSs sampled based on the opacity distribution.
    - Periodically perturb the GSs locations.

    Args:
        cap_max (int): Maximum number of GSs. Default to 1_000_000.
        noise_lr (float): MCMC samping noise learning rate. Default to 5e5.
        refine_start_epochs (int): Start refining GSs after this many epochs. Default is 15.
        refine_stop_epochs (int): Stop refining GSs after this many epochs. Default is 500.
        refine_every_epochs (int): Refine GSs every this many epochs. Default is 3.
        add_every_epochs (int): Add new GSs every this many epochs. Default is 9 (3x refine_every_epochs).
        min_opacity (float): GSs with opacity below this value will be pruned. Default to 0.005.
        prune_scale3d (float): GSs with 3d scale (normalized by scene_scale) above this
          value will be pruned. Default is 0.1.
        prune_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be pruned. Default is 0.15.
        refine_scale2d_stop_iter (int): Stop pruning GSs based on 2d scale after this
          iteration. Default is 0. Set to a positive value to enable this feature.
        growth_factor (float): Factor for growing the number of GSs. Default to 1.05.
        verbose (bool): Whether to print verbose information. Default to False.

    Examples:

        >>> from gsplat import MCMCStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = MCMCStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info, lr=1e-3)

    """

    cap_max: int = 1_000_000
    noise_lr: float = 5e5
    refine_start_epochs: int = 15  # Start refining GSs after this many epochs
    refine_stop_epochs: int = 500  # Stop refining GSs after this many epochs
    refine_every_epochs: int = 3  # Refine GSs every this many epochs
    add_every_epochs: int = 9  # Add new GSs every this many epochs (default 3x refine_every_epochs)
    min_opacity: float = 0.005
    growth_factor: float = 1.05
    verbose: bool = False
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    model_type: str | None = None

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy."""
        n_max = 51
        binoms = torch.zeros((n_max, n_max))
        for n in range(n_max):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)
        return {"binoms": binoms, "scene_scale": scene_scale}
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
        # Initialize epoch statistics at the start of each epoch
        n_gaussian = len(list(params.values())[0])
        device = list(params.values())[0].device

        # Initialize epoch statistics block
        if "epoch_stats" not in state:
            state["epoch_stats"] = EpochStatistics(n_gaussian, device)

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
        # For gradient tracking, we need to retain gradients
        if "gradient_2dgs" in info:
            info["gradient_2dgs"].retain_grad()
        elif "means2d" in info:
            info["means2d"].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        lr: float,
        epoch_ctx: EpochContext,
    ):
        """Callback function to be executed after the `loss.backward()` call.

        Args:
            lr (float): Learning rate for "means" attribute of the GS.
            epoch_ctx (EpochContext): Context information about the current epoch.
        """
        # move to the correct device
        if "means" in params:
            device = params["means"].device
        else:
            device = params["quats"].device

        with torch.no_grad():
            # Update statistics in epoch_stats
            if "epoch_stats" in state:
                # Update all statistics from info
                state["epoch_stats"].update_from_info(
                    info=info,
                    params=params,
                    key_for_gradient="gradient_2dgs",
                    packed=info.get("packed", False)
                )

            if not epoch_ctx.epoch_end:
                return

            # # Prune large gaussians after adding new ones
            # n_pruned = self._prune_large_gs(params, optimizers, state, step)
            # if n_pruned > 0 and self.verbose:
            #     print(
            #         f"Epoch {epoch_ctx.i_epoch} (Step {step}): Pruned {n_pruned} large GSs. "
            #         f"Now having {len(params['means'])} GSs."
            #     )

            state["binoms"] = state["binoms"].to(device)

            binoms = state["binoms"]

            # Check if relocation should happen at this epoch
            should_relocate = (self.refine_stop_epochs > epoch_ctx.i_epoch >= self.refine_start_epochs and
                               epoch_ctx.i_epoch % self.refine_every_epochs == 0)

            # Check if adding new GSs should happen at this epoch
            should_add = (self.refine_stop_epochs > epoch_ctx.i_epoch >= self.refine_start_epochs and
                          epoch_ctx.i_epoch % self.add_every_epochs == 0)

            if should_relocate:
                # teleport GSs
                n_relocated_gs = self._relocate_gs(params, optimizers, binoms, state)
                if self.verbose:
                    print(f"Epoch {epoch_ctx.i_epoch} (Step {step}): Relocated {n_relocated_gs} GSs.")

            if should_add:
                # add new GSs
                n_new_gs = self._add_new_gs(params, optimizers, binoms, state)
                if self.verbose:
                    print(
                        f"Epoch {epoch_ctx.i_epoch} (Step {step}): Added {n_new_gs} GSs. "
                        f"Now having {len(params['means'] if 'means' in params else params['quats'])} GSs."
                    )

            if should_relocate or should_add:
                torch.cuda.empty_cache()

        # add noise to GSs
        if "means" in params:
            inject_noise_to_position(
                params=params, optimizers=optimizers, state={}, scaler=lr * self.noise_lr
            )
        # else:
        #     inject_noise_to_quats(
        #         params=params, optimizers=optimizers, state={}, scaler=lr * self.noise_lr
        #     )

        with torch.no_grad():
            # Reset epoch statistics for next epoch
            if "epoch_stats" in state:
                state["epoch_stats"].reset()

    @torch.no_grad()
    def _relocate_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
        state: Dict[str, Any],
        pct=0.01,
    ) -> int:
        opacities = opacity_activation(params["opacities"].flatten())
        device = params["scales"].device
        
        # Start with dead splats (low opacity)
        dead_mask = opacities <= self.min_opacity
        
        # Add large splats for relocation instead of pruning
        # Check 3D scale
        is_too_big = (
            scaling_activation(params["scales"]).max(dim=-1).values
            > self.prune_scale3d * state["scene_scale"]
        )
        dead_mask |= is_too_big

        if "epoch_stats" in state:
            # Use n_touched from epoch to determine dead splats
            dead_mask |= state["epoch_stats"].n_touched_accum == 0

            # Check 2D scale if enabled
            if hasattr(state["epoch_stats"], "max_touchedPct"):
                # Relocate gaussians that touch too many pixels
                is_too_big_2d = state["epoch_stats"].max_touchedPct > self.prune_scale2d
                dead_mask |= is_too_big_2d


        n_gs = dead_mask.sum().item()
        if n_gs > 0:
            # Compute sampling probabilities
            # probs = _calc_prob1(params, state)
            probs = None

            if self.model_type == "quaternion":
                # Use quaternion-specific relocate for skysphere
                relocate_quaternion(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    mask=dead_mask,
                    binoms=binoms,
                    radius=state["scene_scale"],  # Use scene_scale as radius
                    min_opacity=self.min_opacity,
                    probs=probs,
                )
            else:
                relocate(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    mask=dead_mask,
                    binoms=binoms,
                    min_opacity=self.min_opacity,
                    model_type=self.model_type,
                    probs=probs,
                )
        return n_gs

    @torch.no_grad()
    def _add_new_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
        state: Dict[str, Any],
    ) -> int:
        current_n_points = len(params["scales"])
        n_target = min(self.cap_max, int(self.growth_factor * current_n_points))
        n_gs = max(0, n_target - current_n_points)
        if n_gs > 0:
            # Compute sampling probabilities
            # probs = _calc_prob1(params, state)
            probs=None

            if self.model_type == "quaternion":
                # Use quaternion-specific sample_add for skysphere
                sample_add_quaternion(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    n=n_gs,
                    binoms=binoms,
                    radius=state["scene_scale"],  # Use scene_scale as radius
                    min_opacity=self.min_opacity,
                    probs=probs,
                )
            else:
                sample_add(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    n=n_gs,
                    binoms=binoms,
                    min_opacity=self.min_opacity,
                    model_type=self.model_type,
                    probs=probs,
                )
        return n_gs

    @torch.no_grad()
    def _prune_large_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        """Prune gaussians that are too large."""
        device = params["scales"].device
        n_total = len(params["scales"])
        
        # Initialize prune mask
        is_prune = torch.zeros(n_total, dtype=torch.bool, device=device)
        
        # Prune by 3D scale
        is_too_big = (
            scaling_activation(params["scales"]).max(dim=-1).values
            > self.prune_scale3d * state["scene_scale"]
        )
        is_prune |= is_too_big
        
        # Prune by max touched percentage if enabled
        if self.refine_scale2d_stop_iter > 0 and step < self.refine_scale2d_stop_iter:
            if "epoch_stats" in state and hasattr(state["epoch_stats"], "max_touchedPct"):
                # Prune gaussians that touch too many pixels (e.g., > 1% of image)
                is_too_big_2d = state["epoch_stats"].max_touchedPct > 0.9
                is_prune |= is_too_big_2d
        
        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)
        return n_prune
