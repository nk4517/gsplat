import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor

from .base import Strategy
from .ops import inject_noise_to_position, relocate, sample_add, opacity_activation
from .epoch_stats import EpochStatistics, EpochContext


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
        min_opacity (float): GSs with opacity below this value will be pruned. Default to 0.005.
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
    min_opacity: float = 0.005
    growth_factor: float = 1.05
    verbose: bool = False

    def initialize_state(self) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy."""
        n_max = 51
        binoms = torch.zeros((n_max, n_max))
        for n in range(n_max):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)
        return {"binoms": binoms}

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

    # def step_pre_backward(
    #     self,
    #     params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    #     optimizers: Dict[str, torch.optim.Optimizer],
    #     # state: Dict[str, Any],
    #     step: int,
    #     info: Dict[str, Any],
    # ):
    #     """Callback function to be executed before the `loss.backward()` call."""
    #     pass

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

        if not epoch_ctx.epoch_end:
            return

        state["binoms"] = state["binoms"].to(params["means"].device)

        binoms = state["binoms"]

        # Check if refinement should happen at this epoch
        should_refine = (self.refine_stop_epochs <= epoch_ctx.i_epoch < self.refine_start_epochs and
                         epoch_ctx.i_epoch % self.refine_every_epochs == 0)


        if should_refine:
            # teleport GSs
            n_relocated_gs = self._relocate_gs(params, optimizers, binoms)
            if self.verbose:
                print(f"Epoch {epoch_ctx.i_epoch} (Step {step}): Relocated {n_relocated_gs} GSs.")

            # add new GSs
            n_new_gs = self._add_new_gs(params, optimizers, binoms)
            if self.verbose:
                print(
                    f"Epoch {epoch_ctx.i_epoch} (Step {step}): Added {n_new_gs} GSs. "
                    f"Now having {len(params['means'])} GSs."
                )

            torch.cuda.empty_cache()

        # add noise to GSs
        inject_noise_to_position(
            params=params, optimizers=optimizers, state={}, scaler=lr * self.noise_lr
        )

    @torch.no_grad()
    def _relocate_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
    ) -> int:
        opacities = opacity_activation(params["opacities"].flatten())
        dead_mask = opacities <= self.min_opacity
        n_gs = dead_mask.sum().item()
        if n_gs > 0:
            relocate(
                params=params,
                optimizers=optimizers,
                state={},
                mask=dead_mask,
                binoms=binoms,
                min_opacity=self.min_opacity,
            )
        return n_gs

    @torch.no_grad()
    def _add_new_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
    ) -> int:
        current_n_points = len(params["means"])
        n_target = min(self.cap_max, int(self.growth_factor * current_n_points))
        n_gs = max(0, n_target - current_n_points)
        if n_gs > 0:
            sample_add(
                params=params,
                optimizers=optimizers,
                state={},
                n=n_gs,
                binoms=binoms,
                min_opacity=self.min_opacity,
            )
        return n_gs
