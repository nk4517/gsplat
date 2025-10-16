from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch

from .epoch_stats import EpochContext


class StateWrapper:
    """Wrapper that exposes both state dict and EpochStatistics tensors for unified updates."""

    def __init__(self, state: Dict[str, any]):
        self.state = state
        self.epoch_stats = state.get("epoch_stats", None)
        self._epoch_stats_tensors = {}

        # Map EpochStatistics tensor attributes to prefixed keys
        if self.epoch_stats is not None:
            for attr_name in dir(self.epoch_stats):
                attr = getattr(self.epoch_stats, attr_name)
                if isinstance(attr, torch.Tensor):
                    # Prefix with __epoch_stats__ to avoid conflicts
                    key = f"__epoch_stats__{attr_name}"
                    self._epoch_stats_tensors[key] = attr_name

    def items(self):
        """Iterate over both state items and EpochStatistics tensors."""
        # First yield regular state items (except epoch_stats object itself)
        for k, v in self.state.items():
            if k != "epoch_stats":
                yield k, v

        # Then yield EpochStatistics tensors with special keys
        if self.epoch_stats is not None:
            for key, attr_name in self._epoch_stats_tensors.items():
                yield key, getattr(self.epoch_stats, attr_name)

    def __getitem__(self, key):
        """Get item from state or EpochStatistics."""
        if key.startswith("__epoch_stats__"):
            attr_name = self._epoch_stats_tensors[key]
            return getattr(self.epoch_stats, attr_name)
        return self.state[key]

    def __setitem__(self, key, value):
        """Set item in state or EpochStatistics."""
        if key.startswith("__epoch_stats__"):
            attr_name = self._epoch_stats_tensors[key]
            setattr(self.epoch_stats, attr_name, value)
        else:
            self.state[key] = value

    def __contains__(self, key):
        """Check if key exists in state or EpochStatistics."""
        if key.startswith("__epoch_stats__"):
            return key in self._epoch_stats_tensors
        return key in self.state


@dataclass
class Strategy:
    """Base class for the GS densification strategy.

    This class is an base class that defines the interface for the GS
    densification strategy.
    """

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers."""
        trainable_params = set(
            [name for name, param in params.items() if param.requires_grad]
        )
        assert trainable_params == set(optimizers.keys()), (
            "trainable parameters and optimizers must have the same keys, "
            f"but got {trainable_params} and {optimizers.keys()}"
        )

        for optimizer in optimizers.values():
            assert len(optimizer.param_groups) == 1, (
                "Each optimizer must have exactly one param_group, "
                "that cooresponds to each parameter, "
                f"but got {len(optimizer.param_groups)}"
            )

    def step_epoch_start(
        self,
        *args,
        epoch_ctx: Optional[EpochContext] = None,
        **kwargs,
    ):
        """Callback function to be executed before forward pass of the first camera in batch."""
        pass

    def step_pre_backward(
        self,
        *args,
        epoch_ctx: Optional[EpochContext] = None,
        **kwargs,
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        pass

    def step_post_backward(
        self,
        *args,
        epoch_ctx: Optional[EpochContext] = None,
        **kwargs,
    ):
        """Callback function to be executed after the `loss.backward()` call.
        
        Args:
            epoch_ctx: Context information about the current epoch.
        """
        pass
