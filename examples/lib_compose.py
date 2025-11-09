from enum import Enum
from typing import List, Optional, Tuple

import torch
from torch import Tensor


class CompositingOrder(Enum):
    BACK_TO_FRONT = "back-to-front"
    FRONT_TO_BACK = "front-to-back"


# @torch.compiler.compile()
def compose_renders_back_to_front(
    renders_list: List[Tensor],
    alphas_list: List[Tensor],
    background: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    Compose multiple renders with alpha channels using back-to-front order (painters algorithm).

    Args:
        renders_list: List of render tensors [B, H, W, C] where C is typically 3 (RGB) or 4 (RGB+D)
        alphas_list: List of alpha tensors [B, H, W, 1]
        background: Optional background color [B, H, W, C] or [C] or scalar

    Returns:
        Tuple of (composed_render, composed_alpha) both [B, H, W, C] and [B, H, W, 1]
    """
    if not renders_list:
        raise ValueError("At least one render must be provided")

    # Ensure all renders have the same shape
    shape = renders_list[0].shape
    for i, render in enumerate(renders_list[1:], 1):
        if render.shape[-1] != shape[-1]:
            raise ValueError(f"Render {i} has {render.shape[-1]} channels, expected {shape[-1]} channels")

    # Start with background or zeros
    if background is not None:
        if background.dim() == 1:  # Single color [C]
            composed = background.view(1, 1, 1, -1).expand(shape)
        elif background.dim() == 3:  # [H, W, C]
            composed = background.unsqueeze(0).expand(shape)
        else:  # [B, H, W, C]
            composed = background
        accumulated_alpha = torch.ones_like(alphas_list[0])
    else:
        composed = torch.zeros_like(renders_list[0])
        accumulated_alpha = torch.zeros_like(alphas_list[0])

    # Composite from back to front
    for render, alpha in zip(reversed(renders_list), reversed(alphas_list)):
        composed = render * alpha + composed * (1.0 - alpha)
        accumulated_alpha = alpha + accumulated_alpha * (1.0 - alpha)

    return composed, accumulated_alpha


def compose_renders_front_to_back(
    renders_list: List[Tensor],
    alphas_list: List[Tensor],
    background: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    Compose multiple renders with alpha channels using front-to-back order (early termination possible).

    Args:
        renders_list: List of render tensors [B, H, W, C] where C is typically 3 (RGB) or 4 (RGB+D)
        alphas_list: List of alpha tensors [B, H, W, 1]
        background: Optional background color [B, H, W, C] or [C] or scalar

    Returns:
        Tuple of (composed_render, composed_alpha) both [B, H, W, C] and [B, H, W, 1]
    """
    if not renders_list:
        raise ValueError("At least one render must be provided")

    # Ensure all renders have the same shape
    shape = renders_list[0].shape
    for i, render in enumerate(renders_list[1:], 1):
        if render.shape[-1] != shape[-1]:
            raise ValueError(f"Render {i} has {render.shape[-1]} channels, expected {shape[-1]} channels")

    # Start with first layer
    composed = renders_list[0] * alphas_list[0]
    accumulated_alpha = alphas_list[0]

    # Add subsequent layers
    for render, alpha in zip(renders_list[1:], alphas_list[1:]):
        composed = composed + render * alpha * (1.0 - accumulated_alpha)
        accumulated_alpha = accumulated_alpha + alpha * (1.0 - accumulated_alpha)

    # Add background if not fully opaque
    if background is not None:
        if background.dim() == 1:  # Single color [C]
            bg = background.view(1, 1, 1, -1).expand(shape)
        elif background.dim() == 3:  # [H, W, C]
            bg = background.unsqueeze(0).expand(shape)
        else:  # [B, H, W, C]
            bg = background
        composed = composed + bg * (1.0 - accumulated_alpha)
        accumulated_alpha = torch.ones_like(accumulated_alpha)  # With background, final alpha is 1

    return composed, accumulated_alpha


def compose_renders(
    renders_list: List[Tensor],
    alphas_list: List[Tensor],
    background: Optional[Tensor] = None,
    compositing_order: CompositingOrder = CompositingOrder.BACK_TO_FRONT
) -> Tuple[Tensor, Tensor]:
    """
    Compose multiple renders with alpha channels into a single image.

    Args:
        renders_list: List of render tensors [B, H, W, C] where C is typically 3 (RGB) or 4 (RGB+D)
        alphas_list: List of alpha tensors [B, H, W, 1]
        background: Optional background color [B, H, W, C] or [C] or scalar
        compositing_order: Order of compositing - CompositingOrder.BACK_TO_FRONT (painters algorithm)
                          or CompositingOrder.FRONT_TO_BACK (early termination possible)

    Returns:
        Tuple of (composed_render, composed_alpha) both [B, H, W, C] and [B, H, W, 1]
    """
    if compositing_order == CompositingOrder.BACK_TO_FRONT:
        return compose_renders_back_to_front(renders_list, alphas_list, background)
    else:
        return compose_renders_front_to_back(renders_list, alphas_list, background)
