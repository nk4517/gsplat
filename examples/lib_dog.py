import torch
from torch.nn import functional as F


_gaussian_kernel_cache: dict[tuple, torch.Tensor] = {}


def _get_gaussian_1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (sigma, device, dtype)
    if key not in _gaussian_kernel_cache:
        sz = int(4 * sigma + 1) | 1
        ax = torch.arange(sz, device=device, dtype=dtype) - sz // 2
        g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
        _gaussian_kernel_cache[key] = g / g.sum()
    return _gaussian_kernel_cache[key]


def _separable_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    g = _get_gaussian_1d(sigma, x.device, x.dtype).view(1, 1, -1, 1)
    c = x.shape[1]
    pad_h = g.shape[2] // 2
    x = F.pad(x, (0, 0, pad_h, pad_h), mode='reflect')
    x = F.conv2d(x, g.expand(c, -1, -1, -1), groups=c)
    g = g.transpose(2, 3)
    pad_w = g.shape[3] // 2
    x = F.pad(x, (pad_w, pad_w, 0, 0), mode='reflect')
    x = F.conv2d(x, g.expand(c, -1, -1, -1), groups=c)
    return x

def fast_dog(img: torch.Tensor, sigma: float = 1.0, k: float = 1.6) -> torch.Tensor:
    """DoG как приближение LoG, separable свёртки"""

    squeeze = img.dim() == 3
    if squeeze:
        img = img.unsqueeze(0)

    dog = _separable_blur(img, sigma * k) - _separable_blur(img, sigma)
    norm = dog.abs().max(dim=1, keepdim=True)[0]

    return norm.squeeze(0) if squeeze else norm
