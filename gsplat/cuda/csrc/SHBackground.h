#pragma once

#include <cstdint>

namespace at {
class Tensor;
template <typename T> class optional;
}

namespace gsplat {

void launch_sh_background_fwd_kernel(
    const at::Tensor camtoworlds,
    const at::Tensor Ks,
    const at::Tensor sh_coeffs,
    const uint32_t width,
    const uint32_t height,
    const uint32_t degree,
    at::Tensor colors
);

void launch_sh_background_bwd_kernel(
    const at::Tensor camtoworlds,
    const at::Tensor Ks,
    const at::Tensor sh_coeffs,
    const uint32_t width,
    const uint32_t height,
    const uint32_t degree,
    const at::Tensor v_colors,
    at::Tensor v_sh_coeffs
);

}