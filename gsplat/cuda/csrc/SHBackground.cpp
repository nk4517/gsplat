#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"
#include "Ops.h"
#include "SHBackground.h"

namespace gsplat {

at::Tensor sh_background_fwd(
    const at::Tensor camtoworlds,  // [B, 4, 4]
    const at::Tensor Ks,           // [B, 3, 3]
    const at::Tensor sh_coeffs,    // [K, 3]
    const uint32_t width,
    const uint32_t height,
    const uint32_t degree
) {
    DEVICE_GUARD(camtoworlds);
    CHECK_INPUT(camtoworlds);
    CHECK_INPUT(Ks);
    CHECK_INPUT(sh_coeffs);
    
    TORCH_CHECK(camtoworlds.size(-2) == 4 && camtoworlds.size(-1) == 4,
                "camtoworlds must have shape [..., 4, 4]");
    TORCH_CHECK(Ks.size(-2) == 3 && Ks.size(-1) == 3,
                "Ks must have shape [..., 3, 3]");
    TORCH_CHECK(sh_coeffs.size(-1) == 3,
                "sh_coeffs must have last dimension 3");
    
    uint32_t B = camtoworlds.size(0);
    at::Tensor colors = at::empty({B, height, width, 3}, camtoworlds.options());
    
    launch_sh_background_fwd_kernel(
        camtoworlds, Ks, sh_coeffs, width, height, degree, colors
    );
    
    return colors;
}

at::Tensor sh_background_bwd(
    const at::Tensor camtoworlds,
    const at::Tensor Ks,
    const at::Tensor sh_coeffs,
    const uint32_t width,
    const uint32_t height,
    const uint32_t degree,
    const at::Tensor v_colors
) {
    DEVICE_GUARD(camtoworlds);
    CHECK_INPUT(camtoworlds);
    CHECK_INPUT(Ks);
    CHECK_INPUT(sh_coeffs);
    CHECK_INPUT(v_colors);
    
    TORCH_CHECK(v_colors.size(-1) == 3, "v_colors must have last dimension 3");
    
    at::Tensor v_sh_coeffs = at::zeros_like(sh_coeffs);
    
    launch_sh_background_bwd_kernel(
        camtoworlds,
        Ks,
        sh_coeffs,
        width,
        height,
        degree,
        v_colors,
        v_sh_coeffs
    );
    
    return v_sh_coeffs;
}

} // namespace gsplat