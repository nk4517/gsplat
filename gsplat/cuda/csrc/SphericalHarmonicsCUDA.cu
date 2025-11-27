#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "SphericalHarmonics.h"
#include "Utils.cuh"
#include "SphericalHarmonicsDevice.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void spherical_harmonics_fwd_kernel(
    const uint32_t N,
    const uint32_t K,
    const uint32_t degrees_to_use,
    const vec3 *__restrict__ dirs,       // [N, 3]
    const scalar_t *__restrict__ coeffs, // [N, K, 3]
    const bool *__restrict__ masks,      // [N]
    scalar_t *__restrict__ colors        // [N, 3]
) {
    // parallelize over N * 3
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N * 3) {
        return;
    }
    uint32_t elem_id = idx / 3;
    uint32_t c = idx % 3; // color channel
    if (masks != nullptr && !masks[elem_id]) {
        return;
    }
    sh_coeffs_to_color_fast(
        degrees_to_use,
        c,
        dirs[elem_id],
        coeffs + elem_id * K * 3,
        colors + elem_id * 3
    );
}

void launch_spherical_harmonics_fwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, 3]
    const at::optional<at::Tensor> masks, // [...]
    // outputs
    at::Tensor colors // [..., 2]
) {
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = dirs.numel() / 3;

    // parallelize over N * 3
    int64_t n_elements = N * 3;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        dirs.scalar_type(),
        "spherical_harmonics_fwd_kernel",
        [&]() {
            spherical_harmonics_fwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    N,
                    K,
                    degrees_to_use,
                    reinterpret_cast<vec3 *>(dirs.data_ptr<scalar_t>()),
                    coeffs.data_ptr<scalar_t>(),
                    masks.has_value() ? masks.value().data_ptr<bool>()
                                      : nullptr,
                    colors.data_ptr<scalar_t>()
                );
        }
    );
}

template <typename scalar_t>
__global__ void spherical_harmonics_bwd_kernel(
    const uint32_t N,
    const uint32_t K,
    const uint32_t degrees_to_use,
    const vec3 *__restrict__ dirs,         // [N, 3]
    const scalar_t *__restrict__ coeffs,   // [N, K, 3]
    const bool *__restrict__ masks,        // [N]
    const scalar_t *__restrict__ v_colors, // [N, 3
    scalar_t *__restrict__ v_coeffs,       // [N, K, 3]
    scalar_t *__restrict__ v_dirs          // [N, 3] optional
) {
    // parallelize over N * 3
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N * 3) {
        return;
    }
    uint32_t elem_id = idx / 3;
    uint32_t c = idx % 3; // color channel
    if (masks != nullptr && !masks[elem_id]) {
        return;
    }

    vec3 v_dir = {0.f, 0.f, 0.f};
    sh_coeffs_to_color_fast_vjp(
        degrees_to_use,
        c,
        dirs[elem_id],
        coeffs + elem_id * K * 3,
        v_colors + elem_id * 3,
        v_coeffs + elem_id * K * 3,
        v_dirs == nullptr ? nullptr : &v_dir
    );
    if (v_dirs != nullptr) {
        gpuAtomicAdd(v_dirs + elem_id * 3, v_dir.x);
        gpuAtomicAdd(v_dirs + elem_id * 3 + 1, v_dir.y);
        gpuAtomicAdd(v_dirs + elem_id * 3 + 2, v_dir.z);
    }
}

void launch_spherical_harmonics_bwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, 3]
    const at::optional<at::Tensor> masks, // [...]
    const at::Tensor v_colors,            // [..., 3]
    // outputs
    at::Tensor v_coeffs,            // [..., K, 3]
    at::optional<at::Tensor> v_dirs // [..., 3]
) {
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = dirs.numel() / 3;

    // parallelize over N * 3
    int64_t n_elements = N * 3;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        dirs.scalar_type(),
        "spherical_harmonics_bwd_kernel",
        [&]() {
            spherical_harmonics_bwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    N,
                    K,
                    degrees_to_use,
                    reinterpret_cast<vec3 *>(dirs.data_ptr<scalar_t>()),
                    coeffs.data_ptr<scalar_t>(),
                    masks.has_value() ? masks.value().data_ptr<bool>()
                                      : nullptr,
                    v_colors.data_ptr<scalar_t>(),
                    v_coeffs.data_ptr<scalar_t>(),
                    v_dirs.has_value() ? v_dirs.value().data_ptr<scalar_t>()
                                       : nullptr
                );
        }
    );
}

} // namespace gsplat
