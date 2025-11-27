#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "SHBackground.h"
#include "Utils.cuh"
#include "SphericalHarmonicsDevice.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void sh_background_fwd_kernel(
    const uint32_t B,           // batch size
    const uint32_t H,           // image height
    const uint32_t W,           // image width
    const uint32_t K,           // number of SH coefficients
    const uint32_t degree,      // SH degree
    const scalar_t *__restrict__ camtoworlds,  // [B, 4, 4]
    const scalar_t *__restrict__ Ks,           // [B, 3, 3]
    const scalar_t *__restrict__ sh_coeffs,    // [K, 3]
    scalar_t *__restrict__ colors              // [B, H, W, 3]
) {
    // Load sh_coeffs into shared memory
    extern __shared__ char smem[];
    scalar_t *sh_coeffs_shared = reinterpret_cast<scalar_t *>(smem);
    
    auto block = cg::this_thread_block();
    uint32_t tr = block.thread_rank();
    uint32_t block_size = block.size();
    
    // Cooperatively load sh_coeffs (K * 3 elements)
    for (uint32_t i = tr; i < K * 3; i += block_size) {
        sh_coeffs_shared[i] = sh_coeffs[i];
    }
    block.sync();
    
    uint32_t idx = cg::this_grid().thread_rank();
    uint32_t total_pixels = B * H * W;
    
    if (idx >= total_pixels) {
        return;
    }
    
    uint32_t b = idx / (H * W);
    uint32_t hw = idx % (H * W);
    uint32_t i = hw / W;
    uint32_t j = hw % W;
    
    // Pixel center
    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    
    // Load camera intrinsics
    const scalar_t *cam_K = Ks + b * 9;
    float fx = cam_K[0];
    float fy = cam_K[4];
    float cx = cam_K[2];
    float cy = cam_K[5];
    
    // Ray direction in camera space
    float dx = (px - cx) / fx;
    float dy = (py - cy) / fy;
    float dz = 1.0f;
    
    // Load camera extrinsics (rotation part)
    const scalar_t *c2w = camtoworlds + b * 16;
    float R00 = c2w[0], R01 = c2w[1], R02 = c2w[2];
    float R10 = c2w[4], R11 = c2w[5], R12 = c2w[6];
    float R20 = c2w[8], R21 = c2w[9], R22 = c2w[10];
    
    // Transform to world space
    float wx = R00 * dx + R01 * dy + R02 * dz;
    float wy = R10 * dx + R11 * dy + R12 * dz;
    float wz = R20 * dx + R21 * dy + R22 * dz;
    
    // Normalize direction
    float norm = rsqrtf(wx * wx + wy * wy + wz * wz);
    vec3 dir = {wx * norm, wy * norm, wz * norm};
    
    // Evaluate SH for each color channel
    scalar_t *out = colors + idx * 3;
    
#pragma unroll
    for (uint32_t c = 0; c < 3; ++c) {
        sh_coeffs_to_color_fast(degree, c, dir, sh_coeffs_shared, out);
    }
    
    // Clamp to valid range
#pragma unroll
    for (uint32_t c = 0; c < 3; ++c) {
        out[c] = fminf(fmaxf(out[c], 0.0f), 1.0f);
    }
}

template <typename scalar_t>
__global__ void sh_background_bwd_kernel(
    const uint32_t B,
    const uint32_t H,
    const uint32_t W,
    const uint32_t K,
    const uint32_t degree,
    const scalar_t *__restrict__ camtoworlds,
    const scalar_t *__restrict__ Ks,
    const scalar_t *__restrict__ sh_coeffs,
    const scalar_t *__restrict__ v_colors,     // [B, H, W, 3]
    scalar_t *__restrict__ v_sh_coeffs         // [K, 3]
) {
    // Load sh_coeffs into shared memory
    extern __shared__ char smem[];
    scalar_t *sh_coeffs_shared = reinterpret_cast<scalar_t *>(smem);
    
    auto block = cg::this_thread_block();
    uint32_t tr = block.thread_rank();
    uint32_t block_size = block.size();
    
    // Cooperatively load sh_coeffs (K * 3 elements)
    for (uint32_t i = tr; i < K * 3; i += block_size) {
        sh_coeffs_shared[i] = sh_coeffs[i];
    }
    block.sync();
    
    uint32_t idx = cg::this_grid().thread_rank();
    uint32_t total_pixels = B * H * W;
    
    if (idx >= total_pixels) {
        return;
    }
    
    uint32_t b = idx / (H * W);
    uint32_t hw = idx % (H * W);
    uint32_t i = hw / W;
    uint32_t j = hw % W;
    
    bool valid = (i < H && j < W);
    
    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    
    const scalar_t *cam_K = Ks + b * 9;
    float fx = cam_K[0];
    float fy = cam_K[4];
    float cx = cam_K[2];
    float cy = cam_K[5];
    
    float dx = (px - cx) / fx;
    float dy = (py - cy) / fy;
    float dz = 1.0f;
    
    const scalar_t *c2w = camtoworlds + b * 16;
    float R00 = c2w[0], R01 = c2w[1], R02 = c2w[2];
    float R10 = c2w[4], R11 = c2w[5], R12 = c2w[6];
    float R20 = c2w[8], R21 = c2w[9], R22 = c2w[10];
    
    float wx = R00 * dx + R01 * dy + R02 * dz;
    float wy = R10 * dx + R11 * dy + R12 * dz;
    float wz = R20 * dx + R21 * dy + R22 * dz;
    
    float norm = rsqrtf(wx * wx + wy * wy + wz * wz);
    vec3 dir = {wx * norm, wy * norm, wz * norm};
    
    const scalar_t *v_color = v_colors + idx * 3;
    
    // Local gradient accumulators for all coefficients and channels
    scalar_t v_coeffs_local[25 * 3] = {0.f};  // Max degree 4 -> 25 coeffs, 3 channels
    
    if (valid) {
        // Compute gradients for all color channels
#pragma unroll
        for (uint32_t c = 0; c < 3; ++c) {
            sh_coeffs_to_color_fast_vjp(
                degree,
                c,
                dir,
                sh_coeffs_shared,
                v_color,
                v_coeffs_local,
                nullptr
            );
        }
    }
    
    // Warp-level reduction
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    if (!warp.any(valid)) {
        return;
    }
    
    for (uint32_t k = 0; k < K; ++k) {
#pragma unroll
        for (uint32_t c = 0; c < 3; ++c) {
            float val = v_coeffs_local[k * 3 + c];
#pragma unroll
            for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
                val += warp.shfl_down(val, offset);
            }
            v_coeffs_local[k * 3 + c] = val;
        }
    }
    
    // Only first thread in warp writes to global memory
    if (warp.thread_rank() == 0) {
        for (uint32_t k = 0; k < K; ++k) {
#pragma unroll
            for (uint32_t c = 0; c < 3; ++c) {
                gpuAtomicAdd(v_sh_coeffs + k * 3 + c, v_coeffs_local[k * 3 + c]);
            }
        }
    }
}

void launch_sh_background_fwd_kernel(
    const at::Tensor camtoworlds,
    const at::Tensor Ks,
    const at::Tensor sh_coeffs,
    const uint32_t width,
    const uint32_t height,
    const uint32_t degree,
    at::Tensor colors
) {
    uint32_t B = camtoworlds.size(0);
    uint32_t H = height;
    uint32_t W = width;
    uint32_t K = sh_coeffs.size(0);
    
    uint32_t total_pixels = B * H * W;
    dim3 threads(256);
    dim3 grid((total_pixels + threads.x - 1) / threads.x);
    
    if (total_pixels == 0) {
        return;
    }
    
    AT_DISPATCH_FLOATING_TYPES(
        sh_coeffs.scalar_type(),
        "sh_background_fwd_kernel",
        [&]() {
            sh_background_fwd_kernel<scalar_t>
                <<<grid, threads, K * 3 * sizeof(scalar_t), at::cuda::getCurrentCUDAStream()>>>(
                    B, H, W, K, degree,
                    camtoworlds.data_ptr<scalar_t>(),
                    Ks.data_ptr<scalar_t>(),
                    sh_coeffs.data_ptr<scalar_t>(),
                    colors.data_ptr<scalar_t>()
                );
        }
    );
}

void launch_sh_background_bwd_kernel(
    const at::Tensor camtoworlds,
    const at::Tensor Ks,
    const at::Tensor sh_coeffs,
    const uint32_t width,
    const uint32_t height,
    const uint32_t degree,
    const at::Tensor v_colors,
    at::Tensor v_sh_coeffs
) {
    uint32_t B = camtoworlds.size(0);
    uint32_t H = height;
    uint32_t W = width;
    uint32_t K = sh_coeffs.size(0);
    
    uint32_t total_pixels = B * H * W;
    dim3 threads(256);
    dim3 grid((total_pixels + threads.x - 1) / threads.x);
    
    if (total_pixels == 0) {
        return;
    }
    
    AT_DISPATCH_FLOATING_TYPES(
        sh_coeffs.scalar_type(),
        "sh_background_bwd_kernel",
        [&]() {
            sh_background_bwd_kernel<scalar_t>
                <<<grid, threads, K * 3 * sizeof(scalar_t), at::cuda::getCurrentCUDAStream()>>>(
                    B, H, W, K, degree,
                    camtoworlds.data_ptr<scalar_t>(),
                    Ks.data_ptr<scalar_t>(),
                    sh_coeffs.data_ptr<scalar_t>(),
                    v_colors.data_ptr<scalar_t>(),
                    v_sh_coeffs.data_ptr<scalar_t>()
                );
        }
    );
}

} // namespace gsplat