#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

/**
 * 2DGS Backward Rasterization with Weighted Sum (GaussianImage adaptation)
 * =========================================================================
 * 
 * This implementation is based on the GaussianImage paper:
 * "GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting"
 * Paper: https://arxiv.org/pdf/2403.08551
 * Code: https://github.com/Xinjie-Q/GaussianImage
 * 
 * This is an adaptation of the GaussianImage accumulated summation approach
 * for rendering on arbitrary surfaces in 3D space with camera projection.
 * This backward pass implements gradients for the weighted sum rasterization of 2D gaussian splats (2DGS).
 * 
 * Key simplifications from standard 2DGS backward:
 * 1. No transmittance tracking - gradients are independent
 * 2. No buffer accumulation for later gaussians
 * 3. Simple gradient computation:
 *    ∂L/∂alpha_i = ∂L/∂C * color_i + ∂L/∂N * normal_i
 *    ∂L/∂color_i = ∂L/∂C * alpha_i
 *    ∂L/∂normal_i = ∂L/∂N * alpha_i
 * 4. No early stopping - process all gaussians
 * 5. Only 3D gaussian kernel (no 2D lowpass or antialiasing)
 * 
 * This dramatically simplifies the backward pass as there are no
 * inter-gaussian dependencies through transmittance.
 */

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_2dgs_wsum_bwd_kernel(
    const uint32_t I,        // number of images
    const uint32_t N,        // number of gaussians
    const uint32_t n_isects, // number of ray-primitive intersections
    const bool packed,       // whether the input tensors are packed
    // fwd inputs
    const vec2 *__restrict__ means2d,           // [..., N, 2] or [nnz, 2]
    const scalar_t *__restrict__ ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    const scalar_t *__restrict__ colors,        // [..., N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ normals,       // [..., N, 3] or [nnz, 3]
    const scalar_t *__restrict__ opacities,     // [..., N] or [nnz]
    const scalar_t *__restrict__ backgrounds,   // [..., CDIM]
    const bool *__restrict__ masks,             // [..., tile_height, tile_width]

    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [..., tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]

    // fwd outputs (not used in weighted sum)
    const scalar_t *__restrict__ render_colors,  // [..., image_height, image_width, CDIM]
    const scalar_t *__restrict__ render_alphas,  // [..., image_height, image_width, 1]
    const int32_t *__restrict__ last_ids,        // [..., image_height, image_width]
    const int32_t *__restrict__ median_ids,      // [..., image_height, image_width]

    // grad outputs
    const scalar_t *__restrict__ v_render_colors,  // [..., image_height, image_width, CDIM]
    const scalar_t *__restrict__ v_render_alphas,  // [..., image_height, image_width, 1]
    const scalar_t *__restrict__ v_render_normals, // [..., image_height, image_width, 3]
    const scalar_t *__restrict__ v_render_distort, // [..., image_height, image_width, 1]
    const scalar_t *__restrict__ v_render_median,  // [..., image_height, image_width, 1]

    // grad inputs
    vec2 *__restrict__ v_means2d_abs,        // [..., N, 2] or [nnz, 2]
    vec2 *__restrict__ v_means2d,            // [..., N, 2] or [nnz, 2]
    scalar_t *__restrict__ v_ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    scalar_t *__restrict__ v_colors,         // [..., N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities,      // [..., N] or [nnz]
    scalar_t *__restrict__ v_normals,        // [..., N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_densify         // [..., N, 5] or [nnz, 5]
) {
    auto block = cg::this_thread_block();
    uint32_t image_id = block.group_index().x;
    uint32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += image_id * tile_height * tile_width;
    v_render_colors += image_id * image_height * image_width * CDIM;
    v_render_alphas += image_id * image_height * image_width;
    v_render_normals += image_id * image_height * image_width * 3;

    if (masks != nullptr) {
        masks += image_id * tile_height * tile_width;
    }

    // Skip masked tiles
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    bool inside = (i < image_height && j < image_width);

    // Get tile range
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end = (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
                            ? n_isects
                            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    // Shared memory for batching gaussians
    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3 *xy_opacity_batch = reinterpret_cast<vec3 *>(&id_batch[block_size]);
    vec3 *u_Ms_batch = reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]);
    vec3 *v_Ms_batch = reinterpret_cast<vec3 *>(&u_Ms_batch[block_size]);
    vec3 *w_Ms_batch = reinterpret_cast<vec3 *>(&v_Ms_batch[block_size]);
    float *rgbs_batch = (float *)&w_Ms_batch[block_size];
    float *normals_batch = &rgbs_batch[block_size * CDIM];

    // Fetch gradients for this pixel
    float v_render_c[CDIM];
    float v_render_n[3];

    if (inside) {
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            v_render_c[k] = v_render_colors[pix_id * CDIM + k];
        }
#pragma unroll
        for (uint32_t k = 0; k < 3; ++k) {
            v_render_n[k] = v_render_normals[pix_id * 3 + k];
        }
    }

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // For weighted sum, we don't need bin_final tracking since no early stopping
    // But we can still use warp-level optimizations for efficiency

    // Process gaussians in batches
    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();

        // Load gaussian data
        // Compute indices for backward pass (from near to far)
        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t idx = batch_end - tr;

        if (idx >= range_start) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};

            u_Ms_batch[tr] = {
                ray_transforms[g * 9],
                ray_transforms[g * 9 + 1],
                ray_transforms[g * 9 + 2]
            };
            v_Ms_batch[tr] = {
                ray_transforms[g * 9 + 3],
                ray_transforms[g * 9 + 4],
                ray_transforms[g * 9 + 5]
            };
            w_Ms_batch[tr] = {
                ray_transforms[g * 9 + 6],
                ray_transforms[g * 9 + 7],
                ray_transforms[g * 9 + 8]
            };
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
#pragma unroll
            for (uint32_t k = 0; k < 3; ++k) {
                normals_batch[tr * 3 + k] = normals[g * 3 + k];
            }
        }

        block.sync();

        // Process each gaussian in batch
        uint32_t batch_size = min(block_size, batch_end - range_start + 1);
        for (uint32_t t = 0; t < batch_size; ++t) {
            
            bool valid = inside && (batch_end - t >= range_start);
            
            // Local gradient accumulators
            float v_rgb_local[CDIM] = {0.f};
            float v_normal_local[3] = {0.f};
            vec3 v_u_M_local = {0.f, 0.f, 0.f};
            vec3 v_v_M_local = {0.f, 0.f, 0.f};
            vec3 v_w_M_local = {0.f, 0.f, 0.f};
            float v_opacity_local = 0.f;
            float v_G_local = 0.f;
            vec4 v_densify_local = {0.f, 0.f, 0.f, 0.f};

            if (valid) {
                vec3 xy_opac = xy_opacity_batch[t];
                float opac = xy_opac.z;

                vec3 u_M = u_Ms_batch[t];
                vec3 v_M = v_Ms_batch[t];
                vec3 w_M = w_Ms_batch[t];

                vec3 h_u = px * w_M - u_M;
                vec3 h_v = py * w_M - v_M;
                vec3 ray_cross = glm::cross(h_u, h_v);

                const float RAY_CROSS_EPSILON = 1e-8;
                if (abs(ray_cross.z) < RAY_CROSS_EPSILON)
                    valid = false;

                if (valid) {
                    vec2 s = {ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z};
                    
                    // Simple 3D gaussian weight
                    float gauss_weight_3d = s.x * s.x + s.y * s.y;
                    float sigma = 0.5f * gauss_weight_3d;
                    float vis = __expf(-sigma);
                    float alpha = min(0.999f, opac * vis);

                    if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                        valid = false;
                    }

                    if (valid) {
                        // Simplified gradients for weighted sum
                        // ∂L/∂color_i = ∂L/∂C * alpha_i
#pragma unroll
                        for (uint32_t k = 0; k < CDIM; ++k) {
                            v_rgb_local[k] = alpha * v_render_c[k];
                        }

                        // ∂L/∂normal_i = ∂L/∂N * alpha_i
#pragma unroll
                        for (uint32_t k = 0; k < 3; ++k) {
                            v_normal_local[k] = alpha * v_render_n[k];
                        }

                        // ∂L/∂alpha_i = ∂L/∂C * color_i + ∂L/∂N * normal_i
                        float v_alpha = 0.f;
#pragma unroll
                        for (uint32_t k = 0; k < CDIM; ++k) {
                            v_alpha += rgbs_batch[t * CDIM + k] * v_render_c[k];
                        }
#pragma unroll
                        for (uint32_t k = 0; k < 3; ++k) {
                            v_alpha += normals_batch[t * 3 + k] * v_render_n[k];
                        }

                        // Gradient through gaussian weight
                        float v_G = opac * v_alpha;
                        v_G_local = v_G;
                        
                        // Gradient through s
                        vec2 v_s = {
                            -v_G * vis * s.x,
                            -v_G * vis * s.y
                        };

                        // Gradient through ray intersection
                        float v_sx_pz = v_s.x / ray_cross.z;
                        float v_sy_pz = v_s.y / ray_cross.z;
                        vec3 v_ray_cross = {
                            v_sx_pz, v_sy_pz, -(v_sx_pz * s.x + v_sy_pz * s.y)
                        };
                        vec3 v_h_u = glm::cross(h_v, v_ray_cross);
                        vec3 v_h_v = glm::cross(v_ray_cross, h_u);

                        v_u_M_local = {-v_h_u.x, -v_h_u.y, -v_h_u.z};
                        v_v_M_local = {-v_h_v.x, -v_h_v.y, -v_h_v.z};
                        v_w_M_local = {
                            px * v_h_u.x + py * v_h_v.x,
                            px * v_h_u.y + py * v_h_v.y,
                            px * v_h_u.z + py * v_h_v.z
                        };

                        v_opacity_local = vis * v_alpha;

                        // Densification gradients
                        float depth = w_M.z;
                        v_densify_local.x = v_u_M_local.z * depth;
                        v_densify_local.y = v_v_M_local.z * depth;
                        v_densify_local.z = abs(v_u_M_local.z) * depth;
                        v_densify_local.w = abs(v_v_M_local.z) * depth;
                    }
                }
            }

            // Skip if no threads in warp have valid gradients
            if (!warp.any(valid)) {
                continue;
            }

            // Warp-level reduction
            warpSum<CDIM>(v_rgb_local, warp);
            warpSum<3>(v_normal_local, warp);
            warpSum(v_u_M_local, warp);
            warpSum(v_v_M_local, warp);
            warpSum(v_w_M_local, warp);
            warpSum(v_opacity_local, warp);
            warpSum(v_G_local, warp);
            warpSum(v_densify_local, warp);

            int32_t g = id_batch[t];

            // Write gradients to global memory
            if (warp.thread_rank() == 0) {
                float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_normal_ptr = (float *)(v_normals) + 3 * g;
#pragma unroll
                for (uint32_t k = 0; k < 3; ++k) {
                    gpuAtomicAdd(v_normal_ptr + k, v_normal_local[k]);
                }

                float *v_ray_transforms_ptr = (float *)(v_ray_transforms) + 9 * g;
                gpuAtomicAdd(v_ray_transforms_ptr, v_u_M_local.x);
                gpuAtomicAdd(v_ray_transforms_ptr + 1, v_u_M_local.y);
                gpuAtomicAdd(v_ray_transforms_ptr + 2, v_u_M_local.z);
                gpuAtomicAdd(v_ray_transforms_ptr + 3, v_v_M_local.x);
                gpuAtomicAdd(v_ray_transforms_ptr + 4, v_v_M_local.y);
                gpuAtomicAdd(v_ray_transforms_ptr + 5, v_v_M_local.z);
                gpuAtomicAdd(v_ray_transforms_ptr + 6, v_w_M_local.x);
                gpuAtomicAdd(v_ray_transforms_ptr + 7, v_w_M_local.y);
                gpuAtomicAdd(v_ray_transforms_ptr + 8, v_w_M_local.z);

                gpuAtomicAdd(v_opacities + g, v_opacity_local);

                if (v_densify != nullptr) {
                    float *v_densify_ptr = (float *)(v_densify) + 5 * g;
                    gpuAtomicAdd(v_densify_ptr, v_densify_local.x);
                    gpuAtomicAdd(v_densify_ptr + 1, v_densify_local.y);
                    gpuAtomicAdd(v_densify_ptr + 2, v_densify_local.z);
                    gpuAtomicAdd(v_densify_ptr + 3, v_densify_local.w);
                    gpuAtomicAdd(v_densify_ptr + 4, v_G_local * v_G_local);
                }
            }
        }
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_2dgs_wsum_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor ray_transforms,            // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::Tensor normals,                   // [..., N, 3] or [nnz, 3]
    const at::Tensor densify,                   // [..., N, 5] or [nnz, 5]
    const at::optional<at::Tensor> backgrounds, // [..., CDIM]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_colors, // [..., image_height, image_width, CDIM]
    const at::Tensor render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    const at::Tensor median_ids,    // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors,  // [..., image_height, image_width, 3]
    const at::Tensor v_render_alphas,  // [..., image_height, image_width, 1]
    const at::Tensor v_render_normals, // [..., image_height, image_width, 3]
    const at::Tensor v_render_distort, // [..., image_height, image_width, 1]
    const at::Tensor v_render_median,  // [..., image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_ray_transforms,            // [..., N, 3, 3] or [nnz, 3, 3]
    at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_opacities,                 // [..., N] or [nnz]
    at::Tensor v_normals,                   // [..., N, 3] or [nnz, 3]
    at::Tensor v_densify                    // [..., N, 5] or [nnz, 5]
) {
    bool packed = means2d.dim() == 2;

    uint32_t N = packed ? 0 : means2d.size(-2);
    uint32_t I = render_alphas.numel() / (image_height * image_width);
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.size(0);

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {I, tile_height, tile_width};

    int64_t shmem_size =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(vec3) +
         sizeof(vec3) + sizeof(float) * CDIM + sizeof(float) * 3);

    if (n_isects == 0) {
        return;
    }

    if (cudaFuncSetAttribute(
            rasterize_to_pixels_2dgs_wsum_bwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_2dgs_wsum_bwd_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            I,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
            ray_transforms.data_ptr<float>(),
            colors.data_ptr<float>(),
            normals.data_ptr<float>(),
            opacities.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            render_colors.data_ptr<float>(),
            render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            median_ids.data_ptr<int32_t>(),
            v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            v_render_normals.data_ptr<float>(),
            v_render_distort.data_ptr<float>(),
            v_render_median.data_ptr<float>(),
            v_means2d_abs.has_value()
                ? reinterpret_cast<vec2 *>(
                      v_means2d_abs.value().data_ptr<float>()
                  )
                : nullptr,
            reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
            v_ray_transforms.data_ptr<float>(),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>(),
            v_normals.data_ptr<float>(),
            v_densify.data_ptr<float>()
        );
}

// Explicit Instantiation
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_2dgs_wsum_bwd_kernel<CDIM>(            \
        const at::Tensor means2d,                                              \
        const at::Tensor ray_transforms,                                       \
        const at::Tensor colors,                                               \
        const at::Tensor opacities,                                            \
        const at::Tensor normals,                                              \
        const at::Tensor densify,                                              \
        const at::optional<at::Tensor> backgrounds,                            \
        const at::optional<at::Tensor> masks,                                  \
        const uint32_t image_width,                                            \
        const uint32_t image_height,                                           \
        const uint32_t tile_size,                                              \
        const at::Tensor tile_offsets,                                         \
        const at::Tensor flatten_ids,                                          \
        const at::Tensor render_colors,                                        \
        const at::Tensor render_alphas,                                        \
        const at::Tensor last_ids,                                             \
        const at::Tensor median_ids,                                           \
        const at::Tensor v_render_colors,                                      \
        const at::Tensor v_render_alphas,                                      \
        const at::Tensor v_render_normals,                                     \
        const at::Tensor v_render_distort,                                     \
        const at::Tensor v_render_median,                                      \
        at::optional<at::Tensor> v_means2d_abs,                                \
        const at::Tensor v_means2d,                                            \
        const at::Tensor v_ray_transforms,                                     \
        const at::Tensor v_colors,                                             \
        const at::Tensor v_opacities,                                          \
        const at::Tensor v_normals,                                            \
        const at::Tensor v_densify                                             \
    );

__INS__(1)
__INS__(2)
__INS__(3)
__INS__(4)
__INS__(5)
__INS__(8)
__INS__(9)
__INS__(16)
__INS__(17)
__INS__(32)
__INS__(33)
__INS__(64)
__INS__(65)
__INS__(128)
__INS__(129)
__INS__(256)
__INS__(257)
__INS__(512)
__INS__(513)
#undef __INS__

} // namespace gsplat