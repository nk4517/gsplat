#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Utils.cuh"
#include "Rasterization.h"
#include "MipFilter2DGS.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////

/**
 * 2DGS Forward Rasterization with Weighted Sum (GaussianImage adaptation)
 * ========================================================================
 * 
 * This implementation is based on the GaussianImage paper:
 * "GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting"
 * Paper: https://arxiv.org/pdf/2403.08551
 * Code: https://github.com/Xinjie-Q/GaussianImage
 * 
 * This is an adaptation of the GaussianImage accumulated summation approach
 * for rendering on arbitrary surfaces in 3D space with camera projection.
 * While GaussianImage works with 2D gaussians directly in image space, this implementation
 * uses 2D gaussian splats (2DGS) attached to surfaces in 3D space and renders them using weighted sum.
 * 
 * Key differences from standard 2DGS:
 * 1. No transmittance tracking (T) - each splat contributes independently
 * 2. Simple weighted sum: pixel_color = Σ(alpha_i * color_i)
 * 3. No early stopping - all splats in tile are processed
 * 4. Only 3D gaussian kernel (no 2D lowpass or antialiasing)
 * 
 * Gradient computation simplification:
 * ------------------------------------
 * Standard alpha compositing gradient:
 *   ∂L/∂alpha_i = ∂L/∂C * color_i * T_i - ∂L/∂C * Σ(j>i)[alpha_j * color_j * T_j / (1 - alpha_i)]
 *   where T_i = Π(j<i)[1 - alpha_j] is accumulated transmittance
 *   
 *   This creates complex dependencies between splats due to transmittance chain.
 *   Each splat's gradient depends on all splats before and after it in depth order.
 * 
 * Weighted sum gradient:
 *   ∂L/∂alpha_i = ∂L/∂C * color_i
 *   ∂L/∂color_i = ∂L/∂C * alpha_i
 *   
 *   Gradients are completely independent - no inter-splat dependencies!
 *   This enables:
 *   - Simpler backward pass (no need to track transmittance)
 *   - Better parallelization (no sequential dependencies)
 * 
 * Trade-offs:
 * - Loss of occlusion handling (no depth ordering effect)
 * - All splats contribute regardless of opacity
 * - May require different regularization strategies
 * 
 * This approach is intended for:
 * - Splatting on surfaces in 3D space (arbitrary planes, sphere interiors for sky, etc.)
 * - Scenarios where splats represent surface textures rather than volumetric density
 * 
 * NOT intended for:
 * - Volumetric rendering with proper alpha compositing
 * - Scenes requiring accurate depth-based occlusion
 */

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_2dgs_wsum_fwd_kernel(
    const uint32_t I,        // number of images
    const uint32_t N,        // number of gaussians
    const uint32_t n_isects, // number of ray-primitive intersections.
    const bool packed,       // whether the input tensors are packed
    const vec2
        *__restrict__ means2d, // Projected Gaussian means. [..., N, 2] if
                               // packed is False, [nnz, 2] if packed is True.
    const scalar_t
        *__restrict__ ray_transforms, // transformation matrices that transforms
                                      // xy-planes in pixel spaces into splat
                                      // coordinates. [..., N, 3, 3] if packed is
                                      // False, [nnz, channels] if packed is
                                      // True. This is (KWH)^{-1} in the paper
                                      // (takes screen [x,y] and map to [u,v])
    const scalar_t *__restrict__ colors,    // [..., N, CDIM] or [nnz, CDIM]  //
                                            // Gaussian colors or ND features.
    const scalar_t *__restrict__ opacities, // [..., N] or [nnz] // Gaussian
                                            // opacities that support per-view
                                            // values.
    const scalar_t *__restrict__ normals, // [..., N, 3] or [nnz, 3] // The
                                          // normals in camera space.
    const scalar_t *__restrict__ backgrounds, // [..., CDIM] // Background colors
                                              // on camera basis
    const bool *__restrict__ masks, // [..., tile_height, tile_width] // Optional
                                    // tile mask to skip rendering GS to masked
                                    // tiles.
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t
        *__restrict__ tile_offsets, // [..., tile_height, tile_width]    //
                                    // Intersection offsets outputs from
                                    // `isect_offset_encode()`, this is the
                                    // result of a prefix sum, and gives the
                                    // interval that our gaussians are gonna
                                    // use.
    const int32_t *__restrict__ flatten_ids, // [n_isects] // The global flatten
                                             // indices in [I * N] or [nnz] from
                                             // `isect_tiles()`.

    // outputs
    scalar_t
        *__restrict__ render_colors, // [..., image_height, image_width, CDIM]
    scalar_t *__restrict__ render_alphas,  // [..., image_height, image_width, 1]
    scalar_t *__restrict__ render_normals, // [..., image_height, image_width, 3]
    scalar_t *__restrict__ render_distort, // [..., image_height, image_width, 1]
                                           // // Stores the per-pixel distortion
                                           // error proposed in Mip-NeRF 360.
    scalar_t
        *__restrict__ render_median, // [..., image_height, image_width, 1]  //
                                     // Stores the median depth contribution for
                                     // each pixel "set to the depth of the
                                     // Gaussian that brings the accumulated
                                     // opacity over 0.5."
    int32_t *__restrict__ last_ids,  // [..., image_height, image_width]     //
                                     // Stores the index of the last Gaussian
                                     // that contributed to each pixel.
    int32_t *__restrict__ median_ids, // [..., image_height, image_width]    //
                                     // Stores the index of the Gaussian that
                                     // contributes to the median depth for each
                                     // pixel (bring over 0.5).
    // Additional outputs for dominating gaussian tracking
    int32_t *__restrict__ n_touched,   // [..., N] or [nnz] // Number of pixels touched by each gaussian, in each image
    int32_t *__restrict__ n_dominated, // [..., N] or [nnz] // Number of pixels dominated by each gaussian, in each image
    int32_t *__restrict__ dominating_gauss_ids, // [..., image_height, image_width]
    scalar_t *__restrict__ dominating_weights,  // [..., image_height, image_width]
    scalar_t *__restrict__ dominating_depths    // [..., image_height, image_width]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    /**
     * ==============================
     * Thread and block setup:
     * This sets up the thread and block indices, determining which image,
     * tile, and pixel each thread will process. The grid structure is assigend
     * as: I * tile_height * tile_width blocks (3d grid), each block is a tile.
     * Each thread is responsible for one pixel. (blockSize = tile_size *
     * tile_size)
     * ==============================
     */
    auto block = cg::this_thread_block();
    int32_t image_id = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets +=
        image_id * tile_height *
        tile_width; // get the global offset of the tile w.r.t the image
    render_colors +=
        image_id * image_height * image_width *
        CDIM; // get the global offset of the pixel w.r.t the image
    render_alphas +=
        image_id * image_height *
        image_width; // get the global offset of the pixel w.r.t the image
    last_ids +=
        image_id * image_height *
        image_width; // get the global offset of the pixel w.r.t the image
    render_normals += image_id * image_height * image_width * 3;
    render_distort += image_id * image_height * image_width;
    render_median += image_id * image_height * image_width;
    median_ids += image_id * image_height * image_width;
    
    // Additional outputs for dominating gaussian tracking
    if (!packed && dominating_gauss_ids != nullptr) {
        dominating_gauss_ids += image_id * image_height * image_width;
    }
    if (!packed && dominating_weights != nullptr) {
        dominating_weights += image_id * image_height * image_width;
    }
    if (!packed && dominating_depths != nullptr) {
        dominating_depths += image_id * image_height * image_width;
    }
    // n_touched and n_dominated are NOT offset - g from flatten_ids is already a global index
    // so we use g directly to index into the full [I*N] sized arrays

    // get the global offset of the background and mask
    if (backgrounds != nullptr) {
        backgrounds += image_id * CDIM;
    }
    if (masks != nullptr) {
        masks += image_id * tile_height * tile_width;
    }

    // find the center of the pixel
    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // when the mask is provided, render the background color and return
    // if this tile is labeled as False
    if (masks != nullptr && inside && !masks[tile_id]) {
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] =
                backgrounds == nullptr ? 0.0f : backgrounds[k];
        }
        return;
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile

    // print
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        // see if this is the last tile in the image
        (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    /**
     * ==============================
     * Register computing variables:
     * For each pixel, we need to find its uv intersection with the gaussian
     * primitives. then we retrieve the kernel's parameters and kernel weights
     * do the splatting rendering equation.
     * ==============================
     */
    // Shared memory layout:
    // This memory is laid out as follows:
    // | gaussian indices | x : y : alpha | u | v | w |
    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]

    // stores the concatination for projected primitive source (x, y) and
    // opacity alpha
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]

    // these are row vectors of the ray transformation matrices for the current
    // batch of gaussians
    vec3 *u_Ms_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]
    vec3 *v_Ms_batch =
        reinterpret_cast<vec3 *>(&u_Ms_batch[block_size]); // [block_size]
    vec3 *w_Ms_batch =
        reinterpret_cast<vec3 *>(&v_Ms_batch[block_size]); // [block_size]

    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    float weighted_alpha = 0.f;  // For weighted sum of alphas

    // Variables for tracking dominating gaussian
    float max_weight = 0.f;
    int32_t dominating_gid = -1;
    float dominating_depth_val = 0.f;
    float weighted_depth = 0.f;  // For weighted sum of depths

    /**
     * ==============================
     * Per-pixel rendering: (2DGS Differntiable Rasterizer Forward Pass)
     * This section is responsible for rendering a single pixel.
     * It processes batches of gaussians and accumulates the pixel color and
     * normal.
     * ==============================
     */

    // TODO (WZ): merge pix_out and normal_out to
    //  float pix_out[CDIM + 3] = {0.f}
    float pix_out[CDIM] = {0.f};
    float normal_out[3] = {0.f};
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;

        // only threads within the range of the tile will fetch gaussians
        /**
         * Launch this block with each thread responsible for one gaussian.
         */
        if (idx < range_end) {
            int32_t g = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
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
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        /**
         * ==================================================
         * Forward rasterization pass:
         * ==================================================
         *
         * GSplat computes rasterization point of intersection as:
         * 1. Generate 2 homogeneous plane parameter vectors as sets of points
         * in UV space
         * 2. Find the set of points that satisfy both conditions with the cross
         * product
         * 3. Find where this solution set intersects with UV plane using
         * projective flattening
         *
         * For each gaussian G_i and pixel q_xy:
         *
         * 1. Compute homogeneous plane parameters:
         *    h_u = p_x * M_w - M_u
         *    h_v = p_y * M_w - M_v
         *    where M_u, M_v, M_w are rows of the KWH transform
         *
         * Note: this works because:
         *    for any vector q_uv [u, v, 1], applying co-vector h_u will yield
         * the following expression: h_u * [u, v, 1]^T = P_x * (M_w * q_uv) -
         * M_u * q_uv = P_x * q_ray.z - q_ray.x * q_ray.z
         *    - where P_x is the x-coordinate of the ray origin
         *    Thus: h_u  defines a set of q_uv where q_uv's projected x
         * coordinate in ray space is P_x which aligns with the homogeneous
         * plane definition in original 2DGS paper (similar for h_v)
         *
         * 2. Compute intersection:
         *    zeta = h_u × h_v
         *    This cross product is the only solution that satisfies both
         * homogeneous plane equations (dot product == 0)
         *
         * 3. Project to UV space:
         *    s_uv = [zeta_1/zeta_3, zeta_2/zeta_3]
         *    - since UV space is essentially another ray space, and arbitrary
         * scale of q_uv will not change the result of dot product over
         * orthogonality
         *    - thus, the result is the point of intersection in UV space
         *
         * 4. Evaluate gaussian kernel:
         *    G_i = exp(-(s_u^2 + s_v^2)/2)
         *
         * 5. Accumulate color:
         *    p_xy += alpha_i * c_i * G_i
         *
         * This method efficiently computes the point of intersection and
         * evaluates the gaussian kernel in UV space.
         */
        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {

            const vec3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;

            const vec3 u_M = u_Ms_batch[t];
            const vec3 v_M = v_Ms_batch[t];
            const vec3 w_M = w_Ms_batch[t];

            // h_u and h_v are the homogeneous plane representations (they are
            // contravariant to the points on the primitive plane)
            const vec3 h_u = px * w_M - u_M;
            const vec3 h_v = py * w_M - v_M;

            const vec3 ray_cross = glm::cross(h_u, h_v);

            const float RAY_CROSS_EPSILON = 1e-8;
            if (abs(ray_cross.z) < RAY_CROSS_EPSILON)
                continue;

            const vec2 s =
                vec2(ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z);

            // IMPORTANT: This is where the gaussian kernel is evaluated!!!!!

            // Simple 3D gaussian weight calculation
            const float gauss_weight_3d = s.x * s.x + s.y * s.y;

            const float sigma = 0.5f * gauss_weight_3d;
            // evaluation of the gaussian exponential term
            float alpha = min(0.999f, opac * __expf(-sigma));

            // ignore transparent gaussians
            if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                continue;
            }

            // run volumetric rendering..
            int32_t g = id_batch[t];
            const float vis = alpha;  // Simple weighted sum
            const float *c_ptr = colors + g * CDIM;

            // Вычисление глубины через пересечение луча и плоскости сплата
             const float depth_at_pixel = s.x * w_M.x + s.y * w_M.y + w_M.z;
            // не, лучше не надо, иначе сплаты перехлёстываются и "мигают"
            // upd. они и так и эдак мигают

//            // Get central point depth from the last channel of colors
//            const float depth_at_pixel = c_ptr[CDIM - 1];

            // Track dominating gaussian (the one that consumes most transmittance)
            if (alpha > max_weight) {
                max_weight = alpha;
                dominating_gid = g;
                dominating_depth_val = depth_at_pixel;
            }

#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                pix_out[k] += c_ptr[k] * vis;
            }

            const float *n_ptr = normals + g * 3;
#pragma unroll
            for (uint32_t k = 0; k < 3; ++k) {
                normal_out[k] += n_ptr[k] * vis;
            }

            // Accumulate weighted depth
            weighted_depth += depth_at_pixel * vis;

            // Accumulate weighted sum of alphas
            weighted_alpha += alpha;

            // Track touched gaussians
            if (n_touched != nullptr) {
                atomicAdd(&n_touched[g], 1);
            }


            cur_idx = batch_start + t;
        }
    }
    if (inside) {
        // Store weighted sum of alphas
        render_alphas[pix_id] = weighted_alpha;
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] = pix_out[k];
        }
#pragma unroll
        for (uint32_t k = 0; k < 3; ++k) {
            render_normals[pix_id * 3 + k] = normal_out[k];
        }
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);

        if (render_distort != nullptr) {
            render_distort[pix_id] = 0.0f; // Not used in weighted sum
        }

        // Store weighted average depth in median field
        render_median[pix_id] = weighted_depth;
        // index in bin of gaussian that contributes to median depth
        // Not used in weighted sum
        // ЗАПИСЫВАТЬ -1 НЕЛЬЗЯ, ОНО ИСПОЛЬЗУЕТСЯ ГДЕ-ТО ЕЩЁ
        median_ids[pix_id] = dominating_gid >= 0 ? dominating_gid : 0;

        // Write dominating gaussian information
        if (dominating_gauss_ids != nullptr) {
            dominating_gauss_ids[pix_id] = dominating_gid;
        }
        if (dominating_weights != nullptr) {
            dominating_weights[pix_id] = max_weight;
        }
        if (dominating_depths != nullptr) {
            dominating_depths[pix_id] = dominating_depth_val;
        }

        // Update n_dominated counter for the dominating gaussian
        if (n_dominated != nullptr && dominating_gid >= 0) {
            atomicAdd(&n_dominated[dominating_gid], 1);
        }
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_2dgs_wsum_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,        // [..., N, 2] or [nnz, 2]
    const at::Tensor ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor colors,         // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities,      // [..., N]  or [nnz]
    const at::Tensor normals,        // [..., N, 3] or [nnz, 3]
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // outputs
    at::Tensor renders,        // [..., image_height, image_width, channels]
    at::Tensor alphas,         // [..., image_height, image_width]
    at::Tensor render_normals, // [..., image_height, image_width, 3]
    at::Tensor render_distort, // [..., image_height, image_width]
    at::Tensor render_median,  // [..., image_height, image_width]
    at::Tensor last_ids,       // [..., image_height, image_width]
    at::Tensor median_ids,     // [..., image_height, image_width]
    // Additional outputs for dominating gaussian tracking
    at::Tensor n_touched,      // [..., N] or [nnz] // Number of pixels touched by each gaussian
    at::Tensor n_dominated,    // [..., N] or [nnz] // Number of pixels dominated by each gaussian
    at::Tensor dominating_gauss_ids, // [..., image_height, image_width]
    at::Tensor dominating_weights,  // [..., image_height, image_width]
    at::Tensor dominating_depths    // [..., image_height, image_width]
) {
    bool packed = means2d.dim() == 2;

    uint32_t N = packed ? 0 : means2d.size(-2); // number of gaussians
    uint32_t I = alphas.numel() / (image_height * image_width); // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // I * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {I, tile_height, tile_width};

    int64_t shmem_size = tile_size * tile_size *
                         (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) +
                          sizeof(vec3) + sizeof(vec3));

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_2dgs_wsum_fwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_2dgs_wsum_fwd_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            I,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
            ray_transforms.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            normals.data_ptr<float>(),
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
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_normals.data_ptr<float>(),
            render_distort.data_ptr<float>(),
            render_median.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            median_ids.data_ptr<int32_t>(),
            n_touched.numel() > 0 ? n_touched.data_ptr<int32_t>() : nullptr,
            n_dominated.numel() > 0 ? n_dominated.data_ptr<int32_t>() : nullptr,
            dominating_gauss_ids.numel() > 0 ? dominating_gauss_ids.data_ptr<int32_t>() : nullptr,
            dominating_weights.numel() > 0 ? dominating_weights.data_ptr<float>() : nullptr,
            dominating_depths.numel() > 0 ? dominating_depths.data_ptr<float>() : nullptr
        );
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file.
// TODO: this is slow to compile, can we do something about it?
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_2dgs_wsum_fwd_kernel<CDIM>(            \
        const at::Tensor means2d,                                              \
        const at::Tensor ray_transforms,                                       \
        const at::Tensor colors,                                               \
        const at::Tensor opacities,                                            \
        const at::Tensor normals,                                              \
        const at::optional<at::Tensor> backgrounds,                            \
        const at::optional<at::Tensor> masks,                                  \
        uint32_t image_width,                                                  \
        uint32_t image_height,                                                 \
        uint32_t tile_size,                                                    \
        const at::Tensor tile_offsets,                                         \
        const at::Tensor flatten_ids,                                          \
        at::Tensor renders,                                                    \
        at::Tensor alphas,                                                     \
        at::Tensor render_normals,                                             \
        at::Tensor render_distort,                                             \
        at::Tensor render_median,                                              \
        at::Tensor last_ids,                                                   \
        at::Tensor median_ids,                                                 \
        at::Tensor n_touched,                                                  \
        at::Tensor n_dominated,                                                \
        at::Tensor dominating_gauss_ids,                                       \
        at::Tensor dominating_weights,                                         \
        at::Tensor dominating_depths                                           \
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
