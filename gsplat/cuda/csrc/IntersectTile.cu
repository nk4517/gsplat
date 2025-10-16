#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

// for CUB_WRAPPER
#include <c10/cuda/CUDACachingAllocator.h>
#include <cub/cub.cuh>

#include "Common.h"
#include "Intersect.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

// Evaluate spherical harmonics bases at unit direction for high orders using
// approach described by Efficient Spherical Harmonic Evaluation, Peter-Pike
// Sloan, JCGT 2013 See https://jcgt.org/published/0002/02/06/ for reference
// implementation

__device__ __forceinline__ float
dist_sq_at_px(
    const vec3 u_M,
    const vec3 v_M,
    const vec3 w_M,
    float xy_x,                  // X координата гауссианы в кадре
    float xy_y,                  // Y координата гауссианы в кадре
    float px,                    // X координата пикселя
    float py                     // Y координата пикселя
) {

    
    // Вычисление координат в пространстве гауссианы
    const vec3 h_u = px * w_M - u_M;
    const vec3 h_v = py * w_M - v_M;
    
    const vec3 ray_cross = glm::cross(h_u, h_v);
    
    // Проверка вырожденности
    if (ray_cross.z == 0.0f)
        return 0;
    
    // Координаты s в пространстве гауссианы
    const vec2 s = {
        ray_cross.x / ray_cross.z,
        ray_cross.y / ray_cross.z
    };

    const float dist3d_sq = s.x * s.x + s.y * s.y;
    
    // 2D lowpass
    const float delta_x = px - xy_x;
    const float delta_y = py - xy_y;
    const float dist2d_lp_sq = delta_x * delta_x + delta_y * delta_y;

    const float dist_sq = min(dist3d_sq, dist2d_lp_sq);
    
    return dist_sq;
}

__device__ __forceinline__ bool
check_2dgs_tile_intersection(
    const vec3 u_M,
    const vec3 v_M,
    const vec3 w_M,
    const vec2 mean2d,
    const float tile_radius_x,
    const float tile_radius_y,
    const int32_t tile_x,
    const int32_t tile_y,
    const uint32_t tile_size
) {
    const float cutoff = 3.33f; // N sigmas for 2DGS
    const float cutoff_sq = cutoff * cutoff;
    const float tile_size_f = static_cast<float>(tile_size);
    const float area_tiles = 2.f * tile_radius_x * 2.f * tile_radius_y;
    
    // Проверяем только для больших сплатов (занимающих > 8 тайлов)
    if (area_tiles <= 8) {
        return true;
    }
    
    vec2 tile_center_pix = {
        static_cast<float>(tile_x * tile_size) + 0.5f * tile_size_f,
        static_cast<float>(tile_y * tile_size) + 0.5f * tile_size_f
    };
    
    // Проверка на 4х углах тайла
    const vec2 he1 = {-0.5f, -0.5f};
    const vec2 he2 = {0.5f, -0.5f};
    const vec2 he3 = {0.5f, 0.5f};
    const vec2 he4 = {-0.5f, 0.5f};

    const float dist_sq1 = dist_sq_at_px(
        u_M, v_M, w_M, mean2d.x, mean2d.y,
        tile_center_pix.x + he1.x * tile_size_f,
        tile_center_pix.y + he1.y * tile_size_f
    );

    const float dist_sq2 = dist_sq_at_px(
        u_M, v_M, w_M, mean2d.x, mean2d.y,
        tile_center_pix.x + he2.x * tile_size_f,
        tile_center_pix.y + he2.y * tile_size_f
    );

    const float dist_sq3 = dist_sq_at_px(
        u_M, v_M, w_M, mean2d.x, mean2d.y,
        tile_center_pix.x + he3.x * tile_size_f,
        tile_center_pix.y + he3.y * tile_size_f
    );

    const float dist_sq4 = dist_sq_at_px(
        u_M, v_M, w_M, mean2d.x, mean2d.y,
        tile_center_pix.x + he4.x * tile_size_f,
        tile_center_pix.y + he4.y * tile_size_f
    );
    
    // Проверка, что центр гауссианы находится внутри тайла
    const bool center_in_tile = 
        mean2d.x >= static_cast<float>(tile_x * tile_size) &&
        mean2d.x < static_cast<float>((tile_x + 1) * tile_size) &&
        mean2d.y >= static_cast<float>(tile_y * tile_size) &&
        mean2d.y < static_cast<float>((tile_y + 1) * tile_size);

    const bool tile_has_coverage =
        center_in_tile || (
        dist_sq1 < cutoff_sq || dist_sq2 < cutoff_sq ||
        dist_sq3 < cutoff_sq || dist_sq4 < cutoff_sq
        );

    return tile_has_coverage;
}

template <typename scalar_t>
__global__ void intersect_tile_kernel(
    // if the data is [...,  N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over I * N, only used if packed is False
    const uint32_t I,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ image_ids,    // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const scalar_t *__restrict__ means2d,            // [..., N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,               // [..., N, 2] or [nnz, 2]
    const scalar_t *__restrict__ depths,             // [..., N] or [nnz]
    const scalar_t *__restrict__ ray_transforms_2dgs, // [..., N, 9] or [nnz, 9] optional
    const int64_t *__restrict__ cum_tiles_per_gauss, // [..., N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    const uint32_t image_n_bits,
    int32_t *__restrict__ tiles_per_gauss, // [..., N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
) {
    // parallelize over I * N.
    uint32_t idx = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : I * N)) {
        return;
    }

    const float radius_x = radii[idx * 2];
    const float radius_y = radii[idx * 2 + 1];
    if (radius_x <= 0 || radius_y <= 0) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    vec2 mean2d = glm::make_vec2(means2d + 2 * idx);

    float tile_radius_x = radius_x / static_cast<float>(tile_size);
    float tile_radius_y = radius_y / static_cast<float>(tile_size);
    float tile_x = mean2d.x / static_cast<float>(tile_size);
    float tile_y = mean2d.y / static_cast<float>(tile_size);

    // tile_min is inclusive, tile_max is exclusive
    uint2 tile_min, tile_max;
    tile_min.x = min(max(0, (uint32_t)floor(tile_x - tile_radius_x)), tile_width);
    tile_min.y =
        min(max(0, (uint32_t)floor(tile_y - tile_radius_y)), tile_height);
    tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_radius_x)), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_radius_y)), tile_height);

    int64_t iid; // image id
    if (!first_pass) {
        if (packed) {
            // parallelize over nnz
            iid = image_ids[idx];
        } else {
            // parallelize over I * N
            iid = idx / N;
        }
    }
    const int64_t iid_enc = first_pass ? 0 : (iid << (32 + tile_n_bits));

    // tolerance for negative depth
    int32_t depth_i32 = first_pass ? 0 : *(int32_t *)&(depths[idx]);  // Bit-level reinterpret
    int64_t depth_id_enc = first_pass ? 0 : static_cast<uint32_t>(depth_i32);  // Zero-extend to 64-bit
    
    int64_t cur_idx = first_pass ? 0 : ((idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1]);
    int32_t tile_count = 0;
    
    // 2DGS ray transforms
    const bool use_2dgs = false; //ray_transforms_2dgs != nullptr;
    vec3 u_M, v_M, w_M;

    if (use_2dgs) {
        u_M = {
            ray_transforms_2dgs[idx * 9],
            ray_transforms_2dgs[idx * 9 + 1],
            ray_transforms_2dgs[idx * 9 + 2]
        };
        v_M = {
            ray_transforms_2dgs[idx * 9 + 3],
            ray_transforms_2dgs[idx * 9 + 4],
            ray_transforms_2dgs[idx * 9 + 5]
        };
        w_M = {
            ray_transforms_2dgs[idx * 9 + 6],
            ray_transforms_2dgs[idx * 9 + 7],
            ray_transforms_2dgs[idx * 9 + 8]
        };
    }

    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            

            bool tile_intersects = !use_2dgs || 
                check_2dgs_tile_intersection(
                    u_M, v_M, w_M, mean2d, 
                    tile_radius_x, tile_radius_y,
                    j, i, tile_size
                );
            
            if (tile_intersects) {
                if (first_pass) {
                    tile_count++;
                } else {
                    int64_t tile_id = i * tile_width + j;
                    // e.g. tile_n_bits = 22:
                    // image id (10 bits) | tile id (22 bits) | depth (32 bits)
                    isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;
                    // the flatten index in [I * N] or [nnz]
                    flatten_ids[cur_idx] = static_cast<int32_t>(idx);
                    ++cur_idx;
                }
            }
        }
    }
    
    if (first_pass) {
        tiles_per_gauss[idx] = tile_count;
    }
}

void launch_intersect_tile_kernel(
    // inputs
    const at::Tensor means2d,                    // [..., N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> ray_transforms_2dgs, // [..., N, 9] or [nnz, 9]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const at::optional<at::Tensor> cum_tiles_per_gauss, // [..., N] or [nnz]
    // outputs
    at::optional<at::Tensor> tiles_per_gauss, // [..., N] or [nnz]
    at::optional<at::Tensor> isect_ids,       // [n_isects]
    at::optional<at::Tensor> flatten_ids      // [n_isects]
) {
    bool packed = means2d.dim() == 2;

    uint32_t N, nnz;
    int64_t n_elements;
    if (packed) {
        nnz = means2d.size(0); // total number of gaussians
        n_elements = nnz;
    } else {
        N = means2d.size(-2); // number of gaussians per image
        n_elements = I * N;
    }

    uint32_t n_tiles = tile_width * tile_height;
    // the number of bits needed to encode the image id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t image_n_bits = std::bit_width(I);
    uint32_t image_n_bits = (uint32_t)floor(log2(I)) + 1;
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    // the first 32 bits are used for the image id and tile id altogether, so
    // check if we have enough bits for them.
    assert(image_n_bits + tile_n_bits <= 32);

    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        means2d.scalar_type(),
        "intersect_tile_kernel",
        [&]() {
            intersect_tile_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    packed,
                    I,
                    N,
                    nnz,
                    image_ids.has_value()
                        ? image_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    gaussian_ids.has_value()
                        ? gaussian_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    means2d.data_ptr<scalar_t>(),
                    radii.data_ptr<int32_t>(),
                    depths.data_ptr<scalar_t>(),
                    ray_transforms_2dgs.has_value()
                        ? ray_transforms_2dgs.value().data_ptr<scalar_t>()
                        : nullptr,
                    cum_tiles_per_gauss.has_value()
                        ? cum_tiles_per_gauss.value().data_ptr<int64_t>()
                        : nullptr,
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_n_bits,
                    image_n_bits,
                    tiles_per_gauss.has_value()
                        ? tiles_per_gauss.value().data_ptr<int32_t>()
                        : nullptr,
                    isect_ids.has_value()
                        ? isect_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    flatten_ids.has_value()
                        ? flatten_ids.value().data_ptr<int32_t>()
                        : nullptr
                );
        }
    );
}

__global__ void intersect_offset_kernel(
    const uint32_t n_isects,
    const int64_t *__restrict__ isect_ids,
    const uint32_t I,
    const uint32_t n_tiles,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ offsets // [I, n_tiles]
) {
    // e.g., ids: [1, 1, 1, 3, 3], n_tiles = 6
    // counts: [0, 3, 0, 2, 0, 0]
    // cumsum: [0, 3, 3, 5, 5, 5]
    // offsets: [0, 0, 3, 3, 5, 5]
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_isects)
        return;

    uint32_t image_n_bits = (uint32_t)floor(log2f(float(I))) + 1;

    int64_t isect_id_curr = isect_ids[idx] >> 32;
    int64_t iid_curr = isect_id_curr >> (tile_n_bits);
    int64_t tid_curr = isect_id_curr & ((1 << tile_n_bits) - 1);
    int64_t id_curr = iid_curr * n_tiles + tid_curr;

    if (idx == 0) {
        // write out the offsets until the first valid tile (inclusive)
        for (uint32_t i = 0; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
    if (idx == n_isects - 1) {
        // write out the rest of the offsets
        for (uint32_t i = id_curr + 1; i < I * n_tiles; ++i)
            offsets[i] = static_cast<int32_t>(n_isects);
    }

    if (idx > 0) {
        // visit the current and previous isect_id and check if the (bid, cid,
        // tile_id) tuple changes.
        int64_t isect_id_prev = isect_ids[idx - 1] >> 32; // shift out the depth
        if (isect_id_prev == isect_id_curr)
            return;

        // write out the offsets between the previous and current tiles
        int64_t iid_prev = isect_id_prev >> (tile_n_bits);
        int64_t tid_prev = isect_id_prev & ((1 << tile_n_bits) - 1);
        int64_t id_prev = iid_prev * n_tiles + tid_prev;
        for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
}

void launch_intersect_offset_kernel(
    // inputs
    const at::Tensor isect_ids, // [n_isects]
    const uint32_t I,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // outputs
    at::Tensor offsets // [I, tile_height, tile_width]
) {
    int64_t n_elements = isect_ids.size(0); // total number of intersections
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        offsets.fill_(0);
        return;
    }

    uint32_t n_tiles = tile_width * tile_height;
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    intersect_offset_kernel<<<
        grid,
        threads,
        shmem_size,
        at::cuda::getCurrentCUDAStream()>>>(
        n_elements,
        isect_ids.data_ptr<int64_t>(),
        I,
        n_tiles,
        tile_n_bits,
        offsets.data_ptr<int32_t>()
    );
}

// https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
// DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
void radix_sort_double_buffer(
    const int64_t n_isects,
    const uint32_t image_n_bits,
    const uint32_t tile_n_bits,
    at::Tensor isect_ids,
    at::Tensor flatten_ids,
    at::Tensor isect_ids_sorted,
    at::Tensor flatten_ids_sorted
) {
    if (n_isects <= 0) {
        return;
    }

    // Create a set of DoubleBuffers to wrap pairs of device pointers
    cub::DoubleBuffer<int64_t> d_keys(
        isect_ids.data_ptr<int64_t>(), isect_ids_sorted.data_ptr<int64_t>()
    );
    cub::DoubleBuffer<int32_t> d_values(
        flatten_ids.data_ptr<int32_t>(), flatten_ids_sorted.data_ptr<int32_t>()
    );
    CUB_WRAPPER(
        cub::DeviceRadixSort::SortPairs,
        d_keys,
        d_values,
        n_isects,
        0,
        32 + tile_n_bits + image_n_bits,
        at::cuda::getCurrentCUDAStream()
    );
    switch (d_keys.selector) {
    case 0: // sorted items are stored in isect_ids
        isect_ids_sorted.set_(isect_ids);
        break;
    case 1: // sorted items are stored in isect_ids_sorted
        break;
    }
    switch (d_values.selector) {
    case 0: // sorted items are stored in flatten_ids
        flatten_ids_sorted.set_(flatten_ids);
        break;
    case 1: // sorted items are stored in flatten_ids_sorted
        break;
    }
}

// https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceSegmentedRadixSort.html
// DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
void segmented_radix_sort_double_buffer(
    const int64_t n_isects,
    const uint32_t n_segments,
    const uint32_t image_n_bits,
    const uint32_t tile_n_bits,
    const at::Tensor offsets,
    at::Tensor isect_ids,
    at::Tensor flatten_ids,
    at::Tensor isect_ids_sorted,
    at::Tensor flatten_ids_sorted
) {
    if (n_isects <= 0) {
        return;
    }

    // Create a set of DoubleBuffers to wrap pairs of device pointers
    cub::DoubleBuffer<int64_t> d_keys(
        isect_ids.data_ptr<int64_t>(), isect_ids_sorted.data_ptr<int64_t>()
    );
    cub::DoubleBuffer<int32_t> d_values(
        flatten_ids.data_ptr<int32_t>(), flatten_ids_sorted.data_ptr<int32_t>()
    );
    // image dimensions are contiguous in the isect_ids, 
    // so we can use DeviceSegmentedRadixSort to only sort the lower 
    // (tile_n_bits + 32) bits
    CUB_WRAPPER(
        cub::DeviceSegmentedRadixSort::SortPairs,
        d_keys,
        d_values,
        n_isects,
        n_segments, // number of segments
        offsets.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>() + 1,
        0,
        32 + tile_n_bits,
        at::cuda::getCurrentCUDAStream()
    );
    switch (d_keys.selector) {
    case 0: // sorted items are stored in isect_ids
        isect_ids_sorted.set_(isect_ids);
        break;
    case 1: // sorted items are stored in isect_ids_sorted
        break;
    }
    switch (d_values.selector) {
    case 0: // sorted items are stored in flatten_ids
        flatten_ids_sorted.set_(flatten_ids);
        break;
    case 1: // sorted items are stored in flatten_ids_sorted
        break;
    }
}

} // namespace gsplat
