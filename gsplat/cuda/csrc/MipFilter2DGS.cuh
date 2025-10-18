#ifndef GSPLAT_MIP_FILTER_CUH
#define GSPLAT_MIP_FILTER_CUH

#include "Common.h"

namespace gsplat {

// Antialiasing method selection
enum AAMethod { 
    DEFAULT = 0,  // Original 2DGS method - minimum of ray-intersection and 2D gaussian
    AA_2DGS = 1,  // Mip-NeRF 360 style filter with Jacobian-based covariance (arXiv:2506.11252)
    HDGS = 2      // Frustum-based supersampling (arXiv:2412.01823)
};

// Change this to select antialiasing method globally
constexpr AAMethod AA_METHOD = HDGS;

// Unified version with optional inverse matrix elements for backward pass
__device__ __forceinline__ void compute_mip_filter_weight(
    const vec3& h_u, const vec3& h_v, const vec3& w_M,
    const vec2& s, const float ray_cross_z,
    float& gauss_weight_out, float& norm_factor_out,
    float* inv_a11 = nullptr, float* inv_a12 = nullptr, 
    float* inv_a22 = nullptr, float* inv_det_out = nullptr)
{
    const float inv_z = 1.0f / ray_cross_z;
    const float inv_z_sq = inv_z * inv_z;

    // Partial derivatives of cross product components
    // ∂(h_u × h_v)/∂px = w_M × h_v
    const float dcross_x_dx = w_M.y * h_v.z - w_M.z * h_v.y;
    const float dcross_y_dx = w_M.z * h_v.x - w_M.x * h_v.z;
    const float dcross_z_dx = w_M.x * h_v.y - w_M.y * h_v.x;

    // ∂(h_u × h_v)/∂py = h_u × w_M
    const float dcross_x_dy = h_u.y * w_M.z - h_u.z * w_M.y;
    const float dcross_y_dy = h_u.z * w_M.x - h_u.x * w_M.z;
    const float dcross_z_dy = h_u.x * w_M.y - h_u.y * w_M.x;

    // Jacobian matrix elements using quotient rule
    const float J_11 = dcross_x_dx * inv_z - s.x * dcross_z_dx * inv_z_sq;
    const float J_12 = dcross_x_dy * inv_z - s.x * dcross_z_dy * inv_z_sq;
    const float J_21 = dcross_y_dx * inv_z - s.y * dcross_z_dx * inv_z_sq;
    const float J_22 = dcross_y_dy * inv_z - s.y * dcross_z_dy * inv_z_sq;

    // Mip filter covariance: Σ' = I + σJJ^T (Eq. 16 from paper)
    const float mip_sigma = 0.1f;
    float a11 = 1.0f + mip_sigma * (J_11 * J_11 + J_12 * J_12);
    float a12 = mip_sigma * (J_11 * J_21 + J_12 * J_22);
    float a22 = 1.0f + mip_sigma * (J_21 * J_21 + J_22 * J_22);

    // Determinant and inverse
    float det = a11 * a22 - a12 * a12;

    // Обеспечиваем минимальный детерминант для численной стабильности
    const float min_det = 0.01f;
    if (det < min_det) {
        // Добавляем регуляризацию к диагональным элементам
        const float regularization = min_det - det + 0.01f;
        a11 += regularization;
        a22 += regularization;
        det = a11 * a22 - a12 * a12;
    }

    // Защита от численной нестабильности
    if (det < 1e-6f) {
        gauss_weight_out = 0.0f; // значение приведёт к отбрасыванию гауссиана
        norm_factor_out = 0.0f;
        if (inv_a11 != nullptr) {
            *inv_a11 = 0.0f;
            *inv_a12 = 0.0f;
            *inv_a22 = 0.0f;
            if (inv_det_out != nullptr) *inv_det_out = 0.0f;
        }
        return;
    }

    // Quadratic form: s^T * Σ'^-1 * s
    float inv_det = 1.0f / det;
    
    // Квадратичная форма
    gauss_weight_out = inv_det * (a22 * s.x * s.x - 2.0f * a12 * s.x * s.y + a11 * s.y * s.y);

    // Элементы обратной матрицы Σ'^{-1} (только если нужны для backward pass)
    if (inv_a11 != nullptr) {
        *inv_a11 = a22 * inv_det;
        *inv_a12 = -a12 * inv_det;
        *inv_a22 = a11 * inv_det;
        if (inv_det_out != nullptr) *inv_det_out = inv_det;
    }

    // Normalization factor: sqrt(|I|/|Σ'|) = 1/sqrt(det(Σ'))
    norm_factor_out = rsqrtf(det);

    gauss_weight_out = fminf(gauss_weight_out, 100.0f);
    norm_factor_out = fminf(norm_factor_out, 10.0f);
}

// Frustum-based supersampling for antialiasing (HDGS, arXiv:2412.01823)
__device__ __forceinline__ void compute_supersample_filter_weight(
    const float px, const float py,           // pixel center coordinates
    const vec3& u_M, const vec3& v_M, const vec3& w_M,  // ray transform rows
    float& gauss_weight_out,                  // output: averaged gaussian weight
    vec2& s_center_out,                       // output: center intersection point
    // Optional outputs for backward pass
    vec2* s_samples = nullptr,                // [5] sample intersection points
    float* sample_weights = nullptr           // [5] individual gaussian weights
)
{
    // Frustum sampling configuration with Eq. 22 weights for improved accuracy
    // Center weight: 2/3, Corner weights: 1/12 each
    const float weights[5] = {2.0f/3.0f, 1.0f/12.0f, 1.0f/12.0f, 1.0f/12.0f, 1.0f/12.0f};
    const vec2 offsets[5] = {
        {0.0f, 0.0f},      // center
        {-0.5f, -0.5f},    // top-left corner
        {0.5f, -0.5f},     // top-right corner  
        {-0.5f, 0.5f},     // bottom-left corner
        {0.5f, 0.5f}       // bottom-right corner
    };
    
    float weighted_vis_sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Process all 5 sample points
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        float sample_px = px + offsets[i].x;
        float sample_py = py + offsets[i].y;
        
        // Compute homogeneous plane parameters for this sample
        vec3 h_u = sample_px * w_M - u_M;
        vec3 h_v = sample_py * w_M - v_M;
        vec3 ray_cross = glm::cross(h_u, h_v);
        
        // Check for valid intersection
        const float RAY_CROSS_EPSILON = 1e-8f;
        if (fabsf(ray_cross.z) < RAY_CROSS_EPSILON) {
            // Invalid intersection - mark with large weight to exclude
            if (sample_weights != nullptr) {
                sample_weights[i] = 1e10f;
            }
            if (s_samples != nullptr) {
                s_samples[i] = {0.0f, 0.0f};
            }
            continue;
        }
        
        // Compute intersection point in UV space
        vec2 s = {ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z};
        float gauss_weight_3d = s.x * s.x + s.y * s.y;
        
        // Store sample data if requested
        if (s_samples != nullptr) {
            s_samples[i] = s;
        }
        if (sample_weights != nullptr) {
            sample_weights[i] = gauss_weight_3d;
        }
        
        // Store center point for output
        if (i == 0) {
            s_center_out = s;
        }
        
        // Accumulate weighted visibility (exp(-0.5 * weight))
        weighted_vis_sum += weights[i] * expf(-0.5f * gauss_weight_3d);
        weight_sum += weights[i];
    }
    
    // Handle case where no valid samples exist
    if (weight_sum < 1e-6f) {
        gauss_weight_out = 1e10f; // Will be filtered out
        s_center_out = {0.0f, 0.0f};
        return;
    }
    
    // Weighted average visibility
    float vis_avg = weighted_vis_sum / weight_sum;
    
    // Convert back to effective gaussian weight
    // vis_avg = exp(-0.5 * gauss_weight_effective)
    // gauss_weight_effective = -2.0 * log(vis_avg)
    gauss_weight_out = -2.0f * logf(fmaxf(vis_avg, 1e-10f));
    
    // Clamp to reasonable range for numerical stability
    gauss_weight_out = fminf(gauss_weight_out, 100.0f);
}

// Compute gradients for frustum-based supersampling (backward pass)
__device__ __forceinline__ void compute_supersample_filter_gradients(
    const float v_G,                          // gradient w.r.t. gaussian weight
    const float vis,                          // visibility (exp(-0.5 * gauss_weight))
    const vec2* s_samples,                    // [5] sample intersection points
    const float* sample_weights,              // [5] individual gaussian weights
    const float px, const float py,           // pixel center
    const vec3& u_M, const vec3& v_M, const vec3& w_M,  // transform rows
    // Outputs
    vec3& v_u_M_out,                          // gradient w.r.t. u_M
    vec3& v_v_M_out,                          // gradient w.r.t. v_M
    vec3& v_w_M_out                           // gradient w.r.t. w_M
)
{
    // Eq. 22 weights: center 2/3, corners 1/12 each
    const float weights[5] = {2.0f/3.0f, 1.0f/12.0f, 1.0f/12.0f, 1.0f/12.0f, 1.0f/12.0f};
    
    // Initialize outputs
    v_u_M_out = {0.0f, 0.0f, 0.0f};
    v_v_M_out = {0.0f, 0.0f, 0.0f};
    v_w_M_out = {0.0f, 0.0f, 0.0f};
    
    // Compute visibilities for each sample
    float vis_samples[5];
    float weighted_vis_sum = 0.0f;
    float weight_sum = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        if (sample_weights[i] < 1e9f) {
            vis_samples[i] = expf(-0.5f * sample_weights[i]);
            weighted_vis_sum += weights[i] * vis_samples[i];
            weight_sum += weights[i];
        } else {
            vis_samples[i] = 0.0f;
        }
    }
    
    // Gradient through logarithm and weighted averaging:
    // vis_avg = weighted_vis_sum / weight_sum
    // gauss_weight_effective = -2.0 * log(vis_avg)
    // vis = exp(-0.5 * gauss_weight_effective) = vis_avg
    const float v_vis_avg = v_G * vis / fmaxf(weighted_vis_sum / weight_sum, 1e-10f);
    
    // Gradient for each weighted sample:
    // ∂L/∂vis_i = v_vis_avg * weight_i / weight_sum
    // ∂L/∂weight_i = ∂L/∂vis_i * ∂vis_i/∂weight_i
    //              = (v_vis_avg * weight_i / weight_sum) * (-0.5 * vis_i)
    const float grad_factor = -0.5f * v_vis_avg / weight_sum;
    
    // Gradient through frustum samples
    const vec2 offsets[5] = {
        {0.0f, 0.0f}, {-0.5f, -0.5f}, {0.5f, -0.5f},
        {-0.5f, 0.5f}, {0.5f, 0.5f}
    };
    
    // Accumulate gradients from all samples
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        if (sample_weights[i] >= 1e9f) continue; // skip invalid samples
        
        const vec2& s = s_samples[i];
        const float v_weight_i = grad_factor * weights[i] * vis_samples[i];
        
        // Gradient of gaussian weight w.r.t. s
        const vec2 v_s = {
            v_weight_i * s.x,
            v_weight_i * s.y
        };
        
        // Recompute h_u, h_v for this sample point
        const float sample_px = px + offsets[i].x;
        const float sample_py = py + offsets[i].y;
        const vec3 h_u_i = sample_px * w_M - u_M;
        const vec3 h_v_i = sample_py * w_M - v_M;
        const vec3 ray_cross = glm::cross(h_u_i, h_v_i);
        
        // Gradient through projective transform
        const float inv_z = 1.0f / ray_cross.z;
        const float v_sx_pz = v_s.x * inv_z;
        const float v_sy_pz = v_s.y * inv_z;
        const vec3 v_ray_cross = {
            v_sx_pz, v_sy_pz, -(v_sx_pz * s.x + v_sy_pz * s.y)
        };
        
        const vec3 v_h_u = glm::cross(h_v_i, v_ray_cross);
        const vec3 v_h_v = glm::cross(v_ray_cross, h_u_i);
        
        // Accumulate gradients
        v_u_M_out.x -= v_h_u.x;
        v_u_M_out.y -= v_h_u.y;
        v_u_M_out.z -= v_h_u.z;
        
        v_v_M_out.x -= v_h_v.x;
        v_v_M_out.y -= v_h_v.y;
        v_v_M_out.z -= v_h_v.z;
        
        v_w_M_out.x += sample_px * v_h_u.x + sample_py * v_h_v.x;
        v_w_M_out.y += sample_px * v_h_u.y + sample_py * v_h_v.y;
        v_w_M_out.z += sample_px * v_h_u.z + sample_py * v_h_v.z;
    }
}

} // namespace gsplat


#endif // GSPLAT_MIP_FILTER_CUH
 