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


// Frustum-based supersampling for antialiasing (HDGS, arXiv:2412.01823)
__device__ __forceinline__ void compute_supersample_filter_weight_optimized(
    const float px, const float py,           // pixel center coordinates
    const vec3& u_M, const vec3& v_M, const vec3& w_M,  // ray transform rows
    float& gauss_weight_out,                  // output: averaged gaussian weight
    vec2& s_center_out,                       // output: center intersection point
    // Optional outputs for backward pass
    vec2* s_samples = nullptr,                // [5] sample intersection points
    float* sample_weights = nullptr           // [5] individual gaussian weights
)
{
    // Eq. 22 weights: center 2/3, corners 1/12 each
    const float weights[5] = {2.0f/3.0f, 1.0f/12.0f, 1.0f/12.0f, 1.0f/12.0f, 1.0f/12.0f};
    
    // Предвычисляем общие компоненты для всех сэмплов
    const vec3 base_h_u = px * w_M - u_M;
    const vec3 base_h_v = py * w_M - v_M;
    const vec3 half_w_M = 0.5f * w_M;
    
    // Константы для проверки валидности
    const float RAY_CROSS_EPSILON = 1e-8f;
    const float INVALID_WEIGHT = 1e10f;
    
    // Развёрнутое вычисление для 5 точек с FMA оптимизацией
    // Центр (0, 0)
    float cross_x_0 = __fmaf_rn(base_h_u.y, base_h_v.z, -base_h_u.z * base_h_v.y);
    float cross_y_0 = __fmaf_rn(base_h_u.z, base_h_v.x, -base_h_u.x * base_h_v.z);
    float cross_z_0 = __fmaf_rn(base_h_u.x, base_h_v.y, -base_h_u.y * base_h_v.x);
    
    float inv_z_0 = __frcp_rn(cross_z_0);
    float s0_x = cross_x_0 * inv_z_0;
    float s0_y = cross_y_0 * inv_z_0;
    float weight_0 = __fmaf_rn(s0_x, s0_x, s0_y * s0_y);
    
    // Маскирование невалидных значений без ветвления
    float mask_0 = (fabsf(cross_z_0) >= RAY_CROSS_EPSILON) ? 1.0f : 0.0f;
    weight_0 = mask_0 * weight_0 + (1.0f - mask_0) * INVALID_WEIGHT;
    
    // Сохраняем центральную точку
    s_center_out = {s0_x, s0_y};
    
    // Верхний левый (-0.5, -0.5)
    vec3 h_u_1 = base_h_u - half_w_M;
    vec3 h_v_1 = base_h_v - half_w_M;
    float cross_x_1 = __fmaf_rn(h_u_1.y, h_v_1.z, -h_u_1.z * h_v_1.y);
    float cross_y_1 = __fmaf_rn(h_u_1.z, h_v_1.x, -h_u_1.x * h_v_1.z);
    float cross_z_1 = __fmaf_rn(h_u_1.x, h_v_1.y, -h_u_1.y * h_v_1.x);
    
    float inv_z_1 = __frcp_rn(cross_z_1);
    float s1_x = cross_x_1 * inv_z_1;
    float s1_y = cross_y_1 * inv_z_1;
    float weight_1 = __fmaf_rn(s1_x, s1_x, s1_y * s1_y);
    
    float mask_1 = (fabsf(cross_z_1) >= RAY_CROSS_EPSILON) ? 1.0f : 0.0f;
    weight_1 = mask_1 * weight_1 + (1.0f - mask_1) * INVALID_WEIGHT;
    
    // Верхний правый (0.5, -0.5)
    vec3 h_u_2 = base_h_u + half_w_M;
    vec3 h_v_2 = base_h_v - half_w_M;
    float cross_x_2 = __fmaf_rn(h_u_2.y, h_v_2.z, -h_u_2.z * h_v_2.y);
    float cross_y_2 = __fmaf_rn(h_u_2.z, h_v_2.x, -h_u_2.x * h_v_2.z);
    float cross_z_2 = __fmaf_rn(h_u_2.x, h_v_2.y, -h_u_2.y * h_v_2.x);
    
    float inv_z_2 = __frcp_rn(cross_z_2);
    float s2_x = cross_x_2 * inv_z_2;
    float s2_y = cross_y_2 * inv_z_2;
    float weight_2 = __fmaf_rn(s2_x, s2_x, s2_y * s2_y);
    
    float mask_2 = (fabsf(cross_z_2) >= RAY_CROSS_EPSILON) ? 1.0f : 0.0f;
    weight_2 = mask_2 * weight_2 + (1.0f - mask_2) * INVALID_WEIGHT;
    
    // Нижний левый (-0.5, 0.5)
    vec3 h_u_3 = base_h_u - half_w_M;
    vec3 h_v_3 = base_h_v + half_w_M;
    float cross_x_3 = __fmaf_rn(h_u_3.y, h_v_3.z, -h_u_3.z * h_v_3.y);
    float cross_y_3 = __fmaf_rn(h_u_3.z, h_v_3.x, -h_u_3.x * h_v_3.z);
    float cross_z_3 = __fmaf_rn(h_u_3.x, h_v_3.y, -h_u_3.y * h_v_3.x);
    
    float inv_z_3 = __frcp_rn(cross_z_3);
    float s3_x = cross_x_3 * inv_z_3;
    float s3_y = cross_y_3 * inv_z_3;
    float weight_3 = __fmaf_rn(s3_x, s3_x, s3_y * s3_y);
    
    float mask_3 = (fabsf(cross_z_3) >= RAY_CROSS_EPSILON) ? 1.0f : 0.0f;
    weight_3 = mask_3 * weight_3 + (1.0f - mask_3) * INVALID_WEIGHT;
    
    // Нижний правый (0.5, 0.5)
    vec3 h_u_4 = base_h_u + half_w_M;
    vec3 h_v_4 = base_h_v + half_w_M;
    float cross_x_4 = __fmaf_rn(h_u_4.y, h_v_4.z, -h_u_4.z * h_v_4.y);
    float cross_y_4 = __fmaf_rn(h_u_4.z, h_v_4.x, -h_u_4.x * h_v_4.z);
    float cross_z_4 = __fmaf_rn(h_u_4.x, h_v_4.y, -h_u_4.y * h_v_4.x);
    
    float inv_z_4 = __frcp_rn(cross_z_4);
    float s4_x = cross_x_4 * inv_z_4;
    float s4_y = cross_y_4 * inv_z_4;
    float weight_4 = __fmaf_rn(s4_x, s4_x, s4_y * s4_y);
    
    float mask_4 = (fabsf(cross_z_4) >= RAY_CROSS_EPSILON) ? 1.0f : 0.0f;
    weight_4 = mask_4 * weight_4 + (1.0f - mask_4) * INVALID_WEIGHT;
    
    // Взвешенное усреднение по Eq. 22: среднее от экспонент (видимостей)
    // vis_i = exp(-0.5 * weight_i)
    // vis_avg = Σ(weight_i * vis_i) / Σweight_i
    float weighted_vis_sum = 0.0f;
    float weight_sum = 0.0f;
    
    if (weight_0 < INVALID_WEIGHT) {
        weighted_vis_sum += weights[0] * __expf(-0.5f * weight_0);
        weight_sum += weights[0];
    }
    if (weight_1 < INVALID_WEIGHT) {
        weighted_vis_sum += weights[1] * __expf(-0.5f * weight_1);
        weight_sum += weights[1];
    }
    if (weight_2 < INVALID_WEIGHT) {
        weighted_vis_sum += weights[2] * __expf(-0.5f * weight_2);
        weight_sum += weights[2];
    }
    if (weight_3 < INVALID_WEIGHT) {
        weighted_vis_sum += weights[3] * __expf(-0.5f * weight_3);
        weight_sum += weights[3];
    }
    if (weight_4 < INVALID_WEIGHT) {
        weighted_vis_sum += weights[4] * __expf(-0.5f * weight_4);
        weight_sum += weights[4];
    }
    
    // Проверка на все невалидные сэмплы
    if (weight_sum < 1e-6f) {
        gauss_weight_out = INVALID_WEIGHT;
        s_center_out = {0.0f, 0.0f};
        return;
    }
    
    // Усреднённая видимость
    float vis_avg = weighted_vis_sum / weight_sum;
    
    // Обратное преобразование для получения эффективного веса
    // vis_avg = exp(-0.5 * gauss_weight_effective)
    // gauss_weight_effective = -2.0 * log(vis_avg)
    gauss_weight_out = -2.0f * __logf(fmaxf(vis_avg, 1e-10f));
    
    // Сохранение данных для backward pass (если нужно)
    if (s_samples != nullptr) {
        s_samples[0] = {s0_x, s0_y};
        s_samples[1] = {s1_x, s1_y};
        s_samples[2] = {s2_x, s2_y};
        s_samples[3] = {s3_x, s3_y};
        s_samples[4] = {s4_x, s4_y};
    }
    
    if (sample_weights != nullptr) {
        sample_weights[0] = weight_0;
        sample_weights[1] = weight_1;
        sample_weights[2] = weight_2;
        sample_weights[3] = weight_3;
        sample_weights[4] = weight_4;
    }
    
    // Ограничение для численной стабильности
    gauss_weight_out = fminf(gauss_weight_out, 100.0f);
}

// Compute gradients for frustum-based supersampling (backward pass)
__device__ __forceinline__ void compute_supersample_filter_gradients_optimized(
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
    
    // Вычисляем видимости для каждого сэмпла
    float vis_samples[5];
    float weighted_vis_sum = 0.0f;
    float weight_sum = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        if (sample_weights[i] < 1e9f) {
            vis_samples[i] = __expf(-0.5f * sample_weights[i]);
            weighted_vis_sum += weights[i] * vis_samples[i];
            weight_sum += weights[i];
        } else {
            vis_samples[i] = 0.0f;
        }
    }
    
    // vis_avg = weighted_vis_sum / weight_sum
    // gauss_weight_effective = -2.0 * log(vis_avg)
    // vis = exp(-0.5 * gauss_weight_effective) = vis_avg
    
    // Градиент через логарифм и взвешенное усреднение:
    // ∂L/∂vis_avg = v_G * ∂gauss_weight/∂vis_avg * ∂vis/∂gauss_weight
    //             = v_G * (-2.0 / vis_avg) * (-0.5 * vis)
    //             = v_G * vis / vis_avg
    const float v_vis_avg = v_G * vis / fmaxf(weighted_vis_sum / weight_sum, 1e-10f);
    
    // Градиент для каждого взвешенного сэмпла:
    // ∂L/∂vis_i = v_vis_avg * weight_i / weight_sum
    // ∂L/∂weight_i = ∂L/∂vis_i * ∂vis_i/∂weight_i
    //              = (v_vis_avg * weight_i / weight_sum) * (-0.5 * vis_i)
    const float grad_factor = -0.5f * v_vis_avg / weight_sum;
    
    // Предвычисляем базовые компоненты
    const vec3 base_h_u = px * w_M - u_M;
    const vec3 base_h_v = py * w_M - v_M;
    const vec3 half_w_M = 0.5f * w_M;
    
    // Инициализация аккумуляторов
    float acc_u_x = 0.0f, acc_u_y = 0.0f, acc_u_z = 0.0f;
    float acc_v_x = 0.0f, acc_v_y = 0.0f, acc_v_z = 0.0f;
    float acc_w_x = 0.0f, acc_w_y = 0.0f, acc_w_z = 0.0f;
    
    // Развёрнутая обработка 5 сэмплов
    // Сэмпл 0: центр (0, 0)
    if (sample_weights[0] < 1e9f) {
        const vec2& s0 = s_samples[0];
        const float v_weight_0 = grad_factor * weights[0] * vis_samples[0];
        const vec2 v_s0 = {v_weight_0 * s0.x, v_weight_0 * s0.y};
        
        vec3 ray_cross_0 = glm::cross(base_h_u, base_h_v);
        float inv_z_0 = __frcp_rn(ray_cross_0.z);
        
        float v_sx_pz_0 = v_s0.x * inv_z_0;
        float v_sy_pz_0 = v_s0.y * inv_z_0;
        vec3 v_ray_cross_0 = {
            v_sx_pz_0, v_sy_pz_0,
            -__fmaf_rn(v_sx_pz_0, s0.x, v_sy_pz_0 * s0.y)
        };
        
        // v_h_u = h_v × v_ray_cross
        float v_h_u_x_0 = __fmaf_rn(base_h_v.y, v_ray_cross_0.z, -base_h_v.z * v_ray_cross_0.y);
        float v_h_u_y_0 = __fmaf_rn(base_h_v.z, v_ray_cross_0.x, -base_h_v.x * v_ray_cross_0.z);
        float v_h_u_z_0 = __fmaf_rn(base_h_v.x, v_ray_cross_0.y, -base_h_v.y * v_ray_cross_0.x);
        
        // v_h_v = v_ray_cross × h_u
        float v_h_v_x_0 = __fmaf_rn(v_ray_cross_0.y, base_h_u.z, -v_ray_cross_0.z * base_h_u.y);
        float v_h_v_y_0 = __fmaf_rn(v_ray_cross_0.z, base_h_u.x, -v_ray_cross_0.x * base_h_u.z);
        float v_h_v_z_0 = __fmaf_rn(v_ray_cross_0.x, base_h_u.y, -v_ray_cross_0.y * base_h_u.x);
        
        acc_u_x -= v_h_u_x_0;
        acc_u_y -= v_h_u_y_0;
        acc_u_z -= v_h_u_z_0;
        
        acc_v_x -= v_h_v_x_0;
        acc_v_y -= v_h_v_y_0;
        acc_v_z -= v_h_v_z_0;
        
        acc_w_x = __fmaf_rn(px, v_h_u_x_0, __fmaf_rn(py, v_h_v_x_0, acc_w_x));
        acc_w_y = __fmaf_rn(px, v_h_u_y_0, __fmaf_rn(py, v_h_v_y_0, acc_w_y));
        acc_w_z = __fmaf_rn(px, v_h_u_z_0, __fmaf_rn(py, v_h_v_z_0, acc_w_z));
    }
    
    // Сэмпл 1: (-0.5, -0.5)
    if (sample_weights[1] < 1e9f) {
        const vec2& s1 = s_samples[1];
        const float v_weight_1 = grad_factor * weights[1] * vis_samples[1];
        const vec2 v_s1 = {v_weight_1 * s1.x, v_weight_1 * s1.y};
        
        vec3 h_u_1 = base_h_u - half_w_M;
        vec3 h_v_1 = base_h_v - half_w_M;
        vec3 ray_cross_1 = glm::cross(h_u_1, h_v_1);
        float inv_z_1 = __frcp_rn(ray_cross_1.z);
        
        float v_sx_pz_1 = v_s1.x * inv_z_1;
        float v_sy_pz_1 = v_s1.y * inv_z_1;
        vec3 v_ray_cross_1 = {
            v_sx_pz_1, v_sy_pz_1,
            -__fmaf_rn(v_sx_pz_1, s1.x, v_sy_pz_1 * s1.y)
        };
        
        float v_h_u_x_1 = __fmaf_rn(h_v_1.y, v_ray_cross_1.z, -h_v_1.z * v_ray_cross_1.y);
        float v_h_u_y_1 = __fmaf_rn(h_v_1.z, v_ray_cross_1.x, -h_v_1.x * v_ray_cross_1.z);
        float v_h_u_z_1 = __fmaf_rn(h_v_1.x, v_ray_cross_1.y, -h_v_1.y * v_ray_cross_1.x);
        
        float v_h_v_x_1 = __fmaf_rn(v_ray_cross_1.y, h_u_1.z, -v_ray_cross_1.z * h_u_1.y);
        float v_h_v_y_1 = __fmaf_rn(v_ray_cross_1.z, h_u_1.x, -v_ray_cross_1.x * h_u_1.z);
        float v_h_v_z_1 = __fmaf_rn(v_ray_cross_1.x, h_u_1.y, -v_ray_cross_1.y * h_u_1.x);
        
        const float px_1 = px - 0.5f;
        const float py_1 = py - 0.5f;
        
        acc_u_x -= v_h_u_x_1;
        acc_u_y -= v_h_u_y_1;
        acc_u_z -= v_h_u_z_1;
        
        acc_v_x -= v_h_v_x_1;
        acc_v_y -= v_h_v_y_1;
        acc_v_z -= v_h_v_z_1;
        
        acc_w_x = __fmaf_rn(px_1, v_h_u_x_1, __fmaf_rn(py_1, v_h_v_x_1, acc_w_x));
        acc_w_y = __fmaf_rn(px_1, v_h_u_y_1, __fmaf_rn(py_1, v_h_v_y_1, acc_w_y));
        acc_w_z = __fmaf_rn(px_1, v_h_u_z_1, __fmaf_rn(py_1, v_h_v_z_1, acc_w_z));
    }
    
    // Сэмпл 2: (0.5, -0.5)
    if (sample_weights[2] < 1e9f) {
        const vec2& s2 = s_samples[2];
        const float v_weight_2 = grad_factor * weights[2] * vis_samples[2];
        const vec2 v_s2 = {v_weight_2 * s2.x, v_weight_2 * s2.y};
        
        vec3 h_u_2 = base_h_u + half_w_M;
        vec3 h_v_2 = base_h_v - half_w_M;
        vec3 ray_cross_2 = glm::cross(h_u_2, h_v_2);
        float inv_z_2 = __frcp_rn(ray_cross_2.z);
        
        float v_sx_pz_2 = v_s2.x * inv_z_2;
        float v_sy_pz_2 = v_s2.y * inv_z_2;
        vec3 v_ray_cross_2 = {
            v_sx_pz_2, v_sy_pz_2,
            -__fmaf_rn(v_sx_pz_2, s2.x, v_sy_pz_2 * s2.y)
        };
        
        float v_h_u_x_2 = __fmaf_rn(h_v_2.y, v_ray_cross_2.z, -h_v_2.z * v_ray_cross_2.y);
        float v_h_u_y_2 = __fmaf_rn(h_v_2.z, v_ray_cross_2.x, -h_v_2.x * v_ray_cross_2.z);
        float v_h_u_z_2 = __fmaf_rn(h_v_2.x, v_ray_cross_2.y, -h_v_2.y * v_ray_cross_2.x);
        
        float v_h_v_x_2 = __fmaf_rn(v_ray_cross_2.y, h_u_2.z, -v_ray_cross_2.z * h_u_2.y);
        float v_h_v_y_2 = __fmaf_rn(v_ray_cross_2.z, h_u_2.x, -v_ray_cross_2.x * h_u_2.z);
        float v_h_v_z_2 = __fmaf_rn(v_ray_cross_2.x, h_u_2.y, -v_ray_cross_2.y * h_u_2.x);
        
        const float px_2 = px + 0.5f;
        const float py_2 = py - 0.5f;
        
        acc_u_x -= v_h_u_x_2;
        acc_u_y -= v_h_u_y_2;
        acc_u_z -= v_h_u_z_2;
        
        acc_v_x -= v_h_v_x_2;
        acc_v_y -= v_h_v_y_2;
        acc_v_z -= v_h_v_z_2;
        
        acc_w_x = __fmaf_rn(px_2, v_h_u_x_2, __fmaf_rn(py_2, v_h_v_x_2, acc_w_x));
        acc_w_y = __fmaf_rn(px_2, v_h_u_y_2, __fmaf_rn(py_2, v_h_v_y_2, acc_w_y));
        acc_w_z = __fmaf_rn(px_2, v_h_u_z_2, __fmaf_rn(py_2, v_h_v_z_2, acc_w_z));
    }
    
    // Сэмпл 3: (-0.5, 0.5)
    if (sample_weights[3] < 1e9f) {
        const vec2& s3 = s_samples[3];
        const float v_weight_3 = grad_factor * weights[3] * vis_samples[3];
        const vec2 v_s3 = {v_weight_3 * s3.x, v_weight_3 * s3.y};
        
        vec3 h_u_3 = base_h_u - half_w_M;
        vec3 h_v_3 = base_h_v + half_w_M;
        vec3 ray_cross_3 = glm::cross(h_u_3, h_v_3);
        float inv_z_3 = __frcp_rn(ray_cross_3.z);
        
        float v_sx_pz_3 = v_s3.x * inv_z_3;
        float v_sy_pz_3 = v_s3.y * inv_z_3;
        vec3 v_ray_cross_3 = {
            v_sx_pz_3, v_sy_pz_3,
            -__fmaf_rn(v_sx_pz_3, s3.x, v_sy_pz_3 * s3.y)
        };
        
        float v_h_u_x_3 = __fmaf_rn(h_v_3.y, v_ray_cross_3.z, -h_v_3.z * v_ray_cross_3.y);
        float v_h_u_y_3 = __fmaf_rn(h_v_3.z, v_ray_cross_3.x, -h_v_3.x * v_ray_cross_3.z);
        float v_h_u_z_3 = __fmaf_rn(h_v_3.x, v_ray_cross_3.y, -h_v_3.y * v_ray_cross_3.x);
        
        float v_h_v_x_3 = __fmaf_rn(v_ray_cross_3.y, h_u_3.z, -v_ray_cross_3.z * h_u_3.y);
        float v_h_v_y_3 = __fmaf_rn(v_ray_cross_3.z, h_u_3.x, -v_ray_cross_3.x * h_u_3.z);
        float v_h_v_z_3 = __fmaf_rn(v_ray_cross_3.x, h_u_3.y, -v_ray_cross_3.y * h_u_3.x);
        
        const float px_3 = px - 0.5f;
        const float py_3 = py + 0.5f;
        
        acc_u_x -= v_h_u_x_3;
        acc_u_y -= v_h_u_y_3;
        acc_u_z -= v_h_u_z_3;
        
        acc_v_x -= v_h_v_x_3;
        acc_v_y -= v_h_v_y_3;
        acc_v_z -= v_h_v_z_3;
        
        acc_w_x = __fmaf_rn(px_3, v_h_u_x_3, __fmaf_rn(py_3, v_h_v_x_3, acc_w_x));
        acc_w_y = __fmaf_rn(px_3, v_h_u_y_3, __fmaf_rn(py_3, v_h_v_y_3, acc_w_y));
        acc_w_z = __fmaf_rn(px_3, v_h_u_z_3, __fmaf_rn(py_3, v_h_v_z_3, acc_w_z));
    }
    
    // Сэмпл 4: (0.5, 0.5)
    if (sample_weights[4] < 1e9f) {
        const vec2& s4 = s_samples[4];
        const float v_weight_4 = grad_factor * weights[4] * vis_samples[4];
        const vec2 v_s4 = {v_weight_4 * s4.x, v_weight_4 * s4.y};
        
        vec3 h_u_4 = base_h_u + half_w_M;
        vec3 h_v_4 = base_h_v + half_w_M;
        vec3 ray_cross_4 = glm::cross(h_u_4, h_v_4);
        float inv_z_4 = __frcp_rn(ray_cross_4.z);
        
        float v_sx_pz_4 = v_s4.x * inv_z_4;
        float v_sy_pz_4 = v_s4.y * inv_z_4;
        vec3 v_ray_cross_4 = {
            v_sx_pz_4, v_sy_pz_4,
            -__fmaf_rn(v_sx_pz_4, s4.x, v_sy_pz_4 * s4.y)
        };
        
        float v_h_u_x_4 = __fmaf_rn(h_v_4.y, v_ray_cross_4.z, -h_v_4.z * v_ray_cross_4.y);
        float v_h_u_y_4 = __fmaf_rn(h_v_4.z, v_ray_cross_4.x, -h_v_4.x * v_ray_cross_4.z);
        float v_h_u_z_4 = __fmaf_rn(h_v_4.x, v_ray_cross_4.y, -h_v_4.y * v_ray_cross_4.x);
        
        float v_h_v_x_4 = __fmaf_rn(v_ray_cross_4.y, h_u_4.z, -v_ray_cross_4.z * h_u_4.y);
        float v_h_v_y_4 = __fmaf_rn(v_ray_cross_4.z, h_u_4.x, -v_ray_cross_4.x * h_u_4.z);
        float v_h_v_z_4 = __fmaf_rn(v_ray_cross_4.x, h_u_4.y, -v_ray_cross_4.y * h_u_4.x);
        
        const float px_4 = px + 0.5f;
        const float py_4 = py + 0.5f;
        
        acc_u_x -= v_h_u_x_4;
        acc_u_y -= v_h_u_y_4;
        acc_u_z -= v_h_u_z_4;
        
        acc_v_x -= v_h_v_x_4;
        acc_v_y -= v_h_v_y_4;
        acc_v_z -= v_h_v_z_4;
        
        acc_w_x = __fmaf_rn(px_4, v_h_u_x_4, __fmaf_rn(py_4, v_h_v_x_4, acc_w_x));
        acc_w_y = __fmaf_rn(px_4, v_h_u_y_4, __fmaf_rn(py_4, v_h_v_y_4, acc_w_y));
        acc_w_z = __fmaf_rn(px_4, v_h_u_z_4, __fmaf_rn(py_4, v_h_v_z_4, acc_w_z));
    }
    
    // Запись результатов
    v_u_M_out = {acc_u_x, acc_u_y, acc_u_z};
    v_v_M_out = {acc_v_x, acc_v_y, acc_v_z};
    v_w_M_out = {acc_w_x, acc_w_y, acc_w_z};
}

} // namespace gsplat


#endif // GSPLAT_MIP_FILTER_CUH
 