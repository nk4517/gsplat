#ifndef GSPLAT_MIP_FILTER_CUH
#define GSPLAT_MIP_FILTER_CUH

#include "Common.h"

namespace gsplat {

// Antialiasing method selection
enum AAMethod { 
    DEFAULT = 0,  // Original 2DGS method - minimum of ray-intersection and 2D gaussian
    AA_2DGS = 1,  // Mip-NeRF 360 style filter with Jacobian-based covariance (arXiv:2506.11252)
};

// Change this to select antialiasing method globally
constexpr AAMethod AA_METHOD = AA_2DGS;

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

} // namespace gsplat


#endif // GSPLAT_MIP_FILTER_CUH
 