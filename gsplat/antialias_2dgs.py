from typing import Tuple

import torch


@torch.jit.script
# @torch.compile
def forward_impl(scales: torch.Tensor,
                 opacities: torch.Tensor,
                 sigma_smooth_sq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass implementing Equations (9) and (10) from AA-2DGS.

    Args:
        scales:          [N, 3] - scaling factors (s_u, s_v, unused) of 2D Gaussian primitives
                                  Third component is ignored but preserved for compatibility
        opacities:       [N] - primitive opacities α_k
        sigma_smooth_sq: [N] - Pre-computed σ²_smooth,k = s_reg/ν̂²_k from multiview frequency bounds
                                where ν̂_k is computed using Equation (8):
                                ν̂_k = max{1_n(p_k) · f_n/d_n} over all training views n
                                This must be computed externally based on training camera parameters

    Returns:
        scales_smoothed:    [N, 3] - effective scales after smoothing (third component unchanged)
        opacities_smoothed: [N] - modulated opacities
        opacity_scale:      [N] - opacity modulation factor (cached for backward)
    """
    # Extract only the 2D components (s_u, s_v), ignore third component
    scales_2d = scales[:, :2]  # [N, 2]
    
    # Primitive's intrinsic 2D Gaussian covariance V_k = diag(s²_uk, s²_vk)
    scales_sq = scales_2d * scales_2d  # [N, 2]

    # Effective covariance after convolution (Eq. 9):
    # V_eff_k = V_k + σ²_smooth,k * I_2 = diag(s²_uk + σ²_smooth,k, s²_vk + σ²_smooth,k)
    sigma_smooth_sq_expanded = sigma_smooth_sq.unsqueeze(1)  # [N, 1]
    scales_new_sq = scales_sq + sigma_smooth_sq_expanded     # [N, 2]

    # OPTIMIZATION NOTE: Using determinants to compute opacity modulation
    # Instead of directly computing s_uk * s_vk and √(s²_uk + σ²) * √(s²_vk + σ²),
    # we use the mathematical equivalence:
    # (s_uk * s_vk) / (√(s²_uk + σ²) * √(s²_vk + σ²)) = √(det_old / det_new)
    # where det_old = s²_uk * s²_vk and det_new = (s²_uk + σ²) * (s²_vk + σ²)
    # This reduces the number of sqrt operations from 3 to 1

    # Determinants for opacity modulation
    # det(V_k) = s²_uk * s²_vk (product of diagonal elements)
    # det_old = scales_sq[:, 0] * scales_sq[:, 1]  # [N]
    det_old = scales_sq.prod(dim=1)

    # det(V_eff_k) = (s²_uk + σ²_smooth,k) * (s²_vk + σ²_smooth,k)
    # det_new = scales_new_sq[:, 0] * scales_new_sq[:, 1]  # [N]
    det_new = scales_new_sq.prod(dim=1)

    # Opacity modulation factor from Eq. (10):
    # Original formula: α_smooth_k = α_k * (s_uk * s_vk) / (√(s²_uk + σ²_smooth,k) * √(s²_vk + σ²_smooth,k))
    # Using determinant optimization: α_scale = √(det_old / det_new)
    # Mathematical proof:
    # √(det_old / det_new) = √(s²_uk * s²_vk) / √((s²_uk + σ²) * (s²_vk + σ²))
    #                      = (s_uk * s_vk) / (√(s²_uk + σ²) * √(s²_vk + σ²))
    opacity_scale = torch.sqrt(det_old / det_new)  # [N]

    # Effective scales after flat smoothing
    scales_smoothed_2d = torch.sqrt(scales_new_sq)  # [N, 2]
    
    # Reconstruct full [N, 3] tensor with unchanged third component
    scales_smoothed = torch.cat([scales_smoothed_2d, scales[:, 2:3]], dim=1)  # [N, 3]

    # Modulated opacities maintaining energy conservation
    opacities_smoothed = opacities * opacity_scale  # [N]

    return scales_smoothed, opacities_smoothed, opacity_scale


@torch.jit.script
# @torch.compile
def backward_impl(grad_scales_smoothed: torch.Tensor,
                  grad_opacities_smoothed: torch.Tensor,
                  scales: torch.Tensor,
                  scales_smoothed: torch.Tensor,
                  opacity_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass computing gradients w.r.t. original scales and opacities.

    Gradient derivations:

    1. Through effective scales:
       ∂s_eff/∂s = ∂√(s² + σ²_smooth)/∂s = s/√(s² + σ²_smooth)

    2. Through opacity modulation (Eq. 10):
       ∂α_scale/∂s_u = α_scale * d/ds_u[log(s_u/√(s²_u + σ²_smooth))]
                      = α_scale * (1/s_u - s_u/(s²_u + σ²_smooth))

    3. Through original opacity:
       ∂L/∂α_k = ∂L/∂α_smooth * α_scale
    """
    # Extract scale components (only 2D components)
    s_u = scales[:, 0]  # [N]
    s_v = scales[:, 1]  # [N]
    s_u_smooth = scales_smoothed[:, 0]  # [N]
    s_v_smooth = scales_smoothed[:, 1]  # [N]

    # Gradient through effective scales
    # Chain rule: ∂L/∂s = ∂L/∂s_eff * ∂s_eff/∂s
    # where ∂s_eff/∂s = s/s_eff = s/√(s² + σ²_smooth)
    grad_s_u = grad_scales_smoothed[:, 0] * s_u / s_u_smooth  # [N]
    grad_s_v = grad_scales_smoothed[:, 1] * s_v / s_v_smooth  # [N]

    # Gradient through opacity modulation factor
    # Compute derivatives of opacity_scale w.r.t. s_u and s_v

    # Squared smoothed scales for efficiency
    s_u_smooth_sq = s_u_smooth * s_u_smooth  # [N]
    s_v_smooth_sq = s_v_smooth * s_v_smooth  # [N]

    # Derivative of opacity modulation factor (Eq. 10):
    # ∂α_scale/∂s_u = α_scale * (1/s_u - s_u/(s²_u + σ²_smooth))
    #                = α_scale * (1/s_u - s_u/s²_u_smooth)
    d_scale_d_su = opacity_scale * (1.0 / s_u - s_u / s_u_smooth_sq)  # [N]
    d_scale_d_sv = opacity_scale * (1.0 / s_v - s_v / s_v_smooth_sq)  # [N]

    # Accumulate gradients from opacity modulation
    grad_s_u += grad_opacities_smoothed * d_scale_d_su  # [N]
    grad_s_v += grad_opacities_smoothed * d_scale_d_sv  # [N]

    # Third component gradient is just passed through unchanged
    grad_s_z = grad_scales_smoothed[:, 2]  # [N]

    # Combine gradients back into [N, 3] tensor
    grad_scales = torch.stack([grad_s_u, grad_s_v, grad_s_z], dim=1)  # [N, 3]

    # Gradient w.r.t. original opacities
    # ∂L/∂α_k = ∂L/∂α_smooth * ∂α_smooth/∂α_k = ∂L/∂α_smooth * α_scale
    grad_opacities = grad_opacities_smoothed * opacity_scale  # [N]

    return grad_scales, grad_opacities


class WorldSpaceFlatSmoothingKernel(torch.autograd.Function):
    """
    World-space flat smoothing kernel for 2D Gaussian primitives.

    Following Section 3.2 of AA-2DGS:
    - Projects isotropic 3D smoothing kernel onto the plane of 2D Gaussian primitive
    - Convolves primitive's intrinsic 2D Gaussian with projected 2D smoothing filter
    - Modulates opacity for energy conservation

    IMPORTANT: This function expects pre-computed sigma_smooth_sq values based on
    multiview frequency bounds (Equation 8 from the paper).
    """

    @staticmethod
    def forward(ctx, scales, opacities, sigma_smooth_sq):
        """
        Forward pass with context saving for backward.
        """
        # Compute forward with intermediate values
        scales_smoothed, opacities_smoothed, opacity_scale = \
            forward_impl(scales, opacities, sigma_smooth_sq)

        # Save tensors needed for backward computation
        ctx.save_for_backward(scales, scales_smoothed, opacity_scale)

        return scales_smoothed, opacities_smoothed

    @staticmethod
    def backward(ctx, grad_scales_smoothed, grad_opacities_smoothed):
        """
        Backward pass computing gradients.
        """
        scales, scales_smoothed, opacity_scale = ctx.saved_tensors

        # Compute gradients
        grad_scales, grad_opacities = backward_impl(
            grad_scales_smoothed, grad_opacities_smoothed,
            scales, scales_smoothed, opacity_scale
        )

        # sigma_smooth_sq is computed from training views, not optimized
        return grad_scales, grad_opacities, None


@torch.no_grad()
def calc_sigma_sq(max_sampling_rate: torch.Tensor, s_reg: float, focal: float | None = None, f_orig: float | None = None, blur_mod: float | None = None) -> torch.Tensor:
    # T̂ = 1/ν̂ = d/f
    # ν̂_k = max((1_n(p_k) · f_n / d_n))
    # σ²_smooth = s_reg / ν̂²_max

    # Sampling rate = f/z в pinhole модели камеры представляет собой коэффициент масштабирования между мировым и пиксельным пространством на глубине z.
    #
    # Физический смысл:
    #
    # Определяет количество пикселей на единицу длины в мировом пространстве на расстоянии z от камеры
    # Показывает плотность пространственной дискретизации сцены на данной глубине
    # При увеличении z (удалении от камеры) sampling rate уменьшается - меньше пикселей покрывает ту же физическую область
    # Практическое значение:
    #
    # Объекты ближе к камере имеют больший sampling rate - выше детализация
    # Объекты дальше от камеры имеют меньший sampling rate - ниже детализация
    # Критичен для алгоритмов рендеринга и 3D реконструкции для определения уровня детализации (LOD)

    # Для широкоугольных объективов f/z и f/d существенно отличаются на краях изображения.
    #
    # Геометрическая причина:
    # z - расстояние вдоль оптической оси
    # d - евклидово расстояние от центра камеры до точки
    # Для точки на краю: d = z / cos(θ), где θ - угол между лучом и оптической осью
    # Для широкоугольных объективов:
    #
    # Угол θ на краях может достигать 40-60° и более
    # cos(60°) = 0.5, следовательно d = 2z
    # f/z будет в 2 раза больше f/d на таких углах
    # Практические следствия:
    #
    # f/z переоценивает sampling rate на периферии
    # f/d дает более корректную оценку пространственного разрешения вдоль луча
    # Для fisheye объективов (θ → 90°) разница становится экстремальной


    # Подход с динамическим пересчётом ν̂_k_new = ν̂_k_original × (f_new / f_reference)

    # Математическое обоснование:
    # Из формулы статьи world space sampling interval T̂ = d/f следует, что частота дискретизации ν̂ = f/d.
    # При изменении focal length с f_reference на f_new:
    # ν̂_new = f_new/d
    # ν̂_original = f_reference/d
    # ν̂_new/ν̂_original = f_new/f_reference
    # Соответственно, новые параметры flat smoothing:
    # σ²_smooth,k_new = s_reg/ν̂²_k_new = σ²_smooth,k_original × (f_reference/f_new)²

    # Предсохранение отношения ν̂_k_original / f_reference упрощает динамическое обновление параметров при изменении focal length.
    # Вместо абсолютной частоты ν̂_k сохраняется нормализованная величина:
    # d_k = ν̂_k_original / f_reference = 1/depth_k

    # Вычисления при новом focal:
    # При изменении focal на f_new:
    # ν̂_k_new = d_k × f_new
    # σ²_smooth,k_new = s_reg / (d_k × f_new)²
    # Физический смысл d_k - геометрическая характеристика примитива, не зависящая от параметров камеры

    # From AA-2DGS Eq. (8): ν̂_k = max{f_n/d_n} over training views
    # With max_sampling_rate = max{f_n/d_n}, we have:
    # σ²_smooth,k = s_reg / ν̂²_k = s_reg / max_sampling_rate²

    # For gaussians never visible in training views (max_sampling_rate == 0),
    # set sigma_smooth_sq = 0 to disable smoothing
    # For visible gaussians, compute as usual: σ²_smooth,k = s_reg / max_sampling_rate²

    # σ²smooth,k,0 = sreg · d²/f₀²
    # σ²smooth,k,1 = sreg · d²/f₁²
    # σ²smooth,k,1 = σ²smooth,k,0 · (f₀/f₁)²

    if focal and f_orig:
        mult_sq = s_reg * (focal / f_orig)
    else:
        mult_sq = s_reg

    if blur_mod:
        mult_sq *= blur_mod ** 2

    sigma_smooth_sq = torch.where(
        max_sampling_rate > 0,
        mult_sq / (max_sampling_rate ** 2),
        torch.zeros_like(max_sampling_rate)
    )  # [N]

    return sigma_smooth_sq


@torch.no_grad()
# @torch.compile
def compute_distances_and_frustum_mask(
        means_cam: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
        near_plane: float,
        far_plane: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute euclidean distances and frustum visibility mask for gaussians.
    
    Args:
        means_cam: [N, 3] - Gaussian positions in camera space
        K: [3, 3] - Camera intrinsic matrix
        width: Image width
        height: Image height
        near_plane: Near clipping plane distance
        far_plane: Far clipping plane distance
        
    Returns:
        distances: [N] - Euclidean distances from camera to gaussians
        visible: [N] - Boolean mask of gaussians visible in frustum
    """
    device = means_cam.device
    N = means_cam.shape[0]
    
    # Extract z-coordinate for front/back culling
    z_cam = means_cam[:, 2]  # [N]
    
    # Compute euclidean distance from camera to points in camera space
    # (camera at origin, so distance = norm of position vector)
    # Note: Using euclidean distance instead of z-depth can be more physically correct
    # for wide-angle cameras as it accounts for pixel "stretching" at image periphery.
    # At image edges, pixels cover larger solid angles and world-space areas.
    # Euclidean distance naturally compensates: sampling_rate = f/distance = f·cos(angle)/z,
    # where angle is from optical axis, giving lower sampling rates at edges as expected.
    distances = torch.norm(means_cam, dim=1)  # [N]
    
    # Check if points are in front of camera (z > 0) and within clipping planes
    valid_mask = (z_cam > 0) & (z_cam > near_plane) & (z_cam < far_plane)
    
    # Compute frustum halfplanes for culling
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    margin = 0.15  # 15% margin for Gaussian extent
    w_margin = margin * width
    h_margin = margin * height
    
    left = -w_margin
    top = -h_margin
    bottom = height + h_margin
    right = width + w_margin
    
    # Visible corners in image space
    corners_img = torch.tensor([
        [left,  top,    1],  # Top-left
        [left,  bottom, 1],  # Bottom-left
        [right, bottom, 1],  # Bottom-right
        [right, top,    1]   # Top-right
    ], dtype=torch.float32, device=device)
    
    # Convert corners to camera space at near plane
    corners_cam = torch.zeros((4, 3), device=device)
    corners_cam[:, 0] = (corners_img[:, 0] - cx) * near_plane / fx
    corners_cam[:, 1] = (corners_img[:, 1] - cy) * near_plane / fy
    corners_cam[:, 2] = near_plane
    
    # Compute normals for the half-planes in camera space
    normals = []
    for i in range(4):
        next_i = (i + 1) % 4
        # Two vectors in the plane
        v1 = corners_cam[i, :3]
        v2 = corners_cam[next_i, :3]
        # Cross product to get the normal, pointing inward
        normal = torch.cross(v2, v1, dim=0)
        normal /= torch.norm(normal, dim=0)
        normals.append(normal)
    
    frustum_halfplanes = torch.stack(normals)  # [4, 3]
    
    # Check if points are inside frustum using halfplanes
    dots = torch.matmul(frustum_halfplanes, means_cam.transpose(0, 1))  # [4, N]
    in_frustum = torch.all(dots > 0, dim=0)  # [N]
    
    # Combined visibility mask
    visible = valid_mask & in_frustum
    
    return distances, visible


def apply_flat_smoothing(scales: torch.Tensor, opacities: torch.Tensor, max_sampling_rate: torch.Tensor,
                         s_reg: float = 0.2, focal: float = None, f_orig: float | None = None, blur_mod: float | None = None, threshold: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies world-space flat smoothing kernel to 2D Gaussian primitives.

    Implements Section 3.2 of AA-2DGS paper:
    - Constrains frequency content of 2D Gaussian primitives based on sampling rates
    - Ensures primitives respect Nyquist-Shannon sampling theorem
    - Prevents high-frequency artifacts when zooming in

    Args:
        scales: [N, 3] - primitive scaling factors (s_uk, s_vk, unused)
                Third component is preserved but not used in smoothing
        opacities: [N] - primitive opacities α_k
        max_sampling_rate_sq: [N] Pre-computed max((f/d)^2) from all training cameras
                          Represents squared maximal sampling rate
        s_reg: hyperparameter (typically 0.2 as per paper)
        focal: focal length of the current camera
        f_orig: focal length of the training cameras
        blur_mod: blur modulation factor (optional)
        threshold: threshold for relative change to skip smoothing (default 0.01)

    Returns:
        scales_smoothed: [N, 3] - effective scales after smoothing (third component unchanged)
        opacities_smoothed: [N] - modulated opacities for energy conservation

    Note:
        The sigma_smooth_sq values should be computed
        based on the current primitive positions and training camera parameters.

        # sigma_smooth_sq: [N] - σ²_smooth,k = s_reg/ν̂²_k where:
        #                  - s_reg is a hyperparameter (typically 0.2 as per paper)
        #                  - ν̂²_k is the squared maximal sampling frequency from Eq. (8):
        #                    ν̂_k = max{1_n(p_k) · f_n/d_n} over all training views
        #                   This must be pre-computed based on training camera parameters
    """

    sigma_smooth_sq = calc_sigma_sq(max_sampling_rate_sq, s_reg, focal, f_orig, blur_mod)
    
    # # Check if smoothing would be negligible
    # # Compute relative change: sigma_smooth / min(s_u, s_v)
    # sigma_smooth = torch.sqrt(sigma_smooth_sq)  # [N]
    # min_scales = scales[:, :2].min(dim=1).values  # [N]
    # relative_change = sigma_smooth / min_scales  # [N]
    #
    # # If all changes are below threshold, skip smoothing
    # if (relative_change < threshold).all():
    #     return scales, opacities
    #
    # print("apply scaling", torch.quantile(relative_change, 0.05), torch.quantile(relative_change, 0.95))
    return WorldSpaceFlatSmoothingKernel.apply(scales, opacities, sigma_smooth_sq)

# # Проверка типа скомпилированной функции
# print(type(WorldSpaceFlatSmoothingKernel.forward_impl))
# # Выведет: <class 'torch.jit.ScriptFunction'> если скомпилировано
#
# # Проверка наличия JIT-компиляции
# print(isinstance(WorldSpaceFlatSmoothingKernel.forward_impl, torch.jit.ScriptFunction))
#
# # Просмотр IR-графа JIT-скомпилированной функции
# print(WorldSpaceFlatSmoothingKernel.forward_impl.graph)
#
# # Просмотр сгенерированного кода
# print(WorldSpaceFlatSmoothingKernel.forward_impl.code)

# # Optional: torch.compile optimization for PyTorch 2.0+
# if hasattr(torch, 'compile'):
#     apply_flat_smoothing_compiled = torch.compile(
#         apply_flat_smoothing,
#         mode="reduce-overhead",
#         fullgraph=True
#     )
# else:
#     apply_flat_smoothing_compiled = apply_flat_smoothing


@torch.no_grad()
# @torch.compile
def compute_max_sampling_rate_for_all_cams(
    means: torch.Tensor,
    Ks: list[torch.Tensor],
    widths: list[int],
    heights: list[int],
    viewmats: list[torch.Tensor],
    near_plane: float,
    far_plane: float,
    device: torch.device | str
) -> torch.Tensor:
    """
    Compute maximum sampling rate for each Gaussian across all training views.
    
    Args:
        means: [N, 3] - 3D positions of Gaussians
        Ks: List of camera intrinsic matrices [3, 3]
        widths: List of image widths
        heights: List of image heights
        viewmats: List of view matrices (world-to-camera) [4, 4]
        near_plane: Near clipping plane distance
        far_plane: Far clipping plane distance
        device: Torch device
        
    Returns:
        max_sampling_rate: [N] - Maximum (f/d) for each Gaussian across training views
    """
    N = means.shape[0]
    means = means.clone().detach().to(device)

    # Initialize with zeros to find maximum
    max_sampling_rate = torch.zeros((N, 1), device=device)
    visibility_count = torch.zeros((N, 1), device=device)

    for K, width, height, viewmat in zip(Ks, widths, heights, viewmats):
        K = K.to(device)
        viewmat = viewmat.to(device)
        
        # Transform points to camera space
        means_homo = torch.cat([means, torch.ones(N, 1, device=device)], dim=1)  # [N, 4]
        # [4, 4] @ [4, N] -> [4, N] -> [N, 4]
        means_cam = viewmat @ means_homo.transpose(0, 1)  # [4, N]
        means_cam = means_cam.transpose(0, 1)[:, :3]  # [N, 3]

        # Compute distances and visibility using the new function
        distances, visible = compute_distances_and_frustum_mask(
            means_cam, K, width, height, near_plane, far_plane
        )

        if visible.any():
            # Compute sampling rate for visible Gaussians using real distances
            visible_distances = distances[visible]

            # Use average of fx and fy for more robust frequency estimation
            fx = K[0, 0]
            fy = K[1, 1]
            focal = float((fx + fy) / 2.0)

            # Compute (f/d) for each visible Gaussian
            sampling_rate = focal / visible_distances  # [visible_count]
            sampling_rate = sampling_rate.unsqueeze(1)  # [visible_count, 1]

            # Update maximum sampling rate
            max_sampling_rate[visible] = torch.maximum(max_sampling_rate[visible], sampling_rate)
            visibility_count[visible] += 1

    # Handle Gaussians never visible in training views
    never_visible = visibility_count == 0
    if never_visible.any():
        if (~never_visible).any():
            default_sampling_rate = torch.max(max_sampling_rate[~never_visible])
        else:
            # Fallback if all gaussians are never visible
            default_sampling_rate = 0
        max_sampling_rate[never_visible] = default_sampling_rate

    return max_sampling_rate.squeeze(1)  # [N]


@torch.no_grad()
def update_max_sampling_rate(
    splats: torch.nn.ParameterDict,
    strategy_state: dict,
    trainset,
    near_plane: float,
    far_plane: float,
    device: torch.device | str
):
    """Compute maximum sampling rate for AA-2DGS smoothing.
    
    Args:
        splats: ParameterDict containing gaussian parameters (including max_sampling_rate)
        trainset: Training dataset with camera parameters
        near_plane: Near clipping plane distance
        far_plane: Far clipping plane distance
        device: Torch device
        strategy_state: Optional strategy state dictionary (for backward compatibility)
    """

    if "max_sampling_rate" not in splats:
        return

    last_epoch_max_sampling_rate = None
    last_epoch_valid_mask = None

    # Try to get accumulated max_sampling_rate from strategy_state if provided
    if strategy_state is not None and "epoch_stats" in strategy_state:
        if hasattr(strategy_state["epoch_stats"], "max_sampling_rate"):
            last_epoch_max_sampling_rate = strategy_state["epoch_stats"].max_sampling_rate.clone().detach()
            # Filter out zero values (gaussians never visible in training views)
            last_epoch_valid_mask = last_epoch_max_sampling_rate > 0

    # Extract camera parameters from trainset
    if hasattr(trainset, 'scene_info') and trainset.scene_info is not None:
        # New structure with scene_info
        scene_info = trainset.scene_info
        train_cameras = scene_info.train_cameras
        
        Ks = []
        widths = []
        heights = []
        viewmats = []
        
        for cam in train_cameras:
            Ks.append(torch.from_numpy(cam.K).float().to(device))
            widths.append(cam.width)
            heights.append(cam.height)
            
            # Compute view matrix from R and T
            from loaders.dataset_readers import getWorld2View_npy
            W2C = getWorld2View_npy(cam.R, cam.T)
            viewmats.append(torch.from_numpy(W2C).float().to(device))
    else:
        # Legacy structure with parser
        parser = trainset.parser
        Ks = [torch.asarray(parser.Ks_dict[k_id]).float().to(device) for k_id in parser.camera_ids]
        widths = [parser.imsize_dict[k_id][0] for k_id in parser.camera_ids]
        heights = [parser.imsize_dict[k_id][1] for k_id in parser.camera_ids]
        viewmats = [torch.linalg.inv(torch.asarray(camtoworld).float().to(device)) 
                    for camtoworld in parser.camtoworlds]

    if last_epoch_valid_mask is None or not last_epoch_valid_mask.any():
        max_sampling_rate = compute_max_sampling_rate_for_all_cams(
            means=splats["means"],
            Ks=Ks,
            widths=widths,
            heights=heights,
            viewmats=viewmats,
            near_plane=near_plane,
            far_plane=far_plane,
            device=device
        )
        splats["max_sampling_rate"].data = max_sampling_rate
        return

    # For gaussians with zero values, compute only for invalid ones
    if not last_epoch_valid_mask.all():
        # Get means only for invalid gaussians
        invalid_mask = ~last_epoch_valid_mask

        print("last epoch invalid: ", invalid_mask.sum().item())

        # Compute max_sampling_rate only for invalid gaussians
        uncached_max_sampling_rate = compute_max_sampling_rate_for_all_cams(
            means=splats["means"][invalid_mask],
            Ks=Ks,
            widths=widths,
            heights=heights,
            viewmats=viewmats,
            near_plane=near_plane,
            far_plane=far_plane,
            device=device
        )

        # Update only invalid positions
        last_epoch_max_sampling_rate[invalid_mask] = uncached_max_sampling_rate

    splats["max_sampling_rate"].data = last_epoch_max_sampling_rate