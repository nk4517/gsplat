import typing
from typing import Dict, Optional, Any, TYPE_CHECKING

import torch
from nerfview import apply_float_colormap

from examples.lib_compose import compose_renders, CompositingOrder
from examples.utils import scalar_to_colormap, normalize_robust, index_map_to_pseudocolor
from gsplat.antialias_2dgs import calc_sigma_sq
from gsplat.strategy.ops import scaling_activation

if TYPE_CHECKING:
    from nerfview import CameraState
    from examples.gsplat_viewer_2dgs import GsplatRenderTabState
    from examples.simple_trainer_2dgs import Config
    from gsplat.strategy.epoch_stats import EpochStatistics


def render_inner(
        camera_state: "CameraState",
        render_tab_state: "GsplatRenderTabState",
        device: str,
        splats: torch.nn.ParameterDict,
        parser_Ks_dict: Dict,
        cfg: "Config",
        n_cameras: int,
        rasterize_splats_fn,
        epoch_stats: Optional["EpochStatistics"],
        skysphere_model=None,
        rasterize_sky_fn=None):


    if render_tab_state.preview_render:
        width = render_tab_state.render_width
        height = render_tab_state.render_height
    else:
        width = render_tab_state.viewer_width
        height = render_tab_state.viewer_height
    c2w = camera_state.c2w
    K = camera_state.get_K((width, height))
    c2w = torch.from_numpy(c2w).float().to(device)
    K = torch.from_numpy(K).float().to(device)

    # # Create detached copy of splats for viewer rendering
    # viewer_splats = {k: v.clone().detach() for k, v in splats.items()}
    viewer_splats = splats

    focal = float(K[0, 0] + K[1, 1])/2  # Use focal length from K matrix
    K_orig = list(parser_Ks_dict.values())[0]
    f_orig = float(K_orig[0, 0] + K_orig[1, 1])/2

    blur_mod = render_tab_state.blur_mod

    # Prepare override colors for colormapped visualization
    override_colors = None
    overmax_opacity = False

    if render_tab_state.render_mode == "max_sampling_rate" and "max_sampling_rate" in splats:
        max_sampling_rate = splats["max_sampling_rate"].detach()
        override_colors = scalar_to_colormap(
            max_sampling_rate,
            colormap=render_tab_state.colormap,
            inverse=render_tab_state.inverse,
            explicit_min=10,
            explicit_max=1000,
        ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]

    elif render_tab_state.render_mode == "accumulated_max_sampling_rate":
        # Use accumulated max_sampling_rate from epoch statistics if available
        if epoch_stats and hasattr(epoch_stats, "max_sampling_rate"):
            accumulated_max_sampling = epoch_stats.max_sampling_rate.clone().detach()
            override_colors = scalar_to_colormap(
                accumulated_max_sampling,
                colormap=render_tab_state.colormap,
                inverse=render_tab_state.inverse,
                explicit_min=10,
                explicit_max=1000,
            ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]

    elif render_tab_state.render_mode == "sigma_smooth" and "max_sampling_rate" in splats:
        max_sampling_rate = splats["max_sampling_rate"].clone().detach()
        # Calculate smoothing sigma squared
        # изменения вблизи очень слабозаметны, хотя и применяются правильно
        sigma_smooth = torch.sqrt(calc_sigma_sq(max_sampling_rate, cfg.aa_smoothing_reg, focal, f_orig))
        scales = scaling_activation(viewer_splats["scales"])  # [N, 3]
        min_scales = scales[:, :2].min(dim=1).values  # [N]
        relative_change = sigma_smooth / min_scales  # [N]
        override_colors = scalar_to_colormap(
            relative_change,
            colormap=render_tab_state.colormap,
            inverse=render_tab_state.inverse,
            # explicit_min=0.001,
            # explicit_max=2.0
        ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]

    elif render_tab_state.render_mode == "n_cameras_visible":
        # Use n_cameras_visible_from from epoch statistics if available
        if epoch_stats and hasattr(epoch_stats, "n_cameras_visible_from"):
            n_cameras_seen = epoch_stats.n_cameras_visible_from.clone().float()
            n_cameras_seen[n_cameras_seen == 0] = -n_cameras  # чтобы палитра начиналась с середины, а ноль выделялся цветом
            override_colors = scalar_to_colormap(
                n_cameras_seen,
                colormap=render_tab_state.colormap,
                inverse=render_tab_state.inverse,
                explicit_min=-n_cameras,
                explicit_max=n_cameras,
            ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]

    elif render_tab_state.render_mode == "skyness":
        # Skyness visualization not available (moved to separate skysphere model)
        override_colors = torch.zeros((len(viewer_splats["means"]), 1, 3), device=device)

    elif render_tab_state.render_mode == "elongation":
        # Calculate elongation in log space (scales are stored in log form)
        log_scales = viewer_splats["scales"][..., :2]  # [N, 2] - 2DGS, only x,y scales (no activation)
        log_elongation_ratio = torch.abs(log_scales[:, 0] - log_scales[:, 1])  # [N] - abs(log_x - log_y) = log(max/min)
        elongation_ratio = torch.exp(log_elongation_ratio)  # Convert from log space to actual ratio
        override_colors = scalar_to_colormap(
            elongation_ratio,
            colormap=render_tab_state.colormap,
            inverse=render_tab_state.inverse,
            explicit_min=1.0,
            explicit_max=10.0,
        ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]
        # overmax_opacity = True  # Use maximum opacity for better visibility

    elif render_tab_state.render_mode == "effective_rank":
        # Calculate effective rank based on scale proportions entropy
        # For 2DGS: scales are standard deviations along axes
        activated_scales = scaling_activation(viewer_splats["scales"])[..., :2].detach().clone()  # [N, 2]

        # Square the scales (variance proportional to squared std dev)
        scales_squared = activated_scales ** 2  # [N, 3]

        # Calculate proportions: p_i = scale_i^2 / sum(scales^2)
        sum_scales_squared = scales_squared.sum(dim=1, keepdim=True)  # [N, 1]
        proportions = scales_squared / (sum_scales_squared + 1e-10)  # [N, 3]

        # Calculate entropy: H = -sum(p_i * log(p_i))
        # Add small epsilon to avoid log(0)
        proportions_safe = torch.clamp(proportions, min=1e-10)
        entropy = -(proportions_safe * torch.log(proportions_safe)).sum(dim=1)  # [N]

        # Effective rank = exp(entropy)
        effective_rank = torch.exp(entropy)  # [N]

        override_colors = scalar_to_colormap(
            effective_rank,
            colormap=render_tab_state.colormap,
            inverse=render_tab_state.inverse,
            explicit_min=1.0,
            explicit_max=2.0,
        ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]

    elif render_tab_state.render_mode == "grad2d_accum":
        # Visualize accumulated gradient magnitudes
        if epoch_stats and hasattr(epoch_stats, "grad2d_abs"):
            grad2d = epoch_stats.grad2d_abs.clone()
            count = epoch_stats.count.clone()
            # Average gradient per visibility count
            avg_grad = torch.where(count > 0, grad2d / count.clamp_min(1), torch.zeros_like(grad2d))
            override_colors = scalar_to_colormap(
                avg_grad,
                colormap=render_tab_state.colormap,
                inverse=render_tab_state.inverse,
                explicit_min=0.0,
                explicit_max=cfg.grow_grad2d * 2,  # Scale relative to grow threshold
            ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]
        else:
            # Fallback if no gradient data available
            override_colors = torch.zeros((len(viewer_splats["means"]), 1, 3), device=device)

    elif render_tab_state.render_mode == "grad2d_count":
        # Visualize visibility count (how many times each gaussian was visible)
        if epoch_stats and hasattr(epoch_stats, "count"):
            count = epoch_stats.count.clone()
            override_colors = scalar_to_colormap(
                count,
                colormap=render_tab_state.colormap,
                inverse=render_tab_state.inverse,
                explicit_min=0,
                explicit_max=n_cameras,  # Max is number of training views
            ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]
        else:
            # Fallback if no count data available
            override_colors = torch.zeros((len(viewer_splats["means"]), 1, 3), device=device)

    elif render_tab_state.render_mode == "gcr":
        # Visualize Gradient Consistency Ratio (GCR) from GDAGS
        if epoch_stats and hasattr(epoch_stats, "grad2d") and hasattr(epoch_stats, "grad2d_abs"):
            grad2d = epoch_stats.grad2d.clone()
            grad2d_abs = epoch_stats.grad2d_abs.clone()
            count = epoch_stats.count.clone()
            avg_grad = torch.where(count > 0, grad2d / count.clamp_min(1), torch.zeros_like(grad2d))
            avg_grad_abs = torch.where(count > 0, grad2d_abs / count.clamp_min(1), torch.zeros_like(grad2d_abs))
            # Compute GCR = grad / grad_abs
            gcr = (avg_grad + 1e-8) / (avg_grad_abs + 1e-8)
            gcr = torch.clamp(gcr, 0.0, 1.0)  # Clamp to [0, 1]
            override_colors = scalar_to_colormap(
                gcr,
                colormap=render_tab_state.colormap,
                inverse=render_tab_state.inverse,
                explicit_min=0.0,
                explicit_max=1.0,
            ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]
        else:
            # Fallback if no gradient data available
            override_colors = torch.zeros((len(viewer_splats["means"]), 1, 3), device=device)

    elif render_tab_state.render_mode == "gdags_weight":
        # Visualize GDAGS weight: w = 0.8 + 25 * (1 - GCR)^15
        if epoch_stats and hasattr(epoch_stats, "grad2d") and hasattr(epoch_stats, "grad2d_abs"):
            grad2d = epoch_stats.grad2d.clone()
            grad2d_abs = epoch_stats.grad2d_abs.clone()
            count = epoch_stats.count.clone()
            avg_grad = torch.where(count > 0, grad2d / count.clamp_min(1), torch.zeros_like(grad2d))
            avg_grad_abs = torch.where(count > 0, grad2d_abs / count.clamp_min(1), torch.zeros_like(grad2d_abs))
            # Compute GCR = grad / grad_abs
            gcr = (avg_grad + 1e-8) / (avg_grad_abs + 1e-8)
            gcr = torch.clamp(gcr, 0.0, 1.0)  # Clamp to [0, 1]
            # Compute GDAGS weight
            weight = 0.8 + 25 * torch.pow(1 - gcr, 15)
            override_colors = scalar_to_colormap(
                weight,
                colormap=render_tab_state.colormap,
                inverse=render_tab_state.inverse,
                explicit_min=0.8,
                explicit_max=25.8,
            ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]
        else:
            # Fallback if no gradient data available
            override_colors = torch.zeros((len(viewer_splats["means"]), 1, 3), device=device)

    elif render_tab_state.render_mode == "grad2d_gcr_combined":
        # Combined visualization: grad2d_abs determines intensity, gcr determines hue
        if epoch_stats and hasattr(epoch_stats, "grad2d") and hasattr(epoch_stats, "grad2d_abs"):
            grad2d = epoch_stats.grad2d.clone()
            grad2d_abs = epoch_stats.grad2d_abs.clone()
            count = epoch_stats.count.clone()
            # Average gradients per visibility count
            avg_grad = torch.where(count > 0, grad2d / count.clamp_min(1), torch.zeros_like(grad2d))
            avg_grad_abs = torch.where(count > 0, grad2d_abs / count.clamp_min(1), torch.zeros_like(grad2d_abs))

            # Compute GCR = grad / grad_abs
            gcr = (avg_grad + 1e-8) / (avg_grad_abs + 1e-8)
            gcr = torch.clamp(gcr, 0.0, 1.0)  # Clamp to [0, 1]

            # Normalize grad2d_abs to [0, 1]
            grad_norm = torch.clamp(avg_grad_abs / (cfg.grow_grad2d * 2), 0.0, 1.0)

            # Create color mapping:
            # Low grad_norm → pastel blue (0.7, 0.85, 1.0)
            # High grad_norm + low gcr → pastel red (1.0, 0.7, 0.7)
            # High grad_norm + high gcr → pastel green (0.7, 1.0, 0.7)

            # Base pastel blue
            base_color = torch.tensor([0.3, 0.4, 1.0], device=device)
            # Target colors based on gcr
            red_color = torch.tensor([1.0, 0.3, 0.3], device=device)
            green_color = torch.tensor([0.3, 1.0, 0.3], device=device)

            # Interpolate between red and green based on gcr
            target_color = red_color * (1 - gcr).unsqueeze(-1) + green_color * gcr.unsqueeze(-1)

            # Interpolate between base and target based on grad_norm
            colors = base_color * (1 - grad_norm).unsqueeze(-1) + target_color * grad_norm.unsqueeze(-1)

            override_colors = colors.unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]
        else:
            # Fallback if no gradient data available
            override_colors = torch.zeros((len(viewer_splats["means"]), 1, 3), device=device)

    elif render_tab_state.render_mode == "importance":
        # Visualize importance (vG^2) from accumulated gradients
        if epoch_stats and hasattr(epoch_stats, "importance"):
            importance = epoch_stats.importance.clone()
            count = epoch_stats.count.clone()

            # Normalize by number of cameras where gaussian was visible
            avg_importance = torch.where(count > 0, importance / count.clamp_min(1), torch.zeros_like(importance))
            #
            # Use logarithmic scale for better visualization
            log_importance = torch.log10(avg_importance + 1e-10)

            override_colors = scalar_to_colormap(
                avg_importance,
                colormap=render_tab_state.colormap,
                inverse=render_tab_state.inverse,
                # explicit_min=-6,  # 10^-6
                # explicit_max=-2,  # 10^-2
            ).unsqueeze(1)  # Reshape for rasterization: [N, 1, 3]
        else:
            # Fallback if no importance data available
            override_colors = torch.zeros((len(viewer_splats["means"]), 1, 3), device=device)

    (
        render_colors,
        render_alphas,
        render_normals,
        normals_from_depth,
        render_distort,
        render_median,
        info,
    ) = rasterize_splats_fn(
        camtoworlds=c2w[None],
        Ks=K[None],
        width=width,
        height=height,
        splats=viewer_splats,
        sh_degree=min(render_tab_state.max_sh_degree, cfg.sh_degree),
        near_plane=render_tab_state.near_plane,
        far_plane=render_tab_state.far_plane,
        radius_clip=render_tab_state.radius_clip,
        eps2d=render_tab_state.eps2d,
        render_mode="RGB+ED",
        backgrounds=torch.tensor([render_tab_state.backgrounds], device=device) / 255.0,
        track_domination=True,
        distloss=render_tab_state.render_mode == "distort",
        override_colors=override_colors,
        overmax_opacity=overmax_opacity,
        rasterize_mode=render_tab_state.rasterize_mode,
        f_orig=f_orig,
        blur_mod=blur_mod,
    )  # [1, H, W, 3]
    render_tab_state.total_gs_count = len(viewer_splats["means"])
    render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

    if render_tab_state.render_mode in ("depth(expected)", "depth(dominating)"):
        if render_tab_state.render_mode == "depth(dominating)":
            depth = info["dominating_depths"][0, ..., None]
        else:
            depth = render_median[0, ..., 0:1]
        # normalize depth to [0, 1]
        if render_tab_state.normalize_nearfar:
            near_plane = render_tab_state.near_plane
            far_plane = render_tab_state.far_plane
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
        else:
            depth_norm = normalize_robust(depth)
        if render_tab_state.inverse:
            depth_norm = 1 - depth_norm
        renders = (
            apply_float_colormap(depth_norm, render_tab_state.colormap)
            .cpu()
            .numpy()
        )
    elif render_tab_state.render_mode == "normal":
        render_normals = render_normals[0, ..., 0:3] * 0.5 + 0.5  # normalize to [0, 1]
        renders = render_normals.cpu().numpy()
    elif render_tab_state.render_mode == "alpha":
        alpha = render_alphas[0, ..., 0:1]
        renders = (
            apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
        )
    elif render_tab_state.render_mode == "domination":
        renders = (
            index_map_to_pseudocolor(info["dominating_gauss_ids"][0, ...])
            .cpu()
            .numpy()
        )
    elif render_tab_state.render_mode == "distort":
        dist = render_distort[0, ..., 0:1]
        # normalize distortion to [0, 1]
        if render_tab_state.normalize_nearfar:
            # Use near/far plane for normalization
            near_plane = render_tab_state.near_plane
            far_plane = render_tab_state.far_plane
            dist_norm = (dist - near_plane) / (far_plane - near_plane + 1e-10)
            dist_norm = torch.clip(dist_norm, 0, 1)
        else:
            # Use robust normalization to exclude outliers
            dist_norm = normalize_robust(dist)
        if render_tab_state.inverse:
            dist_norm = 1 - dist_norm
        renders = (
            apply_float_colormap(dist_norm, render_tab_state.colormap)
            .cpu()
            .numpy()
        )
    else:
        render_colors = render_colors[0, ..., 0:3].clamp(0, 1)

        # Composite with skysphere if enabled
        if skysphere_model is not None and rasterize_sky_fn is not None:
            sky_colors, sky_alphas, _ = rasterize_sky_fn(
                camtoworlds=c2w[None],
                Ks=K[None],
                width=width,
                height=height,
            )
            if sky_colors is not None:
                render_colors = render_colors * render_alphas[0] + sky_colors[0] * (1 - render_alphas[0])

        renders = render_colors.cpu().numpy()
    return renders
