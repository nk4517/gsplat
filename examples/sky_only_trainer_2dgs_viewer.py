import torch
from nerfview import CameraState
from nerfview import apply_float_colormap

from examples.utils import index_map_to_pseudocolor, scalar_to_colormap, normalize_robust

def safe_divide(numerator, denominator, min_denom=1):
    """Безопасное деление с проверкой на ноль."""
    return torch.where(denominator > 0, 
                      numerator / denominator.clamp_min(min_denom), 
                      torch.zeros_like(numerator))


def compute_gcr(grad2d, grad2d_abs, count):
    """Вычисление Gradient Consistency Ratio."""
    avg_grad = safe_divide(grad2d, count)
    avg_grad_abs = safe_divide(grad2d_abs, count)
    gcr = (avg_grad + 1e-8) / (avg_grad_abs + 1e-8)
    return torch.clamp(gcr, 0.0, 1.0), avg_grad, avg_grad_abs


def skysphere_renderer(device, rasterize_fn, epoch_stats, trainset_len, grow_grad2d, n_points, sh_background, cfg, camera_state: CameraState, render_tab_state):

    width = render_tab_state.viewer_width
    height = render_tab_state.viewer_height
    c2w = camera_state.c2w
    K = camera_state.get_K((width, height))
    c2w = torch.from_numpy(c2w).float().to(device)
    K = torch.from_numpy(K).float().to(device)

    # Process epoch_stats-based visualizations
    if epoch_stats is not None and render_tab_state.render_mode in [
        "grad2d_accum", "grad2d_count", "gcr", "gdags_weight", 
        "grad2d_gcr_combined", "importance"
    ]:
        override_colors = None
        
        if render_tab_state.render_mode == "grad2d_accum" and hasattr(epoch_stats, "grad2d_abs"):
            # Visualize accumulated gradient magnitudes
            grad2d = epoch_stats.grad2d_abs.clone()
            count = epoch_stats.count.clone()
            avg_grad = safe_divide(grad2d, count)
            override_colors = scalar_to_colormap(
                avg_grad,
                colormap=render_tab_state.colormap,
                inverse=render_tab_state.inverse,
                explicit_min=0.0,
                explicit_max=grow_grad2d * 2,
            )
            
        elif render_tab_state.render_mode == "grad2d_count" and hasattr(epoch_stats, "count"):
            # Visualize visibility count (how many times each gaussian was visible)
            count = epoch_stats.count.clone()
            override_colors = scalar_to_colormap(
                count,
                colormap=render_tab_state.colormap,
                inverse=render_tab_state.inverse,
                explicit_min=0,
                explicit_max=trainset_len,  # Max is number of training views
            )
            
        elif render_tab_state.render_mode == "gcr" and hasattr(epoch_stats, "grad2d") and hasattr(epoch_stats, "grad2d_abs"):
            # Visualize Gradient Consistency Ratio (GCR) from GDAGS
            grad2d = epoch_stats.grad2d.clone()
            grad2d_abs = epoch_stats.grad2d_abs.clone()
            count = epoch_stats.count.clone()
            gcr, _, _ = compute_gcr(grad2d, grad2d_abs, count)
            override_colors = scalar_to_colormap(
                gcr,
                colormap=render_tab_state.colormap,
                inverse=render_tab_state.inverse,
                explicit_min=0.0,
                explicit_max=1.0,
            )
            
        elif render_tab_state.render_mode == "gdags_weight" and hasattr(epoch_stats, "grad2d") and hasattr(epoch_stats, "grad2d_abs"):
            # Visualize GDAGS weight: w = 0.8 + 25 * (1 - GCR)^15
            grad2d = epoch_stats.grad2d.clone()
            grad2d_abs = epoch_stats.grad2d_abs.clone()
            count = epoch_stats.count.clone()
            gcr, _, _ = compute_gcr(grad2d, grad2d_abs, count)
            # Compute GDAGS weight
            weight = 0.8 + 25 * torch.pow(1 - gcr, 15)
            override_colors = scalar_to_colormap(
                weight,
                colormap=render_tab_state.colormap,
                inverse=render_tab_state.inverse,
                explicit_min=0.8,
                explicit_max=25.8,
            )
            
        elif render_tab_state.render_mode == "grad2d_gcr_combined" and hasattr(epoch_stats, "grad2d") and hasattr(epoch_stats, "grad2d_abs"):
            # Combined visualization: grad2d_abs determines intensity, gcr determines hue
            grad2d = epoch_stats.grad2d.clone()
            grad2d_abs = epoch_stats.grad2d_abs.clone()
            count = epoch_stats.count.clone()
            gcr, _, avg_grad_abs = compute_gcr(grad2d, grad2d_abs, count)
            # Normalize grad2d_abs to [0, 1]
            grad_norm = torch.clamp(avg_grad_abs / (0.0002 * 2), 0.0, 1.0)  # Using default grow_grad2d value

            # Create color mapping:
            # Low grad_norm → pastel blue
            # High grad_norm + low gcr → pastel red
            # High grad_norm + high gcr → pastel green

            # Base pastel blue
            base_color = torch.tensor([0.3, 0.4, 1.0], device=device)
            # Target colors based on gcr
            red_color = torch.tensor([1.0, 0.3, 0.3], device=device)
            green_color = torch.tensor([0.3, 1.0, 0.3], device=device)

            # Interpolate between red and green based on gcr
            target_color = red_color * (1 - gcr).unsqueeze(-1) + green_color * gcr.unsqueeze(-1)

            # Interpolate between base and target based on grad_norm
            colors = base_color * (1 - grad_norm).unsqueeze(-1) + target_color * grad_norm.unsqueeze(-1)

            override_colors = colors
            
        elif render_tab_state.render_mode == "importance" and hasattr(epoch_stats, "importance"):
            # Visualize importance (vG^2) from accumulated gradients
            importance = epoch_stats.importance.clone()
            count = epoch_stats.count.clone()

            avg_importance = safe_divide(importance, count)
            #
            # Use logarithmic scale for better visualization
            log_importance = torch.log10(avg_importance + 1e-10)

            override_colors = scalar_to_colormap(
                avg_importance,
                colormap=render_tab_state.colormap,
                inverse=render_tab_state.inverse,
                # explicit_min=-6,  # 10^-6
                # explicit_max=-2,  # 10^-2
            )
        
        # Render with override colors if available
        if override_colors is not None:
            renders = (
                rasterize_fn(
                    camtoworlds=c2w[None],
                    Ks=K[None],
                    width=width,
                    height=height,
                    override_colors=override_colors,
                )[0].squeeze(0).clamp(0, 1).cpu().numpy()
            )
        else:
            # Fallback if no data available for requested mode
            sky_colors, _, _ = rasterize_fn(
                camtoworlds=c2w[None],
                Ks=K[None],
                width=width,
                height=height,
            )
            renders = sky_colors.squeeze(0).clamp(0, 1).cpu().numpy()
            
    else:
        # Non-epoch_stats modes
        track_domination = render_tab_state.render_mode == "domination"
        
        # Render sky
        sky_colors, sky_wsum, info = rasterize_fn(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            track_domination=track_domination,
        )
        
        if render_tab_state.render_mode == "domination" and "median_ids" in info:
            renders = (
                index_map_to_pseudocolor(info["median_ids"][0, ...])
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "domination" and "dominating_gauss_ids" in info:
            renders = (
                index_map_to_pseudocolor(info["dominating_gauss_ids"][0, ...])
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = sky_wsum[0, ..., 0:1]
            alpha_norm = normalize_robust(alpha)
            if render_tab_state.inverse:
                alpha_norm = 1 - alpha_norm
            renders = (
                apply_float_colormap(alpha_norm, render_tab_state.colormap).cpu().numpy()
            )
        else:
            # Default RGB mode
            # Compose with SH background if enabled
            if cfg.use_sh_background and sh_background is not None:
                sh_bg = sh_background.render(
                    camtoworlds=c2w[None],
                    Ks=K[None],
                    width=width,
                    height=height,
                )
                sky_colors = sh_background.blend_with_sky(sky_colors, sky_wsum, sh_bg)
            renders = sky_colors.squeeze(0).cpu().numpy()

    # Update render tab state
    render_tab_state.total_gs_count = n_points
    render_tab_state.rendered_gs_count = n_points

    return renders
