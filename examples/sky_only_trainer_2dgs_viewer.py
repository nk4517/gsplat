import torch
from nerfview import CameraState

from examples.utils import index_map_to_pseudocolor, scalar_to_colormap


def skysphere_renderer(device, rasterize_fn, epoch_stats, trainset_len, grow_grad2d, n_points, camera_state: CameraState, render_tab_state):

    width = render_tab_state.viewer_width
    height = render_tab_state.viewer_height
    c2w = camera_state.c2w
    K = camera_state.get_K((width, height))
    c2w = torch.from_numpy(c2w).float().to(device)
    K = torch.from_numpy(K).float().to(device)

    # Prepare override colors for colormapped visualization
    override_colors = None

    if render_tab_state.render_mode == "domination":
        # For domination mode, we need to track domination info
        track_domination = True
    else:
        track_domination = False

    # Render sky
    sky_colors, info = rasterize_fn(
        camtoworlds=c2w[None],
        Ks=K[None],
        width=width,
        height=height,
        track_domination=track_domination,
        override_colors=override_colors,
    )

    # Handle different render modes
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
    elif render_tab_state.render_mode == "grad2d_accum":
        # Visualize accumulated gradient magnitudes
        if epoch_stats is not None and hasattr(epoch_stats, "grad2d_abs"):
            grad2d = epoch_stats.grad2d_abs.clone()
            count = epoch_stats.count.clone()
            # Average gradient per visibility count
            avg_grad = torch.where(count > 0, grad2d / count.clamp_min(1), torch.zeros_like(grad2d))
            override_colors = scalar_to_colormap(
                avg_grad,
                colormap=render_tab_state.colormap,
                inverse=render_tab_state.inverse,
                explicit_min=0.0,
                explicit_max=grow_grad2d * 2,
            )
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
            # Fallback if no gradient data available
            renders = sky_colors.squeeze(0).clamp(0, 1).cpu().numpy()

    elif render_tab_state.render_mode == "grad2d_count":
        # Visualize visibility count (how many times each gaussian was visible)
        if epoch_stats is not None and hasattr(epoch_stats, "count"):
            count = epoch_stats.count.clone()
            override_colors = scalar_to_colormap(
                count,
                colormap=render_tab_state.colormap,
                inverse=render_tab_state.inverse,
                explicit_min=0,
                explicit_max=trainset_len,  # Max is number of training views
            )
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
            # Fallback if no count data available
            renders = sky_colors.squeeze(0).clamp(0, 1).cpu().numpy()

    elif render_tab_state.render_mode == "gcr":
        # Visualize Gradient Consistency Ratio (GCR) from GDAGS
        if (epoch_stats is not None and
            hasattr(epoch_stats, "grad2d") and
            hasattr(epoch_stats, "grad2d_abs")):
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
            )
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
            # Fallback if no gradient data available
            renders = sky_colors.squeeze(0).clamp(0, 1).cpu().numpy()

    elif render_tab_state.render_mode == "gdags_weight":
        # Visualize GDAGS weight: w = 0.8 + 25 * (1 - GCR)^15
        if (epoch_stats is not None and
            hasattr(epoch_stats, "grad2d") and
            hasattr(epoch_stats, "grad2d_abs")):
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
            )
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
            # Fallback if no gradient data available
            renders = sky_colors.squeeze(0).clamp(0, 1).cpu().numpy()

    elif render_tab_state.render_mode == "grad2d_gcr_combined":
        # Combined visualization: grad2d_abs determines intensity, gcr determines hue
        if (epoch_stats is not None and
            hasattr(epoch_stats, "grad2d") and
            hasattr(epoch_stats, "grad2d_abs")):
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
            # Fallback if no gradient data available
            renders = sky_colors.squeeze(0).clamp(0, 1).cpu().numpy()
    else:
        # Default RGB mode
        renders = sky_colors.squeeze(0).clamp(0, 1).cpu().numpy()

    # Update render tab state
    render_tab_state.total_gs_count = n_points
    render_tab_state.rendered_gs_count = n_points

    return renders
