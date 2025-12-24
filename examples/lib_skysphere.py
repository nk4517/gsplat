import math

import torch
import tqdm


def fibonacci_sphere(samples: int = 1000, radius: float = 300):
    """Generate points on sphere using fibonacci spiral."""

    indices = torch.arange(0, samples, dtype=torch.float32) + 0.5
    phi = torch.tensor(math.pi * (3. - math.sqrt(5.)), dtype=torch.float32)
    y = 1 - (indices / (samples - 1)) * 2
    radius_sqrt = torch.sqrt(1 - y * y)
    theta = phi * indices
    x = torch.cos(theta) * radius_sqrt
    z = torch.sin(theta) * radius_sqrt
    points = torch.stack((radius * x, radius * y, radius * z), dim=-1)
    good_mask = points.isfinite().all(dim=-1)
    return points[good_mask, :]


def reproject_skysphere(trainset, skysphere_radius, samples, device):
    # Generate fibonacci sphere points
    skysphere_pts3d = fibonacci_sphere(
        samples=samples,
        radius=skysphere_radius
    ).float().to(device)
    populated = torch.zeros(skysphere_pts3d.shape[0], dtype=torch.bool, device=device)
    # Collect points and colors from all training cameras
    all_points = []
    all_colors = []

    for i in tqdm.trange(len(trainset), desc="Initializing skysphere"):
        data = trainset[i]
        camtoworld = data["camtoworld"].to(device).unsqueeze(0)
        K = data["K"].to(device).unsqueeze(0)
        image = data["image"].to(device) / 255.0
        height, width = image.shape[:2]

        # Check if sky mask is available
        sky_mask = data.get("sky_mask", None)

        # Project skysphere points to camera
        worldtocam = torch.linalg.inv(camtoworld[0])
        points_cam = (worldtocam[:3, :3] @ skysphere_pts3d.T + worldtocam[:3, 3:4]).T
        points_proj = (K[0] @ points_cam.T).T
        points_2d = points_proj[:, :2] / points_proj[:, 2:3]

        # Filter points inside image bounds and in front of camera
        in_view = (
                (points_2d[:, 0] >= 0) &
                (points_2d[:, 0] < width) &
                (points_2d[:, 1] >= 0) &
                (points_2d[:, 1] < height) &
                (points_cam[:, 2] > 0)
        )

        visible_unpopulated = in_view & ~populated
        if not visible_unpopulated.any():
            continue

        # Sample colors from image
        visible_unpopulated_xy = points_2d[visible_unpopulated].long()
        un_y = visible_unpopulated_xy[:, 1].clamp(0, height - 1)
        un_x = visible_unpopulated_xy[:, 0].clamp(0, width - 1)

        # Filter by sky mask if available
        if sky_mask is not None:
            sky_mask = sky_mask.to(device)
            # Check which points are in sky regions
            is_sky = sky_mask[un_y, un_x]
            # Update visible_unpopulated to only include sky points
            visible_unpopulated_indices = torch.where(visible_unpopulated)[0]
            visible_unpopulated_sky = visible_unpopulated_indices[is_sky]
            visible_unpopulated = torch.zeros_like(visible_unpopulated)
            visible_unpopulated[visible_unpopulated_sky] = True

            if not visible_unpopulated.any():
                continue

            # Re-sample coordinates for filtered points
            visible_unpopulated_xy = points_2d[visible_unpopulated].long()
            un_y = visible_unpopulated_xy[:, 1].clamp(0, height - 1)
            un_x = visible_unpopulated_xy[:, 0].clamp(0, width - 1)

        pts3d_to_add = skysphere_pts3d[visible_unpopulated]
        colors_to_add = image[un_y, un_x]

        all_points.append(pts3d_to_add)
        all_colors.append(colors_to_add)
        populated[visible_unpopulated] = True

    return all_colors, all_points
