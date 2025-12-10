"""Waymo dataset parser for gsplat-2025."""

from pathlib import Path

import numpy as np
from plyfile import PlyData
from tqdm import tqdm
from .waymo_utils import WaymoCoordsHelper

from .dataset import (
    CameraIntrinsics,
    ImagePose,
    PointCloud,
    Scene,
)


WAYMO_CAMERAS = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]


class WaymoParser:
    """Waymo dataset parser."""

    def __init__(
        self,
        data_dir: str,
        camera_angles: list[str] | None = None,
        frame_range: tuple[int, int] = (0, 50),
        test_every: int = 8,
        load_lidar: bool = True,
        output_dir: str | Path | None = None,
        waymo_calib_dir: str | None = None,
    ):
        """
        Args:
            data_dir: Path to Waymo export directory (contains images/, lidar/, skymask/)
            camera_angles: List of cameras to load. Default: ["FRONT"]
            frame_range: (start, end) frame indices to load
            test_every: Every N-th frame goes to test set
            normalize: Normalize world space
            load_lidar: Load lidar point cloud
            waymo_calib_dir: Path to calibration data (default: same as data_dir)
        """

        data_dir = Path(data_dir).absolute()
        if camera_angles is None:
            camera_angles = ["FRONT"]

        calib_dir = Path(waymo_calib_dir) if waymo_calib_dir else data_dir
        waymo_conv = WaymoCoordsHelper(calib_dir)

        camtoworlds = []
        camera_ids_list = []
        image_names = []
        image_paths = []
        # mask_paths = []
        camera_infos: dict[int, CameraIntrinsics] = {}

        frame_start, frame_end = frame_range

        for frame_id in tqdm(range(frame_start, frame_end), desc="Loading frames"):
            for cam_idx, waymo_cam in enumerate(camera_angles):
                img_path = data_dir / "images" / waymo_cam / f"{frame_id:04d}.png"
                if not img_path.is_file():
                    continue

                w, h, K, d, cam_pose = waymo_conv.get_calib_as_ocv_rel_fr0(frame_id, waymo_cam)

                # cam_pose is camera-to-world
                camtoworlds.append(cam_pose)

                # Camera ID based on camera angle
                camera_id = WAYMO_CAMERAS.index(waymo_cam)
                camera_ids_list.append(camera_id)

                if camera_id not in camera_infos:
                    camera_infos[camera_id] = CameraIntrinsics(
                        camera_id=camera_id,
                        K=K.astype(np.float64),
                        width=w,
                        height=h,
                        undistortion=None,  # Waymo images are already undistorted
                    )

                image_names.append(f"{waymo_cam}/{frame_id:04d}")
                image_paths.append(img_path)

                # # Sky mask
                # mask_path = data_dir / "skymask" / waymo_cam / f"{frame_id:04d}.npy"
                # if not mask_path.exists():
                #     mask_path = data_dir / "masks" / waymo_cam / f"{frame_id:04d}.png"
                # mask_paths.append(mask_path if mask_path.exists() else None)

        if not camtoworlds:
            raise ValueError(f"No images found in {data_dir}")

        print(f"[Parser] {len(camtoworlds)} images from {len(camera_infos)} cameras")

        camtoworlds = np.stack(camtoworlds, axis=0)

        # Load lidar point cloud
        points = np.zeros((0, 3), dtype=np.float32)
        points_rgb = np.zeros((0, 3), dtype=np.uint8)
        points_err = np.zeros(0, dtype=np.float32)

        if load_lidar:
            lidar_dir = data_dir / "lidar"
            if lidar_dir.exists():
                all_pts = []
                for frame_id in range(frame_start, frame_end):
                    ply_path = lidar_dir / f"{frame_id:04d}.ply"
                    if not ply_path.exists():
                        continue
                    plydata = PlyData.read(ply_path)
                    vertices = plydata['vertex']
                    pts = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T.astype(np.float32)
                    # Transform to frame 0 and OpenCV coords
                    pts = waymo_conv.points_waymo2ocv(waymo_conv.points_frame2fr0(frame_id, pts))
                    all_pts.append(pts)

                if all_pts:
                    points = np.concatenate(all_pts, axis=0)[::100, ...]
                    points_rgb = np.full((len(points), 3), 128, dtype=np.uint8)
                    points_err = np.zeros(len(points), dtype=np.float32)
                    print(f"[Parser] Loaded {len(points)} lidar points")

        transform = np.eye(4)

        # Build Scene
        scene_images = [
            ImagePose(
                name=image_names[i],
                camera_id=camera_ids_list[i],
                camtoworld=camtoworlds[i],
                image_fpath=image_paths[i],
            )
            for i in range(len(image_names))
        ]

        scene_points = PointCloud(
            xyz=points,
            rgb=points_rgb,
            errors=points_err,
            visibility={},  # No per-image visibility for lidar
        )

        # Scene scale
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        scene_scale = np.max(dists) if len(dists) > 0 else 1.0

        self.scene = Scene(
            cameras=camera_infos,
            images=scene_images,
            points=scene_points,
            transform=transform,
            scene_scale=scene_scale,
            bounds=np.array([0.01, 1.0]),
            extconf={},
            dataset_dir=data_dir,
            output_dir=output_dir,
        )
        self.test_every = test_every
