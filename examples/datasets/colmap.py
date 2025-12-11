import json
from pathlib import Path
from typing import Literal

import cv2
import imageio.v2 as imageio
import numpy as np
import pycolmap
from tqdm import tqdm
from typing_extensions import assert_never
from .dataset import (
    UndistortionMaps,
    CameraIntrinsics,
    ImagePose,
    PointCloud,
    Scene,
)


class Parser:
    """COLMAP parser. Returns Scene with full resolution and distortion maps."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str| Path | None = None,
    ):
        colmap_dir = Path(data_dir) / "sparse/0"
        if not colmap_dir.exists():
            colmap_dir = Path(data_dir) / "sparse"
            if not colmap_dir.exists() and (Path(data_dir) / "images.bin").exists():
                colmap_dir = Path(data_dir)
        assert colmap_dir.exists(), f"COLMAP directory {colmap_dir} does not exist."

        reconstruction = pycolmap.Reconstruction(colmap_dir)

        # Extract extrinsic matrices in world-to-camera format.
        imdata = reconstruction.images
        w2c_mats = []
        camera_ids = []
        camera_infos: dict[int, CameraIntrinsics] = {}
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for image_id, im in imdata.items():
            cfw = im.cam_from_world()
            rot = cfw.rotation.matrix()
            trans = np.array(cfw.translation).reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            if camera_id in camera_infos:
                continue

            # camera intrinsics
            cam = reconstruction.cameras[camera_id]
            params = cam.params
            model_name = cam.model.name if hasattr(cam.model, 'name') else str(cam.model)
            
            # Extract fx, fy, cx, cy based on camera model
            if model_name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:  # PINHOLE, OPENCV, OPENCV_FISHEYE, FULL_OPENCV
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]
            
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            # Get distortion parameters.
            if model_name == "SIMPLE_PINHOLE":
                dist_coefs = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif model_name == "PINHOLE":
                dist_coefs = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif model_name == "SIMPLE_RADIAL":
                dist_coefs = np.array([params[3], 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif model_name == "RADIAL":
                dist_coefs = np.array([params[3], params[4], 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif model_name == "OPENCV":
                dist_coefs = np.array([params[4], params[5], params[6], params[7]], dtype=np.float32)
                camtype = "perspective"
            elif model_name == "OPENCV_FISHEYE":
                dist_coefs = np.array([params[4], params[5], params[6], params[7]], dtype=np.float32)
                camtype = "fisheye"
            elif model_name == "FULL_OPENCV":
                dist_coefs = np.array([params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11]], dtype=np.float32)
                camtype = "perspective"
            else:
                raise ValueError(f"Unsupported camera model: {model_name}")
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {model_name}"

            # Create UndistortionMaps if there's distortion
            undist = None
            if len(dist_coefs) > 0:
                undist = UndistortionMaps(
                    dist_coefs=dist_coefs,
                    camtype=camtype,
                    mapx=None,  # Will be filled later
                    mapy=None,  # Will be filled later
                    roi=None,   # Will be filled later
                    valid_mask=None,  # Will be filled later for fisheye
                )

            camera_infos[camera_id] = CameraIntrinsics(
                camera_id=camera_id,
                K=K,
                width=cam.width,
                height=cam.height,
                undistortion=undist,
            )

        print(f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras.")

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        has_distortion = any(c.undistortion is not None for c in camera_infos.values())
        if has_distortion:
            print("Warning: COLMAP cameras have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [im.name for im in imdata.values()]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = Path(data_dir) / "ext_metadata.json"
        if extconf_file.exists():
            with open(extconf_file) as f:
                extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        bounds = np.array([0.01, 1.0])
        posefile = Path(data_dir) / "poses_bounds.npy"
        if posefile.exists():
            bounds = np.load(posefile)[:, -2:]

        # Load images.
        colmap_image_dir = Path(data_dir) / "images"
        
        # Check if colmap_image_dir exists
        if not colmap_image_dir.exists():
            raise ValueError(f"COLMAP image folder {colmap_image_dir} does not exist.")
        
        # 3D points and {image_name -> [point_idx]}
        points3D = reconstruction.points3D
        n_points = len(points3D)
        points = np.empty((n_points, 3), dtype=np.float32)
        points_err = np.empty(n_points, dtype=np.float32)
        points_rgb = np.empty((n_points, 3), dtype=np.uint8)
        point3D_id_to_idx = {}

        for idx, (pid, p) in enumerate(points3D.items()):
            point3D_id_to_idx[pid] = idx
            points[idx] = p.xyz
            points_err[idx] = p.error
            points_rgb[idx] = p.color
        
        # Build image_id -> image_name mapping
        image_id_to_name = {img_id: img.name for img_id, img in reconstruction.images.items()}
        
        point_indices = dict()
        for point_id, point in points3D.items():
            for track_elem in point.track.elements:
                image_id = track_elem.image_id
                image_name = image_id_to_name[image_id]
                point_idx = point3D_id_to_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        transform = np.eye(4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        first_image_path = colmap_image_dir / image_names[0]
        actual_image = imageio.imread(first_image_path)[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        first_cam = camera_infos[camera_ids[0]]
        colmap_width, colmap_height = first_cam.width, first_cam.height
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, cam in camera_infos.items():
            K = cam.K
            K[0, :] *= s_width
            K[1, :] *= s_height
            cam.width = int(round(cam.width * s_width))
            cam.height = int(round(cam.height * s_height))

        # undistortion
        for camera_id, cam in camera_infos.items():
            if cam.undistortion is None:
                continue  # no distortion
            
            K = cam.K
            width, height = cam.width, cam.height
            dist_coefs = cam.undistortion.dist_coefs
            camtype = cam.undistortion.camtype

            undistort_optimal = True
            if camtype == "perspective":
                if undistort_optimal:
                    K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                        K, dist_coefs, (width, height), 0
                    )
                    mapx, mapy = cv2.initUndistortRectifyMap(
                        K, dist_coefs, None, K_undist, (width, height), cv2.CV_32FC1
                    )
                else:
                    # Use original K matrix instead of computing optimal new matrix
                    K_undist = K.copy()
                    roi_undist = [0, 0, width, height]
                    mapx, mapy = cv2.initUndistortRectifyMap(
                        K, dist_coefs, None, K, (width, height), cv2.CV_32FC1
                    )

                undistort_roi_mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + dist_coefs[0] * theta**2
                    + dist_coefs[1] * theta**4
                    + dist_coefs[2] * theta**6
                    + dist_coefs[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                undistort_roi_mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(undistort_roi_mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                undistort_roi_mask = undistort_roi_mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            cam.undistortion.mapx = mapx
            cam.undistortion.mapy = mapy
            cam.undistortion.roi = roi_undist
            cam.undistortion.valid_mask = undistort_roi_mask
            cam.K = K_undist
            cam.width = roi_undist[2]
            cam.height = roi_undist[3]

        # Build image paths (original, without resize/undistort)
        image_mapping = {name: colmap_image_dir / name for name in image_names}

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        scene_scale = np.max(dists)

        # Build Scene dataclass
        scene_cameras: dict[int, CameraIntrinsics] = {}
        for cam_id, cam in camera_infos.items():
            undist = None
            if cam.undistortion is not None:
                undist = UndistortionMaps(
                    dist_coefs=cam.undistortion.dist_coefs,
                    camtype=cam.undistortion.camtype,
                    mapx=cam.undistortion.mapx,
                    mapy=cam.undistortion.mapy,
                    roi=cam.undistortion.roi,
                    valid_mask=cam.undistortion.valid_mask,
                )
            scene_cameras[cam_id] = CameraIntrinsics(
                camera_id=cam_id,
                K=cam.K,
                width=cam.width,
                height=cam.height,
                undistortion=undist,
            )

        scene_images = [
            ImagePose(
                name=image_names[i],
                camera_id=camera_ids[i],
                camtoworld=camtoworlds[i],
                image_fpath=image_mapping[image_names[i]],
            )
            for i in range(len(image_names))
        ]

        scene_points = PointCloud(
            xyz=points,
            rgb=points_rgb,
            errors=points_err,
            visibility=point_indices,
        )

        self.scene = Scene(
            cameras=scene_cameras,
            images=scene_images,
            points=scene_points,
            transform=transform,
            scene_scale=scene_scale,
            bounds=bounds,
            extconf=extconf,
            dataset_dir=Path(data_dir),
            output_dir=output_dir,
        )


if __name__ == "__main__":
    import argparse

    from examples.datasets.dataset import Dataset
    from examples.datasets.scene_prepare import prepare_scene

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    colmap_parser = Parser(
        data_dir=args.data_dir, normalize=True, test_every=8
    )
    scene = prepare_scene(
        colmap_parser.scene, args.data_dir, factor=args.factor
    )
    dataset = Dataset(scene, test_every=colmap_parser.test_every, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
