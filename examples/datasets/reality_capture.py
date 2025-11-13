import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import lxml.etree
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from tqdm import tqdm

from .normalize_3d import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def parse_RC_xml(xml_string, image_width, image_height):
    """Parse Reality Capture XMP metadata."""
    description = lxml.etree.fromstring(xml_string)[0][0]
    namespace = description.nsmap
    
    rotation = description.find('xcr:Rotation', namespace).text
    rotation = np.array([float(x) for x in rotation.split()]).reshape((3,3))
    
    pos_elt = description.find('xcr:Position', namespace)
    if pos_elt is not None:
        position = pos_elt.text
        position = np.array([float(x) for x in position.split()])
        translation = -np.dot(rotation, position)
    else:
        position = rotation = translation = None
    
    # try:
    #     distortion = description.find('xcr:DistortionCoeficients', namespace).text
    #     # k1, k2, k3, unk, p1, p2 = [float(x) for x in distortion.split()]
    #     k1, k2, p1, p2, k3 = -2.03049154e-01, 3.79533953e-02, -2.47902737e-04, 9.57559238e-06, -2.90281511e-03
    #     # brownT2: Radial1, Radial2, Radial3, unknown 0, Tangential1, Tangential2
    #     d = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    # except AttributeError:
    d = np.zeros(5, dtype=np.float32)
    
    focal_length_35mm = float(str(description.xpath('@xcr:FocalLength35mm', namespaces=namespace)[0]))
    principal_point_u = float(str(description.xpath('@xcr:PrincipalPointU', namespaces=namespace)[0]))
    principal_point_v = float(str(description.xpath('@xcr:PrincipalPointV', namespaces=namespace)[0]))
    
    sensor_width_equivalent = 36  # mm, standard full-frame sensor
    fx_pixels = image_width * focal_length_35mm / sensor_width_equivalent
    
    cx_pixels = image_width * (0.5 + principal_point_u)
    cy_pixels = image_height * (0.5 + principal_point_v)
    
    K = np.array([
        [fx_pixels, 0, cx_pixels],
        [0, fx_pixels, cy_pixels],
        [0, 0, 1]
    ])
    
    return K, d, rotation, translation, position


def _process_mask(
    mask: np.ndarray,
    target_shape: tuple,
    params: np.ndarray,
    mapx: Optional[np.ndarray] = None,
    mapy: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Process mask: resize to target shape and apply undistortion if needed."""
    # Convert to uint8 if boolean
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    
    # Resize mask to match original image size if needed
    mask_h, mask_w = mask.shape[:2]
    target_h, target_w = target_shape
    
    if mask_h != target_h or mask_w != target_w:
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    
    # Apply undistortion if parameters are provided
    if len(params) > 0:
        assert mapx is not None and mapy is not None, "Undistortion maps required"
        
        mask = cv2.remap(mask, mapx, mapy, cv2.INTER_NEAREST)
    
    # Convert back to boolean
    return mask > 127


class Parser:
    """Reality Capture parser."""
    
    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        frame_limit: int = -1,  # -1 for no limit
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        
        path = Path(data_dir)
        
        # Collect camera data from XMP files
        w2c_mats = []
        image_paths = []
        image_names = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()
        mask_dict = dict()
        
        frame_id = -1
        camera_id = 0  # Single camera for now, can be extended
        
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

        for fname_xmp in sorted(path.glob("*.xmp")):
            fname_img = fname_xmp.with_suffix(".png")
            if not fname_img.is_file():
                fname_img = fname_xmp.with_suffix(".jpg")
                if not fname_img.is_file():
                    print(f"Image not found for {fname_xmp.stem}")
                    continue

            # Optional frame limit (e.g., for waymo dataset)
            if frame_limit > 0:
                try:
                    waymo_frame_id = int(fname_img.stem.lstrip("0") or "0")
                    if waymo_frame_id > frame_limit:
                        break
                except ValueError:
                    pass

            # Read image to get dimensions
            img = cv2.imread(str(fname_img), cv2.IMREAD_UNCHANGED)
            h, w = img.shape[:2]

            # Parse XMP metadata
            K, d, R0, t0, C = parse_RC_xml(fname_xmp.read_text("utf-8"), w, h)
            if R0 is None:
                print(f"Unknown pose: {fname_xmp.stem}")
                continue

            frame_id += 1

            # Build world-to-camera transform
            W2C_xform = np.eye(4)
            W2C_xform[:3, :3] = R0
            W2C_xform[:3, 3] = t0
            w2c_mats.append(W2C_xform)

            # Store camera parameters
            K_scaled = K.copy()
            K_scaled[:2, :] /= factor
            Ks_dict[camera_id] = K_scaled
            params_dict[camera_id] = d
            imsize_dict[camera_id] = (w // factor, h // factor)
            mask_dict[camera_id] = None

            # Store image info
            image_paths.append(str(fname_img))
            image_names.append(fname_img.relative_to(path).as_posix())
            camera_ids.append(camera_id)
        
        print(f"[Parser] {len(image_paths)} images loaded from Reality Capture export.")
        
        if len(image_paths) == 0:
            raise ValueError("No images found in Reality Capture export.")
        
        w2c_mats = np.stack(w2c_mats, axis=0)
        
        # Convert extrinsics to camera-to-world
        camtoworlds = np.linalg.inv(w2c_mats)
        
        # Load point cloud
        ply_path = path / "dump.ply"
        if ply_path.exists():
            plydata = PlyData.read(ply_path)
            vertices = plydata['vertex']
            points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T.astype(np.float32)
            points_rgb = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T.astype(np.uint8)
            points_err = np.zeros(len(points), dtype=np.float32)  # No error info in PLY
        else:
            print(f"Warning: Point cloud file {ply_path} not found. Using empty point cloud.")
            points = np.zeros((0, 3), dtype=np.float32)
            points_rgb = np.zeros((0, 3), dtype=np.uint8)
            points_err = np.zeros(0, dtype=np.float32)
        
        # Create empty point indices dict (no point-to-image mapping in Reality Capture export)
        point_indices = {name: np.array([], dtype=np.int32) for name in image_names}
        
        # Normalize the world space
        if normalize and len(camtoworlds) > 0:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            if len(points) > 0:
                points = transform_points(T1, points)
            
            if len(points) > 0:
                T2 = align_principal_axes(points)
                camtoworlds = transform_cameras(T2, camtoworlds)
                points = transform_points(T2, points)
            else:
                T2 = np.eye(4)
            
            transform = T2 @ T1
            
            # Fix for upside down
            if len(points) > 0 and np.median(points[:, 2]) > np.mean(points[:, 2]):
                T3 = np.array([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ])
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                transform = T3 @ transform
        else:
            transform = np.eye(4)
        
        self.image_names = image_names
        self.image_paths = image_paths
        self.camtoworlds = camtoworlds
        self.camera_ids = camera_ids
        self.Ks_dict = Ks_dict
        self.params_dict = params_dict
        self.imsize_dict = imsize_dict
        self.mask_dict = mask_dict
        self.points = points
        self.points_err = points_err
        self.points_rgb = points_rgb
        self.point_indices = point_indices
        self.transform = transform
        
        # Setup undistortion maps
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            
            mapx, mapy = cv2.initUndistortRectifyMap(
                K, params, None, K, (width, height), cv2.CV_32FC1
            )
            
            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            # Keep original K and image size
            # Set ROI to full image size (no cropping)
            self.roi_undist_dict[camera_id] = [0, 0, width, height]
            
        # Check if we need to save undistorted images
        has_distortion = any(len(params) > 0 for params in self.params_dict.values())
        if has_distortion:
            undist_dir = os.path.join(data_dir, 'images_undist')
            if not os.path.exists(undist_dir):
                print(f"Creating undistorted images in {undist_dir}")
                os.makedirs(undist_dir, exist_ok=True)
                
                for idx, image_path in enumerate(tqdm(
                    self.image_paths,
                    desc="Saving undistorted images"
                )):
                    camera_id = self.camera_ids[idx]
                    params = self.params_dict[camera_id]
                    
                    image = imageio.imread(image_path)[..., :3]
                    
                    if len(params) > 0:
                        mapx = self.mapx_dict[camera_id]
                        mapy = self.mapy_dict[camera_id]
                        image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
                    
                    # Save with same relative path structure
                    rel_path = Path(self.image_names[idx])
                    undist_path = Path(undist_dir) / rel_path.with_suffix('.png')
                    undist_path.parent.mkdir(parents=True, exist_ok=True)
                    imageio.imwrite(str(undist_path), image)
                
                print(f"Saved {len(self.image_paths)} undistorted images to {undist_dir}")
            else:
                print(f"Undistorted images already exist in {undist_dir}")
        
        # Scene scale
        if len(camtoworlds) > 0:
            camera_locations = camtoworlds[:, :3, 3]
            scene_center = np.mean(camera_locations, axis=0)
            dists = np.linalg.norm(camera_locations - scene_center, axis=1)
            self.scene_scale = np.max(dists) if len(dists) > 0 else 1.0
        else:
            self.scene_scale = 1.0


class Dataset:
    """A simple dataset class for Reality Capture data."""
    
    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]
        
        # Store original image shape before undistortion
        original_shape = image.shape[:2]
        
        # Process mask if available
        if mask is not None:
            mapx = self.parser.mapx_dict.get(camera_id)
            mapy = self.parser.mapy_dict.get(camera_id)
            
            mask = _process_mask(
                mask,
                original_shape,
                params,
                mapx,
                mapy
            )
        
        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        
        # Load sky mask if available
        sky_mask = None
        skymask_dir = os.path.join(self.parser.data_dir, "sky")
        if not os.path.exists(skymask_dir):
            skymask_dir = os.path.join(self.parser.data_dir, "masks")
        
        if os.path.exists(skymask_dir):
            image_name = self.parser.image_names[index]
            base_name = os.path.splitext(os.path.basename(image_name))[0]
            
            # Try loading .npy or .png
            npy_path = os.path.join(skymask_dir, f"{base_name}.npy")
            png_path = os.path.join(skymask_dir, f"{base_name}.png")
            
            if os.path.exists(npy_path):
                sky_mask = np.load(npy_path)
            elif os.path.exists(png_path):
                sky_mask = imageio.imread(png_path)
                if len(sky_mask.shape) > 2:
                    sky_mask = sky_mask[..., 0]
            
            # Process sky mask
            if sky_mask is not None:
                mapx = self.parser.mapx_dict.get(camera_id)
                mapy = self.parser.mapy_dict.get(camera_id)
                
                sky_mask = _process_mask(
                    sky_mask,
                    original_shape,
                    params,
                    mapx,
                    mapy
                )
                
                sky_mask = ~sky_mask  # Invert for consistency with colmap loader
        
        if self.patch_size is not None:
            # Random crop
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y
            
            # Apply same crop to masks
            if mask is not None:
                mask = mask[y : y + self.patch_size, x : x + self.patch_size]
            
            if sky_mask is not None:
                sky_mask = sky_mask[y : y + self.patch_size, x : x + self.patch_size]
        
        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,
        }
        
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()
        
        if sky_mask is not None:
            data["sky_mask"] = torch.from_numpy(sky_mask).bool()
        
        if self.load_depths:
            # Project points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices.get(image_name, np.array([], dtype=np.int32))
            
            if len(point_indices) > 0:
                points_world = self.parser.points[point_indices]
                points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
                points_proj = (K @ points_cam.T).T
                points = points_proj[:, :2] / points_proj[:, 2:3]
                depths = points_cam[:, 2]
                
                # Filter out points outside the image
                selector = (
                    (points[:, 0] >= 0)
                    & (points[:, 0] < image.shape[1])
                    & (points[:, 1] >= 0)
                    & (points[:, 1] < image.shape[0])
                    & (depths > 0)
                )
                points = points[selector]
                depths = depths[selector]
            else:
                points = np.array([], dtype=np.float32).reshape(0, 2)
                depths = np.array([], dtype=np.float32)
            
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()
        
        return data