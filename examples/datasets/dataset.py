from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Literal

import cv2
import numpy as np
import torch
from imageio import v2 as imageio

@dataclass
class UndistortionMaps:
    """Precomputed undistortion maps and ROI."""
    dist_coefs: np.ndarray  # original distortion coefficients
    camtype: Literal["perspective", "fisheye"]
    mapx: np.ndarray  # (H, W) remap x coordinates
    mapy: np.ndarray  # (H, W) remap y coordinates
    roi: tuple[int, int, int, int]  # x, y, w, h after undistortion
    valid_mask: np.ndarray | None = None  # fisheye valid region mask


@dataclass
class CameraIntrinsics:
    """Интринсики камеры (без привязки к конкретному изображению)."""
    camera_id: int
    K: np.ndarray  # (3, 3)
    width: int
    height: int
    undistortion: UndistortionMaps | None = None


@dataclass
class ImagePose:
    """Поза и пути для одного изображения."""
    name: str  # произвольное уникальное имя
    camera_id: int
    camtoworld: np.ndarray  # (4, 4)
    image_fpath: Path # абсолютный путь до файла
    aux_fpaths: dict[str, Path] = field(default_factory=dict)

@dataclass
class PointCloud:
    """3D points из COLMAP SfM."""
    xyz: np.ndarray  # (N, 3) float32
    rgb: np.ndarray  # (N, 3) uint8
    errors: np.ndarray  # (N,) float32
    # image_name -> indices of points visible in that image
    visibility: dict[str, np.ndarray]

@dataclass
class ImagePrepareEntry:
    """Данные для подготовки одного изображения."""
    rel_path: Path  # относительный путь от images_basedir (с расширением)
    camera_id: int
    target_size: tuple[int, int] | None = None  # (w, h) после resize
    undistort_maps: tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]] | None = None  # (mapx, mapy, roi)
    aux: dict[str, Path] = field(default_factory=dict)  # subdir -> rel_path к вспомогательному файлу (mask, depth, etc.)

@dataclass
class PrepareContext:
    """Контекст для подготовки изображений сцены."""
    dataset_dir: Path
    destination_dir: Path
    transform_suffix: str # _1, если без ресайза/андисторта
    images: list[ImagePrepareEntry]


@dataclass
class Scene:
    """Полные данные сцены."""
    cameras: dict[int, CameraIntrinsics]
    images: list[ImagePose]
    points: PointCloud | None = None
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    scene_scale: float = 1.0
    bounds: np.ndarray = field(default_factory=lambda: np.array([0.01, 1.0]))
    # Extended config (from ext_metadata.json)
    extconf: dict = field(default_factory=dict)
    dataset_dir: Path | None = None
    output_dir: Path | None = None  # для сгенерированного контента (undistort, resize и т.п.)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        scene: Scene,
        test_every: int = 8,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        invert_sky_mask: bool = False,
        soft_sky_mask: bool = False,
        load_sky_mask: bool = False,
        require_sky_mask: bool = False,
    ):
        self.scene = scene
        self.test_every = test_every
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.load_sky_mask = load_sky_mask or require_sky_mask
        self.soft_sky_mask = soft_sky_mask
        self.invert_sky_mask = invert_sky_mask
        self.require_sky_mask = require_sky_mask
        indices = np.arange(len(self.scene.images))
        if split == "train":
            self.indices = indices[indices % self.test_every != 0]
        else:
            self.indices = indices[indices % self.test_every == 0]

        # Filter out images without sky masks if required
        if self.require_sky_mask:
            valid_indices = []
            for idx in self.indices:
                img_data = self.scene.images[idx]
                mask_fpath = img_data.aux_fpaths.get("mask")
                if mask_fpath and mask_fpath.exists():
                    valid_indices.append(idx)
            self.indices = np.array(valid_indices, dtype=np.int64) if valid_indices else np.array([], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        img_data = self.scene.images[index]
        try:
            image = imageio.imread(img_data.image_fpath)[..., :3]
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_data.image_fpath}: {e}") from e

        cam = self.scene.cameras[img_data.camera_id]
        K = cam.K.copy()

        # Load mask if available (already prepared with same undistort/resize as images)
        sky_mask = None
        mask_fpath = img_data.aux_fpaths.get("mask")
        if self.load_sky_mask and mask_fpath and mask_fpath.exists():
            sky_mask = imageio.imread(mask_fpath)
            if len(sky_mask.shape) == 3:
                sky_mask = sky_mask[..., 0]
            # uint8 0-255 -> float 0.0-1.0
            sky_mask = sky_mask.astype(np.float32) / 255.0
            if self.invert_sky_mask:
                sky_mask = 1.0 - sky_mask
            if not self.soft_sky_mask:
                sky_mask = sky_mask > 0.5

        # Fisheye mask (valid region after undistortion)
        mask = cam.undistortion.valid_mask if cam.undistortion is not None else None
        if mask is not None:
            # Resize to match image if needed
            mask_h, mask_w = mask.shape[:2]
            img_h, img_w = image.shape[:2]
            if mask_h != img_h or mask_w != img_w:
                mask = cv2.resize(
                    mask.astype(np.uint8) * 255, (img_w, img_h), interpolation=cv2.INTER_NEAREST
                ) > 127

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

            # Apply same crop to masks
            if mask is not None:
                mask = mask[y : y + self.patch_size, x : x + self.patch_size]

            # Apply same crop to sky mask
            if sky_mask is not None:
                sky_mask = sky_mask[y : y + self.patch_size, x : x + self.patch_size]

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(img_data.camtoworld).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        if sky_mask is not None:
            # sky_mask уже float32 или bool после конвертации выше
            data["sky_mask"] = torch.from_numpy(sky_mask)

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocam = np.linalg.inv(img_data.camtoworld)
            point_indices = self.scene.points.visibility[img_data.name]
            points_world = self.scene.points.xyz[point_indices]
            points_cam = (worldtocam[:3, :3] @ points_world.T + worldtocam[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data
