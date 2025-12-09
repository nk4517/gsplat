from math import ceil
from collections import defaultdict
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from typing import Dict, Any, Optional, TYPE_CHECKING
from tqdm import tqdm
from pathlib import Path

if TYPE_CHECKING:
    from .dataset import Scene, CameraIntrinsics, ImagePose


class PreloadedDataset:
    """Dataset with all images preloaded into memory for fast access.

    Can store data either in pinned CPU memory or directly on GPU.
    """

    def __init__(
            self,
            scene: "Scene",
            test_every: int = 8,
            split: str = "train",
            patch_size: Optional[int] = None,
            load_depths: bool = False,
            device: str = "cuda",
            pin_memory: bool = True,
            to_gpu: bool = False,  # If True, store directly on GPU instead of pinned memory
            require_sky_mask: bool = False,  # If True, exclude images without sky masks from dataset
            load_sky_mask: bool = False,  # If True, load sky masks into memory
            invert_sky_mask: bool = False,
            soft_sky_mask: bool = False,
    ):
        self.scene = scene
        self.test_every = test_every
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.device = device
        self.pin_memory = pin_memory and not to_gpu  # Don't pin if storing on GPU
        self.to_gpu = to_gpu
        self.require_sky_mask = require_sky_mask
        self.load_sky_mask = load_sky_mask or require_sky_mask  # require implies load
        self.invert_sky_mask = invert_sky_mask
        self.soft_sky_mask = soft_sky_mask

        # Determine indices based on split
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

        # Map from dataset index to original parser index
        self.index_mapping = {i: idx for i, idx in enumerate(self.indices)}

        # Preload all data
        print(f"Preloading {len(self.indices)} {split} images into {'GPU' if to_gpu else ('pinned' if self.pin_memory else 'regular')} memory...")
        self._preload_data()

    def _preload_data(self):
        """Preload all images and associated data into memory."""
        self.preloaded_images = []
        self.preloaded_camtoworlds = []
        self.preloaded_Ks = []
        self.preloaded_masks = []
        self.preloaded_camera_ids = []

        if self.load_sky_mask:
            self.preloaded_sky_masks = []

        if self.load_depths:
            self.preloaded_points = []
            self.preloaded_depths = []

        for idx in tqdm(self.indices, desc=f"Loading {self.split} images"):
            img_data = self.scene.images[idx]
            # Load image
            try:
                image = imageio.imread(img_data.image_fpath)[..., :3]
            except Exception as e:
                print(f"Warning: failed to load {img_data.image_fpath}: {e}, skipping")
                continue

            cam_info = self.scene.cameras[img_data.camera_id]
            K = cam_info.K.copy()
            camtoworld = img_data.camtoworld

            # Load sky mask if required and available
            sky_mask = None
            mask_fpath = img_data.aux_fpaths.get("mask")
            if self.load_sky_mask and mask_fpath and mask_fpath.exists():
                try:
                    sky_mask = imageio.imread(mask_fpath)
                    if len(sky_mask.shape) == 3:
                        sky_mask = sky_mask[..., 0]
                    # uint8 0-255 -> float 0.0-1.0
                    sky_mask = sky_mask.astype(np.float32) / 255.0
                    if self.invert_sky_mask:
                        sky_mask = 1.0 - sky_mask
                    if not self.soft_sky_mask:
                        sky_mask = sky_mask > 0.5
                except Exception as e:
                    print(f"Warning: failed to load sky mask {mask_fpath}: {e}")
                    sky_mask = None

            # Fisheye mask (valid region after undistortion)
            mask = cam_info.undistortion.valid_mask if cam_info.undistortion is not None else None
            if mask is not None:
                mask_h, mask_w = mask.shape[:2]
                img_h, img_w = image.shape[:2]
                if mask_h != img_h or mask_w != img_w:
                    mask = cv2.resize(
                        mask.astype(np.uint8) * 255, (img_w, img_h), interpolation=cv2.INTER_NEAREST
                    ) > 127

            # Convert to tensors
            image_tensor = torch.from_numpy(image).float()
            K_tensor = torch.from_numpy(K).float()
            camtoworld_tensor = torch.from_numpy(camtoworld).float()

            # Store on GPU or pin memory
            if self.to_gpu:
                image_tensor = image_tensor.to(self.device)
                K_tensor = K_tensor.to(self.device)
                camtoworld_tensor = camtoworld_tensor.to(self.device)
            elif self.pin_memory:
                image_tensor = image_tensor.pin_memory()
                K_tensor = K_tensor.pin_memory()
                camtoworld_tensor = camtoworld_tensor.pin_memory()

            self.preloaded_images.append(image_tensor)
            self.preloaded_Ks.append(K_tensor)
            self.preloaded_camtoworlds.append(camtoworld_tensor)
            self.preloaded_camera_ids.append(img_data.camera_id)

            if mask is not None:
                mask_tensor = torch.from_numpy(mask).bool()
                if self.to_gpu:
                    mask_tensor = mask_tensor.to(self.device)
                elif self.pin_memory:
                    mask_tensor = mask_tensor.pin_memory()
                self.preloaded_masks.append(mask_tensor)
            else:
                self.preloaded_masks.append(None)

            if self.load_sky_mask:
                if sky_mask is not None:
                    # sky_mask уже float32 или bool после конвертации выше
                    sky_mask_tensor = torch.from_numpy(sky_mask)
                    if self.to_gpu:
                        sky_mask_tensor = sky_mask_tensor.to(self.device)
                    elif self.pin_memory:
                        sky_mask_tensor = sky_mask_tensor.pin_memory()
                    self.preloaded_sky_masks.append(sky_mask_tensor)
                else:
                    self.preloaded_sky_masks.append(None)

            # Load depth data if needed
            if self.load_depths:
                worldtocam = np.linalg.inv(camtoworld)
                if img_data.name in self.scene.points.visibility:
                    point_indices = self.scene.points.visibility[img_data.name]
                    points_world = self.scene.points.xyz[point_indices]
                    points_cam = (worldtocam[:3, :3] @ points_world.T + worldtocam[:3, 3:4]).T
                    points_proj = (K @ points_cam.T).T
                    points = points_proj[:, :2] / points_proj[:, 2:3]
                    depths = points_cam[:, 2]

                    # Filter out points outside the image
                    h, w = image.shape[:2]
                    selector = (
                            (points[:, 0] >= 0)
                            & (points[:, 0] < w)
                            & (points[:, 1] >= 0)
                            & (points[:, 1] < h)
                            & (depths > 0)
                    )
                    points = points[selector]
                    depths = depths[selector]

                    points_tensor = torch.from_numpy(points).float()
                    depths_tensor = torch.from_numpy(depths).float()

                    if self.to_gpu:
                        points_tensor = points_tensor.to(self.device)
                        depths_tensor = depths_tensor.to(self.device)
                    elif self.pin_memory:
                        points_tensor = points_tensor.pin_memory()
                        depths_tensor = depths_tensor.pin_memory()

                    self.preloaded_points.append(points_tensor)
                    self.preloaded_depths.append(depths_tensor)
                else:
                    self.preloaded_points.append(None)
                    self.preloaded_depths.append(None)

        storage_type = "GPU" if self.to_gpu else ("pinned" if self.pin_memory else "regular")
        print(f"Preloaded {len(self.indices)} images into {storage_type} memory")

        if self.to_gpu:
            # Calculate GPU memory usage
            total_bytes = 0
            for img in self.preloaded_images:
                total_bytes += img.element_size() * img.nelement()
            print(f"GPU memory used for images: {total_bytes / 1024 ** 3:.2f} GB")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Get item with optional random cropping for training."""
        image = self.preloaded_images[item]
        K = self.preloaded_Ks[item]
        camtoworld = self.preloaded_camtoworlds[item]
        mask = self.preloaded_masks[item]
        sky_mask = self.preloaded_sky_masks[item] if self.load_sky_mask else None

        # Clone K since we might modify it (already on correct device)
        K = K.clone()

        # Apply random crop if patch_size is specified
        if self.patch_size is not None:
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y: y + self.patch_size, x: x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

            if mask is not None:
                mask = mask[y: y + self.patch_size, x: x + self.patch_size]

            if sky_mask is not None:
                sky_mask = sky_mask[y: y + self.patch_size, x: x + self.patch_size]


        data = {
            "K": K,
            "camtoworld": camtoworld,
            "image": image,
            "image_id": item,  # index in the preloaded dataset
        }

        if mask is not None:
            data["mask"] = mask

        if sky_mask is not None:
            data["sky_mask"] = sky_mask

        if self.load_depths and self.preloaded_points[item] is not None:
            data["points"] = self.preloaded_points[item]
            data["depths"] = self.preloaded_depths[item]

        return data


class PreloadedDataLoader:
    """DataLoader-like interface for preloaded dataset with fast GPU transfer."""

    def __init__(
            self,
            dataset: PreloadedDataset,
            batch_size: int = 1,
            shuffle: bool = True,
            device: str = "cuda",
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.num_samples = len(dataset)
        # Group indices by camera_id for same-resolution batching
        self.indices_by_camera: dict[int, list[int]] = defaultdict(list)
        for idx in range(self.num_samples):
            cam_id = dataset.preloaded_camera_ids[idx]
            self.indices_by_camera[cam_id].append(idx)
        self.camera_ids = list(self.indices_by_camera.keys())

    def __len__(self):
        # Each camera group produces ceil(n_images / batch_size) batches
        return sum(ceil(len(indices) / self.batch_size) for indices in self.indices_by_camera.values())

    def __iter__(self):
        if self.shuffle:
            # Shuffle camera order and indices within each camera
            camera_order = torch.randperm(len(self.camera_ids)).tolist()
            shuffled_groups = []
            for cam_idx in camera_order:
                cam_id = self.camera_ids[cam_idx]
                cam_indices = self.indices_by_camera[cam_id].copy()
                np.random.shuffle(cam_indices)
                shuffled_groups.append(cam_indices)
        else:
            shuffled_groups = [self.indices_by_camera[cam_id].copy() for cam_id in self.camera_ids]

        # Yield batches from each camera group
        for cam_indices in shuffled_groups:
            for i in range(0, len(cam_indices), self.batch_size):
                batch_indices = cam_indices[i:i + self.batch_size]
                batch_data = []

                for idx in batch_indices:
                    data = self.dataset[idx]
                    batch_data.append(data)

                # Collate batch
                batch = {}
                for key in batch_data[0].keys():
                    if batch_data[0][key] is None:
                        continue

                    values = [d[key] for d in batch_data if d[key] is not None]
                    if len(values) == 0:
                        continue

                    if key in ["points", "depths"]:
                        # For batch_size=1, return as tensor with batch dimension
                        # For larger batches, keep as list (not supported in current training code)
                        if self.batch_size == 1 and len(values) == 1:
                            val = values[0]
                            if not val.is_cuda:
                                val = val.to(self.device, non_blocking=True)
                            batch[key] = val.unsqueeze(0)
                        else:
                            # Variable length - keep as list (will break current training code)
                            batch[key] = [v if v.is_cuda else v.to(self.device, non_blocking=True) for v in values]
                    elif isinstance(values[0], torch.Tensor):
                        # Stack tensors and move to device if needed
                        stacked = torch.stack(values)
                        if not stacked.is_cuda:
                            stacked = stacked.to(self.device, non_blocking=True)
                        batch[key] = stacked
                    else:
                        # Keep as tensor for image_id
                        batch[key] = torch.tensor(values, device=self.device)

                yield batch
