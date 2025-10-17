import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from typing import Dict, Any, Optional, List
from tqdm import tqdm


class PreloadedDataset:
    """Dataset with all images preloaded into memory for fast access.
    
    Can store data either in pinned CPU memory or directly on GPU.
    """
    
    def __init__(
        self,
        parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        device: str = "cuda",
        pin_memory: bool = True,
        to_gpu: bool = False,  # If True, store directly on GPU instead of pinned memory
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.device = device
        self.pin_memory = pin_memory and not to_gpu  # Don't pin if storing on GPU
        self.to_gpu = to_gpu
        
        # Determine indices based on split
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]
        
        # Map from dataset index to original parser index
        self.index_mapping = {i: idx for i, idx in enumerate(self.indices)}
        
        # Preload all data
        storage_type = "GPU" if to_gpu else ("pinned" if self.pin_memory else "regular")
        print(f"Preloading {len(self.indices)} {split} images into {storage_type} memory...")
        self._preload_data()
        
    def _preload_data(self):
        """Preload all images and associated data into memory."""
        self.preloaded_images = []
        self.preloaded_camtoworlds = []
        self.preloaded_Ks = []
        self.preloaded_masks = []
        self.preloaded_camera_ids = []
        
        if self.load_depths:
            self.preloaded_points = []
            self.preloaded_depths = []
        
        for idx in tqdm(self.indices, desc=f"Loading {self.split} images"):
            # Load image
            image = imageio.imread(self.parser.image_paths[idx])[..., :3]
            camera_id = self.parser.camera_ids[idx]
            K = self.parser.Ks_dict[camera_id].copy()
            params = self.parser.params_dict[camera_id]
            camtoworld = self.parser.camtoworlds[idx]
            mask = self.parser.mask_dict[camera_id]
            
            # Apply undistortion if needed
            if len(params) > 0:
                mapx, mapy = (
                    self.parser.mapx_dict[camera_id],
                    self.parser.mapy_dict[camera_id],
                )
                image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
                x, y, w, h = self.parser.roi_undist_dict[camera_id]
                image = image[y : y + h, x : x + w]
            
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
            self.preloaded_camera_ids.append(camera_id)
            
            if mask is not None:
                mask_tensor = torch.from_numpy(mask).bool()
                if self.to_gpu:
                    mask_tensor = mask_tensor.to(self.device)
                elif self.pin_memory:
                    mask_tensor = mask_tensor.pin_memory()
                self.preloaded_masks.append(mask_tensor)
            else:
                self.preloaded_masks.append(None)
            
            # Load depth data if needed
            if self.load_depths:
                worldtocam = np.linalg.inv(camtoworld)
                image_name = self.parser.image_names[idx]
                if image_name in self.parser.point_indices:
                    point_indices = self.parser.point_indices[image_name]
                    points_world = self.parser.points[point_indices]
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
            print(f"GPU memory used for images: {total_bytes / 1024**3:.2f} GB")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Get item with optional random cropping for training."""
        image = self.preloaded_images[item]
        K = self.preloaded_Ks[item]
        camtoworld = self.preloaded_camtoworlds[item]
        mask = self.preloaded_masks[item]
        
        # Clone K since we might modify it (already on correct device)
        K = K.clone()
        
        # Apply random crop if patch_size is specified
        if self.patch_size is not None:
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y
            
            if mask is not None:
                mask = mask[y : y + self.patch_size, x : x + self.patch_size]
        
        # Use the mapped index for image_id to maintain consistency
        original_idx = self.index_mapping[item]
        
        data = {
            "K": K,
            "camtoworld": camtoworld,
            "image": image,
            "image_id": item,  # index in the preloaded dataset
        }
        
        if mask is not None:
            data["mask"] = mask
        
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
        
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.num_samples).tolist()
        else:
            indices = list(range(self.num_samples))
        
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
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