from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from typing_extensions import Literal


@dataclass
class DatasetConfig:
    """Base dataset configuration."""
    data_dir: str = ""
    result_dir: str | None = None

    # Normalize the world space
    normalize_world_space: bool = True

    use_sparse_depths: bool = False

    sky_mask_subdir: str | None = "skymask"
    object_mask_subdir: str | None = None

    invert_mask: bool = False
    soft_mask: bool = True

    ckpt: Optional[str] = None

    resume_ckpt: Optional[str] = None
    skysphere_ckpt: Optional[str] = None


@dataclass
class ColmapDatasetConfig(DatasetConfig):
    """COLMAP dataset configuration."""
    parser_type: Literal["colmap"] = "colmap"


@dataclass
class WaymoDatasetConfig(DatasetConfig):
    """Waymo dataset configuration."""
    parser_type: Literal["waymo"] = "waymo"
    waymo_camera_angles: List[str] = field(default_factory=lambda: ["FRONT", "FRONT_LEFT"])
    waymo_frame_range: Tuple[int, int] = (0, 250)
    waymo_calib_dir: Optional[str] = None
    waymo_load_lidar: bool = True


@dataclass
class RCDatasetConfig(DatasetConfig):
    """RealityCapture dataset configuration."""
    parser_type: Literal["rc"] = "rc"


DATASET_DTU_SCAN122 = ColmapDatasetConfig(
    data_dir=r"x:\_ai\_demos\_gsplat\_datasets\2DGS_DTU\scan122",
)
DATASET_YOUTUBE05_TOWEL = ColmapDatasetConfig(
    data_dir=r"x:\_ai\_demos\_gsplat\_datasets\youtube05\towel",
)
DATASET_GUGONG = ColmapDatasetConfig(
    data_dir=r"x:\_ai\_demos\_gsplat\_datasets\gugong",
)
DATASET_DRJOHNSON = ColmapDatasetConfig(
    data_dir=r"x:\_ai\_demos\_gsplat\_datasets\drjohnson",
)
DATASET_GARDEN = ColmapDatasetConfig(
    data_dir=r"X:\_ai\_gsplat\datasets\garden",
)
DATASET_PATIO_HIGH = ColmapDatasetConfig(
    data_dir=r"x:\_ai\_gsplat\datasets\nerfonthego-undistorted\patio-high",
)
DATASET_MOUNTAIN = ColmapDatasetConfig(
    data_dir=r"x:\_ai\_gsplat\datasets\nerfonthego-undistorted\mountain",
)
DATASET_BICYCLE = ColmapDatasetConfig(
    data_dir=r"x:\_ai\_gsplat\datasets\bicycle",
)
DATASET_KITCHEN = ColmapDatasetConfig(
    data_dir=r"X:\_ai\_gsplat\datasets\kitchen",
)
DATASET_FB_COLMAP = ColmapDatasetConfig(
    data_dir=r"x:\_ai\_gsplat\datasets\fb_colmap_res",
)
DATASET_GOPR6996 = ColmapDatasetConfig(
    data_dir=r"y:\_gopro_kv92\extracted_keyframes\GOPR6996_colmap",
)
DATASET_GOPR7015 = ColmapDatasetConfig(
    data_dir=r"y:\_gopro_kv92\calib_charuco\video\extracted_keyframes\GOPR7015\colmap_db",
)
DATASET_GOOD_PARK = ColmapDatasetConfig(
    data_dir=r"y:\_gopro_kv92\2025-10-06-1\3-good-park",
)
DATASET_SEGMENT_102751 = ColmapDatasetConfig(
    data_dir=r"x:\_ai\_demos\_gsplat\_datasets\segment-102751",
)
DATASET_YOUTUBE01 = ColmapDatasetConfig(
    data_dir=r"x:\_ai\_demos\_gsplat\_datasets\youtube01",
)
DATASET_SOUTH_BUILDING = ColmapDatasetConfig(
    data_dir=r"x:\_ai\_glomap\data\south-building",
)
DATASET_WAYMO_F_FL_FR = WaymoDatasetConfig(
    data_dir=r"x:\_ai\_waymo\tensorflow_extractor\colmap_proj_F_FL_FR",
    skysphere_ckpt=r"step_005999\skysphere.pt",
    invert_mask=True,
)
DATASET_WAYMO_COLMAP_PROJ = WaymoDatasetConfig(
    data_dir=r"x:\_ai\_waymo\tensorflow_extractor\colmap_proj",
    skysphere_ckpt=r"step_005999\skysphere.pt",
    invert_mask=True,
)
