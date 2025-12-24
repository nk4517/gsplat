from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple
from typing_extensions import Literal

from examples.datasets.waymo import WAYMO_CAMERAS


@dataclass
class DatasetConfig:
    """Base dataset configuration."""
    dataset_dir: str | Path = ""
    output_dir: str | None = None

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

    def __post_init__(self):
        if self.output_dir is None and self.dataset_dir:
            self.output_dir = str(Path(self.dataset_dir).with_suffix(".out"))

    def resolve_ckpt_path(self, ckpt_path: str | None) -> Path | None:
        """Resolve checkpoint path: absolute or relative to result_dir."""
        if ckpt_path is None:
            return None
        p = Path(ckpt_path)
        if p.is_absolute():
            return p
        # Relative path - resolve from result_dir
        return Path(self.output_dir) / p if self.output_dir else p


@dataclass
class ColmapDatasetConfig(DatasetConfig):
    """COLMAP dataset configuration."""
    parser_type: Literal["colmap"] = "colmap"


@dataclass
class WaymoDatasetConfig(DatasetConfig):
    """Waymo dataset configuration."""
    parser_type: Literal["waymo"] = "waymo"
    waymo_camera_angles: List[str] = field(default_factory=lambda: WAYMO_CAMERAS)
    waymo_frame_range: Tuple[int, int] = (0, 250)
    waymo_calib_dir: Optional[str] = r"X:\_ai\_waymo\tensorflow_extractor\colmap_proj"
    waymo_load_lidar: bool = True


@dataclass
class RCDatasetConfig(DatasetConfig):
    """RealityCapture dataset configuration."""
    parser_type: Literal["rc"] = "rc"


DATASET_DTU_SCAN122 = ColmapDatasetConfig(
    dataset_dir=r"x:\_ai\_demos\_gsplat\_datasets\2DGS_DTU\scan122",
)
DATASET_YOUTUBE05_TOWEL = ColmapDatasetConfig(
    dataset_dir=r"x:\_ai\_demos\_gsplat\_datasets\youtube05\towel",
)
DATASET_GUGONG = ColmapDatasetConfig(
    dataset_dir=r"x:\_ai\_demos\_gsplat\_datasets\gugong",
)
DATASET_DRJOHNSON = ColmapDatasetConfig(
    dataset_dir=r"x:\_ai\_demos\_gsplat\_datasets\drjohnson",
)
DATASET_GARDEN = ColmapDatasetConfig(
    dataset_dir=r"X:\_ai\_gsplat\datasets\garden",
)
DATASET_PATIO_HIGH = ColmapDatasetConfig(
    dataset_dir=r"x:\_ai\_gsplat\datasets\nerfonthego-undistorted\patio-high",
)
DATASET_MOUNTAIN = ColmapDatasetConfig(
    dataset_dir=r"x:\_ai\_gsplat\datasets\nerfonthego-undistorted\mountain",
)
DATASET_BICYCLE = ColmapDatasetConfig(
    dataset_dir=r"x:\_ai\_gsplat\datasets\bicycle",
)
DATASET_KITCHEN = ColmapDatasetConfig(
    dataset_dir=r"X:\_ai\_gsplat\datasets\kitchen",
)
DATASET_FB_COLMAP = ColmapDatasetConfig(
    dataset_dir=r"x:\_ai\_gsplat\datasets\fb_colmap_res",
)
DATASET_GOPR6996 = ColmapDatasetConfig(
    dataset_dir=r"y:\_gopro_kv92\extracted_keyframes\GOPR6996_colmap",
)
DATASET_GOPR7015 = ColmapDatasetConfig(
    dataset_dir=r"y:\_gopro_kv92\calib_charuco\video\extracted_keyframes\GOPR7015\colmap_db",
)
DATASET_GOOD_PARK = ColmapDatasetConfig(
    dataset_dir=r"y:\_gopro_kv92\2025-10-06-1\3-good-park",
)
DATASET_SEGMENT_102751 = ColmapDatasetConfig(
    dataset_dir=r"x:\_ai\_demos\_gsplat\_datasets\segment-102751",
    skysphere_ckpt=r"X:\_ai\_demos\_gsplat\_datasets\segment-102751.out\sky-full\ckpts\step_005999\skysphere.pt",
    invert_mask=True,
)
DATASET_YOUTUBE01 = ColmapDatasetConfig(
    dataset_dir=r"x:\_ai\_demos\_gsplat\_datasets\youtube01",
)
DATASET_SOUTH_BUILDING = ColmapDatasetConfig(
    dataset_dir=r"x:\_ai\_glomap\data\south-building",
)
DATASET_WAYMO_F_FL_FR = WaymoDatasetConfig(
    dataset_dir=r"x:\_ai\_waymo\tensorflow_extractor\colmap_proj_F_FL_FR",
    # skysphere_ckpt=r"step_005999\skysphere.pt",
    invert_mask=True,
)
DATASET_WAYMO_COLMAP_PROJ = WaymoDatasetConfig(
    dataset_dir=r"x:\_ai\_waymo\tensorflow_extractor\colmap_proj",
    # skysphere_ckpt=r"X:\_ai\_waymo\tensorflow_extractor\colmap_proj.result\sky-only\ckpts\step_005999\skysphere.pt",
    invert_mask=True,
    waymo_camera_angles=["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"],
)
DATASET_TRAIN = ColmapDatasetConfig(
    dataset_dir=r"x:\_ai\_demos\_gsplat\_datasets\train",
    skysphere_ckpt=r"X:\_ai\_demos\_gsplat\_datasets\train.out\sky-full\ckpts\step_005999\skysphere.pt",
    resume_ckpt=r"X:\_ai\_demos\_gsplat\_datasets\train.out\ckpts\ckpt_240x133_6023.pt"
)
DATASET_TRUCK = ColmapDatasetConfig(
    dataset_dir=r"x:\_ai\_demos\_gsplat\_datasets\truck",
)

DATASET_TRAIN_DA3 = ColmapDatasetConfig(
    dataset_dir=r"x:\_ai\_my_depthanything_3_workspace\output",
    output_dir=r"X:\_ai\_demos\_gsplat\_datasets\train.out",
)

DATASET_DUCKOV_COLMAP = ColmapDatasetConfig(
    dataset_dir=r"x:\_ai\_gsplat\datasets\__my\duckov_depth_normals\conv"
)
