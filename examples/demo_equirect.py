"""Demo: camera -> equirect -> camera roundtrip."""
import argparse
import shutil
from typing import TYPE_CHECKING

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tqdm import tqdm

from examples import my_datasets
from examples.datasets.scene_prepare import prepare_scene
from examples.datasets.waymo import WaymoParser, WAYMO_CAMERAS
from examples.datasets.colmap import Parser as ColmapParser
from examples.lib_360 import remap_camera_to_equirect, remap_equirect_to_camera, remap_camera_to_eac, remap_eac_to_camera

from examples.datasets.normalize_3d import apply_scene_normalization

if TYPE_CHECKING:
    from examples.datasets.dataset import Scene

# gsplat (Z-up) -> equirect (Y-up) coordinate transform
R_gsplat_to_equirect = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float32)

def iter_frames_with_masks(scene: "Scene", max_frames: int = 10000, invert=False):
    """Итерирует по фреймам сцены, загружая и подготавливая маски.
    
    Yields: (i, img_data, cam, mask_float, c2w_transformed)
    """
    n_frames = min(max_frames, len(scene.images))
    for i in range(n_frames):
        img_data = scene.images[i]
        cam = scene.cameras[img_data.camera_id]
        mask_path = img_data.aux_fpaths.get("mask")
        if mask_path is None or not mask_path.exists():
            print(f"Frame {i}: no mask, skipping")
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
        mask = cv2.resize(mask, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
        mask_float = mask.astype(np.float32) / 255.0
        if invert:
            mask_float = 1 - mask_float
        c2w = R_gsplat_to_equirect @ img_data.camtoworld
        yield i, img_data, cam, mask_float, c2w

def build_equirect_heatmap(cfg, scene: "Scene"):
    """Накапливает маски камер в equirect проекцию и сохраняет в npy."""

    out_npy_path = Path(scene.output_dir) / "sky_equirect_normalized.npy"
    out_npy_path.parent.mkdir(exist_ok=True)

    # Equirect dimensions
    H_eq, W_eq = 1024, 2048

    mask_sum = np.zeros((H_eq, W_eq), dtype=np.float32)

    for i, _, cam, mask_float, c2w in iter_frames_with_masks(scene, invert=cfg.invert_mask):
        equirect_mask, valid = remap_camera_to_equirect(mask_float[..., None], cam.K, c2w, H_eq, W_eq)
        mask_sum += equirect_mask
        print(f"Frame {i}: added mask")

    sum_max = mask_sum.max()
    mask_norm = mask_sum / sum_max if sum_max > 0 else mask_sum
    
    np.save(out_npy_path, mask_norm)
    print(f"Saved heatmap to {out_npy_path}")
    
    # Визуализация inferno
    colored = (plt.cm.inferno(mask_norm)[..., :3] * 255).astype(np.uint8)
    cv2.imwrite(str(out_npy_path.with_suffix(".png")), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))


def build_eac_heatmap(cfg, scene: "Scene", face_size: int = 512):
    """Накапливает маски камер в EAC проекцию и сохраняет в npy."""

    out_npy_path = Path(scene.output_dir) / "sky_eac_normalized.npy"
    out_npy_path.parent.mkdir(exist_ok=True)

    eac_h, eac_w = 2 * face_size, 3 * face_size
    mask_sum = np.zeros((eac_h, eac_w), dtype=np.float32)

    for i, _, cam, mask_float, c2w in iter_frames_with_masks(scene, invert=cfg.invert_mask):
        eac_mask, valid = remap_camera_to_eac(mask_float[..., None], cam.K, c2w, face_size, interpolation=cv2.INTER_AREA)
        mask_sum += eac_mask
        print(f"Frame {i}: added mask")

    sum_max = mask_sum.max()
    mask_norm = mask_sum / sum_max if sum_max > 0 else mask_sum

    np.save(out_npy_path, mask_norm)
    print(f"Saved EAC heatmap to {out_npy_path}")

    colored = (plt.cm.inferno(mask_norm)[..., :3] * 255).astype(np.uint8)
    cv2.imwrite(str(out_npy_path.with_suffix(".png")), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

def prepare_sky_heat_dirs(cfg, scene: "Scene") -> tuple[Path, Path, Path]:
    """Подготавливает директории для sky_heat и определяет images_basedir.
    
    Returns: (sky_heat_out_dir, sky_heat_vis_out_dir, images_basedir)
    """
    sky_heat_out_dir = Path(cfg.output_dir) / "sky_heat"
    shutil.rmtree(sky_heat_out_dir, ignore_errors=True)
    sky_heat_out_dir.mkdir(exist_ok=True)
    sky_heat_vis_out_dir = Path(cfg.output_dir) / "sky_heat_vis"
    shutil.rmtree(sky_heat_vis_out_dir, ignore_errors=True)
    sky_heat_vis_out_dir.mkdir(exist_ok=True)

    try:
        images_subdir = scene.images[0].image_fpath.relative_to(Path(cfg.dataset_dir)).parts[0]
        images_basedir = Path(cfg.dataset_dir) / images_subdir
    except ValueError:
        images_subdir = scene.images[0].image_fpath.relative_to(Path(cfg.output_dir)).parts[0]
        images_basedir = Path(cfg.output_dir) / images_subdir
    return sky_heat_out_dir, sky_heat_vis_out_dir, images_basedir

def project_equirect_heatmap_to_cameras(cfg, scene: "Scene"):
    """Загружает equirect heatmap и проецирует на индивидуальные камеры."""
    
    heatmap_npy_path = Path(scene.output_dir) / "sky_equirect_normalized.npy"
    mask_norm = np.load(heatmap_npy_path)

    sky_heat_out_dir, sky_heat_vis_out_dir, images_basedir = prepare_sky_heat_dirs(cfg, scene)

    image_paths: dict[str, Path] = {}
    sky_heat_paths: dict[str, Path] = {}

    for i, img_data, cam, _, c2w in tqdm(iter_frames_with_masks(scene, invert=cfg.invert_mask), desc="Projecting back"):
        cam_heat = remap_equirect_to_camera(mask_norm, cam.K, c2w, cam.height, cam.width)
        cam_heat = (cam_heat > 0).astype(bool)
        cam_heat_gray = (cam_heat*255).clip(0, 255).astype(np.uint8)

        # Save grayscale
        rel_path = img_data.image_fpath.relative_to(images_basedir)
        out_rel = str(rel_path.with_suffix(".png"))
        sky_heat_path = sky_heat_out_dir / out_rel
        sky_heat_path.parent.mkdir(parents=True, exist_ok=True)

        # colored = (plt.cm.inferno(cam_heat)[..., :3] * 255).astype(np.uint8)
        # cv2.imwrite(str(out_dir / "mask_sum_inferno.png"), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(str(sky_heat_path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(sky_heat_path), cam_heat_gray)

        image_paths[out_rel] = img_data.image_fpath
        sky_heat_paths[out_rel] = sky_heat_path

    # save_masked_visualization(image_paths, sky_heat_paths, sky_heat_vis_out_dir)

    print(f"Saved sky_heat to {sky_heat_out_dir}")
    print(f"Saved sky_heat_vis to {sky_heat_vis_out_dir}")


def project_eac_heatmap_to_cameras(cfg, scene: "Scene"):
    """Загружает EAC heatmap и проецирует на индивидуальные камеры."""
    
    heatmap_npy_path = Path(scene.output_dir) / "sky_eac_normalized.npy"
    mask_norm = np.load(heatmap_npy_path)

    sky_heat_out_dir, sky_heat_vis_out_dir, images_basedir = prepare_sky_heat_dirs(cfg, scene)

    image_paths: dict[str, Path] = {}
    sky_heat_paths: dict[str, Path] = {}

    for i, img_data, cam, _, c2w in tqdm(iter_frames_with_masks(scene, invert=cfg.invert_mask), desc="Projecting EAC back"):
        cam_heat = remap_eac_to_camera(mask_norm, cam.K, c2w, cam.height, cam.width, interpolation=cv2.INTER_AREA)
        cam_heat = (cam_heat > 0).astype(bool)
        cam_heat_gray = (cam_heat*255).clip(0, 255).astype(np.uint8)

        rel_path = img_data.image_fpath.relative_to(images_basedir)
        out_rel = str(rel_path.with_suffix(".png"))
        sky_heat_path = sky_heat_out_dir / out_rel
        sky_heat_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(sky_heat_path), cam_heat_gray)

        image_paths[out_rel] = img_data.image_fpath
        sky_heat_paths[out_rel] = sky_heat_path

    print(f"Saved sky_heat to {sky_heat_out_dir}")
    print(f"Saved sky_heat_vis to {sky_heat_vis_out_dir}")



def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("mode", choices=["build", "project"], help="build=create heatmap, project=apply to cameras")
    # args = parser.parse_args()
    
    cfg = my_datasets.DATASET_TRAIN

    if isinstance(cfg, my_datasets.WaymoDatasetConfig):
        parser = WaymoParser(
            data_dir=cfg.dataset_dir,
            output_dir=cfg.output_dir,
            camera_angles=WAYMO_CAMERAS,
            frame_range=[0, 4000],
            load_lidar=True,
            waymo_calib_dir=cfg.waymo_calib_dir,
        )
    elif isinstance(cfg, my_datasets.ColmapDatasetConfig):
        parser = ColmapParser(
            data_dir=cfg.dataset_dir,
            output_dir=cfg.output_dir,
        )
    else:
        raise ValueError(f"Unknown dataset type: {type(cfg)}")

    apply_scene_normalization(parser.scene)
    scene = prepare_scene(parser.scene, factor=1)

    # if 0:# args.mode == "build":
        # build_equirect_heatmap(cfg, scene)
    # else:
    #     # project_equirect_heatmap_to_cameras(cfg, scene)
    build_eac_heatmap(cfg, scene)
    project_eac_heatmap_to_cameras(cfg, scene)


if __name__ == "__main__":
    main()
 