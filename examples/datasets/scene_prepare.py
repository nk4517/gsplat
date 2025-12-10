"""Подготовка сцены: resize и undistort изображений."""

from pathlib import Path
from typing import Dict, Any

import numpy as np
from tqdm import tqdm

from .dataset import Scene, CameraIntrinsics, ImagePose, PrepareContext, ImagePrepareEntry
from .dir_undistort import (
    resize_and_undistort_file,
    save_masked_visualization,
)

_AUX_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.npy')
_AUX_SUBDIRS = ('mask', 'masks', 'skymask', 'masks_alpha', 'depth', 'depths', "sky_heat")


def _find_aux_files(
    dataset_dir: Path,
    base_path: Path,
    extra_dirs: list[Path] | None = None,
) -> dict[str, Path]:
    """Найти aux-файлы (mask, depth, sky_heat) для изображения. Возвращает {subdir: rel_path}."""
    result: dict[str, Path] = {}
    search_dirs = [dataset_dir] + (extra_dirs or [])
    for subdir in _AUX_SUBDIRS:
        for search_dir in search_dirs:
            aux_dir = Path(search_dir) / subdir
            if not aux_dir.is_dir():
                continue
            for ext in _AUX_EXTENSIONS:
                candidate = aux_dir / base_path.with_suffix(ext)
                if candidate.exists():
                    result[subdir] = base_path.with_suffix(ext)
                    break
            if subdir in result:
                break
    return result


def prepare_scene(
        scene: Scene,
        factor: int | None = None,
        target_resolution: int | tuple[int, int] | None = None,
) -> Scene:
    """Применить resize и undistort к изображениям сцены.
    
    Args:
        scene: Исходная сцена с полным разрешением
        factor: Фактор уменьшения (2, 4, 8...)
        target_resolution: Целевое разрешение (max_side или (w, h))
    
    Returns:
        Модифицированная сцена с обновлёнными путями и интринсиками
    """

    dataset_dir = Path(scene.dataset_dir)
    destination_dir = Path(scene.output_dir)
    if destination_dir is None:
        raise ValueError("scene.output_dir must be set before calling prepare_scene")

    has_distortion = any(c.undistortion is not None for c in scene.cameras.values())
    need_preparation = (factor is not None) or (target_resolution is not None) or has_distortion

    try:
        images_subdir = scene.images[0].image_fpath.relative_to(dataset_dir).parts[0]
        images_basedir = dataset_dir / images_subdir
    except ValueError:
        images_subdir = scene.images[0].image_fpath.relative_to(destination_dir).parts[0]
        images_basedir = destination_dir / images_subdir

    if not need_preparation:
        extra_dirs = [scene.output_dir] if scene.output_dir and scene.output_dir != dataset_dir else None
        
        new_images = []
        for img in scene.images:
            base_path = img.image_fpath.relative_to(images_basedir).with_suffix("")
            aux_raw = _find_aux_files(dataset_dir, base_path, extra_dirs)
            aux_fpaths = {}
            for subdir, rel in aux_raw.items():
                key = "mask" if "mask" in subdir else "depth" if "depth" in subdir else subdir
                aux_fpaths[key] = dataset_dir / subdir / rel
            new_images.append(ImagePose(
                name=img.name,
                camera_id=img.camera_id,
                camtoworld=img.camtoworld,
                image_fpath=img.image_fpath,
                aux_fpaths=aux_fpaths,
            ))
        return Scene(
            cameras=scene.cameras,
            images=new_images,
            points=scene.points,
            transform=scene.transform,
            scene_scale=scene.scene_scale,
            bounds=scene.bounds,
            extconf=scene.extconf,
            dataset_dir=dataset_dir,
            output_dir=destination_dir,
        )
    
    # Построить суффикс для каталогов
    suffix_parts = []
    if target_resolution is not None:
        if isinstance(target_resolution, int):
            suffix_parts.append(f"_r{target_resolution}")
        else:
            suffix_parts.append(f"_r{target_resolution[0]}x{target_resolution[1]}")
    elif factor > 1 and not scene.extconf.get("no_factor_suffix", False):
        suffix_parts.append(f"_{factor}")
    if has_distortion:
        suffix_parts.append("_undist")
    dir_suffix = "".join(suffix_parts)
    
    # Собрать ImagePrepareEntry для каждого изображения
    images_list: list[ImagePrepareEntry] = []
    for img in scene.images:
        cam = scene.cameras[img.camera_id]
        rel_path = img.image_fpath.relative_to(images_basedir)
        base_path = rel_path.with_suffix("")
        
        # target_size
        target_size = None
        if target_resolution is not None or factor > 1:
            src_w, src_h = cam.width, cam.height
            if target_resolution is not None:
                if isinstance(target_resolution, int):
                    scale = target_resolution / max(src_w, src_h)
                else:
                    scale_w = target_resolution[0] / src_w
                    scale_h = target_resolution[1] / src_h
                    scale = min(scale_w, scale_h)
                scale = min(scale, 1.0)
                target_size = (int(src_w * scale), int(src_h * scale))
            else:
                target_size = (src_w // factor, src_h // factor)
        
        # undistort_maps
        undistort_maps = None
        if cam.undistortion is not None:
            undistort_maps = (
                cam.undistortion.mapx,
                cam.undistortion.mapy,
                cam.undistortion.roi,
            )
        
        # aux files
        extra_dirs = [destination_dir] if destination_dir and destination_dir != dataset_dir else None
        aux = _find_aux_files(dataset_dir, base_path, extra_dirs)
        
        images_list.append(ImagePrepareEntry(
            rel_path=rel_path,
            camera_id=img.camera_id,
            target_size=target_size,
            undistort_maps=undistort_maps,
            aux=aux,
        ))
    
    ctx = PrepareContext(
        dataset_dir=dataset_dir,
        destination_dir=Path(destination_dir),
        transform_suffix=dir_suffix,
        images=images_list,
    )
    
    # Подготовить изображения
    images_dst_dir = ctx.destination_dir / (images_subdir + dir_suffix)
    
    image_mapping: dict[Path, Path] = {}
    aux_mappings: dict[str, dict[Path, Path]] = {}  # subdir -> {rel_path -> dst_path}
    
    for entry in tqdm(ctx.images, desc="images"):
        src_fpath = images_basedir / entry.rel_path
        dst_fpath = images_dst_dir / entry.rel_path.with_suffix(".png")
        alpha_mask_dst_dir = ctx.destination_dir / ("masks_alpha" + dir_suffix)
        alpha_mask_fpath = alpha_mask_dst_dir / entry.rel_path.with_suffix(".png")
        
        result = resize_and_undistort_file(
            src_fpath, dst_fpath, entry.target_size, entry.undistort_maps,
            mode='image', alpha_mask_path=alpha_mask_fpath,
        )
        if result:
            image_mapping[entry.rel_path] = result
        if alpha_mask_fpath.exists():
            aux_mappings.setdefault("masks_alpha", {})[entry.rel_path] = alpha_mask_fpath
        
        # aux files (mask, depth, etc.)
        for aux_subdir, aux_rel_path in entry.aux.items():
            aux_src = ctx.dataset_dir / aux_subdir / aux_rel_path
            # Если файл не в dataset_dir, проверить в destination_dir (для сгенерированных файлов типа sky_heat)
            if not aux_src.exists() and ctx.destination_dir:
                alt_src = ctx.destination_dir / aux_subdir / aux_rel_path
                if alt_src.exists():
                    aux_src = alt_src
                else:
                    raise RuntimeError("missing aux")
            aux_dst_dir = ctx.destination_dir / (aux_subdir + dir_suffix)
            aux_dst = aux_dst_dir / aux_rel_path.with_suffix(".png")
            mode = 'mask' if 'mask' in aux_subdir else 'image'
            aux_result = resize_and_undistort_file(
                aux_src, aux_dst, entry.target_size, entry.undistort_maps, mode=mode,
            )
            if aux_result:
                aux_mappings.setdefault(aux_subdir, {})[entry.rel_path] = aux_result
            elif not aux_src.exists():
                print(f"Warning: aux file not found: {aux_src}")
    
    # Визуализация для всех типов масок (не depth)
    for aux_subdir, mapping in aux_mappings.items():
        if 'depth' in aux_subdir:
            continue
        if mapping:
            save_masked_visualization(
                image_mapping, mapping,
                ctx.destination_dir / (aux_subdir + "_vis" + ctx.transform_suffix),
            )
    
    # Alpha masks как fallback
    alpha_mapping = aux_mappings.get("masks_alpha", {})
    mask_mapping = aux_mappings.get("skymask", aux_mappings.get("mask", aux_mappings.get("masks", {})))
    for rel_path, path in alpha_mapping.items():
        if rel_path not in mask_mapping:
            mask_mapping[rel_path] = path
    
    # Обновить пути в images
    new_images = []
    for img in scene.images:
        rel_path = img.image_fpath.relative_to(images_basedir)
        aux_fpaths = {}
        # mask: приоритет mask/masks, fallback на masks_alpha
        m = mask_mapping.get(rel_path) or alpha_mapping.get(rel_path)
        if m:
            aux_fpaths["mask"] = m

        # depth
        depth_mapping = aux_mappings.get("depth", aux_mappings.get("depths", {}))
        d = depth_mapping.get(rel_path)
        if d:
            aux_fpaths["depth"] = d

        # sky_heat
        sky_heat_mapping = aux_mappings.get("sky_heat", {})
        sh = sky_heat_mapping.get(rel_path)
        if sh:
            aux_fpaths["sky_heat"] = sh
        new_fpath = image_mapping.get(rel_path, img.image_fpath)
        new_images.append(ImagePose(
            name=img.name,
            camera_id=img.camera_id,
            camtoworld=img.camtoworld,
            image_fpath=new_fpath,
            aux_fpaths=aux_fpaths,
        ))
    
    # Обновить интринсики камер
    new_cameras: dict[int, CameraIntrinsics] = {}
    need_update_intrinsics = any(e.target_size is not None for e in ctx.images)
    if need_update_intrinsics:
        for cam_id, cam in scene.cameras.items():
            for entry in ctx.images:
                if entry.camera_id == cam_id and entry.target_size is not None:
                    dst_w, dst_h = entry.target_size
                    scale_x = dst_w / cam.width if cam.width else 1.0
                    scale_y = dst_h / cam.height if cam.height else 1.0
                    new_K = cam.K.copy()
                    new_K[0, :] *= scale_x
                    new_K[1, :] *= scale_y
                    new_cameras[cam_id] = CameraIntrinsics(
                        camera_id=cam_id,
                        K=new_K,
                        width=dst_w,
                        height=dst_h,
                        undistortion=cam.undistortion,
                    )
                    break
    else:
        new_cameras = scene.cameras
    
    return Scene(
        cameras=new_cameras,
        images=new_images,
        points=scene.points,
        transform=scene.transform,
        scene_scale=scene.scene_scale,
        bounds=scene.bounds,
        extconf=scene.extconf,
        dataset_dir=dataset_dir,
        output_dir=destination_dir,
    )
 