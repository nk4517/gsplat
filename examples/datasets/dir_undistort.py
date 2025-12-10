"""Unified resize and undistort operations for image/mask directories."""

# !!! LLM INSTRUCTION: ЗАПРЕЩЕНО применять .stem к rel_path - там могут быть вложенные подкаталоги.
# Для отсечения расширения использовать только Path(rel_path).with_suffix("").

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.uint8:
        return data
    if data.dtype in (np.float32, np.float64):
        return (np.clip(data, 0, 1) * 255).astype(np.uint8)
    if data.dtype == np.uint16:
        return (data / 257).astype(np.uint8)
    if data.dtype == bool:
        return data.astype(np.uint8) * 255
    return data.astype(np.uint8)


def load_image_or_mask(path: Path, is_mask: bool = False) -> np.ndarray:
    """Загрузить изображение или маску (.png, .jpg, .npy).
    
    Для масок: нормализует в uint8 0-255 независимо от исходного формата.
    Returns None if file is corrupted.
    """
    try:
        if path.suffix.lower() == ".npy":
            data = np.load(path)
        else:
            data = imageio.imread(path)
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return None
    
    if is_mask:
        data = normalize_to_uint8(data)
    return data


def resize_and_undistort_file(
    src_fpath: Path,
    dst_fpath: Path,
    target_size: tuple[int, int] | None,
    undistort_maps: tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]] | None,
    mode: str = 'image',
    alpha_mask_path: Path | None = None,
    skip_existing: bool = True,
) -> Path | None:
    """Ресайз и/или андисторт одного файла.
    
    Args:
        src_fpath: Путь к исходному файлу
        dst_fpath: Путь для сохранения результата
        target_size: (w, h) целевой размер или None
        undistort_maps: (mapx, mapy, roi) или None
        mode: 'image' | 'mask' - тип обрабатываемых данных
        alpha_mask_path: Путь для сохранения альфа-маски (только для mode='image')
        skip_existing: Пропускать существующие файлы
    
    Returns:
        dst_path если файл обработан/существует, None если ошибка
    """
    if skip_existing and dst_fpath.exists():
        return dst_fpath

    is_mask = mode != 'image'
    data = load_image_or_mask(src_fpath, is_mask=is_mask)
    if data is None:
        return None

    # Для масок - первый канал если многоканальная
    if is_mask and len(data.shape) == 3:
        data = data[..., 0]

    # Андисторт (до ресайза, т.к. maps построены для полного размера)
    if undistort_maps is not None:
        mapx, mapy, roi_undist = undistort_maps
        map_h, map_w = mapx.shape[:2]
        if data.shape[0] != map_h or data.shape[1] != map_w:
            data = cv2.resize(data, (map_w, map_h), interpolation=cv2.INTER_AREA)
        data = cv2.remap(data, mapx, mapy, cv2.INTER_AREA)
        x, y, w, h = roi_undist
        data = data[y : y + h, x : x + w]

    # Ресайз к целевому размеру (после андисторта)
    if target_size is not None:
        data = cv2.resize(data, target_size, interpolation=cv2.INTER_AREA)

    # Обработка альфа-канала для изображений
    if not is_mask and len(data.shape) == 3 and data.shape[2] == 4:
        if alpha_mask_path is not None:
            alpha_data = normalize_to_uint8(data[..., 3])
            alpha_mask_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(alpha_mask_path, alpha_data)
        data = data[..., :3]

    if data.dtype != np.uint8:
        data = data.astype(np.uint8)
    dst_fpath.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(dst_fpath, data)
    return dst_fpath


def build_undistort_params(
    image_names: list[str],
    camera_ids: list[int],
    params_dict: dict,
    mapx_dict: dict,
    mapy_dict: dict,
    roi_undist_dict: dict,
) -> dict[str, tuple[np.ndarray, np.ndarray, list]]:
    """Построить словарь undistort_params из данных парсера.
    
    Returns:
        dict: base_name -> (mapx, mapy, roi_undist) для изображений с дисторсией
    """
    result = {}
    for idx, image_name in enumerate(image_names):
        camera_id = camera_ids[idx]
        params = params_dict[camera_id]
        
        if len(params) == 0:
            continue
        
        base_name = Path(image_name).stem
        result[base_name] = (
            mapx_dict[camera_id],
            mapy_dict[camera_id],
            roi_undist_dict[camera_id],
        )
    
    return result
 

def save_masked_visualization(
    image_paths: dict[str, Path],
    mask_paths: dict[str, Path],
    output_dir: Path,
    alpha: float = 0.5,
    mask_color: tuple[int, int, int] = (255, 0, 255),
) -> None:
    """Сохранить визуализацию изображений с наложенной маской неба.
    
    Args:
        image_paths: rel_path -> Path к изображениям
        mask_paths: rel_path -> Path к маскам
        output_dir: Каталог для сохранения визуализаций
        alpha: Прозрачность маски (0-1)
        mask_color: RGB цвет для маски
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ключи могут иметь разные расширения (.jpg vs .png) - сравниваем без суффиксов
    img_keys_nosuffix = {str(Path(k).with_suffix("")): k for k in image_paths}
    mask_keys_nosuffix = {str(Path(k).with_suffix("")): k for k in mask_paths}
    common_keys_nosuffix = set(img_keys_nosuffix.keys()) & set(mask_keys_nosuffix.keys())
    
    if not common_keys_nosuffix:
        return
    
    print(f"Saving {len(common_keys_nosuffix)} masked visualizations to {output_dir}")
    for key_nosuffix in tqdm(sorted(common_keys_nosuffix), desc="mask_vis"):
        fname = output_dir / f"{key_nosuffix}.png"
        if fname.exists():
            continue
        
        img_rel = img_keys_nosuffix[key_nosuffix]
        mask_rel = mask_keys_nosuffix[key_nosuffix]
        img = imageio.imread(image_paths[img_rel])[..., :3].astype(np.float32)
        mask = imageio.imread(mask_paths[mask_rel])
        if len(mask.shape) == 3:
            mask = mask[..., 0]
        mask_norm = mask.astype(np.float32) / 255.0
        
        overlay = np.array(mask_color, dtype=np.float32)
        blended = img * (1 - alpha * mask_norm[..., None]) + overlay * (alpha * mask_norm[..., None])
        fname.parent.mkdir(exist_ok=True, parents=True)
        imageio.imwrite(fname, blended.clip(0, 255).astype(np.uint8))