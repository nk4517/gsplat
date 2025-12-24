"""Equirect to EAC converter."""
import cv2
import numpy as np
from pathlib import Path

from lib_360 import EAC_FACES, EAC_FACE_GRID, _eac_face_dirs, _rotate_face, _unrotate_face


def equirect_to_eac(equirect: np.ndarray, face_size: int, interpolation = cv2.INTER_LANCZOS4) -> np.ndarray:
    """Convert equirectangular image to EAC projection."""
    H_eq, W_eq = equirect.shape[:2]
    C = equirect.shape[2] if equirect.ndim == 3 else 1
    if equirect.ndim == 2:
        equirect = equirect[..., None]

    eac_h, eac_w = 2 * face_size, 3 * face_size
    result = np.zeros((eac_h, eac_w, C), dtype=equirect.dtype)

    for face in EAC_FACES:
        row, col = EAC_FACE_GRID[face]
        y0, x0 = row * face_size, col * face_size

        dirs = _eac_face_dirs(face, face_size)  # [face_size, face_size, 3]
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]

        # Direction -> spherical -> equirect UV
        theta = np.arctan2(x, z)  # [-pi, pi]
        phi = np.arcsin(np.clip(y, -1, 1))  # [-pi/2, pi/2]

        # pixel-center convention: map to [0.5, W-0.5] range
        map_x = ((theta / np.pi + 1) / 2 * W_eq - 0.5).astype(np.float32)
        map_y = ((0.5 - phi / np.pi) * H_eq - 0.5).astype(np.float32)

        face_img = cv2.remap(equirect, map_x, map_y, interpolation, borderMode=cv2.BORDER_WRAP)
        face_img = _rotate_face(face_img, face)

        if face_img.ndim == 2:
            face_img = face_img[..., None]

        result[y0:y0+face_size, x0:x0+face_size] = face_img

    if C == 1:
        result = result[..., 0]
    return result

def eac_to_equirect(eac: np.ndarray, H_eq: int, W_eq: int, interpolation = cv2.INTER_LANCZOS4) -> np.ndarray:
    """Convert EAC projection back to equirectangular."""
    eac_h, eac_w = eac.shape[:2]
    face_size = eac_h // 2
    C = eac.shape[2] if eac.ndim == 3 else 1
    if eac.ndim == 2:
        eac = eac[..., None]

    # Equirect pixel grid -> spherical -> world direction
    v_eq, u_eq = np.meshgrid(
        np.arange(H_eq, dtype=np.float32),
        np.arange(W_eq, dtype=np.float32),
        indexing='ij'
    )
    # pixel-center convention
    theta = ((u_eq + 0.5) / W_eq - 0.5) * 2 * np.pi      # [-pi, pi]
    phi = (0.5 - (v_eq + 0.5) / H_eq) * np.pi            # [pi/2, -pi/2]

    cos_phi = np.cos(phi)
    x = np.sin(theta) * cos_phi
    y = np.sin(phi)
    z = np.cos(theta) * cos_phi

    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)

    # Determine dominant face
    # face order: front=0, back=1, left=2, right=3, top=4, bottom=5
    face_idx = np.zeros((H_eq, W_eq), dtype=np.int32)
    face_idx = np.where((az >= ax) & (az >= ay) & (z > 0), 0, face_idx)   # front
    face_idx = np.where((az >= ax) & (az >= ay) & (z <= 0), 1, face_idx)  # back
    face_idx = np.where((ax > ay) & (ax > az) & (x < 0), 2, face_idx)     # left
    face_idx = np.where((ax > ay) & (ax > az) & (x >= 0), 3, face_idx)    # right
    face_idx = np.where((ay > ax) & (ay > az) & (y > 0), 4, face_idx)     # top
    face_idx = np.where((ay > ax) & (ay > az) & (y <= 0), 5, face_idx)    # bottom

    eps = 1e-8
    face_u = np.zeros((H_eq, W_eq), dtype=np.float32)
    face_v = np.zeros((H_eq, W_eq), dtype=np.float32)

    # front (+Z): u=x/z, v=-y/z
    m = face_idx == 0; face_u[m] = x[m] / (z[m] + eps); face_v[m] = -y[m] / (z[m] + eps)
    # back (-Z): u=-x/(-z), v=-y/(-z)
    m = face_idx == 1; face_u[m] = -x[m] / (-z[m] + eps); face_v[m] = -y[m] / (-z[m] + eps)
    # left (-X): u=z/(-x), v=-y/(-x)
    m = face_idx == 2; face_u[m] = z[m] / (-x[m] + eps); face_v[m] = -y[m] / (-x[m] + eps)
    # right (+X): u=-z/x, v=-y/x
    m = face_idx == 3; face_u[m] = -z[m] / (x[m] + eps); face_v[m] = -y[m] / (x[m] + eps)
    # top (+Y): u=x/y, v=z/y
    m = face_idx == 4; face_u[m] = x[m] / (y[m] + eps); face_v[m] = z[m] / (y[m] + eps)
    # bottom (-Y): u=x/(-y), v=-z/(-y)
    m = face_idx == 5; face_u[m] = x[m] / (-y[m] + eps); face_v[m] = -z[m] / (-y[m] + eps)

    # EAC inverse mapping
    face_u = np.arctan(face_u) / (np.pi / 4)
    face_v = np.arctan(face_v) / (np.pi / 4)

    # To pixel coords, pixel-center convention
    px = ((face_u + 1) / 2 * face_size - 0.5).astype(np.float32)
    py = ((face_v + 1) / 2 * face_size - 0.5).astype(np.float32)
    px = np.clip(px, 0, face_size - 1)
    py = np.clip(py, 0, face_size - 1)

    # Grid positions and rotations (matching _rotate_face)
    grid_pos = [(0, 1), (1, 1), (0, 0), (0, 2), (1, 2), (1, 0)]  # front,back,left,right,top,bottom
    rotations = [0, 3, 0, 0, 1, 1]  # k for rot90 to undo _rotate_face

    map_x = np.zeros((H_eq, W_eq), dtype=np.float32)
    map_y = np.zeros((H_eq, W_eq), dtype=np.float32)

    for fi, (row, col) in enumerate(grid_pos):
        m = face_idx == fi
        fpx, fpy = px[m], py[m]
        k = rotations[fi]
        if k == 1:
            fpx, fpy = fpy, face_size - 1 - fpx
        elif k == 3:
            fpx, fpy = face_size - 1 - fpy, fpx
        map_x[m] = col * face_size + fpx
        map_y[m] = row * face_size + fpy

    result = cv2.remap(eac, map_x, map_y, interpolation, borderMode=cv2.BORDER_REPLICATE)
    if C == 1 and result.ndim == 3:
        result = result[..., 0]
    return result




def main1():
    """Demo: equirect -> EAC -> equirect roundtrip."""
    input_path = Path(r"P:\3d_printing\_gsplat_sandbox\splatting_app\gsplat-2025\examples\data\1000_F_208486829_9Zf0XvJq5IQTWPf9kcJPes4dOWWMXlNX.jpg")
    eac_path = input_path.with_stem(input_path.stem + "_eac").with_suffix(".png")
    roundtrip_path = input_path.with_stem(input_path.stem + "_roundtrip").with_suffix(".png")
    diff_path = input_path.with_stem(input_path.stem + "_diff").with_suffix(".png")

    equirect = cv2.imread(str(input_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    H_eq, W_eq = equirect.shape[:2]
    face_size = equirect.shape[0] // 3 * 2

    eac = equirect_to_eac(equirect, face_size)
    cv2.imwrite(str(eac_path), eac)
    print(f"Saved EAC: {eac_path}")

    roundtrip = eac_to_equirect(eac, H_eq, W_eq)
    cv2.imwrite(str(roundtrip_path), roundtrip)
    print(f"Saved roundtrip: {roundtrip_path}")

    diff = cv2.absdiff(equirect, roundtrip)
    diff_scaled = np.clip(diff.astype(np.float32) * 10, 0, 255).astype(np.uint8)
    cv2.imwrite(str(diff_path), diff_scaled)
    print(f"Saved diff (10x): {diff_path}")

    mae = np.mean(np.abs(equirect.astype(np.float32) - roundtrip.astype(np.float32)))
    psnr = cv2.PSNR(equirect, roundtrip) if np.any(diff) else float('inf')
    print(f"MAE: {mae:.2f}, PSNR: {psnr:.2f} dB")


def main2():
    """Demo: EAC -> equirect."""
    input_path = Path(r"P:\3d_printing\_gsplat_sandbox\splatting_app\gsplat-2025\examples\data\image-20251210062422-5nmf9au.png")
    output_path = input_path.with_stem(input_path.stem + "_equirect").with_suffix(".png")

    eac = cv2.imread(str(input_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    eac_h, eac_w = eac.shape[:2]
    face_w, face_h = eac_w // 3, eac_h // 2
    face_size = min(face_w, face_h)

    # ресайз каждого face отдельно до квадратного (чтобы интерполяция не замешивала границы)
    if face_w != face_h:
        eac_new = np.zeros((2 * face_size, 3 * face_size, eac.shape[2]) if eac.ndim == 3 else (2 * face_size, 3 * face_size), dtype=eac.dtype)
        for row in range(2):
            for col in range(3):
                y0, x0 = row * face_h, col * face_w
                face = eac[y0:y0+face_h, x0:x0+face_w]
                face_resized = cv2.resize(face, (face_size, face_size), interpolation=cv2.INTER_LANCZOS4)
                y1, x1 = row * face_size, col * face_size
                eac_new[y1:y1+face_size, x1:x1+face_size] = face_resized
        eac = eac_new

    H_eq = face_size * 2
    W_eq = face_size * 4

    equirect = eac_to_equirect(eac, H_eq, W_eq)
    cv2.imwrite(str(output_path), equirect)
    print(f"Saved equirect: {output_path}")

if __name__ == "__main__":
    main2()