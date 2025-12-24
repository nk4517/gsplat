"""
Equirectangular projection utilities.

Coordinate system:
    +Y = up
    +Z = forward (equirect horizontal center, theta=0)
    +X = right

Equirect mapping:
    u=W_eq/2 -> theta=0 -> +Z (forward)
    v=0 -> phi=+pi/2 -> +Y (zenith)
    v=H_eq -> phi=-pi/2 -> -Y (nadir)
"""
import cv2
import numpy as np


def remap_camera_to_equirect(
    image: np.ndarray,         # [h, w, C] или [h, w]
    K: np.ndarray,             # [3, 3]
    c2w: np.ndarray,           # [4, 4]
    H_eq: int,
    W_eq: int,
    interpolation: int = cv2.INTER_AREA,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remap camera image to equirectangular projection.
    Returns (equirect_image, valid_mask).
    """
    # Equirect pixel grid
    v_eq, u_eq = np.meshgrid(
        np.arange(H_eq, dtype=np.float32),
        np.arange(W_eq, dtype=np.float32),
        indexing='ij'
    )

    # Equirect -> spherical (pixel-center convention)
    theta = ((u_eq + 0.5) / W_eq - 0.5) * 2 * np.pi      # [-pi, pi]
    phi = (0.5 - (v_eq + 0.5) / H_eq) * np.pi            # [pi/2, -pi/2]

    # Spherical -> world direction
    cos_phi = np.cos(phi)
    x = np.sin(theta) * cos_phi
    y = np.sin(phi)
    z = np.cos(theta) * cos_phi
    dirs_world = np.stack([x, y, z], axis=-1)    # [H_eq, W_eq, 3]

    # World -> camera
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3]
    dirs_cam = dirs_world @ R.T                  # [H_eq, W_eq, 3]

    # Behind camera mask
    valid = dirs_cam[..., 2] > 0
    dirs_cam_safe = dirs_cam.copy()
    dirs_cam_safe[~valid, 2] = 1.0

    # Project to image
    pts_2d = dirs_cam_safe @ K.T
    map_x = (pts_2d[..., 0] / pts_2d[..., 2]).astype(np.float32)
    map_y = (pts_2d[..., 1] / pts_2d[..., 2]).astype(np.float32)

    # Check bounds
    h, w = image.shape[:2]
    valid &= (map_x >= 0) & (map_x < w) & (map_y >= 0) & (map_y < h)

    map_x[~valid] = -1
    map_y[~valid] = -1

    result = cv2.remap(image, map_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return result, valid


def remap_equirect_to_camera(
    equirect: np.ndarray,      # [H_eq, W_eq, C] или [H_eq, W_eq]
    K: np.ndarray,             # [3, 3]
    c2w: np.ndarray,           # [4, 4]
    h: int,
    w: int,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Remap equirectangular to camera view.
    """
    H_eq, W_eq = equirect.shape[:2]

    # Camera pixel grid -> rays
    v, u = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
    ones = np.ones_like(u)
    uv1 = np.stack([u, v, ones], axis=-1)        # [h, w, 3]

    K_inv = np.linalg.inv(K)
    rays_cam = uv1 @ K_inv.T
    rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True)

    R = c2w[:3, :3]
    rays_world = rays_cam @ R.T

    x, y, z = rays_world[..., 0], rays_world[..., 1], rays_world[..., 2]
    theta = np.arctan2(x, z)
    phi = np.arcsin(np.clip(y, -1, 1))

    # pixel-center convention
    map_x = ((theta / np.pi + 1) / 2 * W_eq - 0.5).astype(np.float32)
    map_y = ((0.5 - phi / np.pi) * H_eq - 0.5).astype(np.float32)

    return cv2.remap(equirect, map_x, map_y, interpolation, borderMode=cv2.BORDER_WRAP)


# EAC layout (3x2 grid):
#   Row 0: Left, Front, Right
#   Row 1: Bottom (rotated CW 90), Back (rotated 180), Top (rotated CCW 90)
EAC_FACES = ['left', 'front', 'right', 'bottom', 'back', 'top']
EAC_FACE_GRID = {
    'left':   (0, 0),
    'front':  (0, 1),
    'right':  (0, 2),
    'bottom': (1, 0),
    'back':   (1, 1),
    'top':    (1, 2),
}

def _eac_face_dirs(face: str, face_size: int) -> np.ndarray:
    """
    Generate world directions for EAC face pixels.
    Returns [face_size, face_size, 3] array of unit vectors.
    """
    # EAC: equi-angular mapping, pixel-center convention
    t = (np.arange(face_size, dtype=np.float32) + 0.5) / face_size * 2 - 1
    # EAC transform: tan(pi/4 * t) maps to equi-angular distribution
    t_eac = np.tan(np.pi / 4 * t)
    u, v = np.meshgrid(t_eac, t_eac, indexing='xy')  # u=horizontal, v=vertical

    ones = np.ones_like(u)

    # Face directions (Y-up coordinate system)
    if face == 'front':    # +Z
        dirs = np.stack([u, -v, ones], axis=-1)
    elif face == 'back':   # -Z
        dirs = np.stack([-u, -v, -ones], axis=-1)
    elif face == 'left':   # -X
        dirs = np.stack([-ones, -v, u], axis=-1)
    elif face == 'right':  # +X
        dirs = np.stack([ones, -v, -u], axis=-1)
    elif face == 'top':    # +Y
        dirs = np.stack([u, ones, v], axis=-1)
    elif face == 'bottom': # -Y
        dirs = np.stack([u, -ones, -v], axis=-1)
    else:
        raise ValueError(f"Unknown face: {face}")

    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs


def _rotate_face(img: np.ndarray, face: str) -> np.ndarray:
    """Apply rotation for bottom row faces in EAC layout."""
    if face == 'bottom':
        return np.rot90(img, k=1)   # CW 90
    elif face == 'back':
        return np.rot90(img, k=3)   # 270 CCW = 90 CW
    elif face == 'top':
        return np.rot90(img, k=1)   # CW 90
    return img


def _unrotate_face(img: np.ndarray, face: str) -> np.ndarray:
    """Reverse rotation for bottom row faces."""
    if face == 'bottom':
        return np.rot90(img, k=-1)
    elif face == 'back':
        return np.rot90(img, k=1)
    elif face == 'top':
        return np.rot90(img, k=-1)
    return img


def remap_camera_to_eac(
    image: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    face_size: int,
    interpolation: int = cv2.INTER_LANCZOS4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remap camera image to EAC projection.
    Output shape: [2*face_size, 3*face_size, C]
    Returns (eac_image, valid_mask).
    """
    h, w = image.shape[:2]
    C = image.shape[2] if image.ndim == 3 else 1
    if image.ndim == 2:
        image = image[..., None]

    eac_h, eac_w = 2 * face_size, 3 * face_size
    result = np.zeros((eac_h, eac_w, C), dtype=image.dtype)
    valid = np.zeros((eac_h, eac_w), dtype=bool)

    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3]

    for face in EAC_FACES:
        row, col = EAC_FACE_GRID[face]
        y0, x0 = row * face_size, col * face_size

        dirs_world = _eac_face_dirs(face, face_size)
        dirs_cam = dirs_world @ R.T

        behind = dirs_cam[..., 2] <= 0
        dirs_cam_safe = dirs_cam.copy()
        dirs_cam_safe[behind, 2] = 1.0

        pts_2d = dirs_cam_safe @ K.T
        map_x = (pts_2d[..., 0] / pts_2d[..., 2]).astype(np.float32)
        map_y = (pts_2d[..., 1] / pts_2d[..., 2]).astype(np.float32)

        face_valid = ~behind & (map_x >= 0) & (map_x < w) & (map_y >= 0) & (map_y < h)
        map_x[~face_valid] = -1
        map_y[~face_valid] = -1

        face_img = cv2.remap(image, map_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        face_img = _rotate_face(face_img, face)
        face_valid = _rotate_face(face_valid, face)

        # после cv2 [h, w, 1] становится [h, w]
        if face_img.ndim == 2:
            face_img = face_img[..., None]

        result[y0:y0+face_size, x0:x0+face_size] = face_img
        valid[y0:y0+face_size, x0:x0+face_size] = face_valid

    if C == 1:
        result = result[..., 0]
    return result, valid


def remap_eac_to_camera(
    eac: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    h: int,
    w: int,
    interpolation: int = cv2.INTER_LANCZOS4,
) -> np.ndarray:
    """
    Remap EAC to camera view.
    Input eac shape: [2*face_size, 3*face_size, C]
    """
    eac_h, eac_w = eac.shape[:2]
    face_size = eac_h // 2

    v, u = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
    ones = np.ones_like(u)
    uv1 = np.stack([u, v, ones], axis=-1)

    K_inv = np.linalg.inv(K)
    rays_cam = uv1 @ K_inv.T
    rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True)

    R = c2w[:3, :3]
    rays_world = rays_cam @ R.T
    x, y, z = rays_world[..., 0], rays_world[..., 1], rays_world[..., 2]
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)

    # Determine dominant face for each ray
    face_idx = np.zeros((h, w), dtype=np.int32)
    face_idx = np.where((az >= ax) & (az >= ay) & (z > 0), 0, face_idx)   # front
    face_idx = np.where((az >= ax) & (az >= ay) & (z <= 0), 1, face_idx)  # back
    face_idx = np.where((ax > ay) & (ax > az) & (x < 0), 2, face_idx)     # left
    face_idx = np.where((ax > ay) & (ax > az) & (x >= 0), 3, face_idx)    # right
    face_idx = np.where((ay > ax) & (ay > az) & (y > 0), 4, face_idx)     # top
    face_idx = np.where((ay > ax) & (ay > az) & (y <= 0), 5, face_idx)    # bottom

    # Face UV calculations (normalized to [-1, 1])
    eps = 1e-8
    face_u = np.zeros((h, w), dtype=np.float32)
    face_v = np.zeros((h, w), dtype=np.float32)

    # front (+Z): u=x/z, v=-y/z
    m = face_idx == 0
    face_u[m] = x[m] / (z[m] + eps)
    face_v[m] = -y[m] / (z[m] + eps)

    # back (-Z): u=-x/(-z), v=-y/(-z)
    m = face_idx == 1
    face_u[m] = -x[m] / (-z[m] + eps)
    face_v[m] = -y[m] / (-z[m] + eps)

    # left (-X): u=-z/(-x), v=-y/(-x)
    m = face_idx == 2
    face_u[m] = z[m] / (-x[m] + eps)
    face_v[m] = -y[m] / (-x[m] + eps)

    # right (+X): u=z/x, v=-y/x
    m = face_idx == 3
    face_u[m] = -z[m] / (x[m] + eps)
    face_v[m] = -y[m] / (x[m] + eps)

    # top (+Y): u=x/y, v=z/y
    m = face_idx == 4
    face_u[m] = x[m] / (y[m] + eps)
    face_v[m] = z[m] / (y[m] + eps)

    # bottom (-Y): u=x/(-y), v=-z/(-y)
    m = face_idx == 5
    face_u[m] = x[m] / (-y[m] + eps)
    face_v[m] = -z[m] / (-y[m] + eps)

    # EAC inverse: atan(t) / (pi/4) maps [-1,1] to [-1,1]
    face_u = np.arctan(face_u) / (np.pi / 4)
    face_v = np.arctan(face_v) / (np.pi / 4)

    # Convert to pixel coords within face [0, face_size)
    px = ((face_u + 1) / 2 * face_size).astype(np.float32)
    py = ((face_v + 1) / 2 * face_size).astype(np.float32)

    # Apply face rotations and grid offsets
    # face order: front=0, back=1, left=2, right=3, top=4, bottom=5
    grid_pos = [(0, 1), (1, 1), (0, 0), (0, 2), (1, 2), (1, 0)]  # (row, col)
    rotations = [0, 3, 0, 0, 1, 1]  # k for rot90 (reverse of _rotate_face)

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for fi, (row, col) in enumerate(grid_pos):
        m = face_idx == fi
        fpx, fpy = px[m], py[m]

        # Unrotate coordinates
        k = rotations[fi]
        if k == 1:  # was rotated CW, so rotate coords CCW
            fpx, fpy = fpy, face_size - 1 - fpx
        elif k == -1:  # was rotated CCW, so rotate coords CW
            fpx, fpy = face_size - 1 - fpy, fpx
        elif k == 2:
            fpx, fpy = face_size - 1 - fpx, face_size - 1 - fpy

        map_x[m] = col * face_size + fpx
        map_y[m] = row * face_size + fpy

    return cv2.remap(eac, map_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)