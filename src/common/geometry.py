"""src/common/geometry.py

Các hàm tính toán hình học 3D cho LiDAR Point Cloud và Bounding Box.
Tuân thủ chuẩn tọa độ nuScenes:
- Hệ trục: x hướng sang phải, y hướng về phía trước, z hướng lên trên (hoặc theo quy ước LIDAR_TOP).
- Cuboid: tâm (cx, cy, cz), kích thước (w, l, h), góc xoay yaw.
"""

from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np


def yaw_to_rotation_matrix(yaw_rad: float) -> np.ndarray:
    """Tạo ma trận xoay quanh trục Z trong mặt phẳng 2D/3D."""
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def transform_points_to_local(
    points_xyz: np.ndarray,
    center: np.ndarray,
    yaw_rad: float
) -> np.ndarray:
    """
    Chuyển đổi điểm LiDAR từ hệ tọa độ sensor sang hệ tọa độ cục bộ của cuboid:
    p_local = R^T * (p_sensor - center)
    """
    if len(points_xyz) == 0:
        return np.empty((0, 3), dtype=np.float64)
    translated = points_xyz - center
    r_mat = yaw_to_rotation_matrix(yaw_rad)
    return np.dot(translated, r_mat)  # r_mat^T * v = v * r_mat vì r_mat là trực giao


def filter_points_in_box(
    points_xyz: np.ndarray,
    center: np.ndarray,
    size_wlh: np.ndarray,
    yaw_rad: float,
    margin: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lọc các điểm nằm bên trong cuboid 3D (có thể mở rộng viền theo margin).
    Trả về:
    - inside_mask: boolean array (N,)
    - points_local: array (N, 3) tọa độ cục bộ theo [w, l, h]
    """
    if len(points_xyz) == 0:
        return np.zeros((0,), dtype=bool), np.empty((0, 3), dtype=np.float64)

    pts_local = transform_points_to_local(points_xyz, center, yaw_rad)
    w, l, h = size_wlh[0], size_wlh[1], size_wlh[2]

    half_w = (w / 2.0) + margin
    half_l = (l / 2.0) + margin
    half_h = (h / 2.0) + margin

    # Trong hệ cục bộ: trục x tương ứng w (hoặc l tùy quy ước, ở đây quy ước dx <= half_w, dy <= half_l, dz <= half_h)
    mask = (
        (np.abs(pts_local[:, 0]) <= half_w) &
        (np.abs(pts_local[:, 1]) <= half_l) &
        (np.abs(pts_local[:, 2]) <= half_h)
    )
    return mask, pts_local


def compute_cuboid_corners(
    center: np.ndarray,
    size_wlh: np.ndarray,
    yaw_rad: float
) -> np.ndarray:
    """
    Tính tọa độ 8 góc của cuboid 3D trong không gian thế giới.
    Thứ tự 8 góc: 4 góc đáy dưới (z - h/2), 4 góc đáy trên (z + h/2).
    """
    w, l, h = size_wlh[0], size_wlh[1], size_wlh[2]
    hw, hl, hh = w / 2.0, l / 2.0, h / 2.0

    # 8 góc trong hệ tọa độ cục bộ
    corners_local = np.array([
        [-hw, -hl, -hh],
        [ hw, -hl, -hh],
        [ hw,  hl, -hh],
        [-hw,  hl, -hh],
        [-hw, -hl,  hh],
        [ hw, -hl,  hh],
        [ hw,  hl,  hh],
        [-hw,  hl,  hh],
    ], dtype=np.float64)

    r_mat = yaw_to_rotation_matrix(yaw_rad)
    corners_world = np.dot(corners_local, r_mat.T) + center
    return corners_world
