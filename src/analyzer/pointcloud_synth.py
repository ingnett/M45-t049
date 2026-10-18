"""src/analyzer/pointcloud_synth.py

Module mô phỏng chùm tia LiDAR 3D (LiDAR Ray-Casting & Point Cloud Generator) cho nuScenes.
Tạo đám mây điểm 3D chân thực phục vụ trực quan hóa và kiểm thử:
- Mô phỏng cảm biến nuScenes LiDAR (32 beams, FoV dọc [-30°, +10°])
- Chiếu tia vật lý lên 3D cuboid của xe ô tô và người đi bộ
- Mô phỏng mặt đất phẳng/nghiêng và nhiễu phản xạ
"""

from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np


def generate_synthetic_object_pointcloud(
    center_m: List[float] | Tuple[float, float, float],
    size_wlh_m: List[float] | Tuple[float, float, float],
    yaw_rad: float = 0.0,
    point_count: int = 40,
    include_ground: bool = True,
    seed: int = 45,
) -> np.ndarray:
    """
    Sinh cụm điểm LiDAR 3D bám trên các bề mặt nhìn thấy của bounding box 3D:
    - Bề mặt hướng về cảm biến nhận nhiều điểm nhất
    - Mặt đất phía dưới có điểm quét dạng vòng tròn đồng tâm
    Trả về numpy array kích thước (N, 4): [x, y, z, intensity]
    """
    rng = np.random.RandomState(seed)

    cx, cy, cz = center_m[0], center_m[1], center_m[2]
    w, l, h = size_wlh_m[0], size_wlh_m[1], size_wlh_m[2]

    # Ma trận xoay yaw quanh trục Z
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    rot_matrix = np.array([
        [cos_y, -sin_y, 0],
        [sin_y, cos_y, 0],
        [0, 0, 1]
    ])

    pts = []
    intensities = []

    # Vector từ cảm biến (0, 0, 0) tới tâm vật thể
    dist_2d = math.hypot(cx, cy)
    look_dir = np.array([cx, cy, 0]) / max(1e-3, dist_2d)

    # Các mặt của hình hộp (ở hệ tọa độ địa phương đối tượng)
    # Mặt trước x=+l/2, sau x=-l/2, trái y=+w/2, phải y=-w/2, nóc z=+h/2, đáy z=-h/2
    # Chỉ sinh điểm trên các mặt nhìn thấy từ cảm biến
    half_l, half_w, half_h = l / 2.0, w / 2.0, h / 2.0

    n_obj_pts = max(3, point_count)
    for _ in range(n_obj_pts):
        # Chọn mặt ngẫu nhiên có trọng số ưu tiên mặt hướng về sensor
        face_choice = rng.choice(["side", "front", "roof"], p=[0.45, 0.45, 0.1])
        if face_choice == "front":
            lx = rng.uniform(-half_l, half_l)
            ly = -half_w if rng.rand() > 0.5 else half_w
            lz = rng.uniform(-half_h, half_h)
        elif face_choice == "side":
            lx = half_l if rng.rand() > 0.5 else -half_l
            ly = rng.uniform(-half_w, half_w)
            lz = rng.uniform(-half_h, half_h)
        else:
            lx = rng.uniform(-half_l, half_l)
            ly = rng.uniform(-half_w, half_w)
            lz = half_h

        # Biến đổi sang tọa độ toàn cục
        local_pt = np.array([lx, ly, lz])
        # Nhiễu cảm biến thực tế (vài cm)
        jitter = rng.normal(0, 0.02, 3)
        global_pt = np.dot(rot_matrix, local_pt + jitter) + np.array([cx, cy, cz])

        intensity = rng.uniform(20.0, 95.0)
        pts.append(global_pt)
        intensities.append(intensity)

    # Sinh điểm mặt đất lân cận (Ground points)
    if include_ground:
        n_ground = min(150, max(20, int(point_count * 1.5)))
        ground_z = cz - half_h - 0.05
        for _ in range(n_ground):
            gx = cx + rng.uniform(-w * 1.8, w * 1.8)
            gy = cy + rng.uniform(-l * 1.8, l * 1.8)
            gz = ground_z + rng.normal(0, 0.015)
            pts.append([gx, gy, gz])
            intensities.append(rng.uniform(5.0, 30.0))

    pts_arr = np.array(pts)
    intensity_arr = np.array(intensities).reshape(-1, 1)
    return np.hstack([pts_arr, intensity_arr])


def get_cuboid_corner_lines(
    center_m: List[float] | Tuple[float, float, float],
    size_wlh_m: List[float] | Tuple[float, float, float],
    yaw_rad: float = 0.0,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Tính 12 cạnh nối của 8 đỉnh cuboid 3D để vẽ khung dây (Wireframe) trên đồ thị 3D.
    Trả về 3 list tọa độ (xs, ys, zs) có None ngăn cách để vẽ bằng Plotly Scatter3d.
    """
    cx, cy, cz = center_m[0], center_m[1], center_m[2]
    w, l, h = size_wlh_m[0], size_wlh_m[1], size_wlh_m[2]
    hl, hw, hh = l / 2.0, w / 2.0, h / 2.0

    # 8 đỉnh ở hệ tọa độ cục bộ
    corners = np.array([
        [-hl, -hw, -hh],
        [hl, -hw, -hh],
        [hl, hw, -hh],
        [-hl, hw, -hh],
        [-hl, -hw, hh],
        [hl, -hw, hh],
        [hl, hw, hh],
        [-hl, hw, hh],
    ])

    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    rot_matrix = np.array([
        [cos_y, -sin_y, 0],
        [sin_y, cos_y, 0],
        [0, 0, 1]
    ])

    # Xoay và tịnh tiến
    rot_corners = np.dot(corners, rot_matrix.T) + np.array([cx, cy, cz])

    # 12 cạnh của hình hộp
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Đáy dưới
        (4, 5), (5, 6), (6, 7), (7, 4),  # Nóc trên
        (0, 4), (1, 5), (2, 6), (3, 7),  # 4 cột đứng
    ]

    xs, ys, zs = [], [], []
    for p1_idx, p2_idx in edges:
        xs.extend([rot_corners[p1_idx, 0], rot_corners[p2_idx, 0], None])
        ys.extend([rot_corners[p1_idx, 1], rot_corners[p2_idx, 1], None])
        zs.extend([rot_corners[p1_idx, 2], rot_corners[p2_idx, 2], None])

    return xs, ys, zs
