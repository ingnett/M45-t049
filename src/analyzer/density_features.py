"""src/analyzer/density_features.py

Module trích xuất và tính toán đặc trưng mật độ điểm LiDAR (Point Density Features).
Dự án: M45 - LiDAR Point Density & Annotation Difficulty Analyzer
Dữ liệu: nuScenes / Synthetic LiDAR Point Clouds

Các đặc trưng chính:
- distance_m: Khoảng cách từ cảm biến LiDAR (LIDAR_TOP) tới tâm đối tượng (d = sqrt(x^2 + y^2 + z^2))
- point_count: Số điểm phản xạ LiDAR rơi vào trong bounding box 3D (N)
- box_volume_m3: Thể tích hộp 3D (V = w * l * h)
- volumetric_density: Mật độ thể tích (points / m^3 = N / V)
- surface_density: Mật độ diện tích bề mặt (points / m^2)
- beam_contact_estimate: Ước lượng số chùm tia LiDAR cắt qua đối tượng dựa trên FoV dọc (-30° đến +10°, 32 beams)
- distance_bin: Phân khoảng cự ly ([0, 10m), [10, 20m), [20, 30m), [30, 50m), >= 50m)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# Các khoảng cự ly chuẩn cho xe tự hành
DISTANCE_BINS = [
    (0.0, 10.0, "0-10m (Rất gần)"),
    (10.0, 20.0, "10-20m (Gần)"),
    (20.0, 30.0, "20-30m (Trung bình)"),
    (30.0, 50.0, "30-50m (Xa)"),
    (50.0, 120.0, "≥50m (Rất xa)"),
]


def classify_distance_bin(distance_m: float) -> str:
    """Phân loại cự ly thành nhãn khoảng chuẩn."""
    for low, high, label in DISTANCE_BINS:
        if low <= distance_m < high:
            return label
    return "≥50m (Rất xa)"


def compute_euclidean_distance(center: List[float] | Tuple[float, float, float]) -> float:
    """Tính khoảng cách Euclid từ gốc tọa độ cảm biến tới tâm đối tượng."""
    x, y, z = center[0], center[1], center[2]
    return float(math.sqrt(x * x + y * y + z * z))


def compute_cuboid_volume(size_wlh: List[float] | Tuple[float, float, float]) -> float:
    """Tính thể tích cuboid 3D (V = width * length * height)."""
    w, l, h = size_wlh[0], size_wlh[1], size_wlh[2]
    return float(max(1e-4, w * l * h))


def compute_cuboid_surface_area(size_wlh: List[float] | Tuple[float, float, float]) -> float:
    """Tính tổng diện tích bề mặt của cuboid 3D."""
    w, l, h = size_wlh[0], size_wlh[1], size_wlh[2]
    return float(2.0 * (w * l + l * h + h * w))


def estimate_theoretical_point_count(
    distance_m: float,
    class_name: str,
    reflectivity_factor: float = 0.8,
) -> int:
    """
    Ước lượng số điểm LiDAR lý thuyết theo mô hình vật lý suy giảm nghịch đảo bình phương khoảng cách:
    N ~ C * A_proj / (d^2) với cảm biến 32-beam LiDAR nuScenes.
    """
    d = max(2.0, distance_m)
    if "vehicle" in class_name.lower():
        # Kích thước xe chuẩn ~ 1.9m x 4.6m x 1.7m, diện tích mặt cắt ~ 5 m^2
        c_const = 18000.0
    else:
        # Người đi bộ ~ 0.7m x 0.7m x 1.7m, diện tích mặt cắt ~ 1.2 m^2
        c_const = 4200.0

    raw_count = (c_const / (d ** 2)) * reflectivity_factor
    # Thêm độ phân rã do chùm tia lọt qua hoặc phản xạ tán xạ
    pts = int(round(raw_count))
    return max(1, pts)


def extract_density_features_for_case(
    case_id: str,
    class_name: str,
    center_m: List[float],
    size_wlh_m: List[float],
    actual_point_count: Optional[int] = None,
    occlusion_ratio: float = 0.0,
) -> Dict[str, Any]:
    """
    Trích xuất toàn bộ chỉ số mật độ điểm cho 1 ca.
    """
    dist_m = compute_euclidean_distance(center_m)
    vol_m3 = compute_cuboid_volume(size_wlh_m)
    surf_m2 = compute_cuboid_surface_area(size_wlh_m)

    if actual_point_count is not None and actual_point_count >= 0:
        pts = actual_point_count
    else:
        # Nếu chưa có point cloud đếm thực, dùng mô hình suy giảm vật lý nuScenes
        theo_pts = estimate_theoretical_point_count(dist_m, class_name)
        # Giảm trừ theo độ che khuất
        pts = max(1, int(round(theo_pts * (1.0 - 0.75 * occlusion_ratio))))

    vol_density = float(pts / vol_m3)
    surf_density = float(pts / max(0.1, surf_m2))
    dist_bin = classify_distance_bin(dist_m)

    return {
        "case_id": case_id,
        "class_name": class_name,
        "distance_m": round(dist_m, 2),
        "distance_bin": dist_bin,
        "point_count": int(pts),
        "volume_m3": round(vol_m3, 3),
        "volumetric_density": round(vol_density, 3),
        "surface_density": round(surf_density, 3),
        "occlusion_ratio": round(occlusion_ratio, 2),
    }


def build_density_dataset(
    cases_df: pd.DataFrame,
    ground_truth_boxes: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Xây dựng bảng dữ liệu mật độ điểm cho danh sách các cases trong dự án.
    Nếu có ground_truth_boxes thật thì dùng, nếu không có thì sinh từ tham số chuẩn nuScenes.
    """
    records = []
    # Seed cố định để đảm bảo tính tái lập khoa học
    rng = np.random.RandomState(45)

    # Phân phối cự ly mô phỏng nuScenes cho các ca demo
    default_car_size = [1.95, 4.65, 1.72]
    default_ped_size = [0.72, 0.78, 1.70]

    for idx, row in cases_df.iterrows():
        cid = str(row["case_id"])
        cname = str(row.get("class_name", "vehicle.car"))
        is_car = "vehicle" in cname.lower()

        if ground_truth_boxes and cid in ground_truth_boxes:
            gt = ground_truth_boxes[cid]
            center = gt.get("center_m", [15.0, 3.0, -0.5])
            size = gt.get("size_wlh_m", default_car_size if is_car else default_ped_size)
            actual_pts = gt.get("point_count", None)
            occ = gt.get("occlusion_ratio", 0.0)
        else:
            # Sinh cự ly phân bố theo từng batch (gần tới xa) để tạo dải đa dạng cho phân tích
            # Batch B1: trung bình (12 - 35m)
            # Batch B2: xa và cực xa (25 - 65m)
            # Batch DEV / TEST: rải đều (8 - 55m)
            batch = str(row.get("batch_id", "B1")).upper()
            if "P" in batch:
                d_target = float(rng.uniform(10.0, 20.0))
            elif "B1" in batch:
                d_target = float(rng.uniform(12.0, 38.0))
            elif "B2" in batch:
                d_target = float(rng.uniform(22.0, 68.0))
            elif "DEV" in batch:
                d_target = float(rng.uniform(14.0, 48.0))
            else:  # TEST
                d_target = float(rng.uniform(15.0, 60.0))

            # Giả lập tọa độ sensor coordinate (LIDAR_TOP)
            angle = rng.uniform(-0.6, 0.6)  # góc quét phía trước
            x = d_target * math.cos(angle)
            y = d_target * math.sin(angle)
            z = float(rng.uniform(-1.2, 0.2))
            center = [x, y, z]

            # Kích thước kèm dao động nhẹ tự nhiên
            base_size = default_car_size if is_car else default_ped_size
            noise_s = rng.normal(0.0, 0.05, 3)
            size = [round(base_size[i] * (1.0 + noise_s[i]), 2) for i in range(3)]

            # Độ che khuất
            occ = float(rng.choice([0.0, 0.1, 0.25, 0.5, 0.7], p=[0.5, 0.2, 0.15, 0.1, 0.05]))
            actual_pts = None

        feat = extract_density_features_for_case(
            case_id=cid,
            class_name=cname,
            center_m=center,
            size_wlh_m=size,
            actual_point_count=actual_pts,
            occlusion_ratio=occ,
        )
        # Giữ lại các metadata gốc
        feat["split"] = row.get("split", "train")
        feat["batch_id"] = row.get("batch_id", "B1")
        feat["is_demo"] = row.get("is_demo", "True")
        records.append(feat)

    df_out = pd.DataFrame(records)
    return df_out
