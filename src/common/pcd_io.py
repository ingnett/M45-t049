"""src/common/pcd_io.py

Module đọc và ghi file Point Cloud PCD (Version 0.7) và Frames JSON.
Tương thích định dạng chuẩn CVAT 3D và CloudCompare.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


def read_pcd_ascii(file_path: Path | str) -> np.ndarray:
    """
    Đọc dữ liệu điểm xyz từ file PCD ASCII 0.7.
    Trả về: np.ndarray shape (N, 3).
    """
    path = Path(file_path)
    if not path.exists():
        return np.empty((0, 3), dtype=np.float64)

    pts: List[List[float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("DATA ascii"):
                in_data = True
                continue
            if in_data:
                parts = line.split()
                if len(parts) >= 3:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])

    if not pts:
        return np.empty((0, 3), dtype=np.float64)
    return np.array(pts, dtype=np.float64)


def write_pcd_ascii(file_path: Path | str, points_xyz: np.ndarray) -> None:
    """
    Ghi dữ liệu điểm xyz ra file PCD ASCII 0.7 chuẩn CVAT 3D.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(points_xyz)
    header = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        f"WIDTH {n}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {n}",
        "DATA ascii",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        for p in points_xyz:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")


def load_frame_points(
    case_id: str,
    work_dir: Path
) -> np.ndarray:
    """
    Tìm và đọc điểm của 1 case_id từ:
    1. data/work/pcd/{case_id}.pcd
    2. data/work/frames/{case_id}.json
    Nếu không thấy, trả về mảng rỗng (0, 3).
    """
    pcd_path = work_dir / "pcd" / f"{case_id}.pcd"
    if pcd_path.exists():
        pts = read_pcd_ascii(pcd_path)
        if len(pts) > 0:
            return pts

    frame_path = work_dir / "frames" / f"{case_id}.json"
    if frame_path.exists():
        try:
            with open(frame_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                raw_pts = data.get("points_xyz", [])
                if raw_pts:
                    return np.array(raw_pts, dtype=np.float64)
        except Exception:
            pass

    return np.empty((0, 3), dtype=np.float64)
