"""src/hung/convert_boxes.py

Chuyển đổi bounding box 3D xuất từ CVAT về file human_boxes.json theo chuẩn nuScenes/M45.
Tuân thủ mục 2 (luật cứng 6), mục 5.7, mục 8.5 của tài liệu M45.

Chuẩn đầu ra của human_boxes.json (mục 5.7):
[
  {
    "case_id": "c001",
    "rater_id": "R1",
    "coord_frame": "LIDAR_TOP",
    "center_m": [x, y, z],
    "size_wlh_m": [w, l, h],
    "rotation_wxyz": [w, x, y, z],
    "status": "completed"
  }
]
Quy tắc: Thiếu box thì null; không tạo box kích thước 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_DIR = REPO_ROOT / "data" / "labels"


def euler_to_quaternion(
    roll: float,
    pitch: float,
    yaw: float,
    order: str = "xyz",
    is_degrees: bool = False,
) -> Tuple[float, float, float, float]:
    """Chuyển đổi góc Euler sang Quaternion [w, x, y, z].
    
    Hỗ trợ thứ tự góc Euler:
    - 'xyz': Roll (X), Pitch (Y), Yaw (Z) - chuẩn phổ biến trong Three.js/CVAT 3D.
    - 'zyx': Yaw (Z), Pitch (Y), Roll (X) - chuẩn hàng không/robotics.
    """
    if is_degrees:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    if order.lower() == "xyz":
        # R = Rx * Ry * Rz
        w = cr * cp * cy - sr * sp * sy
        x = sr * cp * cy + cr * sp * sy
        y = cr * sp * cy - sr * cp * sy
        z = cr * cp * sy + sr * sp * cy
    elif order.lower() == "zyx":
        # R = Rz * Ry * Rx
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = cy * sp * cr + sy * cp * sr
        z = sy * cp * cr - cy * sp * sr
    elif order.lower() == "yaw_only":
        # Chỉ xoay quanh trục Z
        w = cy
        x = 0.0
        y = 0.0
        z = sy
    else:
        raise ValueError(f"Thứ tự góc Euler không được hỗ trợ: {order}")

    # Chuẩn hóa quaternion
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm > 1e-12:
        w /= norm
        x /= norm
        y /= norm
        z /= norm
    else:
        w, x, y, z = 1.0, 0.0, 0.0, 0.0

    return (w, x, y, z)


def quaternion_to_yaw(w: float, x: float, y: float, z: float) -> float:
    """Trích xuất góc yaw (radian) quanh trục Z từ Quaternion [w, x, y, z]."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def parse_datumaro_json(
    data: Dict[str, Any],
    euler_order: str = "xyz",
    force_degrees: Optional[bool] = None,
    dim_order: str = "wlh",
) -> Dict[str, Dict[str, Any]]:
    """Phân tích annotation 3D từ file JSON định dạng Datumaro 1.0 của CVAT."""
    boxes: Dict[str, Dict[str, Any]] = {}
    items = data.get("items", [])

    for item in items:
        item_id = str(item.get("id", "")).strip()
        annotations = item.get("annotations", [])
        cuboid = None
        for ann in annotations:
            if ann.get("type") in ("cuboid_3d", "cuboid3d", "cuboid"):
                cuboid = ann
                break

        if not cuboid:
            boxes[item_id] = {
                "center_m": None,
                "size_wlh_m": None,
                "rotation_wxyz": None,
                "status": "missing_box",
            }
            continue

        pos = cuboid.get("position", [0.0, 0.0, 0.0])
        rot = cuboid.get("rotation", [0.0, 0.0, 0.0])
        scale = cuboid.get("scale", [0.0, 0.0, 0.0])

        # Xác định đơn vị góc: độ hay radian
        max_rot_abs = max(abs(float(r)) for r in rot) if rot else 0.0
        is_deg = force_degrees if force_degrees is not None else (max_rot_abs > 6.284)

        if len(rot) == 3:
            rx, ry, rz = float(rot[0]), float(rot[1]), float(rot[2])
            qw, qx, qy, qz = euler_to_quaternion(rx, ry, rz, order=euler_order, is_degrees=is_deg)
        elif len(rot) == 4:
            # Đã là quaternion [w, x, y, z] hoặc [x, y, z, w]
            qw, qx, qy, qz = float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])
        elif len(rot) == 1:
            qw, qx, qy, qz = euler_to_quaternion(0.0, 0.0, float(rot[0]), order="yaw_only", is_degrees=is_deg)
        else:
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

        # Ánh xạ thứ tự kích thước
        sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
        if dim_order == "wlh":
            w, l, h = sx, sy, sz
        elif dim_order == "lwh":
            l, w, h = sx, sy, sz
        elif dim_order == "hwl":
            h, w, l = sx, sy, sz
        else:
            w, l, h = sx, sy, sz

        center = [float(pos[0]), float(pos[1]), float(pos[2])]
        size = [w, l, h]
        rotation = [qw, qx, qy, qz]

        boxes[item_id] = {
            "center_m": center,
            "size_wlh_m": size,
            "rotation_wxyz": rotation,
            "status": "completed",
        }

    return boxes


def parse_cvat_xml(
    xml_path: Path,
    euler_order: str = "xyz",
    force_degrees: Optional[bool] = None,
    dim_order: str = "wlh",
) -> Dict[str, Dict[str, Any]]:
    """Phân tích annotation 3D từ file XML của CVAT (CVAT for 3D/video 1.1)."""
    boxes: Dict[str, Dict[str, Any]] = {}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Tìm các thẻ image hoặc track/cuboid
    for elem in root.findall(".//image") + root.findall(".//frame"):
        name = elem.get("name", "") or elem.get("id", "")
        # Lấy tên không kèm phần mở rộng
        task_key = Path(name).stem if name else ""

        cuboid = elem.find(".//cuboid")
        if cuboid is None:
            boxes[task_key] = {
                "center_m": None,
                "size_wlh_m": None,
                "rotation_wxyz": None,
                "status": "missing_box",
            }
            continue

        # Các thuộc tính vị trí trong XML
        pos_str = cuboid.get("position", "")
        scale_str = cuboid.get("scale", "")
        rot_str = cuboid.get("rotation", "")

        if pos_str and scale_str:
            pos = [float(p) for p in pos_str.split(",")]
            scale = [float(s) for s in scale_str.split(",")]
            rot = [float(r) for r in rot_str.split(",")] if rot_str else [0.0, 0.0, 0.0]
        else:
            # Một số bản ghi CVAT XML lưu pos_x, pos_y, ...
            pos = [
                float(cuboid.get("pos_x", 0.0)),
                float(cuboid.get("pos_y", 0.0)),
                float(cuboid.get("pos_z", 0.0)),
            ]
            scale = [
                float(cuboid.get("scale_x", 0.0)),
                float(cuboid.get("scale_y", 0.0)),
                float(cuboid.get("scale_z", 0.0)),
            ]
            rot = [
                float(cuboid.get("rot_x", 0.0)),
                float(cuboid.get("rot_y", 0.0)),
                float(cuboid.get("rot_z", 0.0)),
            ]

        max_rot_abs = max(abs(float(r)) for r in rot) if rot else 0.0
        is_deg = force_degrees if force_degrees is not None else (max_rot_abs > 6.284)
        qw, qx, qy, qz = euler_to_quaternion(rot[0], rot[1], rot[2], order=euler_order, is_degrees=is_deg)

        sx, sy, sz = scale[0], scale[1], scale[2]
        if dim_order == "wlh":
            w, l, h = sx, sy, sz
        elif dim_order == "lwh":
            l, w, h = sx, sy, sz
        else:
            w, l, h = sx, sy, sz

        boxes[task_key] = {
            "center_m": [pos[0], pos[1], pos[2]],
            "size_wlh_m": [w, l, h],
            "rotation_wxyz": [qw, qx, qy, qz],
            "status": "completed",
        }

    return boxes


def build_human_boxes_json(
    assignments: List[Dict[str, str]],
    parsed_boxes_by_task: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Gộp các hộp của các rater theo hợp đồng human_boxes.json (mục 5.7)."""
    human_boxes = []

    for a in assignments:
        case_id = a["case_id"]
        task_code = a["task_code"]
        rater_id = a["rater_id"]

        box_info = parsed_boxes_by_task.get(task_code)
        if not box_info or box_info.get("center_m") is None:
            # Thiếu box thì null; không tạo box kích thước 0 (mục 5.7)
            human_boxes.append({
                "case_id": case_id,
                "rater_id": rater_id,
                "coord_frame": "LIDAR_TOP",
                "center_m": None,
                "size_wlh_m": None,
                "rotation_wxyz": None,
                "status": box_info.get("status", "missing_box") if box_info else "missing_box",
            })
        else:
            human_boxes.append({
                "case_id": case_id,
                "rater_id": rater_id,
                "coord_frame": "LIDAR_TOP",
                "center_m": [round(c, 4) for c in box_info["center_m"]],
                "size_wlh_m": [round(s, 4) for s in box_info["size_wlh_m"]],
                "rotation_wxyz": [round(q, 6) for q in box_info["rotation_wxyz"]],
                "status": box_info.get("status", "completed"),
            })

    return human_boxes


def main() -> None:
    parser = argparse.ArgumentParser(description="Chuyển đổi bounding box từ CVAT về human_boxes.json")
    parser.add_argument("--cvat_file", type=str, required=False, default="", help="File export từ CVAT (JSON hoặc XML)")
    parser.add_argument("--assignments", type=str, default="", help="File assignments.csv")
    parser.add_argument("--output", type=str, default="", help="Đường dẫn xuất human_boxes.json")
    parser.add_argument("--euler_order", type=str, default="xyz", help="Thứ tự góc Euler: xyz hoặc zyx")
    parser.add_argument("--dim_order", type=str, default="wlh", help="Thứ tự kích thước: wlh hoặc lwh")
    parser.add_argument("--degrees", action="store_true", help="Buộc góc xoay là độ thay vì radian")
    args = parser.parse_args()

    assignments_path = Path(args.assignments) if args.assignments else LABELS_DIR / "assignments.csv"
    output_path = Path(args.output) if args.output else LABELS_DIR / "human_boxes.json"

    assignments = []
    if assignments_path.exists():
        with open(assignments_path, "r", encoding="utf-8") as f:
            assignments = list(csv.DictReader(f))

    parsed_boxes: Dict[str, Dict[str, Any]] = {}
    if args.cvat_file:
        cvat_file = Path(args.cvat_file)
        if cvat_file.suffix.lower() == ".json":
            with open(cvat_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                parsed_boxes = parse_datumaro_json(
                    data,
                    euler_order=args.euler_order,
                    force_degrees=True if args.degrees else None,
                    dim_order=args.dim_order,
                )
        elif cvat_file.suffix.lower() == ".xml":
            parsed_boxes = parse_cvat_xml(
                cvat_file,
                euler_order=args.euler_order,
                force_degrees=True if args.degrees else None,
                dim_order=args.dim_order,
            )

    human_boxes = build_human_boxes_json(assignments, parsed_boxes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(human_boxes, f, indent=2, ensure_ascii=False)

    print(f"[OK] Đã xuất {len(human_boxes)} mục vào: {output_path}")


if __name__ == "__main__":
    main()
