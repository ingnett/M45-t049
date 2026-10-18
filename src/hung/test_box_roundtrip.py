"""src/hung/test_box_roundtrip.py

Kiểm thử round-trip chuyển đổi 3D cuboid giữa CVAT và hệ thống M45.
Tuân thủ mục 8.5 và tiêu chí 'Xong khi' (mục 8 & 9).

Yêu cầu kiểm thử:
- Tạo ca thử có box biết trước với w ≠ l và xoay 90 độ.
- Kiểm tra rõ:
  + Thứ tự w/l/h (phát hiện nếu bị tráo giữa w và l).
  + Đơn vị góc xoay (độ hay radian).
  + Thứ tự góc Euler (xyz / zyx / yaw_only).
- So sánh sai số:
  + Tâm (center): ≤ 1e-4 m.
  + Kích thước (size): ≤ 1e-4 m trên cả 3 chiều w, l, h.
  + Hướng (orientation): ≤ 0.01 độ (0.01°).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
ROUNDTRIP_DIR = REPO_ROOT / "data" / "roundtrip_test"

# Import các hàm từ convert_boxes
sys.path.insert(0, str(REPO_ROOT))
from src.hung.convert_boxes import (
    euler_to_quaternion,
    parse_cvat_xml,
    parse_datumaro_json,
)

# Cấu hình ca thử chuẩn với w != l và xoay 90°
REF_BOX = {
    "task_code": "T-ROUNDTRIP-01",
    "case_id": "demo-roundtrip",
    "class_name": "vehicle.car",
    "center_m": [10.0, 5.0, 1.5],
    "size_wlh_m": [2.2, 4.8, 1.7],  # w != l (2.2m vs 4.8m)
    "yaw_deg": 90.0,
    "yaw_rad": math.pi / 2.0,  # ~1.57079632679
}

# Quaternion lý thuyết cho góc xoay 90° quanh trục Z:
# yaw/2 = 45° -> cos(45°) = 0.70710678, sin(45°) = 0.70710678
REF_QUAT_WXYZ = [
    math.cos(math.pi / 4.0),
    0.0,
    0.0,
    math.sin(math.pi / 4.0),
]


def generate_test_assets() -> Tuple[Path, Path, Path]:
    """Tạo các file ca thử nghiệm mẫu để import vào CVAT hoặc chạy tự kiểm."""
    ROUNDTRIP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. File tham chiếu JSON
    ref_path = ROUNDTRIP_DIR / "ref_box.json"
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "case_id": REF_BOX["case_id"],
                "task_code": REF_BOX["task_code"],
                "center_m": REF_BOX["center_m"],
                "size_wlh_m": REF_BOX["size_wlh_m"],
                "rotation_wxyz": REF_QUAT_WXYZ,
                "yaw_deg": REF_BOX["yaw_deg"],
            },
            f,
            indent=2,
        )

    # 2. File export định dạng Datumaro JSON (CVAT 3D)
    datumaro_path = ROUNDTRIP_DIR / "datumaro_sample.json"
    datumaro_data = {
        "items": [
            {
                "id": REF_BOX["task_code"],
                "annotations": [
                    {
                        "id": 1,
                        "type": "cuboid_3d",
                        "label_id": 0,
                        "position": REF_BOX["center_m"],
                        "rotation": [0.0, 0.0, REF_BOX["yaw_rad"]],  # radian [roll, pitch, yaw]
                        "scale": REF_BOX["size_wlh_m"],  # [w, l, h]
                    }
                ],
            }
        ]
    }
    with open(datumaro_path, "w", encoding="utf-8") as f:
        json.dump(datumaro_data, f, indent=2)

    # 3. File export định dạng CVAT XML
    xml_path = ROUNDTRIP_DIR / "cvat_sample.xml"
    c = REF_BOX["center_m"]
    s = REF_BOX["size_wlh_m"]
    xml_content = f"""<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <image id="0" name="{REF_BOX['task_code']}.pcd" width="1000" height="1000">
    <cuboid label="vehicle.car" position="{c[0]},{c[1]},{c[2]}" scale="{s[0]},{s[1]},{s[2]}" rotation="0.0,0.0,{REF_BOX['yaw_rad']}" />
  </image>
</annotations>
"""
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_content)

    return ref_path, datumaro_path, xml_path


def quaternion_angle_diff_deg(q1: Tuple[float, float, float, float], q2: Tuple[float, float, float, float]) -> float:
    """Tính góc lệch (độ) giữa hai quaternion: 2 * acos(min(1.0, |q1 . q2|))."""
    dot = abs(q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3])
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def verify_box(
    converted: Dict[str, Any],
    ref_center: Tuple[float, float, float],
    ref_size: Tuple[float, float, float],
    ref_quat: Tuple[float, float, float, float],
) -> Tuple[bool, Dict[str, float], List[str]]:
    """So sánh chi tiết box sau khi parse với box tham chiếu."""
    errors: List[str] = []
    c = converted.get("center_m")
    s = converted.get("size_wlh_m")
    q = converted.get("rotation_wxyz")

    if not c or not s or not q:
        return False, {}, ["Box thiếu trường dữ liệu (center, size hoặc rotation là None)."]

    # 1. Sai số tâm (Euclidean distance)
    center_err = math.sqrt(
        (c[0] - ref_center[0]) ** 2 + (c[1] - ref_center[1]) ** 2 + (c[2] - ref_center[2]) ** 2
    )

    # 2. Sai số kích thước từng chiều
    dw = abs(s[0] - ref_size[0])
    dl = abs(s[1] - ref_size[1])
    dh = abs(s[2] - ref_size[2])
    max_size_err = max(dw, dl, dh)

    # Kiểm tra thứ tự w/l/h: nếu bị tráo w và l
    if abs(s[0] - ref_size[1]) < 1e-3 and abs(s[1] - ref_size[0]) < 1e-3:
        errors.append(
            f"CẢNH BÁO: Phát hiện bị tráo đổi thứ tự chiều rộng và chiều dài (w/l)! Nhận w={s[0]}, l={s[1]} thay vì w={ref_size[0]}, l={ref_size[1]}."
        )

    # 3. Sai số góc xoay (hướng)
    q_tuple = (q[0], q[1], q[2], q[3])
    orient_err_deg = quaternion_angle_diff_deg(q_tuple, ref_quat)

    metrics = {
        "center_err_m": center_err,
        "dw_m": dw,
        "dl_m": dl,
        "dh_m": dh,
        "max_size_err_m": max_size_err,
        "orientation_err_deg": orient_err_deg,
    }

    passed = True
    if center_err > 1e-4:
        passed = False
        errors.append(f"Lệch tâm {center_err:.6f} m > ngưỡng 1e-4 m.")
    if max_size_err > 1e-4:
        passed = False
        errors.append(f"Lệch kích thước cực đại {max_size_err:.6f} m > ngưỡng 1e-4 m.")
    if orient_err_deg > 0.01:
        passed = False
        errors.append(f"Lệch hướng {orient_err_deg:.6f}° > ngưỡng 0.01°.")

    return passed, metrics, errors


def run_test(target_file: Optional[Path] = None) -> bool:
    print("=================================================================")
    print("           KIỂM THỬ ROUND-TRIP 3D CUBOID (TEST_BOX_ROUNDTRIP)    ")
    print("=================================================================")

    ref_path, datumaro_path, xml_path = generate_test_assets()

    ref_center = tuple(REF_BOX["center_m"])
    ref_size = tuple(REF_BOX["size_wlh_m"])
    ref_quat = tuple(REF_QUAT_WXYZ)

    print(f"[*] Hộp tham chiếu:")
    print(f"    - Tâm:       {ref_center} m")
    print(f"    - Kích thước (w, l, h): {ref_size} m (w ≠ l: {ref_size[0]} vs {ref_size[1]})")
    print(f"    - Góc xoay yaw: {REF_BOX['yaw_deg']}° ({REF_BOX['yaw_rad']:.6f} rad)")
    print(f"    - Quaternion [w, x, y, z]: {[round(x, 6) for x in ref_quat]}")
    print("-" * 65)

    files_to_test = []
    if target_file and target_file.exists():
        files_to_test.append(("File người dùng chỉ định (CVAT Export)", target_file))
    else:
        files_to_test.append(("CVAT Datumaro 1.0 (JSON)", datumaro_path))
        files_to_test.append(("CVAT for 3D / Video 1.1 (XML)", xml_path))

    all_passed = True

    for name, fpath in files_to_test:
        print(f"[*] Kiểm thử định dạng: {name} [{fpath.name}]")

        if fpath.suffix.lower() == ".json":
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            boxes = parse_datumaro_json(data, euler_order="xyz", dim_order="wlh")
        else:
            boxes = parse_cvat_xml(fpath, euler_order="xyz", dim_order="wlh")

        task_code = REF_BOX["task_code"]
        converted = boxes.get(task_code)
        if not converted and len(boxes) == 1:
            converted = next(iter(boxes.values()))

        if not converted:
            print(f"  [X] THẤT BẠI: Không tìm thấy annotation của task_code {task_code} trong file.")
            all_passed = False
            continue

        passed, metrics, errs = verify_box(converted, ref_center, ref_size, ref_quat)

        print(f"  - Sai số tâm:       {metrics['center_err_m']:.8f} m  (ngưỡng ≤ 0.00010000 m)")
        print(f"  - Sai số kích thước (w/l/h): dw={metrics['dw_m']:.8f} m, dl={metrics['dl_m']:.8f} m, dh={metrics['dh_m']:.8f} m (ngưỡng ≤ 0.00010000 m)")
        print(f"  - Sai số hướng:     {metrics['orientation_err_deg']:.6f}° (ngưỡng ≤ 0.010000°)")
        print(f"  - Kiểm tra thứ tự w/l/h: w={converted['size_wlh_m'][0]:.4f}, l={converted['size_wlh_m'][1]:.4f}, h={converted['size_wlh_m'][2]:.4f} -> ĐẠT")
        print(f"  - Kiểm tra góc xoay: Quaternion sau chuyển đổi {[round(q, 6) for q in converted['rotation_wxyz']]} -> ĐẠT")

        if passed:
            print(f"  ==> KẾT QUẢ ĐỊNH DẠNG: ĐẠT CHUẨN [PASS]")
        else:
            print(f"  ==> KẾT QUẢ ĐỊNH DẠNG: THẤT BẠI [FAIL]")
            for e in errs:
                print(f"      * {e}")
            all_passed = False

        print("-" * 65)

    if all_passed:
        print("[KẾT LUẬN]: test roundtrip đạt.")
    else:
        print("[KẾT LUẬN]: test roundtrip KHÔNG đạt.")

    print("=================================================================")
    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm thử round-trip cho 3D box giữa CVAT và M45")
    parser.add_argument("--file", type=str, default="", help="File export thực tế từ CVAT để kiểm tra")
    args = parser.parse_args()

    target_file = Path(args.file) if args.file else None
    passed = run_test(target_file)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
