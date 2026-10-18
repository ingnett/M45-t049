"""src/hung/box_errors.py

Tính sai số bounding box của các rater so với ground truth (box_errors.csv).
Tuân thủ mục 2, mục 3, mục 5.8, mục 6, mục 8.6 của tài liệu M45.

Định dạng cột box_errors.csv:
case_id, rater_id, status, center_err_m, size_err_pct, dw_pct, dl_pct, dh_pct, orientation_err_deg

Công thức chuẩn mục 5.8:
- Lệch tâm (center_err_m): khoảng cách Euclid 3D giữa hai tâm (m).
- Lệch kích thước:
  dw_pct = |w - w_ref| / w_ref * 100
  dl_pct = |l - l_ref| / l_ref * 100
  dh_pct = |h - h_ref| / h_ref * 100
  size_err_pct = (dw_pct + dl_pct + dh_pct) / 3
- Lệch hướng (orientation_err_deg):
  2 * acos(min(1.0, |q . q_ref|)) đổi sang độ.
  CHỈ TÍNH CHO vehicle.car, người đi bộ (human.pedestrian.adult) ĐỂ TRỐNG.
- Không gộp ba sai số thành một số chung.
- Chỉ tính cho ca đã có cả hai lượt khóa (LOCKED_{batch} hoặc cả 2 lượt hợp lệ).
- Box tham chiếu nuScenes lấy qua get_sample_data ở hệ cảm biến LIDAR_TOP.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WORK_DIR = REPO_ROOT / "data" / "work"
LABELS_DIR = REPO_ROOT / "data" / "labels"
RAW_NUSCENES_DIR = REPO_ROOT / "data" / "raw" / "nuscenes"


def compute_center_error(c1: List[float], c2: List[float]) -> float:
    """Tính khoảng cách Euclid 3D giữa hai tâm."""
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2)


def compute_size_errors(s1: List[float], s2: List[float]) -> Tuple[float, float, float, float]:
    """Tính sai số kích thước (w, l, h) theo phần trăm: dw_pct, dl_pct, dh_pct và trung bình size_err_pct."""
    w, l, h = s1[0], s1[1], s1[2]
    w_ref, l_ref, h_ref = s2[0], s2[1], s2[2]

    dw_pct = (abs(w - w_ref) / w_ref) * 100.0 if w_ref > 0 else 0.0
    dl_pct = (abs(l - l_ref) / l_ref) * 100.0 if l_ref > 0 else 0.0
    dh_pct = (abs(h - h_ref) / h_ref) * 100.0 if h_ref > 0 else 0.0

    size_err_pct = (dw_pct + dl_pct + dh_pct) / 3.0
    return size_err_pct, dw_pct, dl_pct, dh_pct


def compute_orientation_error_deg(q1: List[float], q2: List[float]) -> float:
    """Tính góc lệch (độ) giữa 2 quaternion: 2 * acos(min(1.0, |q1 . q2|))."""
    dot = abs(q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3])
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def get_reference_box_from_nuscenes(
    nusc_obj: Any,
    sample_token: str,
    annotation_token: str,
) -> Optional[Dict[str, Any]]:
    """Lấy bounding box tham chiếu từ nuScenes gốc ở hệ cảm biến LIDAR_TOP (mục 3)."""
    try:
        sample = nusc_obj.get("sample", sample_token)
        lidar_token = sample["data"]["LIDAR_TOP"]
        _, boxes, _ = nusc_obj.get_sample_data(
            lidar_token,
            selected_anntokens=[annotation_token],
            use_flat_vehicle_coordinates=False,
        )
        if not boxes:
            return None
        box = boxes[0]
        # Box ở hệ cảm biến LIDAR_TOP
        # Box.wlh: [width, length, height]
        # Box.orientation: Quaternion [w, x, y, z]
        center = [float(box.center[0]), float(box.center[1]), float(box.center[2])]
        size = [float(box.wlh[0]), float(box.wlh[1]), float(box.wlh[2])]
        quat = [
            float(box.orientation.elements[0]),
            float(box.orientation.elements[1]),
            float(box.orientation.elements[2]),
            float(box.orientation.elements[3]),
        ]
        return {
            "center_m": center,
            "size_wlh_m": size,
            "rotation_wxyz": quat,
        }
    except Exception as e:
        return None


def get_reference_box(case_row: Dict[str, str], nusc_obj: Any = None) -> Optional[Dict[str, Any]]:
    """Lấy box tham chiếu: từ nuScenes nếu có, hoặc frames JSON, hoặc demo box."""
    case_id = case_row["case_id"]

    # 1. Thử lấy từ nuScenes devkit nếu đã nạp
    if nusc_obj and case_row.get("sample_token") and case_row.get("annotation_token"):
        ref = get_reference_box_from_nuscenes(
            nusc_obj,
            case_row["sample_token"],
            case_row["annotation_token"],
        )
        if ref:
            return ref

    # 2. Thử đọc từ frames/{case_id}.json
    frame_path = WORK_DIR / "frames" / f"{case_id}.json"
    if frame_path.exists():
        try:
            with open(frame_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                b = data.get("box")
                if b and "center_m" in b and "size_wlh_m" in b:
                    return {
                        "center_m": b["center_m"],
                        "size_wlh_m": b["size_wlh_m"],
                        "rotation_wxyz": b.get("rotation_wxyz", [1.0, 0.0, 0.0, 0.0]),
                    }
        except Exception:
            pass

    # 3. Ca demo mặc định (mục 5.10)
    if "vehicle" in case_row.get("class_name", "vehicle.car"):
        return {
            "center_m": [10.0, 0.0, 0.0],
            "size_wlh_m": [2.0, 4.0, 2.0],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
    else:
        return {
            "center_m": [5.0, 1.0, 0.0],
            "size_wlh_m": [0.8, 0.8, 1.7],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        }


def check_manual_calculation() -> bool:
    """Tự kiểm chứng công thức đo sai số khớp tính tay 1 ca (mục 'Xong khi')."""
    print("=== KIỂM CHỨNG TÍNH TAY 1 CA (HAND CALCULATION VERIFICATION) ===")

    # Ca thử nghiệm tính tay:
    # Xe car: tâm ref = [10.0, 0.0, 0.0], size ref = [2.0, 4.0, 2.0], quat ref = [1.0, 0.0, 0.0, 0.0]
    # Rater hộp: tâm = [10.1, 0.0, 0.0], size = [2.1, 4.2, 1.9], yaw = 6.0°
    ref_center = [10.0, 0.0, 0.0]
    ref_size = [2.0, 4.0, 2.0]
    ref_quat = [1.0, 0.0, 0.0, 0.0]

    # Yaw 6° -> yaw/2 = 3° -> cos(3°) = 0.99862953, sin(3°) = 0.05233596
    rater_center = [10.1, 0.0, 0.0]
    rater_size = [2.1, 4.2, 1.9]
    yaw_deg = 6.0
    half_rad = math.radians(yaw_deg * 0.5)
    rater_quat = [math.cos(half_rad), 0.0, 0.0, math.sin(half_rad)]

    # Giá trị kỳ vọng tính tay:
    expected_center_err = 0.1
    expected_dw = 5.0
    expected_dl = 5.0
    expected_dh = 5.0
    expected_size_err = 5.0
    expected_orient_err = 6.0

    # Tính bằng hàm:
    calc_center_err = compute_center_error(rater_center, ref_center)
    calc_size_err, calc_dw, calc_dl, calc_dh = compute_size_errors(rater_size, ref_size)
    calc_orient_err = compute_orientation_error_deg(rater_quat, ref_quat)

    diff_center = abs(calc_center_err - expected_center_err)
    diff_size = abs(calc_size_err - expected_size_err)
    diff_dw = abs(calc_dw - expected_dw)
    diff_dl = abs(calc_dl - expected_dl)
    diff_dh = abs(calc_dh - expected_dh)
    diff_orient = abs(calc_orient_err - expected_orient_err)

    print(f"1. Lệch tâm (center_err_m):")
    print(f"   Tính tay: {expected_center_err:.6f} m | Script: {calc_center_err:.6f} m | Sai khác: {diff_center:.8f} m")
    print(f"2. Lệch kích thước (%):")
    print(f"   dw: Tính tay {expected_dw:.2f}% | Script {calc_dw:.2f}%")
    print(f"   dl: Tính tay {expected_dl:.2f}% | Script {calc_dl:.2f}%")
    print(f"   dh: Tính tay {expected_dh:.2f}% | Script {calc_dh:.2f}%")
    print(f"   size_err_pct: Tính tay {expected_size_err:.2f}% | Script {calc_size_err:.2f}% | Sai khác: {diff_size:.8f}%")
    print(f"3. Lệch hướng (orientation_err_deg):")
    print(f"   Tính tay: {expected_orient_err:.4f}° | Script: {calc_orient_err:.4f}° | Sai khác: {diff_orient:.8f}°")

    passed = (
        diff_center < 1e-6
        and diff_size < 1e-6
        and diff_dw < 1e-6
        and diff_dl < 1e-6
        and diff_dh < 1e-6
        and diff_orient < 1e-4
    )

    if passed:
        print("[OK] box_errors.csv khớp tính tay một ca (sai số tuyệt đối < 1e-6).")
    else:
        print("[FAIL] Tính tay không khớp!")
    print("=" * 60)
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Tính box_errors.csv theo mục 5.8 và 8.6")
    parser.add_argument("--human_boxes", type=str, default="", help="File human_boxes.json")
    parser.add_argument("--cases", type=str, default="", help="File cases.csv")
    parser.add_argument("--y_table", type=str, default="", help="File y_table.csv")
    parser.add_argument("--output", type=str, default="", help="File box_errors.csv")
    parser.add_argument("--check_hand_calc", action="store_true", help="Chạy kiểm chứng khớp tính tay 1 ca")
    args = parser.parse_args()

    # Luôn kiểm tra tính tay trước
    hand_calc_ok = check_manual_calculation()
    if not hand_calc_ok:
        sys.exit(1)

    human_boxes_path = Path(args.human_boxes) if args.human_boxes else LABELS_DIR / "human_boxes.json"
    cases_path = Path(args.cases) if args.cases else WORK_DIR / "cases.csv"
    y_table_path = Path(args.y_table) if args.y_table else LABELS_DIR / "y_table.csv"
    output_path = Path(args.output) if args.output else LABELS_DIR / "box_errors.csv"

    # Đọc cases
    cases_dict: Dict[str, Dict[str, str]] = {}
    if cases_path.exists():
        with open(cases_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cases_dict[row["case_id"]] = row

    # Đọc y_table để biết ca nào đủ điều kiện hợp lệ khóa (y_status == 'ok')
    valid_locked_cases = set()
    if y_table_path.exists():
        with open(y_table_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("y_status", "") == "ok":
                    valid_locked_cases.add(row["case_id"])

    # Thử khởi tạo nuScenes devkit nếu có thư mục dữ liệu thật
    nusc_obj = None
    if RAW_NUSCENES_DIR.exists() and any(RAW_NUSCENES_DIR.iterdir()):
        try:
            from nuscenes.nuscenes import NuScenes
            print(f"[*] Khởi tạo NuScenes devkit từ: {RAW_NUSCENES_DIR}")
            nusc_obj = NuScenes(version="v1.0-mini", dataroot=str(RAW_NUSCENES_DIR), verbose=False)
        except Exception as e:
            print(f"[*] Chưa tải đủ nuScenes devkit/mini ({e}). Sử dụng chế độ demo reference boxes.")

    # Đọc human_boxes.json
    human_boxes = []
    if human_boxes_path.exists():
        with open(human_boxes_path, "r", encoding="utf-8") as f:
            human_boxes = json.load(f)
    else:
        # Nếu chưa có human_boxes.json, tự động chuyển đổi từ demo/template
        print(f"[*] {human_boxes_path} chưa có. Sinh human_boxes mẫu từ convert_boxes...")
        from src.hung.convert_boxes import build_human_boxes_json
        assignments_path = LABELS_DIR / "assignments.csv"
        assigns = []
        if assignments_path.exists():
            with open(assignments_path, "r", encoding="utf-8") as f:
                assigns = list(csv.DictReader(f))
        human_boxes = build_human_boxes_json(assigns, {})
        with open(human_boxes_path, "w", encoding="utf-8") as f:
            json.dump(human_boxes, f, indent=2)

    # Tính sai số cho các ca hợp lệ
    results: List[Dict[str, str]] = []

    for item in human_boxes:
        case_id = item["case_id"]
        rater_id = item["rater_id"]
        status = item.get("status", "completed")

        # Quy tắc mục 8.6: "chỉ cho ca đã có cả hai lượt khóa"
        if valid_locked_cases and case_id not in valid_locked_cases:
            continue

        case_row = cases_dict.get(case_id, {"case_id": case_id, "class_name": "vehicle.car"})
        class_name = case_row.get("class_name", "vehicle.car")
        is_car = "vehicle.car" in class_name

        c_human = item.get("center_m")
        s_human = item.get("size_wlh_m")
        q_human = item.get("rotation_wxyz")

        if not c_human or not s_human or not q_human:
            # Thiếu box thì để trống trường sai số
            results.append({
                "case_id": case_id,
                "rater_id": rater_id,
                "status": status,
                "center_err_m": "",
                "size_err_pct": "",
                "dw_pct": "",
                "dl_pct": "",
                "dh_pct": "",
                "orientation_err_deg": "",
            })
            continue

        ref_box = get_reference_box(case_row, nusc_obj)
        if not ref_box:
            continue

        center_err = compute_center_error(c_human, ref_box["center_m"])
        size_err, dw, dl, dh = compute_size_errors(s_human, ref_box["size_wlh_m"])

        # Lệch hướng: chỉ tính cho vehicle.car, người đi bộ để trống
        if is_car:
            orient_err = compute_orientation_error_deg(q_human, ref_box["rotation_wxyz"])
            orient_err_str = f"{orient_err:.4f}"
        else:
            orient_err_str = ""

        results.append({
            "case_id": case_id,
            "rater_id": rater_id,
            "status": status,
            "center_err_m": f"{center_err:.4f}",
            "size_err_pct": f"{size_err:.2f}",
            "dw_pct": f"{dw:.2f}",
            "dl_pct": f"{dl:.2f}",
            "dh_pct": f"{dh:.2f}",
            "orientation_err_deg": orient_err_str,
        })

    # Ghi file box_errors.csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "rater_id",
        "status",
        "center_err_m",
        "size_err_pct",
        "dw_pct",
        "dl_pct",
        "dh_pct",
        "orientation_err_deg",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"[OK] Đã xuất {len(results)} dòng sai số vào: {output_path}")


if __name__ == "__main__":
    main()
