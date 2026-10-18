"""src/hung/make_cvat_pack.py

Tạo gói dữ liệu dán nhãn 3D CVAT cho từng rater và từng batch.
Tuân thủ mục 2, mục 3, mục 5.3, mục 5.10, mục 6, mục 8.3 của tài liệu M45.

Quy tắc:
- Với mỗi rater (R1, R2, R3) và mỗi batch (P, B1, DEV, B2, TEST), tạo gói chứa đúng các ca được giao.
- Tên file điểm đặt theo task_code (ví dụ T-4821.pcd).
- Định dạng PCD 0.7 chuẩn CVAT 3D chấp nhận.
- Kèm hướng dẫn 1 trang (INSTRUCTIONS.txt).
- TUYỆT ĐỐI KHÔNG CÓ THÔNG TIN NÀO NGOÀI ĐIỂM VÀ CLASS (không có box gốc, không có số điểm N, không có score).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import zipfile
from typing import Dict, List, Tuple

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = REPO_ROOT / "data" / "work"
LABELS_DIR = REPO_ROOT / "data" / "labels"
CVAT_PACKS_DIR = REPO_ROOT / "data" / "cvat_packs"


def format_pcd_ascii(points: List[Tuple[float, float, float]]) -> str:
    """Tạo nội dung file PCD Version 0.7 dạng ASCII theo chuẩn Point Cloud Library / CVAT."""
    n = len(points)
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
    lines = header + [f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in points]
    return "\n".join(lines) + "\n"


def generate_synthetic_points_for_demo(case_id: str, class_name: str) -> List[Tuple[float, float, float]]:
    """Tạo điểm giả lập nếu chưa có file PCD thật từ nuScenes (theo mục 5.10)."""
    # Ca demo-01 chuẩn mục 5.10: tâm (10,0,0), size (2,4,2)
    # Ba điểm trong: (10,0,0), (12,0,0), (10,1,1)
    # Hai điểm ngoài: (12.01,0,0), (10,1.01,0)
    pts = [
        (10.0, 0.0, 0.0),
        (12.0, 0.0, 0.0),
        (10.0, 1.0, 1.0),
        (12.01, 0.0, 0.0),
        (10.0, 1.01, 0.0),
    ]
    # Bổ sung một số điểm bối cảnh quanh tâm trong vùng crop
    if "human" in class_name:
        pts = [
            (5.0, 1.0, 0.0),
            (5.2, 1.1, 0.5),
            (5.1, 0.9, 1.2),
            (6.5, 2.0, 0.0),
        ]
    return pts


def get_case_points(case_id: str, class_name: str) -> List[Tuple[float, float, float]]:
    """Lấy danh sách điểm xyz từ file PCD hoặc frames JSON nếu có, hoặc tạo điểm giả lập."""
    pcd_path = WORK_DIR / "pcd" / f"{case_id}.pcd"
    frame_path = WORK_DIR / "frames" / f"{case_id}.json"

    if pcd_path.exists():
        pts = []
        with open(pcd_path, "r", encoding="utf-8") as f:
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
                        pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
        return pts

    if frame_path.exists():
        with open(frame_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            raw_pts = data.get("points_xyz", [])
            return [(float(p[0]), float(p[1]), float(p[2])) for p in raw_pts]

    # Chưa có PCD hoặc JSON, sinh điểm demo
    return generate_synthetic_points_for_demo(case_id, class_name)


ONE_PAGE_INSTRUCTIONS = """================================================================================
HƯỚNG DẪN DÁN NHÃN 3D CUBOID TRÊN CVAT (CHO RATER R1, R2, R3)
Dự án: M45 - LiDAR Point Density & Annotation Difficulty Analyzer
================================================================================

1. NGUYÊN TẮC BẢO MẬT & ĐỘC LẬP (BẮT BUỘC):
   - Bạn chỉ nhận được danh sách mã nhiệm vụ (task_code) và lớp đối tượng (class).
   - Bạn KHÔNG được biết trước tọa độ hộp gốc, số điểm LiDAR (N), hay kết quả của rater khác.
   - Không trao đổi với thành viên khác về các mã ca trong lúc dán nhãn.
   - Không mở code, app hiển thị hay các file nhãn nội bộ trước khi hoàn tất nộp.

2. CÁCH NẠP DỮ LIỆU VÀO CVAT 3D:
   - Bước 1: Mở CVAT (giao diện web hoặc local container).
   - Bước 2: Tạo Task mới dạng '3D Point Cloud'.
   - Bước 3: Đặt Labels tương ứng:
             * 'vehicle.car' (cho ô tô)
             * 'human.pedestrian.adult' (cho người đi bộ)
   - Bước 4: Tải toàn bộ các file .pcd trong thư mục gói này vào task (hoặc tải trực tiếp file zip).
   - Bước 5: Nhấn 'Submit & Open' để bắt đầu công việc.

3. QUY TRÌNH THAO TÁC TRÊN MỖI KHUNG HÌNH (TASK_CODE):
   - Bước 1: Tra cứu file 'classes.csv' đính kèm để biết task_code này cần dán nhãn loại gì.
   - Bước 2: Bấm giờ dán nhãn (tối đa 180 giây/ca).
   - Bước 3: Chọn công cụ Cuboid (phím tắt N -> chọn label tương ứng).
   - Bước 4: Quan sát cụm điểm trên các cửa sổ hình chiếu:
             * Top View (chiếu từ trên xuống): căn tâm và chiều dài/rộng.
             * Side/Front View: căn chiều cao và vị trí tiếp đất.
             * 3D Perspective: xoay hướng (yaw) sao cho đầu xe/mặt người đúng hướng di chuyển.
   - Bước 5: Kết thúc và ghi nhận:
             * Nếu hoàn thành trong 180s: ghi status = 'completed', ghi thời gian duration_s.
             * Nếu quá 180s chưa xong: ghi status = 'timeout'.
             * Nếu bối cảnh quá thiếu thông tin: ghi status = 'insufficient_context'.
   - Bước 6: Đánh giá độ khó (1-5) và mức tự tin (1-5) theo tài liệu config/rubric_r1.md.
             Điền vào file ratings.csv của bạn.

4. XUẤT VÀ NỘP KẾT QUẢ:
   - Sau khi hoàn thành hết các ca trong gói:
     * Vào Actions -> Export Task Dataset -> Chọn định dạng 'Datumaro 1.0' hoặc 'CVAT for video 1.1'.
     * Đặt tên file export: export_{batch}_{rater}.zip (hoặc .json / .xml).
     * Nộp file export cùng file ratings.csv cho Hưng.

Cảm ơn bạn đã đóng góp vào tính chuẩn xác của nghiên cứu!
================================================================================
"""


def make_cvat_packs() -> None:
    assignments_path = LABELS_DIR / "assignments.csv"
    cases_path = WORK_DIR / "cases.csv"

    if not assignments_path.exists():
        print(f"[!] Không tìm thấy {assignments_path}. Hãy chạy make_assignments.py trước.")
        sys.exit(1)

    # Đọc cases để lấy class_name
    case_classes: Dict[str, str] = {}
    if cases_path.exists():
        with open(cases_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                case_classes[row["case_id"]] = row.get("class_name", "vehicle.car")

    # Đọc assignments
    assignments_by_pack: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    with open(assignments_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["batch_id"], row["rater_id"])
            assignments_by_pack.setdefault(key, []).append(row)

    CVAT_PACKS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[*] Bắt đầu tạo {len(assignments_by_pack)} gói CVAT 3D cho các rater và batch...")

    for (batch_id, rater_id), items in sorted(assignments_by_pack.items()):
        pack_name = f"pack_{batch_id}_{rater_id}"
        pack_dir = CVAT_PACKS_DIR / pack_name
        pack_dir.mkdir(parents=True, exist_ok=True)

        # 1. Tạo file hướng dẫn 1 trang
        with open(pack_dir / "INSTRUCTIONS.txt", "w", encoding="utf-8") as f:
            f.write(ONE_PAGE_INSTRUCTIONS)

        # 2. Tạo file danh sách class của các ca giao
        classes_csv = pack_dir / "classes.csv"
        with open(classes_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["task_code", "class_name"])
            for it in items:
                task_code = it["task_code"]
                case_id = it["case_id"]
                c_name = case_classes.get(case_id, "vehicle.car")
                writer.writerow([task_code, c_name])

        # 3. Tạo các file điểm .pcd mang tên task_code
        pcd_dir = pack_dir / "point_clouds"
        pcd_dir.mkdir(parents=True, exist_ok=True)

        for it in items:
            task_code = it["task_code"]
            case_id = it["case_id"]
            c_name = case_classes.get(case_id, "vehicle.car")

            pts = get_case_points(case_id, c_name)
            pcd_content = format_pcd_ascii(pts)

            pcd_file = pcd_dir / f"{task_code}.pcd"
            with open(pcd_file, "w", encoding="utf-8") as f:
                f.write(pcd_content)

        # 4. Đóng gói thành file ZIP tiện phân phối
        zip_path = CVAT_PACKS_DIR / f"{pack_name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(pack_dir / "INSTRUCTIONS.txt", arcname="INSTRUCTIONS.txt")
            zf.write(classes_csv, arcname="classes.csv")
            for pcd_f in pcd_dir.glob("*.pcd"):
                zf.write(pcd_f, arcname=f"point_clouds/{pcd_f.name}")

        print(f"  [+] Đã tạo gói {pack_name}.zip ({len(items)} ca giao)")

    print(f"[OK] Hoàn tất tạo toàn bộ các gói CVAT tại: {CVAT_PACKS_DIR}")


if __name__ == "__main__":
    make_cvat_packs()
