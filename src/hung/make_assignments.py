"""src/hung/make_assignments.py

Phân chia ca dán nhãn độc lập (double-blind) cho 3 rater (R1, R2, R3).
Tuân thủ mục 1, mục 2, mục 5.4, mục 6, mục 8.2 của tài liệu M45.

Quy tắc:
- Mỗi ca đúng 2 rater khác nhau trong R1, R2, R3.
- Các cặp AB (R1,R2), BC (R2,R3), AC (R1,R3) luân phiên và cân bằng trong từng tập.
- Mỗi rater khoảng 19 lượt trên 28 ca chính (tổng 56 lượt).
- Thứ tự và task_code ngẫu nhiên dùng seed 45.
- Ghi vào data/labels/assignments.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple

# Đảm bảo UTF-8 cho stdout/stderr trên Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Hằng số chuẩn theo quy ước M45
SEED = 45
RATERS = ["R1", "R2", "R3"]
PAIRS: List[Tuple[str, str]] = [("R1", "R2"), ("R2", "R3"), ("R1", "R3")]  # AB, BC, AC

# Thư mục gốc repo
REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = REPO_ROOT / "data" / "work"
LABELS_DIR = REPO_ROOT / "data" / "labels"


def generate_demo_cases() -> List[Dict[str, str]]:
    """Tạo danh sách ca giả lập theo mục 1 và mục 5.10 nếu chưa có cases.csv thật."""
    cases = []

    # 2 ca Pilot
    for i in range(1, 3):
        cases.append({
            "case_id": f"demo-p{i:02d}",
            "split": "pilot",
            "batch_id": "P",
            "class_name": "vehicle.car" if i == 1 else "human.pedestrian.adult",
            "is_demo": "True",
        })

    # 8 ca B1: 4 xe, 4 người đi bộ
    for i in range(1, 9):
        cases.append({
            "case_id": f"demo-b1-{i:02d}",
            "split": "train",
            "batch_id": "B1",
            "class_name": "vehicle.car" if i <= 4 else "human.pedestrian.adult",
            "is_demo": "True",
        })

    # 4 ca DEV: 2 xe, 2 người đi bộ
    for i in range(1, 5):
        cases.append({
            "case_id": f"demo-dev-{i:02d}",
            "split": "dev",
            "batch_id": "DEV",
            "class_name": "vehicle.car" if i <= 2 else "human.pedestrian.adult",
            "is_demo": "True",
        })

    # 8 ca B2: 4 ca lấp khoảng trống + 4 ca ngẫu nhiên
    for i in range(1, 9):
        cases.append({
            "case_id": f"demo-b2-{i:02d}",
            "split": "train",
            "batch_id": "B2",
            "class_name": "vehicle.car" if i % 2 == 1 else "human.pedestrian.adult",
            "is_demo": "True",
        })

    # 8 ca TEST: 4 xe, 4 người đi bộ
    for i in range(1, 9):
        cases.append({
            "case_id": f"demo-test-{i:02d}",
            "split": "test",
            "batch_id": "TEST",
            "class_name": "vehicle.car" if i <= 4 else "human.pedestrian.adult",
            "is_demo": "True",
        })

    return cases


def load_cases(cases_path: Path) -> List[Dict[str, str]]:
    """Đọc cases.csv từ đĩa."""
    cases = []
    with open(cases_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(row)
    return cases


def generate_task_codes(rng: random.Random, count: int) -> List[str]:
    """Sinh danh sách task_code duy nhất dạng T-xxxx ngẫu nhiên."""
    used_numbers = set()
    codes = []
    while len(codes) < count:
        num = rng.randint(1000, 9999)
        if num not in used_numbers:
            used_numbers.add(num)
            codes.append(f"T-{num:04d}")
    return codes


def make_assignments(
    cases: List[Dict[str, str]],
    seed: int = SEED,
) -> List[Dict[str, str]]:
    """Thực hiện chia ca theo đúng quy tắc cân bằng và ngẫu nhiên."""
    rng = random.Random(seed)

    # Nhóm các ca theo batch_id bảo toàn thứ tự logic
    batch_order = ["P", "B1", "DEV", "B2", "TEST"]
    cases_by_batch: Dict[str, List[Dict[str, str]]] = {b: [] for b in batch_order}
    for c in cases:
        b_id = c.get("batch_id", "")
        if b_id in cases_by_batch:
            cases_by_batch[b_id].append(c)
        else:
            cases_by_batch.setdefault(b_id, []).append(c)

    # Đếm số slot cần sinh task_code: mỗi ca có đúng 2 lượt (pass 1 và pass 2)
    total_slots = len(cases) * 2
    task_codes = generate_task_codes(rng, total_slots)
    task_code_idx = 0

    assignments: List[Dict[str, str]] = []
    pair_cycle_idx = 0

    for b_id in batch_order:
        batch_cases = cases_by_batch.get(b_id, [])
        if not batch_cases:
            continue

        # Luân phiên và cân bằng cặp trong từng batch
        for case in batch_cases:
            case_id = case["case_id"]
            pair = PAIRS[pair_cycle_idx % len(PAIRS)]
            pair_cycle_idx += 1

            # Để tránh thiên vị thứ tự cố định (R1 luôn pass 1), ngẫu nhiên hóa ai làm pass 1, ai làm pass 2
            raters_for_case = list(pair)
            if rng.random() < 0.5:
                raters_for_case = [raters_for_case[1], raters_for_case[0]]

            # Pass 1
            code_1 = task_codes[task_code_idx]
            task_code_idx += 1
            assignments.append({
                "case_id": case_id,
                "task_code": code_1,
                "rater_id": raters_for_case[0],
                "pass_no": "1",
                "batch_id": b_id,
            })

            # Pass 2
            code_2 = task_codes[task_code_idx]
            task_code_idx += 1
            assignments.append({
                "case_id": case_id,
                "task_code": code_2,
                "rater_id": raters_for_case[1],
                "pass_no": "2",
                "batch_id": b_id,
            })

    # Xáo trộn thứ tự các nhiệm vụ để người làm không đoán được mối liên hệ
    rng.shuffle(assignments)
    return assignments


def print_statistics(assignments: List[Dict[str, str]]) -> None:
    """In thống kê phân bổ số lượt của từng rater theo từng batch."""
    print("=== THỐNG KÊ PHÂN BỔ NHIỆM VỤ DÁN NHÃN (ASSIGNMENTS) ===")
    total_per_rater: Dict[str, int] = {r: 0 for r in RATERS}
    primary_per_rater: Dict[str, int] = {r: 0 for r in RATERS}

    batches = sorted(list(set(a["batch_id"] for a in assignments)))
    print(f"{'Batch':<8} | {'Tổng lượt':<10} | " + " | ".join(f"{r:<8}" for r in RATERS))
    print("-" * 52)

    for b in batches:
        b_assigns = [a for a in assignments if a["batch_id"] == b]
        counts = {r: sum(1 for a in b_assigns if a["rater_id"] == r) for r in RATERS}
        for r, c in counts.items():
            total_per_rater[r] += c
            if b != "P":
                primary_per_rater[r] += c
        counts_str = " | ".join(f"{counts[r]:<8}" for r in RATERS)
        print(f"{b:<8} | {len(b_assigns):<10} | {counts_str}")

    print("-" * 52)
    tot_str = " | ".join(f"{total_per_rater[r]:<8}" for r in RATERS)
    prim_str = " | ".join(f"{primary_per_rater[r]:<8}" for r in RATERS)
    print(f"{'28 CA CHÍNH':<20} | {prim_str}")
    print(f"{'TỔNG CỘNG':<20} | {tot_str}")
    print("=========================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo file phân bổ nhiệm vụ dán nhãn assignments.csv")
    parser.add_argument("--cases", type=str, default="", help="Đường dẫn file cases.csv")
    parser.add_argument("--demo", action="store_true", help="Sử dụng danh sách ca giả lập demo")
    parser.add_argument("--output", type=str, default="", help="Đường dẫn file xuất assignments.csv")
    args = parser.parse_args()

    cases_file = Path(args.cases) if args.cases else WORK_DIR / "cases.csv"
    output_file = Path(args.output) if args.output else LABELS_DIR / "assignments.csv"

    if args.demo or not cases_file.exists():
        print(f"[*] Không tìm thấy {cases_file} hoặc có cờ --demo. Khởi tạo {30} ca giả lập theo mục 1 và 5.10.")
        cases = generate_demo_cases()
        # Lưu file cases demo nếu thư mục work chưa có
        if not cases_file.exists():
            WORK_DIR.mkdir(parents=True, exist_ok=True)
            with open(cases_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "case_id",
                        "sample_token",
                        "annotation_token",
                        "instance_token",
                        "scene_token",
                        "scene_name",
                        "class_name",
                        "split",
                        "batch_id",
                        "pointcloud_path",
                        "is_demo",
                    ],
                )
                writer.writeheader()
                for c in cases:
                    full_row = {
                        "case_id": c["case_id"],
                        "sample_token": f"sample_{c['case_id']}",
                        "annotation_token": f"ann_{c['case_id']}",
                        "instance_token": f"inst_{c['case_id']}",
                        "scene_token": f"scene_{c['batch_id']}",
                        "scene_name": f"scene_{c['batch_id']}",
                        "class_name": c["class_name"],
                        "split": c["split"],
                        "batch_id": c["batch_id"],
                        "pointcloud_path": f"data/work/pcd/{c['case_id']}.pcd",
                        "is_demo": "True",
                    }
                    writer.writerow(full_row)
            print(f"[+] Đã ghi file ca mẫu tại: {cases_file}")
    else:
        print(f"[*] Đang đọc danh sách ca từ {cases_file}...")
        cases = load_cases(cases_file)

    assignments = make_assignments(cases, seed=SEED)
    print_statistics(assignments)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["case_id", "task_code", "rater_id", "pass_no", "batch_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(assignments)

    print(f"[OK] Đã ghi {len(assignments)} lượt vào: {output_file}")


if __name__ == "__main__":
    main()
