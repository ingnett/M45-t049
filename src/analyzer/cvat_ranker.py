"""src/analyzer/cvat_ranker.py

Công cụ Dòng lệnh (CLI) và Dịch vụ Xếp hạng Độ khó Bounding Box trực tiếp cho CVAT:
- Nhận file export CVAT (XML hoặc JSON) hoặc kết nối trực tiếp qua CVAT REST API.
- Đánh giá định lượng điểm độ khó từng hộp 3D theo quy tắc M45.
- Sắp xếp và gán thứ hạng từ Khó nhất (#01) đến Dễ nhất (#N).
- Xuất lại file CVAT XML / Datumaro JSON đã được đánh tag thứ hạng, mã màu và huy chương.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analyzer.cvat_integration import (
    compute_dataset_box_ranks,
    export_cvat_3d_xml,
    export_datumaro_3d_json,
    parse_cvat_export_file,
    CVATApiClient,
)


def rank_cvat_boxes_file(input_file: Path, output_file: Path) -> List[Dict[str, Any]]:
    """Xếp hạng độ khó các box từ file CVAT và xuất ra file mới."""
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    parsed = parse_cvat_export_file(content, input_file.name)
    if not parsed:
        print(f"[!] Không tìm thấy box 3D nào trong file {input_file}")
        return []

    boxes_list = [
        {
            "case_id": k,
            "class_name": v["class_name"],
            "center_m": v["center_m"],
            "size_wlh_m": v["size_wlh_m"],
            "yaw_rad": v["yaw_rad"],
            "N": v.get("points_count", 20),
            "distance_m": (v["center_m"][0]**2 + v["center_m"][1]**2 + v["center_m"][2]**2)**0.5,
        }
        for k, v in parsed.items()
    ]

    ranked = compute_dataset_box_ranks(boxes_list, sort_by_rank=True)

    if output_file.suffix.lower() == ".json":
        datumaro = export_datumaro_3d_json(ranked, apply_ranking=False)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(datumaro, f, indent=2, ensure_ascii=False)
    else:
        xml_str = export_cvat_3d_xml(ranked, apply_ranking=False)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(xml_str)

    return ranked


def print_leaderboard(ranked_boxes: List[Dict[str, Any]], top_k: int = 15) -> None:
    """In bảng xếp hạng độ khó ra terminal."""
    print("=" * 78)
    print("          🏆 BẢNG XẾP HẠNG ĐỘ KHÓ 3D BOUNDING BOX (M45 CVAT RANKER)      ")
    print("=" * 78)
    print(f"{'HẠNG':<8} {'CASE_ID':<15} {'CLASS':<15} {'ĐIỂM':<8} {'CẤP ĐỘ':<18} {'CỰ LY'}")
    print("-" * 78)

    for b in ranked_boxes[:top_k]:
        rank_str = b.get("rank_medal", f"#{b.get('difficulty_rank', 0)}")
        cid = str(b.get("case_id", ""))[:14]
        cls_name = str(b.get("class_name", "")).split(".")[-1]
        score = f"{b.get('difficulty_score', 0.0):.2f}"
        tier = b.get("rank_tier", b.get("difficulty_tier", ""))
        dist = f"{b.get('distance_m', 0.0):.1f}m"
        print(f"{rank_str:<8} {cid:<15} {cls_name:<15} {score:<8} {tier:<18} {dist}")

    print("-" * 78)
    n_hard = sum(1 for b in ranked_boxes if "HARD" in b.get("difficulty_tier", "") or "CỰC KHÓ" in b.get("rank_tier", ""))
    print(f"Tổng số: {len(ranked_boxes)} boxes | 🔴 Cực khó (High Risk): {n_hard} ca")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="M45 CVAT 3D Box Difficulty Ranker")
    parser.add_argument("--input", type=str, default="", help="File XML/JSON export từ CVAT")
    parser.add_argument("--output", type=str, default="", help="Đường dẫn lưu file XML/JSON sau khi xếp hạng")
    parser.add_argument("--task_id", type=int, default=None, help="Mã CVAT Task ID để đồng bộ qua REST API")
    parser.add_argument("--cvat_url", type=str, default="http://localhost:8080", help="Địa chỉ CVAT server")
    parser.add_argument("--token", type=str, default="", help="Mã token xác thực API")
    args = parser.parse_args()

    if args.input:
        in_p = Path(args.input)
        out_p = Path(args.output) if args.output else in_p.with_name(f"{in_p.stem}_ranked{in_p.suffix}")
        ranked = rank_cvat_boxes_file(in_p, out_p)
        print_leaderboard(ranked)
        print(f"[OK] Đã lưu dữ liệu xếp hạng vào: {out_p}")
    elif args.task_id is not None:
        client = CVATApiClient(base_url=args.cvat_url, token=args.token if args.token else None)
        conn = client.test_connection()
        print(f"[*] Kết nối CVAT Server ({args.cvat_url}): {conn.get('message')}")
        print(f"[*] Đang yêu cầu xếp hạng Task #{args.task_id}...")
    else:
        # Chạy demo với dữ liệu chuẩn M45
        sample_boxes = [
            {"case_id": "c001_near_dense", "class_name": "vehicle.car", "N": 48, "distance_m": 11.5, "center_m": [11.0, 3.0, 0.0], "size_wlh_m": [1.93, 4.62, 1.69], "yaw_rad": 0.0},
            {"case_id": "c002_mid_dist", "class_name": "vehicle.car", "N": 16, "distance_m": 27.0, "center_m": [26.0, 7.0, 0.0], "size_wlh_m": [1.93, 4.62, 1.69], "yaw_rad": 0.5},
            {"case_id": "c003_far_sparse", "class_name": "vehicle.car", "N": 4, "distance_m": 53.0, "center_m": [52.0, 10.0, 0.0], "size_wlh_m": [1.93, 4.62, 1.69], "yaw_rad": 1.57},
            {"case_id": "c004_ped_close", "class_name": "human.pedestrian.adult", "N": 22, "distance_m": 8.0, "center_m": [8.0, 1.0, 0.0], "size_wlh_m": [0.67, 0.73, 1.77], "yaw_rad": 0.0},
            {"case_id": "c005_ped_far", "class_name": "human.pedestrian.adult", "N": 2, "distance_m": 31.0, "center_m": [30.0, 6.0, 0.0], "size_wlh_m": [0.67, 0.73, 1.77], "yaw_rad": 0.2},
        ]
        ranked = compute_dataset_box_ranks(sample_boxes, sort_by_rank=True)
        print_leaderboard(ranked)


if __name__ == "__main__":
    main()
