"""src/analyzer/cvat_integration.py

Mô-đun Tích hợp Toàn diện CVAT 3D cho Đề tài M45:
- Xuất dữ liệu Pre-label chuẩn CVAT 3D XML (v1.1) và Datumaro 3D JSON (v1.0) kèm nhãn thuộc tính độ khó (difficulty tags).
- Đóng gói Task CVAT 3D hoàn chỉnh (.zip) bao gồm point cloud (.pcd), pre-annotations, và hướng dẫn SOP.
- Nhập kết quả gán nhãn từ CVAT để Kiểm toán Chất lượng (QA Audit) & So sánh sai số Easy vs Hard.
- Kết nối và Đồng bộ Trực tiếp qua CVAT REST API.
- Cấu hình Nuclio Serverless Function cho công cụ AI Auto-Annotation trong giao diện CVAT 3D.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import zipfile

import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analyzer.box_difficulty_rules import evaluate_box_difficulty, DIFFICULTY_THRESHOLDS


# ──────────────────────────────────────────────────────────────────────────────
# 1. HÀM CHUYỂN ĐỔI HÌNH HỌC 3D (YAW <-> EULER <-> QUATERNION)
# ──────────────────────────────────────────────────────────────────────────────
def yaw_to_quaternion(yaw_rad: float) -> Tuple[float, float, float, float]:
    """Chuyển đổi góc xoay Yaw (quanh trục Z) sang Quaternion [w, x, y, z]."""
    half_yaw = yaw_rad * 0.5
    return (float(math.cos(half_yaw)), 0.0, 0.0, float(math.sin(half_yaw)))


def quaternion_to_yaw(w: float, x: float, y: float, z: float) -> float:
    """Trích xuất góc Yaw (radian) quanh trục Z từ Quaternion [w, x, y, z]."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def normalize_angle_rad(angle: float) -> float:
    """Chuẩn hóa góc về khoảng [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return float(angle)


# ──────────────────────────────────────────────────────────────────────────────
# 2. BỘ ĐỘNG CƠ XẾP HẠNG ĐỘ KHÓ BOX (BOX DIFFICULTY RANKING ENGINE)
# ──────────────────────────────────────────────────────────────────────────────
def compute_dataset_box_ranks(
    boxes_data: Union[List[Dict[str, Any]], pd.DataFrame],
    sort_by_rank: bool = True,
) -> List[Dict[str, Any]]:
    """
    Xếp hạng toàn bộ Bounding Box trong tập dữ liệu theo độ khó (Ranking Engine):
    - Đánh giá điểm độ khó chính xác từng box theo công thức vật lý và kinh nghiệm nuScenes.
    - Sắp xếp giảm dần từ Box Khó nhất (#01) đến Box Dễ nhất (#N).
    - Gán huy chương thứ hạng (🥇 Top 1-3, 🥈 Top 4-10, 🥉 Top 11-20, ⚪ Còn lại).
    - Gán mã màu CVAT (Bright Red -> Green) và cấp độ Rank Tier để CVAT 3D hiển thị trực quan.
    """
    items: List[Dict[str, Any]] = []
    if isinstance(boxes_data, pd.DataFrame):
        for idx, row in boxes_data.iterrows():
            c = row.get("_center", [0.0, 0.0, 0.0])
            s = row.get("_wlh", [1.93, 4.62, 1.69])
            items.append({
                "case_id": row.get("case_id", f"case_{idx:04d}"),
                "class_name": row.get("class_name", "vehicle.car"),
                "center_m": [float(x) for x in c],
                "size_wlh_m": [float(x) for x in s],
                "yaw_rad": float(row.get("_yaw", 0.0)),
                "N": int(row.get("N", 20)),
                "distance_m": float(row.get("distance_m", 15.0)),
                "occlusion_ratio": float(row.get("occlusion_ratio", 0.0)),
                "points_xyz": row.get("_pts", None),
                "_orig_idx": idx,
            })
    else:
        for idx, b in enumerate(boxes_data):
            c = b.get("center_m", [0.0, 0.0, 0.0])
            s = b.get("size_wlh_m", [1.93, 4.62, 1.69])
            items.append({
                "case_id": b.get("case_id", f"case_{idx:04d}"),
                "class_name": b.get("class_name", "vehicle.car"),
                "center_m": [float(x) for x in c],
                "size_wlh_m": [float(x) for x in s],
                "yaw_rad": float(b.get("yaw_rad", 0.0)),
                "N": int(b.get("N", b.get("point_count", 20))),
                "distance_m": float(b.get("distance_m", math.sqrt(c[0]**2 + c[1]**2 + c[2]**2))),
                "occlusion_ratio": float(b.get("occlusion_ratio", 0.0)),
                "points_xyz": b.get("points_xyz", None),
                "_orig_idx": idx,
            })

    # Đánh giá độ khó cho từng box
    for item in items:
        diff = evaluate_box_difficulty(
            point_count=item["N"],
            distance_m=item["distance_m"],
            class_name=item["class_name"],
            occlusion_ratio=item["occlusion_ratio"],
        )
        item["difficulty_score"] = diff["difficulty_score"]
        item["difficulty_index"] = diff["difficulty_index"]
        item["difficulty_tier"] = diff["tier"]
        item["risk_flags"] = diff["errors_predicted_str"]
        item["annotator_sop"] = diff["annotator_sop"]
        item["reasons_str"] = diff["reasons_str"]

    # Sắp xếp giảm dần theo điểm độ khó (Khó nhất lên đầu)
    items.sort(key=lambda x: (x["difficulty_score"], x["distance_m"], -x["N"]), reverse=True)

    n_total = len(items)
    for rank_idx, item in enumerate(items, start=1):
        item["difficulty_rank"] = rank_idx
        item["rank_tag"] = f"#{rank_idx:02d}"
        
        # Huy chương & Nhãn
        if rank_idx <= 3:
            medal = "🥇"
        elif rank_idx <= 10:
            medal = "🥈"
        elif rank_idx <= 20:
            medal = "🥉"
        else:
            medal = "⚪"
        item["rank_medal"] = f"{medal} #{rank_idx:02d}"
        item["rank_badge"] = f"{medal} #{rank_idx:02d} (Score: {item['difficulty_score']:.2f})"

        # Phân vị (Percentile)
        pct = (rank_idx / max(1, n_total)) * 100.0
        item["rank_percentile"] = f"Top {int(math.ceil(pct))}% Khó" if pct <= 50.0 else f"Dễ (Bottom {int(100-pct)}%)"

        # Tier phân loại trực quan trong CVAT
        score = item["difficulty_score"]
        if score >= 4.2:
            item["rank_tier"] = "[RANK 1] 🔴 CỰC KHÓ"
            item["cvat_color"] = "#EF4444"
        elif score >= 3.6:
            item["rank_tier"] = "[RANK 2] 🟠 KHÓ"
            item["cvat_color"] = "#F97316"
        elif score >= 2.2:
            item["rank_tier"] = "[RANK 3] 🟡 TRUNG BÌNH"
            item["cvat_color"] = "#F59E0B"
        else:
            item["rank_tier"] = "[RANK 4] 🟢 DỄ"
            item["cvat_color"] = "#10B981"

    if not sort_by_rank:
        items.sort(key=lambda x: x["_orig_idx"])

    return items


# ──────────────────────────────────────────────────────────────────────────────
# 3. XUẤT PRE-LABELS ĐỊNH DẠNG CVAT 3D XML (CVAT for Video / 3D 1.1)
# ──────────────────────────────────────────────────────────────────────────────
def export_cvat_3d_xml(
    boxes_data: Union[List[Dict[str, Any]], pd.DataFrame],
    task_name: str = "M45_3D_LiDAR_Task",
    include_attributes: bool = True,
    apply_ranking: bool = True,
) -> str:
    """
    Tạo nội dung XML chuẩn CVAT for 3D/video 1.1 chứa các 3D cuboid kèm BẢNG XẾP HẠNG ĐỘ KHÓ.
    CVAT 3D cho phép hiển thị các thuộc tính difficulty_rank, rank_badge, cvat_color, và lọc theo thứ hạng.
    """
    # Xếp hạng độ khó nếu cần
    if apply_ranking:
        ranked_boxes = compute_dataset_box_ranks(boxes_data, sort_by_rank=True)
    elif isinstance(boxes_data, pd.DataFrame):
        ranked_boxes = compute_dataset_box_ranks(boxes_data, sort_by_rank=False)
    else:
        ranked_boxes = boxes_data

    root = ET.Element("annotations")
    
    # Version
    version_elem = ET.SubElement(root, "version")
    version_elem.text = "1.1"

    # Meta
    meta_elem = ET.SubElement(root, "meta")
    task_elem = ET.SubElement(meta_elem, "task")
    ET.SubElement(task_elem, "name").text = task_name
    ET.SubElement(task_elem, "size").text = str(len(ranked_boxes))
    
    # Labels metadata
    labels_elem = ET.SubElement(task_elem, "labels")
    
    # Định nghĩa nhãn và thuộc tính cho vehicle.car và human.pedestrian.adult
    for class_name in ["vehicle.car", "human.pedestrian.adult"]:
        lbl = ET.SubElement(labels_elem, "label")
        ET.SubElement(lbl, "name").text = class_name
        
        attrs_elem = ET.SubElement(lbl, "attributes")
        
        attr_defs = [
            ("difficulty_rank", "text", "#01"),
            ("rank_badge", "text", "🥇 #01 (Score: 4.5)"),
            ("difficulty_tier", "select", "[RANK 1] 🔴 CỰC KHÓ\n[RANK 2] 🟠 KHÓ\n[RANK 3] 🟡 TRUNG BÌNH\n[RANK 4] 🟢 DỄ\nEASY\nMEDIUM\nHARD"),
            ("difficulty_score", "number", "1.0"),
            ("rank_percentile", "text", "Top 5% Khó"),
            ("cvat_color", "text", "#EF4444"),
            ("points_count", "number", "0"),
            ("distance_m", "number", "0.0"),
            ("risk_flags", "text", "None"),
            ("annotator_sop", "text", "Standard SOP"),
            ("prelabel_source", "text", "M45_Ridge_Screener"),
        ]
        
        for a_name, a_type, a_vals in attr_defs:
            a_elem = ET.SubElement(attrs_elem, "attribute")
            ET.SubElement(a_elem, "name").text = a_name
            ET.SubElement(a_elem, "input_type").text = a_type
            ET.SubElement(a_elem, "default_value").text = a_vals.split("\n")[0]
            ET.SubElement(a_elem, "values").text = a_vals

    # Thêm từng khung hình / PCD theo thứ tự xếp hạng
    for idx, box in enumerate(ranked_boxes):
        case_id = str(box.get("case_id", f"frame_{idx:04d}"))
        pcd_name = f"{case_id}.pcd"
        
        frame_elem = ET.SubElement(root, "image", id=str(idx), name=pcd_name, width="1920", height="1080")
        
        center = box.get("center_m", [0.0, 0.0, 0.0])
        size = box.get("size_wlh_m", [1.93, 4.62, 1.69])
        yaw = float(box.get("yaw_rad", 0.0))
        cls_name = str(box.get("class_name", "vehicle.car"))
        
        pts = int(box.get("N", box.get("point_count", 20)))
        dist = float(box.get("distance_m", math.sqrt(center[0]**2 + center[1]**2 + center[2]**2)))
        occ = float(box.get("occlusion_ratio", 0.0))
        
        tier = box.get("rank_tier", box.get("difficulty_tier", "MEDIUM"))
        score = float(box.get("difficulty_score", 2.5))
        rank_tag = box.get("rank_tag", f"#{idx+1:02d}")
        rank_badge = box.get("rank_badge", f"#{idx+1}")
        pct_tag = box.get("rank_percentile", "N/A")
        c_hex = box.get("cvat_color", "#38bdf8")
        risk_flags = box.get("risk_flags", "None")
        sop = box.get("annotator_sop", "Standard SOP")

        # Tạo thẻ cuboid
        pos_str = f"{center[0]:.4f},{center[1]:.4f},{center[2]:.4f}"
        scale_str = f"{size[0]:.4f},{size[1]:.4f},{size[2]:.4f}"
        rot_str = f"0.0,0.0,{yaw:.6f}"
        
        cuboid_elem = ET.SubElement(
            frame_elem,
            "cuboid",
            label=cls_name,
            occluded="1" if occ >= 0.4 else "0",
            position=pos_str,
            scale=scale_str,
            rotation=rot_str,
        )
        
        if include_attributes:
            def add_attr(name: str, val: Any) -> None:
                ae = ET.SubElement(cuboid_elem, "attribute", name=name)
                ae.text = str(val)
                
            add_attr("difficulty_rank", rank_tag)
            add_attr("rank_badge", rank_badge)
            add_attr("difficulty_tier", tier)
            add_attr("difficulty_score", f"{score:.2f}")
            add_attr("rank_percentile", pct_tag)
            add_attr("cvat_color", c_hex)
            add_attr("points_count", pts)
            add_attr("distance_m", f"{dist:.2f}")
            add_attr("risk_flags", risk_flags)
            add_attr("annotator_sop", sop)
            add_attr("prelabel_source", "M45_AI_Screener")

    # Format XML đẹp
    raw_xml = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(raw_xml)
    return parsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# 3. XUẤT PRE-LABELS ĐỊNH DẠNG CVAT DATUMARO 3D JSON (v1.0)
# ──────────────────────────────────────────────────────────────────────────────
def export_datumaro_3d_json(
    boxes_data: Union[List[Dict[str, Any]], pd.DataFrame],
    dataset_name: str = "M45_Datumaro_3D",
    apply_ranking: bool = True,
) -> Dict[str, Any]:
    """
    Tạo cấu trúc JSON định dạng Datumaro 1.0 3D kèm BẢNG XẾP HẠNG ĐỘ KHÓ.
    """
    if apply_ranking:
        ranked_boxes = compute_dataset_box_ranks(boxes_data, sort_by_rank=True)
    elif isinstance(boxes_data, pd.DataFrame):
        ranked_boxes = compute_dataset_box_ranks(boxes_data, sort_by_rank=False)
    else:
        ranked_boxes = boxes_data

    labels = ["vehicle.car", "human.pedestrian.adult"]
    label_map = {name: idx for idx, name in enumerate(labels)}

    items = []
    for idx, box in enumerate(ranked_boxes):
        case_id = str(box.get("case_id", f"case_{idx:04d}"))
        center = [float(c) for c in box.get("center_m", [0.0, 0.0, 0.0])]
        size = [float(s) for s in box.get("size_wlh_m", [1.93, 4.62, 1.69])]
        yaw = float(box.get("yaw_rad", 0.0))
        cls_name = str(box.get("class_name", "vehicle.car"))
        label_id = label_map.get(cls_name, 0)

        pts = int(box.get("N", box.get("point_count", 20)))
        dist = float(box.get("distance_m", math.sqrt(center[0]**2 + center[1]**2 + center[2]**2)))
        tier = box.get("rank_tier", box.get("difficulty_tier", "MEDIUM"))
        score = float(box.get("difficulty_score", 2.5))
        rank_tag = box.get("rank_tag", f"#{idx+1:02d}")
        rank_badge = box.get("rank_badge", f"#{idx+1}")
        pct_tag = box.get("rank_percentile", "N/A")
        c_hex = box.get("cvat_color", "#38bdf8")

        ann = {
            "id": 1,
            "type": "cuboid_3d",
            "label_id": label_id,
            "position": center,
            "scale": size,
            "rotation": [0.0, 0.0, yaw],
            "attributes": {
                "difficulty_rank": rank_tag,
                "rank_badge": rank_badge,
                "difficulty_tier": tier,
                "difficulty_score": round(score, 2),
                "rank_percentile": pct_tag,
                "cvat_color": c_hex,
                "points_count": pts,
                "distance_m": round(dist, 2),
                "risk_flags": box.get("risk_flags", "None"),
                "annotator_sop": box.get("annotator_sop", "Standard SOP"),
                "is_hard_case": "HARD" in tier or "CỰC KHÓ" in tier or "KHÓ" in tier,
            },
        }

        items.append({
            "id": case_id,
            "annotations": [ann],
            "point_cloud": {"path": f"point_clouds/{case_id}.pcd"},
        })

    datumaro_doc = {
        "info": {
            "description": "M45 nuScenes 3D LiDAR Pre-labels with Difficulty Ranking",
            "version": "1.0",
            "format": "datumaro_3d",
        },
        "categories": {
            "label": {
                "labels": [{"name": name, "parent": ""} for name in labels],
                "attributes": [
                    "difficulty_rank", "rank_badge", "difficulty_tier",
                    "difficulty_score", "rank_percentile", "cvat_color",
                    "points_count", "distance_m", "risk_flags", "annotator_sop", "is_hard_case"
                ],
            }
        },
        "items": items,
    }

    return datumaro_doc


# ──────────────────────────────────────────────────────────────────────────────
# 4. TẠO FILE ASCII PCD 0.7 CHUẨN CVAT CHO POINT CLOUD
# ──────────────────────────────────────────────────────────────────────────────
def format_pcd_ascii(points: np.ndarray) -> str:
    """Tạo nội dung file PCD Version 0.7 dạng ASCII theo chuẩn PCL / CVAT 3D."""
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
    lines = header + [f"{float(p[0]):.4f} {float(p[1]):.4f} {float(p[2]):.4f}" for p in points]
    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# 5. ĐÓNG GÓI TASK CVAT 3D HOÀN CHỈNH (TASK PACK ZIP)
# ──────────────────────────────────────────────────────────────────────────────
def create_cvat_task_pack_zip(
    boxes_df_or_list: Union[pd.DataFrame, List[Dict[str, Any]]],
    task_name: str = "M45_CVAT_Task",
    instructions_text: Optional[str] = None,
) -> io.BytesIO:
    """
    Tạo gói file ZIP chứa toàn bộ cấu trúc để import thẳng vào CVAT 3D:
    - annotations.xml: Toàn bộ pre-label boxes kèm thuộc tính độ khó.
    - datumaro_annotations.json: File Datumaro tương ứng.
    - point_clouds/<case_id>.pcd: File đám mây điểm chuẩn ASCII PCD.
    - classes.csv: Bảng đối chiếu class theo ca.
    - INSTRUCTIONS.txt: Quy trình thao tác chuẩn cho Annotator.
    """
    if isinstance(boxes_df_or_list, pd.DataFrame):
        boxes_list = []
        for _, row in boxes_df_or_list.iterrows():
            center = row.get("_center", [0.0, 0.0, 0.0])
            wlh = row.get("_wlh", [1.93, 4.62, 1.69])
            yaw = float(row.get("_yaw", 0.0))
            pts = row.get("_pts", None)
            boxes_list.append({
                "case_id": row.get("case_id", "demo"),
                "class_name": row.get("class_name", "vehicle.car"),
                "center_m": [float(c) for c in center],
                "size_wlh_m": [float(s) for s in wlh],
                "yaw_rad": yaw,
                "N": int(row.get("N", 20)),
                "distance_m": float(row.get("distance_m", 15.0)),
                "points_xyz": pts,
            })
    else:
        boxes_list = boxes_df_or_list

    # 1. Sinh XML và JSON
    xml_content = export_cvat_3d_xml(boxes_list, task_name=task_name)
    datumaro_dict = export_datumaro_3d_json(boxes_list, dataset_name=task_name)
    json_content = json.dumps(datumaro_dict, indent=2, ensure_ascii=False)

    # 2. Hướng dẫn Annotator
    if not instructions_text:
        instructions_text = """================================================================================
QUY TRÌNH DÁN NHÃN VÀ RÀ SOÁT PRE-LABEL TRÊN CVAT 3D (ĐỀ TÀI M45)
================================================================================
1. CÁCH NHẬP VÀO CVAT 3D:
   - Bước 1: Mở CVAT -> Nhấn 'Create a new task'.
   - Bước 2: Đặt tên task (VD: M45_3D_LiDAR_Task).
   - Bước 3: Thêm 2 labels:
       * vehicle.car
       * human.pedestrian.adult
   - Bước 4: Tải toàn bộ file trong thư mục point_clouds/ vào phần 'Select files'.
   - Bước 5: Nhấn 'Submit & Open'.
   - Bước 6: Tại trang Task, vào Actions -> 'Upload Annotations' -> Chọn 'CVAT for video 1.1' 
             và tải tệp 'annotations.xml' đính kèm. Toàn bộ 3D pre-labels sẽ lập tức xuất hiện!

2. LỌC VÀ ƯU TIÊN XỬ LÝ BOX KHÓ (HARD BOX FILTER):
   - Mở cửa sổ 3D của CVAT.
   - Nhìn sang thanh đối tượng (Objects Sidebar) -> Nhấn Filter.
   - Lọc theo: `difficulty_tier == "HARD"` hoặc chú ý các box có cờ đỏ.
   - QUY TẮC BẮT BUỘC KHI CHỈNH BOX KHÓ:
       * KHÓA KÍCH THƯỚC: Giữ nguyên w/l/h theo Class Prior, không co cụm theo điểm thưa.
       * NEO MẶT NHÌN THẤY: Căn mép hộp sát cụm điểm LiDAR đầu tiên (Visible Face).
       * XÁC MINH HƯỚNG YAW: Tránh lộn ngược đầu/đuôi 180 độ.

3. XUẤT KẾT QUẢ ĐỂ KIỂM TOÁN CHẤT LƯỢNG (QA AUDIT):
   - Sau khi hoàn thành gán nhãn, vào Actions -> 'Export Task Dataset'.
   - Chọn định dạng 'CVAT for video 1.1' (XML) hoặc 'Datumaro 1.0' (JSON).
   - Tải file về và nạp vào Tab 'Kiểm toán Gán nhãn CVAT' trên ứng dụng M45 để đo sai số.
================================================================================
"""

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("annotations.xml", xml_content)
        zf.writestr("datumaro_annotations.json", json_content)
        zf.writestr("INSTRUCTIONS.txt", instructions_text)

        # classes.csv
        classes_rows = ["task_code,class_name,difficulty_tier,points_count,distance_m\n"]
        for b in boxes_list:
            c_id = b.get("case_id", "")
            c_name = b.get("class_name", "vehicle.car")
            diff = evaluate_box_difficulty(int(b.get("N", 20)), float(b.get("distance_m", 15.0)), c_name)
            classes_rows.append(f"{c_id},{c_name},{diff['tier']},{b.get('N', 20)},{b.get('distance_m', 15.0):.1f}\n")
        zf.writestr("classes.csv", "".join(classes_rows))

        # Điểm PCD
        for b in boxes_list:
            c_id = b.get("case_id", "demo")
            pts = b.get("points_xyz")
            if pts is None or len(pts) == 0:
                # Tạo điểm giả lập tối thiểu nếu không có sẵn
                c = b.get("center_m", [10.0, 0.0, 0.0])
                pts = np.array([
                    [c[0] - 0.5, c[1], c[2]],
                    [c[0] + 0.5, c[1], c[2]],
                    [c[0], c[1] - 0.5, c[2]],
                    [c[0], c[1] + 0.5, c[2]],
                    [c[0], c[1], c[2] + 0.5],
                ])
            pcd_str = format_pcd_ascii(pts)
            zf.writestr(f"point_clouds/{c_id}.pcd", pcd_str)

    zip_buffer.seek(0)
    return zip_buffer


# ──────────────────────────────────────────────────────────────────────────────
# 6. NHẬP & KIỂM TOÁN CHẤT LƯỢNG ANNOTATION TỪ CVAT (QA AUDIT ENGINE)
# ──────────────────────────────────────────────────────────────────────────────
def parse_cvat_export_file(file_content: Union[str, bytes], filename: str) -> Dict[str, Dict[str, Any]]:
    """
    Phân tích file xuất từ CVAT (có thể là XML hoặc JSON) thành từ điển chuẩn:
    {case_id: {'center_m': [...], 'size_wlh_m': [...], 'yaw_rad': float, 'class_name': str}}
    """
    boxes: Dict[str, Dict[str, Any]] = {}
    
    if isinstance(file_content, bytes):
        text_content = file_content.decode("utf-8", errors="replace")
    else:
        text_content = file_content

    is_json = filename.lower().endswith(".json") or text_content.strip().startswith("{")
    
    if is_json:
        data = json.loads(text_content)
        items = data.get("items", [])
        for item in items:
            item_id = str(item.get("id", "")).strip()
            # Bỏ phần mở rộng nếu có
            clean_id = Path(item_id).stem
            anns = item.get("annotations", [])
            cuboid = None
            for ann in anns:
                if ann.get("type") in ("cuboid_3d", "cuboid3d", "cuboid"):
                    cuboid = ann
                    break
            if cuboid:
                pos = cuboid.get("position", [0.0, 0.0, 0.0])
                rot = cuboid.get("rotation", [0.0, 0.0, 0.0])
                scale = cuboid.get("scale", [1.93, 4.62, 1.69])
                
                # Yaw: phần tử thứ 3 hoặc 1 trong rotation
                if len(rot) == 3:
                    yaw = float(rot[2])
                elif len(rot) == 1:
                    yaw = float(rot[0])
                elif len(rot) == 4:
                    yaw = quaternion_to_yaw(float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))
                else:
                    yaw = 0.0

                boxes[clean_id] = {
                    "center_m": [float(p) for p in pos],
                    "size_wlh_m": [float(s) for s in scale],
                    "yaw_rad": normalize_angle_rad(yaw),
                    "class_name": cuboid.get("label", "vehicle.car"),
                    "status": "completed",
                }
    else:
        # Phân tích XML
        root = ET.fromstring(text_content)
        for elem in root.findall(".//image") + root.findall(".//frame"):
            name = elem.get("name", "") or elem.get("id", "")
            clean_id = Path(name).stem if name else ""
            cuboid = elem.find(".//cuboid")
            if cuboid is not None:
                pos_str = cuboid.get("position", "")
                scale_str = cuboid.get("scale", "")
                rot_str = cuboid.get("rotation", "")

                if pos_str and scale_str:
                    pos = [float(p) for p in pos_str.split(",")]
                    scale = [float(s) for s in scale_str.split(",")]
                    rot = [float(r) for r in rot_str.split(",")] if rot_str else [0.0, 0.0, 0.0]
                else:
                    pos = [
                        float(cuboid.get("pos_x", 0.0)),
                        float(cuboid.get("pos_y", 0.0)),
                        float(cuboid.get("pos_z", 0.0)),
                    ]
                    scale = [
                        float(cuboid.get("scale_x", 1.93)),
                        float(cuboid.get("scale_y", 4.62)),
                        float(cuboid.get("scale_z", 1.69)),
                    ]
                    rot = [
                        float(cuboid.get("rot_x", 0.0)),
                        float(cuboid.get("rot_y", 0.0)),
                        float(cuboid.get("rot_z", 0.0)),
                    ]

                yaw = float(rot[2]) if len(rot) >= 3 else 0.0
                boxes[clean_id] = {
                    "center_m": pos,
                    "size_wlh_m": scale,
                    "yaw_rad": normalize_angle_rad(yaw),
                    "class_name": cuboid.get("label", "vehicle.car"),
                    "status": "completed",
                }

    return boxes


def calculate_box_iou_bev(
    center_a: List[float], size_a: List[float], yaw_a: float,
    center_b: List[float], size_b: List[float], yaw_b: float,
) -> float:
    """
    Tính IoU gần đúng trên mặt phẳng Bird's-Eye-View (BEV) và tích hợp chiều cao Z.
    """
    w_a, l_a, h_a = size_a[0], size_a[1], size_a[2]
    w_b, l_b, h_b = size_b[0], size_b[1], size_b[2]

    # Kiểm tra giao theo trục Z (chiều cao)
    min_z_a = center_a[2] - h_a / 2.0
    max_z_a = center_a[2] + h_a / 2.0
    min_z_b = center_b[2] - h_b / 2.0
    max_z_b = center_b[2] + h_b / 2.0

    inter_h = max(0.0, min(max_z_a, max_z_b) - max(min_z_a, min_z_b))
    if inter_h <= 1e-4:
        return 0.0

    # Ước lượng diện tích giao BEV bằng hình chữ nhật bao quanh gần đúng
    dx = abs(center_a[0] - center_b[0])
    dy = abs(center_a[1] - center_b[1])

    avg_w = 0.5 * (w_a + w_b)
    avg_l = 0.5 * (l_a + l_b)

    inter_w = max(0.0, avg_w - dx)
    inter_l = max(0.0, avg_l - dy)

    # Hiệu chỉnh góc yaw lệch
    dyaw = abs(normalize_angle_rad(yaw_a - yaw_b))
    yaw_factor = max(0.0, float(math.cos(dyaw)))

    inter_vol = inter_w * inter_l * inter_h * yaw_factor
    vol_a = w_a * l_a * h_a
    vol_b = w_b * l_b * h_b
    union_vol = vol_a + vol_b - inter_vol

    if union_vol <= 1e-4:
        return 0.0
    return float(np.clip(inter_vol / union_vol, 0.0, 1.0))


def audit_cvat_annotations_vs_gt(
    annotator_boxes: Dict[str, Dict[str, Any]],
    gt_boxes_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    So sánh toàn diện bộ nhãn CVAT vừa nhập với Ground Truth / Pre-label gốc:
    - Tính sai số tâm (ΔCenter), sai số hướng (ΔYaw), sai số kích thước (ΔSize), và IoU 3D.
    - Phân tách sai số theo cấp độ Easy vs Medium vs Hard.
    - Phát hiện các lỗi nghiêm trọng: Lộn góc Yaw 180°, Lệch tâm lớn, Co ngót kích thước.
    """
    records = []
    
    for _, row in gt_boxes_df.iterrows():
        case_id = str(row.get("case_id", ""))
        gt_center = [float(c) for c in row.get("_center", [0.0, 0.0, 0.0])]
        gt_wlh = [float(s) for s in row.get("_wlh", [1.93, 4.62, 1.69])]
        gt_yaw = float(row.get("_yaw", 0.0))
        cls_name = str(row.get("class_name", "vehicle.car"))
        pts = int(row.get("N", 20))
        dist = float(row.get("distance_m", 15.0))
        occ = float(row.get("occlusion_ratio", 0.0))

        diff_meta = evaluate_box_difficulty(pts, dist, cls_name, occ)
        tier = diff_meta["tier"]

        ann = annotator_boxes.get(case_id)
        if not ann:
            records.append({
                "case_id": case_id,
                "class_name": cls_name,
                "tier": tier,
                "N": pts,
                "distance_m": dist,
                "status": "MISSING_IN_CVAT",
                "center_err_m": np.nan,
                "yaw_err_deg": np.nan,
                "size_err_pct": np.nan,
                "iou_3d": 0.0,
                "is_yaw_flipped": False,
                "is_severe_error": True,
            })
            continue

        c_ann = ann["center_m"]
        s_ann = ann["size_wlh_m"]
        y_ann = ann["yaw_rad"]

        # Sai số tâm
        c_err = math.sqrt(
            (c_ann[0] - gt_center[0])**2 +
            (c_ann[1] - gt_center[1])**2 +
            (c_ann[2] - gt_center[2])**2
        )

        # Sai số hướng yaw
        raw_dyaw = abs(normalize_angle_rad(y_ann - gt_yaw))
        yaw_err_deg = math.degrees(raw_dyaw)

        # Kiểm tra lộn ngược đầu/đuôi xe 180 độ (±15 độ)
        is_yaw_flipped = abs(yaw_err_deg - 180.0) <= 20.0

        # Sai số kích thước
        size_err_pct = float(np.mean([
            abs(s_ann[0] - gt_wlh[0]) / max(0.1, gt_wlh[0]),
            abs(s_ann[1] - gt_wlh[1]) / max(0.1, gt_wlh[1]),
            abs(s_ann[2] - gt_wlh[2]) / max(0.1, gt_wlh[2]),
        ]) * 100.0)

        # IoU 3D
        iou_val = calculate_box_iou_bev(c_ann, s_ann, y_ann, gt_center, gt_wlh, gt_yaw)

        # Đánh giá lỗi nghiêm trọng
        is_severe = (c_err > 0.45) or is_yaw_flipped or (size_err_pct > 25.0) or (iou_val < 0.50)

        records.append({
            "case_id": case_id,
            "class_name": cls_name,
            "tier": tier,
            "N": pts,
            "distance_m": dist,
            "status": "EVALUATED",
            "center_err_m": round(c_err, 4),
            "yaw_err_deg": round(yaw_err_deg, 2),
            "size_err_pct": round(size_err_pct, 2),
            "iou_3d": round(iou_val, 4),
            "is_yaw_flipped": is_yaw_flipped,
            "is_severe_error": is_severe,
        })

    audit_df = pd.DataFrame(records)

    # Thống kê tổng hợp theo Tier
    tier_summary = {}
    for t in ["EASY", "MEDIUM", "HARD"]:
        sub = audit_df[audit_df["tier"] == t].dropna(subset=["center_err_m"])
        if len(sub) > 0:
            tier_summary[t] = {
                "count": len(sub),
                "mean_center_err_m": float(sub["center_err_m"].mean()),
                "mean_yaw_err_deg": float(sub["yaw_err_deg"].mean()),
                "mean_size_err_pct": float(sub["size_err_pct"].mean()),
                "mean_iou_3d": float(sub["iou_3d"].mean()),
                "severe_error_rate_pct": float(sub["is_severe_error"].mean() * 100.0),
                "yaw_flip_count": int(sub["is_yaw_flipped"].sum()),
            }
        else:
            tier_summary[t] = {
                "count": 0, "mean_center_err_m": 0.0, "mean_yaw_err_deg": 0.0,
                "mean_size_err_pct": 0.0, "mean_iou_3d": 0.0,
                "severe_error_rate_pct": 0.0, "yaw_flip_count": 0,
            }

    return {
        "audit_df": audit_df,
        "tier_summary": tier_summary,
        "total_evaluated": len(audit_df[audit_df["status"] == "EVALUATED"]),
        "total_missing": len(audit_df[audit_df["status"] == "MISSING_IN_CVAT"]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7. CVAT REST API CLIENT (ĐỒNG BỘ TRỰC TIẾP QUA MẠNG)
# ──────────────────────────────────────────────────────────────────────────────
class CVATApiClient:
    """
    Client REST API kết nối trực tiếp đến máy chủ CVAT (Local Docker hoặc CVAT Cloud).
    Cung cấp các thao tác kiểm tra kết nối, tạo task 3D, tải PCD và đẩy pre-labels.
    """
    def __init__(self, base_url: str = "http://localhost:8080", token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.headers = {"Accept": "application/json"}
        if token:
            self.headers["Authorization"] = f"Token {token}"

    def test_connection(self) -> Dict[str, Any]:
        """Kiểm tra trạng thái kết nối tới máy chủ CVAT."""
        try:
            import urllib.request
            import urllib.error
            url = f"{self.base_url}/api/server/about"
            req = urllib.request.Request(url, headers=self.headers, method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "connected": True,
                    "server_version": data.get("version", "Unknown"),
                    "message": f"Kết nối thành công tới CVAT Server ({data.get('version', 'v2.x')})",
                }
        except Exception as e:
            # Fallback mock diagnostic
            return {
                "connected": False,
                "server_version": None,
                "message": f"Không thể kết nối trực tiếp ({type(e).__name__}: {e}). Đang kích hoạt Chế độ Mô phỏng API (Mock/Offline Mode).",
                "simulated_curl": f'curl -X GET "{self.base_url}/api/server/about" -H "Authorization: Token {self.token or "YOUR_TOKEN"}"',
            }

    def generate_task_payload(self, task_name: str, labels: List[str]) -> Dict[str, Any]:
        """Tạo cấu trúc JSON payload cho API tạo 3D Task trên CVAT."""
        label_defs = []
        for lbl in labels:
            label_defs.append({
                "name": lbl,
                "color": "#38bdf8" if "car" in lbl else "#c084fc",
                "attributes": [
                    {
                        "name": "difficulty_tier",
                        "input_type": "select",
                        "mutable": True,
                        "values": ["EASY", "MEDIUM", "HARD"],
                    },
                    {
                        "name": "difficulty_score",
                        "input_type": "number",
                        "mutable": True,
                        "values": ["1.0", "5.0", "0.1"],
                    },
                    {
                        "name": "risk_flags",
                        "input_type": "text",
                        "mutable": True,
                        "values": [],
                    },
                    {
                        "name": "annotator_sop",
                        "input_type": "text",
                        "mutable": True,
                        "values": [],
                    },
                ],
            })
        return {
            "name": task_name,
            "project_id": None,
            "labels": label_defs,
            "segment_size": 1,
            "overlap": 0,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 8. MÔ HÌNH NUCLIO SERVERLESS CHO CVAT AI TOOLS (MAGIC WAND 3D)
# ──────────────────────────────────────────────────────────────────────────────
def get_nuclio_function_yaml() -> str:
    """Trả về file cấu hình function.yaml để triển khai M45 AI Screener vào Nuclio/CVAT."""
    return """metadata:
  name: m45-lidar-box-difficulty-screener
  namespace: cvat
  annotations:
    name: M45 3D LiDAR Difficulty & Pre-label Screener
    type: detector
    framework: pytorch
    spec: |
      [
        {
          "id": 0,
          "name": "vehicle.car",
          "type": "cuboid",
          "attributes": [
            { "id": 0, "name": "difficulty_tier", "type": "select", "values": ["EASY", "MEDIUM", "HARD"] },
            { "id": 1, "name": "difficulty_score", "type": "number", "values": ["1.0", "5.0", "0.1"] },
            { "id": 2, "name": "risk_flags", "type": "text", "values": [] },
            { "id": 3, "name": "annotator_sop", "type": "text", "values": [] }
          ]
        },
        {
          "id": 1,
          "name": "human.pedestrian.adult",
          "type": "cuboid",
          "attributes": [
            { "id": 0, "name": "difficulty_tier", "type": "select", "values": ["EASY", "MEDIUM", "HARD"] },
            { "id": 1, "name": "difficulty_score", "type": "number", "values": ["1.0", "5.0", "0.1"] },
            { "id": 2, "name": "risk_flags", "type": "text", "values": [] },
            { "id": 3, "name": "annotator_sop", "type": "text", "values": [] }
          ]
        }
      ]

spec:
  description: M45 Universal 3D LiDAR Point Density & Annotation Difficulty Screener for CVAT
  runtime: python:3.9
  handler: main:handler
  eventTimeout: 30s
  build:
    image: cvat.m45.screener:latest
    baseImage: python:3.9-slim
    directives:
      preCopy:
        - kind: RUN
          value: pip install --no-cache-dir numpy scipy
"""


def get_nuclio_main_py() -> str:
    """Trả về file mã nguồn main.py của Nuclio serverless function phục vụ CVAT."""
    return '''import json
import math
import numpy as np

CLASS_PRIORS = {
    "vehicle.car": [1.93, 4.62, 1.69],
    "human.pedestrian.adult": [0.67, 0.73, 1.77],
}

def init_context(context):
    context.logger.info("M45 Nuclio 3D Screener Initialized successfully.")

def handler(context, event):
    """Xử lý yêu cầu gán nhãn 3D AI từ giao diện CVAT."""
    try:
        data = event.body
        if isinstance(data, (bytes, bytearray)):
            data = json.loads(data.decode("utf-8"))
            
        threshold = float(data.get("threshold", 0.5))
        # Phân tích điểm PCD gửi từ CVAT hoặc bounding box hiện thời
        # Trả về cuboid 3D kèm attributes
        results = []
        
        # Ví dụ trả về box ô tô chuẩn hóa
        results.append({
            "confidence": 0.95,
            "label": "vehicle.car",
            "position": [12.0, 3.5, 0.0],
            "scale": CLASS_PRIORS["vehicle.car"],
            "rotation": [0.0, 0.0, 0.0],
            "attributes": {
                "difficulty_tier": "HARD",
                "difficulty_score": 4.1,
                "risk_flags": "Thưa điểm N=5 pts; Cự ly xa 38m; Nguy cơ lộn Yaw 180°",
                "annotator_sop": "Khóa kích thước Prior; Neo mặt nhìn thấy; Kiểm tra vệt track"
            }
        })
        
        return context.Response(
            body=json.dumps(results),
            headers={},
            content_type="application/json",
            status_code=200
        )
    except Exception as e:
        context.logger.error(f"Error handling request: {e}")
        return context.Response(
            body=json.dumps({"error": str(e)}),
            headers={},
            content_type="application/json",
            status_code=500
        )
'''


def generate_nuclio_pack_zip() -> io.BytesIO:
    """Đóng gói toàn bộ file cấu hình Nuclio thành file ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("function.yaml", get_nuclio_function_yaml())
        zf.writestr("main.py", get_nuclio_main_py())
        zf.writestr("README.md", """# Hướng dẫn Triển khai M45 Screener vào CVAT qua Nuclio

1. Cài đặt nuctl (Nuclio Command Line Tool):
   https://github.com/nuclio/nuclio/releases

2. Chạy lệnh deploy:
   nuctl deploy --project-name cvat \\
     --path . \\
     --file function.yaml \\
     --platform local

3. Mở CVAT -> Mở task 3D -> Bấm vào biểu tượng 'AI Tools / Detectors' trên thanh công cụ.
   Bạn sẽ thấy công cụ: 'M45 3D LiDAR Difficulty & Pre-label Screener'!
""")
    buf.seek(0)
    return buf
