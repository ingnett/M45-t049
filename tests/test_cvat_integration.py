"""tests/test_cvat_integration.py

Bộ kiểm thử tự động toàn diện cho mô-đun Tích hợp CVAT 3D (src/analyzer/cvat_integration.py).
Kiểm tra các luồng:
1. Xuất CVAT 3D XML (v1.1) kèm đầy đủ metadata và attributes độ khó.
2. Xuất CVAT Datumaro 3D JSON (v1.0).
3. Đóng gói ZIP trọn gói cho CVAT 3D (XML, JSON, PCD, CSV, INSTRUCTIONS).
4. Phân tích (Parse) ngược từ file XML và JSON xuất từ CVAT.
5. Kiểm toán Chất lượng (QA Audit): Tính sai số tâm, hướng yaw, kích thước và IoU 3D so với Ground Truth.
6. Phát hiện các lỗi nghiêm trọng (Lộn góc 180°, Lệch tâm lớn, Co ngót kích thước).
7. CVAT REST API Client & Nuclio AI Tool Pack.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analyzer.cvat_integration import (
    export_cvat_3d_xml,
    export_datumaro_3d_json,
    create_cvat_task_pack_zip,
    parse_cvat_export_file,
    calculate_box_iou_bev,
    audit_cvat_annotations_vs_gt,
    CVATApiClient,
    generate_nuclio_pack_zip,
)


def get_mock_boxes_data():
    """Tạo tập dữ liệu mẫu gồm 3 ca: DỄ, TRUNG BÌNH, KHÓ."""
    return [
        {
            "case_id": "c001_easy",
            "class_name": "vehicle.car",
            "center_m": [12.0, 0.0, 0.5],
            "size_wlh_m": [1.93, 4.62, 1.69],
            "yaw_rad": 0.0,
            "N": 45,
            "distance_m": 12.0,
            "occlusion_ratio": 0.0,
        },
        {
            "case_id": "c002_med",
            "class_name": "vehicle.car",
            "center_m": [26.0, 5.0, 0.5],
            "size_wlh_m": [1.93, 4.62, 1.69],
            "yaw_rad": 0.5,
            "N": 15,
            "distance_m": 26.5,
            "occlusion_ratio": 0.2,
        },
        {
            "case_id": "c003_hard",
            "class_name": "vehicle.car",
            "center_m": [52.0, 10.0, 0.5],
            "size_wlh_m": [1.93, 4.62, 1.69],
            "yaw_rad": 1.57,
            "N": 4,
            "distance_m": 53.0,
            "occlusion_ratio": 0.5,
        },
    ]


def test_export_cvat_3d_xml():
    boxes = get_mock_boxes_data()
    xml_str = export_cvat_3d_xml(boxes, task_name="Test_Task")
    
    # Kiểm tra tính hợp lệ của XML
    root = ET.fromstring(xml_str)
    assert root.tag == "annotations"
    assert root.find("version").text == "1.1"
    
    # Kiểm tra số lượng ảnh/frame
    images = root.findall("image")
    assert len(images) == 3
    
    # Kiểm tra thẻ cuboid và attributes của ca khó
    hard_img = images[2]
    cuboid = hard_img.find("cuboid")
    assert cuboid is not None
    assert cuboid.get("label") == "vehicle.car"
    
    attrs = {a.get("name"): a.text for a in cuboid.findall("attribute")}
    assert "difficulty_tier" in attrs
    assert attrs["difficulty_tier"] == "HARD"
    assert "difficulty_score" in attrs
    assert float(attrs["difficulty_score"]) >= 3.6
    assert "annotator_sop" in attrs
    print("✅ test_export_cvat_3d_xml passed.")


def test_export_datumaro_3d_json():
    boxes = get_mock_boxes_data()
    doc = export_datumaro_3d_json(boxes, dataset_name="Test_Datumaro")
    
    assert doc["info"]["version"] == "1.0"
    assert len(doc["items"]) == 3
    
    easy_item = doc["items"][0]
    ann = easy_item["annotations"][0]
    assert ann["type"] == "cuboid_3d"
    assert ann["attributes"]["difficulty_tier"] == "EASY"
    assert ann["attributes"]["is_hard_case"] is False
    
    hard_item = doc["items"][2]
    hard_ann = hard_item["annotations"][0]
    assert hard_ann["attributes"]["difficulty_tier"] == "HARD"
    assert hard_ann["attributes"]["is_hard_case"] is True
    print("✅ test_export_datumaro_3d_json passed.")


def test_create_cvat_task_pack_zip():
    boxes = get_mock_boxes_data()
    zip_buf = create_cvat_task_pack_zip(boxes, task_name="M45_Pack_Test")
    
    with zipfile.ZipFile(zip_buf, "r") as zf:
        namelist = zf.namelist()
        assert "annotations.xml" in namelist
        assert "datumaro_annotations.json" in namelist
        assert "INSTRUCTIONS.txt" in namelist
        assert "classes.csv" in namelist
        assert "point_clouds/c001_easy.pcd" in namelist
        assert "point_clouds/c003_hard.pcd" in namelist
        
        # Đọc thử file PCD
        pcd_text = zf.read("point_clouds/c001_easy.pcd").decode("utf-8")
        assert "VERSION 0.7" in pcd_text
        assert "DATA ascii" in pcd_text
    print("✅ test_create_cvat_task_pack_zip passed.")


def test_parse_and_audit_roundtrip():
    boxes = get_mock_boxes_data()
    
    # 1. Sinh XML
    xml_str = export_cvat_3d_xml(boxes)
    parsed_boxes = parse_cvat_export_file(xml_str, "test.xml")
    assert len(parsed_boxes) == 3
    assert "c001_easy" in parsed_boxes
    assert "c003_hard" in parsed_boxes
    
    # 2. Tạo DataFrame GT
    gt_df = pd.DataFrame([
        {
            "case_id": b["case_id"],
            "class_name": b["class_name"],
            "_center": b["center_m"],
            "_wlh": b["size_wlh_m"],
            "_yaw": b["yaw_rad"],
            "N": b["N"],
            "distance_m": b["distance_m"],
            "occlusion_ratio": b["occlusion_ratio"],
        }
        for b in boxes
    ])
    
    # 3. Chạy Audit khi nhãn hoàn hảo (perfect match)
    perfect_audit = audit_cvat_annotations_vs_gt(parsed_boxes, gt_df)
    assert perfect_audit["total_evaluated"] == 3
    assert perfect_audit["total_missing"] == 0
    # Sai số tâm phải gần như 0
    assert perfect_audit["tier_summary"]["EASY"]["mean_center_err_m"] < 1e-3
    assert perfect_audit["tier_summary"]["EASY"]["mean_iou_3d"] > 0.95
    
    # 4. Mô phỏng Annotator mắc lỗi:
    # - Ca dễ: lệch nhẹ 0.05m
    # - Ca khó: lộn góc Yaw 180 độ và lệch tâm 0.8m
    import copy
    flawed_boxes = copy.deepcopy(parsed_boxes)
    flawed_boxes["c003_hard"]["yaw_rad"] = float(parsed_boxes["c003_hard"]["yaw_rad"] + math.pi)
    flawed_boxes["c003_hard"]["center_m"][0] += 0.8

    flawed_audit = audit_cvat_annotations_vs_gt(flawed_boxes, gt_df)
    hard_summary = flawed_audit["tier_summary"]["HARD"]
    assert hard_summary["yaw_flip_count"] == 1, f"Expected 1 yaw flip, got {hard_summary['yaw_flip_count']}"
    assert hard_summary["severe_error_rate_pct"] == 100.0
    assert hard_summary["mean_center_err_m"] >= 0.7

    audit_df = flawed_audit["audit_df"]
    hard_row = audit_df[audit_df["case_id"] == "c003_hard"].iloc[0]
    assert bool(hard_row["is_yaw_flipped"]) is True, f"Expected is_yaw_flipped True, row was: {hard_row.to_dict()}"
    assert bool(hard_row["is_severe_error"]) is True
    print("✅ test_parse_and_audit_roundtrip passed.")


def test_cvat_api_client_and_nuclio():
    client = CVATApiClient(base_url="http://localhost:8080", token="demo_token_123")
    payload = client.generate_task_payload("M45_Task", ["vehicle.car", "human.pedestrian.adult"])
    assert payload["name"] == "M45_Task"
    assert len(payload["labels"]) == 2
    assert payload["labels"][0]["attributes"][0]["name"] == "difficulty_tier"
    
    diag = client.test_connection()
    assert "simulated_curl" in diag or diag["connected"] is True
    
    nuclio_buf = generate_nuclio_pack_zip()
    with zipfile.ZipFile(nuclio_buf, "r") as zf:
        names = zf.namelist()
        assert "function.yaml" in names
        assert "main.py" in names
        assert "README.md" in names
    print("✅ test_cvat_api_client_and_nuclio passed.")


if __name__ == "__main__":
    test_export_cvat_3d_xml()
    test_export_datumaro_3d_json()
    test_create_cvat_task_pack_zip()
    test_parse_and_audit_roundtrip()
    test_cvat_api_client_and_nuclio()
    print("\n🎉 TOÀN BỘ 5/5 TEST CVAT INTEGRATION ĐỀU ĐẠT THÀNH CÔNG!")
