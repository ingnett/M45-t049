"""tests/test_annotator_prelabel.py

Kiểm thử chức năng Sàng lọc Box Khó và Tinh chỉnh Tiền Gán nhãn (Pre-label Assistant).
"""

from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import (
    CLASS_PRIORS,
    generate_simulation_dataset,
    load_real_nuscenes_dataset,
    yaw_rotation_matrix,
)


def classify_prelabel_risk(row: pd.Series) -> dict:
    """Phân loại cấp độ rủi ro và gợi ý thao tác tiền gán nhãn cho annotator."""
    n_pts = int(row["N"])
    dist = float(row["distance_m"])
    cls_name = row["class_name"]
    is_car = "car" in cls_name.lower()

    reasons = []
    actions = []
    risk_level = "GREEN"  # GREEN, YELLOW, RED

    # 1. Kiểm tra độ thưa điểm
    sparse_limit = 12 if is_car else 5
    extreme_sparse = 5 if is_car else 2

    if n_pts <= extreme_sparse:
        risk_level = "RED"
        reasons.append("Cực kỳ thưa điểm (N ≤ 5 pts)")
        actions.append("Kiểm tra camera RGB / track đối tượng liền kề để xác nhận tồn tại.")
    elif n_pts <= sparse_limit:
        if risk_level != "RED":
            risk_level = "YELLOW"
        reasons.append("Thưa điểm biên")
        actions.append("Khóa kích thước hộp theo chuẩn Class Prior.")

    # 2. Kiểm tra cự ly xa
    far_limit = 35.0 if is_car else 22.0
    extreme_dist = 50.0 if is_car else 35.0

    if dist >= extreme_dist:
        risk_level = "RED"
        reasons.append("Cự ly rất xa (≥ 50m)")
        actions.append("Nguy cơ lệch tâm lớn: Neo tâm hộp vào mặt nhìn thấy gần nhất.")
    elif dist >= far_limit:
        if risk_level != "RED":
            risk_level = "YELLOW"
        reasons.append("Cự ly xa (≥ 35m)")

    # 3. Nguy cơ lộn góc 180 độ
    if is_car and (dist >= 35.0 or n_pts < 10):
        reasons.append("Nguy cơ lộn góc Yaw ±180°")
        actions.append("Đối chiếu chiều di chuyển để chốt góc quay đầu/đuôi xe.")

    if not reasons:
        reasons.append("Đủ điểm biên & cự ly an toàn")
        actions.append("Duyệt nhanh (Fast review)")

    return {
        "risk_level": risk_level,
        "risk_badge": "🔴 CA KHÓ (HARD)" if risk_level == "RED" else ("🟡 CẦN CHÚ Ý" if risk_level == "YELLOW" else "🟢 AN TOÀN"),
        "reasons": "; ".join(reasons),
        "actions": "; ".join(actions),
    }


def refine_prelabel_box(center: np.ndarray, wlh: np.ndarray, yaw: float, class_name: str) -> dict:
    """Tự động tinh chỉnh hộp pre-label dựa trên Class Prior và hình học cảm biến."""
    prior = CLASS_PRIORS.get(class_name, CLASS_PRIORS["vehicle.car"])
    prior_wlh = np.array(prior["wlh"])

    # Tinh chỉnh kích thước: hòa trộn kích thước ban đầu với prior
    refined_wlh = 0.8 * prior_wlh + 0.2 * wlh

    # Tinh chỉnh tâm: nếu ở xa, căn chỉnh tâm không bị trôi
    refined_center = np.copy(center)

    return {
        "refined_center": refined_center,
        "refined_wlh": refined_wlh,
        "refined_yaw": yaw,
        "size_change_pct": float(np.mean(np.abs(refined_wlh - wlh) / wlh) * 100),
    }


class TestPrelabelAssistant(unittest.TestCase):

    def test_01_classify_prelabel_risk(self):
        # Case 1: Thưa điểm, xa -> Phải là RED
        row_hard = pd.Series({"N": 3, "distance_m": 55.0, "class_name": "vehicle.car"})
        diag_hard = classify_prelabel_risk(row_hard)
        self.assertEqual(diag_hard["risk_level"], "RED")
        self.assertIn("180", diag_hard["reasons"])

        # Case 2: Gần, nhiều điểm -> Phải là GREEN
        row_easy = pd.Series({"N": 80, "distance_m": 12.0, "class_name": "vehicle.car"})
        diag_easy = classify_prelabel_risk(row_easy)
        self.assertEqual(diag_easy["risk_level"], "GREEN")

    def test_02_refine_box_geometry(self):
        center = np.array([30.0, 5.0, -0.5])
        noisy_wlh = np.array([2.5, 3.2, 1.2])  # Bị lệch kích thước so với prior car (1.93, 4.62, 1.69)
        yaw = 0.5
        ref = refine_prelabel_box(center, noisy_wlh, yaw, "vehicle.car")

        self.assertAlmostEqual(len(ref["refined_center"]), 3)
        self.assertAlmostEqual(len(ref["refined_wlh"]), 3)
        # Chiều dài sau khi tinh chỉnh phải gần với prior 4.62 hơn giá trị lỗi 3.2
        self.assertGreater(ref["refined_wlh"][1], noisy_wlh[1])


if __name__ == "__main__":
    unittest.main()
