"""src/analyzer/box_difficulty_rules.py

Hệ thống Quy tắc Định lượng Đánh giá Box Khó và Box Dễ (Box Difficulty Rules Engine)
Dự án: M45 - LiDAR Point Density & Annotation Difficulty Analyzer
Phục vụ: Annotator, AI Pre-labeling Screener, Quality Assurance (QA).
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# Ngưỡng định lượng chuẩn theo vật lý và kinh nghiệm gán nhãn nuScenes / Waymo
DIFFICULTY_THRESHOLDS = {
    "vehicle.car": {
        "easy_min_points": 25,
        "medium_min_points": 8,
        "easy_max_distance": 20.0,
        "medium_max_distance": 35.0,
        "prior_wlh": [1.93, 4.62, 1.69],
        "safe_vol_density": 2.5,
    },
    "human.pedestrian.adult": {
        "easy_min_points": 10,
        "medium_min_points": 4,
        "easy_max_distance": 15.0,
        "medium_max_distance": 25.0,
        "prior_wlh": [0.67, 0.73, 1.77],
        "safe_vol_density": 10.0,
    },
}


def evaluate_box_difficulty(
    point_count: int,
    distance_m: float,
    class_name: str = "vehicle.car",
    occlusion_ratio: float = 0.0,
    visible_faces_count: int = 2,
    center_err_m: Optional[float] = None,
    yaw_err_deg: Optional[float] = None,
    size_err_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Đánh giá định lượng toàn diện xem một Bounding Box là DỄ (EASY), TRUNG BÌNH (MEDIUM) hay KHÓ (HARD).
    
    Trả về:
    - tier: 'EASY', 'MEDIUM', 'HARD'
    - difficulty_score: float từ 1.0 (cực dễ) đến 5.0 (cực khó)
    - difficulty_index: float từ 0 (dễ) đến 100 (cực khó)
    - risk_badge: Nhãn trực quan (🟢 DỄ, 🟡 TRUNG BÌNH, 🔴 KHÓ)
    - reasons: Danh sách nguyên nhân cụ thể
    - errors_predicted: Danh sách lỗi mà annotator hoặc mô hình AI hay mắc phải
    - annotator_sop: Quy trình 3 bước thao tác chuẩn cho Annotator
    """
    is_car = "car" in class_name.lower() or "vehicle" in class_name.lower()
    cfg = DIFFICULTY_THRESHOLDS["vehicle.car"] if is_car else DIFFICULTY_THRESHOLDS["human.pedestrian.adult"]

    reasons: List[str] = []
    errors_predicted: List[str] = []
    
    # 1. Tính toán điểm thành phần theo độ thưa điểm N
    if point_count <= 2:
        pt_score = 5.0
        reasons.append("Cực kỳ thưa điểm (N ≤ 2 điểm)")
        errors_predicted.append("Mất hoàn toàn biên giới 3D; phỏng đoán mơ hồ")
    elif point_count < cfg["medium_min_points"]:
        pt_score = 4.2
        reasons.append(f"Thưa điểm nghiêm trọng (N < {cfg['medium_min_points']} pts)")
        errors_predicted.append("Co cụm kích thước (Size shrinkage); Lệch tâm (Center drift)")
    elif point_count < cfg["easy_min_points"]:
        pt_score = 2.8
        reasons.append(f"Mật độ điểm mức trung bình ({cfg['medium_min_points']} ≤ N < {cfg['easy_min_points']} pts)")
    else:
        pt_score = 1.3
        reasons.append(f"Mật độ điểm dồi dào (N ≥ {cfg['easy_min_points']} pts)")

    # 2. Tính toán điểm thành phần theo Cự ly d
    if distance_m >= (50.0 if is_car else 30.0):
        dist_score = 4.8
        reasons.append(f"Cự ly rất xa (d = {distance_m:.1f}m ≥ 50m)")
        errors_predicted.append("Góc mở chùm tia LiDAR giãn rộng; suy giảm 95% số tia phản xạ")
    elif distance_m >= cfg["medium_max_distance"]:
        dist_score = 3.8
        reasons.append(f"Cự ly xa ({cfg['medium_max_distance']}m ≤ d < {distance_m:.1f}m)")
    elif distance_m >= cfg["easy_max_distance"]:
        dist_score = 2.4
        reasons.append(f"Cự ly trung bình ({cfg['easy_max_distance']}m ≤ d < {cfg['medium_max_distance']}m)")
    else:
        dist_score = 1.2
        reasons.append(f"Cự ly gần an toàn (d < {cfg['easy_max_distance']}m)")

    # 3. Phân tích nguy cơ lộn góc Yaw (±180 deg)
    if is_car and (distance_m >= 32.0 or point_count < 10 or visible_faces_count <= 1):
        errors_predicted.append("Nguy cơ lộn góc xoay Yaw ±180° do thiếu điểm đầu/đuôi xe")
        reasons.append("Mất mặt trước/sau để xác định hướng tiến")

    # 4. Phân tích mức độ che khuất (Occlusion)
    if occlusion_ratio >= 0.4:
        reasons.append(f"Bị che khuất nặng ({int(occlusion_ratio * 100)}%)")
        errors_predicted.append("Cắt cụt biên dạng; tâm bị kéo lệch về phía nhìn thấy")
        occ_penalty = 1.2
    elif occlusion_ratio >= 0.15:
        reasons.append(f"Che khuất nhẹ ({int(occlusion_ratio * 100)}%)")
        occ_penalty = 0.5
    else:
        occ_penalty = 0.0

    # 5. Tổng hợp điểm độ khó (1.0 - 5.0)
    raw_score = 0.55 * pt_score + 0.35 * dist_score + 0.10 * occ_penalty
    difficulty_score = float(np.clip(round(raw_score, 2), 1.0, 5.0))
    difficulty_index = float(np.clip(round((difficulty_score - 1.0) / 4.0 * 100.0, 1), 0.0, 100.0))

    # 6. Phân định Cấp độ (Tier)
    # Nguyên tắc an toàn xe tự hành: Nếu N cực thưa HOẶC Cự ly rất xa -> tự động xếp vào HARD
    is_hard_condition = (
        point_count < cfg["medium_min_points"]
        or distance_m >= cfg["medium_max_distance"]
        or occlusion_ratio >= 0.4
        or difficulty_score >= 3.6
    )

    is_easy_condition = (
        point_count >= cfg["easy_min_points"]
        and distance_m < cfg["easy_max_distance"]
        and occlusion_ratio < 0.15
        and difficulty_score <= 2.2
    )

    if is_hard_condition:
        tier = "HARD"
        risk_badge = "🔴 BOX KHÓ (HARD)"
        annotator_sop = (
            "1. KHÓA KÍCH THƯỚC: Áp dụng Class Prior chuẩn, không tự ý co nhỏ box.\n"
            "2. NEO MẶT NHÌN THẤY: Căn mép hộp sát vào điểm LiDAR đầu tiên (Visible face snap).\n"
            "3. XÁC MINH YAW: Bật ảnh camera RGB hoặc vệt Track liên khung để tránh lộn đầu/đuôi 180°."
        )
    elif is_easy_condition:
        tier = "EASY"
        risk_badge = "🟢 BOX DỄ (EASY)"
        annotator_sop = "Duyệt nhanh: Cụm điểm rõ nét L-shape, chỉ cần 5s kiểm tra tâm và kích thước."
    else:
        tier = "MEDIUM"
        risk_badge = "🟡 TRUNG BÌNH (MEDIUM)"
        annotator_sop = (
            "1. Kiểm tra 2 góc nhìn (Top & Side view).\n"
            "2. Căn chỉnh viền hộp khớp với điểm phản xạ ngoài cùng.\n"
            "3. Kiểm tra độ cao đáy xe cách mặt đường ~0.1m."
        )

    # Nếu không có lỗi tiềm ẩn nào thì gán nhãn an toàn
    if not errors_predicted:
        errors_predicted.append("Không phát hiện rủi ro hình học nghiêm trọng")

    return {
        "tier": tier,
        "difficulty_score": difficulty_score,
        "difficulty_index": difficulty_index,
        "risk_badge": risk_badge,
        "reasons": reasons,
        "reasons_str": "; ".join(reasons),
        "errors_predicted": errors_predicted,
        "errors_predicted_str": "; ".join(errors_predicted),
        "annotator_sop": annotator_sop,
        "prior_wlh": cfg["prior_wlh"],
    }


def explain_rule_summary() -> str:
    """Trả về chuỗi tóm tắt bộ quy tắc bằng Markdown để hiển thị trên UI."""
    return r"""
### 📋 BỘ QUY TẮC ĐỊNH LƯỢNG M45: THẾ NÀO LÀ BOX DỄ vs BOX KHÓ?

1. **🟢 BOX DỄ (EASY - Điểm 1.0 – 2.0 | Index ≤ 35):**
   - **Ô tô:** $N \ge 25$ điểm, cự ly gần $d < 20\text{m}$, không che khuất.
   - **Người đi bộ:** $N \ge 10$ điểm, cự ly gần $d < 15\text{m}$.
   - **Hình học:** Rõ ít nhất 2 mặt vuông góc (L-Shape), annotator hoàn tất dưới 35 giây.

2. **🟡 BOX TRUNG BÌNH (MEDIUM - Điểm 2.1 – 3.5 | Index 36 – 65):**
   - **Ô tô:** $8 \le N < 25$ điểm, cự ly trung bình $20\text{m} \le d < 35\text{m}$, che khuất nhẹ $<35\%$.
   - **Người đi bộ:** $4 \le N < 10$ điểm, cự ly $15\text{m} \le d < 25\text{m}$.
   - **Hình học:** Thấy 1 mặt rõ, cần đối chiếu góc nhìn và chỉnh trong 35–75 giây.

3. **🔴 BOX KHÓ (HARD - Điểm 3.6 – 5.0 | Index 66 – 100):**
   - **Ô tô:** $N < 8$ điểm **HOẶC** cự ly xa $d \ge 35\text{m}$ **HOẶC** che khuất $\ge 40\%$.
   - **Người đi bộ:** $N < 4$ điểm **HOẶC** cự ly xa $d \ge 25\text{m}$.
   - **3 Rủi ro chí mạng:** Lộn góc Yaw $\pm 180^\circ$, Lệch tâm $>0.35\text{m}$, Co cụm kích thước.
   - **SOP Bắt buộc:** Khóa Class Prior, neo mặt nhìn thấy, đối chiếu Camera RGB.
"""


if __name__ == "__main__":
    # Test thử 3 trường hợp tiêu biểu
    print("=== KIỂM THỬ BỘ QUY TẮC PHÂN LOẠI M45 ===")
    
    # Ca 1: Xe gần, nhiều điểm
    res1 = evaluate_box_difficulty(point_count=50, distance_m=12.0, class_name="vehicle.car")
    print(f"\n[Ca 1] Xe 12m, 50 pts  -> {res1['risk_badge']} (Score: {res1['difficulty_score']}, Index: {res1['difficulty_index']})")
    
    # Ca 2: Xe trung bình
    res2 = evaluate_box_difficulty(point_count=18, distance_m=26.0, class_name="vehicle.car")
    print(f"[Ca 2] Xe 26m, 18 pts  -> {res2['risk_badge']} (Score: {res2['difficulty_score']}, Index: {res2['difficulty_index']})")
    
    # Ca 3: Xe xa, ít điểm (Ca khó)
    res3 = evaluate_box_difficulty(point_count=3, distance_m=56.0, class_name="vehicle.car")
    print(f"[Ca 3] Xe 56m, 3 pts   -> {res3['risk_badge']} (Score: {res3['difficulty_score']}, Index: {res3['difficulty_index']})")
    print("  • Lý do:", res3["reasons_str"])
    print("  • Lỗi tiềm ẩn:", res3["errors_predicted_str"])
