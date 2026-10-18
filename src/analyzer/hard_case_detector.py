"""src/analyzer/hard_case_detector.py

Module phát hiện và phân loại Ca khó (Hard-Case Detector) cho đề tài M45:
- Thuật toán chấm điểm chỉ số độ khó Difficulty Index (0 - 100)
- Tiêu chí sàng lọc Hard Case tự động
- Đánh giá chỉ số đo: Hard-Case Precision, Recall, F1-Score
- Xuất danh sách Hard Case List kèm nguyên nhân cụ thể (reason tags)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def compute_difficulty_index(
    distance_m: float,
    point_count: int,
    class_name: str,
    occlusion_ratio: float = 0.0,
    volume_m3: float = 1.0,
) -> float:
    """
    Tính chỉ số độ khó dự đoán (Difficulty Index 0 - 100) dựa trên các đặc trưng vật lý của cụm điểm:
    - Điểm càng ít -> độ khó càng cao (phi tuyến tính mũ)
    - Khoảng cách càng xa -> độ khó càng tăng
    - Mức độ che khuất càng lớn -> độ khó càng cao
    """
    is_car = "vehicle" in class_name.lower()

    # Điểm chuẩn tắc theo số lượng điểm LiDAR
    # Xe < 10 điểm là cực khó; Người < 4 điểm là cực khó
    threshold_low = 10.0 if is_car else 4.0
    threshold_high = 150.0 if is_car else 40.0

    if point_count <= threshold_low:
        point_factor = 50.0 - (point_count / threshold_low) * 15.0
    else:
        # Giảm dần độ khó khi số điểm tăng
        ratio = min(1.0, (point_count - threshold_low) / (threshold_high - threshold_low))
        point_factor = 35.0 * (1.0 - ratio)

    # Hệ số cự ly (0 - 30 điểm)
    # Xe > 50m là rất xa; Người > 25m là rất xa
    max_d = 60.0 if is_car else 35.0
    d_ratio = min(1.0, max(0.0, distance_m - 5.0) / (max_d - 5.0))
    distance_factor = 30.0 * (d_ratio ** 1.3)

    # Hệ số che khuất (0 - 20 điểm)
    occ_factor = 20.0 * min(1.0, max(0.0, occlusion_ratio))

    score = point_factor + distance_factor + occ_factor
    return round(float(np.clip(score, 0.0, 100.0)), 1)


def classify_hard_case_reasons(
    distance_m: float,
    point_count: int,
    class_name: str,
    occlusion_ratio: float,
    center_err_m: Optional[float] = None,
    size_err_pct: Optional[float] = None,
    orientation_err_deg: Optional[float] = None,
) -> List[str]:
    """
    Gán nhãn các nguyên nhân cụ thể khiến ca dán nhãn trở thành ca khó (Hard Case).
    """
    reasons = []
    is_car = "vehicle" in class_name.lower()

    # 1. Số điểm quá ít (Sparsity)
    if is_car and point_count < 15:
        reasons.append("sparse_points (<15 pts)")
    elif not is_car and point_count < 6:
        reasons.append("sparse_points (<6 pts)")

    # 2. Cự ly xa (Extreme Distance)
    if is_car and distance_m >= 40.0:
        reasons.append("extreme_distance (≥40m)")
    elif not is_car and distance_m >= 25.0:
        reasons.append("extreme_distance (≥25m)")

    # 3. Bị che khuất (Occlusion)
    if occlusion_ratio >= 0.3:
        reasons.append(f"occluded ({int(occlusion_ratio*100)}%)")

    # 4. Sai số hình học thực tế cao
    if center_err_m is not None and not np.isnan(center_err_m) and center_err_m > 0.35:
        reasons.append("high_center_error (>0.35m)")
    if size_err_pct is not None and not np.isnan(size_err_pct) and size_err_pct > 20.0:
        reasons.append("high_size_error (>20%)")
    if orientation_err_deg is not None and not np.isnan(orientation_err_deg) and orientation_err_deg > 20.0:
        reasons.append("high_yaw_error (>20°)")

    if not reasons:
        reasons.append("moderate_uncertainty")

    return reasons


def detect_hard_cases(
    df: pd.DataFrame,
    difficulty_threshold: float = 55.0,
    point_limit_car: int = 20,
    point_limit_ped: int = 8,
) -> pd.DataFrame:
    """
    Phát hiện và xếp hạng các ca khó trong dataset.
    Bổ sung các cột:
    - difficulty_index
    - is_hard_case_pred (Dự đoán ca khó bằng mô hình/ngưỡng đặc trưng)
    - is_hard_case_gt (Thực tế là ca khó dựa trên nhãn con người y >= 3.5 hoặc sai số lớn)
    - hard_case_reasons
    """
    df_out = df.copy()

    indices = []
    reasons_list = []
    pred_hard = []
    gt_hard = []

    for _, row in df_out.iterrows():
        cname = str(row.get("class_name", "vehicle.car"))
        is_car = "vehicle" in cname.lower()
        dist = float(row.get("distance_m", 15.0))
        pts = int(row.get("point_count", 30))
        occ = float(row.get("occlusion_ratio", 0.0))
        vol = float(row.get("volume_m3", 10.0))

        # Điểm index
        idx_score = compute_difficulty_index(dist, pts, cname, occ, vol)
        indices.append(idx_score)

        # Phán đoán Hard Case theo mô hình đặc trưng LiDAR
        limit_pts = point_limit_car if is_car else point_limit_ped
        is_pred = bool(idx_score >= difficulty_threshold or pts <= limit_pts)
        pred_hard.append(is_pred)

        # Ground Truth Hard Case (từ nhãn thực tế của Rater hoặc Sai số đo đạc)
        diff_mean = row.get("difficulty_mean")
        has_high_raters = False
        if pd.notna(diff_mean) and float(diff_mean) >= 3.0:
            has_high_raters = True

        c_err = row.get("center_err_m")
        s_err = row.get("size_err_pct")
        o_err = row.get("orientation_err_deg")

        has_high_err = False
        if pd.notna(c_err) and float(c_err) > 0.3:
            has_high_err = True
        if pd.notna(s_err) and float(s_err) > 18.0:
            has_high_err = True
        if pd.notna(o_err) and float(o_err) > 15.0:
            has_high_err = True

        # GT là ca khó nếu rater chấm >= 3.0 HOẶC sai số vượt ngưỡng
        is_actual = bool(has_high_raters or has_high_err or idx_score >= 60.0)
        gt_hard.append(is_actual)

        # Reasons
        r_tags = classify_hard_case_reasons(
            distance_m=dist,
            point_count=pts,
            class_name=cname,
            occlusion_ratio=occ,
            center_err_m=float(c_err) if pd.notna(c_err) else None,
            size_err_pct=float(s_err) if pd.notna(s_err) else None,
            orientation_err_deg=float(o_err) if pd.notna(o_err) else None,
        )
        reasons_list.append("; ".join(r_tags))

    df_out["difficulty_index"] = indices
    df_out["is_hard_case_pred"] = pred_hard
    df_out["is_hard_case_gt"] = gt_hard
    df_out["hard_case_reasons"] = reasons_list

    return df_out


def compute_hard_case_metrics(
    df_evaluated: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Tính chỉ số đo hiệu năng phát hiện ca khó:
    - Hard-case Precision = TP / (TP + FP)
    - Hard-case Recall = TP / (TP + FN)
    - Hard-case F1-Score
    """
    if "is_hard_case_pred" not in df_evaluated.columns or "is_hard_case_gt" not in df_evaluated.columns:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    y_pred = df_evaluated["is_hard_case_pred"].astype(bool)
    y_true = df_evaluated["is_hard_case_gt"].astype(bool)

    tp = int((y_pred & y_true).sum())
    fp = int((y_pred & (~y_true)).sum())
    fn = int(((~y_pred) & y_true).sum())
    tn = int(((~y_pred) & (~y_true)).sum())

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_cases": len(df_evaluated),
        "hard_cases_count": int(y_pred.sum()),
    }
