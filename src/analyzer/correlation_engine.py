"""src/analyzer/correlation_engine.py

Module tính toán tương quan (Correlation Engine) cho đề tài M45:
- Correlation density-error (Pearson r & Spearman rho, p-value)
- Phân tích tương quan giữa Mật độ điểm / Số điểm / Khoảng cách và các Sai số dán nhãn (Lệch tâm, Lệch kích thước, Lệch hướng, Điểm độ khó y)
- Phân nhóm thống kê theo Class và Distance Bin
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def compute_correlation_pair(
    x_series: pd.Series,
    y_series: pd.Series,
) -> Dict[str, Any]:
    """
    Tính cả Pearson r và Spearman rho giữa 2 chuỗi số liệu (bỏ qua NaN/None).
    """
    valid_mask = (~x_series.isna()) & (~y_series.isna())
    x_clean = x_series[valid_mask].astype(float).to_numpy()
    y_clean = y_series[valid_mask].astype(float).to_numpy()

    n = len(x_clean)
    if n < 3 or float(np.std(x_clean)) < 1e-6 or float(np.std(y_clean)) < 1e-6:
        return {
            "n_samples": int(n),
            "pearson_r": 0.0,
            "pearson_p": 1.0,
            "spearman_rho": 0.0,
            "spearman_p": 1.0,
            "is_significant": False,
        }

    # Pearson
    try:
        p_r, p_p = stats.pearsonr(x_clean, y_clean)
    except Exception:
        p_r, p_p = 0.0, 1.0

    # Spearman
    try:
        s_r, s_p = stats.spearmanr(x_clean, y_clean)
    except Exception:
        s_r, s_p = 0.0, 1.0

    return {
        "n_samples": int(n),
        "pearson_r": round(float(p_r), 4),
        "pearson_p": round(float(p_p), 6),
        "spearman_rho": round(float(s_r), 4),
        "spearman_p": round(float(s_p), 6),
        "is_significant": bool(p_p < 0.05 or s_p < 0.05),
    }


def analyze_all_correlations(
    merged_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Chạy toàn diện phân tích tương quan giữa các biến mật độ/khoảng cách và sai số dán nhãn.
    """
    results: Dict[str, Any] = {}

    # Danh sách biến độc lập (X) và biến phụ thuộc sai số (Y)
    x_vars = ["point_count", "volumetric_density", "distance_m"]
    y_vars = ["center_err_m", "size_err_pct", "orientation_err_deg", "difficulty_mean"]

    matrix_rows = []

    for x_col in x_vars:
        if x_col not in merged_df.columns:
            continue
        for y_col in y_vars:
            if y_col not in merged_df.columns:
                continue

            # Toàn bộ tập
            overall_corr = compute_correlation_pair(merged_df[x_col], merged_df[y_col])

            # Theo từng class nếu có
            class_corrs = {}
            if "class_name" in merged_df.columns:
                for cname, group in merged_df.groupby("class_name"):
                    c_res = compute_correlation_pair(group[x_col], group[y_col])
                    class_corrs[str(cname)] = c_res

            matrix_rows.append({
                "x_var": x_col,
                "y_var": y_col,
                "pearson_r": overall_corr["pearson_r"],
                "pearson_p": overall_corr["pearson_p"],
                "spearman_rho": overall_corr["spearman_rho"],
                "spearman_p": overall_corr["spearman_p"],
                "n_samples": overall_corr["n_samples"],
                "is_significant": overall_corr["is_significant"],
                "by_class": class_corrs,
            })

    results["correlation_matrix"] = matrix_rows

    # Tổng hợp chỉ số đo chính của đề tài (Correlation density-error)
    # Tìm cặp tương quan tiêu biểu nhất: point_count vs difficulty_mean và point_count vs center_err_m
    primary_corr = 0.0
    for r in matrix_rows:
        if r["x_var"] == "point_count" and r["y_var"] == "difficulty_mean":
            primary_corr = r["spearman_rho"]
            break
        elif r["x_var"] == "point_count" and r["y_var"] == "center_err_m":
            primary_corr = r["spearman_rho"]

    results["key_metric_density_error_corr"] = primary_corr

    return results


def summarize_by_distance_and_class(
    merged_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tổng hợp bảng thống kê phân tích Mật độ theo Khoảng cách và Phân lớp đối tượng (Report density vs distance/class).
    """
    group_cols = []
    if "class_name" in merged_df.columns:
        group_cols.append("class_name")
    if "distance_bin" in merged_df.columns:
        group_cols.append("distance_bin")

    if not group_cols:
        return pd.DataFrame()

    summary = (
        merged_df.groupby(group_cols)
        .agg(
            n_cases=("case_id", "count"),
            avg_distance_m=("distance_m", "mean"),
            median_points=("point_count", "median"),
            min_points=("point_count", "min"),
            max_points=("point_count", "max"),
            avg_vol_density=("volumetric_density", "mean"),
            avg_center_err_m=("center_err_m", "mean"),
            avg_size_err_pct=("size_err_pct", "mean"),
            avg_orientation_err_deg=("orientation_err_deg", "mean"),
            avg_difficulty=("difficulty_mean", "mean"),
        )
        .reset_index()
    )

    # Làm tròn số
    for col in summary.columns:
        if summary[col].dtype == "float64":
            summary[col] = summary[col].round(3)

    return summary
