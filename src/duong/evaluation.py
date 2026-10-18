"""src/duong/evaluation.py

Module Đánh giá Mô hình và Chỉ số Đo cốt lõi của Đề tài M45.
Chỉ số đo:
1. Sai số dự đoán: MAE, RMSE
2. Hệ số tương quan:
   - Pearson r
   - Spearman rank correlation rho (Correlation density-error)
3. Phát hiện Ca khó (Hard-case Detection, ngưỡng difficulty >= 4):
   - Hard-case Precision
   - Hard-case Recall
   - Hard-case F1-score
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple
import numpy as np
import scipy.stats as stats


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Tính toán MAE, RMSE, Pearson r, Spearman rho."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Pearson
    if np.std(y_true) > 1e-6 and np.std(y_pred) > 1e-6:
        pearson_r, _ = stats.pearsonr(y_true, y_pred)
        spearman_rho, _ = stats.spearmanr(y_true, y_pred)
    else:
        pearson_r = 1.0
        spearman_rho = 1.0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "pearson_r": round(float(pearson_r), 4),
        "spearman_rho": round(float(spearman_rho), 4),
    }


def compute_hard_case_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 4.0
) -> Dict[str, Any]:
    """
    Tính ma trận nhầm lẫn và các chỉ số đo cho bài toán phân loại Ca khó (Hard Case).
    Định nghĩa Ca khó: nhãn >= threshold (mặc định 4.0).
    """
    actual_hard = y_true >= threshold
    predicted_hard = y_pred >= threshold

    tp = int(np.sum(actual_hard & predicted_hard))
    fp = int(np.sum((~actual_hard) & predicted_hard))
    fn = int(np.sum(actual_hard & (~predicted_hard)))
    tn = int(np.sum((~actual_hard) & (~predicted_hard)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "threshold": threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "hard_case_precision": round(precision * 100.0, 2),
        "hard_case_recall": round(recall * 100.0, 2),
        "hard_case_f1": round(f1, 4),
    }
