"""src/duong/run_duong.py

CLI Runner & Verification Suite cho Vai trò ĐƯỜNG (Mô hình hóa & Machine Learning).
Dự án: M45 - LiDAR Point Density & Annotation Difficulty Analyzer
Thực hiện:
1. Huấn luyện mô hình Ridge Regression từ nhãn rater (y_table.csv) và đặc trưng hình học (dung_features.csv).
2. Dự đoán độ khó [1.0, 5.0] cho toàn bộ tập dữ liệu.
3. Xuất kết quả và đánh giá các chỉ số cốt lõi đề tài (Correlation density-error, Hard-case precision/recall/F1).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.duong.train_and_predict import train_and_predict_pipeline


def print_duong_report(results: dict) -> None:
    print("\n" + "=" * 68)
    print("      BÁO CÁO MÔ HÌNH HÓA ĐỘ KHÓ & ĐÁNH GIÁ (VAI TRÒ ĐƯỜNG)")
    print("=" * 68)

    print(f"[*] Thuật toán:       {results['model_type']}")
    print(f"[*] Tham số điều chuẩn: alpha = {results['alpha']}")
    print(f"[*] Hệ số bias học được: b = {results['bias']}")
    print(f"[*] Số mẫu huấn luyện: {results['train_samples_count']} / {results['total_samples_count']}")

    print("\n--- TRỌNG SỐ ĐẶC TRƯNG HỌC ĐƯỢC (FEATURE IMPORTANCES) ---")
    weights = results["feature_weights"]
    for feat_name, w in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  • {feat_name:<24}: {w:>8.4f}")

    reg = results["regression_metrics"]
    print("\n--- CHỈ SỐ SAI SỐ HỒI QUY (REGRESSION METRICS) ---")
    print(f"  • MAE (Mean Absolute Error)     : {reg['mae']}")
    print(f"  • RMSE (Root Mean Squared Error): {reg['rmse']}")
    print(f"  • Pearson Correlation r         : {reg['pearson_r']}")
    print(f"  • Spearman Correlation rho      : {reg['spearman_rho']}")

    hard = results["hard_case_metrics"]
    print("\n--- CHỈ SỐ ĐO PHÁT HIỆN CA KHÓ (HARD-CASE DETECTION) ---")
    print(f"  • Ngưỡng Ca khó (Difficulty Threshold) : >= {hard['threshold']}")
    print(f"  • Confusion Matrix (TP / FP / FN / TN) : {hard['true_positives']} / {hard['false_positives']} / {hard['false_negatives']} / {hard['true_negatives']}")
    print(f"  • Hard-case Precision                  : {hard['hard_case_precision']}%")
    print(f"  • Hard-case Recall                     : {hard['hard_case_recall']}%")
    print(f"  • Hard-case F1-Score                   : {hard['hard_case_f1']}")

    print("\n" + "=" * 68)
    print(" [✓] HUẤN LUYỆN VÀ ĐÁNH GIÁ VAI TRÒ ĐƯỜNG HOÀN TẤT THÀNH CÔNG!")
    print("=" * 68)


def main() -> None:
    parser = argparse.ArgumentParser(description="M45 Runner cho Vai trò ĐƯỜNG - Mô hình hóa & Machine Learning")
    parser.add_argument("--alpha", type=float, default=1.0, help="Tham số điều chuẩn L2 Ridge")
    args = parser.parse_args()

    print("🤖 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH HỒI QUY RIDGE VAI TRÒ ĐƯỜNG...")
    df_pred, results = train_and_predict_pipeline(alpha=args.alpha)
    print_duong_report(results)


if __name__ == "__main__":
    main()
