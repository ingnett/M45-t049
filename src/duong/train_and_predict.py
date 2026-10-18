"""src/duong/train_and_predict.py

Pipeline Huấn luyện và Dự đoán Độ khó cho Toàn bộ Dữ liệu.
Vai trò: ĐƯỜNG (Mô hình hóa & Machine Learning).
Dữ liệu vào:
- data/work/dung_features.csv (Đặc trưng hình học từ Dũng)
- data/labels/y_table.csv (Nhãn độ khó ground-truth tổng hợp từ Hưng)
Dữ liệu ra:
- outputs/difficulty_predictions.csv
- outputs/duong_model_results.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.duong.evaluation import compute_hard_case_metrics, compute_regression_metrics
from src.duong.model import M45RidgeDifficultyModel

WORK_DIR = REPO_ROOT / "data" / "work"
LABELS_DIR = REPO_ROOT / "data" / "labels"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def train_and_predict_pipeline(
    alpha: float = 1.0,
    features_csv: Optional[Path] = None,
    y_table_csv: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    feat_file = features_csv or (WORK_DIR / "dung_features.csv")
    y_file = y_table_csv or (LABELS_DIR / "y_table.csv")

    if not feat_file.exists():
        raise FileNotFoundError(f"Chưa có file đặc trưng {feat_file}. Hãy chạy src/dung/run_dung.py trước.")
    if not y_file.exists():
        raise FileNotFoundError(f"Chưa có file nhãn {y_file}. Hãy chạy src/hung/validate_ratings.py trước.")

    df_feats = pd.read_csv(feat_file)
    df_y = pd.read_csv(y_file)

    # Đổi tên chuẩn hóa để sử dụng thống nhất
    if "difficulty_mean" in df_y.columns:
        df_y["y_mean"] = df_y["difficulty_mean"]
    if "y_status" in df_y.columns:
        df_y["status"] = df_y["y_status"]
    if "excluded" not in df_y.columns:
        df_y["excluded"] = df_y.get("y_status", "ok") != "ok"

    # Ghép bảng dựa trên case_id
    df_merged = pd.merge(df_feats, df_y[["case_id", "y_mean", "status", "excluded"]], on="case_id", how="inner")

    # Tách tập train (các batch train: B1, B2) và test/dev
    train_mask = df_merged["batch_id"].isin(["B1", "B2"]) & (~df_merged["excluded"])
    if train_mask.sum() == 0:
        # Nếu chưa chia rõ thì dùng toàn bộ các ca không bị loại
        train_mask = ~df_merged["excluded"]

    df_train = df_merged[train_mask]
    y_train = df_train["y_mean"].to_numpy(dtype=np.float64)

    # Huấn luyện mô hình Ridge
    model = M45RidgeDifficultyModel(alpha=alpha)
    model.fit(df_train, y_train)

    # Dự đoán trên toàn bộ các ca
    y_pred_cont = model.predict_continuous(df_merged)
    y_pred_disc = model.predict(df_merged)

    df_merged["pred_difficulty_score"] = np.round(y_pred_cont, 2)
    df_merged["pred_difficulty_1_5"] = y_pred_disc
    df_merged["is_hard_predicted"] = y_pred_disc >= 4
    df_merged["is_hard_actual"] = df_merged["y_mean"] >= 4.0

    # Đánh giá chỉ số
    reg_metrics = compute_regression_metrics(
        df_merged["y_mean"].to_numpy(),
        y_pred_cont
    )
    hard_metrics = compute_hard_case_metrics(
        df_merged["y_mean"].to_numpy(),
        y_pred_cont,
        threshold=4.0
    )

    weights_dict = model.get_feature_importances()

    results: Dict[str, Any] = {
        "model_type": "Ridge Regression (Pure NumPy)",
        "alpha": alpha,
        "bias": round(model.bias, 4),
        "feature_weights": {k: round(v, 4) for k, v in weights_dict.items()},
        "regression_metrics": reg_metrics,
        "hard_case_metrics": hard_metrics,
        "train_samples_count": int(train_mask.sum()),
        "total_samples_count": len(df_merged),
    }

    # Xuất file kết quả
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUTS_DIR / "difficulty_predictions.csv"
    out_json = OUTPUTS_DIR / "duong_model_results.json"

    export_cols = [
        "case_id", "batch_id", "class_name", "distance_m", "N",
        "y_mean", "pred_difficulty_score", "pred_difficulty_1_5",
        "is_hard_actual", "is_hard_predicted"
    ]
    cols_to_save = [c for c in export_cols if c in df_merged.columns]
    df_merged[cols_to_save].to_csv(out_csv, index=False, encoding="utf-8")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return df_merged, results
