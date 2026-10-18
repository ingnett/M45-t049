"""src/duong/model.py

Mô hình Hồi quy Ridge (Ridge Regression) Dự đoán Độ khó Dán nhãn 3D Cuboid.
Phụ trách: ĐƯỜNG (Mô hình hóa & Machine Learning).
Tuân thủ tài liệu: M45_chiso_do_kho_label_3D.docx

Quy trình 3 bước chuẩn:
1. Chuẩn hóa z-score (đảo dấu các chỉ số 'càng lớn càng dễ': N, density, coverage, entropy, ring_count).
2. Ridge Regression học trọng số: y = sum(w_i * x_i_norm) + b (bằng numpy thuần).
3. Scale và clip về [1.0, 5.0] và làm tròn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Danh sách các đặc trưng tuyển chọn từ 7 nhóm A-G của Dũng
FEATURE_SPECS = [
    # (Tên cột, Có đảo dấu không: True nếu 'càng lớn càng dễ')
    ("N", True),                    # Nhóm A: Số điểm
    ("density", True),              # Nhóm A: Mật độ thể tích
    ("ring_count", True),           # Nhóm A: Số vòng quét laser
    ("coverage_avg", True),         # Nhóm B: Độ bao phủ 3 trục
    ("entropy_8_octant", True),     # Nhóm B: Entropy phân bố octant
    ("patchiness", False),          # Nhóm B: Tính vá víu / tỷ lệ voxel rỗng (càng lớn càng khó)
    ("distance_m", False),          # Nhóm C: Cự ly (càng xa càng khó)
    ("size_anomaly", False),        # Nhóm C: Lệch kích thước chuẩn class (càng lớn càng khó)
    ("boundary_clutter", False),    # Nhóm D: Nhiễu biên
    ("l_shape_fit_residual", False),# Nhóm E: Sai số khớp L-shape
]


class M45RidgeDifficultyModel:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.feature_names = [f[0] for f in FEATURE_SPECS]
        self.sign_flips = [f[1] for f in FEATURE_SPECS]
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 3.0

    def fit(self, X_df: pd.DataFrame, y: np.ndarray) -> M45RidgeDifficultyModel:
        """
        Huấn luyện mô hình Ridge Regression bằng NumPy thuần.
        y là mảng nhãn độ khó thực tế từ rater (thang 1 - 5).
        """
        # Trích xuất ma trận X
        X_raw = X_df[self.feature_names].to_numpy(dtype=np.float64)

        # Tính mean và std trên tập huấn luyện
        self.means = np.mean(X_raw, axis=0)
        self.stds = np.std(X_raw, axis=0)
        # Tránh chia cho 0
        self.stds[self.stds < 1e-6] = 1.0

        # Chuẩn hóa Z-score và đảo dấu
        X_norm = (X_raw - self.means) / self.stds
        for i, flip in enumerate(self.sign_flips):
            if flip:
                X_norm[:, i] = -X_norm[:, i]

        # Bổ sung cột bias (cột 1)
        n_samples = len(X_norm)
        X_design = np.hstack([X_norm, np.ones((n_samples, 1), dtype=np.float64)])

        # Ma trận điều chuẩn L2: alpha * I (không phạt bias)
        n_feats = len(self.feature_names)
        reg_matrix = self.alpha * np.eye(n_feats + 1, dtype=np.float64)
        reg_matrix[-1, -1] = 0.0  # Không điều chuẩn bias

        # Nghiệm đóng Ridge Regression: w = (X^T * X + reg)^(-1) * X^T * y
        lhs = np.dot(X_design.T, X_design) + reg_matrix
        rhs = np.dot(X_design.T, y)

        solution = np.linalg.solve(lhs, rhs)
        self.weights = solution[:n_feats]
        self.bias = float(solution[-1])
        return self

    def predict_continuous(self, X_df: pd.DataFrame) -> np.ndarray:
        """Dự đoán giá trị độ khó liên tục trong khoảng [1.0, 5.0]."""
        if self.weights is None or self.means is None or self.stds is None:
            raise RuntimeError("Mô hình chưa được huấn luyện! Hãy gọi fit() trước.")

        X_raw = X_df[self.feature_names].to_numpy(dtype=np.float64)
        X_norm = (X_raw - self.means) / self.stds
        for i, flip in enumerate(self.sign_flips):
            if flip:
                X_norm[:, i] = -X_norm[:, i]

        y_pred = np.dot(X_norm, self.weights) + self.bias
        # Clip về thang [1.0, 5.0]
        return np.clip(y_pred, 1.0, 5.0)

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        """Dự đoán độ khó rời rạc thang 1 - 5 (làm tròn nguyên)."""
        cont = self.predict_continuous(X_df)
        return np.clip(np.round(cont), 1.0, 5.0).astype(int)

    def get_feature_importances(self) -> Dict[str, float]:
        """Lấy trọng số học được của từng đặc trưng."""
        if self.weights is None:
            return {}
        return {
            name: float(w) for name, w in zip(self.feature_names, self.weights)
        }
