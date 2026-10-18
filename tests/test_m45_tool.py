"""tests/test_m45_tool.py

Bộ kiểm thử tự động toàn diện cho Tool Đề tài M45:
LiDAR Point Density & Annotation Difficulty Analyzer
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
    DISTANCE_BINS,
    RIDGE_FEATS,
    compute_cuboid_wireframe,
    compute_features_from_cloud,
    generate_simulation_dataset,
    generate_synthetic_cloud,
    load_real_nuscenes_dataset,
    points_in_box_mask,
    theoretical_points,
    train_ridge_pure_numpy,
    yaw_rotation_matrix,
)


class TestM45Tool(unittest.TestCase):

    def test_01_theoretical_points_physics_law(self):
        """Kiểm tra định luật nghịch đảo bình phương: d tăng thì N giảm mạnh."""
        d_close = 10.0
        d_mid = 25.0
        d_far = 50.0

        n_close = theoretical_points(d_close, "vehicle.car")
        n_mid = theoretical_points(d_mid, "vehicle.car")
        n_far = theoretical_points(d_far, "vehicle.car")

        self.assertGreater(n_close, n_mid, "N ở 10m phải lớn hơn N ở 25m")
        self.assertGreater(n_mid, n_far, "N ở 25m phải lớn hơn N ở 50m")
        self.assertGreaterEqual(n_far, 1, "N ở cự ly xa tối thiểu phải là 1 điểm")

    def test_02_synthetic_cloud_generation(self):
        """Kiểm tra sinh đám mây điểm 3D bám trên hình hộp."""
        rng = np.random.RandomState(42)
        center = np.array([20.0, 0.0, 0.0])
        wlh = np.array([1.93, 4.62, 1.69])
        yaw = 0.0
        n_pts = 30

        cloud = generate_synthetic_cloud(center, wlh, yaw, n_pts, rng)
        self.assertIsInstance(cloud, np.ndarray)
        self.assertGreaterEqual(len(cloud), n_pts)
        self.assertEqual(cloud.shape[1], 3)

    def test_03_points_in_box_mask(self):
        """Kiểm tra thuật toán lọc điểm nằm trong 3D bounding box."""
        center = np.array([10.0, 5.0, 0.0])
        wlh = np.array([2.0, 4.0, 2.0])
        yaw = 0.0

        # Điểm tại tâm -> phải nằm trong
        pt_inside = center.reshape(1, 3)
        self.assertTrue(points_in_box_mask(pt_inside, center, wlh, yaw)[0])

        # Điểm cách xa 100m -> phải nằm ngoài
        pt_outside = np.array([[100.0, 100.0, 100.0]])
        self.assertFalse(points_in_box_mask(pt_outside, center, wlh, yaw)[0])

    def test_04_cuboid_wireframe_structure(self):
        """Kiểm tra 12 cạnh wireframe của hình hộp 3D."""
        center = np.array([15.0, 2.0, -0.5])
        wlh = np.array([1.9, 4.5, 1.6])
        yaw = 0.5

        xs, ys, zs = compute_cuboid_wireframe(center, wlh, yaw)
        self.assertEqual(len(xs), len(ys))
        self.assertEqual(len(ys), len(zs))
        # 12 cạnh, mỗi cạnh gồm 2 đỉnh và 1 điểm None phân tách = 36 phần tử
        self.assertEqual(len(xs), 36)

    def test_05_simulation_dataset_integrity(self):
        """Kiểm tra tập dữ liệu mô phỏng tương tác."""
        df_sim = generate_simulation_dataset(n_cases=40, seed=123)
        self.assertEqual(len(df_sim), 40)

        required_cols = [
            "case_id", "batch", "class_name", "distance_m", "distance_bin",
            "N", "density", "center_err_m", "yaw_err_deg", "size_err_pct",
            "difficulty_gt", "is_hard_gt", "_pts", "_wlh", "_center", "_yaw"
        ]
        for col in required_cols:
            self.assertIn(col, df_sim.columns)

        # Kiểm tra khoảng giá trị hợp lệ
        self.assertTrue((df_sim["N"] >= 1).all())
        self.assertTrue((df_sim["distance_m"] > 0).all())
        self.assertTrue((df_sim["difficulty_gt"] >= 1.0).all())
        self.assertTrue((df_sim["difficulty_gt"] <= 5.0).all())

    def test_06_ridge_regression_pure_numpy(self):
        """Kiểm tra mô hình huấn luyện Ridge Regression bằng Pure NumPy."""
        df_sim = generate_simulation_dataset(n_cases=50, seed=45)
        pred_fn, weights, bias = train_ridge_pure_numpy(df_sim, alpha=1.0)

        self.assertIsInstance(weights, dict)
        self.assertEqual(len(weights), len(RIDGE_FEATS))
        self.assertIsInstance(bias, float)

        preds = pred_fn(df_sim)
        self.assertEqual(len(preds), len(df_sim))
        self.assertTrue((preds >= 1.0).all())
        self.assertTrue((preds <= 5.0).all())

    def test_07_real_nuscenes_dataset_loader(self):
        """Kiểm tra tải tập dữ liệu thực nghiệm nuScenes."""
        df_real = load_real_nuscenes_dataset()
        self.assertGreaterEqual(len(df_real), 30)
        self.assertIn("N", df_real.columns)
        self.assertIn("distance_m", df_real.columns)
        self.assertIn("difficulty_gt", df_real.columns)

    def test_08_core_metrics_target_check(self):
        """Kiểm tra các chỉ số cốt lõi đề tài M45."""
        df_sim = generate_simulation_dataset(n_cases=60, seed=45)
        pred_fn, _, _ = train_ridge_pure_numpy(df_sim, alpha=1.0)
        df_sim["difficulty_pred"] = pred_fn(df_sim)

        thresh = 4.0
        act_hard = df_sim["difficulty_gt"] >= thresh
        pred_hard = df_sim["difficulty_pred"] >= thresh

        tp = int(np.sum(act_hard & pred_hard))
        fp = int(np.sum((~act_hard) & pred_hard))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0

        # Hard-case precision phải đạt tối thiểu >= 80% (đề tài đạt 100%)
        self.assertGreaterEqual(precision, 0.80)


if __name__ == "__main__":
    unittest.main()
