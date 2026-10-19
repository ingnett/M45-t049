"""tests/test_full_app_integration.py

Kiểm thử chức năng chuyên sâu (End-to-End Functional Test) cho toàn bộ Tool M45.
Xác thực mọi tính toán, luồng dữ liệu và xuất file.
"""

from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import scipy.stats as stats

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
)


class TestFullAppIntegration(unittest.TestCase):

    def test_01_real_dataset_pipeline_integration(self):
        """Kiểm tra toàn bộ pipeline dữ liệu thực tế nuScenes (Vai trò Hưng -> Dũng -> Đường)."""
        df_real = load_real_nuscenes_dataset()
        self.assertGreaterEqual(len(df_real), 30, "Tập dữ liệu thực tế phải có ít nhất 30 ca")

        # Kiểm tra các cột cốt lõi
        expected_cols = [
            "case_id", "batch", "class_name", "distance_m", "distance_bin",
            "N", "density", "center_err_m", "size_err_pct", "yaw_err_deg",
            "difficulty_gt", "is_hard_gt", "_pts", "_wlh", "_center", "_yaw"
        ]
        for col in expected_cols:
            self.assertIn(col, df_real.columns, f"Thiếu cột {col} trong tập dữ liệu thực tế")

        # Kiểm tra tính toán tương quan (Correlation density-error)
        rho_center, p_val = stats.spearmanr(df_real["N"], df_real["center_err_m"])
        print(f"\n[Test 1] Real Dataset Spearman rho (N vs center_err): {rho_center:.4f} (p={p_val:.4e})")
        self.assertFalse(np.isnan(rho_center))

    def test_02_simulation_generator_scaling(self):
        """Kiểm tra bộ sinh mô phỏng vật lý có mở rộng linh hoạt theo số ca hay không."""
        for n_cases in [40, 80, 120]:
            df_sim = generate_simulation_dataset(n_cases=n_cases, seed=45)
            self.assertEqual(len(df_sim), n_cases, f"Số ca sinh ra ({len(df_sim)}) phải khớp ({n_cases})")
            self.assertTrue((df_sim["N"] >= 1).all())
            self.assertTrue((df_sim["distance_m"] > 0).all())

    def test_03_ridge_regression_learning_and_inference(self):
        """Kiểm tra mô hình hồi quy Ridge học được trọng số và dự đoán chính xác trong miền [1, 5]."""
        df_sim = generate_simulation_dataset(n_cases=80, seed=45)
        pred_fn, weights, bias = train_ridge_pure_numpy(df_sim, alpha=1.0)

        # Kiểm tra trọng số đặc trưng
        self.assertEqual(len(weights), 10)
        self.assertIsInstance(bias, float)

        # Dự đoán
        preds = pred_fn(df_sim)
        self.assertEqual(len(preds), len(df_sim))
        self.assertTrue((preds >= 1.0).all())
        self.assertTrue((preds <= 5.0).all())

        # MAE và RMSE phải nằm trong ngưỡng chấp nhận được (< 0.7 trên thang 1-5)
        mae = np.mean(np.abs(preds - df_sim["difficulty_gt"].to_numpy()))
        rmse = np.sqrt(np.mean((preds - df_sim["difficulty_gt"].to_numpy()) ** 2))
        print(f"[Test 3] Ridge Regression MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        self.assertLess(mae, 0.70)
        self.assertLess(rmse, 0.85)

    def test_04_hard_case_detection_metrics(self):
        """Kiểm tra bộ lọc ca khó và các chỉ số Precision, Recall, F1."""
        df_sim = generate_simulation_dataset(n_cases=80, seed=45)
        pred_fn, _, _ = train_ridge_pure_numpy(df_sim, alpha=1.0)
        df_sim["difficulty_pred"] = pred_fn(df_sim)

        thresh = 4.0
        act_hard = (df_sim["difficulty_gt"] >= thresh).to_numpy()
        pred_hard = (df_sim["difficulty_pred"] >= thresh).to_numpy()

        tp = int(np.sum(act_hard & pred_hard))
        fp = int(np.sum((~act_hard) & pred_hard))
        fn = int(np.sum(act_hard & (~pred_hard)))
        tn = int(np.sum((~act_hard) & (~pred_hard)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        print(f"[Test 4] Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"[Test 4] Hard-Case Precision: {prec * 100:.1f}%, Recall: {rec * 100:.1f}%, F1: {f1:.4f}")

        self.assertGreaterEqual(prec, 0.80, "Hard-case Precision phải >= 80%")
        self.assertGreaterEqual(rec, 0.70, "Hard-case Recall phải >= 70%")

    def test_05_3d_pointcloud_and_wireframe(self):
        """Kiểm tra hình học đám mây điểm 3D và wireframe cuboid."""
        rng = np.random.RandomState(45)
        center = np.array([25.0, -3.0, -0.6])
        wlh = np.array([1.93, 4.62, 1.69])
        yaw = 0.35

        cloud = generate_synthetic_cloud(center, wlh, yaw, point_count=25, rng=rng)
        self.assertGreaterEqual(len(cloud), 25)
        self.assertEqual(cloud.shape[1], 3)

        # Wireframe 12 cạnh
        xs, ys, zs = compute_cuboid_wireframe(center, wlh, yaw)
        self.assertEqual(len(xs), 36)
        # Kiểm tra không có giá trị NaN trong tọa độ đỉnh (chỉ None ngăn cách các đoạn thẳng)
        coords_x = [x for x in xs if x is not None]
        self.assertFalse(np.isnan(coords_x).any())

    def test_06_reports_and_export_integrity(self):
        """Kiểm tra tệp báo cáo markdown và CSV đã tạo ra trong thư mục outputs/reports."""
        report_md = REPO_ROOT / "outputs" / "reports" / "report_density_vs_distance_class.md"
        hard_csv = REPO_ROOT / "outputs" / "reports" / "hard_case_list.csv"
        dist_csv = REPO_ROOT / "outputs" / "reports" / "density_vs_distance_summary.csv"

        self.assertTrue(report_md.exists(), "Tệp report_density_vs_distance_class.md phải tồn tại")
        self.assertTrue(hard_csv.exists(), "Tệp hard_case_list.csv phải tồn tại")
        self.assertTrue(dist_csv.exists(), "Tệp density_vs_distance_summary.csv phải tồn tại")

        # Đọc và kiểm tra nội dung
        with open(report_md, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Correlation density - error", content)
            self.assertIn("Hard-case Precision", content)

        df_hard = pd.read_csv(hard_csv)
        self.assertGreater(len(df_hard), 0, "Danh sách ca khó không được rỗng")

        df_dist = pd.read_csv(dist_csv)
        self.assertGreater(len(df_dist), 0, "Bảng tổng hợp cự ly không được rỗng")


if __name__ == "__main__":
    unittest.main()
