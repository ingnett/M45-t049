"""tests/test_box_difficulty_rules.py

Kiểm thử tự động cho Bộ quy tắc phân loại Box Khó vs Box Dễ.
"""

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analyzer.box_difficulty_rules import evaluate_box_difficulty


class TestBoxDifficultyRules(unittest.TestCase):

    def test_01_easy_car(self):
        """Xe gần, nhiều điểm -> Phải là EASY."""
        res = evaluate_box_difficulty(point_count=60, distance_m=11.0, class_name="vehicle.car")
        self.assertEqual(res["tier"], "EASY")
        self.assertLessEqual(res["difficulty_score"], 2.2)
        self.assertIn("DỄ", res["risk_badge"])

    def test_02_medium_car(self):
        """Xe cự ly trung bình 25m, 15 điểm -> Phải là MEDIUM."""
        res = evaluate_box_difficulty(point_count=15, distance_m=25.0, class_name="vehicle.car")
        self.assertEqual(res["tier"], "MEDIUM")
        self.assertGreater(res["difficulty_score"], 2.0)
        self.assertLess(res["difficulty_score"], 3.6)

    def test_03_hard_car_distance(self):
        """Xe ở xa 55m, ít điểm (3 pts) -> Phải là HARD."""
        res = evaluate_box_difficulty(point_count=3, distance_m=55.0, class_name="vehicle.car")
        self.assertEqual(res["tier"], "HARD")
        self.assertGreaterEqual(res["difficulty_score"], 3.6)
        self.assertIn("KHÓ", res["risk_badge"])
        self.assertTrue(any("180" in err for err in res["errors_predicted"]))

    def test_04_hard_pedestrian_sparse(self):
        """Người đi bộ chỉ có 2 điểm -> Phải là HARD."""
        res = evaluate_box_difficulty(point_count=2, distance_m=28.0, class_name="human.pedestrian.adult")
        self.assertEqual(res["tier"], "HARD")

    def test_05_occlusion_hard_trigger(self):
        """Đối tượng bị che khuất >= 40% -> Phải là HARD."""
        res = evaluate_box_difficulty(point_count=12, distance_m=18.0, class_name="vehicle.car", occlusion_ratio=0.5)
        self.assertEqual(res["tier"], "HARD")
        self.assertTrue(any("che khuất" in r.lower() for r in res["reasons"]))


if __name__ == "__main__":
    unittest.main()
