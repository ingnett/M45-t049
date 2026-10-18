"""src/dung/run_dung.py

CLI Runner & Verification Suite cho Vai trò DŨNG (Dữ liệu & Hình học).
Dự án: M45 - LiDAR Point Density & Annotation Difficulty Analyzer
Thực hiện:
1. Trích xuất đặc trưng 7 nhóm A-G cho toàn bộ ca trong data/work/cases.csv.
2. Kiểm tra tính hợp lệ hình học (Volume > 0, N >= 0, Entropy >= 0, Coverage trong [0, 1]).
3. Xuất kết quả ra data/work/dung_features.csv.
4. Báo cáo thống kê tổng kết phân phối đặc trưng.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dung.extract_features import extract_features_pipeline

WORK_DIR = REPO_ROOT / "data" / "work"


def validate_and_report_features(df: pd.DataFrame) -> bool:
    print("\n" + "=" * 68)
    print("      BÁO CÁO THỐNG KÊ ĐẶC TRƯNG HÌNH HỌC (VAI TRÒ DŨNG)")
    print("=" * 68)

    total_cases = len(df)
    print(f"[*] Tổng số ca đã trích xuất: {total_cases}")

    # Kiểm tra tính toàn vẹn
    invalid_vol = (df["volume_m3"] <= 0).sum()
    invalid_n = (df["N"] < 0).sum()
    invalid_cov = ((df["coverage_avg"] < 0) | (df["coverage_avg"] > 1.0)).sum()
    invalid_ent = (df["entropy_8_octant"] < 0).sum()

    print(f"[*] Kiểm tra tính hợp lệ hình học:")
    print(f"    - Ca có thể tích không hợp lệ (<= 0):     {invalid_vol}")
    print(f"    - Ca có số điểm âm (< 0):                 {invalid_n}")
    print(f"    - Ca có coverage ngoài [0, 1]:            {invalid_cov}")
    print(f"    - Ca có entropy âm (< 0):                 {invalid_ent}")

    if invalid_vol + invalid_n + invalid_cov + invalid_ent > 0:
        print("[!] Phát hiện vi phạm tính hợp lệ đặc trưng!")
        return False

    print("\n--- BẢNG THỐNG KÊ CÁC ĐẶC TRƯNG CỐT LÕI (MEAN ± STD) ---")
    metrics = [
        ("N (số điểm)", "N"),
        ("Mật độ thể tích (pts/m³)", "density"),
        ("Mật độ chuẩn hóa (N·d²)", "N_d2"),
        ("Voxel occupancy (0.2m)", "voxel_occupancy"),
        ("Số vòng laser (ring_count)", "ring_count"),
        ("Coverage trung bình", "coverage_avg"),
        ("Entropy 8 octant", "entropy_8_octant"),
        ("Khoảng cách d (m)", "distance_m"),
        ("Thể tích V (m³)", "volume_m3"),
        ("L-shape residual", "l_shape_fit_residual"),
        ("Boundary clutter", "boundary_clutter"),
    ]

    for label, col in metrics:
        if col in df.columns:
            m = df[col].mean()
            s = df[col].std()
            min_v = df[col].min()
            max_v = df[col].max()
            print(f"  • {label:<28}: {m:>8.2f} ± {s:>6.2f} (Min: {min_v:>6.2f}, Max: {max_v:>6.2f})")

    # Phân nhóm theo Class
    print("\n--- PHÂN BỔ TRUNG BÌNH THEO PHÂN LỚP ĐỐI TƯỢNG (CLASS) ---")
    grp = df.groupby("class_name")[["N", "distance_m", "density", "coverage_avg"]].mean()
    print(grp.to_string())

    print("\n" + "=" * 68)
    print(" [✓] KIỂM TRA ĐẶC TRƯNG HÌNH HỌC VAI TRÒ DŨNG ĐẠT CHUẨN!")
    print("=" * 68)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="M45 Runner cho Vai trò DŨNG - Trích xuất đặc trưng hình học & LiDAR")
    parser.add_argument("--cases", type=str, default=str(WORK_DIR / "cases.csv"), help="Đường dẫn file cases.csv")
    parser.add_argument("--output", type=str, default=str(WORK_DIR / "dung_features.csv"), help="Đường dẫn file dung_features.csv")
    args = parser.parse_args()

    print("🚀 BẮT ĐẦU QUY TRÌNH TRÍCH XUẤT ĐẶC TRƯNG CHO VAI TRÒ DŨNG...")
    df_feats = extract_features_pipeline(Path(args.cases), Path(args.output))
    ok = validate_and_report_features(df_feats)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
