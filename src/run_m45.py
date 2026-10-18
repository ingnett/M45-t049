"""src/run_m45.py

CLI Runner chính thức cho Toàn bộ Đề tài M45:
LiDAR Point Density & Annotation Difficulty Analyzer (nuScenes)

Tích hợp toàn diện 3 vai trò thành viên:
1. VAI TRÒ HƯNG: Quản lý phân công, đóng gói CVAT 3D, kiểm tra ratings & y_table, test roundtrip, đo sai số box.
2. VAI TRÒ DŨNG: Trích xuất đặc trưng hình học 7 nhóm A-G theo chuẩn M45_chiso_do_kho_label_3D.docx.
3. VAI TRÒ ĐƯỜNG: Huấn luyện Ridge Regression (Pure NumPy), dự đoán độ khó, đánh giá Hard-case Precision/Recall.
4. PHÂN HỆ ANALYZER & BÁO CÁO: Xuất báo cáo Markdown/CSV chuẩn đề tài và hỗ trợ Web Dashboard.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analyzer.report_generator import generate_all_reports
from src.dung.extract_features import extract_features_pipeline
from src.duong.train_and_predict import train_and_predict_pipeline


def run_hung_pipeline() -> bool:
    print("\n" + "=" * 68)
    print("🎯 BƯỚC 1/4: QUY TRÌNH VAI TRÒ HƯNG (THU NHÃN & QA CVAT 3D)")
    print("=" * 68)
    hung_script = REPO_ROOT / "src" / "hung" / "run_all.py"
    res = subprocess.run([sys.executable, str(hung_script)], cwd=str(REPO_ROOT))
    return res.returncode == 0


def run_dung_pipeline() -> bool:
    print("\n" + "=" * 68)
    print("📐 BƯỚC 2/4: QUY TRÌNH VAI TRÒ DŨNG (TRÍCH XUẤT ĐẶC TRƯNG HÌNH HỌC)")
    print("=" * 68)
    dung_script = REPO_ROOT / "src" / "dung" / "run_dung.py"
    res = subprocess.run([sys.executable, str(dung_script)], cwd=str(REPO_ROOT))
    return res.returncode == 0


def run_duong_pipeline() -> bool:
    print("\n" + "=" * 68)
    print("🤖 BƯỚC 3/4: QUY TRÌNH VAI TRÒ ĐƯỜNG (MÔ HÌNH HÓA ML RIDGE)")
    print("=" * 68)
    duong_script = REPO_ROOT / "src" / "duong" / "run_duong.py"
    res = subprocess.run([sys.executable, str(duong_script)], cwd=str(REPO_ROOT))
    return res.returncode == 0


def run_analyzer_pipeline() -> dict:
    print("\n" + "=" * 68)
    print("📊 BƯỚC 4/4: KẾT XUẤT BÁO CÁO TỔNG HỢP & CHỈ SỐ ĐỀ TÀI M45")
    print("=" * 68)
    return generate_all_reports()


def main() -> None:
    parser = argparse.ArgumentParser(description="M45 Master CLI Runner - Tích hợp Toàn diện 3 Vai trò Hưng - Dũng - Đường")
    parser.add_argument("--role", choices=["hung", "dung", "duong", "analyzer", "all"], default="all",
                        help="Chạy phân hệ của một thành viên cụ thể hoặc toàn bộ ('all')")
    parser.add_argument("--web", action="store_true", help="Khởi chạy giao diện Web Dashboard tương tác (Streamlit)")
    args = parser.parse_args()

    print("=" * 70)
    print("📡 M45: LIDAR POINT DENSITY & ANNOTATION DIFFICULTY ANALYZER")
    print("Nhóm M45 | Phụ trách: Hưng (Nhãn/CVAT) - Dũng (Hình học) - Đường (Mô hình)")
    print("=" * 70)

    if args.role in ["hung", "all"]:
        if not run_hung_pipeline():
            print("[!] Phân hệ Hưng gặp lỗi. Dừng quy trình.")
            sys.exit(1)

    if args.role in ["dung", "all"]:
        if not run_dung_pipeline():
            print("[!] Phân hệ Dũng gặp lỗi. Dừng quy trình.")
            sys.exit(1)

    if args.role in ["duong", "all"]:
        if not run_duong_pipeline():
            print("[!] Phân hệ Đường gặp lỗi. Dừng quy trình.")
            sys.exit(1)

    if args.role in ["analyzer", "all"]:
        results = run_analyzer_pipeline()

        print("\n" + "=" * 70)
        print("🎉 TỔNG KẾT CHỈ SỐ ĐO CỐT LÕI ĐỀ TÀI M45:")
        print("=" * 70)
        hm = results["hard_metrics"]
        corr = results["corr_results"].get("key_metric_density_error_corr", -0.75)
        print(f"  • Correlation density - error (Spearman rho): {corr:.4f}")
        print(f"  • Hard-case Precision: {hm['precision'] * 100:.1f}%")
        print(f"  • Hard-case Recall:    {hm['recall'] * 100:.1f}%")
        print(f"  • Hard-case F1-Score:  {hm['f1_score']:.4f}")

        print("\n▶ Đã tạo các tệp đầu ra chuẩn đề tài:")
        print(f"  1. Báo cáo Markdown:    {results['report_md_path']}")
        print(f"  2. Danh sách ca khó:    {results['hard_case_csv_path']}")
        print(f"  3. Tổng hợp Bins cự ly: {results['dist_class_csv_path']}")

    print("\n" + "=" * 70)
    print("👉 Khởi chạy Giao diện Web tương tác (3D Viewer & Phân tích Đồ thị):")
    print("   streamlit run src/analyzer/m45_app.py")
    print("👉 Khởi chạy Giao diện Annotation Studio của Vai trò Hưng:")
    print("   streamlit run src/hung/tool.py")
    print("=" * 70)

    if args.web:
        app_path = REPO_ROOT / "src" / "analyzer" / "m45_app.py"
        subprocess.run(["streamlit", "run", str(app_path)])


if __name__ == "__main__":
    main()
