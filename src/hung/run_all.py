"""src/hung/run_all.py

Chạy toàn bộ quy trình thu nhãn, kiểm tra và đo sai số của Vai HƯNG.
Tự động gọi lần lượt:
1. make_assignments.py
2. make_cvat_pack.py
3. validate_ratings.py
4. test_box_roundtrip.py
5. box_errors.py
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

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_step(step_idx: int, step_name: str, script_name: str, extra_args: list[str] | None = None) -> bool:
    print(f"\n==================================================================")
    print(f"▶ [{step_idx}/5] {step_name} (src/hung/{script_name})")
    print(f"==================================================================")
    sys.stdout.flush()
    script_path = REPO_ROOT / "src" / "hung" / script_name
    cmd = [sys.executable, str(script_path)] + (extra_args or [])
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    sys.stdout.flush()
    if res.returncode != 0:
        print(f"\n[!] BƯỚC {step_idx} ({step_name}) GẶP LỖI (Exit code {res.returncode})")
        return False
    print(f"[✓] Bước {step_idx} hoàn thành xuất sắc!")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="M45 - All-in-one CLI Runner cho Vai trò Hưng")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5], help="Chỉ chạy riêng 1 bước cụ thể (1 đến 5)")
    args = parser.parse_args()

    print("🎯 BẮT ĐẦU QUY TRÌNH TOÀN DIỆN CHO VAI TRÒ HƯNG (NHÓM M45)")
    steps = [
        (1, "Phân chia nhiệm vụ dán nhãn (assignments.csv)", "make_assignments.py", []),
        (2, "Đóng gói dữ liệu CVAT 3D (cvat_packs/*.zip)", "make_cvat_pack.py", []),
        (3, "Kiểm tra ratings & xuất y_table (y_table.csv)", "validate_ratings.py", []),
        (4, "Kiểm thử Round-trip Bounding Box (G1 test bench)", "test_box_roundtrip.py", []),
        (5, "Đo sai số Bounding Box (box_errors.csv)", "box_errors.py", []),
    ]

    if args.step:
        steps = [s for s in steps if s[0] == args.step]

    all_success = True
    for idx, name, script, extra_args in steps:
        ok = run_step(idx, name, script, extra_args)
        if not ok:
            all_success = False
            break

    print("\n" + "=" * 66)
    if all_success:
        print("🎉 TẤT CẢ CÁC MODULE ĐÃ HOÀN TẤT THÀNH CÔNG VÀ ĐẠT TOÀN BỘ CHỈ TIÊU!")
    else:
        print("❌ CÓ BƯỚC CHƯA ĐẠT. VUI LÒNG KIỂM TRA LỖI Ở TRÊN.")
    print("=" * 66)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
