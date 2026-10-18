"""src/hung/validate_ratings.py

Kiểm tra tính hợp lệ của file ratings.csv, xuất y_table.csv và in báo cáo.
Tuân thủ mục 2, mục 3, mục 5.5, mục 5.6, mục 6, mục 8.4 của tài liệu M45.

Quy tắc kiểm tra:
- Kiểm trường bắt buộc: case_id, rater_id, status, rubric_version, is_demo.
- Khóa duy nhất: case_id + rater_id.
- rubric_version phải đúng bản đã khóa (mặc định 'r1').
- status ∈ {'completed', 'timeout', 'insufficient_context'}.
- duration_s <= 180 với ca completed (chỉ dùng ca completed khi phân tích tốc độ).
- difficulty_1_5, confidence_1_5 là số nguyên 1-5 hoặc để trống.
- reason: các thẻ cách nhau dấu ';' từ {center, size, orientation, context, other}.
- Ghi y_table.csv theo mục 5.6:
  + Một lượt hợp lệ nếu difficulty_1_5 không trống (bất kể status).
  + y_status=ok khi có đúng hai lượt hợp lệ; difficulty_mean là trung bình hai điểm.
  + Còn lại excluded kèm lý do (ví dụ lt2_valid). Không tự bổ sung điểm.
- In báo cáo: số ca thiếu, số timeout, số insufficient_context,
  tỷ lệ hai điểm khó lệch <= 1 bậc, danh sách ca lệch >= 2 bậc. Ca thiếu được báo, không xóa âm thầm.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Set, Tuple

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = REPO_ROOT / "data" / "work"
LABELS_DIR = REPO_ROOT / "data" / "labels"

VALID_STATUSES = {"completed", "timeout", "insufficient_context"}
VALID_REASONS = {"center", "size", "orientation", "context", "other"}
VALID_RATERS = {"R1", "R2", "R3"}
REQUIRED_FIELDS = ["case_id", "rater_id", "status", "rubric_version", "is_demo"]


def parse_int_or_none(val: str, min_v: int = 1, max_v: int = 5) -> Tuple[Optional[int], Optional[str]]:
    val = val.strip()
    if not val:
        return None, None
    try:
        iv = int(val)
        if iv < min_v or iv > max_v:
            return None, f"Giá trị {iv} nằm ngoài khoảng [{min_v}, {max_v}]"
        return iv, None
    except ValueError:
        return None, f"Không phải số nguyên hợp lệ: '{val}'"


def parse_float_or_none(val: str) -> Tuple[Optional[float], Optional[str]]:
    val = val.strip()
    if not val:
        return None, None
    try:
        fv = float(val)
        return fv, None
    except ValueError:
        return None, f"Không phải số thực hợp lệ: '{val}'"


def validate_ratings_file(
    ratings_path: Path,
    expected_rubric_version: str = "r1",
) -> Tuple[List[Dict[str, str]], List[str]]:
    """Đọc và kiểm tra tính hợp lệ của từng dòng trong ratings.csv."""
    errors: List[str] = []
    rows: List[Dict[str, str]] = []

    if not ratings_path.exists():
        errors.append(f"File không tồn tại: {ratings_path}")
        return rows, errors

    seen_keys: Set[Tuple[str, str]] = set()

    with open(ratings_path, "r", encoding="utf-8") as f:
        # Bỏ qua các dòng comment bắt đầu bằng '#'
        filtered_lines = [line for line in f if not line.strip().startswith("#")]
        reader = csv.DictReader(filtered_lines)

        # Kiểm tra tiêu đề cột
        if not reader.fieldnames:
            errors.append("File rỗng hoặc không có tiêu đề cột.")
            return rows, errors

        missing_headers = [req for req in REQUIRED_FIELDS if req not in reader.fieldnames]
        if missing_headers:
            errors.append(f"Thiếu các cột bắt buộc trong header: {missing_headers}")
            return rows, errors

        for idx, row in enumerate(reader, start=2):
            case_id = row.get("case_id", "").strip()
            rater_id = row.get("rater_id", "").strip()
            status = row.get("status", "").strip()
            rubric_version = row.get("rubric_version", "").strip()
            duration_str = row.get("duration_s", "").strip()
            diff_str = row.get("difficulty_1_5", "").strip()
            conf_str = row.get("confidence_1_5", "").strip()
            reason_str = row.get("reason", "").strip()

            line_prefix = f"Dòng {idx} [case={case_id}, rater={rater_id}]:"

            if not case_id:
                errors.append(f"{line_prefix} case_id không được để trống.")
            if not rater_id or rater_id not in VALID_RATERS:
                errors.append(f"{line_prefix} rater_id không hợp lệ: '{rater_id}' (chỉ nhận R1, R2, R3).")

            # Khóa duy nhất (case_id + rater_id)
            key = (case_id, rater_id)
            if key in seen_keys:
                errors.append(f"{line_prefix} Trùng lặp khóa duy nhất (case_id + rater_id): {key}")
            else:
                seen_keys.add(key)

            # rubric_version
            if rubric_version != expected_rubric_version:
                errors.append(
                    f"{line_prefix} rubric_version sai bản: '{rubric_version}' (bản đã khóa là '{expected_rubric_version}')."
                )

            # status
            if status not in VALID_STATUSES:
                errors.append(
                    f"{line_prefix} status không hợp lệ: '{status}' (chỉ nhận completed, timeout, insufficient_context)."
                )

            # duration_s
            dur_val, dur_err = parse_float_or_none(duration_str)
            if dur_err:
                errors.append(f"{line_prefix} duration_s: {dur_err}")
            else:
                if status == "completed":
                    if dur_val is None:
                        errors.append(f"{line_prefix} Ca completed bắt buộc phải có duration_s.")
                    elif dur_val > 180.0:
                        errors.append(f"{line_prefix} Ca completed có duration_s = {dur_val}s > 180s (vi phạm luật cứng).")
                    elif dur_val < 0.0:
                        errors.append(f"{line_prefix} duration_s không được âm: {dur_val}s.")

            # difficulty_1_5
            _, diff_err = parse_int_or_none(diff_str, 1, 5)
            if diff_err:
                errors.append(f"{line_prefix} difficulty_1_5: {diff_err}")

            # confidence_1_5
            _, conf_err = parse_int_or_none(conf_str, 1, 5)
            if conf_err:
                errors.append(f"{line_prefix} confidence_1_5: {conf_err}")

            # reason
            if reason_str:
                reasons = [r.strip() for r in reason_str.split(";") if r.strip()]
                for r in reasons:
                    if r not in VALID_REASONS:
                        errors.append(
                            f"{line_prefix} reason có thẻ không hợp lệ: '{r}' (chỉ nhận: center, size, orientation, context, other)."
                        )

            rows.append(row)

    return rows, errors


def build_y_table(
    rows: List[Dict[str, str]],
    expected_case_ids: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, str]], Dict[str, any]]:
    """Xây dựng y_table.csv theo mục 5.6 và thống kê các chỉ số chất lượng nhãn."""
    ratings_by_case: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        cid = r["case_id"].strip()
        ratings_by_case.setdefault(cid, []).append(r)

    # Tập hợp tất cả các case_id cần xem xét
    all_case_ids = set(ratings_by_case.keys())
    if expected_case_ids:
        all_case_ids.update(expected_case_ids)

    y_table: List[Dict[str, str]] = []
    stats = {
        "total_cases": len(all_case_ids),
        "missing_cases": [],
        "n_timeout": 0,
        "n_insufficient_context": 0,
        "n_ok": 0,
        "n_excluded": 0,
        "diff_le_1": 0,
        "diff_ge_2": [],
        "completed_durations": [],
    }

    for cid in sorted(all_case_ids):
        case_ratings = ratings_by_case.get(cid, [])
        if not case_ratings:
            # Ca hoàn toàn thiếu
            stats["missing_cases"].append(cid)
            y_table.append({
                "case_id": cid,
                "difficulty_mean": "",
                "n_valid_ratings": "0",
                "y_status": "excluded",
                "y_exclusion_reason": "missing_ratings",
            })
            continue

        valid_difficulties: List[int] = []
        for r in case_ratings:
            st = r.get("status", "").strip()
            if st == "timeout":
                stats["n_timeout"] += 1
            elif st == "insufficient_context":
                stats["n_insufficient_context"] += 1
            elif st == "completed":
                dur, _ = parse_float_or_none(r.get("duration_s", ""))
                if dur is not None:
                    stats["completed_durations"].append(dur)

            d_val, _ = parse_int_or_none(r.get("difficulty_1_5", ""), 1, 5)
            if d_val is not None:
                valid_difficulties.append(d_val)

        n_valid = len(valid_difficulties)

        if n_valid == 2:
            mean_diff = sum(valid_difficulties) / 2.0
            y_status = "ok"
            exclusion_reason = ""
            stats["n_ok"] += 1

            gap = abs(valid_difficulties[0] - valid_difficulties[1])
            if gap <= 1:
                stats["diff_le_1"] += 1
            else:
                stats["diff_ge_2"].append((cid, valid_difficulties[0], valid_difficulties[1], gap))

            y_table.append({
                "case_id": cid,
                "difficulty_mean": f"{mean_diff:.4f}",
                "n_valid_ratings": "2",
                "y_status": y_status,
                "y_exclusion_reason": exclusion_reason,
            })
        else:
            y_status = "excluded"
            stats["n_excluded"] += 1
            reason = "lt2_valid" if n_valid < 2 else "gt2_valid"
            y_table.append({
                "case_id": cid,
                "difficulty_mean": "",
                "n_valid_ratings": str(n_valid),
                "y_status": y_status,
                "y_exclusion_reason": reason,
            })

    return y_table, stats


def print_validation_report(stats: Dict[str, any], errors: List[str]) -> None:
    """In báo cáo chi tiết theo đúng yêu cầu mục 8.4."""
    print("=================================================================")
    print("                BÁO CÁO KIỂM TRA NHÃN & Y_TABLE                  ")
    print("=================================================================")

    if errors:
        print(f"[!] PHÁT HIỆN {len(errors)} LỖI CHẶN (VALIDATION ERRORS):")
        for err in errors[:20]:
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... và {len(errors) - 20} lỗi khác.")
        print("-" * 65)
    else:
        print("[OK] Toàn bộ các dòng ratings.csv đều hợp lệ theo rubric và luật cứng.")
        print("-" * 65)

    print(f"- Tổng số ca xem xét:            {stats['total_cases']}")
    print(f"- Số ca hoàn thành hợp lệ (ok):   {stats['n_ok']}")
    print(f"- Số ca bị loại trừ (excluded):  {stats['n_excluded']}")
    print(f"- Số ca thiếu dữ liệu nhãn:       {len(stats['missing_cases'])}")
    if stats["missing_cases"]:
        print(f"  -> Danh sách ca thiếu: {', '.join(stats['missing_cases'][:10])}")
    print(f"- Số lượt timeout:               {stats['n_timeout']}")
    print(f"- Số lượt insufficient_context:  {stats['n_insufficient_context']}")

    # Tỷ lệ lệch điểm khó
    total_compared = stats["n_ok"]
    if total_compared > 0:
        ratio_le_1 = (stats["diff_le_1"] / total_compared) * 100.0
        print(f"- Tỷ lệ hai điểm khó lệch ≤ 1 bậc: {ratio_le_1:.1f}% ({stats['diff_le_1']}/{total_compared})")
        print(f"- Số ca có điểm khó lệch ≥ 2 bậc:  {len(stats['diff_ge_2'])}")
        if stats["diff_ge_2"]:
            print("  Chi tiết các ca lệch ≥ 2 bậc:")
            for cid, d1, d2, gap in stats["diff_ge_2"]:
                print(f"    * {cid}: rater 1 = {d1}, rater 2 = {d2} (chênh lệch {gap} bậc)")
    else:
        print("- Chưa có ca nào đủ 2 lượt hợp lệ để so sánh độ lệch điểm khó.")

    # Tốc độ
    durs = stats["completed_durations"]
    if durs:
        mean_dur = sum(durs) / len(durs)
        print(f"- Tốc độ dán nhãn (ca completed): Trung bình {mean_dur:.1f}s | Min {min(durs):.1f}s | Max {max(durs):.1f}s")

    print("=================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra ratings.csv, xuất y_table.csv và in báo cáo.")
    parser.add_argument("--ratings", type=str, default="", help="Đường dẫn file ratings.csv")
    parser.add_argument("--assignments", type=str, default="", help="Đường dẫn file assignments.csv")
    parser.add_argument("--output_y", type=str, default="", help="Đường dẫn xuất file y_table.csv")
    parser.add_argument("--rubric", type=str, default="r1", help="Phiên bản rubric đã khóa (mặc định r1)")
    args = parser.parse_args()

    ratings_path = Path(args.ratings) if args.ratings else LABELS_DIR / "ratings.csv"
    assignments_path = Path(args.assignments) if args.assignments else LABELS_DIR / "assignments.csv"
    output_y_path = Path(args.output_y) if args.output_y else LABELS_DIR / "y_table.csv"

    # Lấy danh sách case_id mong đợi từ assignments.csv nếu có
    expected_case_ids: Set[str] = set()
    if assignments_path.exists():
        with open(assignments_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = row.get("case_id", "").strip()
                if cid:
                    expected_case_ids.add(cid)

    # Nếu ratings.csv chưa tồn tại (chưa dán nhãn thực tế), tạo file demo_ratings.csv theo mục 5.10
    if not ratings_path.exists():
        print(f"[*] {ratings_path} chưa tồn tại. Tạo dữ liệu ratings giả lập demo (is_demo=True) để kiểm thử...")
        ratings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ratings_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "case_id",
                "rater_id",
                "difficulty_1_5",
                "confidence_1_5",
                "duration_s",
                "status",
                "rubric_version",
                "reason",
                "is_demo",
            ])
            # Tạo demo ratings cho các ca từ assignments nếu có
            if expected_case_ids:
                # Đọc features từ dung_features.csv hoặc cases.csv nếu có
                feat_map: Dict[str, float] = {}
                feats_csv = WORK_DIR / "dung_features.csv"
                if feats_csv.exists():
                    try:
                        with open(feats_csv, "r", encoding="utf-8") as fcf:
                            for r in csv.DictReader(fcf):
                                cid_f = r["case_id"]
                                d_m = float(r.get("distance_m", 25.0))
                                n_pts = float(r.get("N", 10.0))
                                # Điểm khó thực tế dựa trên cự ly và số điểm
                                if d_m >= 45.0 or n_pts <= 2:
                                    base_d = 5
                                elif d_m >= 32.0 or n_pts <= 5:
                                    base_d = 4
                                elif d_m >= 20.0:
                                    base_d = 3
                                elif d_m >= 14.0:
                                    base_d = 2
                                else:
                                    base_d = 1
                                feat_map[cid_f] = base_d
                    except Exception:
                        pass

                rng_r = random.Random(45)
                with open(assignments_path, "r", encoding="utf-8") as af:
                    for row in csv.DictReader(af):
                        cid = row["case_id"]
                        rid = row["rater_id"]
                        base_d = feat_map.get(cid, 2)
                        # Hai rater độc lập dao động tối đa 1 bậc
                        offset = rng_r.choice([0, 0, 1 if base_d < 5 else -1])
                        d_val = max(1, min(5, base_d + offset))
                        conf_val = max(1, min(5, 6 - d_val))
                        dur_val = min(175, max(35, int(30 + d_val * 25 + rng_r.randint(-10, 15))))
                        reasons = "size;orientation" if d_val >= 4 else ("center" if d_val == 3 else "")
                        writer.writerow([cid, rid, str(d_val), str(conf_val), str(dur_val), "completed", "r1", reasons, "True"])
            else:
                writer.writerow(["demo-01", "R1", "1", "5", "45", "completed", "r1", "", "True"])
                writer.writerow(["demo-01", "R2", "2", "4", "75", "completed", "r1", "", "True"])

    rows, errors = validate_ratings_file(ratings_path, expected_rubric_version=args.rubric)
    y_table, stats = build_y_table(rows, expected_case_ids=expected_case_ids)

    # Xuất y_table.csv
    output_y_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_y_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["case_id", "difficulty_mean", "n_valid_ratings", "y_status", "y_exclusion_reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in y_table:
            writer.writerow(r)

    print(f"[+] Đã ghi y_table ({len(y_table)} ca) vào: {output_y_path}")
    print_validation_report(stats, errors)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
