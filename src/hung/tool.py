"""src/hung/tool.py

M45 ANNOTATION & QA STUDIO (BỘ CÔNG CỤ DÀNH CHO VAI TRÒ HƯNG)
Dự án: M45 - LiDAR Point Density & Annotation Difficulty Analyzer
Phụ trách: Hưng (Thu nhãn, đo sai số, quản lý phân công, kiểm thử CVAT)

Tuân thủ:
- Mục 2: Luật cứng (hiển thị banner DEMO khi có dữ liệu giả lập, không bịa số/nhãn).
- Mục 6: Quyết định đã chốt.
- Mục 8: Toàn bộ 6 module của Hưng và 3 cổng con người G1, G3, G4.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Cấu hình encoding UTF-8
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hung.box_errors import (
    check_manual_calculation,
    compute_center_error,
    compute_orientation_error_deg,
    compute_size_errors,
)
from src.hung.convert_boxes import parse_cvat_xml, parse_datumaro_json
from src.hung.make_assignments import generate_demo_cases, make_assignments
from src.hung.make_cvat_pack import make_cvat_packs
from src.hung.test_box_roundtrip import (
    REF_BOX,
    REF_QUAT_WXYZ,
    generate_test_assets,
    verify_box,
)
from src.hung.validate_ratings import (
    build_y_table,
    validate_ratings_file,
)

WORK_DIR = REPO_ROOT / "data" / "work"
LABELS_DIR = REPO_ROOT / "data" / "labels"
CVAT_PACKS_DIR = REPO_ROOT / "data" / "cvat_packs"
ROUNDTRIP_DIR = REPO_ROOT / "data" / "roundtrip_test"
BLOCKED_FILE = REPO_ROOT / "BLOCKED.md"


def check_is_demo() -> bool:
    """Kiểm tra xem hệ thống có đang chạy trên dữ liệu giả lập (is_demo=True) hay không."""
    cases_path = WORK_DIR / "cases.csv"
    if cases_path.exists():
        try:
            df = pd.read_csv(cases_path)
            if "is_demo" in df.columns:
                return (df["is_demo"].astype(str).str.lower() == "true").any()
        except Exception:
            pass
    ratings_path = LABELS_DIR / "ratings.csv"
    if ratings_path.exists():
        try:
            df = pd.read_csv(ratings_path, comment="#")
            if "is_demo" in df.columns:
                return (df["is_demo"].astype(str).str.lower() == "true").any()
        except Exception:
            pass
    return True


st.set_page_config(
    page_title="M45 Annotation Studio - Vai Hưng",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hiển thị Banner DEMO theo Luật cứng số 2
if check_is_demo():
    st.warning("⚠️ **CHẾ ĐỘ DEMO ĐANG HOẠT ĐỘNG (is_demo=True)** – Dữ liệu giả lập theo mục 5.10; không vào báo cáo thật.")

st.title("🎯 M45 Annotation & Quality Assurance Studio")
st.caption("Bộ công cụ chuyên biệt cho vai trò Hưng – Thu nhãn, đo sai số & nghiệm thu CVAT 3D")

tabs = st.tabs([
    "📋 1. Quản lý phân công",
    "📦 2. CVAT 3D Pack Center",
    "✅ 3. Kiểm tra nhãn & y_table",
    "📊 4. Đo sai số 3D Box",
    "🔄 5. Kiểm thử Roundtrip (G1)",
    "🚧 6. Cổng con người (BLOCKED)",
])

# ==============================================================================
# TAB 1: QUẢN LÝ PHÂN CÔNG (ASSIGNMENTS)
# ==============================================================================
with tabs[0]:
    st.subheader("1. Quản lý phân công & Cân bằng lượt dán nhãn (Mục 8.2)")
    assignments_path = LABELS_DIR / "assignments.csv"

    col_btn1, col_btn2 = st.columns([2, 4])
    with col_btn1:
        if st.button("🔄 Tạo lại phân công (Seed 45)", use_container_width=True):
            cases_path = WORK_DIR / "cases.csv"
            if not cases_path.exists():
                cases = generate_demo_cases()
            else:
                cases = []
                with open(cases_path, "r", encoding="utf-8") as f:
                    cases = list(csv.DictReader(f))
            assigns = make_assignments(cases, seed=45)
            assignments_path.parent.mkdir(parents=True, exist_ok=True)
            with open(assignments_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["case_id", "task_code", "rater_id", "pass_no", "batch_id"])
                writer.writeheader()
                writer.writerows(assigns)
            st.success(f"Đã tạo {len(assigns)} lượt dán nhãn thành công!")

    if assignments_path.exists():
        df_assign = pd.read_csv(assignments_path)

        # Thống kê tổng quan
        c1, c2, c3, c4, c5 = st.columns(5)
        total_slots = len(df_assign)
        r1_cnt = (df_assign["rater_id"] == "R1").sum()
        r2_cnt = (df_assign["rater_id"] == "R2").sum()
        r3_cnt = (df_assign["rater_id"] == "R3").sum()
        c1.metric("Tổng lượt phân công", total_slots)
        c2.metric("Số ca duy nhất", df_assign["case_id"].nunique())
        c3.metric("Rater R1", f"{r1_cnt} lượt")
        c4.metric("Rater R2", f"{r2_cnt} lượt")
        c5.metric("Rater R3", f"{r3_cnt} lượt")

        # Phân bố theo Batch
        st.markdown("#### Phân bố lượt dán nhãn theo từng Batch")
        batch_stats = df_assign.groupby(["batch_id", "rater_id"]).size().unstack(fill_value=0)
        batch_stats["Tổng"] = batch_stats.sum(axis=1)
        st.dataframe(batch_stats, use_container_width=True)

        fig_bar = px.bar(
            df_assign,
            x="batch_id",
            color="rater_id",
            barmode="group",
            title="Số lượt dán nhãn của từng Rater qua các Batch (Cân bằng & Luân phiên)",
            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"],
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Bộ lọc bảng
        st.markdown("#### Chi tiết danh sách phân công (Bảo mật: Chỉ đưa task_code cho Rater)")
        f_batch = st.multiselect("Lọc theo Batch:", sorted(df_assign["batch_id"].unique()), default=[])
        f_rater = st.multiselect("Lọc theo Rater:", ["R1", "R2", "R3"], default=[])

        filtered_assign = df_assign.copy()
        if f_batch:
            filtered_assign = filtered_assign[filtered_assign["batch_id"].isin(f_batch)]
        if f_rater:
            filtered_assign = filtered_assign[filtered_assign["rater_id"].isin(f_rater)]

        st.dataframe(filtered_assign, use_container_width=True, height=350)
    else:
        st.info("Chưa có file assignments.csv. Hãy nhấn 'Tạo lại phân công' ở trên.")

# ==============================================================================
# TAB 2: CVAT 3D PACK CENTER
# ==============================================================================
with tabs[1]:
    st.subheader("2. Trung tâm Đóng gói & Tải CVAT 3D Packs (Mục 8.3)")
    st.markdown(
        "> [!IMPORTANT]\n"
        "> **Nguyên tắc che thông tin (Blinding):** Gói dữ liệu CVAT chỉ chứa các file điểm `.pcd` đặt tên theo `task_code` và phân loại lớp (`classes.csv`). Tuyệt đối không chứa box gốc, số điểm $N$, score hay thông tin rater khác."
    )

    if st.button("📦 Chạy đóng gói toàn bộ CVAT Packs"):
        make_cvat_packs()
        st.success("Đã hoàn tất tạo 15 gói CVAT 3D!")

    zip_files = list(CVAT_PACKS_DIR.glob("*.zip")) if CVAT_PACKS_DIR.exists() else []
    if zip_files:
        st.markdown(f"Đã tạo **{len(zip_files)} gói ZIP** sẵn sàng tải về:")

        pack_cols = st.columns(3)
        for i, zf in enumerate(sorted(zip_files)):
            col_idx = i % 3
            with pack_cols[col_idx]:
                with open(zf, "rb") as f_bytes:
                    btn = st.download_button(
                        label=f"⬇️ {zf.name}",
                        data=f_bytes,
                        file_name=zf.name,
                        mime="application/zip",
                        use_container_width=True,
                    )
    else:
        st.info("Chưa có gói CVAT ZIP nào trong data/cvat_packs/. Hãy nhấn nút đóng gói ở trên.")

    # Xem trước tài liệu hướng dẫn
    sample_inst = CVAT_PACKS_DIR / "pack_B1_R1" / "INSTRUCTIONS.txt"
    if sample_inst.exists():
        with st.expander("📖 Xem tài liệu hướng dẫn 1 trang (INSTRUCTIONS.txt) đính kèm trong mỗi gói"):
            st.text(sample_inst.read_text(encoding="utf-8"))

# ==============================================================================
# TAB 3: KIỂM TRA NHÃN & Y_TABLE
# ==============================================================================
with tabs[2]:
    st.subheader("3. Kiểm tra tính hợp lệ Phiếu nhãn & Xuất y_table (Mục 8.4)")
    ratings_path = LABELS_DIR / "ratings.csv"
    y_table_path = LABELS_DIR / "y_table.csv"

    col_up, col_check = st.columns([2, 4])
    with col_up:
        uploaded_ratings = st.file_uploader("Tải lên file ratings.csv mới:", type=["csv"])
        if uploaded_ratings is not None:
            with open(ratings_path, "wb") as f_out:
                f_out.write(uploaded_ratings.getbuffer())
            st.success("Đã lưu file ratings.csv tải lên!")

    if ratings_path.exists():
        rows, errors = validate_ratings_file(ratings_path, expected_rubric_version="r1")

        assignments_path = LABELS_DIR / "assignments.csv"
        exp_cases = set()
        if assignments_path.exists():
            with open(assignments_path, "r", encoding="utf-8") as af:
                exp_cases = set(r["case_id"] for r in csv.DictReader(af) if r.get("case_id"))

        y_table, stats = build_y_table(rows, expected_case_ids=exp_cases)

        # Cột trạng thái
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Lỗi chặn (Validation Errors)", len(errors), delta=None if not errors else f"-{len(errors)} lỗi")
        m2.metric("Số ca hợp lệ (y_status=ok)", f"{stats['n_ok']}/{stats['total_cases']}")
        m3.metric("Số ca thiếu dữ liệu", len(stats["missing_cases"]))
        total_comp = stats["n_ok"]
        ratio_le_1 = (stats["diff_le_1"] / total_comp * 100.0) if total_comp > 0 else 0.0
        m4.metric("Tỷ lệ lệch độ khó ≤ 1 bậc", f"{ratio_le_1:.1f}%")

        if errors:
            st.error(f"Phát hiện {len(errors)} lỗi vi phạm hợp đồng dữ liệu / rubric:")
            for e in errors[:10]:
                st.write(f"- ❌ {e}")
            if len(errors) > 10:
                st.caption(f"... và {len(errors) - 10} lỗi khác.")
        else:
            st.success("✅ Toàn bộ các dòng ratings.csv đều hợp lệ theo rubric r1 và luật cứng!")

        # Hiển thị y_table
        st.markdown("#### Bảng nhãn hợp nhất y_table.csv (Mục 5.6)")
        df_y = pd.DataFrame(y_table)
        st.dataframe(df_y, use_container_width=True, height=250)

        # Nút ghi file y_table
        if st.button("💾 Lưu y_table.csv vào data/labels/"):
            y_table_path.parent.mkdir(parents=True, exist_ok=True)
            df_y.to_csv(y_table_path, index=False, encoding="utf-8")
            st.success(f"Đã lưu thành công y_table.csv tại {y_table_path}")

        # Biểu đồ phân bố độ khó
        if not df_y.empty and "difficulty_mean" in df_y.columns:
            valid_y = df_y[df_y["difficulty_mean"] != ""]["difficulty_mean"].astype(float)
            if not valid_y.empty:
                fig_hist_y = px.histogram(
                    valid_y,
                    nbins=10,
                    title="Phân bố điểm nhãn trung bình y (difficulty_mean)",
                    labels={"value": "Điểm độ khó trung bình (1-5)"},
                    color_discrete_sequence=["#2ca02c"],
                )
                st.plotly_chart(fig_hist_y, use_container_width=True)

# ==============================================================================
# TAB 4: ĐO SAI SỐ 3D BOX (BOX ERRORS)
# ==============================================================================
with tabs[3]:
    st.subheader("4. Đo sai số Bounding Box 3D của Rater (Mục 5.8 & 8.6)")
    box_errors_path = LABELS_DIR / "box_errors.csv"

    col_btn_err1, col_btn_err2 = st.columns([3, 3])
    with col_btn_err1:
        if st.button("⚙️ Chạy kiểm chứng tính tay 1 ca (Analytical Check)"):
            ok = check_manual_calculation()
            if ok:
                st.success("✅ Tuyệt vời: Thuật toán đo sai số khớp tính tay 1 ca với sai số tuyệt đối < 1e-6!")
            else:
                st.error("❌ Tính tay không khớp!")

    with col_btn_err2:
        if st.button("🔄 Tính toán lại toàn bộ box_errors.csv"):
            import subprocess
            res = subprocess.run([sys.executable, str(REPO_ROOT / "src" / "hung" / "box_errors.py")], capture_output=True, text=True)
            if res.returncode == 0:
                st.success("Đã tính toán và cập nhật box_errors.csv!")
            else:
                st.error(f"Lỗi: {res.stderr}")

    if box_errors_path.exists():
        df_err = pd.read_csv(box_errors_path)
        # Chuyển đổi các cột số
        num_cols = ["center_err_m", "size_err_pct", "dw_pct", "dl_pct", "dh_pct", "orientation_err_deg"]
        for c in num_cols:
            if c in df_err.columns:
                df_err[c] = pd.to_numeric(df_err[c], errors="coerce")

        # Thống kê tổng hợp
        e1, e2, e3 = st.columns(3)
        mean_center = df_err["center_err_m"].mean()
        mean_size = df_err["size_err_pct"].mean()
        mean_orient = df_err["orientation_err_deg"].mean()
        e1.metric("Lệch tâm trung bình", f"{mean_center:.3f} m" if not np.isnan(mean_center) else "chưa đo")
        e2.metric("Lệch kích thước TB (size_err_pct)", f"{mean_size:.1f}%" if not np.isnan(mean_size) else "chưa đo")
        e3.metric("Lệch hướng TB (chỉ vehicle.car)", f"{mean_orient:.2f}°" if not np.isnan(mean_orient) else "chưa đo")

        # Biểu đồ phân tích Plotly
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig_center = px.box(
                df_err,
                x="rater_id",
                y="center_err_m",
                points="all",
                color="rater_id",
                title="Phân bố sai số lệch tâm (center_err_m) theo Rater",
            )
            st.plotly_chart(fig_center, use_container_width=True)

        with col_chart2:
            fig_size = px.box(
                df_err.melt(id_vars=["case_id", "rater_id"], value_vars=["dw_pct", "dl_pct", "dh_pct"]),
                x="variable",
                y="value",
                color="variable",
                title="Sai số kích thước từng chiều: Chiều rộng (dw), Chiều dài (dl), Chiều cao (dh)",
                labels={"variable": "Chiều kích thước", "value": "Sai số (%)"},
            )
            st.plotly_chart(fig_size, use_container_width=True)

        st.markdown("#### Chi tiết bảng số liệu sai số (box_errors.csv)")
        st.dataframe(df_err, use_container_width=True, height=350)
    else:
        st.info("Chưa có file box_errors.csv. Hãy nhấn 'Tính toán lại toàn bộ' ở trên.")

# ==============================================================================
# TAB 5: KIỂM THỬ ROUNDTRIP CVAT (CỔNG G1)
# ==============================================================================
with tabs[4]:
    st.subheader("5. Khu vực kiểm thử Round-trip Bounding Box CVAT (Cổng G1 & Mục 8.5)")
    st.markdown(
        "> [!TIP]\n"
        "> **Yêu cầu kiểm thử:** Tạo ca thử có box biết trước với $w \\ne l$ và xoay $90^\\circ$. Người import vào CVAT và export lại, script so sánh: tâm/kích thước $\\le 10^{-4}$ m, hướng $\\le 0.01^\\circ$. Kiểm tra rõ thứ tự $w/l/h$, đơn vị độ/radian, thứ tự góc Euler."
    )

    c_box1, c_box2 = st.columns(2)
    with c_box1:
        st.markdown("#### Hộp chuẩn thử nghiệm (Known Reference Box):")
        st.write(f"- **Tâm (x, y, z):** `{REF_BOX['center_m']}` m")
        st.write(f"- **Kích thước (w, l, h):** `{REF_BOX['size_wlh_m']}` m (w ≠ l: 2.2m vs 4.8m)")
        st.write(f"- **Góc xoay Yaw:** `{REF_BOX['yaw_deg']}°` ({REF_BOX['yaw_rad']:.6f} rad)")
        st.write(f"- **Quaternion [w, x, y, z]:** `{[round(q, 6) for q in REF_QUAT_WXYZ]}`")

    with c_box2:
        st.markdown("#### Tải file mẫu để import vào CVAT:")
        ref_path, datumaro_path, xml_path = generate_test_assets()
        with open(datumaro_path, "rb") as f_dat:
            st.download_button(
                "⬇️ Tải mẫu Datumaro 1.0 (JSON)",
                data=f_dat,
                file_name="datumaro_sample.json",
                mime="application/json",
            )
        with open(xml_path, "rb") as f_xml:
            st.download_button(
                "⬇️ Tải mẫu CVAT XML 1.1 (XML)",
                data=f_xml,
                file_name="cvat_sample.xml",
                mime="application/xml",
            )

    st.markdown("---")
    st.markdown("#### Tải lên file export từ CVAT thực tế để kiểm tra Round-trip:")
    uploaded_cvat = st.file_uploader("Chọn file export từ CVAT (JSON hoặc XML):", type=["json", "xml"])

    if st.button("🧪 Chạy kiểm tra Round-trip"):
        if uploaded_cvat is not None:
            temp_path = ROUNDTRIP_DIR / uploaded_cvat.name
            with open(temp_path, "wb") as f_tmp:
                f_tmp.write(uploaded_cvat.getbuffer())
            target_f = temp_path
        else:
            st.info("Chưa tải file lên. Tiến hành tự kiểm trên các file sinh chuẩn (Self-test).")
            target_f = datumaro_path

        if target_f.suffix.lower() == ".json":
            with open(target_f, "r", encoding="utf-8") as f:
                d = json.load(f)
            boxes = parse_datumaro_json(d, euler_order="xyz", dim_order="wlh")
        else:
            boxes = parse_cvat_xml(target_f, euler_order="xyz", dim_order="wlh")

        task_code = REF_BOX["task_code"]
        converted = boxes.get(task_code)
        if not converted and len(boxes) == 1:
            converted = next(iter(boxes.values()))

        if converted:
            passed, metrics, errs = verify_box(
                converted,
                tuple(REF_BOX["center_m"]),
                tuple(REF_BOX["size_wlh_m"]),
                tuple(REF_QUAT_WXYZ),
            )

            r1, r2, r3 = st.columns(3)
            r1.metric("Sai số tâm (≤ 1e-4 m)", f"{metrics['center_err_m']:.8f} m", delta="Đạt" if metrics['center_err_m'] <= 1e-4 else "Vượt ngưỡng")
            r2.metric("Sai số kích thước max (≤ 1e-4 m)", f"{metrics['max_size_err_m']:.8f} m", delta="Đạt" if metrics['max_size_err_m'] <= 1e-4 else "Vượt ngưỡng")
            r3.metric("Sai số góc hướng (≤ 0.01°)", f"{metrics['orientation_err_deg']:.6f}°", delta="Đạt" if metrics['orientation_err_deg'] <= 0.01 else "Vượt ngưỡng")

            if passed:
                st.success("🎉 **KẾT LUẬN: test roundtrip đạt.** Đã kiểm tra rõ thứ tự w/l/h, đơn vị radian và thứ tự góc Euler.")
            else:
                st.error("❌ **KẾT LUẬN: test roundtrip KHÔNG đạt.**")
                for e in errs:
                    st.write(f"- {e}")
        else:
            st.error("Không tìm thấy dữ liệu bounding box trong file.")

# ==============================================================================
# TAB 6: CỔNG CON NGƯỜI & BLOCKED.MD
# ==============================================================================
with tabs[5]:
    st.subheader("6. Quản lý Cổng con người (Human-in-the-loop & BLOCKED.md)")
    st.markdown("Theo mục 0.3 và mục 10: Giúp Hưng theo dõi và điều phối các cổng cần sự can thiệp của con người.")

    if BLOCKED_FILE.exists():
        content = BLOCKED_FILE.read_text(encoding="utf-8")
        st.markdown(content)

        st.markdown("---")
        st.markdown("#### Cập nhật nội dung BLOCKED.md:")
        new_content = st.text_area("Chỉnh sửa ghi chú cổng:", value=content, height=220)
        if st.button("💾 Lưu cập nhật BLOCKED.md"):
            BLOCKED_FILE.write_text(new_content, encoding="utf-8")
            st.success("Đã lưu cập nhật BLOCKED.md thành công!")
    else:
        st.info("Chưa tìm thấy file BLOCKED.md.")
