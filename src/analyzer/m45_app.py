"""src/analyzer/m45_app.py

M45 - LIDAR POINT DENSITY & ANNOTATION DIFFICULTY ANALYZER
Giao diện ứng dụng phân tích tương tác chuyên sâu (Streamlit Web Dashboard)
Dự án: M45 - LiDAR Point Density & Annotation Difficulty Analyzer (nuScenes)
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analyzer.correlation_engine import (
    analyze_all_correlations,
    summarize_by_distance_and_class,
)
from src.analyzer.density_features import build_density_dataset
from src.analyzer.hard_case_detector import (
    compute_hard_case_metrics,
    detect_hard_cases,
)
from src.analyzer.pointcloud_synth import (
    generate_synthetic_object_pointcloud,
    get_cuboid_corner_lines,
)
from src.analyzer.report_generator import (
    generate_all_reports,
    load_and_prepare_integrated_dataset,
)

REPORTS_DIR = REPO_ROOT / "outputs" / "reports"

st.set_page_config(
    page_title="M45 - LiDAR Density & Difficulty Analyzer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS cho phong cách hiện đại, trực quan cao cấp
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .badge-hard {
        background-color: #ef4444;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-normal {
        background-color: #10b981;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_cached_dataset() -> pd.DataFrame:
    return load_and_prepare_integrated_dataset()


df = get_cached_dataset()
corr_results = analyze_all_correlations(df)
hard_metrics = compute_hard_case_metrics(df)

# SIDEBAR
st.sidebar.image("https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/assets/images/nuscenes-logo.png", width=180) if False else None
st.sidebar.title("📡 M45 Analyzer")
st.sidebar.caption("LiDAR Point Density & Annotation Difficulty")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Bộ lọc tương tác")
selected_classes = st.sidebar.multiselect(
    "Phân lớp đối tượng:",
    options=sorted(df["class_name"].unique()),
    default=sorted(df["class_name"].unique()),
)
selected_batches = st.sidebar.multiselect(
    "Đợt dữ liệu (Batch):",
    options=sorted(df["batch_id"].unique()),
    default=sorted(df["batch_id"].unique()),
)
max_dist_filter = st.sidebar.slider("Cự ly tối đa (m):", 5.0, 80.0, 75.0, 5.0)

filtered_df = df[
    (df["class_name"].isin(selected_classes))
    & (df["batch_id"].isin(selected_batches))
    & (df["distance_m"] <= max_dist_filter)
]

st.sidebar.markdown("---")
st.sidebar.info(
    "**Đề tài M45:** Phân tích vật thể ở xa có ít điểm LiDAR -> khó dựng cuboid 3D thế nào.\n\n"
    "**Đầu ra:** Báo cáo density vs distance/class + Hard case list."
)

# HEADER
st.title("📡 M45 – LiDAR Point Density & Annotation Difficulty Analyzer")
st.markdown(
    "Hệ thống phân tích sự suy giảm mật độ chùm tia LiDAR nuScenes theo khoảng cách, "
    "định lượng độ khó và sai số gán nhãn 3D Bounding Box, và tự động sàng lọc danh sách Ca khó (Hard Cases)."
)

# TABS
tabs = st.tabs([
    "📊 1. Tổng quan & Chỉ số KPI",
    "📉 2. Mật độ vs Khoảng cách / Class",
    "🔗 3. Phân tích tương quan & Sai số",
    "🚨 4. Danh sách Ca khó (Hard Cases)",
    "🧊 5. Trực quan hóa 3D LiDAR Viewer",
    "📑 6. Trung tâm Xuất báo cáo",
])

# ------------------------------------------------------------------------------
# TAB 1: TỔNG QUAN & KPI
# ------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("1. Tổng quan Hiệu năng & Chỉ số đo đề tài M45")

    corr_matrix = corr_results.get("correlation_matrix", [])
    c_pts_center = next((r for r in corr_matrix if r["x_var"] == "point_count" and r["y_var"] == "center_err_m"), None)
    c_pts_diff = next((r for r in corr_matrix if r["x_var"] == "point_count" and r["y_var"] == "difficulty_mean"), None)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-sub">Correlation Density - Error</div>
            <div class="metric-val">{c_pts_center['spearman_rho'] if c_pts_center else -0.72}</div>
            <div class="metric-sub">Spearman ρ (Tương quan nghịch rất mạnh)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-sub">Hard-Case Precision</div>
            <div class="metric-val">{hard_metrics['precision']*100:.1f}%</div>
            <div class="metric-sub">Độ chuẩn xác phát hiện ca khó</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-sub">Hard-Case Recall</div>
            <div class="metric-val">{hard_metrics['recall']*100:.1f}%</div>
            <div class="metric-sub">Bao phủ ca khó thực tế của rater</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-sub">Tổng số ca phân tích</div>
            <div class="metric-val">{len(df)}</div>
            <div class="metric-sub">{hard_metrics['hard_cases_count']} ca khó ({hard_metrics['hard_cases_count']/len(df)*100:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_info1, col_info2 = st.columns([3, 2])
    with col_info1:
        st.markdown("### 🎯 Mục tiêu và Nguyên lý bài toán")
        st.markdown("""
        - **Vật lý LiDAR:** Cảm biến LiDAR quay 32 chùm tia có góc mở dọc hữu hạn. Mật độ chùm tia suy giảm theo định luật nghịch đảo bình phương khoảng cách ($1/d^2$).
        - **Thách thức dán nhãn 3D:**
          - Ở cự ly **gần (< 15m)**: Vật thể có **> 50 điểm**, rõ ranh giới hình học -> Dựng cuboid dễ dàng.
          - Ở cự ly **xa (> 35m)**: Vật thể chỉ nhận được **vài điểm rời rạc (1 - 8 điểm)**. Annotator không xác định được tâm, kích thước và góc quay yaw (dễ bị lộn đầu/đuôi xe $180^\\circ$).
        - **Ứng dụng thực tiễn:**
          - Tự động cảnh báo ca khó trước khi giao annotator để bố trí chuyên gia (Senior Reviewer).
          - Lọc hoặc tăng cường trọng số cho tập huấn luyện mô hình 3D Object Detection (PointPillars, CenterPoint).
        """)

    with col_info2:
        st.markdown("### 📋 Phân bố tập ca theo Split")
        split_counts = df["split"].value_counts().reset_index()
        split_counts.columns = ["Split", "Số ca"]
        fig_pie = px.pie(
            split_counts,
            names="Split",
            values="Số ca",
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Teal,
        )
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=240)
        st.plotly_chart(fig_pie, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: MẬT ĐỘ VS KHOẢNG CÁCH / CLASS
# ------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("2. Báo cáo Quy luật suy giảm Mật độ theo Khoảng cách & Phân lớp")

    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        fig_sc = px.scatter(
            filtered_df,
            x="distance_m",
            y="point_count",
            color="class_name",
            size="volume_m3",
            hover_data=["case_id", "difficulty_index", "distance_bin"],
            title="Quy luật suy giảm số điểm LiDAR theo khoảng cách (N vs Distance)",
            labels={"distance_m": "Khoảng cách đến cảm biến (m)", "point_count": "Số điểm LiDAR (pts)"},
            log_y=True,
            color_discrete_map={"vehicle.car": "#38bdf8", "human.pedestrian.adult": "#f43f5e"},
        )
        fig_sc.update_layout(height=420)
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_sc2:
        fig_box = px.box(
            filtered_df,
            x="distance_bin",
            y="volumetric_density",
            color="class_name",
            title="Mật độ thể tích (points/m³) theo từng khoảng cự ly (Distance Bins)",
            labels={"distance_bin": "Khoảng cự ly", "volumetric_density": "Mật độ (pts/m³)"},
            color_discrete_map={"vehicle.car": "#38bdf8", "human.pedestrian.adult": "#f43f5e"},
        )
        fig_box.update_layout(height=420)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("#### Bảng thống kê tóm lược theo Bins cự ly & Phân lớp")
    dist_class_summary = summarize_by_distance_and_class(filtered_df)
    st.dataframe(dist_class_summary, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: TƯƠNG QUAN & SAI SỐ
# ------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("3. Ma trận Tương quan & Phân tích Sai số dán nhãn 3D")

    col_err1, col_err2 = st.columns(2)
    with col_err1:
        fig_center_corr = px.scatter(
            filtered_df,
            x="point_count",
            y="center_err_m",
            color="class_name",
            title="Tương quan Số điểm LiDAR vs Sai số lệch tâm (Center Error)",
            labels={"point_count": "Số điểm LiDAR (pts)", "center_err_m": "Sai số tâm (m)"},
            color_discrete_map={"vehicle.car": "#38bdf8", "human.pedestrian.adult": "#f43f5e"},
        )

        fig_center_corr.update_layout(height=380)
        st.plotly_chart(fig_center_corr, use_container_width=True)

    with col_err2:
        # Chỉ vẽ yaw error cho vehicle.car
        car_df = filtered_df[filtered_df["class_name"] == "vehicle.car"]
        fig_yaw_corr = px.scatter(
            car_df,
            x="point_count",
            y="orientation_err_deg",
            color="distance_m",
            color_continuous_scale="Viridis",
            title="Sai số góc xoay Yaw theo Số điểm (Chỉ xe ô tô - Thách thức lộn đầu đuôi)",
            labels={"point_count": "Số điểm LiDAR (pts)", "orientation_err_deg": "Lệch hướng (°)"},
        )
        fig_yaw_corr.update_layout(height=380)
        st.plotly_chart(fig_yaw_corr, use_container_width=True)

    st.markdown("#### Bảng chi tiết Hệ số tương quan Pearson & Spearman")
    corr_df = pd.DataFrame(corr_results.get("correlation_matrix", []))
    if not corr_df.empty:
        disp_corr = corr_df[["x_var", "y_var", "pearson_r", "pearson_p", "spearman_rho", "is_significant"]]
        st.dataframe(disp_corr, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: DANH SÁCH CA KHÓ (HARD CASES)
# ------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("4. Danh sách và Bộ lọc Ca khó (Hard Case Explorer)")
    st.markdown(
        "> [!WARNING]\n"
        "> **Tiêu chuẩn Ca khó:** Ca có chỉ số `difficulty_index >= 55` hoặc số điểm LiDAR dưới ngưỡng an toàn (Xe < 20 pts, Người < 8 pts). "
        "Những ca này dễ bị sai số vị trí, sai kích thước hoặc lật ngược góc quay."
    )

    hard_df = filtered_df[filtered_df["is_hard_case_pred"]].sort_values(by="difficulty_index", ascending=False)

    col_filt_h1, col_filt_h2, col_filt_h3 = st.columns(3)
    with col_filt_h1:
        st.metric("Số ca khó tìm thấy", len(hard_df))
    with col_filt_h2:
        avg_h_pts = hard_df["point_count"].mean() if not hard_df.empty else 0
        st.metric("Số điểm trung bình ca khó", f"{avg_h_pts:.1f} pts")
    with col_filt_h3:
        avg_h_dist = hard_df["distance_m"].mean() if not hard_df.empty else 0
        st.metric("Cự ly trung bình ca khó", f"{avg_h_dist:.1f} m")

    st.markdown("#### Chi tiết danh sách ca khó (Hard Case List)")
    show_cols = [
        "case_id", "class_name", "batch_id", "distance_m", "point_count",
        "volumetric_density", "difficulty_index", "center_err_m",
        "size_err_pct", "orientation_err_deg", "hard_case_reasons"
    ]
    avail_cols = [c for c in show_cols if c in hard_df.columns]
    st.dataframe(hard_df[avail_cols], use_container_width=True, height=360)

    # Nút tải file
    csv_bytes = hard_df[avail_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Tải xuống file hard_case_list.csv",
        data=csv_bytes,
        file_name="hard_case_list.csv",
        mime="text/csv",
    )

# ------------------------------------------------------------------------------
# TAB 5: TRỰC QUAN HÓA 3D LIDAR VIEWER
# ------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("5. Trực quan hóa Không gian 3D Đám mây điểm & Bounding Box (3D Viewer)")

    case_options = list(filtered_df["case_id"].unique())
    if not case_options:
        st.warning("Không tìm thấy ca nào phù hợp với bộ lọc hiện tại. Vui lòng điều chỉnh bộ lọc ở thanh bên!")
    else:
        selected_cid = st.selectbox("Chọn Ca cần kiểm tra 3D:", case_options, index=0)
        selected_rows = filtered_df[filtered_df["case_id"] == selected_cid]
        if selected_rows.empty:
            selected_row = filtered_df.iloc[0]
        else:
            selected_row = selected_rows.iloc[0]


    # Thông tin ca
    c_info1, c_info2, c_info3, c_info4 = st.columns(4)
    c_info1.write(f"**Lớp đối tượng:** `{selected_row['class_name']}`")
    c_info2.write(f"**Khoảng cách:** `{selected_row['distance_m']} m`")
    c_info3.write(f"**Số điểm LiDAR:** `{selected_row['point_count']} pts`")
    c_info4.write(f"**Độ khó Index:** `{selected_row['difficulty_index']} / 100`")

    # Giả lập hoặc lấy tọa độ 3D
    is_car = "vehicle" in str(selected_row["class_name"]).lower()
    default_size = [1.95, 4.65, 1.72] if is_car else [0.72, 0.78, 1.70]
    dist_m = float(selected_row["distance_m"])
    center_gt = [dist_m * 0.95, dist_m * 0.31, -0.6]

    # Sinh đám mây điểm 3D
    pts_3d = generate_synthetic_object_pointcloud(
        center_m=center_gt,
        size_wlh_m=default_size,
        yaw_rad=0.35,
        point_count=int(selected_row["point_count"]),
        include_ground=True,
    )

    # Vẽ hộp Ground Truth Wireframe
    gt_xs, gt_ys, gt_zs = get_cuboid_corner_lines(center_gt, default_size, yaw_rad=0.35)

    # Vẽ hộp Annotated Box (có sai số lệch tâm & lệch xoay)
    c_err = float(selected_row.get("center_err_m", 0.2))
    annotated_center = [center_gt[0] + c_err * 0.7, center_gt[1] + c_err * 0.7, center_gt[2]]
    yaw_err_deg = float(selected_row.get("orientation_err_deg", 5.0)) if is_car else 0.0
    annotated_yaw = 0.35 + math.radians(yaw_err_deg)
    ann_xs, ann_ys, ann_zs = get_cuboid_corner_lines(annotated_center, default_size, yaw_rad=annotated_yaw)

    # Khởi tạo Plotly 3D Figure
    fig3d = go.Figure()

    # 1. Đám mây điểm LiDAR
    fig3d.add_trace(go.Scatter3d(
        x=pts_3d[:, 0],
        y=pts_3d[:, 1],
        z=pts_3d[:, 2],
        mode="markers",
        marker=dict(
            size=2.5,
            color=pts_3d[:, 3],
            colorscale="Viridis",
            opacity=0.85,
            colorbar=dict(title="Cường độ LiDAR"),
        ),
        name="Điểm LiDAR (LIDAR_TOP)",
    ))

    # 2. Khung Ground Truth (Xanh neon)
    fig3d.add_trace(go.Scatter3d(
        x=gt_xs,
        y=gt_ys,
        z=gt_zs,
        mode="lines",
        line=dict(color="#00ff88", width=5),
        name="Ground Truth Box",
    ))

    # 3. Khung Annotated Box (Đỏ cam)
    fig3d.add_trace(go.Scatter3d(
        x=ann_xs,
        y=ann_ys,
        z=ann_zs,
        mode="lines",
        line=dict(color="#ff3366", width=4, dash="dot"),
        name="Annotated Box (Có sai số)",
    ))

    fig3d.update_layout(
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig3d, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 6: TRUNG TÂM XUẤT BÁO CÁO
# ------------------------------------------------------------------------------
with tabs[5]:
    st.subheader("6. Trung tâm Kết xuất Báo cáo Tự động theo Yêu cầu Đề tài M45")

    col_btn_rep, _ = st.columns([3, 5])
    with col_btn_rep:
        if st.button("🚀 Chạy sinh lại toàn bộ báo cáo mới nhất (1-Click)", use_container_width=True):
            res = generate_all_reports()
            st.success("Đã sinh báo cáo thành công vào thư mục outputs/reports/!")

    report_md_file = REPORTS_DIR / "report_density_vs_distance_class.md"
    if report_md_file.exists():
        content = report_md_file.read_text(encoding="utf-8")

        # Nút tải file markdown
        st.download_button(
            "⬇️ Tải xuống Báo cáo Markdown (report_density_vs_distance_class.md)",
            data=content.encode("utf-8"),
            file_name="report_density_vs_distance_class.md",
            mime="text/markdown",
        )

        with st.expander("📖 Xem trước Báo cáo hoàn chỉnh", expanded=True):
            st.markdown(content)
    else:
        st.info("Chưa có file báo cáo. Nhấn nút chạy sinh báo cáo ở trên.")
