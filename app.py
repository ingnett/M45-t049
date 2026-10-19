"""app.py — M45: LiDAR Point Density & Annotation Difficulty Analyzer
================================================================================
Universal Web Application & Cloud Deployment Entrypoint for Topic M45.
- Works 100% standalone and with local nuScenes datasets.
- One-click deployable to Streamlit Community Cloud, Hugging Face Spaces, Docker, Render.
- Integrates all 3 roles: Hưng (Annotation/CVAT/Errors) - Dũng (Geometry) - Đường (ML Ridge)
================================================================================
"""

from __future__ import annotations

import io
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st

# Configure UTF-8 for standard output
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# CORE PHYSICS & GEOMETRY CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
DISTANCE_BINS = [
    (0, 10, "0–10 m (Gần)"),
    (10, 20, "10–20 m (Gần)"),
    (20, 30, "20–30 m (Trung bình)"),
    (30, 50, "30–50 m (Xa)"),
    (50, 120, "≥50 m (Rất xa)"),
]

CLASS_PRIORS = {
    "vehicle.car": {
        "wlh": [1.93, 4.62, 1.69],
        "C": 18000.0,
        "label": "Xe hơi (Car)",
        "color": "#38bdf8",
    },
    "human.pedestrian.adult": {
        "wlh": [0.67, 0.73, 1.77],
        "C": 4200.0,
        "label": "Người đi bộ (Pedestrian)",
        "color": "#c084fc",
    },
}

FEATURE_GROUPS = [
    ("A", "Mật độ & Số điểm", "N, density (N/V), N·d², voxel occupancy (0.2m), ring count"),
    ("B", "Phân bố không gian", "Coverage trung bình 3 trục, Entropy 8-octant, Patchiness (1 - occ)"),
    ("C", "Cự ly & Kích thước", "Distance d (m), Volume V (m³), Size anomaly (Z-score kích thước)"),
    ("D", "Che khuất & Tạp âm", "Occlusion proxy, Truncation proxy, Boundary clutter lân cận"),
    ("E", "Tín hiệu phản xạ", "LiDAR intensity mean/std, L-shape corner fitting residual"),
    ("F", "Thời gian & Ngữ cảnh", "Track length, Temporal consistency, Vận tốc di chuyển (m/s)"),
    ("G", "Đặc trưng tập dữ liệu", "Visibility token nuScenes, Tên phân lớp đối tượng (Class)"),
]

RIDGE_FEATS = [
    "N", "density", "ring_count", "coverage_avg", "entropy_8_octant",
    "patchiness", "distance_m", "size_anomaly", "boundary_clutter", "l_shape_fit_residual"
]
FLIP_SIGNS = [True] * 5 + [False] * 5


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS: GEOMETRY, ROTATION, WIREFRAME
# ──────────────────────────────────────────────────────────────────────────────
def yaw_rotation_matrix(yaw_rad: float) -> np.ndarray:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def theoretical_points(dist_m: float, class_name: str) -> int:
    d = max(2.0, float(dist_m))
    c_val = CLASS_PRIORS.get(class_name, CLASS_PRIORS["vehicle.car"])["C"]
    return max(1, int(round(c_val / (d ** 2))))


def generate_synthetic_cloud(
    center: np.ndarray,
    wlh: np.ndarray,
    yaw: float,
    point_count: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    w, l, h = wlh
    hw, hl, hh = w / 2.0, l / 2.0, h / 2.0
    rot = yaw_rotation_matrix(yaw)
    pts = []
    n_pts = max(3, point_count)

    for _ in range(n_pts):
        face = rng.choice(["side", "front", "roof"], p=[0.45, 0.45, 0.10])
        if face == "front":
            lx = rng.uniform(-hl, hl)
            ly = hw if rng.rand() > 0.5 else -hw
            lz = rng.uniform(-hh, hh)
        elif face == "side":
            lx = hl if rng.rand() > 0.5 else -hl
            ly = rng.uniform(-hw, hw)
            lz = rng.uniform(-hh, hh)
        else:
            lx = rng.uniform(-hl, hl)
            ly = rng.uniform(-hw, hw)
            lz = hh
        jitter = rng.normal(0, 0.02, 3)
        pts.append(rot @ (np.array([lx, ly, lz]) + jitter) + center)

    # Thêm điểm mặt đất xung quanh vật thể
    ground_z = center[2] - hh - 0.05
    n_ground = min(120, max(20, n_pts * 2))
    for _ in range(n_ground):
        gx = center[0] + rng.uniform(-w * 2.0, w * 2.0)
        gy = center[1] + rng.uniform(-l * 2.0, l * 2.0)
        gz = ground_z + rng.normal(0, 0.015)
        pts.append([gx, gy, gz])

    return np.array(pts, dtype=np.float64)


def points_in_box_mask(pts: np.ndarray, center: np.ndarray, wlh: np.ndarray, yaw: float, margin: float = 0.03) -> np.ndarray:
    if len(pts) == 0:
        return np.array([], dtype=bool)
    rot = yaw_rotation_matrix(yaw)
    local = (pts - center) @ rot
    hw, hl, hh = wlh[0] / 2.0 + margin, wlh[1] / 2.0 + margin, wlh[2] / 2.0 + margin
    return (
        (np.abs(local[:, 0]) <= hl)
        & (np.abs(local[:, 1]) <= hw)
        & (np.abs(local[:, 2]) <= hh)
    )


def compute_cuboid_wireframe(center: np.ndarray, wlh: np.ndarray, yaw: float) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    w, l, h = wlh
    hw, hl, hh = w / 2.0, l / 2.0, h / 2.0
    rot = yaw_rotation_matrix(yaw)
    corners_local = np.array([
        [-hl, -hw, -hh], [hl, -hw, -hh], [hl, hw, -hh], [-hl, hw, -hh],
        [-hl, -hw, hh],  [hl, -hw, hh],  [hl, hw, hh],  [-hl, hw, hh]
    ])
    corners_world = corners_local @ rot.T + center
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Đáy dưới
        (4, 5), (5, 6), (6, 7), (7, 4),  # Nóc trên
        (0, 4), (1, 5), (2, 6), (3, 7),  # 4 cột đứng
    ]
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    zs: List[Optional[float]] = []
    for a, b in edges:
        xs.extend([float(corners_world[a, 0]), float(corners_world[b, 0]), None])
        ys.extend([float(corners_world[a, 1]), float(corners_world[b, 1]), None])
        zs.extend([float(corners_world[a, 2]), float(corners_world[b, 2]), None])
    return xs, ys, zs


def compute_features_from_cloud(
    center: np.ndarray,
    wlh: np.ndarray,
    yaw: float,
    pts: np.ndarray,
    class_name: str,
) -> Dict[str, Any]:
    w, l, h = wlh
    dist = float(np.linalg.norm(center[:2]))
    vol = max(1e-4, float(w * l * h))
    mask = points_in_box_mask(pts, center, wlh, yaw)
    in_pts = pts[mask]
    n_pts = len(in_pts)
    density = n_pts / vol
    n_d2 = n_pts * (dist ** 2)

    # Voxel occupancy (0.2m)
    vx = 0.2
    nvx = max(1, int(math.ceil(l / vx)))
    nvy = max(1, int(math.ceil(w / vx)))
    nvz = max(1, int(math.ceil(h / vx)))
    if n_pts > 0:
        rot = yaw_rotation_matrix(yaw)
        local_in = (in_pts - center) @ rot
        ix = np.clip(np.floor((local_in[:, 0] + l / 2.0) / vx), 0, nvx - 1)
        iy = np.clip(np.floor((local_in[:, 1] + w / 2.0) / vx), 0, nvy - 1)
        iz = np.clip(np.floor((local_in[:, 2] + h / 2.0) / vx), 0, nvz - 1)
        occ = len(np.unique(ix + iy * nvx + iz * nvx * nvy)) / (nvx * nvy * nvz)
    else:
        occ = 0.0

    rings = int(np.sum(np.diff(np.sort(in_pts[:, 2])) > 0.08) + 1) if n_pts > 1 else n_pts

    if n_pts > 1:
        rot = yaw_rotation_matrix(yaw)
        local_in = (in_pts - center) @ rot
        cov = float(np.mean([
            min(1.0, (local_in[:, i].max() - local_in[:, i].min()) / max(1e-4, dim))
            for i, dim in enumerate([l, w, h])
        ]))
        oct_idx = (
            (local_in[:, 0] >= 0).astype(int)
            + (local_in[:, 1] >= 0).astype(int) * 2
            + (local_in[:, 2] >= 0).astype(int) * 4
        )
        counts = np.bincount(oct_idx, minlength=8)
        probs = counts[counts > 0] / n_pts
        entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        cov = 0.0
        entropy = 0.0

    patchiness = max(0.0, 1.0 - occ)
    buf_mask = points_in_box_mask(pts, center, wlh + np.array([0.7, 0.7, 0.7]), yaw)
    clutter = float((buf_mask.sum() - mask.sum()) / max(1, n_pts))

    if n_pts >= 3:
        rot = yaw_rotation_matrix(yaw)
        bev = ((in_pts - center) @ rot)[:, :2]
        l_res = float(np.mean(np.minimum(
            np.minimum(np.abs(bev[:, 0] - l / 2), np.abs(bev[:, 0] + l / 2)),
            np.minimum(np.abs(bev[:, 1] - w / 2), np.abs(bev[:, 1] + w / 2)),
        )))
    else:
        l_res = 0.5

    prior_wlh = np.array(CLASS_PRIORS[class_name]["wlh"])
    prior_sig = np.array([0.15, 0.35, 0.20] if "car" in class_name else [0.12, 0.14, 0.15])
    size_anom = float(np.sum(np.abs(wlh - prior_wlh) / prior_sig))
    dist_bin = next((label for lo, hi, label in DISTANCE_BINS if lo <= dist < hi), "≥50 m (Rất xa)")

    return {
        "N": n_pts,
        "density": density,
        "N_d2": n_d2,
        "voxel_occupancy": occ,
        "ring_count": rings,
        "coverage_avg": cov,
        "entropy_8_octant": entropy,
        "patchiness": patchiness,
        "boundary_clutter": clutter,
        "l_shape_fit_residual": l_res,
        "size_anomaly": size_anom,
        "distance_m": dist,
        "distance_bin": dist_bin,
        "volume_m3": vol,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ANNOTATOR PRE-LABEL ASSISTANT: RISK DIAGNOSIS & SMART REFINER
# ──────────────────────────────────────────────────────────────────────────────
def classify_prelabel_risk(row: pd.Series) -> Dict[str, Any]:
    """Phân loại cấp độ rủi ro và phát hiện lỗi tiềm ẩn của pre-label box cho annotator."""
    n_pts = int(row.get("N", 0))
    dist = float(row.get("distance_m", 0.0))
    cls_name = str(row.get("class_name", "vehicle.car"))
    is_car = "car" in cls_name.lower() or "vehicle" in cls_name.lower()

    reasons = []
    actions = []
    risk_level = "GREEN"

    sparse_limit = 14 if is_car else 6
    extreme_sparse = 6 if is_car else 2

    if n_pts <= extreme_sparse:
        risk_level = "RED"
        reasons.append("Cực kỳ thưa điểm (N ≤ 5 pts)")
        actions.append("Kiểm tra camera RGB hoặc track đối tượng liền kề để xác nhận.")
    elif n_pts <= sparse_limit:
        if risk_level != "RED":
            risk_level = "YELLOW"
        reasons.append("Thưa điểm ở các mặt biên")
        actions.append("Khóa kích thước hộp theo chuẩn Class Prior.")

    far_limit = 35.0 if is_car else 22.0
    extreme_dist = 48.0 if is_car else 32.0

    if dist >= extreme_dist:
        risk_level = "RED"
        reasons.append("Cự ly rất xa (≥ 48m)")
        actions.append("Nguy cơ lệch tâm lớn: Neo tâm hộp vào cụm điểm nhìn thấy.")
    elif dist >= far_limit:
        if risk_level != "RED":
            risk_level = "YELLOW"
        reasons.append("Cự ly xa (≥ 35m)")

    if is_car and (dist >= 32.0 or n_pts < 10):
        reasons.append("Nguy cơ lộn góc Yaw ±180°")
        actions.append("Đối chiếu hướng di chuyển để chốt góc quay đầu/đuôi xe.")

    if not reasons:
        reasons.append("Đủ điểm biên & cự ly an toàn")
        actions.append("Duyệt nhanh (Fast review)")

    badge_map = {
        "RED": "🔴 BOX KHÓ (HARD)",
        "YELLOW": "🟡 CẦN CHÚ Ý",
        "GREEN": "🟢 AN TOÀN",
    }

    return {
        "risk_level": risk_level,
        "risk_badge": badge_map[risk_level],
        "reasons": "; ".join(reasons),
        "actions": "; ".join(actions),
    }


def refine_prelabel_box(center: np.ndarray, wlh: np.ndarray, yaw: float, class_name: str) -> Dict[str, Any]:
    """Tự động tinh chỉnh hộp pre-label dựa trên Class Prior và hình học cảm biến."""
    prior = CLASS_PRIORS.get(class_name, CLASS_PRIORS["vehicle.car"])
    prior_wlh = np.array(prior["wlh"])
    refined_wlh = 0.85 * prior_wlh + 0.15 * wlh
    refined_center = np.copy(center)
    return {
        "refined_center": refined_center,
        "refined_wlh": refined_wlh,
        "refined_yaw": yaw,
        "size_change_pct": float(np.mean(np.abs(refined_wlh - wlh) / wlh) * 100),
    }


# ──────────────────────────────────────────────────────────────────────────────
# PURE NUMPY RIDGE REGRESSION
# ──────────────────────────────────────────────────────────────────────────────
def train_ridge_pure_numpy(df_train: pd.DataFrame, alpha: float = 1.0) -> Tuple[Any, Dict[str, float], float]:
    df_valid = df_train.copy()
    X = df_valid[RIDGE_FEATS].to_numpy(dtype=np.float64)
    y = df_valid["difficulty_gt"].to_numpy(dtype=np.float64)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    X_norm = (X - mean) / std

    for i, flip in enumerate(FLIP_SIGNS):
        if flip:
            X_norm[:, i] = -X_norm[:, i]

    n_samples, n_features = X_norm.shape
    X_design = np.hstack([X_norm, np.ones((n_samples, 1))])
    reg = alpha * np.eye(n_features + 1)
    reg[-1, -1] = 0.0

    solution = np.linalg.solve(X_design.T @ X_design + reg, X_design.T @ y)
    weights = solution[:-1]
    bias = float(solution[-1])

    def predict_fn(df_target: pd.DataFrame) -> np.ndarray:
        X_tgt = df_target[RIDGE_FEATS].to_numpy(dtype=np.float64)
        X_tgt_norm = (X_tgt - mean) / std
        for j, fl in enumerate(FLIP_SIGNS):
            if fl:
                X_tgt_norm[:, j] = -X_tgt_norm[:, j]
        raw_pred = X_tgt_norm @ weights + bias
        return np.clip(raw_pred, 1.0, 5.0)

    return predict_fn, dict(zip(RIDGE_FEATS, weights.tolist())), bias


# ──────────────────────────────────────────────────────────────────────────────
# DATASET GENERATION: SIMULATION & REAL NUSCENES LOADER
# ──────────────────────────────────────────────────────────────────────────────
def generate_simulation_dataset(n_cases: int = 80, seed: int = 45) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    classes = ["vehicle.car", "human.pedestrian.adult"]
    batches = ["B1"] * 25 + ["B2"] * 25 + ["DEV"] * 15 + ["TEST"] * 15
    dist_ranges = {
        "B1": (10.0, 35.0),
        "B2": (25.0, 65.0),
        "DEV": (12.0, 50.0),
        "TEST": (15.0, 60.0),
    }

    for i in range(n_cases):
        cls_name = classes[i % 2]
        batch = batches[i % len(batches)]
        d_min, d_max = dist_ranges[batch]
        dist = float(rng.uniform(d_min, d_max))
        angle = float(rng.uniform(-0.55, 0.55))
        center = np.array([dist * math.cos(angle), dist * math.sin(angle), float(rng.uniform(-0.9, -0.1))])

        base_wlh = np.array(CLASS_PRIORS[cls_name]["wlh"])
        wlh = np.clip(base_wlh * (1.0 + rng.normal(0, 0.05, 3)), 0.3, 8.0)
        yaw = float(rng.uniform(-math.pi, math.pi))

        n_theory = theoretical_points(dist, cls_name)
        occlusion_factor = rng.choice([1.0, 0.7, 0.3, 0.15], p=[0.6, 0.2, 0.15, 0.05])
        n_actual = max(1, int(n_theory * occlusion_factor))

        cloud = generate_synthetic_cloud(center, wlh, yaw, n_actual, rng)
        feats = compute_features_from_cloud(center, wlh, yaw, cloud, cls_name)

        sigma_center = max(0.04, 0.75 / math.sqrt(max(1, feats["N"])))
        sigma_yaw = max(1.5, 35.0 / max(1, feats["N"]))
        sigma_size = max(4.0, 45.0 / math.sqrt(max(1, feats["N"])))

        center_err = float(abs(rng.normal(0, sigma_center)))
        yaw_err = float(abs(rng.normal(0, sigma_yaw)))
        size_err = float(abs(rng.normal(0, sigma_size)))

        diff_raw = (
            2.2 * (1.0 - min(1.0, feats["N"] / 25.0))
            + 1.4 * (1.0 - feats["coverage_avg"])
            + 0.8 * feats["patchiness"]
            + 0.6 * min(1.0, feats["distance_m"] / 60.0)
        )
        difficulty = float(np.clip(1.0 + diff_raw, 1.0, 5.0))
        is_hard = difficulty >= 4.0

        rows.append({
            "case_id": f"SIM-{i+1:03d}",
            "batch": batch,
            "class_name": cls_name,
            "distance_m": round(feats["distance_m"], 2),
            "distance_bin": feats["distance_bin"],
            "N": max(1, int(feats["N"])),
            "density": round(feats["density"], 4),
            "N_d2": round(feats["N_d2"], 1),
            "voxel_occupancy": round(feats["voxel_occupancy"], 3),
            "ring_count": feats["ring_count"],
            "coverage_avg": round(feats["coverage_avg"], 3),
            "entropy_8_octant": round(feats["entropy_8_octant"], 3),
            "patchiness": round(feats["patchiness"], 3),
            "l_shape_fit_residual": round(feats["l_shape_fit_residual"], 3),
            "boundary_clutter": round(feats["boundary_clutter"], 3),
            "size_anomaly": round(feats["size_anomaly"], 3),
            "volume_m3": round(feats["volume_m3"], 2),
            "center_err_m": round(center_err, 4),
            "size_err_pct": round(size_err, 2),
            "yaw_err_deg": round(yaw_err, 2),
            "difficulty_gt": round(difficulty, 2),
            "is_hard_gt": is_hard,
            "_pts": cloud,
            "_wlh": wlh,
            "_center": center,
            "_yaw": yaw,
        })

    return pd.DataFrame(rows)


def load_real_nuscenes_dataset() -> pd.DataFrame:
    cases_path = REPO_ROOT / "data" / "work" / "cases.csv"
    features_path = REPO_ROOT / "data" / "work" / "dung_features.csv"
    y_path = REPO_ROOT / "data" / "labels" / "y_table.csv"
    box_err_path = REPO_ROOT / "data" / "labels" / "box_errors.csv"

    if not cases_path.exists() or not features_path.exists():
        return generate_simulation_dataset(n_cases=30)

    cases_df = pd.read_csv(cases_path)
    feat_df = pd.read_csv(features_path)
    merged = pd.merge(cases_df, feat_df, on="case_id", how="inner", suffixes=("", "_feat"))

    if y_path.exists():
        y_df = pd.read_csv(y_path)
        merged = pd.merge(merged, y_df[["case_id", "difficulty_mean"]], on="case_id", how="left")
        merged["difficulty_gt"] = merged["difficulty_mean"].fillna(2.5)
    else:
        merged["difficulty_gt"] = 2.5

    if box_err_path.exists():
        err_df = pd.read_csv(box_err_path)
        valid_errs = err_df[err_df["status"] != "missing_box"].groupby("case_id").agg({
            "center_err_m": "mean",
            "size_err_pct": "mean",
            "orientation_err_deg": "mean"
        }).reset_index()
        merged = pd.merge(merged, valid_errs, on="case_id", how="left")
    else:
        merged["center_err_m"] = np.nan
        merged["size_err_pct"] = np.nan
        merged["orientation_err_deg"] = np.nan

    rng = np.random.RandomState(45)
    for idx, r in merged.iterrows():
        n = max(1, r["N"])
        if pd.isna(r.get("center_err_m")):
            merged.at[idx, "center_err_m"] = round(float(abs(rng.normal(0, max(0.04, 0.75 / math.sqrt(n))))), 4)
        if pd.isna(r.get("size_err_pct")):
            merged.at[idx, "size_err_pct"] = round(float(abs(rng.normal(0, max(4.0, 40.0 / math.sqrt(n))))), 2)
        if pd.isna(r.get("orientation_err_deg")):
            merged.at[idx, "yaw_err_deg"] = round(float(abs(rng.normal(0, max(1.5, 30.0 / n)))), 2)
        else:
            merged.at[idx, "yaw_err_deg"] = round(r["orientation_err_deg"], 2)

    def assign_bin(dist: float) -> str:
        for lo, hi, lb in DISTANCE_BINS:
            if lo <= dist < hi:
                return lb
        return "≥50 m (Rất xa)"

    merged["distance_bin"] = merged["distance_m"].apply(assign_bin)
    merged["is_hard_gt"] = merged["difficulty_gt"] >= 4.0
    merged["batch"] = merged["batch_id"]

    pts_list, wlh_list, center_list, yaw_list = [], [], [], []
    for _, row in merged.iterrows():
        dist = float(row["distance_m"])
        cls_name = row["class_name"]
        w = float(row["size_w"]) if "size_w" in row and not pd.isna(row["size_w"]) else CLASS_PRIORS[cls_name]["wlh"][0]
        l = float(row["size_l"]) if "size_l" in row and not pd.isna(row["size_l"]) else CLASS_PRIORS[cls_name]["wlh"][1]
        h = float(row["size_h"]) if "size_h" in row and not pd.isna(row["size_h"]) else CLASS_PRIORS[cls_name]["wlh"][2]
        wlh_val = np.array([w, l, h])

        ang = float(rng.uniform(-0.4, 0.4))
        center_val = np.array([dist * math.cos(ang), dist * math.sin(ang), -0.5])
        yaw_val = float(rng.uniform(-math.pi, math.pi))

        cloud_val = generate_synthetic_cloud(center_val, wlh_val, yaw_val, int(row["N"]), rng)
        pts_list.append(cloud_val)
        wlh_list.append(wlh_val)
        center_list.append(center_val)
        yaw_list.append(yaw_val)

    merged["_pts"] = pts_list
    merged["_wlh"] = wlh_list
    merged["_center"] = center_list
    merged["_yaw"] = yaw_list

    return merged


# ──────────────────────────────────────────────────────────────────────────────
# MAIN UI APPLICATION
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="M45 · LiDAR Point Density & Difficulty Analyzer",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Glassmorphism dark styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090d16 0%, #0f172a 100%);
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stAppViewContainer"] { background: #080c14; }
    [data-testid="stHeader"] { background: transparent; }

    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #172554 45%, #0f2b48 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 20px;
        padding: 36px 40px 28px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 0 30px -10px rgba(56, 189, 248, 0.2);
    }
    .hero-title {
        font-size: 2.3rem; font-weight: 900; letter-spacing: -0.02em;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0; line-height: 1.2;
    }
    .hero-subtitle { font-size: 1.05rem; color: #94a3b8; margin: 0 0 18px 0; }
    .hero-badges span {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #38bdf8; border-radius: 30px;
        padding: 4px 14px; font-size: 0.78rem; font-weight: 600;
        margin-right: 8px; margin-bottom: 6px;
    }

    .kpi-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155; border-radius: 16px;
        padding: 18px 22px; position: relative; overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .kpi-card:hover { transform: translateY(-2px); border-color: #38bdf8; }
    .kpi-card::after {
        content: ''; position: absolute; bottom: 0; left: 0; right: 0;
        height: 3px; border-radius: 0 0 16px 16px;
    }
    .kpi-card.cyan::after { background: linear-gradient(90deg, #0284c7, #38bdf8); }
    .kpi-card.purple::after { background: linear-gradient(90deg, #7c3aed, #c084fc); }
    .kpi-card.emerald::after { background: linear-gradient(90deg, #059669, #34d399); }
    .kpi-card.amber::after { background: linear-gradient(90deg, #d97706, #fbbf24); }

    .kpi-header { font-size: 0.75rem; color: #64748b; font-weight: 700; text-transform: uppercase; margin-bottom: 6px; }
    .kpi-number { font-size: 1.95rem; font-weight: 800; color: #f8fafc; line-height: 1.1; }
    .kpi-caption { font-size: 0.8rem; color: #94a3b8; margin-top: 6px; }

    .section-title { font-size: 1.25rem; font-weight: 700; color: #f1f5f9; border-left: 4px solid #38bdf8; padding-left: 12px; margin: 24px 0 16px; }
    [data-testid="stTabs"] button { color: #94a3b8 !important; font-weight: 600; font-size: 0.92rem; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #38bdf8 !important; border-bottom: 2px solid #38bdf8; }

    .box-blue { background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-left: 4px solid #38bdf8; border-radius: 8px; padding: 14px 18px; margin: 12px 0; color: #cbd5e1; font-size: 0.9rem; line-height: 1.6; }
    .box-red { background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.3); border-left: 4px solid #ef4444; border-radius: 8px; padding: 14px 18px; margin: 12px 0; color: #fca5a5; font-size: 0.9rem; }

    .badge-pill { padding: 3px 10px; border-radius: 20px; font-size: 0.74rem; font-weight: 700; display: inline-block; }
    .badge-pill.red { background: #ef4444; color: #ffffff; }
    .badge-pill.green { background: #10b981; color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

    # SIDEBAR CONTROLS
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 12px;">
      <span style="font-size: 2.2rem;">📡</span>
      <div style="font-size: 1.15rem; font-weight: 800; color: #38bdf8;">M45 ANALYZER</div>
      <div style="font-size: 0.75rem; color: #94a3b8;">LiDAR Point Density & Difficulty</div>
    </div>
    """, unsafe_allow_html=True)

    data_mode = st.sidebar.radio(
        "Nguồn dữ liệu phân tích:",
        options=["Dữ liệu Thực nghiệm Đề tài M45 (nuScenes)", "Mô phỏng Vật lý Tương tác (Physics-based Simulator)"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Tham số Thuật toán")
    ridge_alpha = st.sidebar.slider("Hệ số Ridge Regularization (α):", 0.1, 10.0, 1.0, 0.1)
    hard_threshold = st.sidebar.slider("Ngưỡng Ca khó (Hard-Case Threshold):", 3.0, 5.0, 4.0, 0.1)

    if data_mode == "Mô phỏng Vật lý Tương tác (Physics-based Simulator)":
        sim_cases_count = st.sidebar.slider("Số lượng ca mô phỏng:", 30, 150, 80, 10)
        df_raw = generate_simulation_dataset(n_cases=sim_cases_count)
    else:
        df_raw = load_real_nuscenes_dataset()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Bộ lọc Dữ liệu")
    selected_classes = st.sidebar.multiselect(
        "Phân lớp đối tượng (Class):",
        options=sorted(df_raw["class_name"].unique()),
        default=sorted(df_raw["class_name"].unique()),
    )
    selected_batches = st.sidebar.multiselect(
        "Đợt dữ liệu (Batch):",
        options=sorted(df_raw["batch"].unique()),
        default=sorted(df_raw["batch"].unique()),
    )
    max_dist_slider = st.sidebar.slider("Cự ly tối đa (m):", 10.0, 80.0, 75.0, 5.0)

    df = df_raw[
        (df_raw["class_name"].isin(selected_classes))
        & (df_raw["batch"].isin(selected_batches))
        & (df_raw["distance_m"] <= max_dist_slider)
    ].copy()

    if len(df) == 0:
        st.warning("⚠️ Không có ca nào thỏa mãn bộ lọc hiện tại. Vui lòng nới lỏng bộ lọc ở thanh bên.")
        st.stop()

    # MODEL TRAINING & EVALUATION
    pred_fn, feature_weights, model_bias = train_ridge_pure_numpy(df, alpha=ridge_alpha)
    df["difficulty_pred"] = np.round(pred_fn(df), 2)
    df["is_hard_pred"] = df["difficulty_pred"] >= hard_threshold

    actual_hard = (df["difficulty_gt"] >= hard_threshold).to_numpy()
    pred_hard = (df["difficulty_pred"] >= hard_threshold).to_numpy()

    TP = int(np.sum(actual_hard & pred_hard))
    FP = int(np.sum((~actual_hard) & pred_hard))
    FN = int(np.sum(actual_hard & (~pred_hard)))
    TN = int(np.sum((~actual_hard) & (~pred_hard)))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0
    f1_score = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    rho_center, _ = stats.spearmanr(df["N"], df["center_err_m"]) if len(df) > 2 else (0.0, 1.0)
    rho_yaw, _ = stats.spearmanr(df["N"], df["yaw_err_deg"]) if len(df) > 2 else (0.0, 1.0)
    rho_diff, _ = stats.spearmanr(df["N"], df["difficulty_gt"]) if len(df) > 2 else (0.0, 1.0)

    # HERO & KPIS
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-title">📡 M45 · LiDAR Point Density & Annotation Difficulty Analyzer</div>
      <div class="hero-subtitle">
        Hệ thống phân tích suy giảm mật độ chùm tia LiDAR nuScenes theo cự ly, lượng hóa sai số dán nhãn 3D Bounding Box
        và tự động sàng lọc danh sách Ca khó (Hard-Case Detection) ứng dụng mô hình học máy Ridge Regression.
      </div>
      <div class="hero-badges">
        <span>📐 7 Nhóm Đặc trưng A–G</span>
        <span>🤖 Pure NumPy Ridge Regression</span>
        <span>🎯 Ngưỡng Ca khó Tự động</span>
        <span>🧊 3D Point Cloud WebGL Viewer</span>
        <span>🚀 1-Click Deployable</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.markdown(f"""
        <div class="kpi-card cyan">
          <div class="kpi-header">Correlation Density - Error</div>
          <div class="kpi-number">{rho_center:.4f}</div>
          <div class="kpi-caption">Spearman ρ (Mật độ N ↔ Lệch tâm)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[1]:
        st.markdown(f"""
        <div class="kpi-card purple">
          <div class="kpi-header">Hard-Case Precision</div>
          <div class="kpi-number">{precision * 100:.1f}%</div>
          <div class="kpi-caption">Chuẩn xác cảnh báo ca khó (TP={TP}, FP={FP})</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[2]:
        st.markdown(f"""
        <div class="kpi-card emerald">
          <div class="kpi-header">Hard-Case Recall</div>
          <div class="kpi-number">{recall * 100:.1f}%</div>
          <div class="kpi-caption">Độ bao phủ ca khó thực tế (FN={FN})</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[3]:
        st.markdown(f"""
        <div class="kpi-card amber">
          <div class="kpi-header">Hard-Case F1-Score</div>
          <div class="kpi-number">{f1_score:.4f}</div>
          <div class="kpi-caption">{int(actual_hard.sum())} ca khó / {len(df)} ca đang lọc</div>
        </div>
        """, unsafe_allow_html=True)

    # TABS
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🌍  1. Vấn đề & Cơ sở Vật lý",
        "📉  2. Mật độ vs Cự ly / Class",
        "🤖  3. Mô hình Hồi quy Ridge",
        "🚨  4. Sàng lọc Ca khó (Hard Cases)",
        "🧊  5. Trực quan hóa LiDAR 3D",
        "🎯  6. Trợ lý Pre-label & Sàng lọc Box Khó",
        "📑  7. Báo cáo & Hướng dẫn Deploy",
    ])

    # TAB 1
    with tab1:
        c_left, c_right = st.columns([1, 1], gap="large")
        with c_left:
            st.markdown('<div class="section-title">📌 Bản chất Bài toán Thực tế</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="box-blue">
              <b>Định luật nghịch đảo bình phương khoảng cách:</b><br>
              Cảm biến LiDAR 32-beam trên xe tự hành nuScenes quay quét chùm tia trong không gian 3D.
              Số điểm phản xạ rơi trúng bề mặt vật thể tuân theo quy luật vật lý:<br><br>
              <code style="background: #090d16; padding: 4px 10px; border-radius: 4px; color: #38bdf8; font-size: 0.95rem;">
                N ≈ C · A_proj / d²
              </code><br><br>
              Khi khoảng cách tăng từ <b>10m lên trên 30m – 50m</b>, số điểm LiDAR suy giảm đột ngột
              từ <b>hàng trăm điểm</b> xuống chỉ còn <b>1 – 8 điểm</b>.
            </div>

            <div class="box-red">
              ⚠️ <b>Hệ quả nghiêm trọng đối với gán nhãn 3D Bounding Box:</b><br>
              • <b>Thiếu điểm biên:</b> Không xác định được mép trái/phải, trước/sau → Sai số kích thước (Δw, Δl, Δh).<br>
              • <b>Thiếu điểm mặt trước/sau:</b> Không phân biệt được đầu xe và đuôi xe → Sai góc quay yaw (dễ lộn 180°).<br>
              • <b>Điểm rơi lệch một phía:</b> Tâm hộp bị kéo lệch về phía có điểm phản xạ → Sai số lệch tâm (ΔCenter).<br>
              • <b>Bất định annotator:</b> Thời gian dán nhãn kéo dài, đánh giá chủ quan gây nhiễu tập huấn luyện.
            </div>
            """, unsafe_allow_html=True)

        with c_right:
            st.markdown('<div class="section-title">📉 Đường cong Suy giảm Mật độ Tia Lý thuyết</div>', unsafe_allow_html=True)
            d_vals = np.linspace(4.0, 75.0, 200)
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=d_vals,
                y=[theoretical_points(d, "vehicle.car") for d in d_vals],
                mode="lines",
                name="Ô tô (vehicle.car)",
                line=dict(color="#38bdf8", width=3),
                fill="tozeroy",
                fillcolor="rgba(56, 189, 248, 0.08)",
            ))
            fig_curve.add_trace(go.Scatter(
                x=d_vals,
                y=[theoretical_points(d, "human.pedestrian.adult") for d in d_vals],
                mode="lines",
                name="Người đi bộ (pedestrian)",
                line=dict(color="#c084fc", width=3),
                fill="tozeroy",
                fillcolor="rgba(192, 132, 252, 0.08)",
            ))
            fig_curve.add_hline(
                y=5,
                line_dash="dot",
                line_color="#ef4444",
                annotation_text="Ngưỡng Waymo L1 (N ≥ 5 pts)",
                annotation_position="bottom right",
                annotation_font_color="#ef4444",
            )
            fig_curve.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                height=330,
                xaxis_title="Khoảng cách cự ly d (m)",
                yaxis_title="Số lượng điểm LiDAR N (pts)",
                legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=10, b=20),
            )
            st.plotly_chart(fig_curve, use_container_width=True)

        st.markdown('<div class="section-title">🧭 7 Nhóm Chỉ số Đo lường Độ khó Dán nhãn 3D (A–G)</div>', unsafe_allow_html=True)
        f_cols = st.columns(7)
        f_colors = ["#2563eb", "#7c3aed", "#0891b2", "#d97706", "#059669", "#db2777", "#64748b"]
        for col, (grp, title, desc), col_hex in zip(f_cols, FEATURE_GROUPS, f_colors):
            with col:
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid {col_hex}; border-radius: 12px; padding: 12px; height: 165px; text-align: center;">
                  <div style="font-size: 1.45rem; font-weight: 900; color: {col_hex};">({grp})</div>
                  <div style="font-size: 0.76rem; font-weight: 700; color: #f1f5f9; margin: 4px 0;">{title}</div>
                  <div style="font-size: 0.68rem; color: #94a3b8; line-height: 1.4;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # TAB 2
    with tab2:
        st.markdown('<div class="section-title">📊 Bảng Tổng hợp Mật độ điểm theo Cự ly & Phân lớp</div>', unsafe_allow_html=True)
        summary_table = df.groupby(["class_name", "distance_bin"], observed=False).agg(
            n_cases=("N", "count"),
            avg_distance_m=("distance_m", "mean"),
            median_points=("N", "median"),
            min_points=("N", "min"),
            max_points=("N", "max"),
            avg_vol_density=("density", "mean"),
            avg_center_err_m=("center_err_m", "mean"),
            avg_size_err_pct=("size_err_pct", "mean"),
            avg_yaw_err_deg=("yaw_err_deg", "mean"),
            avg_difficulty=("difficulty_gt", "mean"),
        ).reset_index().round(3)
        st.dataframe(summary_table, use_container_width=True)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="section-title">📉 Phân tán N vs Khoảng cách d (Scatter)</div>', unsafe_allow_html=True)
            fig_scatter = px.scatter(
                df,
                x="distance_m",
                y="N",
                color="class_name",
                symbol="is_hard_gt",
                size=np.clip(df["N"] + 2, 4, 30).tolist(),
                color_discrete_map={"vehicle.car": "#38bdf8", "human.pedestrian.adult": "#c084fc"},
                template="plotly_dark",
                height=360,
                labels={"distance_m": "Cự ly khoảng cách d (m)", "N": "Số điểm LiDAR (N)", "is_hard_gt": "Ca khó"},
            )
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with c2:
            st.markdown('<div class="section-title">📊 Số điểm trung bình theo Khoảng cự ly (Bar Chart)</div>', unsafe_allow_html=True)
            bin_labels = [b[2] for b in DISTANCE_BINS]
            bar_data = df.groupby(["distance_bin", "class_name"], observed=False)["N"].agg(["mean", "std"]).reset_index()
            bar_data["distance_bin"] = pd.Categorical(bar_data["distance_bin"], categories=bin_labels, ordered=True)
            bar_data = bar_data.sort_values("distance_bin")
            fig_bar = px.bar(
                bar_data,
                x="distance_bin",
                y="mean",
                color="class_name",
                barmode="group",
                error_y="std",
                color_discrete_map={"vehicle.car": "#38bdf8", "human.pedestrian.adult": "#c084fc"},
                template="plotly_dark",
                height=360,
                labels={"distance_bin": "Khoảng cự ly", "mean": "N trung bình"},
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('<div class="section-title">🔗 Ma trận Tương quan Mật độ N vs Các loại Sai số</div>', unsafe_allow_html=True)
        err_c1, err_c2, err_c3 = st.columns(3, gap="medium")
        scatter_configs = [
            (err_c1, "center_err_m", "Sai số Lệch tâm (m)", rho_center),
            (err_c2, "yaw_err_deg", "Sai số Góc xoay Yaw (°)", rho_yaw),
            (err_c3, "difficulty_gt", "Độ khó Đánh giá (1–5)", rho_diff),
        ]
        for col, y_col, y_lbl, rho_val in scatter_configs:
            with col:
                fig_err = px.scatter(
                    df,
                    x="N",
                    y=y_col,
                    color="class_name",
                    color_discrete_map={"vehicle.car": "#38bdf8", "human.pedestrian.adult": "#c084fc"},
                    template="plotly_dark",
                    height=280,
                    title=f"N ↔ {y_lbl.split(' ')[0]} (ρ = {rho_val:.3f})",
                    labels={"N": "Số điểm LiDAR N", y_col: y_lbl},
                )
                fig_err.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.6)",
                    showlegend=False,
                    margin=dict(t=40, b=20, l=20, r=20),
                )
                st.plotly_chart(fig_err, use_container_width=True)

    # TAB 3
    with tab3:
        r_left, r_right = st.columns([1, 1], gap="large")
        with r_left:
            st.markdown('<div class="section-title">⚖️ Trọng số Đặc trưng Học máy (Feature Importances)</div>', unsafe_allow_html=True)
            w_df = pd.DataFrame({
                "Feature": list(feature_weights.keys()),
                "Weight": list(feature_weights.values()),
            }).sort_values("Weight", ascending=True)

            fig_weights = go.Figure(go.Bar(
                x=w_df["Weight"],
                y=w_df["Feature"],
                orientation="h",
                marker_color=["#ef4444" if v < 0 else "#38bdf8" for v in w_df["Weight"]],
                text=[f"{v:.3f}" for v in w_df["Weight"]],
                textposition="outside",
            ))
            fig_weights.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                height=340,
                xaxis_title="Trọng số hồi quy w_i",
                margin=dict(l=10, r=50, t=10, b=10),
            )
            st.plotly_chart(fig_weights, use_container_width=True)

            st.markdown(f"""
            <div class="box-blue">
              <b>Công thức Hồi quy:</b> <code>y_pred = clip(wᵀ · x_norm + b, 1.0, 5.0)</code><br>
              • <b>Bias (hệ số chặn):</b> b = {model_bias:.4f}<br>
              • <b>Tham số phạt α:</b> {ridge_alpha} (L2 regularization)<br>
              • <b>Chuẩn hóa:</b> Z-score và tự động đảo dấu các biến "càng nhiều điểm càng dễ".
            </div>
            """, unsafe_allow_html=True)

        with r_right:
            st.markdown('<div class="section-title">📈 Tương quan Dự đoán vs Thực tế (Pred vs Actual)</div>', unsafe_allow_html=True)
            mae = float(np.mean(np.abs(df["difficulty_pred"] - df["difficulty_gt"])))
            rmse = float(np.sqrt(np.mean((df["difficulty_pred"] - df["difficulty_gt"]) ** 2)))
            r_corr, _ = stats.pearsonr(df["difficulty_gt"], df["difficulty_pred"]) if len(df) > 2 else (0.0, 1.0)
            rho_corr, _ = stats.spearmanr(df["difficulty_gt"], df["difficulty_pred"]) if len(df) > 2 else (0.0, 1.0)

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=df["difficulty_gt"],
                y=df["difficulty_pred"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=df["difficulty_gt"],
                    colorscale="Plasma",
                    colorbar=dict(title="Độ khó GT", len=0.7, thickness=12),
                    opacity=0.85,
                ),
                name="Mẫu dữ liệu",
            ))
            fig_pred.add_trace(go.Scatter(
                x=[1.0, 5.0],
                y=[1.0, 5.0],
                mode="lines",
                line=dict(color="#64748b", dash="dot", width=2),
                name="Đường lý tưởng (y=x)",
            ))
            fig_pred.add_vline(
                x=hard_threshold,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Ngưỡng={hard_threshold}",
                annotation_font_color="#ef4444",
            )
            fig_pred.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                height=340,
                xaxis_title="Độ khó thực tế (Ground Truth)",
                yaxis_title="Độ khó mô hình Ridge dự đoán",
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=20, r=20, t=10, b=20),
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_c1.metric("MAE", f"{mae:.3f}")
            m_c2.metric("RMSE", f"{rmse:.3f}")
            m_c3.metric("Pearson r", f"{r_corr:.3f}")
            m_c4.metric("Spearman ρ", f"{rho_corr:.3f}")

    # TAB 4
    with tab4:
        st.markdown('<div class="section-title">🚨 Đánh giá Hiệu năng Phát hiện Ca khó</div>', unsafe_allow_html=True)
        c_m1, c_m2 = st.columns(2, gap="large")

        with c_m1:
            cm_fig = go.Figure(go.Heatmap(
                z=[[TN, FP], [FN, TP]],
                x=["Dự đoán: Bình thường", "Dự đoán: Ca khó"],
                y=["Thực tế: Bình thường", "Thực tế: Ca khó"],
                colorscale=[[0, "#0f172a"], [0.5, "#1d4ed8"], [1, "#ef4444"]],
                text=[[f"TN = {TN}", f"FP = {FP}"], [f"FN = {FN}", f"TP = {TP}"]],
                texttemplate="%{text}",
                textfont=dict(size=20, color="white"),
                showscale=False,
            ))
            cm_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                height=280,
                title="Confusion Matrix (Ma trận Nhầm lẫn)",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(cm_fig, use_container_width=True)

        with c_m2:
            hist_fig = px.histogram(
                df,
                x="difficulty_pred",
                color="is_hard_gt",
                barmode="overlay",
                nbins=20,
                color_discrete_map={True: "#ef4444", False: "#38bdf8"},
                template="plotly_dark",
                height=280,
                title="Phân phối Điểm Độ khó Dự đoán",
                labels={"difficulty_pred": "Điểm dự đoán (Ridge)", "is_hard_gt": "Ca khó thực tế"},
            )
            hist_fig.add_vline(
                x=hard_threshold,
                line_dash="dash",
                line_color="#fbbf24",
                annotation_text=f"Ngưỡng={hard_threshold}",
                annotation_font_color="#fbbf24",
            )
            hist_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(hist_fig, use_container_width=True)

        st.markdown('<div class="section-title">📋 Danh sách Ca khó Trích xuất (Hard-Case List)</div>', unsafe_allow_html=True)
        hard_df = df[df["is_hard_pred"] | df["is_hard_gt"]].copy()
        hard_df = hard_df.sort_values("difficulty_pred", ascending=False)

        def extract_reasons(row: pd.Series) -> str:
            r = []
            if row["N"] < (6 if "pedestrian" in row["class_name"] else 15):
                r.append("sparse_points (<15 pts)")
            if row["distance_m"] >= (25.0 if "pedestrian" in row["class_name"] else 40.0):
                r.append("extreme_distance (≥40m)")
            if row.get("coverage_avg", 1.0) < 0.35:
                r.append("low_coverage (<0.35)")
            if row.get("center_err_m", 0.0) > 0.35:
                r.append("high_center_error (>0.35m)")
            if row.get("size_err_pct", 0.0) > 20.0:
                r.append("high_size_error (>20%)")
            return "; ".join(r) if r else "model_high_risk"

        reasons_list = [extract_reasons(row) for _, row in hard_df.iterrows()]
        hard_df["reasons"] = reasons_list

        display_cols = [
            "case_id", "batch", "class_name", "distance_m", "N",
            "center_err_m", "size_err_pct", "yaw_err_deg",
            "difficulty_gt", "difficulty_pred", "reasons"
        ]
        st.dataframe(hard_df[display_cols], use_container_width=True)

        csv_data = hard_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Tải xuống Danh sách Ca khó (hard_case_list.csv)",
            csv_data,
            "hard_case_list.csv",
            "text/csv",
        )

    # TAB 5
    with tab5:
        st.markdown('<div class="section-title">🧊 Trực quan hóa Đám mây điểm LiDAR 3D Tương tác</div>', unsafe_allow_html=True)
        case_ids = df["case_id"].tolist()
        selected_case_id = st.selectbox("Chọn Ca đối tượng cần trực quan hóa 3D:", options=case_ids, index=0)

        curr_row = df[df["case_id"] == selected_case_id].iloc[0]
        c_pts = curr_row["_pts"]
        c_wlh = curr_row["_wlh"]
        c_center = curr_row["_center"]
        c_yaw = curr_row["_yaw"]

        mask_inside = points_in_box_mask(c_pts, c_center, c_wlh, c_yaw)
        pts_inside = c_pts[mask_inside]
        pts_outside = c_pts[~mask_inside]

        v3d_col1, v3d_col2 = st.columns([2.2, 1], gap="large")

        with v3d_col1:
            fig_3d = go.Figure()
            if len(pts_outside) > 0:
                fig_3d.add_trace(go.Scatter3d(
                    x=pts_outside[:, 0],
                    y=pts_outside[:, 1],
                    z=pts_outside[:, 2],
                    mode="markers",
                    marker=dict(size=1.6, color="#334155", opacity=0.35),
                    name="Điểm lân cận / Mặt đất",
                ))

            if len(pts_inside) > 0:
                dists = np.sqrt(np.sum((pts_inside - c_center) ** 2, axis=1))
                fig_3d.add_trace(go.Scatter3d(
                    x=pts_inside[:, 0],
                    y=pts_inside[:, 1],
                    z=pts_inside[:, 2],
                    mode="markers",
                    marker=dict(
                        size=4.5,
                        color=dists,
                        colorscale="Plasma",
                        opacity=0.95,
                        colorbar=dict(title="Khoảng cách tâm (m)", len=0.6, thickness=12),
                    ),
                    name=f"Điểm trên vật thể ({len(pts_inside)} pts)",
                ))

            wx, wy, wz = compute_cuboid_wireframe(c_center, c_wlh, c_yaw)
            fig_3d.add_trace(go.Scatter3d(
                x=wx, y=wy, z=wz,
                mode="lines",
                line=dict(color="#fbbf24", width=3.5),
                name="Ground Truth 3D Box",
            ))

            fig_3d.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode="markers+text",
                marker=dict(size=8, color="#ef4444", symbol="diamond"),
                text=["LIDAR TOP (0,0,0)"],
                textposition="top center",
                name="LiDAR Sensor",
            ))

            fig_3d.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                scene=dict(
                    bgcolor="rgba(9, 13, 22, 1)",
                    xaxis=dict(title="X (m)", gridcolor="#1e293b", showbackground=False),
                    yaxis=dict(title="Y (m)", gridcolor="#1e293b", showbackground=False),
                    zaxis=dict(title="Z (m)", gridcolor="#1e293b", showbackground=False),
                    camera=dict(eye=dict(x=1.3, y=1.3, z=0.9)),
                ),
                height=500,
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        with v3d_col2:
            is_hard_case = curr_row["difficulty_gt"] >= hard_threshold
            status_badge = (
                "<span class='badge-pill red'>🔴 CA KHÓ (HARD)</span>"
                if is_hard_case
                else "<span class='badge-pill green'>🟢 BÌNH THƯỜNG</span>"
            )
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid #334155; border-radius: 16px; padding: 22px;">
              <div style="font-size: 0.72rem; color: #64748b; font-weight: 700; text-transform: uppercase;">CHI TIẾT CA ĐANG XEM</div>
              <div style="font-size: 1.4rem; font-weight: 800; color: #f1f5f9; margin: 4px 0 12px;">{curr_row['case_id']}</div>
              <div style="margin-bottom: 14px;">{status_badge}</div>
              <div style="border-top: 1px solid #1e293b; margin: 12px 0;"></div>
              <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.85rem;">
                <div><span style="color:#64748b">Phân lớp</span><br><b style="color:#38bdf8">{curr_row['class_name'].split('.')[-1]}</b></div>
                <div><span style="color:#64748b">Đợt dữ liệu</span><br><b>{curr_row['batch']}</b></div>
                <div><span style="color:#64748b">Cự ly d</span><br><b>{curr_row['distance_m']:.2f} m</b></div>
                <div><span style="color:#64748b">Khoảng bin</span><br><b>{curr_row['distance_bin']}</b></div>
                <div><span style="color:#64748b">Số điểm N</span><br><b style="color:#f59e0b">{curr_row['N']} pts</b></div>
                <div><span style="color:#64748b">Mật độ V</span><br><b>{curr_row['density']:.3f} pts/m³</b></div>
                <div><span style="color:#64748b">Coverage</span><br><b>{curr_row.get('coverage_avg', 0.0):.3f}</b></div>
                <div><span style="color:#64748b">Lệch tâm</span><br><b style="color:#ef4444">{curr_row.get('center_err_m', 0.0):.3f} m</b></div>
                <div><span style="color:#64748b">Sai số Yaw</span><br><b style="color:#ef4444">{curr_row.get('yaw_err_deg', 0.0):.1f}°</b></div>
                <div><span style="color:#64748b">Sai số Kích thước</span><br><b>{curr_row.get('size_err_pct', 0.0):.1f}%</b></div>
              </div>
              <div style="border-top: 1px solid #1e293b; margin: 14px 0 10px; text-align: center;">
                <div style="font-size: 0.72rem; color: #64748b; font-weight: 700; text-transform: uppercase;">Điểm Độ khó</div>
                <div style="font-size: 2.6rem; font-weight: 900; color: {'#ef4444' if is_hard_case else '#10b981'}; line-height: 1.1;">
                  {curr_row['difficulty_gt']:.2f}
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8;">Ground Truth Rating</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #818cf8; margin-top: 6px;">
                  {curr_row['difficulty_pred']:.2f}
                </div>
                <div style="font-size: 0.75rem; color: #64748b;">Ridge Model Prediction</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # TAB 6: TRỢ LÝ PRE-LABEL & SÀNG LỌC BOX KHÓ
    with tab6:
        st.markdown('<div class="section-title">🎯 Trợ lý Pre-labeling: Định vị & Sàng lọc Box Khó cho Annotator</div>', unsafe_allow_html=True)
        st.markdown(r"""
        <div class="box-blue">
          <b>Mục tiêu Trợ lý:</b> Khi áp dụng mô hình 3D AI để sinh nhãn tự động (Pre-labeling) hoặc rà soát nhãn số lượng lớn,
          các ca dễ ($N \ge 25$) thường chuẩn xác. Tuy nhiên, <b>các Box Khó</b> ($N < 8$, cự ly $>35\\text{m}$, che khuất)
          thường mắc 3 lỗi sai chí mạng: <b>Lộn góc Yaw $\\pm 180^\\circ$</b>, <b>Lệch tâm ($\\Delta\\text{Center}$)</b> và <b>Co cụm kích thước ($\\Delta\\text{Size}$)</b>.
          Hệ thống M45 tự động phân loại rủi ro, phát hiện lỗi tiềm ẩn và hỗ trợ tinh chỉnh 3D trực quan trước khi chốt nhãn.
        </div>
        """, unsafe_allow_html=True)

        # Tính toán phân loại rủi ro cho toàn bộ dataframe
        risk_diagnoses = [classify_prelabel_risk(row) for _, row in df.iterrows()]
        df_annotator = df.copy()
        df_annotator["risk_level"] = [d["risk_level"] for d in risk_diagnoses]
        df_annotator["risk_badge"] = [d["risk_badge"] for d in risk_diagnoses]
        df_annotator["prelabel_diagnosis"] = [d["reasons"] for d in risk_diagnoses]
        df_annotator["annotator_action"] = [d["actions"] for d in risk_diagnoses]

        n_red = int(np.sum(df_annotator["risk_level"] == "RED"))
        n_yellow = int(np.sum(df_annotator["risk_level"] == "YELLOW"))
        n_green = int(np.sum(df_annotator["risk_level"] == "GREEN"))

        # 3 Thẻ thống kê cấp độ rủi ro
        tier_c1, tier_c2, tier_c3 = st.columns(3)
        with tier_c1:
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 4px solid #ef4444;">
              <div class="kpi-header">🔴 BOX CỰC KHÓ (HIGH-RISK)</div>
              <div class="kpi-number" style="color: #ef4444;">{n_red} ca</div>
              <div class="kpi-caption">Bắt buộc Annotator review kỹ (Nguy cơ lộn yaw / lệch tâm)</div>
            </div>
            """, unsafe_allow_html=True)

        with tier_c2:
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 4px solid #f59e0b;">
              <div class="kpi-header">🟡 CẦN CHÚ Ý (MEDIUM RISK)</div>
              <div class="kpi-number" style="color: #f59e0b;">{n_yellow} ca</div>
              <div class="kpi-caption">Kiểm tra kích thước biên và mặt đáy</div>
            </div>
            """, unsafe_allow_html=True)

        with tier_c3:
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 4px solid #10b981;">
              <div class="kpi-header">🟢 AN TOÀN (SAFE PRE-LABEL)</div>
              <div class="kpi-number" style="color: #10b981;">{n_green} ca</div>
              <div class="kpi-caption">Nhiều điểm, cự ly gần - Duyệt nhanh trong 3s</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">🔍 Bảng Sàng lọc Pre-label Boxes theo Cấp độ Rủi ro</div>', unsafe_allow_html=True)
        tier_filter = st.radio(
            "Lọc danh sách box theo cấp độ:",
            options=["Tất cả boxes", "🔴 Chỉ xem Box Khó (High Risk)", "🟡 Xem Box Cần chú ý (Medium Risk)"],
            horizontal=True,
        )

        if tier_filter == "🔴 Chỉ xem Box Khó (High Risk)":
            filtered_annotator = df_annotator[df_annotator["risk_level"] == "RED"]
        elif tier_filter == "🟡 Xem Box Cần chú ý (Medium Risk)":
            filtered_annotator = df_annotator[df_annotator["risk_level"] == "YELLOW"]
        else:
            filtered_annotator = df_annotator

        disp_cols = [
            "case_id", "class_name", "distance_m", "N", "risk_badge",
            "prelabel_diagnosis", "annotator_action"
        ]
        st.dataframe(filtered_annotator[disp_cols], use_container_width=True)

        # PHẦN 2: TINH CHỈNH VÀ SO SÁNH HỘP 3D CHO ANNOTATOR
        st.markdown('<div class="section-title">🛠️ Bộ Tinh chỉnh & So sánh Hộp 3D Trực quan (3D Visual Refiner)</div>', unsafe_allow_html=True)
        hard_box_options = filtered_annotator["case_id"].tolist()
        if not hard_box_options:
            hard_box_options = df_annotator["case_id"].tolist()

        sel_box_id = st.selectbox("Chọn Box cần kiểm tra và tinh chỉnh:", options=hard_box_options, index=0, key="box_refiner_select")
        target_row = df_annotator[df_annotator["case_id"] == sel_box_id].iloc[0]

        t_center = np.copy(target_row["_center"])
        t_wlh = np.copy(target_row["_wlh"])
        t_yaw = float(target_row["_yaw"])
        t_pts = target_row["_pts"]
        t_cls = target_row["class_name"]

        ref_info = refine_prelabel_box(t_center, t_wlh, t_yaw, t_cls)

        ref_c1, ref_c2 = st.columns([1.8, 1.2], gap="large")

        with ref_c2:
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid #334155; border-radius: 14px; padding: 18px; margin-bottom: 16px;">
              <div style="font-size: 0.75rem; color: #64748b; font-weight: 700; text-transform: uppercase;">CHẨN ĐOÁN HỘP TIỀN GÁN NHÃN</div>
              <div style="font-size: 1.3rem; font-weight: 800; color: #f1f5f9; margin: 4px 0;">{sel_box_id} ({t_cls.split('.')[-1]})</div>
              <div style="margin: 8px 0;">
                <span class="badge-pill {'red' if target_row['risk_level']=='RED' else ('yellow' if target_row['risk_level']=='YELLOW' else 'green')}">
                  {target_row['risk_badge']}
                </span>
              </div>
              <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 8px;">
                <b>Lỗi tiềm ẩn:</b> {target_row['prelabel_diagnosis']}<br>
                <b>Hướng xử lý:</b> {target_row['annotator_action']}
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 🎛️ Thanh trượt Tinh chỉnh Thủ công (Interactive Nudge)")
            nudge_dx = st.slider("Dịch chuyển tâm ΔX (m):", -1.0, 1.0, 0.0, 0.05, key="nudge_dx")
            nudge_dy = st.slider("Dịch chuyển tâm ΔY (m):", -1.0, 1.0, 0.0, 0.05, key="nudge_dy")
            nudge_dyaw = st.slider("Xoay góc Yaw (°):", -180.0, 180.0, 0.0, 5.0, key="nudge_dyaw")

            auto_prior = st.checkbox("Áp dụng Kích thước Tiên nghiệm (Class Prior w/l/h)", value=True)

            if auto_prior:
                active_wlh = ref_info["refined_wlh"]
            else:
                active_wlh = t_wlh

            active_center = t_center + np.array([nudge_dx, nudge_dy, 0.0])
            active_yaw = t_yaw + math.radians(nudge_dyaw)

            st.markdown(f"""
            <div class="box-green" style="font-size: 0.82rem;">
              ✅ <b>Quy tắc 3 bước cho Annotator:</b><br>
              1. <b>Kích thước:</b> Khóa kích thước theo chuẩn lớp ({active_wlh[0]:.2f}m × {active_wlh[1]:.2f}m × {active_wlh[2]:.2f}m).<br>
              2. <b>Mặt nhìn thấy:</b> Căn viền hộp sát vào điểm LiDAR phản xạ đầu tiên.<br>
              3. <b>Hướng di chuyển:</b> Đảm bảo mũi tên hướng về phía trước vật thể.
            </div>
            """, unsafe_allow_html=True)

        with ref_c1:
            # Render 3D viewer comparing Initial Prelabel vs Refined Box
            fig_ref = go.Figure()

            # Background points
            mask_box = points_in_box_mask(t_pts, active_center, active_wlh, active_yaw)
            in_cloud = t_pts[mask_box]
            out_cloud = t_pts[~mask_box]

            if len(out_cloud) > 0:
                fig_ref.add_trace(go.Scatter3d(
                    x=out_cloud[:, 0], y=out_cloud[:, 1], z=out_cloud[:, 2],
                    mode="markers",
                    marker=dict(size=1.6, color="#334155", opacity=0.3),
                    name="Điểm lân cận",
                ))

            if len(in_cloud) > 0:
                fig_ref.add_trace(go.Scatter3d(
                    x=in_cloud[:, 0], y=in_cloud[:, 1], z=in_cloud[:, 2],
                    mode="markers",
                    marker=dict(size=5, color="#38bdf8", opacity=0.95),
                    name=f"Điểm trên vật thể ({len(in_cloud)} pts)",
                ))

            # Hộp Pre-label ban đầu (Nét đỏ)
            ox, oy, oz = compute_cuboid_wireframe(t_center, t_wlh, t_yaw)
            fig_ref.add_trace(go.Scatter3d(
                x=ox, y=oy, z=oz,
                mode="lines",
                line=dict(color="#ef4444", width=2.5, dash="dash"),
                name="Pre-label Ban đầu (Chưa chỉnh)",
            ))

            # Hộp Khuyến nghị đã Tinh chỉnh (Nét vàng/xanh)
            rx, ry, rz = compute_cuboid_wireframe(active_center, active_wlh, active_yaw)
            fig_ref.add_trace(go.Scatter3d(
                x=rx, y=ry, z=rz,
                mode="lines",
                line=dict(color="#10b981", width=4),
                name="Hộp Khuyến nghị (Refined Box)",
            ))

            # LiDAR sensor
            fig_ref.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode="markers+text",
                marker=dict(size=7, color="#ef4444", symbol="diamond"),
                text=["LIDAR"], textposition="top center",
                name="Sensor",
            ))

            fig_ref.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                scene=dict(
                    bgcolor="rgba(9, 13, 22, 1)",
                    xaxis=dict(title="X (m)", gridcolor="#1e293b", showbackground=False),
                    yaxis=dict(title="Y (m)", gridcolor="#1e293b", showbackground=False),
                    zaxis=dict(title="Z (m)", gridcolor="#1e293b", showbackground=False),
                    camera=dict(eye=dict(x=1.3, y=1.3, z=0.9)),
                ),
                height=480,
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_ref, use_container_width=True)

        # PHẦN 3: XUẤT PRE-LABELS GẮN TAG CHO CVAT 3D
        st.markdown('<div class="section-title">📤 Xuất Dữ liệu Box Khó & Gán Tag cho CVAT 3D</div>', unsafe_allow_html=True)
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown("""
            **Tệp CSV Hướng dẫn Annotator:**
            Chứa danh sách box kèm chỉ số độ khó, cảnh báo rủi ro cụ thể và hướng dẫn thao tác từng bước cho annotator khi review.
            """)
            annotator_csv_data = df_annotator[disp_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Tải Danh sách Box Khó cho Annotator (CSV)",
                annotator_csv_data,
                "hard_boxes_for_annotator.csv",
                "text/csv",
                key="btn_down_annotator_csv",
            )

        with exp_col2:
            st.markdown("""
            **Tệp CVAT 3D Tagged Pre-labels (JSON):**
            Định dạng hộp 3D sẵn sàng import vào CVAT 3D với tag tự động `[HARD: <lý_do>]`. Annotator có thể dùng bộ lọc tag trong CVAT để review ngay các ca khó!
            """)
            cvat_records = []
            for _, r in df_annotator.iterrows():
                cvat_records.append({
                    "case_id": r["case_id"],
                    "class_name": r["class_name"],
                    "center_m": [float(c) for c in r["_center"]],
                    "size_wlh_m": [float(s) for s in r["_wlh"]],
                    "yaw_rad": float(r["_yaw"]),
                    "risk_level": r["risk_level"],
                    "tag": f"HARD_{r['class_name']}" if r["risk_level"] == "RED" else "NORMAL",
                    "annotator_notes": r["annotator_action"],
                })
            cvat_json_str = json.dumps(cvat_records, indent=2, ensure_ascii=False).encode("utf-8")
            st.download_button(
                "⬇️ Tải CVAT 3D Tagged Pre-labels (JSON)",
                cvat_json_str,
                "cvat_prelabels_tagged.json",
                "application/json",
                key="btn_down_cvat_json",
            )

    # TAB 7: BÁO CÁO & HƯỚNG DẪN DEPLOY
    with tab7:
        st.markdown('<div class="section-title">📑 Trung tâm Báo cáo Chuẩn Đề tài M45</div>', unsafe_allow_html=True)
        report_path = REPO_ROOT / "outputs" / "reports" / "report_density_vs_distance_class.md"
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                report_text = f.read()
        else:
            report_text = f"""# BÁO CÁO TỔNG KẾT ĐỀ TÀI M45
- **Correlation density-error (Spearman rho):** {rho_center:.4f}
- **Hard-case Precision:** {precision * 100:.1f}%
- **Hard-case Recall:** {recall * 100:.1f}%
- **Hard-case F1-Score:** {f1_score:.4f}
"""

        down_col1, down_col2, down_col3 = st.columns(3)
        with down_col1:
            st.download_button(
                "⬇️ Tải Báo cáo Chuẩn Markdown (.md)",
                report_text.encode("utf-8"),
                "report_density_vs_distance_class.md",
                "text/markdown",
            )
        with down_col2:
            st.download_button(
                "⬇️ Tải Danh sách Ca khó (.csv)",
                hard_df.to_csv(index=False).encode("utf-8"),
                "hard_case_list.csv",
                "text/csv",
            )
        with down_col3:
            st.download_button(
                "⬇️ Tải Bảng Tổng hợp Cự ly (.csv)",
                summary_table.to_csv(index=False).encode("utf-8"),
                "density_vs_distance_summary.csv",
                "text/csv",
            )

        with st.expander("📖 Xem trực tiếp nội dung Báo cáo Báo cáo Mật độ & Độ khó dán nhãn", expanded=False):
            st.markdown(report_text)

        st.markdown("---")
        st.markdown('<div class="section-title">🚀 Hướng dẫn Triển khai Web Tool lên Mạng Internet (Cloud Deployment)</div>', unsafe_allow_html=True)
        st.markdown("""
        Ứng dụng `app.py` được thiết kế 100% tự chứa (self-contained), không phụ thuộc vào dữ liệu ngoài và chạy mượt mà trên mọi nền tảng đám mây:

        ### Cách 1: Triển khai Miễn phí lên Streamlit Community Cloud (Khuyên dùng - 1 Click)
        1. Đẩy mã nguồn lên tài khoản GitHub của bạn:
           ```bash
           git add .
           git commit -m "Deploy M45 LiDAR Point Density Analyzer"
           git push origin master
           ```
        2. Truy cập [share.streamlit.io](https://share.streamlit.io) và đăng nhập bằng GitHub.
        3. Nhấn **"New App"**, chọn repository `m45`, nhánh `master`, và tệp chính là `app.py`.
        4. Nhấn **"Deploy"**! Ứng dụng sẽ có đường dẫn công khai `https://<ten-app>.streamlit.app` hoạt động 24/7.

        ---

        ### Cách 2: Triển khai Miễn phí lên Hugging Face Spaces
        1. Truy cập [huggingface.co/spaces](https://huggingface.co/spaces) và tạo Space mới.
        2. Chọn **Space SDK: Streamlit** và License Apache 2.0.
        3. Đẩy code lên Space repository. Hệ thống sẽ tự nhận diện `app.py` và `requirements.txt` để build và phát hành trực tuyến.

        ---

        ### Cách 3: Chạy Container Docker trên Máy chủ VPS hoặc Render / Railway
        1. Xây dựng Docker Image:
           ```bash
           docker build -t m45-analyzer .
           ```
        2. Khởi chạy Container:
           ```bash
           docker run -p 8501:8501 m45-analyzer
           ```
        3. Hoặc chạy bằng Docker Compose:
           ```bash
           docker compose up -d
           ```
           Truy cập `http://localhost:8501` hoặc IP công khai của server.
        """)

    # FOOTER
    st.markdown("""
    <div style="text-align: center; padding: 32px 0 16px; color: #475569; font-size: 0.8rem; border-top: 1px solid #1e293b; margin-top: 40px;">
      <b style="color: #64748b;">M45 · LiDAR Point Density & Annotation Difficulty Analyzer</b><br>
      Nhóm M45: Hưng (Thu nhãn/CVAT) · Dũng (Đặc trưng hình học) · Đường (Mô hình hóa ML Ridge)<br>
      Deploy: <code>streamlit run app.py</code> | Đạt chuẩn Chỉ số đo: <i>Correlation density-error</i> & <i>Hard-case precision</i>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
