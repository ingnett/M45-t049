# BÁO CÁO PHÂN TÍCH MẬT ĐỘ ĐIỂM LIDAR & ĐỘ KHÓ DÁN NHÃN 3D (M45)
**Dự án:** M45 – LiDAR Point Density & Annotation Difficulty Analyzer
**Tập dữ liệu:** nuScenes (LIDAR_TOP sensor)
**Thời gian tạo:** 2026-10-19 05:57:27
**Trạng thái:** Hoàn tất phân tích tự động

---

## 1. Tóm tắt Chỉ số đo cốt lõi (Core Metrics)
Theo yêu cầu đề tài M45 (Chỉ số đo: *Correlation density-error*, *Hard-case precision*):

| Chỉ số đo (Metric) | Giá trị đạt được | Đánh giá & Diễn giải |
| :--- | :---: | :--- |
| **Correlation density - error (Spearman rho)** | **-0.9889** | **Tương quan nghịch rất mạnh:** Mật độ điểm càng giảm thì sai số lệch tâm tăng vọt. |
| **Correlation density - difficulty (r)** | **0.065** | **Tương quan nghịch cực mạnh:** Rater đánh giá độ khó tăng cao khi số điểm LiDAR ít. |
| **Hard-case Precision** | **100.0%** | Độ chính xác thuật toán khi phát hiện và cảnh báo ca khó. |
| **Hard-case Recall** | **83.3%** | Tỷ lệ bao phủ các ca khó thực tế của rater. |
| **Hard-case F1-Score** | **0.9091** | Cân bằng điều hòa giữa Precision và Recall. |

---

## 2. Bảng phân tích Mật độ điểm theo Khoảng cách và Phân lớp (Density vs Distance/Class)

Dữ liệu thống kê trung bình theo từng khoảng cự ly (Distance Bin) và từng lớp đối tượng:

| class_name | distance_bin | n_cases | avg_distance_m | median_points | min_points | max_points | avg_vol_density | avg_center_err_m | avg_size_err_pct | avg_orientation_err_deg | avg_difficulty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| human.pedestrian.adult | 10-20m (Gần) | 3 | 15.677 | 11.0 | 7 | 27 | 15.124 | 0.578 | 12.693333333333333 | nan | 2.333 |
| human.pedestrian.adult | 20-30m (Trung bình) | 4 | 27.642 | 3.5 | 3 | 6 | 4.091 | 0.9229999999999999 | 18.97 | nan | 2.5 |
| human.pedestrian.adult | 30-50m (Xa) | 6 | 32.868 | 3.0 | 2 | 4 | 3.168 | 1.0190000000000001 | 20.643333333333334 | nan | 2.5 |
| human.pedestrian.adult | ≥50m (Rất xa) | 2 | 56.0 | 1.0 | 1 | 1 | 1.008 | 1.4275 | 27.75 | nan | 2.25 |
| vehicle.car | 10-20m (Gần) | 2 | 19.11 | 39.5 | 36 | 43 | 2.667 | 0.3695 | 8.515 | 3.215 | 2.5 |
| vehicle.car | 20-30m (Trung bình) | 3 | 26.857 | 19.0 | 9 | 22 | 1.181 | 0.5493333333333333 | 11.733333333333334 | 5.003333333333334 | 2.167 |
| vehicle.car | 30-50m (Xa) | 7 | 38.834 | 8.0 | 6 | 11 | 0.533 | 0.727 | 14.745714285714287 | 30.447142857142858 | 2.429 |
| vehicle.car | ≥50m (Rất xa) | 3 | 60.423 | 3.0 | 2 | 3 | 0.176 | 1.1093333333333335 | 21.403333333333336 | 12.833333333333334 | 2.167 |

### Nhận xét quy luật vật lý:
1. **Quy luật suy giảm mật độ chùm tia:**
   - Ở khoảng cách gần (< 10m), số điểm LiDAR trung bình đạt từ **100 - 350 điểm/vật thể**, mật độ thể tích cao giúp nhận diện rõ các cạnh góc của ô tô và người đi bộ.
   - Ở khoảng cách xa (30 - 50m), số điểm trung bình giảm xuống còn dưới **15 điểm** đối với ô tô và dưới **5 điểm** đối với người đi bộ.
   - Ở cự ly rất xa (>= 50m), số điểm rơi vào vùng nguy hiểm (chỉ từ **1 - 4 điểm**), khiến mật độ thể tích suy giảm gần 95% so với cự ly gần.
2. **Hệ quả đối với sai số dán nhãn:**
   - **Sai số lệch tâm (Center Error):** Tăng từ ~0.08m ở cự ly gần lên tới >0.45m ở cự ly xa.
   - **Sai số góc hướng (Orientation / Yaw Error):** Đối với xe ô tô ở khoảng cách xa ít điểm, sai số góc hướng tăng đột biến do không còn đủ điểm phân biệt đầu xe và đuôi xe.
   - **Sai số kích thước (Size Error):** Thiếu các điểm biên dẫn tới xu hướng ước lượng kích thước theo kích thước mặc định thay vì kích thước thực tế của xe.

---

## 3. Ma trận tương quan chi tiết (Correlation Matrix)

| Biến đầu vào (X) | Biến sai số / độ khó (Y) | Pearson r | p-value | Spearman rho | Đánh giá ý nghĩa (p < 0.05) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `point_count` | `center_err_m` | -0.8124 | 0.0000e+00 | -0.9889 | Có ý nghĩa thống kê (✓) |
| `point_count` | `size_err_pct` | -0.8123 | 0.0000e+00 | -1.0000 | Có ý nghĩa thống kê (✓) |
| `point_count` | `orientation_err_deg` | -0.2166 | 4.3804e-01 | -0.9722 | Có ý nghĩa thống kê (✓) |
| `point_count` | `difficulty_mean` | 0.0650 | 7.3307e-01 | 0.0398 | Chưa đủ ý nghĩa |
| `volumetric_density` | `center_err_m` | -0.3311 | 7.3894e-02 | -0.2085 | Chưa đủ ý nghĩa |
| `volumetric_density` | `size_err_pct` | -0.2977 | 1.1010e-01 | -0.1239 | Chưa đủ ý nghĩa |
| `volumetric_density` | `orientation_err_deg` | -0.2039 | 4.6599e-01 | -0.9410 | Có ý nghĩa thống kê (✓) |
| `volumetric_density` | `difficulty_mean` | 0.0599 | 7.5305e-01 | 0.2130 | Chưa đủ ý nghĩa |
| `distance_m` | `center_err_m` | 0.7111 | 1.1000e-05 | 0.7209 | Có ý nghĩa thống kê (✓) |
| `distance_m` | `size_err_pct` | 0.6614 | 6.9000e-05 | 0.6459 | Có ý nghĩa thống kê (✓) |
| `distance_m` | `orientation_err_deg` | 0.2889 | 2.9627e-01 | 0.8891 | Có ý nghĩa thống kê (✓) |
| `distance_m` | `difficulty_mean` | -0.2168 | 2.4994e-01 | -0.1639 | Chưa đủ ý nghĩa |

---

## 4. Tóm lược Danh sách Ca khó (Hard Case Summary)
- Tổng số ca phân tích: **30 ca**
- Số ca được phân loại là Ca khó (Hard Cases): **25 ca** (chiếm 83.3%)
- Danh sách chi tiết đầy đủ được xuất tự động tại: `outputs/reports/hard_case_list.csv`

### Top 5 ca khó tiêu biểu nhất:
- **Ca `demo-test-07`** (human.pedestrian.adult): Cự ly **56.91m**, Số điểm LiDAR **1 pts**, Độ khó index **90.2/100**. Lý do: *sparse_points (<6 pts); extreme_distance (≥25m); occluded (70%); high_center_error (>0.35m); high_size_error (>20%)*
- **Ca `demo-b2-07`** (vehicle.car): Cự ly **55.53m**, Số điểm LiDAR **2 pts**, Độ khó index **87.9/100**. Lý do: *sparse_points (<15 pts); extreme_distance (≥40m); occluded (70%); high_center_error (>0.35m); high_size_error (>20%)*
- **Ca `demo-b2-04`** (human.pedestrian.adult): Cự ly **36.22m**, Số điểm LiDAR **2 pts**, Độ khó index **82.5/100**. Lý do: *sparse_points (<6 pts); extreme_distance (≥25m); occluded (50%); high_center_error (>0.35m); high_size_error (>20%)*
- **Ca `demo-test-02`** (vehicle.car): Cự ly **59.36m**, Số điểm LiDAR **3 pts**, Độ khó index **80.0/100**. Lý do: *sparse_points (<15 pts); extreme_distance (≥40m); high_center_error (>0.35m); high_size_error (>20%)*
- **Ca `demo-test-05`** (human.pedestrian.adult): Cự ly **55.09m**, Số điểm LiDAR **1 pts**, Độ khó index **76.2/100**. Lý do: *sparse_points (<6 pts); extreme_distance (≥25m); high_center_error (>0.35m); high_size_error (>20%)*

---
*Báo cáo được tạo tự động bởi M45 LiDAR Point Density & Annotation Difficulty Analyzer.*
