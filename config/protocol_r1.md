# QUY TRÌNH THU THẬP NHÃN VÀ ĐÁNH GIÁ ĐỘ KHÓ ANNOTATION (ĐỢT R1)
**Dự án:** M45 – LiDAR Point Density & Annotation Difficulty Analyzer (nuScenes mini)  
**Phiên bản quy trình:** `protocol_r1`  
**Người phụ trách:** Hưng (Thu nhãn và đo sai số)  
**Ngày ban hành:** 09/10/2026  

---

## 1. Phạm vi và Đối tượng
1. **Tổng số ca:** 28 ca chính + 2 ca luyện tập (pilot) = 30 ca tổng cộng.
   - **Pilot (Luyện tập):** 2 ca (thử quy trình CVAT và phiếu, không train, không đánh giá).
   - **B1 (Batch 1 - Train V1):** 8 ca (4 xe `vehicle.car` + 4 người đi bộ `human.pedestrian.adult`).
   - **DEV (Tập phát triển):** 4 ca (2 xe + 2 người đi bộ).
   - **B2 (Batch 2 - Bổ sung V2):** 8 ca (4 ca lấp khoảng trống + 4 ca ngẫu nhiên).
   - **TEST (Tập kiểm thử cuối):** 8 ca (4 xe + 4 người đi bộ; nhãn được cất kín tới khi có `outputs/LOCK_V1V2.json`).
2. **Quy tắc dán nhãn độc lập (Double-blind):**
   - Mỗi ca được giao cho đúng 2 người dán nhãn độc lập (rater R1, R2, R3).
   - Tổng cộng 28 ca chính × 2 = 56 lượt dán nhãn (khoảng 18–19 lượt/người).
   - Cặp dán nhãn luân phiên và cân bằng: {R1, R2}, {R2, R3}, {R1, R3}.

---

## 2. Vùng quan sát (Crop Window)
Dữ liệu điểm LiDAR được cắt trích xuất cục bộ quanh đối tượng theo cấu hình `config/crop.yaml`:
- **Hệ tọa độ:** Hệ cảm biến LiDAR gốc `LIDAR_TOP`.
- **Dạng cửa sổ:** Cửa sổ vuông theo trục tọa độ (axis-aligned), tâm đặt tại tâm đối tượng.
- **Kích thước nửa cạnh (half-extent):**
  - `vehicle.car`: nửa cạnh **12 m** (cửa sổ 24 m × 24 m trong mặt phẳng XY).
  - `human.pedestrian.adult`: nửa cạnh **8 m** (cửa sổ 16 m × 16 m trong mặt phẳng XY).
  - Chiều cao trục Z: trong khoảng **±5 m** quanh tâm đối tượng.
- Cửa sổ cắt đảm bảo luôn rộng hơn đường chéo bounding box của đối tượng và giữ lại bối cảnh lân cận, không cắt sát viền box.

---

## 3. Quy định bảo mật và che thông tin (Blinding)
Nhằm đảm bảo tính khách quan khoa học, người dán nhãn tuân thủ nghiêm ngặt:
1. **Thông tin người làm ĐƯỢC BIẾT:**
   - Phân loại đối tượng (`class_name`: `vehicle.car` hoặc `human.pedestrian.adult`).
   - Mã nhiệm vụ ngẫu nhiên (`task_code` dạng `T-xxxx`), không thể suy đoán batch hay độ khó.
2. **Thông tin BỊ CHE (TUYỆT ĐỐI KHÔNG BIẾT):**
   - Box gốc ground truth từ nuScenes.
   - Số điểm LiDAR của vật thể (`point_count` $N$).
   - Điểm số dự đoán từ mô hình (`predicted_difficulty`, `score_0_100`).
   - Phiếu đánh giá, thời gian hay bounding box do người làm nhãn còn lại thực hiện.
3. **Quy định tác vụ (Luật cứng số 10):**
   - Người làm nhãn **không được mở giao diện web/app**, không mở file `frames/*.json` hay box gốc nuScenes trước khi nộp xong toàn bộ các lượt của mình trong đợt.

---

## 4. Giới hạn thời gian và 3 trạng thái kết quả (Status)
1. **Giới hạn thời gian:**
   - Tối đa **180 giây/ca** (3 phút).
   - Thời gian tính từ lúc bắt đầu quan sát và điều chỉnh box đến khi hoàn tất nộp. Không tính thời gian tải dữ liệu vào công cụ.
   - Người làm nhãn tự bấm giờ độc lập bằng đồng hồ bấm giờ hoặc công cụ đo thời gian của CVAT.
2. **Ba trạng thái kết thúc ca (`status`):**
   - `completed`: Hoàn tất việc điều chỉnh cuboid 3D và chấm điểm trong thời gian quy định ($duration\_s \le 180$).
   - `timeout`: Quá 180 giây mà chưa thể hoàn tất căn chỉnh box hoặc chấm điểm.
   - `insufficient_context`: Bối cảnh dữ liệu quá thiếu thông tin (điểm LiDAR quá ít hoặc bị che khuất hoàn toàn, không thể xác định đối tượng).
3. **Phân tích tốc độ:**
   - Chỉ sử dụng chỉ số `duration_s` của các ca đạt trạng thái `completed` để phân tích tốc độ dán nhãn.

---

## 5. Quy trình thực hiện qua CVAT 3D
1. **Nhận gói nhiệm vụ:**
   - Mỗi rater nhận file gói dữ liệu tương ứng: `pack_{batch_id}_{rater_id}.zip`.
   - Gói chỉ chứa các file point cloud `.pcd` mang tên `{task_code}.pcd` và bảng ánh xạ lớp `{task_code} -> class_name`.
2. **Thao tác trên CVAT 3D:**
   - Dựng task 3D Point Cloud trên CVAT (hoặc dùng instance CVAT được chỉ định).
   - Điều chỉnh 3D cuboid bao quanh cụm điểm:
     - Định vị tâm (`center_m`: x, y, z).
     - Điều chỉnh kích thước (`size_wlh_m`: width, length, height).
     - Xoay hướng đối tượng (`rotation`: yaw đối với xe ô tô).
3. **Xuất kết quả box:**
   - Xuất annotation định dạng Datumaro JSON hoặc CVAT XML.
   - Nộp file annotation đã xuất cho Hưng.
4. **Điền phiếu đánh giá (`ratings.csv`):**
   - Mở file mẫu `data/labels/ratings_template.csv`.
   - Điền các trường: `case_id` (hoặc `task_code`), `rater_id`, `difficulty_1_5`, `confidence_1_5`, `duration_s`, `status`, `rubric_version` (ghi `r1`), `reason`.
   - Nộp phiếu cho Hưng độc lập.

---

## 6. Khóa nhãn và Nghiệm thu
1. Hưng kiểm tra tính hợp lệ của phiếu qua công cụ `validate_ratings.py`.
2. Hưng chuyển đổi box qua `convert_boxes.py` và lưu trữ vào `human_boxes.json`.
3. Khi đủ đúng 2 lượt hợp lệ cho mỗi ca trong batch, Hưng tạo file đánh dấu `data/labels/LOCKED_{batch}` để khóa batch.
4. Dữ liệu tập TEST được giữ bảo mật, tuyệt đối không mở cho tới khi có file `outputs/LOCK_V1V2.json`.
