# Hướng dẫn Triển khai M45 Screener vào CVAT 3D qua Nuclio Serverless

Thư mục này chứa toàn bộ cấu hình để biến hệ thống **M45 LiDAR Box Difficulty & Pre-label Screener** thành một công cụ Trí tuệ Nhân tạo (AI Auto-Annotation Tool) chạy trực tiếp bên trong giao diện web của **CVAT 3D**.

---

## 1. Cơ chế Hoạt động trong CVAT 3D
- Khi người gán nhãn (Annotator) mở một tác vụ 3D Point Cloud trên CVAT và bấm vào biểu tượng chiếc đũa thần (**AI Tools / Detectors**).
- CVAT sẽ gọi hàm Nuclio `m45-lidar-box-difficulty-screener`.
- Hàm tự động phân tích điểm LiDAR, tính toán khoảng cách $d$, số điểm $N$, và tự động gắn các thuộc tính:
  * `difficulty_tier`: **EASY**, **MEDIUM**, hoặc **HARD**.
  * `difficulty_score`: Điểm định lượng từ 1.0 đến 5.0.
  * `risk_flags`: Cảnh báo rủi ro (lộn góc Yaw $180^\circ$, lệch tâm, co cụm).
  * `annotator_sop`: Hướng dẫn thao tác chuẩn 3 bước cho Annotator.

---

## 2. Các Bước Cài đặt & Triển khai

### Bước 1: Tải công cụ nuctl CLI
Tải phiên bản `nuctl` phù hợp với hệ điều hành của bạn từ GitHub Releases của Nuclio:
- **Windows / Linux / macOS:** [https://github.com/nuclio/nuclio/releases](https://github.com/nuclio/nuclio/releases)
- Đổi tên thành `nuctl` và thêm vào `PATH`.

### Bước 2: Triển khai Hàm vào CVAT
Chạy lệnh sau tại thư mục chứa file `function.yaml`:

```bash
nuctl deploy --project-name cvat \
  --path . \
  --file function.yaml \
  --platform local
```

Nếu CVAT của bạn chạy trên Docker cùng máy, Nuclio sẽ tự động build container Docker `cvat.m45.screener:latest` và kết nối trực tiếp với backend của CVAT.

### Bước 3: Sử dụng trên Giao diện CVAT 3D
1. Mở CVAT tại `http://localhost:8080`.
2. Tạo hoặc mở một tác vụ 3D (`3D Point Cloud`).
3. Mở thanh công cụ AI bên trái -> Chọn **M45 3D LiDAR Difficulty & Pre-label Screener**.
4. Chọn nhãn `vehicle.car` hoặc `human.pedestrian.adult`.
5. Công cụ sẽ tự động sinh hộp kèm toàn bộ thuộc tính độ khó!
