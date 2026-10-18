# BLOCKED.md - Danh sách các cổng cần can thiệp của người (Human-in-the-loop)
Nhóm: M45 | Dự án: LiDAR Point Density & Annotation Difficulty Analyzer

Theo quy định mục 0.3 và mục 10: Khi gặp cổng 🧑 của mình, agent ghi nhận tại file này, báo cáo trong trao đổi, và tiếp tục triển khai các phần công việc không bị phụ thuộc.

| Vai | Cổng | Cần gì từ con người | Trạng thái | Ghi chú |
| :--- | :---: | :--- | :---: | :--- |
| **Hưng** | **G1** | Dựng công cụ CVAT 3D, nhập ca thử nghiệm do `test_box_roundtrip.py` sinh ra, xuất lại annotation từ CVAT, xác nhận kết quả kiểm thử roundtrip đạt (tâm/kích thước ≤ 1e-4 m, hướng ≤ 0.01°). Nếu không import được, nhập tay giá trị hộp vào CVAT rồi export. | **CHỜ NGƯỜI THỰC HIỆN** | Agent đã tạo script `test_box_roundtrip.py` và `convert_boxes.py`, sinh sẵn ca thử nghiệm chuẩn. |
| **Hưng** | **G3** | Khóa phiên bản `rubric_r1.md` và `protocol_r1.md` trước khi tiến hành dán nhãn đợt B1. Xác nhận không thay đổi rubric sau khi thu nhãn. | **CHỜ NGƯỜI XÁC NHẬN** | Agent đã hoàn tất tài liệu tại `config/rubric_r1.md` và `config/protocol_r1.md`. |
| **Hưng** | **G4** | Điều phối 3 thành viên (R1, R2, R3) dán nhãn các đợt B1, DEV, TEST, B2 (bấm giờ tay ≤ 180s/ca, nộp box và phiếu độc lập); Hưng khóa nhãn từng batch bằng cách tạo file `data/labels/LOCKED_{batch}` sau khi đủ 2 lượt độc lập. | **CHỜ DỮ LIỆU THỰC** | Agent đã xây dựng `make_assignments.py`, `make_cvat_pack.py`, `ratings_template.csv`, `validate_ratings.py`. |
