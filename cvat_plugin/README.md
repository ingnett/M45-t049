# Hướng dẫn Cài đặt Plugin Xếp hạng Box Khó trực tiếp vào Giao diện CVAT 3D

Plugin **M45 Box Difficulty & Priority Ranker** cho phép nhúng trực tiếp thanh công cụ xếp hạng và định vị box khó vào bên trong trang web của **CVAT 3D** (`http://localhost:8080` hoặc `app.cvat.ai`).

---

## 1. Các Tính năng Trực tiếp trong CVAT 3D
- 🏆 **Bảng Xếp hạng (Leaderboard) theo Thời gian thực:** Tự động quét toàn bộ 3D cuboid trong khung hình và hiển thị danh sách từ **Khó nhất (#01) đến Dễ nhất (#N)** với huy chương 🥇, 🥈, 🥉.
- ⚡ **Phím tắt Nhảy tới Box Khó (`Alt + H`):** Bấm `Alt + H` (hoặc nút **Next Hard**) để camera 3D trong CVAT tự động xoay và phóng to vào box khó tiếp theo cần annotator căn chỉnh.
- 🎨 **Mã màu Trực quan:** Viền đối tượng trong danh sách CVAT tự động đổi màu:
  * 🔴 **Đỏ (#EF4444):** Box Cực khó ($N < 8$, cự ly xa $\ge 35\text{m}$, che khuất).
  * 🟡 **Vàng (#F59E0B):** Box Trung bình.
  * 🟢 **Xanh (#10B981):** Box Dễ, an toàn duyệt nhanh.

---

## 2. Cách Cài đặt (2 Cách Rất Nhanh)

### Cách 1: Chạy trực tiếp qua Trình duyệt (Không cần cài Extension - Nhanh nhất)
1. Mở CVAT trên trình duyệt (Chrome, Edge, Firefox, Brave).
2. Nhấn phím `F12` để mở **Developer Tools** -> Chọn tab **Console**.
3. Mở file [`cvat_m45_ranking_plugin.user.js`](cvat_m45_ranking_plugin.user.js), copy toàn bộ nội dung và dán vào Console rồi nhấn `Enter`.
4. Bảng điều khiển nổi **🎯 M45 BOX RANKER** sẽ xuất hiện ngay lập tức ở góc trên bên phải màn hình CVAT!

---

### Cách 2: Tự động chạy vĩnh viễn qua Tampermonkey / Violentmonkey
1. Cài tiện ích mở rộng **Tampermonkey** từ Chrome Web Store hoặc Firefox Add-ons:
   [https://www.tampermonkey.net/](https://www.tampermonkey.net/)
2. Bấm vào biểu tượng Tampermonkey -> Chọn **Create a new script...**
3. Dán toàn bộ nội dung file [`cvat_m45_ranking_plugin.user.js`](cvat_m45_ranking_plugin.user.js) vào và nhấn `Ctrl + S` để lưu.
4. Từ nay mỗi khi mở CVAT, công cụ xếp hạng sẽ tự động kích hoạt 100%!
