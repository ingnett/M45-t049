# RUBRIC ĐÁNH GIÁ ĐỘ KHÓ DÁN NHÃN 3D CUBOID (PHIÊN BẢN R1)
**Dự án:** M45 – LiDAR Point Density & Annotation Difficulty Analyzer (nuScenes mini)  
**Phiên bản rubric:** `r1`  
**Người phụ trách:** Hưng (Thu nhãn và đo sai số)  
**Khóa phiên bản:** Cổng G3 (Khóa cố định trước đợt B1)  

---

## 1. Thang điểm độ khó (`difficulty_1_5`)
Thang điểm nguyên từ 1 đến 5 đánh giá mức độ khó khăn, thời gian suy xét và sự bất định khi xác định 3D cuboid bao quanh đối tượng:

| Điểm | Mức độ | Định nghĩa và Tiêu chí cụ thể |
| :---: | :--- | :--- |
| **1** | **Xác định rõ ràng** | Cụm điểm LiDAR đầy đủ, mật độ cao, hình dạng hình học đối tượng hiển nhiên. Biên giới và góc cạnh rõ nét; tách biệt hoàn toàn khỏi nền, mặt đất và các vật cản lân cận. Hướng đối tượng (orientation) và kích thước (w, l, h) nhận diện được ngay lập tức không cần do dự. |
| **2** | **Chút bất định nhưng hoàn tất dễ** | Hình dạng đối tượng tương đối rõ ràng. Có thể thiếu một số điểm ở mặt sau, cạnh bên hoặc mật độ điểm hơi thưa, nhưng đặc trưng cấu trúc chính vẫn quan sát tốt. Căn chỉnh nhanh chóng, mức độ tự tin cao. |
| **3** | **Phải đối chiếu nhiều góc nhìn và chỉnh đáng kể** | Mật độ điểm LiDAR mức trung bình hoặc bị che khuất một phần. Điểm LiDAR có thể gần mặt đường hoặc vật thể lân cận. Người làm phải chuyển đổi qua lại giữa nhiều góc chiếu (Top, Side, Front, BEV 3D), phóng to/thu nhỏ và tinh chỉnh kích thước/tâm/hướng nhiều lần mới định vị được. |
| **4** | **Khó xác định ít nhất một yếu tố, còn bất định khi nộp** | Cụm điểm rất thưa (ví dụ ở cự ly xa hoặc bị che khuất mạnh) hoặc nằm lẫn trong cụm điểm của vật thể khác / cây cối lề đường. Người làm gặp khó khăn nghiêm trọng khi xác định ít nhất một trong ba yếu tố: vị trí tâm, kích thước 3 chiều, hoặc hướng xoay. Vẫn còn sự bất định rõ rệt sau khi hoàn thành. |
| **5** | **Rất khó hoàn tất đáng tin cậy** | Cụm điểm cực kỳ thưa thớt (chỉ vài điểm rời rạc 1–3 điểm) hoặc bị che khuất gần như triệt để. Không đủ cơ sở dữ liệu điểm để suy đoán đáng tin cậy về ranh giới hộp 3D và hướng đầu xe / người. Việc đặt box phần lớn dựa vào phỏng đoán mơ hồ. |

*Lưu ý:* Nếu người làm nhãn không thể xác định điểm số (ví dụ gặp lỗi dữ liệu hoặc không đủ bối cảnh), ô `difficulty_1_5` để trống (không điền 0 hay giá trị ngoài [1, 5]).

---

## 2. Thang điểm mức độ tự tin (`confidence_1_5`)
Đánh giá mức độ tự tin của chính người dán nhãn đối với bounding box mà mình vừa tạo:
- **1:** Hoàn toàn không tự tin (suy đoán chủ quan, khả năng sai lệch lớn).
- **2:** Tự tin thấp (còn nhiều điểm nghi ngờ về vị trí hoặc kích thước).
- **3:** Tự tin trung bình (chấp nhận được nhưng chưa tối ưu).
- **4:** Tự tin cao (hộp khớp tốt với phần lớn cụm điểm).
- **5:** Rất tự tin (hộp bao khít hoàn hảo, chính xác cao).

*Lưu ý:* Để trống nếu ca không chấm được.

---

## 3. Danh mục nguyên nhân thách thức (`reason`)
Khi gặp khó khăn (đặc biệt các ca điểm khó từ mức 3 trở lên), người dán nhãn ghi nhận nguyên nhân chính bằng cách chọn một hoặc nhiều thẻ phân tách bởi dấu chấm phẩy `;` từ tập hợp sau:

- `center`: Khó định vị tọa độ tâm 3D $(x, y, z)$ do cụm điểm không cân đối hoặc bị cắt cụt.
- `size`: Khó xác định chính xác kích thước $(w, l, h)$ do mất điểm ở các bề mặt biên.
- `orientation`: Khó xác định hướng xoay (góc yaw / quaternion), đặc biệt là phân biệt đầu xe và đuôi xe ở khoảng cách xa.
- `context`: Bối cảnh xung quanh phức tạp gây nhiễu (mặt đất dốc/gồ ghề, dính chùm với người/xe khác, tường rào, bóng phản xạ LiDAR).
- `other`: Nguyên nhân kỹ thuật khác (cần ghi chú rõ nếu có).

Ví dụ hợp lệ: `orientation`, `center;size`, `orientation;context`.

---

## 4. Quy ước xử lý dữ liệu nhãn (y)
1. **Một lượt hợp lệ:** Lượt dán nhãn được tính là hợp lệ nếu trường `difficulty_1_5` không bị để trống (bất kể `status` là `completed`, `timeout` hay `insufficient_context`).
2. **Nhãn hợp nhất ($y$):**
   - Một ca đạt `y_status = ok` khi và chỉ khi có **đúng hai lượt đánh giá hợp lệ** từ 2 rater độc lập.
   - Điểm nhãn mục tiêu $y$ (`difficulty_mean`) được tính bằng trung bình cộng của 2 điểm:
     $$y = \frac{\text{difficulty}_1 + \text{difficulty}_2}{2}$$
   - Nếu có ít hơn 2 lượt hợp lệ: gán `y_status = excluded` kèm lý do (ví dụ `lt2_valid`).
   - Tuyệt đối không tự bịa điểm hoặc bổ sung điểm giả định khi thiếu lượt.
