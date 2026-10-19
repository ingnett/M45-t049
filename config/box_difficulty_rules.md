# BỘ QUY TẮC ĐỊNH LƯỢNG ĐÁNH GIÁ BOX KHÓ VÀ BOX DỄ (BOX DIFFICULTY RULES)
**Dự án:** M45 – LiDAR Point Density & Annotation Difficulty Analyzer (nuScenes mini)  
**Tài liệu:** Bộ tiêu chí phân loại & quy tắc đánh giá Bounding Box 3D  
**Đối tượng áp dụng:** Annotator, Reviewer, Thuật toán AI Pre-labeling & Quality Assurance  

---

## 1. Tổng quan & Mục đích

Trong gán nhãn dữ liệu LiDAR 3D (3D Bounding Box Annotation), độ khó của một ca không phải là cảm tính cá nhân mà bắt nguồn trực tiếp từ **quy luật vật lý suy giảm chùm tia ($1/d^2$)**, **hình học quan sát** và **mức độ bất định thông tin**.

Bộ quy tắc này cung cấp:
1. **Tiêu chí định lượng (Quantitative Criteria):** Định lượng chính xác qua số điểm $N$, cự ly $d$, tỷ lệ che khuất, số mặt nhìn thấy.
2. **Quy tắc phân loại 3 Cấp độ (3 Tiers):** 🟢 **Box Dễ (Easy)**, 🟡 **Box Trung bình (Medium)**, 🔴 **Box Khó (Hard)**.
3. **Cây quyết định (Decision Tree):** Giúp Annotator và hệ thống AI tự động phân loại trong chưa đầy 1 giây.
4. **Hướng dẫn Thao tác chuẩn cho Annotator (SOP):** Ứng xử với từng loại box.

---

## 2. Bảng Tiêu chuẩn Phân loại Chi tiết (Quantitative Rubric)

| Tiêu chí Đánh giá | 🟢 BOX DỄ (EASY) | 🟡 BOX TRUNG BÌNH (MEDIUM) | 🔴 BOX KHÓ (HARD) |
| :--- | :--- | :--- | :--- |
| **Thang điểm Độ khó** | **1.0 – 2.0 / 5.0** | **2.1 – 3.5 / 5.0** | **3.6 – 5.0 / 5.0** (Nguy cơ cao) |
| **Chỉ số Difficulty Index** | **0 – 35 / 100** | **36 – 65 / 100** | **66 – 100 / 100** |
| **Số điểm LiDAR Ô tô (`vehicle.car`)** | **$N \ge 25$ điểm** (thường 40–150+ pts) | **$8 \le N < 25$ điểm** | **$N < 8$ điểm** (Cực khó khi $N \le 4$ pts) |
| **Số điểm LiDAR Người (`pedestrian`)** | **$N \ge 10$ điểm** (thường 12–35 pts) | **$4 \le N < 10$ điểm** | **$N < 4$ điểm** (Chỉ còn 1–3 điểm rời rạc) |
| **Khoảng cách Cự ly Ô tô ($d$)** | **$d < 20\text{ m}$** (Cự ly gần) | **$20\text{ m} \le d < 35\text{ m}$** | **$d \ge 35\text{ m}$** (Rất xa khi $d \ge 50\text{ m}$) |
| **Khoảng cách Cự ly Người ($d$)** | **$d < 15\text{ m}$** | **$15\text{ m} \le d < 25\text{ m}$** | **$d \ge 25\text{ m}$** |
| **Mật độ Thể tích ($N/V$)** | $\ge 2.5\text{ pts/m}^3$ (xe) / $\ge 10$ (người) | $0.6 - 2.5\text{ pts/m}^3$ | $< 0.6\text{ pts/m}^3$ (xe) / $< 4$ (người) |
| **Đặc trưng Hình học nhìn thấy** | Nhìn rõ ít nhất 2 mặt vuông góc (L-Shape) hoặc nóc | Chỉ thấy 1 mặt bên hoặc mặt trước rõ nét | Chỉ có 1 đường thẳng điểm hoặc 1 cụm điểm dính |
| **Mức độ Che khuất (Occlusion)** | **$0\%$** (Hoàn toàn thông thoáng) | **$10\% - 35\%$** (Che khuất nhẹ) | **$\ge 40\%$** hoặc bị cắt cụt (Truncated) |
| **Tạp âm Lân cận (Clutter)** | Độc lập, cách xa mặt đất và vật thể khác | Gần mặt đường hoặc có 1 vật thể lân cận | Dính chùm tán cây, rào chắn hoặc bóng phản xạ |
| **Thời gian Annotator hoàn tất** | **$< 35$ giây** | **$35 - 75$ giây** | **$> 90$ giây** (hoặc chạm timeout 180s) |
| **Độ tự tin của Rater** | $4 - 5 / 5$ (Tuyệt đối tin cậy) | $3 / 5$ (Tương đối tự tin) | $1 - 2 / 5$ (Bất định lớn, phỏng đoán) |
| **Rủi ro Pre-label sai lệch** | Rất thấp ($\Delta \text{Center} \le 0.1\text{m}$) | Trung bình ($\Delta \text{Center} \le 0.25\text{m}$) | Rất cao: Lộn Yaw $180^\circ$, Lệch tâm $>0.35\text{m}$ |

---

## 3. Cây Quyết định Phân loại Nhanh (Decision Flowchart)

```
                            [ BẮT ĐẦU ĐÁNH GIÁ 1 BOX ]
                                        │
           ┌────────────────────────────┴───────────────────────────┐
           ▼                                                        ▼
      [ Lớp Ô tô (vehicle.car) ]                       [ Lớp Người đi bộ (pedestrian) ]
           │                                                        │
   ┌───────┴───────┐                                        ┌───────┴───────┐
   ▼               ▼                                        ▼               ▼
(N < 8 pts    (N >= 8 pts                              (N < 4 pts      (N >= 4 pts
 hoặc d >= 35m  và d < 35m                               hoặc d >= 25m   và d < 25m
 hoặc Occ >= 40%) và Occ < 40%)                           hoặc Occ >= 40%) và Occ < 40%)
   │               │                                        │               │
   ▼               ▼                                        ▼               ▼
[🔴 BOX KHÓ]    [Kiểm tra tiếp]                          [🔴 BOX KHÓ]    [Kiểm tra tiếp]
                   │                                                        │
           ┌───────┴───────┐                                        ┌───────┴───────┐
           ▼               ▼                                        ▼               ▼
     (N >= 25 pts    (8 <= N < 25 pts)                        (N >= 10 pts    (4 <= N < 10 pts)
      và d < 20m)          │                                   và d < 15m)          │
           │               ▼                                        │               ▼
           ▼         [🟡 TRUNG BÌNH]                                ▼         [🟡 TRUNG BÌNH]
      [🟢 BOX DỄ]                                              [🟢 BOX DỄ]
```

---

## 4. Danh mục 5 Lỗi Chí mạng của Box Khó & Cách Khắc phục

### 4.1. Lỗi 1: Lộn góc Yaw $180^\circ$ (Yaw Ambiguity)
- **Dấu hiệu:** Đối tượng ở khoảng cách $>35\text{m}$, chùm tia chỉ quét trúng 1 đầu xe.
- **Hậu quả:** Đầu xe bị gán nhầm thành đuôi xe $\to$ Mô hình xe tự hành dự đoán xe đối diện đi lùi về phía mình.
- **Quy tắc giải quyết:**
  1. Tuyệt đối không phỏng đoán nếu chỉ có điểm 1 đầu.
  2. Bật chế độ chiếu camera RGB phía trước/sau (Front/Back Camera view).
  3. Bật hiển thị vệt lịch sử quỹ đạo (Temporal Track): Xe chạy tiến thì vector vận tốc trùng với góc Yaw.

### 4.2. Lỗi 2: Lệch tâm Hộp ($\Delta\text{Center}$ Drift)
- **Dấu hiệu:** Cảm biến LiDAR chỉ nhận phản xạ ở mặt hướng về xe tự hành (mặt trước/mặt bên). Mặt khuất phía sau hoàn toàn rỗng.
- **Hậu quả:** Annotator hoặc mô hình AI tự động gom tâm hộp vào chính giữa cụm điểm $\to$ Tâm thật của xe bị lệch ra ngoài $0.3\text{m} - 0.8\text{m}$.
- **Quy tắc giải quyết (Face-Snapping Rule):**
  1. Khóa kích thước hộp theo chuẩn tiên nghiệm: $w=1.93\text{m}, l=4.62\text{m}, h=1.69\text{m}$.
  2. Căn mặt trước/mặt bên của hộp bám sát điểm LiDAR đầu tiên.
  3. Đẩy tâm hộp lùi về phía sau theo đúng $l/2$ và $w/2$.

### 4.3. Lỗi 3: Co cụm Kích thước ($\Delta\text{Size}$ Shrinkage)
- **Dấu hiệu:** Cụm điểm quá thưa khiến thuật toán AI tự động co kích thước xe xuống còn $3.2\text{m}$ (thay vì $4.62\text{m}$) hoặc người chỉ còn $0.4\text{m}$.
- **Quy tắc giải quyết:**
  - Nhấn nút **"Khóa Prior Kích thước"**. Không được thu nhỏ kích thước chỉ vì số điểm thưa thớt, trừ khi có bằng chứng rõ ràng đó là xe mini hoặc trẻ em.

### 4.4. Lỗi 4: Dính tạp âm Mặt đất / Rào chắn (Ground & Wall Clutter)
- **Dấu hiệu:** Các điểm dưới bánh xe bị lẫn với tia quét dội lên từ mặt đường nhựa hoặc vỉa hè.
- **Hậu quả:** Chiều cao $h$ bị kéo dài quá mức, tâm $z$ bị tụt sâu xuống lòng đất.
- **Quy tắc giải quyết:**
  - Cắt mặt phẳng đất (Ground plane slicing at $z_{\text{ground}}$): Mặt đáy hộp xe phải nằm cách mặt đường khoảng $0.05\text{m} - 0.1\text{m}$, không chìm dưới lòng đường.

### 4.5. Lỗi 5: Thưa điểm Cực hạn ($N \le 3$ điểm)
- **Dấu hiệu:** Ca ở cự ly $\ge 50\text{m}$, chỉ có 1 đến 3 chấm rời rạc.
- **Quy tắc giải quyết:**
  - Nếu không thể xác định hướng Yaw với độ tự tin $>50\%$: Đánh dấu cờ `insufficient_yaw` hoặc `hard_case`.
  - Giữ kích thước mặc định và ghi chú nguyên nhân `sparse_extreme`.

---

## 5. Quy trình Thao tác Chuẩn dành cho Annotator (3-Step SOP)

Khi gặp bất kỳ box nào được hệ thống M45 gắn nhãn 🔴 **BOX KHÓ**:
- **Bước 1 (Lock Size):** Không kéo chỉnh kích thước $w, l, h$ tự do. Khóa kích thước theo chuẩn lớp (`vehicle.car`: $1.93 \times 4.62 \times 1.69\text{m}$; `pedestrian`: $0.67 \times 0.73 \times 1.77\text{m}$).
- **Bước 2 (Snap to Visible Face):** Dùng Top-view (BEV) căn chỉnh mặt nhìn thấy áp sát vào cụm điểm LiDAR phản xạ.
- **Bước 3 (Verify Yaw with Temporal/RGB):** Liếc nhanh sang ảnh camera RGB hoặc khung hình liền kề để xác định hướng đầu xe. Bấm nộp nhãn và ghi chú tag `[HARD: sparse_pts]`.
