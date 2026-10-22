// ========================================================
// VLEARN K04-L23-P1-T0 COMPREHENSIVE KNOWLEDGE DATABASE
// 14 Days of Vision & Autonomous Driving AI Curriculum
// ========================================================

const courseData = {
  courseId: "K04-L23-P1-T0",
  courseName: "K04-L23-P1-T0 · Khóa 4 Phase 1",
  totalDays: 14,
  
  days: [
    {
      id: "D01",
      dayNumber: 1,
      title: "Data (Cơ sở Dữ liệu AI)",
      tag: "Data Foundations",
      slidesCount: 4,
      summary: "Tổng quan về vòng đời dữ liệu AI, các tiêu chuẩn chất lượng dữ liệu huấn luyện, bias trong tập dữ liệu và phân loại bài toán thị giác máy tính.",
      sections: [
        {
          title: "1. Vòng đời dữ liệu AI (AI Data Lifecycle)",
          content: "Dữ liệu là cốt lõi của mô hình Deep Learning. Vòng đời dữ liệu bao gồm: Thu thập (Collection) → Tiền xử lý & Làm sạch (Cleaning) → Gán nhãn (Annotation) → Đảm bảo chất lượng (QA/QC) → Quản lý phiên bản (Versioning) → Giám sát & Bổ sung (Active Learning).",
          bullets: [
            "Định luật 'Garbage In, Garbage Out' (GIGO): Nhãn sai lệch gây ảnh hưởng nghiêm trọng hơn nhiều so với việc thiếu dữ liệu.",
            "Tập Train, Val, Test phải phân phối độc lập và đồng nhất (i.i.d), tránh hiện tượng rò rỉ dữ liệu (Data Leakage) giữa các frame liên tiếp trong video.",
            "Class Imbalance: Tình trạng mẫu hiếm (vd: người đi xe lăn, động vật băng qua đường) cần chiến lược oversampling hoặc cost-sensitive loss."
          ]
        },
        {
          title: "2. Các chuẩn định dạng nhãn cơ bản",
          content: "Tùy thuộc bài toán (Phân loại, Định vị, Phân vùng), nhãn có định dạng biểu diễn toán học khác nhau:",
          bullets: [
            "Classification: One-hot vector hoặc nhãn số nguyên.",
            "2D Bounding Box: [x_min, y_min, width, height] (COCO) hoặc [x_min, y_min, x_max, y_max] (VOC).",
            "Polygon / Mask: Tập hợp các điểm (x, y) nối nhau hoặc ma trận nhị phân 0-1."
          ]
        }
      ],
      examTrap: "Lưu ý đề thi: Chuẩn COCO lưu width và height tuyệt đối theo pixel, trong khi YOLO chuẩn hóa (normalized) từ 0 đến 1 dựa vào kích thước ảnh gốc!"
    },
    {
      id: "D02",
      dayNumber: 2,
      title: "Object Detection (Phát hiện vật thể 2D)",
      tag: "2D Detection",
      slidesCount: 2,
      summary: "Quy chuẩn gán nhãn 2D Bounding Box, xử lý che khuất (Occlusion), cắt viền (Truncation), các chỉ số IoU, Precision, Recall và mAP.",
      sections: [
        {
          title: "1. Tiêu chuẩn gán Bounding Box (Bounding Box Guidelines)",
          content: "Bounding Box phải bao sát mép ngoài cùng của đối tượng (Tight bounding box). Không được bao quá rộng (thừa nền - background noise) hoặc quá chật (mất đặc trưng nhận dạng).",
          bullets: [
            "Occlusion (Che khuất): Chỉ đóng hộp phần nhìn thấy được (Visible box) hay dự đoán cả phần bị che (Amodal box) tùy theo guideline của từng bài toán.",
            "Truncation (Cắt viền): Khi đối tượng nằm ở mép ảnh và bị cụt một phần, phải gán nhãn cờ truncated = True.",
            "Ngưỡng kích thước tối thiểu: Đối tượng quá nhỏ (< 10x10 px hoặc quá mờ) thường gắn nhãn 'Do not care' hoặc 'Ignore' để tránh làm nhiễu gradient khi huấn luyện."
          ]
        },
        {
          title: "2. Đánh giá mô hình: IoU & mAP",
          content: "IoU (Intersection over Union) đo độ trùng khớp giữa Box dự đoán và Box nhãn chuẩn. Ngưỡng IoU = 0.5 là ranh giới phân định True Positive (TP) và False Positive (FP).",
          bullets: [
            "Precision = TP / (TP + FP); Recall = TP / (TP + FN).",
            "NMS (Non-Maximum Suppression): Loại bỏ các box trùng lặp cùng dự đoán một đối tượng bằng cách giữ box có confidence cao nhất và xóa các box có IoU > threshold với nó.",
            "mAP@0.5 vs mAP@[0.5:0.95]: mAP COCO tính trung bình cộng AP tại 10 mốc IoU từ 0.50 đến 0.95 với bước nhảy 0.05."
          ]
        }
      ],
      examTrap: "Nếu Box dự đoán đúng class nhưng IoU = 0.49 (nhỏ hơn ngưỡng 0.5), mô hình sẽ bị phạt cả 1 False Positive (dự đoán sai vị trí) VÀ 1 False Negative (bỏ sót đối tượng chuẩn)!"
    },
    {
      id: "D03",
      dayNumber: 3,
      title: "MultiFrame Tracking (Theo dõi đa đối tượng)",
      tag: "Tracking & Video",
      slidesCount: 3,
      summary: "Theo dõi đối tượng qua chuỗi khung hình video, gán ID liên tục, thuật toán SORT, DeepSORT, ma trận chi phí Hungarian và chỉ số MOTA/MOTP.",
      sections: [
        {
          title: "1. Khái niệm Multi-Object Tracking (MOT)",
          content: "MOT không chỉ phát hiện vật thể ở từng frame riêng lẻ mà phải liên kết (associate) các vật thể qua thời gian để tạo thành quỹ đạo (tracklet) với Track ID duy nhất.",
          bullets: [
            "Hiện tượng ID Switch (Đổi ID): Xảy ra khi hai vật thể đi cắt chéo nhau hoặc bị che khuất tạm thời (Lost track), khi xuất hiện lại bị gán ID mới.",
            "Kalman Filter: Dự đoán vị trí tiếp theo của đối tượng dựa trên trạng thái vận tốc và gia tốc trước đó.",
            "Hungarian Algorithm (Thuật toán Hungary): Giải bài toán ghép cặp tối ưu giữa các Detection ở frame hiện tại và Tracklet dự đoán từ Kalman Filter."
          ]
        },
        {
          title: "2. Thuật toán SORT và DeepSORT",
          content: "SORT chỉ dùng vị trí và kích thước hộp (Spatial distance / IoU distance) để ghép cặp, dễ bị văng ID khi xe chạy nhanh hoặc che khuất. DeepSORT bổ sung mạng trích xuất đặc trưng ngoại hình (Appearance Embedding) và Cosine Distance để tái nhận diện (Re-ID).",
          bullets: [
            "MOTA (Multiple Object Tracking Accuracy): Đo độ chính xác tổng thể, trừ điểm theo False Positives, False Negatives và ID Switches.",
            "MOTP (Multiple Object Tracking Precision): Đo sai số vị trí trung bình giữa các bounding box ghép cặp thành công."
          ]
        }
      ],
      examTrap: "SORT thất bại nặng nề khi vật thể bị che khuất dài (long-term occlusion) vì Kalman Filter mất dấu và không có bộ nhớ đặc trưng hình ảnh như DeepSORT!"
    },
    {
      id: "D04",
      dayNumber: 4,
      title: "Keypoint and Pose Data (Điểm mấu chốt & Tư thế)",
      tag: "Pose Estimation",
      slidesCount: 3,
      summary: "Gán nhãn khung xương người (Skeleton), các điểm mấu chốt (Keypoints COCO 17 điểm), trạng thái nhìn thấy (visibility flag), heatmap và độ đo OKS.",
      sections: [
        {
          title: "1. Cấu trúc nhãn Pose & Keypoint",
          content: "Chuẩn COCO quy định 17 điểm mấu chốt trên cơ thể người (mũi, mắt, tai, vai, khuỷu tay, cổ tay, hông, đầu gối, mắt cá chân). Mỗi điểm gồm tọa độ (x, y) và cờ trạng thái v:",
          bullets: [
            "v = 0: Điểm không được gắn nhãn hoặc nằm hoàn toàn ngoài khung hình.",
            "v = 1: Điểm bị che khuất (Occluded / Labeled but not visible) nhưng dự đoán được vị trí giải phẫu.",
            "v = 2: Điểm nhìn thấy rõ ràng (Visible & clearly labeled)."
          ]
        },
        {
          title: "2. Chỉ số OKS (Object Keypoint Similarity)",
          content: "Thay vì dùng IoU như Bounding Box, Pose Estimation dùng OKS để tính độ trùng khớp giữa các điểm:",
          bullets: [
            "Công thức: OKS = ∑ exp(-d_i² / (2 * s² * k_i²)) * v_i / ∑ v_i.",
            "d_i là khoảng cách Euclidean giữa điểm dự đoán và ground-truth.",
            "s là căn bậc hai diện tích bao quanh người (scale factor).",
            "k_i là hệ số sai số cho phép của từng khớp (ví dụ: khớp vai k_i lớn hơn điểm mắt/mũi vì vai khó xác định điểm chính xác hơn)."
          ]
        }
      ],
      examTrap: "Trạng thái v=1 cực kỳ quan trọng: Người gán nhãn phải dùng kiến thức giải phẫu học cơ thể để suy luận điểm bị che khuất sau lưng áo hoặc cánh tay!"
    },
    {
      id: "D05",
      dayNumber: 5,
      title: "Segmentation Dataset (Tập dữ liệu Phân vùng)",
      tag: "Segmentation",
      slidesCount: 3,
      summary: "Phân biệt Semantic, Instance và Panoptic Segmentation; kỹ thuật mã hóa Run-Length Encoding (RLE), độ đo Dice Coefficient và mIoU.",
      sections: [
        {
          title: "1. Ba cấp độ Phân vùng ảnh",
          content: "Thị giác máy tính chia phân vùng thành 3 bài toán lớn:",
          bullets: [
            "Semantic Segmentation: Gán từng pixel vào một lớp (vd: toàn bộ xe ô tô cùng màu đỏ, mặt đường màu xám). Không phân biệt từng cá thể xe.",
            "Instance Segmentation: Chỉ phân vùng các đối tượng đếm được (things) và phân biệt riêng rẽ từng cá thể (xe A, xe B, người 1, người 2).",
            "Panoptic Segmentation: Hợp nhất cả hai: phân biệt từng cá thể (things: car, person) VÀ phân loại nền vô định hình (stuff: sky, road, grass)."
          ]
        },
        {
          title: "2. Run-Length Encoding (RLE) & Độ đo mIoU / Dice",
          content: "Lưu trữ mask nhị phân dạng ma trận tốn rất nhiều dung lượng. RLE nén chuỗi pixel liên tục cùng giá trị (vd: 50 pixel 0, 120 pixel 1) giúp giảm dung lượng file xuống hàng chục lần.",
          bullets: [
            "mIoU (Mean Intersection over Union): Tỉ lệ diện tích giao trên hợp của các vùng mask theo từng class.",
            "Dice Coefficient = 2 * |A ∩ B| / (|A| + |B|). Khi IoU = 0.5, Dice = 2*0.5 / (1 + 0.5) = 0.667."
          ]
        }
      ],
      examTrap: "Semantic segmentation KHÔNG thể đếm số lượng xe trên đường vì các xe đỗ sát nhau sẽ dính liền thành một vùng pixel chung!"
    },
    {
      id: "D06",
      dayNumber: 6,
      title: "Data Workflow & Quality Control (Quy trình & Đảm bảo Chất lượng)",
      tag: "QA / QC Process",
      slidesCount: 3,
      summary: "Xây dựng hướng dẫn gán nhãn (Annotation Guideline), đo lường mức độ đồng thuận giữa người gán (Inter-Annotator Agreement - Cohen/Fleiss Kappa), quy trình kiểm thử QA/QC nhiều tầng.",
      sections: [
        {
          title: "1. Quy trình QA/QC chuẩn công nghiệp",
          content: "Chất lượng dữ liệu quyết định giới hạn trên của mô hình. Một pipeline gán nhãn chuyên nghiệp gồm 3 vòng: Vòng 1 (Annotator gán ban đầu) → Vòng 2 (Reviewer/QC kiểm tra mẫu 20-100%) → Vòng 3 (Gold Standard Audit / Chuyên gia thẩm định).",
          bullets: [
            "Honeypot / Benchmark Tasks: Trộn các tác vụ đã biết trước đáp án chuẩn vào luồng công việc để phát hiện người gán cẩu thả.",
            "Phân tích lỗi (Error Taxonomy): Phân loại lỗi thành: Sai class (Misclassification), Sai ranh giới (Boundary error), Bỏ sót (False Negative), Gán nhầm rác (False Positive)."
          ]
        },
        {
          title: "2. Đo mức độ đồng thuận (Inter-Annotator Agreement)",
          content: "Để biết hướng dẫn gán nhãn có mơ hồ hay không, cần cho nhiều người gán độc lập cùng một tập ảnh và đo hệ số Fleiss' Kappa hoặc Cohen's Kappa:",
          bullets: [
            "Kappa < 0.4: Đồng thuận kém, guideline đang rất mơ hồ, cần viết lại.",
            "0.6 <= Kappa < 0.8: Đồng thuận khá tốt.",
            "Kappa >= 0.8: Đồng thuận xuất sắc, nhãn đáng tin cậy để huấn luyện mô hình sản xuất."
          ]
        }
      ],
      examTrap: "Tỉ lệ phần trăm trùng khớp đơn thuần (Percent Agreement) thường bị thổi phồng do yếu tố ngẫu nhiên; bắt buộc phải dùng hệ số Cohen's/Fleiss' Kappa để loại trừ yếu tố may rủi!"
    },
    {
      id: "D07",
      dayNumber: 7,
      title: "Data Pipeline & Governance (Đường ống Dữ liệu & Quản trị)",
      tag: "Data Ops & MLOps",
      slidesCount: 2,
      summary: "Kiến trúc đường ống ETL cho dữ liệu thị giác máy tính, công cụ Data Version Control (DVC), Data Lineage và bảo mật quyền riêng tư (che mờ mặt và biển số xe).",
      sections: [
        {
          title: "1. Data Version Control (DVC)",
          content: "Git không thể lưu trữ các file ảnh/video dung lượng Gigabyte. DVC kết hợp với Git bằng cách chỉ commit file con trỏ mã hash (.dvc) vào Git, trong khi dữ liệu thô nhị phân được lưu trữ ở Cloud Storage (S3, GCS, MinIO).",
          bullets: [
            "Data Lineage (Nguồn gốc dữ liệu): Truy vết được mô hình v1.2 được huấn luyện chính xác từ snapshot dữ liệu ngày nào, do ai review.",
            "Dataset Drift & Concept Drift: Dữ liệu thực tế thay đổi theo thời tiết, mùa màng đòi hỏi pipeline phải liên tục cập nhật và kiểm thử tự động."
          ]
        },
        {
          title: "2. Tuân thủ quyền riêng tư (Privacy & Anonymization)",
          content: "Dữ liệu camera đường phố bắt buộc phải tuân thủ chuẩn GDPR/bảo vệ dữ liệu cá nhân:",
          bullets: [
            "Face Blurring: Nhận diện và làm mờ khuôn mặt người đi bộ/tài xế.",
            "License Plate Blurring: Nhận diện và làm mờ biển số xe cơ giới trước khi đưa vào kho dữ liệu chia sẻ chung."
          ]
        }
      ],
      examTrap: "DVC không lưu dữ liệu trực tiếp trong Git commit mà lưu meta-file con trỏ md5 hash chỉ tới Data Remote Storage!"
    },
    {
      id: "D08",
      dayNumber: 8,
      title: "AI-Assisted Data & Active Learning (Gán nhãn Hỗ trợ bởi AI)",
      tag: "Active Learning",
      slidesCount: 2,
      summary: "Tận dụng mô hình tiền huấn luyện để Pre-annotate (gán nhãn trước), Human-in-the-Loop, chiến lược chọn mẫu không chắc chắn (Uncertainty Sampling) trong Active Learning.",
      sections: [
        {
          title: "1. AI-Assisted Annotation (Gán nhãn bán tự động)",
          content: "Thay vì vẽ từ đầu, người gán nhãn chỉ cần kiểm tra và chỉnh sửa (Audit & Correct) các box do mô hình sơ bộ (Foundation Model / Pre-trained YOLO) dự đoán trước, giúp tăng tốc độ gán từ 3x đến 10x.",
          bullets: [
            "Model-Assisted Bias: Rủi ro người gán nhãn lười biếng chấp nhận luôn nhãn sai của AI mà không sửa kỹ.",
            "Interactive Segmentation (SAM - Segment Anything Model): Người dùng chỉ cần click 1 điểm (prompt point) là AI sinh ra polygon chính xác."
          ]
        },
        {
          title: "2. Chiến lược Active Learning",
          content: "Khi có hàng triệu ảnh thô chưa gán nhãn, không thể gán tất cả. Active Learning tự động chọn ra những mẫu mang lại nhiều thông tin nhất cho mô hình học:",
          bullets: [
            "Uncertainty Sampling: Chọn các ảnh mô hình dự đoán với độ tự tin thấp nhất (Entropy cao hoặc khoảng cách giữa top 1 và top 2 xác suất rất nhỏ).",
            "Diversity Sampling: Chọn các mẫu có đặc trưng hình ảnh lạ nhất chưa từng có trong tập huấn luyện (Out-of-distribution)."
          ]
        }
      ],
      examTrap: "Gán nhãn ngẫu nhiên 10,000 ảnh thường kém hiệu quả hơn nhiều so với việc chỉ gán nhãn 2,000 ảnh được chọn lọc theo chiến lược Uncertainty Sampling của Active Learning!"
    },
    {
      id: "D09",
      dayNumber: 9,
      title: "ROAD ELEMENTS (Nhận diện Thành phần Đường bộ)",
      tag: "Autonomous Driving",
      slidesCount: 2,
      summary: "Gán nhãn chuyên biệt cho xe tự hành: Vạch kẻ đường (Lane markings), Vỉa hè (Curbs), Vùng xe chạy được (Drivable Area), Biển báo và Đèn tín hiệu giao thông.",
      sections: [
        {
          title: "1. Vạch kẻ đường & Vỉa hè (Lane Lines & Curbs)",
          content: "Xe tự hành cần nhận biết chính xác làn đường để điều khiển bám làn (Lane Keeping Assist).",
          bullets: [
            "Phân loại vạch kẻ: Vạch liền (Solid) - cấm chuyển làn; Vạch đứt (Dashed) - được phép chuyển làn; Vạch đôi, vạch dừng xe (Stop line), vạch mắt võng.",
            "Curbs (Bó vỉa hè): Ranh giới vật lý giữa lòng đường và vỉa hè dành cho người đi bộ, có ý nghĩa an toàn sinh tử để tránh va quẹt bánh xe.",
            "Biểu diễn Polyline: Vạch kẻ đường thường được biểu diễn dưới dạng đường gấp khúc có thứ tự điểm (Ordered Spline / Polyline) thay vì bounding box."
          ]
        },
        {
          title: "2. Vùng xe chạy được (Drivable Area)",
          content: "Drivable area là polygon phân vùng toàn bộ diện tích mặt đường xe có thể di chuyển hợp pháp và an toàn, trừ đi các vật cản, đảo giao thông và vỉa hè.",
          bullets: [
            "Biển báo giao thông (Traffic Signs): Cần gắn nhãn cả vị trí 2D box và thuộc tính ngữ nghĩa (Sign Type: Cấm rẽ, Tốc độ tối đa, Biển nguy hiểm).",
            "Đèn tín hiệu (Traffic Lights): Phải gắn nhãn trạng thái màu (Red, Yellow, Green, Off) và hướng mũi tên (rẽ trái/phải)."
          ]
        }
      ],
      examTrap: "Vạch kẻ đường bị mờ hoặc đứt gãy do thời tiết/bóng râm cây cối yêu cầu người gán nhãn phải nội suy theo hình học đường cong của cả đoạn đường!"
    },
    {
      id: "D10",
      dayNumber: 10,
      title: "Vinfast OMS & DMS (Hệ thống Giám sát trong Khoang xe)",
      tag: "In-Cabin AI",
      slidesCount: 1,
      summary: "Giám sát tài xế (DMS - Driver Monitoring System) và khoang hành khách (OMS - Occupant Monitoring System): Nhận diện hướng nhìn, buồn ngủ, tư thế hành khách và an toàn túi khí.",
      sections: [
        {
          title: "1. Driver Monitoring System (DMS)",
          content: "Sử dụng camera hồng ngoại (IR Camera) chiếu vào mặt tài xế kể cả ban đêm hoặc khi đeo kính râm:",
          bullets: [
            "Gaze Tracking (Hướng nhìn): Đo góc quay đầu (Head Pose: Pitch, Yaw, Roll) và vector hướng mắt để cảnh báo mất tập trung quá 2 giây.",
            "PERCLOS (Percentage of Eye Closure): Tỉ lệ thời gian mắt nhắm lại trong một khoảng thời gian (chuẩn quốc tế xác định tài xế đang buồn ngủ / ngủ gật).",
            "Hành vi nguy hiểm: Dùng điện thoại di động, hút thuốc, ngáp liên tục, thả hai tay khỏi vô lăng."
          ]
        },
        {
          title: "2. Occupant Monitoring System (OMS)",
          content: "Camera góc rộng bao quát toàn bộ hàng ghế trước và sau trong cabin:",
          bullets: [
            "Phát hiện thắt dây an toàn (Seatbelt Detection): Phân biệt chính xác giữa việc cài dây thật và việc cài chốt giả để tắt chuông cảnh báo.",
            "Phát hiện trẻ em bị bỏ quên (Child Left Behind Detection): Cảnh báo sống còn khi tài xế khóa xe rời đi mà vẫn còn em bé trong ghế phụ hoặc ghế sau.",
            "Tư thế hành khách (Passenger Posture): Định lượng khoảng cách đầu/ngực đến bảng taplo để điều chỉnh lực bung túi khí thông minh khi xảy ra va chạm."
          ]
        }
      ],
      examTrap: "Camera DMS trong xe hơi luôn là camera dải hồng ngoại (Near-Infrared / NIR 940nm) để hoạt động hoàn hảo trong bóng tối mà không làm chói mắt tài xế!"
    },
    {
      id: "D11",
      dayNumber: 11,
      title: "SVM-360 View (Hệ thống Camera Toàn cảnh 360)",
      tag: "Surround View & BEV",
      slidesCount: 2,
      summary: "Hệ thống 4 camera mắt cá (Fisheye) xung quanh xe, hiệu chỉnh nội suy/ngoại suy, khử méo góc rộng, phép chiếu Homography và ghép ảnh Bird's Eye View (BEV).",
      sections: [
        {
          title: "1. Cấu trúc hệ thống SVM (Surround View Monitoring)",
          content: "SVM gồm 4 camera góc siêu rộng (thường > 180° fisheye) lắp ở: Đầu xe (Front), Đuôi xe (Rear), Gương chiếu hậu trái (Left Mirror), Gương chiếu hậu phải (Right Mirror).",
          bullets: [
            "Méo hình mắt cá (Barrel / Fisheye Distortion): Hình ảnh bị uốn cong mạnh ở rìa, cần mô hình hiệu chuẩn Scaramuzza hoặc Kannala-Brandt để khử méo phẳng.",
            "Ma trận Ngoại suy (Extrinsics): Xác định vị trí tọa độ 3D (X, Y, Z) và góc quay (Roll, Pitch, Yaw) của từng camera đối với trục tâm xe."
          ]
        },
        {
          title: "2. Biến đổi Bird's Eye View (BEV) & Ghép ảnh (Stitching)",
          content: "Hình ảnh từ 4 hướng được chiếu phẳng xuống mặt đất thông qua ma trận Homography:",
          bullets: [
            "Tạo góc nhìn từ trên trời nhìn thẳng xuống (Top-down view / Bird's Eye View) giúp tài xế quan sát toàn bộ điểm mù xung quanh xe.",
            "Ghép nối (Stitching & Blending): Vùng giao nhau giữa 2 camera (vd: góc trước-trái) được khử đường viền bằng thuật toán Alpha Blending đa độ phân giải để tạo ảnh mượt mà không bị giật nếp gấp."
          ]
        }
      ],
      examTrap: "Phép chiếu Homography trong SVM chỉ bảo toàn tính đúng đắn trên mặt phẳng mặt đường; các vật thể cao như cột đèn, xe bên cạnh sẽ bị hiệu ứng kéo dài méo mó (stretching artifact) khi nhìn trên ảnh BEV!"
    },
    {
      id: "D12",
      dayNumber: 12,
      title: "Data Annotation Point Cloud & Cuboid (Gán nhãn Đám mây điểm 3D)",
      tag: "3D LiDAR Annotation",
      slidesCount: 2,
      summary: "Đám mây điểm LiDAR 3D, hộp 3D Bounding Box (Cuboid 7 bậc tự do: tâm X, Y, Z, kích thước L, W, H, góc quay Yaw), mặt trước vật thể (Heading angle).",
      sections: [
        {
          title: "1. Bản chất dữ liệu LiDAR Point Cloud",
          content: "Cảm biến LiDAR bắn hàng trăm nghìn tia laser mỗi giây và đo thời gian bay (Time-of-Flight) để trả về danh sách các điểm (x, y, z, intensity). Khác với ảnh 2D, đám mây điểm có tọa độ khoảng cách mét thực tế tuyệt đối nhưng không có màu sắc chi tiết.",
          bullets: [
            "3D Bounding Box (Cuboid): Cần định nghĩa chính xác 7 tham số: Tọa độ tâm (center_x, center_y, center_z), Kích thước (length, width, height) và Góc xoay quanh trục thẳng đứng (Yaw / Heading angle).",
            "Mặt trước vật thể (Front indicator): Cuboid bắt buộc phải có véc-tơ chỉ rõ đâu là đầu xe/mặt trước của người đi bộ để thuật toán dự báo hướng chuyển động trong tương lai."
          ]
        },
        {
          title: "2. Thách thức trong gán nhãn 3D",
          content: "Gán nhãn đám mây điểm đòi hỏi quan sát đồng thời trên 3 khung chiếu trực giao: Top View (BEV), Side View, và Front View:",
          bullets: [
            "Vật thể ở xa: Càng xa cảm biến, mật độ tia laser càng thưa (Sparsity). Một chiếc ô tô ở cự ly 60m có thể chỉ thu được 5-10 điểm phản xạ, người gán nhãn phải dùng kích thước xe chuẩn (Prior template) để khớp hộp.",
            "Mặt đất (Ground plane): Đáy hộp cuboid phải đặt sát mặt đất, không được cắm xuống lòng đất hoặc lơ lửng trên không trung."
          ]
        }
      ],
      examTrap: "Góc quay của xe trong hệ tọa độ phẳng mặt đất là góc Yaw (xoay quanh trục Z hướng lên trời). Pitch và Roll của xe thường bằng 0 trừ khi xe đang leo dốc nghiêng!"
    },
    {
      id: "D13",
      dayNumber: 13,
      title: "Robotaxi A — LiDAR 3D Object & Sensor Fusion (Hợp nhất Cảm biến)",
      tag: "NuScenes & Fusion",
      slidesCount: 1,
      summary: "Tập dữ liệu chuẩn công nghiệp NuScenes / Waymo, hợp nhất đa cảm biến (Camera + LiDAR Fusion), ma trận biến đổi tọa độ Sensor-to-Ego-to-World.",
      sections: [
        {
          title: "1. Cấu trúc tập dữ liệu NuScenes",
          content: "NuScenes là bộ dữ liệu chuẩn mực nhất cho xe tự hành gồm 1 xe thử nghiệm trang bị 1 LiDAR 32 tia, 5 Radar và 6 Camera 360 độ:",
          bullets: [
            "Cấu trúc bảng quan hệ (Relational Schema): Gồm scene → sample → sample_data → sample_annotation → instance.",
            "Sample: Một điểm thời gian được đồng bộ giữa LiDAR và các camera (tần số 2Hz cho nhãn nhắm).",
            "Instance: Thực thể vật thể duy nhất xuyên suốt cả một video (scene)."
          ]
        },
        {
          title: "2. Sensor Fusion & Phép chiếu 3D sang 2D",
          content: "Chiếu đám mây điểm 3D lên ảnh camera 2D để kiểm chứng độ khớp của hộp nhãn:",
          bullets: [
            "Công thức biến đổi: P_cam = K * (R * P_lidar + T).",
            "K là ma trận nội suy camera (Camera Intrinsic: tiêu cự fx, fy, quang tâm cx, cy).",
            "[R | T] là ma trận ngoại suy hiệu chuẩn giữa tâm LiDAR và tâm Camera (Extrinsic calibration).",
            "Khi nhãn 3D chính xác, phép chiếu 8 đỉnh của hộp 3D xuống ảnh 2D phải ôm trọn khít vật thể trên ảnh camera."
          ]
        }
      ],
      examTrap: "Nếu hộp 3D khớp đẹp trên điểm LiDAR nhưng khi chiếu lên ảnh 2D bị lệch vị trí: 99% nguyên nhân là do ma trận hiệu chuẩn ngoại suy (Extrinsics) bị sai hoặc lỗi đồng bộ thời gian (Time synchronization lag) giữa Camera và LiDAR!"
    },
    {
      id: "D14",
      dayNumber: 14,
      title: "Robotaxi B — Across Time (Tích lũy Đám mây điểm Đa khung hình)",
      tag: "Multi-Frame LiDAR",
      slidesCount: 1,
      summary: "Tích lũy đám mây điểm qua thời gian (Multi-sweep / Point accumulation), chuyển động bản thân xe (Ego-motion compensation) bằng Odometry/IMU, Scene Flow và bài toán vật thể di động.",
      sections: [
        {
          title: "1. Tại sao cần tích lũy đám mây điểm (Multi-sweep)?",
          content: "Một vòng quét đơn lẻ (1 sweep = 50ms) của LiDAR cho điểm rất thưa, đặc biệt với vật thể xa. Để nhìn rõ cấu trúc hình học, người ta tích lũy 10 đến 20 sweep liên tiếp (khoảng 0.5 - 1.0 giây).",
          bullets: [
            "Bù chuyển động xe tự hành (Ego-Motion Compensation): Vì xe tự hành đang di chuyển trong lúc quét, các điểm quét ở thời điểm t1 và t2 phải được chuyển về cùng một hệ tọa độ quy chiếu chung thông qua dữ liệu IMU/GPS định vị (Pose Odometry).",
            "Vật thể tĩnh (Static Background): Khi bù ego-motion đúng, toàn bộ nhà cửa, mặt đường, cây cối giữa các frame sẽ trùng khít tuyệt đối và trở nên đặc quánh, sắc nét."
          ]
        },
        {
          title: "2. Vật thể di động & Vệt bóng mờ (Ghosting Artifact)",
          content: "Đối với các xe ô tô hoặc người đi bộ đang di chuyển trên đường:",
          bullets: [
            "Hiện tượng Ghosting / Smearing: Khi gom 10 khung hình lại, vật thể đang chạy sẽ để lại một vệt bóng mờ kéo dài như đoàn tàu.",
            "Xử lý nhãn: Người gán nhãn phải tính toán véc-tơ vận tốc (Velocity vector / Scene Flow) của từng vật thể để dịch chuyển hộp 3D theo đúng vị trí của từng khung quét thành phần.",
            "Dự báo quỹ đạo (Trajectory Prediction): Nhãn đa khung thời gian là đầu vào để xe tự hành học cách đoán trước xe đối diện sẽ rẽ trái hay đi thẳng trong 3 giây tiếp theo."
          ]
        }
      ],
      examTrap: "Nếu chỉ bù chuyển động của xe mình (Ego-motion) mà không tính vận tốc của xe đối diện, vệt điểm của xe đối diện sẽ bị nhòe dài (ghost trail) trên đám mây điểm tích lũy!"
    }
  ],

  // ========================================================
  // NGÂN HÀNG ĐỀ THI TRẮC NGHIỆM VLEARN (60 CÂU HỎI THỰC CHIẾN)
  // ========================================================
  questions: [
    {
      id: 1,
      day: "D01",
      question: "Trong quy chuẩn nhãn COCO format, một bounding box 2D được định dạng theo cấu trúc nào sau đây?",
      options: [
        "[x_min, y_min, x_max, y_max]",
        "[x_min, y_min, width, height]",
        "[x_center, y_center, width, height] (chuẩn hóa 0-1)",
        "[x1, y1, x2, y2, x3, y3, x4, y4]"
      ],
      correct: 1,
      explanation: "COCO format sử dụng tọa độ góc trên bên trái [x_min, y_min] kết hợp với chiều rộng width và chiều cao height tính theo pixel thực tế. Định dạng [x_min, y_min, x_max, y_max] thuộc về PASCAL VOC, còn [x_center, y_center, w, h] chuẩn hóa thuộc về YOLO."
    },
    {
      id: 2,
      day: "D01",
      question: "Hiện tượng 'Data Leakage' (rò rỉ dữ liệu) phổ biến nhất khi chia tập Train/Val/Test cho bài toán video camera là gì?",
      options: [
        "Sử dụng ảnh có độ phân giải quá cao trong tập Test",
        "Để các frame liền kề nhau từ cùng một video clip rơi vào cả tập Train và tập Test",
        "Số lượng ảnh trong tập Test nhiều hơn tập Train",
        "Không làm mờ biển số xe trước khi huấn luyện"
      ],
      correct: 1,
      explanation: "Trong video, hai khung hình liên tiếp nhau (cách nhau 1/30 giây) gần như giống hệt nhau. Nếu chia ngẫu nhiên từng frame, tập Train và Test sẽ chứa các frame gần như trùng khớp, khiến mô hình chỉ ghi nhớ (overfit) thay vì tổng quát hóa. Phải chia tập theo toàn bộ clip/scene độc lập."
    },
    {
      id: 3,
      day: "D02",
      question: "Cho 2 bounding box A và B có diện tích lần lượt là 100 và 120. Diện tích phần giao nhau (Intersection) là 40. Giá trị IoU của 2 box này là bao nhiêu?",
      options: [
        "0.333",
        "0.222",
        "0.400",
        "0.500"
      ],
      correct: 1,
      explanation: "Diện tích phần hợp (Union) = Area(A) + Area(B) - Area(Intersection) = 100 + 120 - 40 = 180. IoU = Area(Intersection) / Area(Union) = 40 / 180 ≈ 0.222."
    },
    {
      id: 4,
      day: "D02",
      question: "Thuật toán NMS (Non-Maximum Suppression) trong Object Detection được sử dụng để làm gì?",
      options: [
        "Tăng độ sáng của ảnh ban đêm",
        "Loại bỏ các bounding box trùng lặp dự đoán cùng một đối tượng và giữ lại box có confidence cao nhất",
        "Dự đoán góc quay 3D của ô tô",
        "Tự động vẽ mask phân vùng cho vật thể"
      ],
      correct: 1,
      explanation: "Mô hình phát hiện vật thể thường sinh ra nhiều box chồng lấn lên cùng một đối tượng. NMS sắp xếp các box theo điểm số tự tin (confidence score), giữ box cao nhất và loại bỏ các box xung quanh có chỉ số IoU vượt ngưỡng trùng lặp."
    },
    {
      id: 5,
      day: "D02",
      question: "Nếu một nhãn chuẩn (Ground Truth) có một chiếc ô tô, nhưng mô hình phát hiện ra ô tô đó với chỉ số IoU = 0.42 (trong khi ngưỡng yêu cầu là IoU >= 0.5). Kết quả đánh giá này sẽ ghi nhận như thế nào?",
      options: [
        "Ghi nhận 1 True Positive",
        "Ghi nhận 1 False Positive và 1 False Negative",
        "Chỉ ghi nhận 1 True Negative",
        "Bỏ qua không tính điểm câu này"
      ],
      correct: 1,
      explanation: "Vì IoU < 0.5, box dự đoán bị coi là đoán sai vị trí nên tính là 1 False Positive (FP). Đồng thời, vật thể thật trên ảnh không có box nào đạt chuẩn che phủ nên bị tính là bỏ sót: 1 False Negative (FN)."
    },
    {
      id: 6,
      day: "D03",
      question: "Điểm khác biệt cốt lõi giữa thuật toán theo dõi vật thể DeepSORT so với SORT truyền thống là gì?",
      options: [
        "DeepSORT chạy nhanh hơn SORT gấp 10 lần",
        "DeepSORT bổ sung mạng trích xuất đặc trưng ngoại hình (Appearance Embeddings) để hỗ trợ Re-Identification khi bị che khuất",
        "DeepSORT không cần dùng Kalman Filter",
        "DeepSORT chỉ dùng cho ảnh đen trắng"
      ],
      correct: 1,
      explanation: "SORT chỉ dùng khoảng cách không gian (IoU và Kalman Filter), do đó khi vật thể bị che khuất lâu hoặc chuyển động bất thường sẽ bị mất dấu hoặc văng ID. DeepSORT trích xuất vector ngoại hình (deep visual feature) để nhận diện lại đối tượng cũ khi xuất hiện trở lại."
    },
    {
      id: 7,
      day: "D03",
      question: "Trong bài toán Multi-Object Tracking, đại lượng 'ID Switch' (IDSW) phản ánh điều gì?",
      options: [
        "Vật thể thay đổi kích thước đột ngột",
        "Track ID của một đối tượng bị nhảy sang một ID khác hoặc hoán đổi cho đối tượng bên cạnh",
        "Xe tự hành chuyển sang chế độ lái tự động",
        "Camera đổi góc nhìn"
      ],
      correct: 1,
      explanation: "ID Switch xảy ra khi hệ thống theo dõi gán nhầm định danh của đối tượng (ví dụ: người đi bộ mang ID 5 sau khi đi qua cột đèn bị đổi thành ID 12, hoặc tráo đổi với người đi ngược chiều)."
    },
    {
      id: 8,
      day: "D04",
      question: "Trong định dạng nhãn Keypoint của bộ dữ liệu COCO, giá trị cờ hiển thị v = 1 đại diện cho trạng thái nào?",
      options: [
        "Khớp xương nhìn thấy rõ ràng trên ảnh",
        "Khớp xương hoàn toàn không tồn tại trên cơ thể",
        "Khớp xương bị che khuất nhưng người gán nhãn dự đoán được tọa độ giải phẫu",
        "Ảnh bị mờ không thể nhận diện"
      ],
      correct: 2,
      explanation: "Quy ước COCO Keypoint: v=0 (không được gán / ngoài ảnh), v=1 (bị che khuất nhưng suy luận được vị trí), v=2 (nhìn thấy rõ ràng)."
    },
    {
      id: 9,
      day: "D04",
      question: "Chỉ số OKS (Object Keypoint Similarity) sử dụng yếu tố nào để dung sai cho sai số định vị giữa các khớp xương khác nhau?",
      options: [
        "Màu sắc trang phục",
        "Hệ số sai số cho phép k_i đặc thù cho từng khớp xương (ví dụ: khớp vai k_i lớn hơn điểm mắt)",
        "Tốc độ di chuyển của người",
        "Độ phân giải của camera"
      ],
      correct: 1,
      explanation: "Trong công thức OKS, hệ số k_i kiểm soát độ biến thiên tiêu chuẩn của từng khớp cơ thể người. Các khớp khó xác định chính xác như hông, vai có k_i lớn hơn mắt, mũi."
    },
    {
      id: 10,
      day: "D05",
      question: "Phát biểu nào sau đây phân biệt ĐÚNG NHẤT giữa Semantic Segmentation và Instance Segmentation?",
      options: [
        "Semantic segmentation phân biệt từng xe riêng lẻ, Instance gộp chung tất cả các xe",
        "Semantic segmentation gán cùng nhãn cho tất cả pixel thuộc cùng lớp (không phân biệt cá thể), còn Instance segmentation tách biệt từng cá thể đối tượng",
        "Semantic segmentation chỉ chạy trên ảnh 3D, Instance chỉ chạy trên ảnh 2D",
        "Cả hai hoàn toàn giống nhau không có điểm khác biệt"
      ],
      correct: 1,
      explanation: "Semantic Segmentation phân loại từng pixel (ví dụ: mọi pixel ô tô đều tô màu xanh). Instance Segmentation chỉ quan tâm đến các vật thể đếm được (things) và phân biệt riêng Car 1, Car 2, Car 3."
    },
    {
      id: 11,
      day: "D05",
      question: "Kỹ thuật mã hóa Run-Length Encoding (RLE) mang lại lợi ích gì lớn nhất cho tập dữ liệu Phân vùng (Segmentation)?",
      options: [
        "Tăng độ sáng của ảnh",
        "Nén kích thước lưu trữ ma trận mặt nạ nhị phân xuống gấp nhiều lần",
        "Tự động sửa lỗi nhãn gán sai",
        "Chuyển đổi ảnh 2D thành đám mây điểm 3D"
      ],
      correct: 1,
      explanation: "RLE ghi lại chuỗi các pixel liên tục cùng giá trị (0 hoặc 1) thay vì lưu từng pixel đơn lẻ trong ma trận HxW, giúp giảm dung lượng file JSON lưu nhãn mask đáng kể."
    },
    {
      id: 12,
      day: "D05",
      question: "Nếu hai vùng phân vùng có chỉ số IoU = 0.6. Giá trị Dice Coefficient tương ứng sẽ là bao nhiêu?",
      options: [
        "0.600",
        "0.750",
        "0.500",
        "0.850"
      ],
      correct: 1,
      explanation: "Mối liên hệ giữa Dice và IoU là: Dice = (2 * IoU) / (1 + IoU). Thay số: Dice = (2 * 0.6) / (1 + 0.6) = 1.2 / 1.6 = 0.75 (hay 75%)."
    },
    {
      id: 13,
      day: "D06",
      question: "Để đo lường mức độ đồng thuận giữa nhiều người gán nhãn độc lập (Inter-Annotator Agreement) và loại trừ yếu tố trùng khớp ngẫu nhiên, chỉ số nào sau đây được ưu tiên sử dụng?",
      options: [
        "Accuracy thuần",
        "Fleiss' Kappa (hoặc Cohen's Kappa)",
        "Mean Squared Error",
        "Loss Entropy"
      ],
      correct: 1,
      explanation: "Hệ số Cohen's Kappa (cho 2 người) và Fleiss' Kappa (cho nhiều người) là tiêu chuẩn vàng trong đo lường độ đồng thuận gán nhãn vì nó trừ đi xác suất trùng hợp ngẫu nhiên (chance agreement)."
    },
    {
      id: 14,
      day: "D06",
      question: "Nhiệm vụ 'Honeypot' (hoặc Gold Standard Task) trong quy trình kiểm soát chất lượng dữ liệu QA/QC có vai trò gì?",
      options: [
        "Tự động làm đẹp ảnh",
        "Trộn các tác vụ đã biết trước đáp án chuẩn vào luồng công việc để đánh giá độ chính xác thực tế của nhân sự gán nhãn",
        "Khóa tài khoản của người dùng khi làm bài quá nhanh",
        "Nén dung lượng ảnh"
      ],
      correct: 1,
      explanation: "Honeypot là các câu hỏi chuẩn đã có đáp án chuyên gia được cài ngầm vào hệ thống. Nếu người gán nhãn làm sai các tác vụ này, hệ thống sẽ cảnh báo hoặc ngừng phân phối việc vì chất lượng không đảm bảo."
    },
    {
      id: 15,
      day: "D07",
      question: "Công cụ DVC (Data Version Control) quản lý các tệp dữ liệu hình ảnh dung lượng lớn bằng cách nào?",
      options: [
        "Lưu toàn bộ ảnh trực tiếp vào cơ sở dữ liệu MySQL",
        "Commit file mã hash (.dvc) siêu nhỏ vào Git repository, đồng thời đồng bộ file nhị phân nặng lên Cloud Remote Storage",
        "Nén toàn bộ ảnh thành file zip rồi gửi qua email",
        "Xóa bớt một nửa số lượng ảnh"
      ],
      correct: 1,
      explanation: "DVC tạo các file .dvc chứa checksum MD5 để Git quản lý phiên bản nhẹ nhàng, trong khi các file dữ liệu nặng thật sự được lưu ở remote storage như AWS S3, Google Cloud Storage hoặc MinIO."
    },
    {
      id: 16,
      day: "D08",
      question: "Trong quy trình Active Learning, chiến lược 'Uncertainty Sampling' ưu tiên chọn mẫu dữ liệu nào để gửi cho con người gán nhãn?",
      options: [
        "Các mẫu mà mô hình hiện tại dự đoán tự tin 99.9%",
        "Các mẫu mà mô hình hiện tại dự đoán với độ phân vân/không chắc chắn cao nhất (độ tự tin thấp)",
        "Các mẫu ngẫu nhiên không cần điều kiện",
        "Các ảnh chỉ chụp bầu trời"
      ],
      correct: 1,
      explanation: "Mục tiêu của Active Learning là tối ưu chi phí gán nhãn. Việc gán các mẫu mô hình đã nắm chắc không mang lại kiến thức mới, do đó hệ thống ưu tiên chọn các mẫu mô hình đang phân vân nhiều nhất (entropy cao) để con người hỗ trợ."
    },
    {
      id: 17,
      day: "D09",
      question: "Tại sao vạch kẻ đường (Lane markings) trong bài toán xe tự hành thường được gán nhãn dạng Polyline/Spline thay vì Bounding Box 2D?",
      options: [
        "Vì Bounding box vẽ quá nhanh",
        "Vì vạch kẻ đường uốn lượn cong theo không gian và chứa nhiều diện tích trống; Polyline phản ánh trung thực hình học dẫn đường cho thuật toán bám làn",
        "Vì Bounding box không thể tô màu đỏ",
        "Vì camera xe tự hành chỉ đọc được đường thẳng"
      ],
      correct: 1,
      explanation: "Vạch kẻ làn đường là các đường mảnh kéo dài uốn lượn. Nếu dùng Bounding box bao quanh, hộp sẽ chứa 95% diện tích mặt đường không liên quan và không thể hiện được độ cong chính xác để xe bám làn (steering control)."
    },
    {
      id: 18,
      day: "D10",
      question: "Camera giám sát người lái (DMS) trong xe hơi thường sử dụng loại cảm biến ánh sáng nào?",
      options: [
        "Camera RGB thông thường không có đèn",
        "Camera hồng ngoại gần (Near-Infrared / NIR 940nm) kết hợp đèn LED hồng ngoại",
        "Camera tia X",
        "Camera siêu âm"
      ],
      correct: 1,
      explanation: "Camera hồng ngoại gần (NIR 940nm) cho phép quan sát rõ khuôn mặt và đồng tử mắt của tài xế cả ban đêm mà mắt người không nhìn thấy ánh sáng đèn chiếu, tránh làm chói mắt nguy hiểm khi lái xe."
    },
    {
      id: 19,
      day: "D10",
      question: "Chỉ số PERCLOS trong hệ thống DMS dùng để đo lường điều gì?",
      options: [
        "Tốc độ xe chạy trên cao tốc",
        "Tỉ lệ phần trăm thời gian mí mắt nhắm lại quá 80% trong một khoảng thời gian quan sát (dấu hiệu buồn ngủ)",
        "Khoảng cách giữa hai xe",
        "Nhiệt độ bên trong khoang lái"
      ],
      correct: 1,
      explanation: "PERCLOS (Percentage of Eye Closure) là tiêu chuẩn y khoa và kỹ thuật phổ biến nhất để đo mức độ buồn ngủ của tài xế. Khi PERCLOS vượt ngưỡng 80% trong chu kỳ theo dõi, hệ thống sẽ phát âm thanh cảnh báo nguy hiểm."
    },
    {
      id: 20,
      day: "D11",
      question: "Hệ thống Camera Toàn cảnh SVM-360 View tạo ra góc nhìn Bird's Eye View (BEV) dựa trên nguyên lý quang học nào?",
      options: [
        "Chụp ảnh từ flycam trên nóc xe",
        "Sử dụng 4 camera mắt cá (Fisheye), khử méo ống kính và áp dụng phép biến đổi Homography chiếu phẳng xuống mặt đất kết hợp ghép ảnh (Stitching)",
        "Dùng gương cầu lồi lắp quanh xe",
        "Chiếu laser quét toàn cảnh"
      ],
      correct: 1,
      explanation: "SVM sử dụng 4 camera góc siêu rộng (trước, sau, hai gương), khử méo quang học bằng mô hình thấu kính, dùng Homography chiếu phẳng xuống tọa độ mặt đất và ghép vùng viền (stitching) để tạo ảnh giả lập mắt chim nhìn từ trên cao."
    },
    {
      id: 21,
      day: "D12",
      question: "Một hộp gán nhãn 3D Bounding Box (Cuboid) tiêu chuẩn cho đám mây điểm LiDAR yêu cầu tối thiểu bao nhiêu bậc tự do (DoF)?",
      options: [
        "4 tham số (x, y, w, h)",
        "7 tham số: Tọa độ tâm (x, y, z), kích thước (dx, dy, dz) và góc xoay hướng (Yaw)",
        "2 tham số (khoảng cách và độ cao)",
        "12 tham số tự do"
      ],
      correct: 1,
      explanation: "Trong không gian 3D mặt đường, một hộp bao 3D được xác định bởi: 3 tham số vị trí tâm (x, y, z), 3 tham số kích thước (length, width, height) và 1 góc xoay chính quanh trục thẳng đứng (Yaw angle / Heading), tổng cộng là 7 bậc tự do (7 DoF)."
    },
    {
      id: 22,
      day: "D12",
      question: "Tại sao việc xác định 'Heading Angle' (hướng đầu xe) trong hộp gán nhãn 3D lại mang tính sống còn đối với xe tự hành?",
      options: [
        "Để xe biết bật đèn pha",
        "Để thuật toán dự đoán quỹ đạo (Trajectory Planning) biết hướng chuyển động trong tương lai của phương tiện nhằm phanh hoặc né tránh kịp thời",
        "Để tính toán giá trị của chiếc xe",
        "Để chụp ảnh phạt nguội"
      ],
      correct: 1,
      explanation: "Xe ô tô di chuyển theo hướng đầu xe. Nếu hộp 3D bị quay ngược 180 độ (nhầm đầu xe thành đuôi xe), hệ thống tự hành sẽ dự báo xe đó lùi lại thay vì tiến tới, dẫn tới ra quyết định xử lý thảm họa."
    },
    {
      id: 23,
      day: "D13",
      question: "Trong tập dữ liệu NuScenes, đối tượng 'Sample' được định nghĩa là gì?",
      options: [
        "Toàn bộ một video clip dài 20 giây",
        "Một mốc thời gian (keyframe) được đồng bộ hóa dữ liệu cảm biến (LiDAR + Camera) có gán nhãn 3D đầy đủ ở tần số 2Hz",
        "Một bức ảnh bị hỏng cần xóa bỏ",
        "Một cá thể xe ô tô cụ thể"
      ],
      correct: 1,
      explanation: "Trong kiến trúc NuScenes, một Scene kéo dài 20s gồm nhiều Sample. Mỗi Sample là một điểm thời gian được trích xuất (tần số 2Hz) mà tại đó tất cả cảm biến (LiDAR, Camera, Radar) được đồng bộ chính xác và có nhãn 3D đầy đủ."
    },
    {
      id: 24,
      day: "D13",
      question: "Để chiếu một điểm 3D P_lidar từ hệ tọa độ LiDAR sang tọa độ điểm ảnh 2D trên Camera, ta cần áp dụng công thức nào?",
      options: [
        "P_pixel = P_lidar / 100",
        "P_pixel = K * [R | T] * P_lidar (trong đó K là ma trận nội suy Intrinsic, [R|T] là ma trận ngoại suy Extrinsic)",
        "P_pixel = P_lidar * 3.14",
        "P_pixel = K + P_lidar"
      ],
      correct: 1,
      explanation: "Công thức chiếu hình học Pin-hole chuẩn: Điểm 3D LiDAR được chuyển sang hệ tọa độ Camera qua phép biến đổi cứng Extrinsic [R|T], sau đó nhân với ma trận nội suy Camera Intrinsic K để ra tọa độ điểm ảnh 2D (u, v)."
    },
    {
      id: 25,
      day: "D14",
      question: "Hiện tượng vệt bóng mờ kéo dài (Ghosting / Smearing artifact) xuất hiện khi tích lũy đa khung hình LiDAR (Multi-sweep) là do nguyên nhân nào?",
      options: [
        "Do thời tiết sương mù làm tia laser bị tán xạ",
        "Do các đối tượng trên đường đang chuyển động trong khi các khung quét LiDAR cách nhau một khoảng thời gian",
        "Do cảm biến LiDAR bị hỏng động cơ xoay",
        "Do camera chụp ảnh bị rung tay"
      ],
      correct: 1,
      explanation: "Khi gom nhiều vòng quét LiDAR qua các thời điểm t1, t2, t3,... nếu xe khác đang chạy, các điểm laser ở mỗi vòng quét sẽ nằm ở các vị trí dịch chuyển khác nhau, tạo thành vệt điểm kéo dài như đoàn tàu (ghost trail)."
    },
    {
      id: 26,
      day: "D14",
      question: "Khi thực hiện tích lũy điểm LiDAR đa khung (Multi-frame accumulation), 'Bù chuyển động xe tự hành' (Ego-motion compensation) sử dụng dữ liệu từ nguồn nào?",
      options: [
        "Dữ liệu bản đồ Google Maps vệ tinh",
        "Dữ liệu định vị từ hệ thống IMU / GPS / Odometry đo chuyển động thực tế của chính xe tự hành",
        "Dữ liệu từ cảm biến áp suất lốp",
        "Dữ liệu âm thanh còi xe"
      ],
      correct: 1,
      explanation: "Ego-motion compensation dùng ma trận biến đổi tọa độ xe (Vehicle Pose) ghi nhận từ IMU/GPS ở từng thời điểm để xoay và tịnh tiến các đám mây điểm về cùng một hệ tọa độ quy chiếu cố định chung."
    },
    {
      id: 27,
      day: "D02",
      question: "Khi huấn luyện mô hình phát hiện người đi bộ, nếu người gán nhãn bỏ sót một người trong ảnh, chỉ số nào sẽ bị sụt giảm?",
      options: [
        "Chỉ số Recall",
        "Chỉ số Precision",
        "Chỉ số FPS (tốc độ khung hình)",
        "Độ phân giải của ảnh"
      ],
      correct: 0,
      explanation: "Bỏ sót đối tượng dẫn tới tăng False Negative (FN). Công thức Recall = TP / (TP + FN), do đó khi FN tăng thì Recall sẽ giảm trực tiếp."
    },
    {
      id: 28,
      day: "D06",
      question: "Hệ số Fleiss' Kappa bằng 0.25 thể hiện điều gì về chất lượng của bộ nhãn và tài liệu hướng dẫn gán nhãn?",
      options: [
        "Đồng thuận hoàn hảo tuyệt đối",
        "Đồng thuận kém (Poor agreement), tài liệu hướng dẫn còn nhiều mâu thuẫn hoặc mơ hồ khiến các annotator gán theo ý chủ quan",
        "Dữ liệu sẵn sàng đưa vào thương mại hóa",
        "Mô hình đạt độ chính xác 99%"
      ],
      correct: 1,
      explanation: "Fleiss' Kappa dưới 0.40 được phân loại là Poor / Slight Agreement. Điều này cảnh báo guideline chưa rõ ràng, cần họp lại đội ngũ để chuẩn hóa tiêu chuẩn gán."
    },
    {
      id: 29,
      day: "D12",
      question: "Đối với vật thể nằm cách xa cảm biến LiDAR (> 50m), khó khăn lớn nhất của người gán nhãn 3D là gì?",
      options: [
        "Vật thể bị phóng to quá mức",
        "Đám mây điểm rất thưa thớt (sparsity) chỉ có vài điểm, rất khó xác định chính xác kích thước và góc quay (heading)",
        "Điểm LiDAR bị đổi màu sang màu đỏ",
        "Không thể lưu file"
      ],
      correct: 1,
      explanation: "Do chùm tia laser phân kỳ theo khoảng cách, các vật thể ở xa nhận được rất ít tia phản xạ (chỉ vài chấm rời rạc). Annotator phải kết hợp nhìn ảnh camera 2D và kích thước xe chuẩn để căn chỉnh hộp 3D."
    },
    {
      id: 30,
      day: "D11",
      question: "Hạn chế vật lý lớn nhất của ảnh Bird's Eye View (BEV) được tạo ra từ 4 camera SVM-360 là gì?",
      options: [
        "Không thể hiển thị vạch kẻ đường",
        "Các vật thể có chiều cao nhô lên khỏi mặt đất (như xe bên cạnh, người đi bộ) sẽ bị méo mó kéo dài (stretching artifact) vì phép chiếu Homography giả định toàn bộ mặt phẳng nằm trên mặt đường",
        "Chỉ hiển thị được màu đen trắng",
        "Gây chập điện ắc quy xe"
      ],
      correct: 1,
      explanation: "Homography giả lập phép chiếu từ phẳng sang phẳng (Planar assumption). Bất kỳ điểm nào có độ cao Z > 0 sẽ bị kéo dãn hình học tạo thành hình ảnh nghiêng ngả biến dạng đặc trưng trên màn hình BEV."
    }
  ],

  // ========================================================
  // BỘ FLASHCARDS ÔN NHANH (36 THẺ THUẬT NGỮ & CÔNG THỨC)
  // ========================================================
  flashcards: [
    {
      id: 1,
      day: "Day 1",
      term: "GIGO (Garbage In, Garbage Out)",
      definition: "Nguyên lý cốt lõi của Machine Learning: Dữ liệu nhãn huấn luyện sai lệch hoặc kém chất lượng sẽ tạo ra mô hình dự đoán sai lệch, bất kể kiến trúc mô hình hiện đại đến đâu.",
      formula: "Data Quality > Model Complexity"
    },
    {
      id: 2,
      day: "Day 2",
      term: "IoU (Intersection over Union)",
      definition: "Tỉ lệ giữa diện tích phần giao nhau và diện tích phần hợp nhất của hai bounding box. Ngưỡng IoU >= 0.5 là ranh giới chuẩn xác định True Positive.",
      formula: "IoU = Area(A ∩ B) / Area(A ∪ B)"
    },
    {
      id: 3,
      day: "Day 2",
      term: "Precision vs Recall",
      definition: "Precision đo tỉ lệ dự đoán đúng trong các mẫu mô hình báo Dương; Recall đo tỉ lệ phát hiện được trong toàn bộ mẫu Dương thực tế ngoài đời.",
      formula: "Precision = TP/(TP+FP) | Recall = TP/(TP+FN)"
    },
    {
      id: 4,
      day: "Day 2",
      term: "NMS (Non-Maximum Suppression)",
      definition: "Thuật toán lọc bỏ các bounding box trùng lặp cùng dự đoán một đối tượng, giữ lại box có điểm tin cậy cao nhất và xóa các box lân cận có IoU vượt ngưỡng.",
      formula: "IoU(Box_keep, Box_i) > NMS_Threshold -> Remove"
    },
    {
      id: 5,
      day: "Day 3",
      term: "MOTA (Multi-Object Tracking Accuracy)",
      definition: "Chỉ số đánh giá độ chính xác tổng thể của thuật toán theo dõi video, trừ điểm phạt dựa trên số lỗi bỏ sót (FN), dự đoán nhầm (FP) và nhảy định danh (ID Switch).",
      formula: "MOTA = 1 - ∑(FN + FP + IDSW) / ∑ GT"
    },
    {
      id: 6,
      day: "Day 3",
      term: "Hungarian Algorithm",
      definition: "Thuật toán giải bài toán phân công ghép cặp tối ưu (Bipartite matching) giữa các box phát hiện ở frame hiện tại với các tracklet dự đoán từ Kalman Filter.",
      formula: "Cost Matrix = 1 - IoU_Distance (or Cosine_Dist)"
    },
    {
      id: 7,
      day: "Day 4",
      term: "OKS (Object Keypoint Similarity)",
      definition: "Độ đo độ tương đồng giữa các điểm mấu chốt khung xương người dự đoán so với nhãn chuẩn, có xét đến tỉ lệ cơ thể và dung sai riêng của từng khớp.",
      formula: "OKS = ∑ exp(-d_i² / 2s²k_i²) * v_i / ∑ v_i"
    },
    {
      id: 8,
      day: "Day 5",
      term: "Dice Coefficient (F1 Score Mask)",
      definition: "Chỉ số đo độ trùng khớp giữa hai vùng phân vùng nhị phân. Khi IoU = 0.5, Dice Coefficient tương ứng là 0.667.",
      formula: "Dice = 2 * |A ∩ B| / (|A| + |B|) = 2*IoU / (1+IoU)"
    },
    {
      id: 9,
      day: "Day 5",
      term: "RLE (Run-Length Encoding)",
      definition: "Kỹ thuật nén ma trận mặt nạ phân vùng nhị phân bằng cách ghi lại độ dài chuỗi pixel 0 và 1 liên tiếp, giúp tối ưu dung lượng file JSON lưu nhãn.",
      formula: "Counts: [start1, len1, start2, len2,...]"
    },
    {
      id: 10,
      day: "Day 6",
      term: "Fleiss' Kappa",
      definition: "Hệ số thống kê đo lường mức độ đồng thuận gán nhãn giữa nhiều kiểm định viên độc lập sau khi đã loại trừ xác suất trùng khớp ngẫu nhiên. Ngưỡng >= 0.8 là xuất sắc.",
      formula: "Kappa = (P_observed - P_chance) / (1 - P_chance)"
    },
    {
      id: 11,
      day: "Day 7",
      term: "DVC (Data Version Control)",
      definition: "Công cụ quản lý phiên bản dữ liệu lớn tích hợp với Git: Chỉ lưu file con trỏ text mã hash trong Git, trong khi dữ liệu nhị phân nặng được đẩy lên Cloud Storage.",
      formula: ".dvc pointer -> S3 / GCS / MinIO"
    },
    {
      id: 12,
      day: "Day 8",
      term: "Active Learning - Uncertainty Sampling",
      definition: "Chiến lược lựa chọn dữ liệu tự động gửi cho con người gán nhãn dựa trên các mẫu mà mô hình hiện tại đang có độ tự tin thấp nhất hoặc độ hỗn loạn cao nhất.",
      formula: "Query: argmax_x Entropy(P(y|x))"
    },
    {
      id: 13,
      day: "Day 10",
      term: "PERCLOS (Percentage of Eye Closure)",
      definition: "Tỉ lệ phần trăm thời gian mí mắt tài xế nhắm lại quá 80% trong một cửa sổ thời gian quan sát. Là tiêu chuẩn quốc tế số 1 để cảnh báo ngủ gật.",
      formula: "PERCLOS = (Time_Eyes_Closed_80% / Total_Window) * 100%"
    },
    {
      id: 14,
      day: "Day 11",
      term: "Homography Matrix (SVM-360)",
      definition: "Ma trận biến đổi xạ ảnh 3x3 dùng để chiếu hình ảnh từ các camera mắt cá xuống mặt phẳng mặt đường để tạo ra góc nhìn trên cao Bird's Eye View (BEV).",
      formula: "x_bev ~ H * x_cam"
    },
    {
      id: 15,
      day: "Day 12",
      term: "3D Cuboid Bounding Box (7 DoF)",
      definition: "Hộp bao không gian 3 chiều xác định vị trí và kích thước vật thể trong đám mây điểm LiDAR, bao gồm tâm 3D, kích thước bao và góc xoay hướng Yaw.",
      formula: "[x, y, z, length, width, height, yaw]"
    },
    {
      id: 16,
      day: "Day 13",
      term: "Sensor Fusion (LiDAR to Camera)",
      definition: "Phép chiếu hình học biến đổi tọa độ điểm 3D từ LiDAR sang tọa độ pixel 2D trên ảnh camera thông qua ma trận hiệu chuẩn Intrinsic K và Extrinsic [R|T].",
      formula: "P_pixel = K * (R * P_lidar + T)"
    },
    {
      id: 17,
      day: "Day 14",
      term: "Ego-Motion Compensation (LiDAR)",
      definition: "Kỹ thuật hiệu chỉnh tọa độ các vòng quét LiDAR theo thời gian bằng cách sử dụng dữ liệu định vị Pose từ IMU/GPS để bù trừ chuyển động của chính xe tự hành.",
      formula: "P_world = T_ego_to_world * P_ego"
    },
    {
      id: 18,
      day: "Day 14",
      term: "Ghosting / Smearing Artifact",
      definition: "Vệt bóng mờ kéo dài của các vật thể di động (như xe khác chạy vượt qua) khi tích lũy nhiều khung quét LiDAR liên tiếp mà chưa xử lý bù véc-tơ vận tốc vật thể.",
      formula: "Multi-sweep without Scene Flow -> Ghost Trail"
    }
  ]
};
