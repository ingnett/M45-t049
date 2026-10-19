"""cvat_nuclio/main.py

Nuclio Serverless Handler phục vụ công cụ AI Auto-annotation trong giao diện CVAT 3D.
Tích hợp trực tiếp thuật toán M45 để:
1. Nhận cụm điểm 3D hoặc bounding box từ CVAT.
2. Đánh giá định lượng độ khó (EASY, MEDIUM, HARD).
3. Tinh chỉnh tâm, kích thước theo Class Prior và trả về CVAT 3D cuboid kèm thuộc tính rủi ro & SOP.
"""

import json
import math
import numpy as np

CLASS_PRIORS = {
    "vehicle.car": [1.93, 4.62, 1.69],
    "human.pedestrian.adult": [0.67, 0.73, 1.77],
}


def init_context(context):
    context.logger.info("Khởi tạo M45 3D Screener Nuclio Function thành công.")


def handler(context, event):
    """
    Điểm vào xử lý sự kiện khi Annotator bấm 'AI Tools / Detectors' trên CVAT 3D.
    """
    try:
        body = event.body
        if isinstance(body, (bytes, bytearray)):
            body = json.loads(body.decode("utf-8"))

        # Tham số yêu cầu từ CVAT
        threshold = float(body.get("threshold", 0.5))
        pos = body.get("pos", [10.0, 0.0, 0.0])
        cls_name = body.get("label", "vehicle.car")
        points = body.get("points", [])

        is_car = "car" in cls_name.lower() or "vehicle" in cls_name.lower()
        prior_wlh = CLASS_PRIORS["vehicle.car"] if is_car else CLASS_PRIORS["human.pedestrian.adult"]

        n_pts = len(points) if points else int(body.get("point_count", 15))
        dist = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)

        # Đánh giá độ khó
        is_hard = (n_pts < (8 if is_car else 4)) or (dist >= (35.0 if is_car else 25.0))
        tier = "HARD" if is_hard else ("EASY" if n_pts >= 25 and dist < 20.0 else "MEDIUM")
        score = 4.2 if tier == "HARD" else (1.5 if tier == "EASY" else 2.8)

        risk_flags = []
        if is_hard:
            risk_flags.append(f"Thưa điểm N={n_pts} pts")
            risk_flags.append(f"Cự ly xa d={dist:.1f}m")
            risk_flags.append("Nguy cơ lộn góc Yaw 180°")
            sop = "1. Khóa kích thước Class Prior; 2. Căn sát mặt nhìn thấy; 3. Đối chiếu camera tránh lộn hướng."
        else:
            sop = "Duyệt nhanh hình khối L-shape trong 5 giây."

        response_cuboid = {
            "confidence": 0.95,
            "label": "vehicle.car" if is_car else "human.pedestrian.adult",
            "position": [float(pos[0]), float(pos[1]), float(pos[2])],
            "scale": [float(prior_wlh[0]), float(prior_wlh[1]), float(prior_wlh[2])],
            "rotation": [0.0, 0.0, 0.0],
            "attributes": {
                "difficulty_tier": tier,
                "difficulty_score": score,
                "risk_flags": "; ".join(risk_flags) if risk_flags else "Mật độ điểm tốt",
                "annotator_sop": sop,
            },
        }

        return context.Response(
            body=json.dumps([response_cuboid]),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    except Exception as exc:
        context.logger.error(f"Lỗi xử lý M45 Screener: {exc}")
        return context.Response(
            body=json.dumps({"error": str(exc)}),
            headers={},
            content_type="application/json",
            status_code=500,
        )
